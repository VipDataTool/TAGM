"""
TASM Analyzer - The Alignment Stress Map + Lateral Tension Profile
Session-based data collection platform for alignment signal analysis.
"""

import os
import io
import csv
import json
import math
import time
import base64
import logging
import threading
import traceback
from pathlib import Path
from typing import Optional

import torch

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from misc.old.tasm.engine.model_manager import ModelManager, KNOWN_PAIRS, _load_model_registry, _save_model_registry
from misc.old.tasm.engine.analyzer import Analyzer, result_to_dict
from misc.old.tasm.engine.baselines import get_all_prompts, add_prompt as csv_add_prompt
from misc.old.tasm.engine.statistics import aggregate_batch
from misc.old.tasm.engine.visualizations import (
    plot_signed_attribution, plot_stress_per_token,
    plot_amplitude_trajectory, plot_heatmap,
    plot_distribution_metrics, plot_batch_summary,
    plot_separability,
    # LTP visualizations
    plot_ltp_profiles, plot_ltp_tension_magnitudes,
    plot_ltp_dual_trajectory, plot_ltp_summary_stats,
    plot_ltp_profile_heatmap,
    # SFD visualizations
    plot_sfd_density,
    plot_rank_displacement,
)
from misc.old.tasm.engine.comparative import generate_all_comparative
from misc.old.tasm.engine.dataset import DatasetSession
from misc.old.tasm.engine.reports import generate_single_report, generate_batch_report
from misc.old.tasm.engine.modules import ModuleRunner
from misc.old.tasm.engine.modules.domain_surface import (embed_and_cache_probes, _probe_cache_path,
                                            _load_probes, _parse_meta,
                                            _load_probe_cache, _detect_level_cols)
from misc.old.tasm.engine import engine_config

# ─── Logging ─────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "tasm.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("tasm")

# ─── Global state ────────────────────────────────────────────────
mm = ModelManager()
analyzer = None
session: Optional[DatasetSession] = None
progress_log = []
loading_state = {"active": False, "error": None}
user_info = {"name": "", "organization": "", "project": ""}
module_runner = ModuleRunner()

# Locks protecting shared mutable state from concurrent access.
# _analysis_lock: serializes forward passes, activation caches, and session writes.
# _loading_lock: makes the loading_state check-and-set atomic.
_analysis_lock = threading.Lock()
mm.inference_lock = _analysis_lock  # share lock with ModelManager for module thread safety
_loading_lock = threading.Lock()
_batch_running = False


def log_progress(stage, message):
    progress_log.append({"stage": stage, "message": message, "time": time.time()})
    logger.info(f"[{stage}] {message}")


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def _validate_prompt(prompt: str) -> Optional[str]:
    """Validate a prompt string. Returns error message or None."""
    if not prompt or not prompt.strip():
        return "Prompt cannot be empty."
    if len(prompt) > 5000:
        return "Prompt exceeds maximum length (5000 characters)."
    return None


def _validate_category(category: str) -> str:
    """Normalize and validate category."""
    valid = {"benign", "mild", "harmful", "jailbreak", "adversarial", "dual-use", ""}
    cat = category.strip().lower()
    return cat if cat in valid else "unknown"


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TASM Analyzer starting up")
    # Load persisted engine config if present
    if ENGINE_CONFIG_FILE.exists():
        try:
            saved = json.loads(ENGINE_CONFIG_FILE.read_text())
            engine_config.update(saved)
            logger.info(f"Loaded engine config from disk: {len(saved)} params")
        except Exception as e:
            logger.warning(f"Failed to load engine config: {e}")
    # Load probe file selection
    _load_probe_config()
    # Clear probe caches on startup if persistence is disabled
    if not engine_config.get("persist_probe_caches"):
        cache_dir = Path(__file__).parent / "probe_cache"
        if cache_dir.exists():
            cleared = 0
            for fn in cache_dir.iterdir():
                if fn.suffix == ".json":
                    fn.unlink()
                    cleared += 1
            if cleared:
                logger.info(f"Cleared {cleared} probe cache(s) (persist disabled)")
        _active_probes.clear()
        _save_probe_config()
        logger.info("Probe set cleared (persist disabled)")
    else:
        logger.info(f"Active probe files: {sorted(_active_probes)}")
    # Clean stale session data from previous server run.
    # This is deliberate: schema changes between versions could make
    # old results.json incompatible with new code. Fresh start is safe.
    # Users can manually restore via /api/session/restore if they know
    # the data is compatible.
    stale = Path("datasets/current")
    if stale.exists():
        import shutil
        shutil.rmtree(stale, ignore_errors=True)
        logger.info(f"Cleaned stale session directory: {stale}")
    yield
    logger.info("TASM Analyzer shutting down")

app = FastAPI(title="TASM Analyzer", lifespan=lifespan)


# ─── Background model loading ────────────────────────────────────

def _load_model_worker(pair_id, base_id, instruct_id):
    global analyzer, session
    try:
        mm.load_pair(pair_id=pair_id, base_id=base_id,
                     instruct_id=instruct_id, callback=log_progress)
        analyzer = Analyzer(mm)
        session = DatasetSession()
        session.set_model(mm.state.display_name)

        # Give probe generator access to the loaded model
        pg = module_runner.get_module("probe_generator")
        if pg and hasattr(pg, 'set_model_manager'):
            pg.set_model_manager(mm)

        # Give backscatter module access for ΔW projection
        bs = module_runner.get_module("correction_backscatter")
        if bs and hasattr(bs, 'set_model_manager'):
            bs.set_model_manager(mm)

        loading_state["active"] = False
        loading_state["error"] = None
        log_progress("ready", "Model loaded. Session started. Ready to analyze.")
    except Exception as e:
        loading_state["active"] = False
        loading_state["error"] = str(e)
        logger.error(f"Model loading failed: {traceback.format_exc()}")
        log_progress("error", f"Loading failed: {e}")


def _get_layer_fracs():
    """Read and validate domain layer fractions from config."""
    subj = engine_config.get("domain_embedding_layer_frac") or 0.50
    esc = engine_config.get("domain_escalation_layer_frac") or 0.75
    for name, val in [("domain_embedding_layer_frac", subj),
                      ("domain_escalation_layer_frac", esc)]:
        if val < 0 or val > 1:
            logger.warning(f"[CONFIG] {name}={val} out of range [0,1] — clamping")
    return max(0, min(1, subj)), max(0, min(1, esc))


def _analyze_and_record(prompt, category, compute_kl, compute_trajectory,
                        capture_responses, compute_ltp=False,
                        compute_sfd=False,
                        full_capture=False,
                        ltp_k=8, ltp_layer_strategy="signal",
                        ltp_svd_rank=0,
                        skip_plots=True,
                        base_cache=None):
    # Serialize access to model activations, hooks, and session state.
    # Without this lock, concurrent API calls can corrupt activation caches
    # (one prompt's hidden states overwriting another's mid-extraction)
    # and race on session.results.
    with _analysis_lock:
        result = analyzer.analyze_prompt(
            prompt, category=category,
            compute_kl=compute_kl,
            compute_full_trajectory=compute_trajectory,
            capture_responses=capture_responses,
            full_capture=full_capture,
            compute_ltp=compute_ltp,
            compute_sfd=compute_sfd,
            ltp_k=ltp_k,
            ltp_layer_strategy=ltp_layer_strategy,
            ltp_svd_rank=ltp_svd_rank,
            base_cache=base_cache)

        plots = {}
        if not skip_plots:
            plots = {
                "signed_attribution": plot_signed_attribution(result),
                "stress_per_token": plot_stress_per_token(result),
                "distribution_metrics": plot_distribution_metrics(result),
            }
            if compute_trajectory:
                plots["amplitude_trajectory"] = plot_amplitude_trajectory(result)
                plots["heatmap"] = plot_heatmap(result)

            # LTP plots
            if compute_ltp and result.ltp is not None:
                plots["ltp_profiles"] = plot_ltp_profiles(result.ltp, result.tokens)
                plots["ltp_tension_magnitudes"] = plot_ltp_tension_magnitudes(result.ltp, result.tokens)
                plots["ltp_dual_trajectory"] = plot_ltp_dual_trajectory(result.ltp)
                plots["ltp_summary_stats"] = plot_ltp_summary_stats(result.ltp)
                plots["ltp_profile_heatmap"] = plot_ltp_profile_heatmap(result.ltp, result.tokens)

            # SFD plots
            if compute_sfd and result.sfd is not None:
                plots["sfd_density"] = plot_sfd_density(result)
            if result.rank_displacement is not None:
                plots["rank_displacement"] = plot_rank_displacement(result)

        result_dict = result_to_dict(result)
        if session:
            idx = session.add_result(result_dict)

            # Save per-prompt plots to disk (not in JSON)
            if plots:
                plot_dir = session.session_dir / "plots" / "individual"
                plot_dir.mkdir(parents=True, exist_ok=True)
                for name, b64_str in plots.items():
                    if b64_str:
                        path = plot_dir / f"{idx:04d}_{name}.png"
                        path.write_bytes(base64.b64decode(b64_str))

    plot_keys = _plot_keys_for_result(result_dict) if skip_plots else [k for k, v in plots.items() if v]
    return result_dict, plot_keys


def _plot_keys_for_result(r):
    """Determine which plots are available for a result without rendering them."""
    keys = ["signed_attribution", "stress_per_token", "distribution_metrics"]
    if r.get("amplitude_trajectory"):
        keys.append("amplitude_trajectory")
    if r.get("heatmap"):
        keys.append("heatmap")
    ltp = r.get("ltp")
    if ltp and ltp.get("profiles"):
        keys.extend(["ltp_profiles", "ltp_tension_magnitudes",
                      "ltp_dual_trajectory", "ltp_summary_stats",
                      "ltp_profile_heatmap"])
    if r.get("sfd"):
        keys.extend(["sfd_density"])
    if r.get("rank_displacement"):
        keys.append("rank_displacement")
    return keys


def _generate_deferred_plots(sess):
    """Generate per-prompt plots for results that were analyzed in batch mode
    (skip_plots=True). Reconstitutes minimal PromptResult objects from stored
    dicts and generates plots one at a time to control memory."""
    from misc.old.tasm.engine.analyzer import PromptResult
    import gc

    plot_dir = sess.session_dir / "plots" / "individual"
    generated = 0

    for idx, r in enumerate(sess.results):
        # Check if plots already exist for this prompt
        marker = plot_dir / f"{idx:04d}_stress_per_token.png"
        if marker.exists():
            continue

        try:
            pr = PromptResult.from_dict(r, mode="plot")

            plots = {
                "signed_attribution": plot_signed_attribution(pr),
                "stress_per_token": plot_stress_per_token(pr),
                "distribution_metrics": plot_distribution_metrics(pr),
            }
            if pr.amplitude_trajectory:
                plots["amplitude_trajectory"] = plot_amplitude_trajectory(pr)
            if pr.heatmap:
                plots["heatmap"] = plot_heatmap(pr)

            # LTP plots
            if pr.ltp and pr.ltp.profiles:
                plots["ltp_profiles"] = plot_ltp_profiles(pr.ltp, pr.tokens)
                plots["ltp_tension_magnitudes"] = plot_ltp_tension_magnitudes(pr.ltp, pr.tokens)
                plots["ltp_dual_trajectory"] = plot_ltp_dual_trajectory(pr.ltp)
                plots["ltp_summary_stats"] = plot_ltp_summary_stats(pr.ltp)
                plots["ltp_profile_heatmap"] = plot_ltp_profile_heatmap(pr.ltp, pr.tokens)

            # SFD plots
            if pr.sfd is not None:
                plots["sfd_density"] = plot_sfd_density(pr)
            if pr.rank_displacement is not None:
                plots["rank_displacement"] = plot_rank_displacement(pr)

            # Write to disk
            for name, b64_str in plots.items():
                if b64_str:
                    path = plot_dir / f"{idx:04d}_{name}.png"
                    path.write_bytes(base64.b64decode(b64_str))

            generated += 1

            if generated % 10 == 0:
                gc.collect()

        except Exception as e:
            logger.warning(f"Deferred plot {idx} failed: {e}")

    if generated > 0:
        logger.info(f"Generated deferred plots for {generated} prompts")


# ─── API Routes ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(
        content=html_path.read_text(),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


@app.get("/favicon.svg")
async def serve_favicon():
    favicon_path = Path(__file__).parent / "static" / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(str(favicon_path), media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"error": "No favicon."})


@app.get("/api/status")
async def get_status():
    disk_info = DatasetSession.has_session_on_disk() if not session else None
    return {
        "model_loaded": mm.is_loaded() and not loading_state["active"],
        "loading": loading_state["active"],
        "loading_error": loading_state["error"],
        "model_pair": mm.state.pair_id if mm.state else None,
        "model_name": mm.state.display_name if mm.state else None,
        "available_pairs": {k: v[2] for k, v in KNOWN_PAIRS.items()},
        "session": {
            "n_results": session.n_results if session else 0,
            "categories": session.categories if session else {},
            "cache_size_bytes": session.get_cache_size() if session else 0,
            "model": session.model_name if session else "",
        } if session else None,
        "restorable": disk_info,
        "user_info": user_info,
    }


@app.post("/api/user_info")
async def set_user_info(name: str = Form(""), organization: str = Form(""), project: str = Form("")):
    user_info["name"] = name.strip()
    user_info["organization"] = organization.strip()
    user_info["project"] = project.strip()
    logger.info(f"User info updated: {user_info}")
    return {"ok": True, "user_info": user_info}


@app.get("/api/models")
async def get_models():
    """Return the model registry from models.json."""
    models_file = Path(__file__).parent / "models.json"
    if models_file.exists():
        with open(models_file) as f:
            return {"models": json.load(f)}
    return {"models": []}


@app.post("/api/models")
async def add_model(id: str = Form(...), name: str = Form(...),
                    base: str = Form(...), instruct: str = Form(...)):
    """Add a model pair to models.json."""
    global KNOWN_PAIRS
    id_clean = id.strip().lower().replace(" ", "-")
    if not id_clean or not base.strip() or not instruct.strip():
        return JSONResponse(status_code=400, content={"error": "All fields required."})
    KNOWN_PAIRS[id_clean] = (base.strip(), instruct.strip(), name.strip())
    _save_model_registry(KNOWN_PAIRS)
    logger.info(f"Model added: {id_clean} = {name.strip()}")
    return {"ok": True}


@app.post("/api/load_model")
async def load_model(pair_id: str = Form(None),
                     base_id: str = Form(None),
                     instruct_id: str = Form(None)):
    # Atomic check-and-set prevents two concurrent load_model requests
    # from both passing the "active" check before either sets the flag.
    with _loading_lock:
        if loading_state["active"]:
            return {"ok": False, "message": "Already loading a model."}
        loading_state["active"] = True
        loading_state["error"] = None

    # Validate
    if not pair_id and not (base_id and instruct_id):
        with _loading_lock:
            loading_state["active"] = False
        return JSONResponse(status_code=400,
                            content={"error": "Select a model pair or provide custom IDs."})

    progress_log.clear()
    log_progress("starting", "Initializing model pair...")

    thread = threading.Thread(
        target=_load_model_worker,
        args=(pair_id, base_id, instruct_id), daemon=True)
    thread.start()
    return {"ok": True, "message": "Loading started."}


@app.post("/api/set_inference_model")
async def set_inference_model(model_class: str = Form(...)):
    if model_class not in ("instruct", "base"):
        return JSONResponse(status_code=400,
                            content={"error": "Must be 'instruct' or 'base'"})
    if not mm.state or not mm.state.loaded:
        return {"ok": False, "message": "No model loaded."}
    if model_class == "base" and mm.state.model_base is None:
        try:
            mm.load_base_for_kl(callback=log_progress)
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            return {"ok": False, "message": f"Failed to load base model: {e}"}
    mm.state.inference_class = model_class
    return {"ok": True, "active": model_class}


@app.post("/api/reset")
async def reset_all():
    global analyzer, session
    progress_log.clear()
    if session:
        session.clear()
        session = None
    mm.reset()
    analyzer = None
    loading_state["active"] = False
    loading_state["error"] = None
    logger.info("Full reset performed")
    log_progress("reset", "All resources released. Session cleared.")
    return {"ok": True, "message": "Reset complete."}


@app.post("/api/session/clear_plots")
async def clear_session_plots():
    """Delete all cached plot files. Keeps CSV, JSON, and session data."""
    if not session:
        return JSONResponse(status_code=400, content={"error": "No active session."})
    freed = session.clear_plots()
    freed_mb = freed / 1024 / 1024
    logger.info(f"Cleared plot cache: {freed_mb:.1f}MB freed")
    return {"ok": True, "freed_bytes": freed, "freed_mb": round(freed_mb, 1),
            "cache_size_bytes": session.get_cache_size()}


@app.post("/api/session/clear_all")
async def clear_session_all():
    """Delete all session data (plots, CSV, JSON). Session remains active but empty."""
    if not session:
        return JSONResponse(status_code=400, content={"error": "No active session."})
    freed = session.get_cache_size()
    session.clear()
    session.session_dir.mkdir(parents=True, exist_ok=True)
    session._csv_initialized = False
    freed_mb = freed / 1024 / 1024
    logger.info(f"Cleared all session data: {freed_mb:.1f}MB freed")
    return {"ok": True, "freed_bytes": freed, "freed_mb": round(freed_mb, 1),
            "n_remaining": 0, "cache_size_bytes": 0}


@app.post("/api/session/restore")
async def restore_session():
    """Restore session from disk. Used after browser crash or page refresh."""
    global session
    disk_info = DatasetSession.has_session_on_disk()
    if not disk_info or not disk_info.get("has_results"):
        return JSONResponse(status_code=404,
                            content={"ok": False, "error": "No session data found on disk."})

    restored = DatasetSession.restore()
    if not restored or restored.n_results == 0:
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": "Session data exists but could not be parsed."})

    session = restored
    logger.info(f"[SESSION] Manual restore: {session.n_results} results, "
                f"model: {session.model_name or 'unknown'}")
    return {
        "ok": True,
        "n_results": session.n_results,
        "categories": session.categories,
        "model": session.model_name,
        "cache_size_bytes": session.get_cache_size(),
    }


@app.get("/api/progress")
async def get_progress():
    return {"log": progress_log}


# ─── Prompts ──────────────────────────────────────────────────────

@app.get("/api/prompts")
async def get_prompts():
    """Return all prompts from the unified prompts.csv."""
    return {"prompts": get_all_prompts()}


@app.post("/api/prompts")
async def add_prompt_route(prompt: str = Form(...),
                           category: str = Form("benign")):
    err = _validate_prompt(prompt)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    cat = _validate_category(category)
    csv_add_prompt(prompt, cat)
    logger.info(f"Prompt added to library: [{cat}] {prompt[:60]}...")
    return {"ok": True, "prompts": get_all_prompts()}


@app.post("/api/analyze")
async def analyze_single(prompt: str = Form(...),
                         category: str = Form(""),
                         compute_kl: bool = Form(False),
                         compute_trajectory: bool = Form(True),
                         capture_responses: bool = Form(False),
                         full_capture: bool = Form(False),
                         compute_ltp: bool = Form(False),
                         compute_sfd: bool = Form(False),
                         ltp_k: int = Form(8),
                         ltp_layer_strategy: str = Form("signal"),
                         ltp_svd_rank: int = Form(0),
):
    # Validation
    err = _validate_prompt(prompt)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    if not analyzer:
        return JSONResponse(status_code=400, content={"error": "No model loaded."})

    category = _validate_category(category)

    try:
        logger.info(f"Analyzing: [{category}] {prompt[:60]}... (LTP={compute_ltp}, SFD={compute_sfd}, k={ltp_k}, strategy={ltp_layer_strategy}, svd={ltp_svd_rank})")

        # ── Sequential pipeline: batch of one ──
        # Same path for single and batch — no concurrent model loading.
        needs_base = compute_kl or capture_responses or compute_ltp
        base_cache = None

        if needs_base:
            prompts_batch = [{"prompt": prompt, "category": category}]
            base_caches = analyzer.run_base_phase(
                prompts_batch, ltp_k=ltp_k, compute_kl=compute_kl,
                capture_responses=capture_responses,
                progress=log_progress)
            base_cache = base_caches[0] if base_caches else None

        result_dict, plot_keys = _analyze_and_record(
            prompt, category, compute_kl, compute_trajectory, capture_responses,
            full_capture=full_capture,
            compute_ltp=compute_ltp, compute_sfd=compute_sfd,
            ltp_k=ltp_k, ltp_layer_strategy=ltp_layer_strategy,
            ltp_svd_rank=ltp_svd_rank,
            base_cache=base_cache)

        return sanitize_for_json({
            "ok": True,
            "result": result_dict,
            "plot_keys": plot_keys,
            "session_n": session.n_results if session else 0,
            "cache_size_bytes": session.get_cache_size() if session else 0,
        })
    except Exception as e:
        logger.error(f"Analysis failed: {traceback.format_exc()}")
        return JSONResponse(status_code=500,
                            content={"error": str(e), "trace": traceback.format_exc()})


@app.post("/api/analyze_batch")
async def analyze_batch(file: UploadFile = File(...),
                        compute_kl: bool = Form(False),
                        compute_trajectory: bool = Form(False),
                        capture_responses: bool = Form(False),
                        full_capture: bool = Form(False),
                        compute_ltp: bool = Form(False),
                        compute_sfd: bool = Form(False),
                        ltp_k: int = Form(8),
                        ltp_layer_strategy: str = Form("signal"),
                        ltp_svd_rank: int = Form(0)):
    if not analyzer:
        return JSONResponse(status_code=400, content={"error": "No model loaded."})

    global _batch_running
    if _batch_running:
        return JSONResponse(status_code=409,
                            content={"error": "A batch is already running. Wait for it to finish."})

    # Read files on the event loop (fast async I/O)
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    filename = file.filename

    # Parse prompts synchronously (fast)
    reader = csv.DictReader(io.StringIO(content))
    prompts = []
    for row in reader:
        p = (row.get("prompt") or row.get("Prompt") or "").strip()
        c = (row.get("category") or row.get("Category") or "unknown").strip()
        if p:
            prompts.append({"prompt": p, "category": _validate_category(c)})

    if not prompts:
        return JSONResponse(status_code=400,
                            content={"error": "No valid prompts found in CSV."})

    # Fire off the heavy work in a background thread and return immediately
    _batch_running = True
    threading.Thread(
        target=_run_batch_sync,
        args=(content, filename,
              compute_kl, compute_trajectory, capture_responses,
              full_capture,
              compute_ltp, compute_sfd, ltp_k, ltp_layer_strategy,
              ltp_svd_rank),
        daemon=True).start()

    return {"ok": True, "started": True, "n_prompts": len(prompts),
            "message": f"Batch started: {len(prompts)} prompts. Watch progress log."}


def _run_batch_sync(content, filename,
                    compute_kl, compute_trajectory, capture_responses,
                    full_capture,
                    compute_ltp, compute_sfd, ltp_k, ltp_layer_strategy,
                    ltp_svd_rank):
    """Synchronous batch processing — runs in a background thread.

    Uses a two-phase sequential pipeline when the base model is needed:
      Phase 1: Load base model, run all prompts, cache outputs, unload base.
      Phase 2: Load instruct model (+ deltas), run all prompts with cached
               base data. No concurrent model loading required.
    This enables larger models to run on memory-constrained hardware.
    """
    try:
        reader = csv.DictReader(io.StringIO(content))
        prompts = []
        for row in reader:
            p = (row.get("prompt") or row.get("Prompt") or "").strip()
            c = (row.get("category") or row.get("Category") or "unknown").strip()
            if p:
                prompts.append({"prompt": p, "category": _validate_category(c)})

        if not prompts:
            log_progress("error", "No valid prompts found in CSV.")
            return

        logger.info(f"Batch: {len(prompts)} prompts from {filename} (LTP={compute_ltp}, SFD={compute_sfd}, full_capture={full_capture}, svd={ltp_svd_rank})")
        log_progress("batch", f"Loaded {len(prompts)} prompts from CSV")

        # ── Determine if base model is needed ──
        needs_base = compute_kl or capture_responses or compute_ltp
        base_caches = None

        if needs_base:
            # ── Phase 1: Sequential base-model pass ──
            log_progress("base_phase", f"Phase 1: Running base model on {len(prompts)} prompts...")
            base_caches = analyzer.run_base_phase(
                prompts, ltp_k=ltp_k, compute_kl=compute_kl,
                capture_responses=capture_responses,
                progress=log_progress)
            log_progress("base_phase", f"Phase 1 complete. Base model unloaded. Starting instruct phase...")

        # ── Phase 2: Instruct model pass (base model already freed) ──
        for i, p in enumerate(prompts):
            log_progress("analyzing", f"[{i+1}/{len(prompts)}] {p['prompt'][:60]}...")
            try:
                base_cache = base_caches[i] if base_caches and i < len(base_caches) else None
                _analyze_and_record(
                    p["prompt"], p["category"],
                    compute_kl, compute_trajectory, capture_responses,
                    compute_ltp=compute_ltp, compute_sfd=compute_sfd,
                    full_capture=full_capture,
                    ltp_k=ltp_k,
                    ltp_layer_strategy=ltp_layer_strategy,
                    ltp_svd_rank=ltp_svd_rank,
                    skip_plots=True,
                    base_cache=base_cache)
            except Exception as prompt_err:
                logger.error(f"Prompt {i+1} failed: {prompt_err}")
                log_progress("warning", f"[{i+1}] FAILED: {str(prompt_err)[:80]}")

            # Free memory between prompts
            if (i + 1) % 10 == 0:
                import gc as _gc
                _gc.collect()
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except ImportError:
                    pass
                try:
                    import resource
                    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    log_progress("analyzing", f"[{i+1}/{len(prompts)}] Peak RSS: {mem_mb:.0f}MB")
                except Exception:
                    log_progress("analyzing", f"[{i+1}/{len(prompts)}] Memory cleaned")

        # Free base caches
        del base_caches

        log_progress("done", f"Batch complete: {len(prompts)} prompts added to session")

    except Exception as e:
        logger.error(f"Batch failed: {traceback.format_exc()}")
        log_progress("error", f"Batch failed: {str(e)[:100]}")
    finally:
        global _batch_running
        _batch_running = False


@app.get("/api/session/results")
async def get_session_results(page: int = 1, per_page: int = 10):
    """Return a page of session results (chronological order)."""
    if not session or session.n_results == 0:
        return {"ok": False, "results": [], "total": 0, "page": 1, "per_page": per_page, "total_pages": 0}

    total = session.n_results
    total_pages = max(1, -(-total // per_page))  # ceil division
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = min(start + per_page, total)

    # Build the page slice
    results = []
    for i in range(start, end):
        r = session.results[i]
        r_copy = dict(r)

        # List available plot keys based on result data (not filesystem —
        # plots are lazy-generated and may not exist yet on disk).
        r_copy["_plot_keys"] = _plot_keys_for_result(r)

        results.append(r_copy)

    return {"ok": True, "results": results,
            "page": page, "per_page": per_page,
            "total": total, "total_pages": total_pages,
            "cache_size_bytes": session.get_cache_size()}


@app.post("/api/session/remove")
async def remove_from_session(request: Request):
    """Remove specific results from the session by index."""
    if not session:
        return JSONResponse(status_code=400, content={"error": "No active session."})
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body."})

    indices = sorted(set(body.get("indices", [])), reverse=True)
    if not indices:
        return JSONResponse(status_code=400, content={"error": "No indices provided."})

    removed = 0
    for idx in indices:
        if 0 <= idx < len(session.results):
            session.results.pop(idx)
            removed += 1

    # Reindex _index fields
    for i, r in enumerate(session.results):
        r["_index"] = i

    # Rewrite the CSV from scratch to stay in sync
    session._csv_initialized = False
    if session.csv_path.exists():
        session.csv_path.unlink()
    for r in session.results:
        session._write_csv_row(r)

    logger.info(f"Removed {removed} results from session ({session.n_results} remaining)")
    return {"ok": True, "removed": removed, "remaining": session.n_results}


@app.post("/api/session/rerun")
async def rerun_prompts(request: Request):
    """Rerun specific prompts from the session by index. The old results are
    removed and fresh analyses are appended at the end."""
    if not session:
        return JSONResponse(status_code=400, content={"error": "No active session."})
    if not analyzer:
        return JSONResponse(status_code=400, content={"error": "No model loaded."})
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body."})

    indices = sorted(set(body.get("indices", [])))
    if not indices:
        return JSONResponse(status_code=400, content={"error": "No indices provided."})

    # Read analysis options from the request (with defaults)
    opts = body.get("options", {})
    compute_kl = opts.get("compute_kl", False)
    compute_trajectory = opts.get("compute_trajectory", True)
    capture_responses = opts.get("capture_responses", False)
    full_capture = opts.get("full_capture", False)
    compute_ltp = opts.get("compute_ltp", False)
    compute_sfd = opts.get("compute_sfd", False)
    ltp_k = opts.get("ltp_k", 8)
    ltp_layer_strategy = opts.get("ltp_layer_strategy", "signal")
    ltp_svd_rank = opts.get("ltp_svd_rank", 0)

    # Collect prompts to rerun before removing them
    to_rerun = []
    for idx in indices:
        if 0 <= idx < len(session.results):
            r = session.results[idx]
            to_rerun.append({"prompt": r["prompt"], "category": r.get("category", "")})

    if not to_rerun:
        return JSONResponse(status_code=400, content={"error": "No valid indices."})

    # Remove the old results (reverse order to keep indices valid)
    for idx in sorted(indices, reverse=True):
        if 0 <= idx < len(session.results):
            session.results.pop(idx)

    # Reindex
    for i, r in enumerate(session.results):
        r["_index"] = i

    # ── Sequential pipeline for rerun ──
    needs_base = compute_kl or capture_responses or compute_ltp
    base_caches = None

    if needs_base:
        base_caches = analyzer.run_base_phase(
            to_rerun, ltp_k=ltp_k, compute_kl=compute_kl,
            capture_responses=capture_responses,
            progress=log_progress)

    rerun_count = 0
    for i, item in enumerate(to_rerun):
        try:
            base_cache = base_caches[i] if base_caches and i < len(base_caches) else None
            _analyze_and_record(
                item["prompt"], item["category"],
                compute_kl, compute_trajectory, capture_responses,
                full_capture=full_capture,
                compute_ltp=compute_ltp, compute_sfd=compute_sfd,
                ltp_k=ltp_k,
                ltp_layer_strategy=ltp_layer_strategy,
                ltp_svd_rank=ltp_svd_rank,
                base_cache=base_cache)
            rerun_count += 1
        except Exception as e:
            logger.error(f"Rerun failed for '{item['prompt'][:40]}': {e}")

    # Rewrite CSV from scratch
    session._csv_initialized = False
    if session.csv_path.exists():
        session.csv_path.unlink()
    for r in session.results:
        session._write_csv_row(r)

    logger.info(f"Reran {rerun_count}/{len(to_rerun)} prompts")
    return {"ok": True, "rerun": rerun_count, "total": session.n_results}


@app.get("/api/dashboard")
async def get_dashboard(force: bool = False):
    if not session or session.n_results == 0:
        return {"ok": False, "error": "No data in session."}

    import asyncio
    try:
        return await asyncio.to_thread(_run_dashboard_sync, force)
    except Exception as e:
        logger.error(f"Dashboard failed: {traceback.format_exc()}")
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": str(e)})


def _run_dashboard_sync(force: bool = False):
    """Synchronous session data refresh — runs in a thread.

    Returns lightweight slim results for the data table and session info.
    Aggregate statistics are now computed by the Comparative Analysis module.
    """
    logger.info(f"Session refresh: {session.n_results} prompts")
    results = session.results

    # ── Build lightweight result list (scalars only) ──
    slim_results = []
    for i, r in enumerate(results):
        slim = {"_index": i}
        for k in ["prompt", "category", "role", "seq_len", "stress_score",
                   "net_correction", "entropy", "top2_share",
                   "middle_share", "interior_cv", "kl_divergence",
                   "n_negative_tokens", "has_negative_tokens"]:
            slim[k] = r.get(k)

        # Instrument scalars only
        ltp = r.get("ltp")
        if ltp:
            slim["ltp"] = {k: ltp.get(k) for k in [
                "mean_M", "mean_V",
                "max_prc", "n_directional"]}
        sfd = r.get("sfd")
        if sfd:
            slim["sfd"] = {k: sfd.get(k) for k in [
                "density_mean"]}
        rd = r.get("rank_displacement")
        if rd:
            slim["rank_displacement"] = {k: rd.get(k) for k in [
                "mean_tau", "mean_overlap", "mean_matched", "mean_replacement",
                "mean_concentration", "mean_disp_per_token", "total_displacement",
                "high_replacement_frac", "low_match_frac"]}

        # Pre-computed candidate graph summary
        slim["candidate_graph"] = _compute_candidate_graph_summary(r)

        # Model predictions (top-k next token)
        if r.get("instruct_topk"):
            slim["instruct_topk"] = r["instruct_topk"]
        if r.get("base_topk"):
            slim["base_topk"] = r["base_topk"]

        slim_results.append(slim)

    return sanitize_for_json({
        "ok": True,
        "results": slim_results,
        "session_info": {
            "n_results": session.n_results,
            "categories": session.categories,
            "model": session.model_name,
            "timestamp": session.timestamp,
        },
        "cache_size_bytes": session.get_cache_size(),
    })


@app.get("/api/plots/{plot_key}")
async def get_plot(plot_key: str):
    """Serve a plot as a PNG file, generating it on demand if needed."""
    if not session:
        return JSONResponse(status_code=404, content={"error": "No session."})
    safe_key = "".join(c for c in plot_key if c.isalnum() or c in "_-")
    plots_dir = session.session_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f"{safe_key}.png"

    # Serve from cache if already generated
    if plot_path.exists():
        return FileResponse(plot_path, media_type="image/png",
                            headers={"Cache-Control": "no-cache"})

    # Generate on demand
    import asyncio
    try:
        png_bytes = await asyncio.to_thread(_generate_plot_sync, safe_key)
        if png_bytes:
            plot_path.write_bytes(png_bytes)
            return FileResponse(plot_path, media_type="image/png",
                                headers={"Cache-Control": "no-cache"})
        else:
            return JSONResponse(status_code=404,
                                content={"error": f"Plot '{plot_key}' could not be generated."})
    except Exception as e:
        logger.error(f"Plot generation failed for {safe_key}: {e}")
        return JSONResponse(status_code=500,
                            content={"error": f"Plot generation failed: {str(e)[:80]}"})


# Lock to prevent duplicate generation of the same plot
_plot_gen_lock = threading.Lock()

def _generate_plot_sync(plot_key: str) -> Optional[bytes]:
    """Generate a single plot by key. Returns PNG bytes or None."""
    with _plot_gen_lock:
        # Double-check: another thread may have generated while we waited
        plots_dir = session.session_dir / "plots"
        plot_path = plots_dir / f"{plot_key}.png"
        if plot_path.exists():
            return plot_path.read_bytes()

        logger.info(f"[PLOT] Generating on demand: {plot_key}")
        results = session.results

        # Load aggregate stats if needed
        agg = None
        agg_path = session.session_dir / "aggregate_statistics.json"
        if agg_path.exists():
            try:
                with open(agg_path) as f:
                    agg = json.load(f)
            except Exception:
                pass

        # If agg not on disk, compute it
        if agg is None:
            from misc.old.tasm.engine.analyzer import PromptResult
            pr_list = [PromptResult.from_dict(r, mode="scalar") for r in results]
            agg = aggregate_batch(pr_list)

        b64 = None

        # Aggregate-based plots
        if plot_key == "batch_summary":
            b64 = plot_batch_summary(agg)
        elif plot_key == "separability":
            b64 = plot_separability(agg)

        # Comparative plots (proven)
        elif plot_key == "key_scatters":
            from misc.old.tasm.engine.comparative import plot_key_scatters
            b64 = plot_key_scatters(results)
        elif plot_key == "discriminative_sublayers":
            from misc.old.tasm.engine.comparative import plot_discriminative_sublayers
            b64 = plot_discriminative_sublayers(results)
        elif plot_key == "proof1_summary":
            from misc.old.tasm.engine.comparative import plot_proof1_summary
            b64 = plot_proof1_summary(results)

        # Comparative plots (experimental)
        elif plot_key == "exp_trajectory_overlay":
            from misc.old.tasm.engine.comparative import plot_trajectory_overlay
            b64 = plot_trajectory_overlay(results)
        elif plot_key == "exp_difference_from_benign":
            from misc.old.tasm.engine.comparative import plot_difference_from_benign
            b64 = plot_difference_from_benign(results)
        elif plot_key == "exp_metric_scatters":
            from misc.old.tasm.engine.comparative import plot_metric_scatters
            b64 = plot_metric_scatters(results)
        elif plot_key == "exp_behavioral_comparison":
            from misc.old.tasm.engine.comparative import plot_behavioral_comparison
            b64 = plot_behavioral_comparison(results)
        elif plot_key == "exp_ltp_category_comparison":
            from misc.old.tasm.engine.comparative import plot_ltp_category_comparison
            b64 = plot_ltp_category_comparison(results)
        elif plot_key == "exp_ltp_m_vs_stress":
            from misc.old.tasm.engine.comparative import plot_ltp_m_vs_stress
            b64 = plot_ltp_m_vs_stress(results)
        elif plot_key == "exp_ltp_profile_shapes":
            from misc.old.tasm.engine.comparative import plot_ltp_profile_shape_distribution
            b64 = plot_ltp_profile_shape_distribution(results)
        elif plot_key == "exp_sfd_category_comparison":
            from misc.old.tasm.engine.comparative import plot_sfd_category_comparison
            b64 = plot_sfd_category_comparison(results)
        elif plot_key == "exp_sfd_vs_asm":
            from misc.old.tasm.engine.comparative import plot_sfd_vs_asm
            b64 = plot_sfd_vs_asm(results)
        elif plot_key == "exp_rank_displacement":
            from misc.old.tasm.engine.comparative import plot_rank_displacement_by_category
            b64 = plot_rank_displacement_by_category(results)
        else:
            logger.warning(f"[PLOT] Unknown plot key: {plot_key}")
            return None

        if b64:
            png_bytes = base64.b64decode(b64)
            elapsed = "generated"
            logger.info(f"[PLOT] {plot_key} {elapsed}")
            return png_bytes
        return None


@app.get("/api/plots/individual/{index}/{plot_key}")
async def get_individual_plot(index: int, plot_key: str):
    """Serve a per-prompt plot, generating it on demand if needed."""
    if not session:
        return JSONResponse(status_code=404, content={"error": "No session."})
    if index < 0 or index >= len(session.results):
        return JSONResponse(status_code=404, content={"error": "Index out of range."})

    safe_key = "".join(c for c in plot_key if c.isalnum() or c in "_-")
    plot_dir = session.session_dir / "plots" / "individual"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{index:04d}_{safe_key}.png"

    # Serve from cache if already generated
    if plot_path.exists():
        return FileResponse(plot_path, media_type="image/png",
                            headers={"Cache-Control": "no-cache"})

    # Generate on demand
    import asyncio
    try:
        png_bytes = await asyncio.to_thread(
            _generate_individual_plot_sync, index, safe_key)
        if png_bytes:
            plot_path.write_bytes(png_bytes)
            return FileResponse(plot_path, media_type="image/png",
                                headers={"Cache-Control": "no-cache"})
        else:
            return JSONResponse(status_code=404,
                                content={"error": f"Plot '{plot_key}' could not be generated."})
    except Exception as e:
        logger.error(f"Individual plot generation failed for {index}/{safe_key}: {e}")
        return JSONResponse(status_code=500,
                            content={"error": f"Plot generation failed: {str(e)[:80]}"})


def _generate_individual_plot_sync(index: int, plot_key: str) -> Optional[bytes]:
    """Generate a single per-prompt plot on demand. Returns PNG bytes or None."""
    from misc.old.tasm.engine.analyzer import PromptResult

    with _plot_gen_lock:
        # Double-check: another thread may have generated while we waited
        plot_dir = session.session_dir / "plots" / "individual"
        plot_path = plot_dir / f"{index:04d}_{plot_key}.png"
        if plot_path.exists():
            return plot_path.read_bytes()

        logger.info(f"[PLOT] Generating on demand: {index}/{plot_key}")

        r = session.results[index]
        pr = PromptResult.from_dict(r, mode="plot")

        b64 = None

        # Core plots
        if plot_key == "signed_attribution":
            b64 = plot_signed_attribution(pr)
        elif plot_key == "stress_per_token":
            b64 = plot_stress_per_token(pr)
        elif plot_key == "distribution_metrics":
            b64 = plot_distribution_metrics(pr)
        elif plot_key == "amplitude_trajectory":
            b64 = plot_amplitude_trajectory(pr)
        elif plot_key == "heatmap":
            b64 = plot_heatmap(pr)

        # LTP plots
        elif plot_key == "ltp_profiles" and pr.ltp:
            b64 = plot_ltp_profiles(pr.ltp, pr.tokens)
        elif plot_key == "ltp_tension_magnitudes" and pr.ltp:
            b64 = plot_ltp_tension_magnitudes(pr.ltp, pr.tokens)
        elif plot_key == "ltp_dual_trajectory" and pr.ltp:
            b64 = plot_ltp_dual_trajectory(pr.ltp)
        elif plot_key == "ltp_summary_stats" and pr.ltp:
            b64 = plot_ltp_summary_stats(pr.ltp)
        elif plot_key == "ltp_profile_heatmap" and pr.ltp:
            b64 = plot_ltp_profile_heatmap(pr.ltp, pr.tokens)

        # SFD plots
        elif plot_key == "sfd_density" and pr.sfd:
            b64 = plot_sfd_density(pr)
        elif plot_key == "rank_displacement" and pr.rank_displacement:
            b64 = plot_rank_displacement(pr)

        else:
            logger.warning(f"[PLOT] Unknown or unavailable: {index}/{plot_key}")
            return None

        if b64:
            return base64.b64decode(b64)
        return None


def _compute_candidate_graph_summary(r):
    """Pre-compute candidate graph topology metrics for one result.
    Returns lightweight summary dict — no raw arrays."""
    tokens = r.get("tokens", [])
    inst_cf = (r.get("ltp") or {}).get("counterfactual_tokens", [])
    base_cf = r.get("base_counterfactual_tokens", [])
    n_pos = min(len(inst_cf), len(base_cf), len(tokens))
    if n_pos == 0:
        return None

    candidates = {}  # token -> {promoted: [], demoted: [], matched: []}
    pos_stats = []

    for pos in range(n_pos):
        inst_set = {t: p for t, p in inst_cf[pos]} if inst_cf[pos] else {}
        base_set = {t: p for t, p in base_cf[pos]} if base_cf[pos] else {}

        n_prom, n_demo, n_match = 0, 0, 0
        for t in inst_set:
            if t not in candidates:
                candidates[t] = {"promoted": [], "demoted": [], "matched": [], "inst": [], "base": []}
            candidates[t]["inst"].append(pos)
            if t in base_set:
                candidates[t]["matched"].append(pos)
                n_match += 1
            else:
                candidates[t]["promoted"].append(pos)
                n_prom += 1
        for t in base_set:
            if t not in candidates:
                candidates[t] = {"promoted": [], "demoted": [], "matched": [], "inst": [], "base": []}
            candidates[t]["base"].append(pos)
            if t not in inst_set:
                candidates[t]["demoted"].append(pos)
                n_demo += 1

        contested = n_prom > 0 and n_demo > 0
        pos_stats.append({"contested": contested, "intensity": n_prom + n_demo})

    # Metrics
    n_contested = sum(1 for ps in pos_stats if ps["contested"])
    contested_frac = n_contested / n_pos if n_pos > 0 else 0

    # Dual-role candidates
    n_dual = sum(1 for c in candidates.values() if c["promoted"] and c["demoted"])

    # Role switches
    n_switches = 0
    for t, c in candidates.items():
        all_pos = sorted(set(c["inst"] + c["base"]))
        if len(all_pos) < 2:
            continue
        for i in range(len(all_pos) - 1):
            if all_pos[i + 1] - all_pos[i] > 2:
                continue
            p1, p2 = all_pos[i], all_pos[i + 1]
            r1 = "P" if p1 in c["promoted"] else ("D" if p1 in c["demoted"] else "M")
            r2 = "P" if p2 in c["promoted"] else ("D" if p2 in c["demoted"] else "M")
            if r1 != r2:
                n_switches += 1

    # Multi-position count
    n_multi = sum(1 for c in candidates.values() if len(set(c["inst"] + c["base"])) >= 2)

    switch_rate = n_switches / n_pos if n_pos > 0 else 0

    # Contestation map (compact string)
    cmap = ""
    for ps in pos_stats:
        if not ps["contested"]:
            cmap += "."
        elif ps["intensity"] >= 6:
            cmap += "H"
        elif ps["intensity"] >= 3:
            cmap += "M"
        else:
            cmap += "L"

    return {
        "contested_frac": round(contested_frac, 4),
        "n_dual_role": n_dual,
        "n_role_switches": n_switches,
        "switch_rate": round(switch_rate, 4),
        "n_multi_position": n_multi,
        "n_unique_candidates": len(candidates),
        "n_positions": n_pos,
        "contest_map": cmap,
    }


@app.get("/api/results/detail")
async def get_results_detail(start: int = 0, count: int = 25):
    """Return heavy per-token arrays for a slice of results.
    Used by terrain viewer and other components that need raw data."""
    if not session or session.n_results == 0:
        return {"ok": False, "error": "No data in session."}

    results = session.results
    end = min(start + count, len(results))
    detail = []
    for i in range(start, end):
        r = results[i]
        d = {"_index": i, "tokens": r.get("tokens", []),
             "per_token_kl": r.get("per_token_kl"),
             "signed_attr": r.get("signed_attr"),
             "per_token_stress": r.get("per_token_stress"),
             "base_counterfactual_tokens": r.get("base_counterfactual_tokens"),
             "prompt": r.get("prompt", ""), "category": r.get("category", "")}
        ltp = r.get("ltp")
        if ltp:
            d["ltp"] = {}
            for k in ["profiles", "base_profiles", "counterfactual_tokens",
                       "tension_magnitudes", "profile_shapes"]:
                if ltp.get(k):
                    d["ltp"][k] = ltp[k]
        rd = r.get("rank_displacement")
        if rd:
            d["rank_displacement"] = {}
            for k in ["instruct_disp_profiles", "base_disp_profiles", "per_position"]:
                if rd.get(k):
                    d["rank_displacement"][k] = rd[k]
        detail.append(d)

    return sanitize_for_json({
        "ok": True, "start": start, "count": len(detail),
        "total": len(results), "results": detail,
    })


@app.post("/api/export")
async def export_session(request: Request):
    if not session or session.n_results == 0:
        return JSONResponse(status_code=400, content={"error": "No data to export."})

    # Parse export options from request body
    try:
        opts = await request.json()
    except Exception:
        opts = {}
    export_opts = {
        "csv": True,  # always
        "pdf": opts.get("pdf", False),
        "json": opts.get("json", False),
        "charts": opts.get("charts", False),
        "includeArrays": opts.get("includeArrays", True),
        "exportPath": opts.get("exportPath", ""),
    }

    # Run export in background thread
    threading.Thread(target=_run_export_sync, args=(export_opts,), daemon=True).start()
    return {"ok": True, "message": "Export started. Watch progress log."}


@app.get("/api/export/download")
async def download_export():
    """Serve the prepared ZIP file after export completes."""
    if not session:
        return JSONResponse(status_code=400, content={"error": "No session."})
    zip_path = session.session_dir / "tasm_session.zip"
    if not zip_path.exists():
        return JSONResponse(status_code=404, content={"error": "Export not ready yet."})
    return FileResponse(
        str(zip_path), media_type="application/zip",
        filename=f"tasm_session_{session.timestamp}.zip")


def _run_export_sync(opts=None):
    """Synchronous export — runs in a background thread."""
    if opts is None:
        opts = {"csv": True, "pdf": False, "json": False, "charts": False, "exportPath": ""}
    import shutil

    do_pdf = opts.get("pdf", False)
    do_json = opts.get("json", False)
    do_charts = opts.get("charts", False)
    export_path = opts.get("exportPath", "").strip()

    # ── Aggregate stats (needed for PDF, charts, and aggregate JSON) ─
    agg = None
    all_plots = {}
    if do_pdf or do_charts:
        log_progress("exporting", "Generating aggregate statistics...")
        try:
            from misc.old.tasm.engine.analyzer import PromptResult
            results = session.results
            pr_list = [PromptResult.from_dict(r, mode="scalar") for r in results]

            log_progress("exporting", "Computing separability and comparative plots...")
            agg = aggregate_batch(pr_list)
            agg_plots = {"batch_summary": plot_batch_summary(agg),
                         "separability": plot_separability(agg)}
            comp_plots = generate_all_comparative(results)
            all_plots = {**agg_plots, **comp_plots}

            if do_charts:
                session.save_comparative_plots(all_plots)
            session.save_aggregate_json(agg)
        except Exception as e:
            logger.error(f"Export aggregate/stats failed: {e}")
            log_progress("warning", f"Aggregate stats failed: {str(e)[:80]}")
    else:
        log_progress("exporting", "Skipping charts and PDF (disabled)...")

    if do_pdf and agg is not None:
        try:
            log_progress("exporting", "Generating PDF report...")
            pdf_path = generate_batch_report(
                agg, session.results, all_plots,
                model_name=session.model_name, n_prompts=session.n_results,
                user_info=user_info)
            shutil.copy2(pdf_path, session.session_dir / "report.pdf")
        except Exception as e:
            logger.error(f"Export PDF failed: {e}")
            log_progress("warning", f"PDF generation failed: {str(e)[:80]}")
    elif not do_pdf:
        log_progress("exporting", "Skipping PDF report (disabled)...")

    if do_json:
        try:
            include_arrays = opts.get("includeArrays", True)
            label = "full (with per-token arrays)" if include_arrays else "compact (scalars only)"
            log_progress("exporting", f"Saving results JSON ({label})...")
            session.save_results_json(include_arrays=include_arrays)
        except Exception as e:
            logger.error(f"Export results JSON failed: {e}")
    else:
        log_progress("exporting", "Skipping JSON results (disabled)...")

    if do_charts:
        try:
            n = session.n_results
            log_progress("exporting", f"Generating per-prompt plots ({n} prompts)...")
            _generate_deferred_plots(session)
        except Exception as e:
            logger.error(f"Deferred plot generation failed: {e}")
    else:
        log_progress("exporting", "Skipping per-prompt plots (disabled)...")

    try:
        log_progress("exporting", "Packaging ZIP...")
        zip_bytes = session.export_zip(
            exclude_plots=not do_charts,
            exclude_pdf=not do_pdf,
            exclude_json=not do_json)
        size_mb = len(zip_bytes) / 1024 / 1024
        logger.info(f"Session exported: {session.n_results} prompts, {size_mb:.1f}MB")

        # Copy to custom export path if specified
        if export_path:
            try:
                export_dir = Path(export_path)
                export_dir.mkdir(parents=True, exist_ok=True)
                zip_src = session.session_dir / "tasm_session.zip"
                dest = export_dir / f"tasm_session_{session.timestamp}.zip"
                shutil.copy2(zip_src, dest)
                log_progress("done", f"Export ready: {session.n_results} prompts, {size_mb:.1f}MB → {dest}")
            except Exception as e:
                logger.error(f"Export copy to {export_path} failed: {e}")
                log_progress("warning", f"Could not copy to {export_path}: {str(e)[:80]}")
                log_progress("done", f"Export ready (path failed): {session.n_results} prompts. Click Download.")
        else:
            log_progress("done", f"Export ready: {session.n_results} prompts. Click Download.")
    except Exception as e:
        logger.error(f"Export ZIP failed: {traceback.format_exc()}")
        log_progress("error", f"Export ZIP failed: {str(e)[:80]}")


# ─── Module Framework Endpoints ──────────────────────────────────

@app.get("/api/modules")
async def list_modules():
    """List all available analysis modules with status."""
    return {"ok": True, "modules": module_runner.list_modules()}


@app.post("/api/modules/upload_template")
async def upload_module_template(file: UploadFile = File(...)):
    """Upload a template CSV for module use. Saves to project root."""
    project_root = Path(__file__).parent
    dest = project_root / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return {"ok": True, "filename": file.filename}


@app.post("/api/modules/{module_name}/run")
async def run_module(module_name: str, request: Request):
    """Start a module in a background thread."""
    # Check if module needs session data
    mod = module_runner.get_module(module_name)
    if mod and mod.min_results > 0:
        if not session or session.n_results == 0:
            return JSONResponse(status_code=400,
                                content={"ok": False, "error": "No session data."})

    session_results = session.results if session else []

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    params = body.get("params", {})

    result = module_runner.run_module(
        module_name,
        session_results,
        params,
        session_dir=session.session_dir if session else None,
    )
    if not result["ok"]:
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/api/modules/{module_name}/status")
async def module_status(module_name: str):
    """Check module execution status."""
    return module_runner.get_status(module_name)


@app.get("/api/modules/{module_name}/results")
async def module_results(module_name: str):
    """Get module results."""
    results = module_runner.get_results(module_name)
    if results is None:
        return JSONResponse(status_code=404,
                            content={"ok": False, "error": "No results available."})
    return sanitize_for_json({"ok": True, "results": results})


@app.post("/api/modules/{module_name}/reset")
async def reset_module(module_name: str):
    """Reset a module to idle state."""
    result = module_runner.reset_module(module_name)
    if not result["ok"]:
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/api/modules/{module_name}/download_log")
async def download_module_log(module_name: str):
    """Download the standalone log file for a completed module."""
    log_path = module_runner.get_log_path(module_name)
    if not log_path or not Path(log_path).exists():
        return JSONResponse(status_code=404,
                            content={"ok": False, "error": "No log file available."})
    return FileResponse(
        log_path, media_type="application/json",
        filename=f"module_{module_name}_log.json")


@app.get("/api/log")
async def get_log():
    """Return the log file for download."""
    if LOG_FILE.exists():
        return FileResponse(str(LOG_FILE), media_type="text/plain", filename="tasm.log")
    return JSONResponse(status_code=404, content={"error": "No log file."})


CONFIG_FILE = Path(__file__).parent / "ui_config.json"
ENGINE_CONFIG_FILE = Path(__file__).parent / "engine_config.json"
PROBE_CONFIG_FILE = Path(__file__).parent / "probe_config.json"

# Active probe files — loaded at startup, persisted on change.
_active_probes = set()  # filenames, e.g. {"alignment_probes.csv"}

def _load_probe_config():
    """Load active probe file list from disk."""
    global _active_probes
    if PROBE_CONFIG_FILE.exists():
        try:
            data = json.loads(PROBE_CONFIG_FILE.read_text())
            _active_probes = set(data.get("active", []))
        except Exception:
            _active_probes = set()
    return _active_probes

def _save_probe_config():
    """Persist active probe file list to disk."""
    PROBE_CONFIG_FILE.write_text(json.dumps({"active": sorted(_active_probes)}, indent=2))

@app.get("/api/config")
async def get_config():
    """Load saved UI configuration."""
    if CONFIG_FILE.exists():
        import json
        try:
            data = json.loads(CONFIG_FILE.read_text())
            return JSONResponse(content={"ok": True, "config": data})
        except Exception:
            pass
    return JSONResponse(content={"ok": True, "config": {}})

@app.post("/api/config")
async def save_config(request: Request):
    """Save UI configuration to disk."""
    import json
    try:
        data = await request.json()
        CONFIG_FILE.write_text(json.dumps(data, indent=2))
        return JSONResponse(content={"ok": True})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


# ─── Engine Configuration Endpoints ─────────────────────────────

@app.get("/api/engine_config")
async def get_engine_config():
    """Return current engine parameters and their defaults."""
    return {
        "ok": True,
        "config": engine_config.as_dict(),
        "defaults": dict(engine_config.DEFAULTS),
    }


@app.post("/api/engine_config")
async def update_engine_config(request: Request):
    """Update engine parameters. Changes take effect on next analysis."""
    try:
        data = await request.json()
        engine_config.update(data)
        # Persist to disk
        ENGINE_CONFIG_FILE.write_text(json.dumps(engine_config.as_dict(), indent=2))
        logger.info(f"[CONFIG] Engine config updated and saved: {list(data.keys())}")
        return {"ok": True, "config": engine_config.as_dict()}
    except Exception as e:
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": str(e)})


@app.post("/api/engine_config/reset")
async def reset_engine_config():
    """Reset all engine parameters to defaults."""
    engine_config.reset()
    # Remove persisted file so startup uses defaults
    if ENGINE_CONFIG_FILE.exists():
        ENGINE_CONFIG_FILE.unlink()
    logger.info("[CONFIG] Engine config reset to defaults")
    return {"ok": True, "config": engine_config.as_dict()}



_probe_apply_state = {"active": False, "progress": "", "error": None, "result": None}


@app.post("/api/probe_set/apply")
async def apply_probe_set(file: UploadFile = File(...)):
    """Upload a probe CSV, validate, kick off background embedding."""
    if not mm.state or not mm.state.model_instruct:
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": "No model loaded."})

    if _probe_apply_state["active"]:
        return JSONResponse(status_code=409,
                            content={"ok": False, "error": "Probe embedding already in progress."})

    project_root = str(Path(__file__).parent)
    filename = file.filename

    # Save file
    dest = os.path.join(project_root, filename)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # Validate
    level_cols, level_names = _detect_level_cols(dest)
    if not level_cols:
        os.remove(dest)
        return JSONResponse(status_code=400,
                            content={"ok": False,
                                     "error": "Not a valid probe file — needs 'subject' column and subclass columns."})

    probes = _load_probes(dest)
    subjects = sorted(set(p["subject"] for p in probes))

    # Start background thread
    _probe_apply_state["active"] = True
    _probe_apply_state["progress"] = "Starting..."
    _probe_apply_state["error"] = None
    _probe_apply_state["result"] = None

    import threading
    threading.Thread(
        target=_probe_apply_worker,
        args=(filename, project_root, len(probes), len(subjects),
              len(level_cols), level_names, subjects),
        daemon=True
    ).start()

    return {"ok": True, "status": "started", "filename": filename}


@app.get("/api/probe_set/apply_status")
async def probe_apply_status():
    """Poll embedding progress."""
    return {
        "ok": True,
        "active": _probe_apply_state["active"],
        "progress": _probe_apply_state["progress"],
        "error": _probe_apply_state["error"],
        "result": _probe_apply_state["result"],
    }


def _probe_apply_worker(filename, project_root, n_probes, n_subjects,
                         n_levels, level_names, subjects):
    """Background worker: embed probes and set as active."""
    try:
        state = mm.state
        model_id = state.instruct_model_id or state.pair_id
        use_proj = engine_config.get("probe_projection_space")

        # Check for template-specified layer depths
        csv_path = os.path.join(project_root, filename)
        meta = _parse_meta(csv_path)
        if "layer_low" in meta and "layer_high" in meta:
            subj_frac = max(0.0, min(1.0, float(meta["layer_low"])))
            esc_frac = max(0.0, min(1.0, float(meta["layer_high"])))
            logger.info(f"[PROBE_SET] Using template depths: "
                        f"L{int(subj_frac*100)}, L{int(esc_frac*100)}")
        else:
            subj_frac, esc_frac = _get_layer_fracs()

        depths = sorted(set([subj_frac, esc_frac]))
        embedded = 0

        with _analysis_lock:
            for frac in depths:
                _probe_apply_state["progress"] = f"Embedding at L{int(frac*100)}..."

                def _probe_progress(stage, msg, _frac=frac):
                    _probe_apply_state["progress"] = f"L{int(_frac*100)}: {msg}"
                    log_progress(stage, msg)

                delta = None
                if use_proj:
                    target_layer = max(0, min(state.n_layers - 1, int(frac * state.n_layers)))
                    delta = state.o_delta(target_layer)
                try:
                    embed_and_cache_probes(
                        state.model_instruct, state.tokenizer,
                        project_root, filename, model_id,
                        layer_frac=frac,
                        progress=_probe_progress,
                        delta_matrix=delta if use_proj else None)
                    embedded += 1
                except Exception as e:
                    logger.warning(f"[PROBE_SET] Failed at L{int(frac*100)}: {e}")

        if embedded == 0:
            _probe_apply_state["error"] = "Failed to embed probes at any depth."
            _probe_apply_state["active"] = False
            return

        # Set as sole active probe
        _active_probes.clear()
        _active_probes.add(filename)
        _save_probe_config()

        _probe_apply_state["result"] = {
            "filename": filename,
            "n_probes": n_probes,
            "n_subjects": n_subjects,
            "n_levels": n_levels,
            "levels": level_names,
            "subjects": subjects,
            "layer_L50": int(subj_frac * 100),
            "layer_L75": int(esc_frac * 100),
        }
        _probe_apply_state["progress"] = "Complete."
        logger.info(f"[PROBE_SET] Applied {filename}: {n_probes} probes embedded")

    except Exception as e:
        logger.error(f"[PROBE_SET] Worker failed: {e}")
        _probe_apply_state["error"] = str(e)
    finally:
        _probe_apply_state["active"] = False


@app.get("/api/probe_set/status")
async def probe_set_status():
    """Return info about the currently active probe set."""
    project_root = str(Path(__file__).parent)
    if not _active_probes:
        return {"ok": True, "active": False}

    filename = next(iter(_active_probes))
    csv_path = os.path.join(project_root, filename)
    if not os.path.exists(csv_path):
        return {"ok": True, "active": False}

    level_cols, level_names = _detect_level_cols(csv_path)
    probes = _load_probes(csv_path)
    subjects = sorted(set(p["subject"] for p in probes))

    # Check cache
    state = mm.state
    model_id = (state.instruct_model_id or state.pair_id) if state else None
    subj_frac = max(0, min(1, engine_config.get("domain_embedding_layer_frac") or 0.50))
    cached = False
    if model_id:
        cache_path = _probe_cache_path(
            project_root, filename, model_id, subj_frac,
            projected=engine_config.get("probe_projection_space"))
        cached = os.path.exists(cache_path)

    return {
        "ok": True,
        "active": True,
        "filename": filename,
        "n_probes": len(probes),
        "n_subjects": len(subjects),
        "n_levels": len(level_cols),
        "levels": level_names,
        "cached": cached,
    }


@app.post("/api/probe_set/clear_caches")
async def clear_probe_caches():
    """Delete all probe cache files."""
    project_root = str(Path(__file__).parent)
    cache_dir = os.path.join(project_root, "probe_cache")
    deleted = 0
    if os.path.isdir(cache_dir):
        for fn in os.listdir(cache_dir):
            fp = os.path.join(cache_dir, fn)
            if fn.endswith(".json") and os.path.isfile(fp):
                os.remove(fp)
                deleted += 1
    logger.info(f"[PROBE_SET] Cleared {deleted} cache files")
    return {"ok": True, "deleted": deleted}


@app.post("/api/chat")
async def chat(request: Request):
    """Generate text from the active inference model, optionally analyzing prompts/responses."""
    if analyzer is None or analyzer.mm.state is None or analyzer.mm.active_model is None:
        return JSONResponse(content={"ok": False, "error": "No model loaded"})

    try:
        body = await request.json()
        messages = body.get("messages", [])
        max_tokens = min(body.get("max_tokens", 256), engine_config.get("chat_max_tokens"))
        do_analyze = body.get("analyze", False)
        do_analyze_response = body.get("analyze_response", False)
        category = body.get("category", "")
        compute_ltp = body.get("compute_ltp", True)
        compute_sfd = body.get("compute_sfd", True)

        if not messages:
            return JSONResponse(content={"ok": False, "error": "No messages"})

        prompt_text = messages[-1].get("content", "")

        # Generate response
        with _analysis_lock:
            state = analyzer.mm.state
            model = analyzer.mm.active_model
            tokenizer = state.tokenizer
            device = state.device
            using_class = state.inference_class

            # Base models lack chat templates — fall back to raw prompt
            if using_class == "base":
                text = prompt_text
            else:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=engine_config.get("chat_temperature"),
                    top_p=engine_config.get("chat_top_p"),
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        result = {"ok": True, "response": response, "model_class": using_class}

        # Analyze user prompt — goes into session as a regular result
        if do_analyze and prompt_text:
            try:
                rd, _ = _analyze_and_record(
                    prompt_text, category=category,
                    compute_kl=True, compute_trajectory=False,
                    capture_responses=False,
                    compute_ltp=compute_ltp, compute_sfd=compute_sfd,
                    skip_plots=True,
                )
                # Tag it
                if session and session.results:
                    session.results[-1]['role'] = 'user'
                result['prompt_analyzed'] = True
            except Exception as e:
                logging.exception("Chat prompt analysis failed")
                result['prompt_analysis_error'] = str(e)

        # Analyze model response — also goes into session
        if do_analyze_response and response:
            try:
                rd, _ = _analyze_and_record(
                    response, category='model_response',
                    compute_kl=True, compute_trajectory=False,
                    capture_responses=False,
                    compute_ltp=compute_ltp, compute_sfd=compute_sfd,
                    skip_plots=True,
                )
                if session and session.results:
                    session.results[-1]['role'] = 'assistant'
                result['response_analyzed'] = True
            except Exception as e:
                logging.exception("Chat response analysis failed")
                result['response_analysis_error'] = str(e)

        return JSONResponse(content=result)
    except Exception as e:
        logging.exception("Chat generation failed")
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """Serve the chat interface."""
    chat_html = Path(__file__).parent / "static" / "chat.html"
    if chat_html.exists():
        return HTMLResponse(content=chat_html.read_text())
    return HTMLResponse(content="<p>chat.html not found</p>", status_code=404)


@app.get("/domain_surface_viz", response_class=HTMLResponse)
async def domain_surface_viz():
    """Serve the domain surface interactive visualization."""
    viz_html = Path(__file__).parent / "static" / "domain_surface_viz.html"
    if viz_html.exists():
        return HTMLResponse(content=viz_html.read_text())
    return HTMLResponse(content="<p>domain_surface_viz.html not found</p>", status_code=404)


@app.get("/correction_manifold_viz", response_class=HTMLResponse)
async def correction_manifold_viz():
    """Serve the correction manifold interactive visualization."""
    viz_html = Path(__file__).parent / "static" / "correction_manifold_viz.html"
    if viz_html.exists():
        return HTMLResponse(content=viz_html.read_text())
    return HTMLResponse(content="<p>correction_manifold_viz.html not found</p>", status_code=404)


@app.get("/correction_heatmap_viz", response_class=HTMLResponse)
async def correction_heatmap_viz():
    """Serve the correction heatmap interactive visualization."""
    viz_html = Path(__file__).parent / "static" / "correction_heatmap_viz.html"
    if viz_html.exists():
        return HTMLResponse(content=viz_html.read_text())
    return HTMLResponse(content="<p>correction_heatmap_viz.html not found</p>", status_code=404)


@app.get("/correction_backscatter_viz", response_class=HTMLResponse)
async def correction_backscatter_viz():
    """Serve the correction backscatter interactive visualization."""
    viz_html = Path(__file__).parent / "static" / "correction_backscatter_viz.html"
    if viz_html.exists():
        return HTMLResponse(content=viz_html.read_text())
    return HTMLResponse(content="<p>correction_backscatter_viz.html not found</p>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
