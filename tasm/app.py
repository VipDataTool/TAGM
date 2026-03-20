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

from engine.model_manager import ModelManager, KNOWN_PAIRS, _load_model_registry, _save_model_registry
from engine.analyzer import Analyzer, result_to_dict
from engine.baselines import BaselineManager, get_all_prompts, add_prompt as csv_add_prompt
from engine.statistics import aggregate_batch
from engine.visualizations import (
    plot_signed_attribution, plot_stress_per_token,
    plot_amplitude_trajectory, plot_heatmap,
    plot_distribution_metrics, plot_batch_summary,
    plot_separability,
    # LTP visualizations
    plot_ltp_profiles, plot_ltp_tension_magnitudes,
    plot_ltp_dual_trajectory, plot_ltp_summary_stats,
    plot_ltp_profile_heatmap,
    # SFD visualizations
    plot_sfd_density, plot_sfd_energy, plot_sfd_entropy,
    plot_rank_displacement,
)
from engine.comparative import generate_all_comparative
from engine.dataset import DatasetSession
from engine.reports import generate_single_report, generate_batch_report

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
baselines = BaselineManager()
session: Optional[DatasetSession] = None
progress_log = []
loading_state = {"active": False, "error": None}
user_info = {"name": "", "organization": ""}

# Locks protecting shared mutable state from concurrent access.
# _analysis_lock: serializes forward passes, activation caches, and session writes.
# _loading_lock: makes the loading_state check-and-set atomic.
_analysis_lock = threading.Lock()
_loading_lock = threading.Lock()


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
    valid = {"benign", "mild", "harmful", "jailbreak", "adversarial", ""}
    cat = category.strip().lower()
    return cat if cat in valid else "unknown"


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TASM Analyzer starting up")
    yield
    logger.info("TASM Analyzer shutting down")

app = FastAPI(title="TASM Analyzer", lifespan=lifespan)


# ─── Background model loading ────────────────────────────────────

def _load_model_worker(pair_id, base_id, instruct_id):
    global analyzer, baselines, session
    try:
        mm.load_pair(pair_id=pair_id, base_id=base_id,
                     instruct_id=instruct_id, callback=log_progress)
        analyzer = Analyzer(mm)
        baselines = BaselineManager()
        # Baselines are NOT computed automatically — the user enables them
        # via the toggle, which triggers /api/baselines/compute.
        session = DatasetSession()
        session.set_model(mm.state.display_name)
        loading_state["active"] = False
        loading_state["error"] = None
        log_progress("ready", "Model loaded. Session started. Ready to analyze.")
    except Exception as e:
        loading_state["active"] = False
        loading_state["error"] = str(e)
        logger.error(f"Model loading failed: {traceback.format_exc()}")
        log_progress("error", f"Loading failed: {e}")


def _analyze_and_record(prompt, category, compute_kl, compute_trajectory,
                        capture_responses, compute_ltp=False,
                        compute_sfd=False,
                        full_capture=False,
                        ltp_k=8, ltp_layer_strategy="signal",
                        ltp_svd_rank=0, ltp_tuned_lens=False,
                        skip_plots=False):
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
            ltp_tuned_lens=ltp_tuned_lens)

        baselines.normalize_result(result)

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
                plots["sfd_energy"] = plot_sfd_energy(result)
                plots["sfd_entropy"] = plot_sfd_entropy(result)
            if result.rank_displacement is not None:
                plots["rank_displacement"] = plot_rank_displacement(result)

        result_dict = result_to_dict(result)
        if session:
            session.add_result(result_dict, plots)

    return result_dict, plots


def _generate_deferred_plots(sess):
    """Generate per-prompt plots for results that were analyzed in batch mode
    (skip_plots=True). Reconstitutes minimal PromptResult objects from stored
    dicts and generates plots one at a time to control memory."""
    from engine.analyzer import PromptResult
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
                plots["sfd_energy"] = plot_sfd_energy(pr)
                plots["sfd_entropy"] = plot_sfd_entropy(pr)
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
    return {
        "model_loaded": mm.is_loaded() and not loading_state["active"],
        "loading": loading_state["active"],
        "loading_error": loading_state["error"],
        "model_pair": mm.state.pair_id if mm.state else None,
        "model_name": mm.state.display_name if mm.state else None,
        "available_pairs": {k: v[2] for k, v in KNOWN_PAIRS.items()},
        "baseline_summary": baselines.get_summary(),
        "session": {
            "n_results": session.n_results if session else 0,
            "categories": session.categories if session else {},
        } if session else None,
        "user_info": user_info,
    }


@app.post("/api/user_info")
async def set_user_info(name: str = Form(""), organization: str = Form("")):
    user_info["name"] = name.strip()
    user_info["organization"] = organization.strip()
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
    log_progress("starting", "Starting model load...")

    thread = threading.Thread(
        target=_load_model_worker,
        args=(pair_id, base_id, instruct_id), daemon=True)
    thread.start()
    return {"ok": True, "message": "Loading started."}


@app.post("/api/reset")
async def reset_all():
    global analyzer, baselines, session
    progress_log.clear()
    if session:
        session.clear()
        session = None
    mm.reset()
    analyzer = None
    baselines = BaselineManager()
    loading_state["active"] = False
    loading_state["error"] = None
    logger.info("Full reset performed")
    log_progress("reset", "All resources released. Session cleared.")
    return {"ok": True, "message": "Reset complete."}


@app.post("/api/recalibrate")
async def recalibrate_classifier():
    """Recalibrate the v2 classifier's class parameters from the current
    session's labeled results. Requires at least 2 prompts per category
    to compute stable mean/std estimates. Reports what changed."""
    if not session or session.n_results < 4:
        return JSONResponse(status_code=400,
                            content={"error": "Need at least 4 labeled prompts in session."})

    from engine.classifier import update_params, CLASS_PARAMS, CLASSES, FEATURES

    results = session.results
    new_params = update_params(results)

    # Build a diff report showing what changed
    changes = []
    for cls in CLASSES:
        for feat in FEATURES:
            old_mu, old_sigma = CLASS_PARAMS[cls][feat]
            new_mu, new_sigma = new_params[cls][feat]
            if abs(new_mu - old_mu) > 1e-6 or abs(new_sigma - old_sigma) > 1e-6:
                changes.append({
                    "class": cls, "feature": feat,
                    "old_mean": round(old_mu, 6), "new_mean": round(new_mu, 6),
                    "old_std": round(old_sigma, 6), "new_std": round(new_sigma, 6),
                    "delta_mean": round(new_mu - old_mu, 6),
                })

    # Count per-category sample sizes
    cat_counts = {}
    for r in results:
        cat = r.get("category", "")
        if cat in CLASSES:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Apply the new parameters
    for cls in CLASSES:
        CLASS_PARAMS[cls] = new_params[cls]

    logger.info(f"Classifier v2 recalibrated from {len(results)} prompts: "
                f"{len(changes)} parameters changed, categories={cat_counts}")

    return {
        "ok": True,
        "n_prompts": len(results),
        "category_counts": cat_counts,
        "n_changes": len(changes),
        "changes": changes,
        "message": f"v2 classifier recalibrated from {len(results)} prompts. "
                   f"{len(changes)} parameters updated.",
    }


@app.get("/api/progress")
async def get_progress():
    return {"log": progress_log}


# ─── Baselines ────────────────────────────────────────────────────

@app.get("/api/baselines")
async def get_baselines():
    """Return baseline system status."""
    return {"ok": True, "baselines": baselines.get_summary()}


@app.post("/api/baselines/toggle")
async def toggle_baselines(request: Request):
    """Enable or disable baseline length normalization.
    When enabling for the first time, triggers computation from baselines.csv."""
    if not analyzer:
        return JSONResponse(status_code=400, content={"error": "No model loaded."})
    try:
        body = await request.json()
    except Exception:
        body = {}
    enable = body.get("enabled", not baselines.enabled)

    if enable and not baselines._computed:
        # First time enabling — compute from baselines.csv
        log_progress("baseline", "Computing baselines (first enable)...")
        import asyncio
        def _compute():
            with _analysis_lock:
                baselines.compute_baselines(analyzer, callback=log_progress)
        await asyncio.to_thread(_compute)
    elif enable:
        baselines.enabled = True
        log_progress("baseline", "Baselines enabled (already computed)")
    else:
        baselines.enabled = False
        log_progress("baseline", "Baselines disabled — _ln fields will be None")

    return {"ok": True, "baselines": baselines.get_summary()}


@app.post("/api/baselines/recompute")
async def recompute_baselines():
    """Force recomputation of baselines from baselines.csv."""
    if not analyzer:
        return JSONResponse(status_code=400, content={"error": "No model loaded."})
    log_progress("baseline", "Recomputing baselines from baselines.csv...")
    import asyncio
    def _compute():
        with _analysis_lock:
            baselines.compute_baselines(analyzer, callback=log_progress)
    await asyncio.to_thread(_compute)
    return {"ok": True, "baselines": baselines.get_summary()}


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
                         ltp_tuned_lens: bool = Form(False)):
    # Validation
    err = _validate_prompt(prompt)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    if not analyzer:
        return JSONResponse(status_code=400, content={"error": "No model loaded."})

    category = _validate_category(category)

    try:
        logger.info(f"Analyzing: [{category}] {prompt[:60]}... (LTP={compute_ltp}, SFD={compute_sfd}, k={ltp_k}, strategy={ltp_layer_strategy}, svd={ltp_svd_rank}, tl={ltp_tuned_lens})")
        if compute_kl or capture_responses or compute_ltp:
            mm.load_base_for_kl(callback=log_progress)

        result_dict, plots = _analyze_and_record(
            prompt, category, compute_kl, compute_trajectory, capture_responses,
            full_capture=full_capture,
            compute_ltp=compute_ltp, compute_sfd=compute_sfd,
            ltp_k=ltp_k, ltp_layer_strategy=ltp_layer_strategy,
            ltp_svd_rank=ltp_svd_rank, ltp_tuned_lens=ltp_tuned_lens)

        if compute_kl or capture_responses or compute_ltp:
            mm.unload_base()

        return sanitize_for_json({
            "ok": True,
            "result": result_dict,
            "plots": plots,
            "session_n": session.n_results if session else 0,
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
                        ltp_svd_rank: int = Form(0),
                        ltp_tuned_lens: bool = Form(False),
                        baseline_file: Optional[UploadFile] = File(None)):
    if not analyzer:
        return JSONResponse(status_code=400, content={"error": "No model loaded."})

    # Read files on the event loop (fast async I/O)
    content = (await file.read()).decode("utf-8")
    bl_content = None
    if baseline_file:
        bl_content = (await baseline_file.read()).decode("utf-8")
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
    threading.Thread(
        target=_run_batch_sync,
        args=(content, bl_content, filename,
              compute_kl, compute_trajectory, capture_responses,
              full_capture,
              compute_ltp, compute_sfd, ltp_k, ltp_layer_strategy,
              ltp_svd_rank, ltp_tuned_lens),
        daemon=True).start()

    return {"ok": True, "started": True, "n_prompts": len(prompts),
            "message": f"Batch started: {len(prompts)} prompts. Watch progress log."}


def _run_batch_sync(content, bl_content, filename,
                    compute_kl, compute_trajectory, capture_responses,
                    full_capture,
                    compute_ltp, compute_sfd, ltp_k, ltp_layer_strategy,
                    ltp_svd_rank, ltp_tuned_lens):
    """Synchronous batch processing — runs in a background thread."""
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

        logger.info(f"Batch: {len(prompts)} prompts from {filename} (LTP={compute_ltp}, SFD={compute_sfd}, full_capture={full_capture}, svd={ltp_svd_rank}, tl={ltp_tuned_lens})")
        log_progress("batch", f"Loaded {len(prompts)} prompts from CSV")

        if bl_content:
            bl_reader = csv.DictReader(io.StringIO(bl_content))
            bl_prompts = [(row.get("prompt") or row.get("Prompt") or "").strip()
                          for row in bl_reader]
            bl_prompts = [p for p in bl_prompts if p]
            if bl_prompts:
                baselines.add_user_baselines(bl_prompts, analyzer, callback=log_progress)

        if compute_kl or capture_responses or compute_ltp:
            mm.load_base_for_kl(callback=log_progress)

        for i, p in enumerate(prompts):
            log_progress("analyzing", f"[{i+1}/{len(prompts)}] {p['prompt'][:60]}...")
            try:
                _analyze_and_record(
                    p["prompt"], p["category"],
                    compute_kl, compute_trajectory, capture_responses,
                    compute_ltp=compute_ltp, compute_sfd=compute_sfd,
                    full_capture=full_capture,
                    ltp_k=ltp_k,
                    ltp_layer_strategy=ltp_layer_strategy,
                    ltp_svd_rank=ltp_svd_rank, ltp_tuned_lens=ltp_tuned_lens,
                    skip_plots=(len(prompts) > 100))
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

        if compute_kl or capture_responses or compute_ltp:
            mm.unload_base()

        log_progress("done", f"Batch complete: {len(prompts)} prompts added to session")

    except Exception as e:
        logger.error(f"Batch failed: {traceback.format_exc()}")
        log_progress("error", f"Batch failed: {str(e)[:100]}")


@app.get("/api/session/results")
async def get_session_results():
    """Return all session results with per-prompt plots loaded from disk."""
    if not session or session.n_results == 0:
        return {"ok": False, "results": []}
    results = []
    for i, r in enumerate(session.results):
        r_copy = dict(r)
        # Load plots from disk
        plots = {}
        plot_dir = session.session_dir / "plots" / "individual"
        if plot_dir.exists():
            for pfile in sorted(plot_dir.glob(f"{i:04d}_*.png")):
                plot_name = pfile.stem[5:]  # strip "0000_"
                try:
                    plots[plot_name] = base64.b64encode(pfile.read_bytes()).decode()
                except Exception:
                    pass
        r_copy["_plots"] = plots
        results.append(r_copy)
    return {"ok": True, "results": results}


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
    ltp_tuned_lens = opts.get("ltp_tuned_lens", False)

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

    # Load base model if needed
    needs_base = compute_kl or capture_responses or compute_ltp
    if needs_base:
        mm.load_base_for_kl(callback=log_progress)

    # Rerun in the request thread (these are individual prompts, not a huge batch)
    rerun_count = 0
    for item in to_rerun:
        try:
            _analyze_and_record(
                item["prompt"], item["category"],
                compute_kl, compute_trajectory, capture_responses,
                full_capture=full_capture,
                compute_ltp=compute_ltp, compute_sfd=compute_sfd,
                ltp_k=ltp_k,
                ltp_layer_strategy=ltp_layer_strategy,
                ltp_svd_rank=ltp_svd_rank, ltp_tuned_lens=ltp_tuned_lens)
            rerun_count += 1
        except Exception as e:
            logger.error(f"Rerun failed for '{item['prompt'][:40]}': {e}")

    if needs_base:
        mm.unload_base()

    # Rewrite CSV from scratch
    session._csv_initialized = False
    if session.csv_path.exists():
        session.csv_path.unlink()
    for r in session.results:
        session._write_csv_row(r)

    logger.info(f"Reran {rerun_count}/{len(to_rerun)} prompts")
    return {"ok": True, "rerun": rerun_count, "total": session.n_results}


@app.get("/api/dashboard")
async def get_dashboard():
    if not session or session.n_results == 0:
        return {"ok": False, "error": "No data in session."}

    import asyncio
    try:
        return await asyncio.to_thread(_run_dashboard_sync)
    except Exception as e:
        logger.error(f"Dashboard failed: {traceback.format_exc()}")
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": str(e)})


def _run_dashboard_sync():
    """Synchronous dashboard generation — runs in a thread."""
    logger.info(f"Dashboard request: {session.n_results} prompts in session")
    results = session.results
    from engine.analyzer import PromptResult
    pr_list = [PromptResult.from_dict(r, mode="scalar") for r in results]

    agg = aggregate_batch(pr_list)
    agg_plots = {"batch_summary": plot_batch_summary(agg),
                 "separability": plot_separability(agg)}
    comp_plots = generate_all_comparative(results)
    all_plots = {**agg_plots, **comp_plots}

    session.save_comparative_plots(all_plots)
    session.save_aggregate_json(agg)
    session.save_results_json()

    logger.info(f"Dashboard generated: {session.n_results} prompts")

    slim_results = []
    for r in results:
        slim = {k: r.get(k) for k in [
            "prompt", "category", "seq_len", "stress_score", "net_correction",
            "entropy", "gini", "top2_share", "middle_share", "interior_cv",
            "kl_divergence", "stress_score_ln", "entropy_ln", "top2_share_ln",
            "middle_share_ln", "n_negative_tokens", "has_negative_tokens",
            "instruct_topk", "base_topk", "base_counterfactual_tokens",
            # Terrain viewer needs these per-token arrays
            "tokens", "per_token_kl", "signed_attr", "per_token_stress",
        ]}
        ltp = r.get("ltp")
        if ltp:
            slim["ltp"] = {k: ltp.get(k) for k in [
                "mean_M", "mean_C", "mean_V", "mean_L",
                "max_prc", "n_directional",
                "layer_strategy", "k", "svd_rank", "tuned_lens",
                # Terrain viewer needs these per-token arrays
                "profiles", "base_profiles", "counterfactual_tokens",
                "tension_magnitudes", "profile_shapes",
            ]}
        # SFD data
        sfd = r.get("sfd")
        if sfd:
            slim["sfd"] = {k: sfd.get(k) for k in [
                "density_mean", "density_max", "density_var", "density_p90",
                "entropy_mean", "entropy_max", "entropy_var", "entropy_p90",
                "energy_mean", "energy_max", "energy_var", "energy_p90",
                "global_erank", "n_layers_monitored", "k",
                "per_token_density", "per_token_entropy", "per_token_energy",
            ]}
        # Rank displacement
        slim["rank_displacement"] = r.get("rank_displacement")
        slim_results.append(slim)

    return sanitize_for_json({
        "ok": True, "aggregate": agg, "plots": all_plots, "results": slim_results,
        "session_info": {
            "n_results": session.n_results, "categories": session.categories,
            "model": session.model_name, "timestamp": session.timestamp,
        },
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
        "pdf": opts.get("pdf", True),
        "json": opts.get("json", True),
        "charts": opts.get("charts", True),
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
        opts = {"csv": True, "pdf": True, "json": True, "charts": True, "exportPath": ""}
    import shutil

    do_pdf = opts.get("pdf", True)
    do_json = opts.get("json", True)
    do_charts = opts.get("charts", True)
    export_path = opts.get("exportPath", "").strip()

    # Aggregate stats are needed for PDF and charts, so compute if either is on
    agg = None
    all_plots = {}
    if do_pdf or do_charts:
        log_progress("exporting", "Generating aggregate statistics...")
        try:
            results = session.results
            from engine.analyzer import PromptResult
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
            log_progress("exporting", "Saving full results JSON...")
            session.save_results_json()
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


@app.get("/api/log")
async def get_log():
    """Return the log file for download."""
    if LOG_FILE.exists():
        return FileResponse(str(LOG_FILE), media_type="text/plain", filename="tasm.log")
    return JSONResponse(status_code=404, content={"error": "No log file."})


CONFIG_FILE = Path(__file__).parent / "ui_config.json"

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


@app.post("/api/chat")
async def chat(request: Request):
    """Generate text from the loaded instruct model, optionally with analysis."""
    if analyzer is None or analyzer.mm.state is None or analyzer.mm.state.model_instruct is None:
        return JSONResponse(content={"ok": False, "error": "No model loaded"})

    try:
        body = await request.json()
        messages = body.get("messages", [])
        max_tokens = min(body.get("max_tokens", 256), 512)
        do_analyze = body.get("analyze", False)
        category = body.get("category", "")
        compute_ltp = body.get("compute_ltp", True)
        compute_sfd = body.get("compute_sfd", True)

        if not messages:
            return JSONResponse(content={"ok": False, "error": "No messages"})

        prompt_text = messages[-1].get("content", "")

        with _analysis_lock:
            state = analyzer.mm.state
            model = state.model_instruct
            tokenizer = state.tokenizer
            device = state.device

            # Build chat template from full history
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        result = {"ok": True, "response": response}

        # Optionally run analysis on the user's prompt
        if do_analyze and prompt_text:
            try:
                result_dict, plots = _analyze_and_record(
                    prompt_text, category=category,
                    compute_kl=True, compute_trajectory=False,
                    capture_responses=False,
                    compute_ltp=compute_ltp, compute_sfd=compute_sfd,
                    skip_plots=True,
                )
                # Return a slim analysis payload
                slim = {}
                for k in ['stress_score', 'net_correction', 'entropy', 'middle_share',
                          'interior_cv', 'gini', 'top2_share', 'kl_divergence',
                          'seq_len', 'n_negative_tokens']:
                    if k in result_dict:
                        slim[k] = result_dict[k]
                if result_dict.get('ltp'):
                    slim['ltp'] = {k: result_dict['ltp'].get(k) for k in
                                   ['mean_M', 'mean_C', 'mean_V', 'mean_L'] if result_dict['ltp'].get(k) is not None}
                if result_dict.get('sfd'):
                    slim['sfd'] = result_dict['sfd']
                if result_dict.get('rank_displacement'):
                    slim['rank_displacement'] = result_dict['rank_displacement']
                if result_dict.get('classifiers'):
                    slim['classifiers'] = {}
                    for cid, cl in result_dict['classifiers'].items():
                        slim['classifiers'][cid] = {
                            'predicted': cl.get('predicted'),
                            'confidence': cl.get('confidence'),
                        }
                result['analysis'] = slim
            except Exception as e:
                logging.exception("Chat analysis failed")
                result['analysis_error'] = str(e)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
