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
import logging
import threading
import traceback
from pathlib import Path
from typing import Optional

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
        baselines.compute_builtin_baselines(analyzer, callback=log_progress)
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
                        ltp_k=8, ltp_layer_strategy="signal",
                        ltp_svd_rank=0, ltp_tuned_lens=False,
                        skip_plots=False):
    result = analyzer.analyze_prompt(
        prompt, category=category,
        compute_kl=compute_kl,
        compute_full_trajectory=compute_trajectory,
        capture_responses=capture_responses,
        compute_ltp=compute_ltp,
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

    result_dict = result_to_dict(result)
    if session:
        session.add_result(result_dict, plots)

    return result_dict, plots


def _generate_deferred_plots(sess):
    """Generate per-prompt plots for results that were analyzed in batch mode
    (skip_plots=True). Reconstitutes minimal PromptResult objects from stored
    dicts and generates plots one at a time to control memory."""
    from engine.analyzer import PromptResult
    from engine.ltp import LTPResult
    import numpy as np

    plot_dir = sess.session_dir / "plots" / "individual"
    generated = 0

    for idx, r in enumerate(sess.results):
        # Check if plots already exist for this prompt
        marker = plot_dir / f"{idx:04d}_stress_per_token.png"
        if marker.exists():
            continue

        try:
            # Reconstitute a minimal PromptResult for the plot functions
            pr = PromptResult()
            pr.tokens = r.get("tokens", [])
            pr.seq_len = r.get("seq_len", len(pr.tokens))
            pr.signed_attr = r.get("signed_attr", [])
            pr.per_token_stress = r.get("per_token_stress", [])
            pr.stress_score = r.get("stress_score", 0)
            pr.net_correction = r.get("net_correction", 0)
            pr.entropy = r.get("entropy", 0)
            pr.gini = r.get("gini", 0)
            pr.top2_share = r.get("top2_share", 0)
            pr.middle_share = r.get("middle_share", 0)
            pr.interior_cv = r.get("interior_cv", 0)
            pr.amplitude_trajectory = r.get("amplitude_trajectory", [])
            pr.amplitude_normalized = r.get("amplitude_normalized", [])
            pr.heatmap = r.get("heatmap", [])
            pr.signal_layer_indices = r.get("signal_layer_indices", [])

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
            ltp_data = r.get("ltp")
            if ltp_data and ltp_data.get("profiles"):
                ltp_r = LTPResult()
                ltp_r.profiles = [np.array(p) for p in ltp_data.get("profiles", [])]
                ltp_r.tension_magnitudes = ltp_data.get("tension_magnitudes", [])
                ltp_r.profile_shapes = ltp_data.get("profile_shapes", [])
                ltp_r.counterfactual_tokens = ltp_data.get("counterfactual_tokens", [])
                ltp_r.mean_M = ltp_data.get("mean_M", 0)
                ltp_r.mean_C = ltp_data.get("mean_C", 0)
                ltp_r.mean_V = ltp_data.get("mean_V", 0)
                ltp_r.mean_L = ltp_data.get("mean_L", 0)
                ltp_r.k = ltp_data.get("k", 8)
                ltp_r.offset_magnitude = {int(k): v for k, v in ltp_data.get("offset_magnitude", {}).items()}
                ltp_r.offset_consistency = {int(k): v for k, v in ltp_data.get("offset_consistency", {}).items()}
                sem = ltp_data.get("semantic_trajectory_2d", [])
                ten = ltp_data.get("tension_trajectory_2d", [])
                if sem:
                    ltp_r.semantic_trajectory = np.array(sem)
                if ten:
                    ltp_r.tension_trajectory = np.array(ten)

                plots["ltp_profiles"] = plot_ltp_profiles(ltp_r, pr.tokens)
                plots["ltp_tension_magnitudes"] = plot_ltp_tension_magnitudes(ltp_r, pr.tokens)
                plots["ltp_dual_trajectory"] = plot_ltp_dual_trajectory(ltp_r)
                plots["ltp_summary_stats"] = plot_ltp_summary_stats(ltp_r)
                plots["ltp_profile_heatmap"] = plot_ltp_profile_heatmap(ltp_r, pr.tokens)

            # Write to disk
            import base64 as b64mod
            for name, b64_str in plots.items():
                if b64_str:
                    path = plot_dir / f"{idx:04d}_{name}.png"
                    path.write_bytes(b64mod.b64decode(b64_str))

            generated += 1

            # GC every 10 plots batches
            if generated % 10 == 0:
                import gc as _gc
                _gc.collect()

        except Exception as e:
            logger.warning(f"Deferred plot {idx} failed: {e}")

    if generated > 0:
        logger.info(f"Generated deferred plots for {generated} prompts")


# ─── API Routes ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


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
    if loading_state["active"]:
        return {"ok": False, "message": "Already loading a model."}

    # Validate
    if not pair_id and not (base_id and instruct_id):
        return JSONResponse(status_code=400,
                            content={"error": "Select a model pair or provide custom IDs."})

    progress_log.clear()
    loading_state["active"] = True
    loading_state["error"] = None
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


@app.get("/api/progress")
async def get_progress():
    return {"log": progress_log}


@app.get("/api/prompts")
async def get_prompts():
    """Return all prompts from the unified prompts.csv."""
    return {"prompts": get_all_prompts()}


@app.post("/api/prompts")
async def add_prompt_route(prompt: str = Form(...),
                           category: str = Form("benign"),
                           baseline: bool = Form(False)):
    err = _validate_prompt(prompt)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    cat = _validate_category(category)
    csv_add_prompt(prompt, cat, baseline)
    logger.info(f"Prompt added to library: [{cat}] {prompt[:60]}...")
    return {"ok": True, "prompts": get_all_prompts()}


@app.post("/api/analyze")
async def analyze_single(prompt: str = Form(...),
                         category: str = Form(""),
                         compute_kl: bool = Form(False),
                         compute_trajectory: bool = Form(True),
                         capture_responses: bool = Form(False),
                         compute_ltp: bool = Form(False),
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
        logger.info(f"Analyzing: [{category}] {prompt[:60]}... (LTP={compute_ltp}, k={ltp_k}, strategy={ltp_layer_strategy}, svd={ltp_svd_rank}, tl={ltp_tuned_lens})")
        if compute_kl or capture_responses:
            mm.load_base_for_kl(callback=log_progress)

        result_dict, plots = _analyze_and_record(
            prompt, category, compute_kl, compute_trajectory, capture_responses,
            compute_ltp=compute_ltp, ltp_k=ltp_k, ltp_layer_strategy=ltp_layer_strategy,
            ltp_svd_rank=ltp_svd_rank, ltp_tuned_lens=ltp_tuned_lens)

        if compute_kl or capture_responses:
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
                        compute_ltp: bool = Form(False),
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
              compute_ltp, ltp_k, ltp_layer_strategy,
              ltp_svd_rank, ltp_tuned_lens),
        daemon=True).start()

    return {"ok": True, "started": True, "n_prompts": len(prompts),
            "message": f"Batch started: {len(prompts)} prompts. Watch progress log."}


def _run_batch_sync(content, bl_content, filename,
                    compute_kl, compute_trajectory, capture_responses,
                    compute_ltp, ltp_k, ltp_layer_strategy,
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

        logger.info(f"Batch: {len(prompts)} prompts from {filename} (LTP={compute_ltp}, svd={ltp_svd_rank}, tl={ltp_tuned_lens})")
        log_progress("batch", f"Loaded {len(prompts)} prompts from CSV")

        if bl_content:
            bl_reader = csv.DictReader(io.StringIO(bl_content))
            bl_prompts = [(row.get("prompt") or row.get("Prompt") or "").strip()
                          for row in bl_reader]
            bl_prompts = [p for p in bl_prompts if p]
            if bl_prompts:
                baselines.add_user_baselines(bl_prompts, analyzer, callback=log_progress)

        if compute_kl or capture_responses:
            mm.load_base_for_kl(callback=log_progress)

        for i, p in enumerate(prompts):
            log_progress("analyzing", f"[{i+1}/{len(prompts)}] {p['prompt'][:60]}...")
            try:
                _analyze_and_record(
                    p["prompt"], p["category"],
                    compute_kl, compute_trajectory, capture_responses,
                    compute_ltp=compute_ltp, ltp_k=ltp_k,
                    ltp_layer_strategy=ltp_layer_strategy,
                    ltp_svd_rank=ltp_svd_rank, ltp_tuned_lens=ltp_tuned_lens,
                    skip_plots=True)
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

        if compute_kl or capture_responses:
            mm.unload_base()

        log_progress("done", f"Batch complete: {len(prompts)} prompts added to session")

    except Exception as e:
        logger.error(f"Batch failed: {traceback.format_exc()}")
        log_progress("error", f"Batch failed: {str(e)[:100]}")


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
    pr_list = []
    for r in results:
        pr = PromptResult()
        for key in ["stress_score", "net_correction", "entropy", "gini",
                     "top2_share", "middle_share", "interior_cv",
                     "kl_divergence", "entropy_ln", "top2_share_ln",
                     "middle_share_ln", "stress_score_ln",
                     "n_negative_tokens", "has_negative_tokens",
                     "category", "seq_len"]:
            val = r.get(key)
            if val is not None:
                setattr(pr, key, val)

        ltp_data = r.get("ltp")
        if ltp_data:
            from engine.ltp import LTPResult
            ltp_r = LTPResult()
            ltp_r.mean_M = ltp_data.get("mean_M", 0.0) or 0.0
            ltp_r.mean_C = ltp_data.get("mean_C", 0.0) or 0.0
            ltp_r.mean_V = ltp_data.get("mean_V", 0.0) or 0.0
            ltp_r.mean_L = ltp_data.get("mean_L", 0.0) or 0.0
            pr.ltp = ltp_r

        pr_list.append(pr)

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
            "instruct_topk", "base_topk",
        ]}
        ltp = r.get("ltp")
        if ltp:
            slim["ltp"] = {k: ltp.get(k) for k in [
                "mean_M", "mean_C", "mean_V", "mean_L",
                "layer_strategy", "k", "svd_rank", "tuned_lens",
            ]}
        slim_results.append(slim)

    return sanitize_for_json({
        "ok": True, "aggregate": agg, "plots": all_plots, "results": slim_results,
        "session_info": {
            "n_results": session.n_results, "categories": session.categories,
            "model": session.model_name, "timestamp": session.timestamp,
        },
    })


@app.get("/api/export")
async def export_session():
    if not session or session.n_results == 0:
        return JSONResponse(status_code=400, content={"error": "No data to export."})

    # Run export in background thread
    threading.Thread(target=_run_export_sync, daemon=True).start()
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


def _run_export_sync():
    """Synchronous export — runs in a background thread."""
    log_progress("exporting", "Generating aggregate statistics...")
    try:
        results = session.results
        from engine.analyzer import PromptResult
        pr_list = []
        for r in results:
            pr = PromptResult()
            for key in ["stress_score", "net_correction", "entropy", "gini",
                         "top2_share", "middle_share", "interior_cv",
                         "kl_divergence", "category", "seq_len",
                         "has_negative_tokens", "n_negative_tokens"]:
                val = r.get(key)
                if val is not None: setattr(pr, key, val)

            ltp_data = r.get("ltp")
            if ltp_data:
                from engine.ltp import LTPResult
                ltp_r = LTPResult()
                ltp_r.mean_M = ltp_data.get("mean_M", 0.0) or 0.0
                ltp_r.mean_C = ltp_data.get("mean_C", 0.0) or 0.0
                ltp_r.mean_V = ltp_data.get("mean_V", 0.0) or 0.0
                ltp_r.mean_L = ltp_data.get("mean_L", 0.0) or 0.0
                pr.ltp = ltp_r

            pr_list.append(pr)

        log_progress("exporting", "Computing separability and comparative plots...")
        agg = aggregate_batch(pr_list)
        agg_plots = {"batch_summary": plot_batch_summary(agg),
                     "separability": plot_separability(agg)}
        comp_plots = generate_all_comparative(results)
        all_plots = {**agg_plots, **comp_plots}

        log_progress("exporting", "Generating PDF report...")
        pdf_path = generate_batch_report(
            agg, results, all_plots,
            model_name=session.model_name, n_prompts=session.n_results,
            user_info=user_info)
        import shutil
        shutil.copy2(pdf_path, session.session_dir / "report.pdf")
        session.save_comparative_plots(all_plots)
        session.save_aggregate_json(agg)
    except Exception as e:
        logger.error(f"Export PDF/stats failed: {e}")
        log_progress("warning", f"PDF generation failed: {str(e)[:80]}")

    try:
        log_progress("exporting", "Saving full results JSON...")
        session.save_results_json()
    except Exception as e:
        logger.error(f"Export results JSON failed: {e}")

    try:
        n = session.n_results
        log_progress("exporting", f"Generating per-prompt plots ({n} prompts)...")
        _generate_deferred_plots(session)
    except Exception as e:
        logger.error(f"Deferred plot generation failed: {e}")

    try:
        log_progress("exporting", "Packaging ZIP...")
        zip_bytes = session.export_zip()
        # Save to disk for download endpoint
        zip_path = session.session_dir / "tasm_session.zip"
        zip_path.write_bytes(zip_bytes)
        logger.info(f"Session exported: {session.n_results} prompts, {len(zip_bytes)/1024/1024:.1f}MB")
        log_progress("done", f"Export ready: {session.n_results} prompts. Click Download.")
    except Exception as e:
        logger.error(f"Export ZIP failed: {traceback.format_exc()}")
        log_progress("error", f"Export ZIP failed: {str(e)[:80]}")
    except Exception as e:
        logger.error(f"Export failed: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/log")
async def get_log():
    """Return the log file for download."""
    if LOG_FILE.exists():
        return FileResponse(str(LOG_FILE), media_type="text/plain", filename="tasm.log")
    return JSONResponse(status_code=404, content={"error": "No log file."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
