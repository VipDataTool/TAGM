"""TAGM app core: TASM-native API surface backed by TAGM's engine.

This module provides the global state, startup logic, and all critical
endpoint handlers. It replaces the orchestrator-based flow with the
engine's Analyzer, producing flat result dicts the frontend reads directly.

Integration: replace the corresponding sections in app.py with these
functions. The static file serving, probe management, chat, and viz
endpoints can remain as-is — they don't touch the analyze pipeline.
"""
from __future__ import annotations

import json
import math
import time
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from tagm.core.pipeline import Pipeline
from tagm.engine.analyzer import Analyzer
from tagm.engine.result import result_to_dict
from tagm.engine import config as engine_config
from tagm.engine.session import Session

logger = logging.getLogger("tagm")

# app_core.py lives at tagm/tagm/engine/app_core.py
# Project root (where models.json, start.sh, static/ live) is 3 levels up
_PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_FILE = _PROJECT_ROOT / "models.json"
CONFIG_FILE = _PROJECT_ROOT / "ui_config.json"

# Log at import time so path issues are visible immediately
logger.info(f"[app_core] Project root: {_PROJECT_ROOT}")
logger.info(f"[app_core] MODELS_FILE: {MODELS_FILE} (exists={MODELS_FILE.exists()})")


# ─── Global state ───────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.analyzer: Optional[Analyzer] = None
        self.session = Session()
        self.loading_state = {"active": False, "error": None}
        self.progress_log: list[dict] = []
        self.user_info: dict = {"name": "", "organization": "", "project": ""}
        self.inference_class: str = "instruct"
        self._post_load_callbacks: list = []

    def on_model_loaded(self, callback):
        """Register a callback to run after model load completes.
        Callback receives the pipeline as its argument."""
        self._post_load_callbacks.append(callback)

    def _fire_post_load(self):
        """Call all registered post-load callbacks."""
        for cb in self._post_load_callbacks:
            try:
                cb(self.pipeline)
            except Exception as e:
                logger.warning(f"[post_load] Callback failed: {e}")

    def progress(self, stage: str, message: str):
        self.progress_log.append({
            "stage": stage, "message": message, "time": time.time(),
        })
        logger.info(f"[{stage}] {message}")
        if len(self.progress_log) > 1000:
            self.progress_log = self.progress_log[-500:]


state = AppState()
_analysis_lock = threading.Lock()
_loading_lock = threading.Lock()
_batch_running = False
_batch_lock = threading.Lock()


# ─── Model registry ─────────────────────────────────────────────

def _load_model_registry() -> list[dict]:
    if not MODELS_FILE.exists():
        return []
    with open(MODELS_FILE) as f:
        return json.load(f)


# ─── Status ─────────────────────────────────────────────────────

def api_status_handler():
    """GET /api/status — returns TASM's native field shape."""
    model_loaded = (state.pipeline is not None
                    and state.pipeline.loaded
                    and not state.loading_state.get("active"))
    current_model = ""
    if model_loaded and state.pipeline:
        current_model = state.pipeline.instruct_model_id

    disk_info = Session.has_session_on_disk() if state.session.n_results == 0 else None

    return {
        "model_loaded": model_loaded,
        "loading": state.loading_state.get("active", False),
        "loading_error": state.loading_state.get("error"),
        "current_model": current_model,
        "model_pair": state.pipeline.instruct_model_id if model_loaded else None,
        "model_name": current_model,
        "session": {
            "n_results": state.session.n_results,
            "categories": state.session.categories,
            "cache_size_bytes": state.session.get_cache_size(),
            "model": state.session.model_name,
        } if state.session else None,
        "session_id": state.session.session_id,
        "n_results": state.session.n_results,
        "cache_bytes": state.session.get_cache_size(),
        "restorable": disk_info,
        "user_info": state.user_info,
    }


# ─── Config (UI preferences store) ──────────────────────────────

def api_config_get_handler():
    """GET /api/config — load saved UI preferences."""
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            return {"ok": True, "config": data}
        except Exception:
            pass
    return {"ok": True, "config": {}}


def api_config_post_handler(data: dict):
    """POST /api/config — save UI preferences to disk."""
    try:
        CONFIG_FILE.write_text(json.dumps(data, indent=2))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── Model loading ──────────────────────────────────────────────

def _load_worker(instruct_id: str, base_id: str,
                 layer_filter=None, compute_spectral=True):
    """Background worker: load model, compute deltas, create Analyzer."""
    import traceback
    logger.info(f"[LOAD] Thread started: {instruct_id} / {base_id}")
    state.progress("loading", "Worker thread started")

    try:
        if state.pipeline is not None:
            state.progress("loading", "Unloading previous pipeline")
            state.pipeline.unload()

        state.progress("loading", f"Downloading weights: {instruct_id}")
        state.pipeline = Pipeline(
            instruct_model_id=instruct_id,
            base_model_id=base_id,
            device="cpu",
            dtype=torch.bfloat16,
        )
        state.pipeline.load(
            layer_filter=layer_filter,
            compute_spectral=compute_spectral,
            progress=state.progress,
        )

        # Create the Analyzer — the engine that produces TASM-shaped results
        state.analyzer = Analyzer(state.pipeline)
        state.session = Session()
        state.session.set_model(instruct_id)

        state.loading_state = {"active": False, "error": None}
        state._fire_post_load()
        state.progress("ready", f"Model pair loaded: {instruct_id}")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[LOAD] FAILED:\n{tb}")
        state.loading_state = {"active": False, "error": str(e)}
        state.progress("error", f"Load failed: {type(e).__name__}: {e}")


def api_load_model_handler(pair_id: str = "", instruct_id: str = "",
                           base_id: str = "", layer_filter_raw: str = "",
                           compute_spectral_raw: str = "true"):
    """POST /api/load_model — form data, starts background load."""
    if pair_id and not (instruct_id and base_id):
        pairs = _load_model_registry()
        match = next((p for p in pairs if p.get("id") == pair_id), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Unknown pair_id '{pair_id}'")
        instruct_id = match.get("instruct", "")
        base_id = match.get("base", "")

    if not instruct_id or not base_id:
        raise HTTPException(status_code=400,
                            detail="Provide pair_id or both instruct_id and base_id")

    if state.loading_state.get("active"):
        raise HTTPException(status_code=409, detail="Load already in progress")

    layer_filter = None
    if layer_filter_raw.strip():
        try:
            layer_filter = [int(x.strip()) for x in layer_filter_raw.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid layer_filter")

    compute_spectral = compute_spectral_raw.lower() not in ("0", "false", "no", "")

    state.loading_state = {"active": True, "error": None}
    state.progress("loading", f"Starting load: {instruct_id} / {base_id}")

    thread = threading.Thread(
        target=_load_worker,
        args=(instruct_id, base_id, layer_filter, compute_spectral),
        daemon=True,
    )
    thread.start()

    return {"ok": True, "started": True,
            "message": f"Loading {instruct_id} in background."}


# ─── Analyze (single prompt) ────────────────────────────────────

def _plot_keys_for_result(r: dict) -> list[str]:
    """Which plots are available for a result, without rendering them."""
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
        keys.append("sfd_density")
    if r.get("rank_displacement"):
        keys.append("rank_displacement")
    return keys


async def api_analyze_handler(request: Request):
    """POST /api/analyze — accepts form data with TASM flags.

    Reads per-request flags, calls analyzer.analyze_prompt() with those
    flags, serializes via result_to_dict(), stores in session, returns
    the flat result dict the frontend reads.
    """
    if state.analyzer is None:
        return {"ok": False, "error": "No model loaded."}

    form = await request.form()
    prompt = (form.get("prompt") or "").strip()
    category = (form.get("category") or "").strip()

    if not prompt:
        return {"ok": False, "error": "prompt required"}
    if len(prompt) > 5000:
        return {"ok": False, "error": "Prompt too long (max 5000)"}

    # Read per-request flags — this IS how TASM works
    compute_kl = _form_bool(form, "compute_kl")
    compute_trajectory = _form_bool(form, "compute_trajectory", default=True)
    capture_responses = _form_bool(form, "capture_responses")
    full_capture = _form_bool(form, "full_capture")
    compute_ltp = _form_bool(form, "compute_ltp")
    compute_sfd = _form_bool(form, "compute_sfd")
    ltp_k = int(form.get("ltp_k") or 8)
    ltp_layer_strategy = (form.get("ltp_layer_strategy") or "signal").strip()
    ltp_svd_rank = int(form.get("ltp_svd_rank") or 0)

    # Sequential base-model phase if needed
    needs_base = compute_kl or capture_responses or compute_ltp
    base_cache = None

    def _do_analyze():
        nonlocal base_cache
        with _analysis_lock:
            if needs_base:
                caches = state.analyzer.run_base_phase(
                    [{"prompt": prompt, "category": category}],
                    ltp_k=ltp_k,
                    compute_kl=compute_kl,
                    capture_responses=capture_responses,
                    progress=state.progress,
                )
                base_cache = caches[0] if caches else None

            result = state.analyzer.analyze_prompt(
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
                base_cache=base_cache,
            )
            return result_to_dict(result)

    try:
        result_dict = await run_in_threadpool(_do_analyze)
    except Exception as e:
        logger.exception("Analysis failed")
        return {"ok": False, "error": str(e)}

    # Store in session
    idx = state.session.add_result(result_dict)
    plot_keys = _plot_keys_for_result(result_dict)
    result_dict["_plot_keys"] = plot_keys

    # Snapshot to disk (non-blocking)
    await run_in_threadpool(state.session.save_to_disk)

    return _sanitize_for_json({
        "ok": True,
        "result": result_dict,
        "plot_keys": plot_keys,
        "session_n": state.session.n_results,
        "cache_size_bytes": state.session.get_cache_size(),
    })


# ─── Batch analyze ──────────────────────────────────────────────

async def api_analyze_batch_handler(request: Request):
    """POST /api/analyze_batch — form data with CSV file + TASM flags."""
    global _batch_running
    import csv as _csv
    import io as _io

    if state.analyzer is None:
        return {"ok": False, "error": "No model loaded."}

    with _batch_lock:
        if _batch_running:
            return {"ok": False, "error": "Batch already running."}

    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "read"):
        return {"ok": False, "error": "CSV file required. Select a file before clicking Run Batch."}

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    reader = _csv.DictReader(_io.StringIO(content))
    prompts = []
    for row in reader:
        p = (row.get("prompt") or row.get("Prompt") or "").strip()
        c = (row.get("category") or row.get("Category") or "unknown").strip()
        if p:
            prompts.append({"prompt": p, "category": c})

    if not prompts:
        return {"ok": False, "error": "No valid prompts in CSV. Needs 'prompt' column."}

    # Read flags
    compute_kl = _form_bool(form, "compute_kl")
    compute_trajectory = _form_bool(form, "compute_trajectory")
    capture_responses = _form_bool(form, "capture_responses")
    full_capture = _form_bool(form, "full_capture")
    compute_ltp = _form_bool(form, "compute_ltp")
    compute_sfd = _form_bool(form, "compute_sfd")
    ltp_k = int(form.get("ltp_k") or 8)
    ltp_layer_strategy = (form.get("ltp_layer_strategy") or "signal").strip()

    with _batch_lock:
        _batch_running = True

    def _run_batch():
        global _batch_running
        try:
            needs_base = compute_kl or capture_responses or compute_ltp
            all_base_caches = []

            if needs_base:
                all_base_caches = state.analyzer.run_base_phase(
                    prompts, ltp_k=ltp_k, compute_kl=compute_kl,
                    capture_responses=capture_responses,
                    progress=state.progress,
                )

            for i, p in enumerate(prompts):
                state.progress("analyzing",
                               f"[{i+1}/{len(prompts)}] {p['prompt'][:50]}...")
                try:
                    with _analysis_lock:
                        bc = all_base_caches[i] if i < len(all_base_caches) else None
                        result = state.analyzer.analyze_prompt(
                            p["prompt"], category=p.get("category", ""),
                            compute_kl=compute_kl,
                            compute_full_trajectory=compute_trajectory,
                            capture_responses=capture_responses,
                            full_capture=full_capture,
                            compute_ltp=compute_ltp,
                            compute_sfd=compute_sfd,
                            ltp_k=ltp_k,
                            ltp_layer_strategy=ltp_layer_strategy,
                            base_cache=bc,
                        )
                        rd = result_to_dict(result)
                        state.session.add_result(rd)
                except Exception as e:
                    logger.exception(f"Batch prompt {i} failed")
                    state.progress("error", f"Prompt {i} failed: {e}")

            state.session.save_to_disk()
            state.progress("done", f"Batch complete: {len(prompts)} prompts analyzed")
        except Exception as e:
            logger.exception("Batch failed")
            state.progress("error", f"Batch failed: {e}")
        finally:
            with _batch_lock:
                _batch_running = False

    threading.Thread(target=_run_batch, daemon=True).start()

    return {
        "ok": True, "started": True, "n_prompts": len(prompts),
        "message": f"Batch started: {len(prompts)} prompts.",
    }


# ─── Session results ────────────────────────────────────────────

def api_session_results_handler(page: int = 1, per_page: int = 10):
    """GET /api/session/results — paginated flat results."""
    results = state.session.results
    total = len(results)
    if total == 0:
        return {"ok": False, "results": [], "total": 0, "page": 1,
                "per_page": per_page, "total_pages": 0,
                "cache_size_bytes": state.session.get_cache_size()}

    per_page = max(1, min(per_page, 10000))
    total_pages = max(1, -(-total // per_page))
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = min(start + per_page, total)

    # Results already have _index; add _plot_keys if missing
    out = []
    for i in range(start, end):
        r = dict(results[i])
        if "_plot_keys" not in r:
            r["_plot_keys"] = _plot_keys_for_result(r)
        out.append(r)

    return {
        "ok": True, "results": out, "page": page,
        "per_page": per_page, "total": total,
        "total_pages": total_pages,
        "cache_size_bytes": state.session.get_cache_size(),
    }


def api_dashboard_handler(force: bool = False):
    """GET /api/dashboard — slim results for the data table."""
    results = state.session.results
    if not results:
        return {"ok": False, "error": "No data in session.",
                "results": [], "session_info": {"n_results": 0},
                "cache_size_bytes": state.session.get_cache_size()}

    slim = []
    for i, r in enumerate(results):
        s = {"_index": i}
        for k in ("prompt", "category", "seq_len", "stress_score",
                   "net_correction", "entropy", "top2_share",
                   "middle_share", "interior_cv", "kl_divergence",
                   "n_negative_tokens", "has_negative_tokens"):
            if k in r:
                s[k] = r[k]
        ltp = r.get("ltp")
        if isinstance(ltp, dict):
            s["ltp"] = {k: ltp.get(k) for k in ("mean_M", "mean_V", "max_prc", "n_directional")}
        sfd = r.get("sfd")
        if isinstance(sfd, dict):
            s["sfd"] = {"density_mean": sfd.get("density_mean")}
        rd = r.get("rank_displacement")
        if isinstance(rd, dict):
            s["rank_displacement"] = {k: rd.get(k) for k in
                                       ("mean_tau", "mean_overlap", "mean_replacement",
                                        "mean_disp_per_token", "total_displacement")}
        slim.append(s)

    session_info = {
        "n_results": len(results),
        "categories": state.session.categories,
        "session_id": state.session.session_id,
        "model": state.session.model_name,
    }

    return {
        "ok": True, "results": slim,
        "session_info": session_info,
        "cache_size_bytes": state.session.get_cache_size(),
    }


def api_results_detail_handler(start: int = 0, count: int = 50):
    """GET /api/results/detail — chunked detail window."""
    results = state.session.results
    total = len(results)
    start = max(0, start)
    count = max(1, min(count, 1000))
    end = min(start + count, total)

    out = []
    for i in range(start, end):
        r = dict(results[i])
        if "_plot_keys" not in r:
            r["_plot_keys"] = _plot_keys_for_result(r)
        out.append(r)

    return {"ok": True, "results": out, "start": start,
            "count": end - start, "total": total}


# ─── Progress / reset / user_info ───────────────────────────────

def api_progress_handler():
    """GET /api/progress"""
    return {"log": list(state.progress_log), "now": time.time()}


def api_reset_handler():
    """POST /api/reset — full reset."""
    if state.pipeline is not None:
        try:
            state.pipeline.unload()
        except Exception:
            pass
    state.pipeline = None
    state.analyzer = None
    state.session = Session()
    state.loading_state = {"active": False, "error": None}
    state.progress("reset", "Pipeline and session reset")
    return {"ok": True, "message": "Reset complete"}


def api_session_restore_handler():
    """POST /api/session/restore"""
    restored = Session.restore()
    if restored is None:
        return {"ok": False, "error": "No session to restore."}
    state.session = restored
    return {
        "ok": True,
        "n_results": restored.n_results,
        "model": restored.model_name,
        "message": f"Restored {restored.n_results} results",
    }


def api_session_remove_handler(indices: list[int]):
    """POST /api/session/remove — remove specific indices."""
    state.session.remove_indices(indices)
    state.session.save_to_disk()
    return {"ok": True, "n_results": state.session.n_results}


# ─── Helpers ────────────────────────────────────────────────────

def _form_bool(form, key: str, default: bool = False) -> bool:
    """Parse a form-data boolean field (handles 'true'/'false' strings)."""
    val = form.get(key)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes", "on")


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj
