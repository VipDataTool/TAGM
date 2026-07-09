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
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from src.core.pipeline import Pipeline
from src.core.db import get_db, migrate_json_to_db
from src.engine.analyzer import Analyzer
from src.engine.result import result_to_dict
from src.engine import config as engine_config
from src.engine.session import Session
from src.engine.deconstruct import expand_prompts
from src.service.events import broker

logger = logging.getLogger("src")

# app_core.py lives at tagm/tagm/engine/app_core.py
# Project root (where models.json, start.sh, static/ live) is 3 levels up
_PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_FILE = _PROJECT_ROOT / "models.json"    # legacy, kept for reference
CONFIG_FILE = _PROJECT_ROOT / "ui_config.json"  # legacy, kept for reference

# Log at import time so path issues are visible immediately
logger.info(f"[app_core] Project root: {_PROJECT_ROOT}")

# ─── Database bootstrap + migration ─────────────────────────────
_db = get_db()
_migration_summary = migrate_json_to_db(_db, _PROJECT_ROOT)
if any(v for v in _migration_summary.values()):
    logger.info(f"[app_core] Migration summary: {_migration_summary}")


# ─── Global state ───────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.analyzer: Optional[Analyzer] = None
        self.session = Session(db=_db)
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
        broker.publish("progress", {"stage": stage, "message": message})


state = AppState()
# THE model lock. Analysis, chat, probe embedding, and every module that
# runs a forward pass or generate() all acquire this same lock — see
# src/core/locks.py for the hook-corruption rationale.
from src.core.locks import MODEL_LOCK as _analysis_lock
_loading_lock = threading.Lock()
# Job-level guard: at most one analysis job (single or batch) runs at a time.
# This is distinct from _analysis_lock, which serializes individual inference
# calls; _job_running serializes whole submissions so the unified async
# contract can't have two jobs in flight at once.
_job_running = False
_job_lock = threading.Lock()


def job_active() -> bool:
    """True while an analysis job (single or batch) is in flight."""
    with _job_lock:
        return _job_running


# ─── Model registry ─────────────────────────────────────────────

def _load_model_registry() -> list[dict]:
    return _db.list_models()


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
        "inference_class": state.pipeline.inference_class if model_loaded else "instruct",
    }


# ─── Config (UI preferences store) ──────────────────────────────

def api_config_get_handler():
    """GET /api/config — load saved UI preferences."""
    data = _db.get_config("ui")
    return {"ok": True, "config": data}


def api_config_post_handler(data: dict):
    """POST /api/config — save UI preferences."""
    try:
        _db.set_config("ui", data)
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
            svd_k=int(engine_config.get("delta_svd_k") or 64),
            progress=state.progress,
        )

        state.analyzer = Analyzer(state.pipeline)
        state.session = Session(db=_db)
        state.session.set_model(instruct_id)

        state.loading_state = {"active": False, "error": None}
        state._fire_post_load()
        state.progress("ready", f"Model pair loaded: {instruct_id}")

        broker.publish("model_loaded", {
            "model_name": instruct_id,
            "instruct": instruct_id,
            "base": base_id,
            "n_deltas": len(state.pipeline.delta_store) if state.pipeline.delta_store else 0,
        })

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[LOAD] FAILED:\n{tb}")
        state.loading_state = {"active": False, "error": str(e)}
        state.progress("error", f"Load failed: {type(e).__name__}: {e}")

        broker.publish("model_error", {
            "error": str(e),
            "stage": "loading",
        })


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
    if job_active():
        raise HTTPException(
            status_code=409,
            detail="An analysis job is running. Wait for it to finish "
                   "before loading a different model.")

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


def _read_analyze_flags(form, *, trajectory_default: bool) -> dict:
    """Read the TASM analyze flags shared by the single and batch paths.

    The only per-path difference is the trajectory default (single
    computes it by default, batch doesn't), so it's a parameter. Every
    other flag is read identically, which is why both handlers can share
    one analysis core.
    """
    return {
        "compute_kl": _form_bool(form, "compute_kl"),
        "compute_full_trajectory": _form_bool(form, "compute_trajectory",
                                              default=trajectory_default),
        "capture_responses": _form_bool(form, "capture_responses"),
        "full_capture": _form_bool(form, "full_capture"),
        "compute_ltp": _form_bool(form, "compute_ltp"),
        "compute_sfd": _form_bool(form, "compute_sfd"),
        "compute_ecm": _form_bool(form, "compute_ecm"),
        "ecm_harvest_tokens": int(form.get("ecm_harvest_tokens") or 0),
        "ltp_k": int(form.get("ltp_k") or 8),
        "ltp_layer_strategy": (form.get("ltp_layer_strategy") or "signal").strip(),
        "ltp_svd_rank": int(form.get("ltp_svd_rank") or 0),
    }


def _analyze_prompt_list(prompts: list[dict], flags: dict, *,
                         deconstruct: bool = False,
                         progress=None) -> tuple[list[dict], list[tuple]]:
    """Shared analysis body for both the single and batch endpoints.

    A single prompt is just a one-element list, so this is the one place
    prompts become result records — there is no separate single-vs-batch
    pipeline anymore. The two endpoints differ only in their response
    contract (sync inline result vs. async 'started'), which stays in the
    thin wrappers.

    ``prompts``: list of ``{"prompt", "category"}`` dicts.
    ``flags``:   analyze_prompt kwargs from ``_read_analyze_flags``.
    ``deconstruct``: when True, each prompt is expanded into its prefix
        ladder (see deconstruct.py) and every rung result carries
        ``family_index`` / ``rung_index`` so the records regroup later.
    ``progress``: optional ``(stage, message)`` callback for per-rung
        logging; pass None to keep the console quiet (the single path's
        behavior when not deconstructing).

    Returns ``(results, errors)``:
        results — result dicts analyzed AND persisted, in order
        errors  — ``(work_index, message)`` for any that failed
    Each result is persisted via ``add_result`` as it is produced, per
    the engine's 'no separate save step' contract.
    """
    work = expand_prompts(prompts, deconstruct)

    # One session-stable family base for this whole call, so ladders
    # never collide with families already stored (or from other calls).
    family_base = (state.session._db.next_family_base(state.session.session_id)
                   if deconstruct else 0)

    needs_base = (flags.get("compute_kl") or flags.get("capture_responses")
                  or flags.get("compute_ltp"))
    base_caches: list = []
    if needs_base:
        base_caches = state.analyzer.run_base_phase(
            work,
            ltp_k=flags.get("ltp_k", 8),
            compute_kl=flags.get("compute_kl", False),
            capture_responses=flags.get("capture_responses", False),
            progress=state.progress,
        )

    results: list[dict] = []
    errors: list[tuple] = []
    n = len(work)
    for i, p in enumerate(work):
        if progress:
            progress("analyzing", f"[{i+1}/{n}] {(p.get('prompt') or '')[:50]}...")
        try:
            with _analysis_lock:
                bc = base_caches[i] if i < len(base_caches) else None
                result = state.analyzer.analyze_prompt(
                    p["prompt"], category=p.get("category", ""),
                    compute_kl=flags.get("compute_kl", False),
                    compute_full_trajectory=flags.get("compute_full_trajectory", True),
                    capture_responses=flags.get("capture_responses", False),
                    full_capture=flags.get("full_capture", False),
                    compute_ltp=flags.get("compute_ltp", False),
                    compute_sfd=flags.get("compute_sfd", False),
                    ltp_k=flags.get("ltp_k", 8),
                    ltp_layer_strategy=flags.get("ltp_layer_strategy", "signal"),
                    ltp_svd_rank=flags.get("ltp_svd_rank", 0),
                    base_cache=bc,
                )
                rd = result_to_dict(result)
                if flags.get("compute_ecm"):
                    from src.engine.ecm_analysis import attach_ecm_analysis
                    attach_ecm_analysis(rd)
                if deconstruct:
                    rd["family_index"] = family_base + p.get("family_local", 0)
                    rd["rung_index"] = p.get("rung_index", 0)
                state.session.add_result(rd)
                results.append(rd)

                # ── ECM harvest: generate response + analyze it ───
                # The ECM checkbox is the single gate.  Token count
                # comes from the form field (falls back to config).
                harvest_tokens = int(
                    flags.get("ecm_harvest_tokens")
                    or engine_config.get("ecm_harvest_tokens") or 0)
                if flags.get("compute_ecm") and harvest_tokens > 0:
                    try:
                        state.progress(
                            "harvesting",
                            f"[{i+1}/{n}] generating response "
                            f"({harvest_tokens} tokens)...")

                        from src.engine.ecm_harvest import (
                            generate_harvest_response)
                        harvest = generate_harvest_response(
                            state.analyzer,
                            p["prompt"],
                            max_new_tokens=harvest_tokens,
                        )

                        resp_text = harvest["response_text"]
                        if resp_text:
                            resp_cat = (p.get("category", "")
                                        + ":response").lstrip(":")
                            resp_result = state.analyzer.analyze_prompt(
                                resp_text,
                                category=resp_cat,
                                compute_kl=flags.get("compute_kl", False),
                                compute_full_trajectory=flags.get(
                                    "compute_full_trajectory", True),
                                capture_responses=flags.get(
                                    "capture_responses", False),
                                full_capture=flags.get(
                                    "full_capture", False),
                                compute_ltp=flags.get(
                                    "compute_ltp", False),
                                compute_sfd=flags.get(
                                    "compute_sfd", False),
                                ltp_k=flags.get("ltp_k", 8),
                                ltp_layer_strategy=flags.get(
                                    "ltp_layer_strategy", "signal"),
                                ltp_svd_rank=flags.get(
                                    "ltp_svd_rank", 0),
                                base_cache=None,
                            )
                            resp_rd = result_to_dict(resp_result)
                            from src.engine.ecm_analysis import (
                                attach_ecm_analysis as _attach_ecm)
                            _attach_ecm(resp_rd)
                            resp_rd["role"] = "assistant"
                            resp_rd["ecm_harvest"] = {
                                "source_prompt": p["prompt"],
                                "ecm_diagnostics": (
                                    harvest["ecm_diagnostics"]),
                                "seed": harvest["seed"],
                                "max_new_tokens": harvest_tokens,
                                "n_generated_tokens": harvest["n_tokens"],
                            }
                            state.session.add_result(resp_rd)
                            results.append(resp_rd)

                            n_iv = harvest["ecm_diagnostics"].get(
                                "n_interventions", 0)
                            state.progress(
                                "harvesting",
                                f"  response: {harvest['n_tokens']} "
                                f"tokens, {n_iv} interventions")
                    except Exception as he:
                        logger.exception(
                            f"Harvest failed for prompt {i}: "
                            f"{p.get('prompt', '')[:60]}")
                        errors.append((i, f"ECM harvest: {he}"))
                        state.progress("error",
                                       f"Harvest {i} failed: {he}")

        except Exception as e:
            logger.exception(f"Analysis failed for work item {i}")
            errors.append((i, str(e)))
            if progress:
                progress("error", f"Prompt {i} failed: {e}")
    return results, errors


def _start_analysis_job(prompts: list[dict], flags: dict, *,
                        deconstruct: bool = False,
                        n_prompts: int = 0,
                        progress=None) -> dict:
    """Start an analysis job in the background — the single contract shared by
    the single-prompt and batch endpoints.

    Returns immediately with ``{"ok": True, "started": True, "n_prompts": N}``.
    Completion (or fatal failure) is announced exactly once via the
    ``analyze_done`` SSE event, whose payload carries the full outcome so the
    client can surface failures loudly and never silently:

        ``{ok, n_results, n_prompts, n_errors, error}``

    - ``ok=False`` → a fatal/infrastructure error; nothing was produced.
    - ``ok=True, n_results=0`` → the job ran but every prompt failed
      (``error`` = first failure message).
    - ``ok=True, n_errors>0``  → partial: some prompts failed, some succeeded.

    Individual inference calls are still serialized by ``_analysis_lock``
    inside ``_analyze_prompt_list``; this only adds a job-level guard so two
    whole submissions can't overlap. ``progress`` is forwarded to the per-rung
    logger — the single endpoint passes ``None`` when not deconstructing to
    keep the interactive console quiet, its long-standing behavior.
    """
    global _job_running
    with _job_lock:
        if _job_running:
            return {"ok": False, "error": "Analysis already running."}
        _job_running = True

    def _run():
        global _job_running
        try:
            results, errors = _analyze_prompt_list(
                prompts, flags, deconstruct=deconstruct, progress=progress)
            state.session.save_to_disk()
            n = len(results)
            if progress is not None:
                msg = (f"Analysis complete: {n} record(s), {len(errors)} failed"
                       if errors else f"Analysis complete: {n} record(s)")
                progress("done", msg)
            broker.publish("analyze_done", {
                "ok": True,
                "n_results": n,
                "n_prompts": n_prompts,
                "n_errors": len(errors),
                "error": (errors[0][1] if errors else None),
            })
        except Exception as e:
            logger.exception("Analysis job failed")
            if progress is not None:
                progress("error", f"Analysis failed: {e}")
            broker.publish("analyze_done", {
                "ok": False,
                "n_results": 0,
                "n_prompts": n_prompts,
                "n_errors": 0,
                "error": str(e),
            })
        finally:
            with _job_lock:
                _job_running = False

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "started": True, "n_prompts": n_prompts}


async def api_analyze_handler(request: Request):
    """POST /api/analyze — a single prompt as a one-item analysis job.

    Asynchronous, like the batch endpoint: returns ``{"started": True}`` and
    announces completion via the ``analyze_done`` SSE event. With deconstruct
    on, the one prompt expands into its prefix ladder; all rungs are stored.
    Per-rung console logging is enabled only when deconstructing, so an
    ordinary single-prompt run keeps the console quiet.
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

    deconstruct = _form_bool(form, "deconstruct")
    flags = _read_analyze_flags(form, trajectory_default=True)

    return _start_analysis_job(
        [{"prompt": prompt, "category": category}],
        flags, deconstruct=deconstruct, n_prompts=1,
        progress=(state.progress if deconstruct else None),
    )


# ─── Batch analyze ──────────────────────────────────────────────

async def api_analyze_batch_handler(request: Request):
    """POST /api/analyze_batch — a CSV of prompts as one analysis job.

    Same async contract as the single endpoint via ``_start_analysis_job``;
    the only differences are CSV parsing and that trajectories default off for
    batch volume. Completion arrives on the shared ``analyze_done`` event.
    """
    import csv as _csv
    import io as _io

    if state.analyzer is None:
        return {"ok": False, "error": "No model loaded."}

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

    deconstruct = _form_bool(form, "deconstruct")
    flags = _read_analyze_flags(form, trajectory_default=False)

    return _start_analysis_job(
        prompts, flags, deconstruct=deconstruct,
        n_prompts=len(prompts), progress=state.progress,
    )


# ─── Session results ────────────────────────────────────────────

def api_session_results_handler(page: int = 1, per_page: int = 10):
    """GET /api/session/results — paginated flat results."""
    total = state.session.n_results
    if total == 0:
        return {"ok": False, "results": [], "total": 0, "page": 1,
                "per_page": per_page, "total_pages": 0,
                "cache_size_bytes": state.session.get_cache_size()}

    per_page = max(1, min(per_page, 10000))
    total_pages = max(1, -(-total // per_page))
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page

    out = state.session.get_results_page(offset=start, limit=per_page)
    for r in out:
        if "_plot_keys" not in r:
            r["_plot_keys"] = _plot_keys_for_result(r)

    return {
        "ok": True, "results": out, "page": page,
        "per_page": per_page, "total": total,
        "total_pages": total_pages,
        "cache_size_bytes": state.session.get_cache_size(),
    }


def api_dashboard_handler(force: bool = False):
    """GET /api/dashboard — slim results via indexed SQL columns."""
    n = state.session.n_results
    if n == 0:
        return {"ok": False, "error": "No data in session.",
                "results": [], "session_info": {"n_results": 0},
                "cache_size_bytes": state.session.get_cache_size()}

    slim = state.session.get_dashboard_rows()

    session_info = {
        "n_results": n,
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
    """GET /api/results/detail — paginated full results from DB."""
    total = state.session.n_results
    start = max(0, start)
    count = max(1, min(count, 1000))

    out = state.session.get_results_page(offset=start, limit=count)
    for r in out:
        if "_plot_keys" not in r:
            r["_plot_keys"] = _plot_keys_for_result(r)

    return {"ok": True, "results": out, "start": start,
            "count": len(out), "total": total}


# ─── Progress / reset / user_info ───────────────────────────────

def api_progress_handler():
    """GET /api/progress"""
    return {"log": list(state.progress_log), "now": time.time()}


def api_reset_handler():
    """POST /api/reset — full reset. Refused while a job is in flight."""
    if job_active():
        return {"ok": False,
                "error": "An analysis job is running. Wait for it to "
                         "finish before resetting."}
    if state.pipeline is not None:
        try:
            state.pipeline.unload()
        except Exception:
            pass
    state.pipeline = None
    state.analyzer = None
    state.session = Session(db=_db)
    state.loading_state = {"active": False, "error": None}
    state.progress("reset", "Pipeline and session reset")
    return {"ok": True, "message": "Reset complete"}


def api_session_restore_handler():
    """POST /api/session/restore"""
    restored = Session.restore(db=_db)
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
