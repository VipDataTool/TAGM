"""TAGM FastAPI application — TASM-native API surface.

Uses tagm.engine for all per-prompt computation. Produces flat result
dicts the frontend reads directly. No orchestrator, no tasm_compat.
"""
from __future__ import annotations

import csv as _csv
import io as _io
import json
import logging
import math
import os
import shutil
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool

from src.core.pipeline import Pipeline
from src.engine.analyzer import Analyzer
from src.engine.result import result_to_dict
from src.engine import config as engine_config
from src.engine.session import Session
from src.engine.app_core import (
    AppState, state,
    _load_model_registry,
    api_status_handler,
    api_config_get_handler, api_config_post_handler,
    api_load_model_handler,
    api_analyze_handler,
    api_analyze_batch_handler,
    cancel_analysis_job,
    api_session_results_handler,
    api_dashboard_handler,
    api_results_detail_handler,
    api_progress_handler,
    api_reset_handler,
    api_session_restore_handler,
    api_session_remove_handler,
    _plot_keys_for_result,
    _sanitize_for_json,
    _form_bool,
    _analysis_lock,
)
from src.core.cache import Cache
from src.core.locks import MODEL_LOCK
from src.service.events import broker

# API routers. Each owns a slice of the surface that used to live here;
# routes, methods and response shapes are unchanged by the split.
from src.api import ecm_config as _ecm_config_api
from src.api import hep as _hep_api
from src.api import modules as _modules_api
from src.api import probes as _probes_api
from src.api import roundtable as _roundtable_api
from src.api._state import module_runner as _module_runner

logger = logging.getLogger("src")

_PACKAGE_DIR = Path(__file__).parent
LOG_FILE = _PACKAGE_DIR.parent / "tagm.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# ─── Additional global state ────────────────────────────────────
# _module_runner now lives in src/api/_state.py so the routers can share
# the one instance without importing this module (which would be circular).
_cache = Cache()

# When a model loads, propagate the pipeline to modules that need it
state.on_model_loaded(_module_runner.set_pipeline)

# ─── App ────────────────────────────────────────────────────────
app = FastAPI(title="TAGM", version="2.0.0")

# ─── Routers ────────────────────────────────────────────────────
# Mounted before the static mount / catch-all HTML routes below so route
# resolution order is unchanged from when these were defined inline.
# Literal-path routers first (probes owns two literal
# /api/modules/probe_generator/* paths), then the parameterised ones.
app.include_router(_hep_api.router)
app.include_router(_ecm_config_api.router)
app.include_router(_probes_api.router)
app.include_router(_modules_api.router)
app.include_router(_roundtable_api.router)

# Static files
#
# Served with `Cache-Control: no-cache` — which (despite the name) means
# "cache, but revalidate before use."  StaticFiles already sends ETag and
# Last-Modified, so revalidation is a 304 round-trip: always fresh after
# a code change, near-free when nothing changed.  Without this header,
# browsers apply heuristic freshness (~10% of file age) and serve stale
# JS/CSS for days after an edit — the blank-page-until-hard-refresh bug.
#
# Deliberately NOT a BaseHTTPMiddleware: that would wrap every response
# including the SSE streams (chat tokens, analyze_done events), a known
# Starlette streaming pitfall.  Subclassing touches static files only.
class RevalidatedStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response

_static_dir = _PACKAGE_DIR.parent / "static"
if _static_dir.exists():
    app.mount("/static", RevalidatedStaticFiles(directory=str(_static_dir)),
              name="static")

# ─── Restore HEP state from DB ────────────────────────────────
# If HEP was active when the server last ran, re-enable it so
# cached mmap delta files are found on the next model load.
try:
    from src.core.db import get_db
    _hep_state = get_db().get_config("hep")
    if _hep_state.get("active"):
        engine_config.update({
            "delta_backend": "mmap",
            "hep_active": True,
            "hep_evict_base_cache": _hep_state.get("evict_base_cache", True),
        })
        logger.info("[HEP] Restored: active, delta_backend=mmap")
    del _hep_state
except Exception as e:
    logger.info(f"[HEP] No saved state ({e})")


# ═══════════════════════════════════════════════════════════════
# HTML routes
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    index = _static_dir / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>TAGM</h1><p>No frontend found.</p>")

@app.get("/favicon.svg")
async def favicon():
    p = _static_dir / "favicon.svg"
    if p.exists():
        return FileResponse(str(p), media_type="image/svg+xml")
    raise HTTPException(status_code=404)

for _viz in ("chat", "roundtable", "domain_surface_viz",
             "template_maker",
             "correction_prism_viz",
             "probe_diagnostic_viz",
             "correction_field_topology_viz"):
    def _make_viz_route(name):
        async def handler():
            p = _static_dir / f"{name}.html"
            if p.exists():
                return HTMLResponse(p.read_text(encoding="utf-8"))
            raise HTTPException(status_code=404)
        return handler
    app.get(f"/{_viz}", include_in_schema=False)(_make_viz_route(_viz))


# ═══════════════════════════════════════════════════════════════
# SSE event stream — replaces polling for all state transitions
# ═══════════════════════════════════════════════════════════════

@app.get("/api/events")
async def events(request: Request):
    return broker.sse_response(request)


# ═══════════════════════════════════════════════════════════════
# Core API — delegated to engine.app_core
# ═══════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    # Thread-pooled: the handler reads session.get_cache_size(), a
    # SUM(LENGTH(blob)) table scan taken under the DB's RLock, which
    # analysis writers also hold. On a ~2s poll that stalled the event
    # loop for as long as a write was in flight.
    return await run_in_threadpool(api_status_handler)

@app.get("/api/config")
async def get_config():
    return api_config_get_handler()

@app.post("/api/config")
async def save_config(request: Request):
    data = await request.json()
    return api_config_post_handler(data)

@app.get("/api/models")
async def get_models():
    return {"models": _load_model_registry()}

@app.post("/api/models")
async def add_model(id: str = Form(...), name: str = Form(...),
                    base: str = Form(...), instruct: str = Form(...)):
    id_clean = id.strip().lower().replace(" ", "-")
    if not id_clean or not base.strip() or not instruct.strip():
        # ok:false added so the failure shape matches every other endpoint;
        # the frontend reads `error`, which is unchanged.
        return JSONResponse(status_code=400,
                            content={"ok": False,
                                     "error": "All fields required."})
    from src.core.db import get_db
    get_db().upsert_model(id_clean, name.strip(), base.strip(), instruct.strip())
    return {"ok": True}

@app.post("/api/load_model")
async def load_model(request: Request):
    form = await request.form()
    return api_load_model_handler(
        pair_id=(form.get("pair_id") or "").strip(),
        instruct_id=(form.get("instruct_id") or form.get("instruct") or "").strip(),
        base_id=(form.get("base_id") or form.get("base") or "").strip(),
        layer_filter_raw=(form.get("layer_filter") or "").strip(),
        compute_spectral_raw=(form.get("compute_spectral") or "true"),
    )

@app.post("/api/set_inference_model")
async def set_inference_model(request: Request):
    form = await request.form()
    cls = (form.get("model_class") or "").strip()
    if cls not in ("instruct", "base"):
        return {"ok": False, "error": "Must be 'instruct' or 'base'"}
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "error": "No model loaded."}
    if cls == "base" and state.pipeline.base_model is None:
        try:
            state.progress("loading", "Loading base model for chat...")
            await run_in_threadpool(state.pipeline.load_base)
            state.progress("ready", "Base model loaded")
        except Exception as e:
            logger.exception("Failed to load base model for chat")
            return {"ok": False, "error": f"Failed to load base model: {e}"}
    state.pipeline.inference_class = cls
    state.inference_class = cls  # keep state in sync
    return {"ok": True, "inference_class": cls}

@app.post("/api/reset")
async def reset():
    # Thread-pooled: api_reset_handler now takes MODEL_LOCK to tear the
    # pipeline down safely, and that lock can be held by an in-flight
    # chat generation — blocking on it from the event loop would freeze
    # every other request.
    return await run_in_threadpool(api_reset_handler)


@app.post("/api/analyze")
async def analyze(request: Request):
    return await api_analyze_handler(request)

@app.post("/api/analyze_batch")
async def analyze_batch(request: Request):
    return await api_analyze_batch_handler(request)

@app.post("/api/analyze/cancel")
async def analyze_cancel():
    """Request cancellation of the running analysis job.

    Cooperative: takes effect after the current prompt finishes, since a
    forward pass cannot be safely interrupted. Prompts already analyzed are
    kept — each is a complete, already-persisted record. The job still emits
    exactly one analyze_done (with cancelled=True), so a waiting client is
    never left hanging.
    """
    return cancel_analysis_job()

@app.get("/api/session/results")
async def session_results(page: int = 1, per_page: int = 10):
    return api_session_results_handler(page, per_page)

@app.get("/api/dashboard")
async def dashboard(force: bool = False):
    return api_dashboard_handler(force)

@app.get("/api/results/detail")
async def results_detail(start: int = 0, count: int = 50):
    return api_results_detail_handler(start, count)

@app.get("/api/progress")
async def progress():
    return api_progress_handler()

@app.get("/api/log")
async def log_alias(since: float = 0):
    return api_progress_handler()

@app.get("/api/log/download")
async def log_download():
    """Download the full backend log file (tagm.log).

    This is the unfiltered application log written by Python's logging
    framework — a strict superset of the in-memory progress buffer that
    drives the sidebar widget. Returned with Content-Disposition so the
    browser saves it instead of rendering inline.
    """
    if not LOG_FILE.exists():
        raise HTTPException(status_code=404, detail="No log file on disk yet.")
    return FileResponse(
        str(LOG_FILE),
        media_type="text/plain",
        filename="tagm.log",
    )

@app.post("/api/session/restore")
async def restore():
    return api_session_restore_handler()

@app.post("/api/session/clear_plots")
async def clear_plots():
    # Must report cache_size_bytes even though it deletes nothing: the client
    # reads it into the session-size badge (main.js), so omitting it made
    # "clear plot cache" blank out a number it had not changed.
    size = await run_in_threadpool(state.session.get_cache_size)
    return {"ok": True, "message": "No server-side plot cache.",
            "freed_mb": 0, "cache_size_bytes": size}

@app.post("/api/session/clear_all")
async def clear_all():
    return await run_in_threadpool(api_reset_handler)

@app.post("/api/session/remove")
async def session_remove(request: Request):
    body = await request.json()
    return api_session_remove_handler(body.get("indices", []))

@app.post("/api/session/rerun")
async def session_rerun(request: Request):
    """Re-analyze specific prompts with current settings."""
    body = await request.json()
    indices = body.get("indices", [])
    if not indices or state.analyzer is None:
        return {"ok": False, "error": "No indices or no model loaded."}
    options = body.get("options", {})

    def _rerun() -> tuple[int, int]:
        n_ok = 0
        n_err = 0
        for idx in indices:
            if idx >= len(state.session.results):
                continue
            old = state.session.results[idx]
            try:
                with _analysis_lock:
                    result = state.analyzer.analyze_prompt(
                        old["prompt"], category=old.get("category", ""),
                        compute_ltp=options.get("compute_ltp", False),
                        compute_sfd=options.get("compute_sfd", False),
                    )
                    rd = result_to_dict(result)
                    if options.get("compute_ecm"):
                        from src.engine.ecm_analysis import attach_ecm_analysis
                        attach_ecm_analysis(rd)
                    rd["_index"] = idx
                    rd["_plot_keys"] = _plot_keys_for_result(rd)
                    # Preserve deconstruction-ladder identity: the rerun result
                    # has no family/rung fields, and dropping them from the blob
                    # made the detail view disagree with the dashboard columns.
                    for kf in ("family_index", "rung_index"):
                        if old.get(kf) is not None:
                            rd[kf] = old[kf]
                    state.session.results[idx] = rd
                    n_ok += 1
            except Exception:
                # Previously counted as neither success nor failure: the
                # response reported ok/n_rerun only, so a run where every
                # index blew up looked identical to a partial success.
                logger.exception(f"Rerun {idx} failed")
                n_err += 1
        state.session.save_to_disk()
        return n_ok, n_err

    # analyze_prompt is a blocking forward pass; every other analysis path
    # already goes through run_in_threadpool, this one did not and stalled
    # the event loop for the whole rerun.
    n_rerun, n_errors = await run_in_threadpool(_rerun)
    return {"ok": True, "n_rerun": n_rerun, "n_errors": n_errors,
            "n_results": state.session.n_results}


# ═══════════════════════════════════════════════════════════════
# User info
# ═══════════════════════════════════════════════════════════════

@app.post("/api/user_info")
async def set_user_info(name: str = Form(""), organization: str = Form(""), project: str = Form("")):
    state.user_info = {"name": name.strip(), "organization": organization.strip(), "project": project.strip()}
    return {"ok": True, "user_info": state.user_info}


# ═══════════════════════════════════════════════════════════════
# Prompt library
# ═══════════════════════════════════════════════════════════════

_PROMPTS_FILE = _PACKAGE_DIR.parent / "prompts.csv"

@app.get("/api/prompts")
async def get_prompts():
    from src.core.db import get_db
    prompts = get_db().list_prompts()
    return {"prompts": prompts}

@app.post("/api/prompts")
async def add_prompt(prompt: str = Form(...), category: str = Form("")):
    from src.core.db import get_db
    db = get_db()
    db.add_prompt(prompt.strip(), category.strip())
    # Also append to CSV for backward compat
    exists = _PROMPTS_FILE.exists()
    with open(_PROMPTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["prompt", "category"])
        if not exists:
            writer.writeheader()
        writer.writerow({"prompt": prompt.strip(), "category": category.strip()})
    return {"ok": True, "prompts": db.list_prompts()}


# ═══════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════

# Export job state. Mirrors the _pg_embed_state pattern: one dict, one
# lock, one "already running" flag. The previous globals (_export_ready /
# _export_path) were written from a daemon thread with no synchronisation
# and no guard, so overlapping exports raced on the same file, and every
# export left a zip behind in Path.cwd() forever.
_export_lock = threading.Lock()
_export_state: dict[str, Any] = {
    "active": False,
    "ready": False,
    "path": None,   # Optional[Path]
    "dir": None,    # Optional[Path] — private tempdir holding `path`
}


def _discard_previous_export():
    """Delete the artifact from the last export, if any. Caller holds the lock."""
    old_path = _export_state.get("path")
    old_dir = _export_state.get("dir")
    _export_state["ready"] = False
    _export_state["path"] = None
    _export_state["dir"] = None
    try:
        if old_path is not None and Path(old_path).exists():
            Path(old_path).unlink()
        if old_dir is not None and Path(old_dir).exists():
            shutil.rmtree(old_dir, ignore_errors=True)
    except OSError:
        logger.warning(f"Could not clean up previous export at {old_path}",
                       exc_info=True)


@app.post("/api/export")
async def export_session(request: Request):
    from src.service.export import export_session_split

    if not state.session.results:
        return {"ok": False, "error": "No data to export."}

    # Read options from request body
    try:
        opts = await request.json()
    except Exception:
        opts = {}
    emb_precision = int(opts.get("embeddingPrecision", 12))
    emb_precision = max(4, min(emb_precision, 17))  # sane bounds

    with _export_lock:
        if _export_state["active"]:
            return {"ok": False, "error": "Export already in progress."}
        _discard_previous_export()
        _export_state["active"] = True
        # Private temp dir, not Path.cwd(): the server's working directory
        # is the repo, so exports were dropped into the source tree and
        # never removed.
        export_dir = Path(tempfile.mkdtemp(prefix="tagm_export_"))
        _export_state["dir"] = export_dir

    def _do_export():
        try:
            state.progress("exporting", "Preparing export...")
            p = export_dir / f"session_{state.session.session_id}.zip"
            mod_results = _module_runner.collect_results(skip={"comparative_analysis"})
            export_session_split(state.session, p,
                                 float_precision=emb_precision,
                                 module_results=mod_results)
            with _export_lock:
                _export_state["path"] = p
                _export_state["ready"] = True
            state.progress("done", f"Export ready: {p.name}")
            broker.publish("export_ready", {"filename": p.name})
        except Exception as e:
            # Without this, "ready" stays False forever and the UI waits on
            # an event that never comes. export_error has been in the
            # broker's SNAPSHOT_TYPES all along — now something actually
            # publishes it.
            logger.exception("Export failed")
            state.progress("error", f"Export failed: {e}")
            broker.publish("export_error", {"error": str(e)})
        finally:
            with _export_lock:
                _export_state["active"] = False

    threading.Thread(target=_do_export, daemon=True).start()
    return {"ok": True, "ready": False}

@app.get("/api/export/download")
async def export_download():
    with _export_lock:
        ready = _export_state["ready"]
        path = _export_state["path"]
    if not ready or path is None or not Path(path).exists():
        raise HTTPException(status_code=404, detail="No export available.")
    path = Path(path)
    media = "application/zip" if path.suffix == ".zip" else "application/gzip"
    return FileResponse(str(path), media_type=media, filename=path.name)


# ═══════════════════════════════════════════════════════════════
# Plots (server-side matplotlib)
# ═══════════════════════════════════════════════════════════════

@app.get("/api/plots/{plot_key}")
async def get_plot(plot_key: str):
    """Batch or per-prompt plot."""
    if not state.session.results:
        raise HTTPException(status_code=404, detail="No data in session.")
    try:
        # One registry lookup — service.plots knows which keys are batch
        # plots and which are per-prompt, so the old "try batch, fall back
        # to per-prompt" chain (which silently rendered result[0] when a
        # batch plot legitimately produced no image) is gone.
        from src.service.plots import render_session_plot
        img = await run_in_threadpool(render_session_plot, plot_key,
                                      state.session.results)
        if img is not None:
            return StreamingResponse(_io.BytesIO(img), media_type="image/png")

        raise HTTPException(status_code=404, detail=f"Unknown plot '{plot_key}'")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Plot {plot_key} failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plots/individual/{index}/{plot_key}")
async def get_individual_plot(index: int, plot_key: str):
    from src.service.plots import render_plot
    if index >= len(state.session.results):
        raise HTTPException(status_code=404)
    try:
        img = await run_in_threadpool(render_plot, plot_key, state.session.results[index])
        if img is None:
            raise HTTPException(status_code=404, detail=f"Unknown plot '{plot_key}'")
        return StreamingResponse(_io.BytesIO(img), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# Chat
# ═══════════════════════════════════════════════════════════════

def _load_chat_config():
    """Load chat config from disk, layered over the engine defaults.

    Persisted keys win; missing keys fall back to defaults, so a config
    file written before a new setting existed (e.g. context_window)
    still yields a complete config.
    """
    cfg = {
        "temperature": engine_config.get("chat_temperature"),
        "top_p": engine_config.get("chat_top_p"),
        "max_tokens": engine_config.get("chat_max_tokens"),
        "analyze_prompts": True,
        "analyze_responses": False,
        "compute_ltp": True,
        "compute_sfd": True,
        # Number of prior turns the client replays into `messages`.
        "context_window": 10,
    }
    config_path = _PACKAGE_DIR.parent / "chat_config.json"
    if config_path.exists():
        try:
            saved = json.loads(config_path.read_text())
        except Exception as e:
            # Silently discarding the file meant a corrupt chat_config.json
            # reverted every persisted setting with no indication why.
            logger.warning(f"[chat] Ignoring unreadable chat_config.json: {e}")
        else:
            if isinstance(saved, dict):
                cfg.update(saved)
            else:
                logger.warning(
                    "[chat] chat_config.json is not a JSON object — ignored")
    return cfg

@app.get("/api/chat/config")
async def get_chat_config():
    return {"ok": True, "config": _load_chat_config()}

@app.post("/api/chat/config")
async def set_chat_config(request: Request):
    """Merge posted keys into the chat config and persist."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON body."}
    cfg = _load_chat_config()
    cfg.update(body)
    config_path = _PACKAGE_DIR.parent / "chat_config.json"
    try:
        config_path.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        return {"ok": False, "error": f"Write failed: {e}"}
    return {"ok": True, "config": cfg}

@app.post("/api/chat")
async def chat(request: Request):
    from src.service.chat import generate_chat_response_streaming
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "error": "No model loaded."}
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON body."}
    messages = body.get("messages")
    if not messages:
        prompt = body.get("prompt") or body.get("message", "")
        if not prompt:
            return {"ok": False, "error": "No message provided."}
        messages = [{"role": "user", "content": prompt}]

    cfg = _load_chat_config()
    cfg_max = int(cfg.get("max_tokens",
                          engine_config.get("chat_max_tokens")))
    hard_cap = max(int(engine_config.get("chat_max_tokens")), cfg_max)
    max_tokens = min(int(body.get("max_tokens", cfg_max)), hard_cap)

    do_analyze = body.get("analyze", cfg.get("analyze_prompts", True))
    do_analyze_resp = body.get("analyze_response", cfg.get("analyze_responses", False))
    compute_ltp = body.get("compute_ltp", cfg.get("compute_ltp", True))
    compute_sfd = body.get("compute_sfd", cfg.get("compute_sfd", True))
    full_analysis = body.get("full_analysis", False)
    category = body.get("category", "chat")

    async def event_stream():
        import json as _json

        done_result = None

        # Phase 1: stream tokens from generation.
        #
        # generate_chat_response_streaming is a *synchronous* generator: it
        # blocks on the TextIteratorStreamer queue and on thread.join().
        # Iterating it directly from this coroutine pinned the event loop
        # for the entire generation, freezing every other request (status
        # polls, the SSE broker, analyze_done). iterate_in_threadpool drives
        # it on a worker thread and hands chunks back asynchronously.
        sync_stream = generate_chat_response_streaming(
            state.pipeline, messages,
            max_tokens=max_tokens,
            temperature=float(cfg.get("temperature",
                              engine_config.get("chat_temperature"))),
            top_p=float(cfg.get("top_p", engine_config.get("chat_top_p"))),
            analyzer=state.analyzer,
        )
        async for sse_line in iterate_in_threadpool(sync_stream):
            # Capture the done event for analysis
            if sse_line.startswith("data: "):
                try:
                    evt = _json.loads(sse_line[6:].strip())
                    if evt.get("type") == "done":
                        done_result = evt
                except Exception:
                    pass
            # No client-disconnect branch here: a disconnect closes this
            # async generator, which raises GeneratorExit (a BaseException)
            # at the yield. The old `except Exception` around the yield
            # could never catch it, so the drain logic was dead code.
            yield sse_line

        # Phase 2: run analysis (after generation completes)
        if done_result and done_result.get("ok"):
            ecm_was_active = done_result.get("ecm_active", False)
            ecm_summary = None
            if ecm_was_active and done_result.get("ecm_diagnostics"):
                d = done_result["ecm_diagnostics"]
                ecm_summary = {
                    "n_interventions": d.get("n_interventions", 0),
                    "n_tokens": d.get("n_tokens", 0),
                    "max_cascade_signal": d.get("max_cascade_signal", 0),
                    "intervention_rate": d.get("intervention_rate", 0.0),
                    "n_loop_releases": d.get("n_loop_releases", 0),
                }

            # Analyze user prompt
            if do_analyze:
                prompt_text = messages[-1].get("content", "")
                if prompt_text and state.analyzer:
                    try:
                        await run_in_threadpool(
                            _analyze_chat_turn, prompt_text, category,
                            compute_ltp, compute_sfd, "user",
                            ecm_active=ecm_was_active,
                            full_analysis=full_analysis)
                        yield f"data: {_json.dumps({'type': 'analyzed', 'target': 'prompt'})}\n\n"
                    except Exception as e:
                        logger.warning(f"Chat prompt analysis failed: {e}")

            # Analyze model response
            if do_analyze_resp:
                response_text = done_result.get("response", "")
                if response_text and state.analyzer:
                    try:
                        await run_in_threadpool(
                            _analyze_chat_turn, response_text,
                            "model_response",
                            full_analysis, full_analysis, "assistant",
                            ecm_active=ecm_was_active,
                            ecm_summary=ecm_summary,
                            full_analysis=full_analysis)
                        yield f"data: {_json.dumps({'type': 'analyzed', 'target': 'response'})}\n\n"
                    except Exception as e:
                        logger.warning(f"Chat response analysis failed: {e}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _analyze_chat_turn(text, category, compute_ltp, compute_sfd, role,
                       ecm_active=False, ecm_summary=None,
                       full_analysis=False):
    """Analyze a chat turn and add it to the session with a role tag.

    When full_analysis=True, runs the base model phase first to compute
    rank displacement and full KL — required for topology visualization.
    """
    from src.engine.result import result_to_dict
    with _analysis_lock:
        base_cache = None
        if full_analysis:
            caches = state.analyzer.run_base_phase(
                [{"prompt": text}],
                compute_kl=True,
                capture_responses=False,
                progress=state.progress,
            )
            if caches and caches[0]:
                base_cache = caches[0]

        result = state.analyzer.analyze_prompt(
            text, category=category,
            compute_kl=True, compute_full_trajectory=False,
            compute_ltp=compute_ltp, compute_sfd=compute_sfd,
            base_cache=base_cache,
        )
        rd = result_to_dict(result)
        rd["role"] = role
        rd["ecm_active"] = ecm_active
        if ecm_summary:
            rd["ecm_summary"] = ecm_summary
        state.session.add_result(rd)
        return rd
