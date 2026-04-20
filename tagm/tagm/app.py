"""TAGM FastAPI surface.

Run with:
    python -m uvicorn tagm.app:app --host 0.0.0.0 --port 8000

Endpoints (all under /api/*):

  Pipeline / models
    GET    /api/status               pipeline + session state
    GET    /api/models               list configured model pairs (models.json)
    POST   /api/load                 load a model pair
    POST   /api/unload               free the model pair

  Measurement and analysis introspection
    GET    /api/measurements         list registered measurements + metadata
    GET    /api/analyses             list registered analyses + metadata
    GET    /api/adapters             list registered adapter families

  Per-prompt analysis
    POST   /api/configure            select measurements + parameter values
    POST   /api/analyze              run selected measurements on one prompt
    POST   /api/batch                run on a list of prompts

  Post-session analysis
    POST   /api/analysis/{name}      run a named post-session analysis

  Probes
    GET    /api/templates            list available probe templates
    GET    /api/probes               list stored probe sets
    POST   /api/probes/generate      generate a probe set from a template
    DELETE /api/probes/{set_id}      delete a probe set

  Session I/O
    GET    /api/session              current session snapshot
    POST   /api/session/reset        clear session state
    GET    /api/session/export       export session as gzipped JSON
    POST   /api/session/import       replace session from uploaded file

  Presets
    GET    /api/presets              list saved capture-config presets
    POST   /api/presets              save a preset
    GET    /api/presets/{name}       load a preset
    DELETE /api/presets/{name}       delete a preset
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional
import threading

import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Trigger registration of all measurements, analyses, adapters
from tagm.core.adapter import list_families, find_adapter  # noqa: F401
from tagm.core.adapter import registry as _adapter_registry  # noqa: F401
from tagm.core.cache import Cache
from tagm.core.capture.config import CaptureConfig, CapturePoint
from tagm.core.pipeline import Pipeline
from tagm.measurement import modules as _measurement_modules  # noqa: F401  (registers)
from tagm.measurement.registry import list_measurements
import tagm.analysis  # noqa: F401  (registers all analyses)
from tagm.analysis.registry import list_analyses


# Verify that registration actually populated both registries.
# A silent zero-registration would otherwise present as an empty /api/measurements
# response with no clear cause.
_n_measurements = len(list_measurements())
_n_analyses = len(list_analyses())
if _n_measurements == 0:
    raise RuntimeError(
        "TAGM startup: no measurements registered. Check that "
        "tagm.measurement.modules.__init__ imported each module file.")
if _n_analyses == 0:
    raise RuntimeError(
        "TAGM startup: no analyses registered. Check that "
        "tagm.analysis.__init__ imported each analysis file.")
from tagm.probes.store import ProbeStore
from tagm.probes.generator import EmbeddingGenerator, GenerationParams
from tagm.probes.template import load_template, parse_template_csv
from tagm.service.export import export_session, load_session
from tagm.service.orchestration import Orchestrator
from tagm.service.session import Session, PromptRecord


# ─── Logging ────────────────────────────────────────────────────────
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
logger = logging.getLogger("tagm")


# ─── Global state ───────────────────────────────────────────────────
class AppState:
    """Process-wide mutable state. Single-user semantics per the design:
    no tenancy, no auth. Multi-user scenarios run multiple TAGM instances."""

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.cache = Cache()
        self.probe_store = ProbeStore(root=self.cache.layout.probes)
        self.session = Session()
        self.orchestrator: Optional[Orchestrator] = None
        self.selected_measurements: list[tuple[str, dict]] = []
        self.loading_state = {"active": False, "error": None}
        self.progress_log: list[dict] = []
        self.user_info: dict = {}
        self.inference_class: str = "instruct"   # for chat model toggle

    def progress(self, stage: str, message: str) -> None:
        entry = {"ts": time.time(), "stage": stage, "message": message}
        self.progress_log.append(entry)
        logger.info(f"[{stage}] {message}")
        # Trim
        if len(self.progress_log) > 1000:
            self.progress_log = self.progress_log[-1000:]

    def require_pipeline(self) -> Pipeline:
        if self.pipeline is None or not self.pipeline.loaded:
            raise HTTPException(status_code=409,
                                 detail="No model pair loaded. POST /api/load first.")
        return self.pipeline

    def require_orchestrator(self) -> Orchestrator:
        self.require_pipeline()
        if self.orchestrator is None:
            self.orchestrator = Orchestrator(self.pipeline, self.probe_store)
        return self.orchestrator


state = AppState()


# ─── Model registry (models.json at repo root) ──────────────────────
MODELS_FILE = _PACKAGE_DIR.parent / "models.json"


def _load_model_registry() -> list[dict]:
    if not MODELS_FILE.exists():
        return []
    try:
        with open(MODELS_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read models.json: {e}")
        return []


# ─── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(title="TAGM", description="Transformer Alignment Geometric Metrology")


# ─── Security headers ───────────────────────────────────────────────
# TAGM is a single-user local instrument — it runs on localhost against
# a model the operator loaded themselves, served to a browser the
# operator owns. The threat model that motivates a strict CSP (untrusted
# content injection from a multi-tenant backend) does not apply here.
# The TASM-derived UI uses inline event-handler attributes (onclick=,
# onchange=) throughout, and the viz pages contain inline <script> blocks
# that run the visualization logic. Both require 'unsafe-inline' in
# script-src to execute under a CSP at all.
#
# Rather than refactor 32+ inline handlers across five HTML files into
# addEventListener calls — which would be a net negative for readability
# of TASM's UI that TAGM was supposed to keep verbatim — we allow
# 'unsafe-inline' and explicitly whitelist the two CDN origins the
# frontend actually uses: cdnjs (Three.js for domain_surface_viz) and
# fonts.googleapis.com / fonts.gstatic.com (IBM Plex font family).
# cdn.plot.ly is kept whitelisted against the day Plotly is introduced.
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' "
            "https://cdn.plot.ly https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "img-src 'self' data:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
    return response

# Static file serving for the frontend
_STATIC_DIR = _PACKAGE_DIR.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the frontend index."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({
        "service": "TAGM",
        "version": "0.1.0",
        "note": "Frontend index.html not found; API endpoints at /api/*",
    })


# ─── Status / models ────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    cap_cfg = state.orchestrator.capture_config if state.orchestrator else None
    pipeline_info = (state.pipeline.describe() if state.pipeline else
                     {"loaded": False})

    # TASM-flat fields. main.js (TASM-derived frontend) reads these
    # directly. Carried alongside TAGM's nested shape so both styles work.
    #
    # model_loaded combines two TAGM conditions that TASM rolled into
    # one: the pipeline itself is loaded, AND the orchestrator has a
    # CaptureConfig + at least one selected measurement. Without the
    # second condition we'd briefly report model_loaded=True during the
    # window between pipeline.load() returning and _load_worker's
    # auto-configure step, and main.js would enable the Analyze button
    # against a half-ready backend.
    pipeline_loaded = bool(pipeline_info.get("loaded"))
    orch_ready = (state.orchestrator is not None
                  and state.orchestrator.capture_config is not None
                  and len(state.selected_measurements) > 0)
    model_loaded = pipeline_loaded and orch_ready
    current_model = ""
    if pipeline_loaded:
        current_model = (pipeline_info.get("model_pair", {})
                                       .get("instruct", ""))

    # TASM's frontend reads `st.loading` as a boolean — `if(st.loading)`
    # branches into the LOADING state. Our `state.loading_state` is a
    # dict `{active, error}`. We need to return a truthy value ONLY when
    # a load is actually in progress. Falsy otherwise, regardless of
    # whether a previous load errored (those get surfaced via
    # loading_error, not loading itself).
    loading_active = bool(state.loading_state.get("active"))
    loading_payload = (dict(state.loading_state) if loading_active else False)

    # Session block. main.js bootstrap reads `st.session.n_results`
    # (not `n_prompts`), `st.session.cache_size_bytes`, and
    # `st.session.model` directly (see the "Restored session" block near
    # the init IIFE). Include both name variants so both TAGM-native
    # and TASM-compat readers find what they expect.
    session_block = {
        "session_id": state.session.session_id,
        "n_prompts": len(state.session.prompts),
        "n_results": len(state.session.prompts),       # TASM alias
        "categories": state.session.categories(),
        "measurement_names": state.session.measurement_names(),
        "n_analyses": len(state.session.record.analyses),
        "cache_size_bytes": state.cache.disk_usage(),  # TASM alias
        "model": current_model,                         # TASM alias
    }

    return {
        # Native TAGM nested shape
        "service": "TAGM",
        "pipeline": pipeline_info,
        "loading": loading_payload,
        "capture_config": (cap_cfg.to_dict() if cap_cfg is not None else None),
        "session": session_block,
        "selected_measurements": [
            {"name": n, "params": p} for n, p in state.selected_measurements
        ],
        "cache_usage_bytes": state.cache.disk_usage(),

        # TASM-flat compatibility fields
        "model_loaded": model_loaded,
        "loading_error": state.loading_state.get("error"),
        "current_model": current_model,
        "session_id": state.session.session_id,
        "n_results": len(state.session.prompts),
        "cache_bytes": state.cache.disk_usage(),

        # User info — main.js reads st.user_info.name/organization/project
        # at bootstrap to rehydrate the Analyst form fields.
        "user_info": dict(state.user_info) if state.user_info else None,
    }


@app.get("/api/models")
async def api_models():
    """Return the model registry. TASM-shape: `{models: [...]}`. Each
    entry is {id, name, instruct, base}."""
    return {"models": _load_model_registry(), "pairs": _load_model_registry()}


@app.post("/api/models")
async def api_models_add(request: Request):
    """Add or update a pair in models.json. Form data: id, name, base, instruct."""
    form = await request.form()
    id_ = (form.get("id") or "").strip().lower().replace(" ", "-")
    name = (form.get("name") or "").strip()
    base = (form.get("base") or "").strip()
    instruct = (form.get("instruct") or "").strip()
    if not id_ or not base or not instruct:
        raise HTTPException(status_code=400, detail="All fields required.")

    pairs = _load_model_registry()
    # Update if exists, else append
    found = False
    for p in pairs:
        if p.get("id") == id_:
            p["name"] = name or id_
            p["base"] = base
            p["instruct"] = instruct
            found = True
            break
    if not found:
        pairs.append({"id": id_, "name": name or id_,
                      "base": base, "instruct": instruct})

    try:
        with open(MODELS_FILE, "w") as f:
            json.dump(pairs, f, indent=2)
    except OSError as e:
        raise HTTPException(status_code=500,
                             detail=f"Could not write models.json: {e}")

    return {"ok": True, "models": pairs}


@app.get("/api/adapters")
async def api_adapters():
    return {"families": list_families()}


# ─── Load / unload ──────────────────────────────────────────────────

def _default_capture_config(n_layers: int) -> CaptureConfig:
    """Build a CaptureConfig that satisfies every Wave-1 measurement's
    CaptureExpectation without any user input.

    Covers the union of what the registered measurements need:
      - `pre_attn_norm` hidden at every layer (stress_score, LPA, LTP,
         amplitude_trajectory, SFD)
      - `post_attn_norm` hidden at every layer (amplitude_trajectory)
      - `attn_output` attention_weights at every layer (LPA)

    Measurements whose expectation this doesn't cover (none at present,
    but the orchestrator's report/violation path handles any future
    additions gracefully) are simply skipped at configure time.

    The default is intentionally generous — a user who wants a narrow
    capture for memory reasons can POST /api/capture with their own
    CaptureConfig, which replaces this one.
    """
    points = []
    layers = list(range(n_layers))
    for layer in layers:
        points.append(CapturePoint(
            layer=layer,
            hook_point="pre_attn_norm",
            capture=frozenset({"hidden"}),
        ))
        points.append(CapturePoint(
            layer=layer,
            hook_point="post_attn_norm",
            capture=frozenset({"hidden"}),
        ))
        points.append(CapturePoint(
            layer=layer,
            hook_point="attn_output",
            capture=frozenset({"attention_weights"}),
        ))
        # residual_post_block is what per_token_embedding reads for
        # depth-labeled snapshots (default depths "subject:12,escalation:18").
        points.append(CapturePoint(
            layer=layer,
            hook_point="residual_post_block",
            capture=frozenset({"hidden"}),
        ))
    # final_norm is layer-independent; per_token_embedding wants it
    # by default (include_final_norm=True).
    points.append(CapturePoint(
        layer=None,
        hook_point="final_norm",
        capture=frozenset({"hidden"}),
    ))
    return CaptureConfig(
        name="tagm_default_full",
        description="Auto-installed on model load; covers all Wave-1 "
                    "measurement CaptureExpectations.",
        points=tuple(points),
    )


def _load_worker(instruct_id: str, base_id: str,
                 layer_filter, compute_spectral: bool) -> None:
    """Background worker: runs the actual (slow, blocking) model load.
    Updates state.loading_state + state.progress so the frontend's
    polling can observe completion.

    Every step is logged both to the Python logger (→ tagm.log) and to
    state.progress (→ UI log pane via /api/progress). If anything
    fails, the error surfaces in both places.
    """
    import traceback
    logger.info(f"[_load_worker] Thread started: instruct={instruct_id} base={base_id}")
    state.progress("loading", f"Worker thread started")

    try:
        # Step 1: Unload previous pipeline if any
        if state.pipeline is not None:
            logger.info("[_load_worker] Unloading previous pipeline")
            state.progress("loading", "Unloading previous pipeline")
            state.pipeline.unload()

        # Step 2: Construct Pipeline
        logger.info(f"[_load_worker] Constructing Pipeline object")
        state.progress("loading", f"Constructing pipeline")
        state.pipeline = Pipeline(
            instruct_model_id=instruct_id,
            base_model_id=base_id,
            device="cpu",
            dtype=torch.bfloat16,
        )
        logger.info(f"[_load_worker] Pipeline constructed; calling load()")
        state.progress("loading",
                        f"Downloading weights from HF: {instruct_id}")

        # Step 3: Run the load (downloads weights, computes deltas)
        state.pipeline.load(
            layer_filter=layer_filter,
            compute_spectral=compute_spectral,
            progress=state.progress,
        )
        logger.info("[_load_worker] pipeline.load() returned successfully")

        # Step 4: Create orchestrator
        logger.info("[_load_worker] Creating orchestrator")
        state.progress("loading", "Creating orchestrator")
        state.orchestrator = Orchestrator(state.pipeline, state.probe_store)

        # Step 4b: Install a default CaptureConfig and select every
        # registered measurement that the default covers.
        #
        # TASM's UI assumed that "model loaded" implied "ready to
        # analyze." There was no separate capture-configuration step —
        # TASM derived capture implicitly from the measurement set.
        # TAGM's backend separates the two concerns, but the TASM-
        # derived frontend doesn't expose a capture UI. Rather than 409
        # every /api/analyze until the user discovers /api/capture, we
        # install a generous default covering all Wave-1 measurements'
        # expectations. Users who want a narrower capture can hit
        # /api/capture or /api/engine_setup directly; this default is
        # just "work out of the box."
        try:
            n_layers = state.pipeline.adapter.n_layers(
                state.pipeline.instruct_model)
            default_cap = _default_capture_config(n_layers)
            state.orchestrator.set_capture_config(default_cap)
            logger.info(f"[_load_worker] Default CaptureConfig installed "
                        f"({len(default_cap.points)} points over "
                        f"{n_layers} layers)")

            all_names = [m["name"] for m in list_measurements()]
            report = state.orchestrator.configure_measurements(
                [(n, {}) for n in all_names])
            state.selected_measurements = [
                (n, p) for n, p in report["selected"]]
            logger.info(f"[_load_worker] Default measurement selection: "
                        f"{len(report['selected'])} selected, "
                        f"{len(report['skipped'])} skipped "
                        f"(skipped: {report['skipped']})")
            state.progress("loading",
                f"Configured {len(report['selected'])} measurements "
                f"(skipped {len(report['skipped'])}: "
                f"{', '.join(report['skipped']) or 'none'})")
        except Exception as cfg_err:
            # Don't fail the load if default configuration fails — the
            # user can still configure manually via /api/capture and
            # /api/configure. But surface the problem in the UI log.
            logger.warning(f"[_load_worker] Default configuration failed: "
                           f"{cfg_err}")
            state.progress("loading",
                f"Warning: default capture/measurement configuration "
                f"failed ({cfg_err}). Configure manually.")

        # Step 5: Mark done
        state.loading_state = {"active": False, "error": None}
        state.progress("ready", f"Model pair loaded: {instruct_id}")
        logger.info(f"[_load_worker] DONE: loading_state reset, ready emitted")

    except Exception as e:
        # Full traceback to both channels. Users reading the UI shouldn't
        # need to open tagm.log to see what broke.
        tb = traceback.format_exc()
        logger.error(f"[_load_worker] FAILED:\n{tb}")
        state.loading_state = {"active": False, "error": str(e)}
        state.progress("error", f"Load failed: {type(e).__name__}: {e}")
        # Also log the first few traceback lines to the UI log
        for line in tb.split("\n")[-6:]:
            if line.strip():
                state.progress("error", line)


def _do_load(instruct_id: str, base_id: str,
             layer_filter=None, compute_spectral: bool = True) -> dict:
    """Start a model load in a background thread; return immediately.

    The frontend polls /api/status (for model_loaded / loading_error)
    and /api/progress (for log messages) every 2s. A synchronous load
    would block the event loop and starve those polls."""
    logger.info(f"[_do_load] called: instruct={instruct_id} base={base_id}")

    if not instruct_id or not base_id:
        logger.warning(f"[_do_load] rejecting: missing instruct/base")
        raise HTTPException(status_code=400,
                             detail="Provide pair_id or both instruct_id and base_id")

    if state.loading_state["active"]:
        logger.warning(f"[_do_load] rejecting: load already active")
        raise HTTPException(status_code=409, detail="Load already in progress")

    state.loading_state = {"active": True, "error": None}
    state.progress("loading", f"Starting load: {instruct_id} / {base_id}")

    try:
        thread = threading.Thread(
            target=_load_worker,
            args=(instruct_id, base_id, layer_filter, compute_spectral),
            daemon=True,
            name=f"tagm-load-{instruct_id}",
        )
        thread.start()
        logger.info(f"[_do_load] thread spawned: {thread.name} (alive={thread.is_alive()})")
    except Exception as e:
        # If we can't even spawn the thread, unwind state and fail loudly
        logger.exception("[_do_load] failed to spawn thread")
        state.loading_state = {"active": False, "error": str(e)}
        state.progress("error", f"Could not start load thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True, "started": True,
            "message": f"Loading {instruct_id} in background."}


@app.post("/api/load")
async def api_load(request: Request):
    """JSON load endpoint. Body: {pair_id} or {instruct_id, base_id}, plus
    optional layer_filter and compute_spectral."""
    body = await request.json()
    pair_id = body.get("pair_id")
    instruct_id = body.get("instruct_id")
    base_id = body.get("base_id")
    layer_filter = body.get("layer_filter")
    compute_spectral = bool(body.get("compute_spectral", True))

    if pair_id:
        pairs = _load_model_registry()
        match = next((p for p in pairs if p.get("id") == pair_id), None)
        if not match:
            raise HTTPException(status_code=404,
                                 detail=f"Unknown pair_id '{pair_id}'")
        instruct_id = match.get("instruct")
        base_id = match.get("base")

    return _do_load(instruct_id, base_id, layer_filter, compute_spectral)


@app.post("/api/load_model")
async def api_load_model(request: Request):
    """TASM-compat form-data load. main.js posts a FormData with
    `instruct`, `base`, optional `layer_filter` (comma-separated layer
    indices), optional `compute_spectral` (truthy string)."""
    form = await request.form()
    instruct_id = (form.get("instruct") or form.get("instruct_id") or "").strip()
    base_id = (form.get("base") or form.get("base_id") or "").strip()
    pair_id = (form.get("pair_id") or "").strip()
    logger.info(f"[api_load_model] received: pair_id={pair_id!r} "
                f"instruct={instruct_id!r} base={base_id!r}")

    if pair_id and not (instruct_id and base_id):
        pairs = _load_model_registry()
        match = next((p for p in pairs if p.get("id") == pair_id), None)
        if not match:
            logger.warning(f"[api_load_model] unknown pair_id {pair_id!r}")
            raise HTTPException(status_code=404,
                                 detail=f"Unknown pair_id '{pair_id}'")
        instruct_id = match.get("instruct")
        base_id = match.get("base")
        logger.info(f"[api_load_model] resolved pair_id={pair_id!r} to "
                    f"instruct={instruct_id!r} base={base_id!r}")

    layer_filter = None
    layer_filter_raw = (form.get("layer_filter") or "").strip()
    if layer_filter_raw:
        try:
            layer_filter = [int(x.strip()) for x in layer_filter_raw.split(",")
                            if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400,
                                 detail=f"Invalid layer_filter: {layer_filter_raw}")

    compute_spectral_raw = (form.get("compute_spectral") or "true").lower()
    compute_spectral = compute_spectral_raw not in ("0", "false", "no", "")

    return _do_load(instruct_id, base_id, layer_filter, compute_spectral)


@app.post("/api/unload")
async def api_unload():
    if state.pipeline is not None:
        state.pipeline.unload()
        state.pipeline = None
        state.orchestrator = None
        state.selected_measurements = []
    state.progress("unload", "Pipeline unloaded")
    return {"ok": True}


# ─── Measurements / analyses ────────────────────────────────────────
@app.get("/api/measurements")
async def api_measurements():
    return {"measurements": list_measurements()}


@app.get("/api/analyses")
async def api_analyses():
    return {"analyses": list_analyses()}


# ─── Configure / analyze ────────────────────────────────────────────
#
# The flow is two-step:
#   1. POST /api/capture  — set the CaptureConfig. This is the user's
#      choice about what the pipeline records during forward passes.
#      Independent of which measurements will run.
#   2. POST /api/configure — select measurements and their parameters.
#      The orchestrator validates each measurement against the active
#      CaptureConfig. Measurements whose expectations aren't met are
#      rejected with specific reasons.
#
# Configure will fail with 409 if capture has not been set first.

@app.post("/api/capture")
async def api_capture(request: Request):
    """Set the CaptureConfig used by the pipeline for forward passes.

    Body shape:
      {
        "capture": { ... CaptureConfig.to_dict() ... },
        "base_capture": { ... optional, for paired runs with both-sided activations ... }
      }
    """
    orch = state.require_orchestrator()
    body = await request.json()

    cap_dict = body.get("capture")
    if not cap_dict:
        raise HTTPException(status_code=400,
                             detail="'capture' (CaptureConfig dict) required")
    try:
        cap = CaptureConfig.from_dict(cap_dict)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400,
                             detail=f"Invalid capture config: {e}")

    base_cap = None
    base_cap_dict = body.get("base_capture")
    if base_cap_dict:
        try:
            base_cap = CaptureConfig.from_dict(base_cap_dict)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(status_code=400,
                                 detail=f"Invalid base_capture config: {e}")

    orch.set_capture_config(cap, base_cap)

    # Changing capture invalidates any existing measurement selection (their
    # CaptureExpectations were validated against the previous capture).
    # Clear selected_measurements so the UI prompts the user to re-select.
    orch._selected = []
    state.selected_measurements = []

    state.progress("capture", f"Capture config set ({len(cap.points)} points, "
                               f"signature={cap.signature()})")
    return {
        "ok": True,
        "capture": cap.to_dict(),
        "signature": cap.signature(),
        "n_points": len(cap.points),
        "selected_measurements_cleared": True,
    }


@app.get("/api/capture")
async def api_capture_get():
    """Return the current CaptureConfig (or None if not set)."""
    orch = state.require_orchestrator()
    cap = orch.capture_config
    if cap is None:
        return {"capture": None}
    return {"capture": cap.to_dict(), "signature": cap.signature()}


@app.post("/api/configure")
async def api_configure(request: Request):
    """Select which measurements to run and their scope parameters.

    Body shape:
      { "selections": [{"name": "stress_score", "params": {"layers": [10,11,12]}}, ...] }

    Returns report:
      { ok, errors, skipped, violations, selected }

    Requires /api/capture to have been called first. 409 if not.
    """
    orch = state.require_orchestrator()
    if orch.capture_config is None:
        raise HTTPException(
            status_code=409,
            detail=("No CaptureConfig set. POST /api/capture first. "
                    "The user controls what the pipeline captures; "
                    "measurements only consume from that capture."),
        )

    body = await request.json()
    selections = body.get("selections") or []
    parsed = [(s["name"], s.get("params") or {}) for s in selections]

    try:
        report = orch.configure_measurements(parsed)
    except RuntimeError as e:
        # set_capture_config guard; shouldn't happen given our check above
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("Configure failed")
        raise HTTPException(status_code=500, detail=str(e))

    state.selected_measurements = [(name, p) for name, p in report["selected"]]
    ok = not report["errors"] and not report["violations"]
    return {"ok": ok, **report}


@app.post("/api/analyze")
async def api_analyze(request: Request):
    """Analyze a single prompt.

    Accepts either:
      - JSON body: {"prompt": "...", "category": "..."}  → TAGM-native response
      - Form data: prompt, category, plus TASM legacy fields → TASM-shape response

    Form-data callers get the TASM-shape result main.js expects;
    JSON callers get TAGM's native nested PromptRecord shape.

    The actual analyze_prompt call runs in a threadpool so the event
    loop can keep serving /api/status and /api/progress polls.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    is_form = ("multipart/form-data" in content_type
               or "application/x-www-form-urlencoded" in content_type)

    if is_form:
        form = await request.form()
        prompt = (form.get("prompt") or "").strip()
        category = (form.get("category") or "").strip()
    else:
        body = await request.json()
        prompt = body.get("prompt") or ""
        category = body.get("category", "")

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    orch = state.require_orchestrator()
    if orch.capture_config is None:
        raise HTTPException(status_code=409,
                             detail="No CaptureConfig set. POST /api/capture first.")
    if not orch._selected:
        raise HTTPException(status_code=409,
                             detail="No measurements selected. POST /api/configure first.")

    def _do_analyze():
        return orch.analyze_prompt(prompt, category=category,
                                     session=state.session)

    try:
        prec = await run_in_threadpool(_do_analyze)
    except Exception as e:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Snapshot session for /api/session/restore (non-blocking cost)
    await run_in_threadpool(_snapshot_session_to_disk)

    if is_form:
        # TASM-shape response: flat result dict + session metadata
        # Index is the last prompt in the session (just appended).
        idx = len(state.session.record.prompts) - 1
        tasm_result = _tasm_shape_with_meta(prec.to_dict(), idx)
        return {
            "ok": True,
            "result": tasm_result,
            "plot_keys": tasm_result.get("_plot_keys", []),
            "session_n": len(state.session.record.prompts),
            "cache_size_bytes": state.cache.disk_usage(),
        }

    # JSON callers get TAGM-native shape
    return {"ok": True, "prompt": prec.to_dict()}


@app.post("/api/batch")
async def api_batch(request: Request):
    body = await request.json()
    prompts = body.get("prompts") or []
    if not prompts:
        raise HTTPException(status_code=400, detail="prompts required (list)")
    orch = state.require_orchestrator()
    if orch.capture_config is None:
        raise HTTPException(status_code=409,
                             detail="No CaptureConfig set. POST /api/capture first.")
    if not orch._selected:
        raise HTTPException(status_code=409,
                             detail="No measurements selected. POST /api/configure first.")

    def _do_batch():
        return orch.analyze_batch(prompts, session=state.session,
                                    progress=state.progress)

    try:
        records = await run_in_threadpool(_do_batch)
    except Exception as e:
        logger.exception("Batch failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "n_records": len(records),
            "prompts": [r.to_dict() for r in records]}


# Module-level state for the form-data batch endpoint. TASM ran batches
# in a background thread and the UI polled /api/progress until it saw
# "Batch complete". TAGM mirrors that.
_batch_running = False
_batch_lock = None  # initialized below in module scope

import threading as _threading
_batch_lock = _threading.Lock()


def _run_batch_in_thread(prompts: list[dict]) -> None:
    """Background batch worker. Logs progress messages the frontend can
    parse out of /api/progress; emits "Batch complete" on done."""
    global _batch_running
    try:
        orch = state.orchestrator
        if orch is None:
            state.progress("error", "Batch: orchestrator not initialized")
            return
        state.progress("batch", f"Starting batch of {len(prompts)} prompts")
        orch.analyze_batch(prompts, session=state.session,
                            progress=state.progress)
        _snapshot_session_to_disk()
        state.progress("done", f"Batch complete: {len(prompts)} prompts processed")
    except Exception as e:
        logger.exception("Batch thread failed")
        state.progress("error", f"Batch failed: {e}")
    finally:
        with _batch_lock:
            _batch_running = False


@app.post("/api/analyze_batch")
async def api_analyze_batch(request: Request):
    """TASM-compat batch endpoint. Accepts form data with a CSV file
    upload and assorted legacy flags. Parses the CSV, kicks off a
    background thread, returns immediately with {ok, started, n_prompts}.
    The frontend polls /api/progress until it sees 'Batch complete'."""
    global _batch_running

    orch = state.require_orchestrator()
    if orch.capture_config is None:
        raise HTTPException(status_code=409,
                             detail="No CaptureConfig set. POST /api/capture first.")
    if not orch._selected:
        raise HTTPException(status_code=409,
                             detail="No measurements selected. POST /api/configure first.")

    with _batch_lock:
        if _batch_running:
            raise HTTPException(
                status_code=409,
                detail="A batch is already running. Wait for it to finish.")

    form = await request.form()
    file = form.get("file")
    if not isinstance(file, UploadFile):
        raise HTTPException(status_code=400, detail="CSV file required")

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    import csv as _csv
    import io as _io
    reader = _csv.DictReader(_io.StringIO(content))
    prompts: list[dict] = []
    for row in reader:
        p = (row.get("prompt") or row.get("Prompt") or "").strip()
        c = (row.get("category") or row.get("Category") or "unknown").strip()
        if p:
            prompts.append({"prompt": p, "category": c})

    if not prompts:
        raise HTTPException(status_code=400,
                             detail="No valid prompts found in CSV.")

    with _batch_lock:
        _batch_running = True

    _threading.Thread(target=_run_batch_in_thread, args=(prompts,),
                       daemon=True).start()

    return {
        "ok": True,
        "started": True,
        "n_prompts": len(prompts),
        "message": f"Batch started: {len(prompts)} prompts. Watch progress log.",
    }


@app.post("/api/analysis/{name}")
async def api_run_analysis(name: str, request: Request):
    # Read body once. Empty bodies are fine — caller may run an analysis
    # with default parameters and no JSON payload.
    raw = await request.body()
    if raw:
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Body must be JSON or empty")
    else:
        body = {}
    params = body.get("params") or {}
    orch = state.require_orchestrator()

    def _do_run():
        return orch.run_analysis(name, state.session, params=params)

    try:
        result = await run_in_threadpool(_do_run)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
    return result


# ─── Templates ──────────────────────────────────────────────────────
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"


@app.get("/api/templates")
async def api_templates():
    entries = []
    if _TEMPLATES_DIR.exists():
        for csv_path in sorted(_TEMPLATES_DIR.glob("*.csv")):
            try:
                tmpl = parse_template_csv(csv_path)
                entries.append({
                    "name": tmpl.name,
                    "template_id": tmpl.template_id,
                    "rows": len(tmpl.rows),
                    "columns": len(tmpl.columns),
                    "n_cells": len(tmpl.cells),
                    "path": str(csv_path.relative_to(_PACKAGE_DIR.parent)),
                })
            except Exception as e:
                logger.warning(f"Template parse failed for {csv_path}: {e}")
    return {"templates": entries}


# ─── Probes ─────────────────────────────────────────────────────────
@app.get("/api/probes")
async def api_probes_list():
    return {"probe_sets": state.probe_store.list()}


@app.post("/api/probes/generate")
async def api_probes_generate(request: Request):
    body = await request.json()
    template_name = body.get("template")
    if not template_name:
        raise HTTPException(status_code=400, detail="template name required")
    depth_layers = body.get("depth_layers") or {}
    if not depth_layers:
        raise HTTPException(status_code=400, detail="depth_layers required "
                             "(dict mapping label -> layer_idx)")

    pipeline = state.require_pipeline()
    tmpl = load_template(template_name, templates_dir=_TEMPLATES_DIR)

    params = GenerationParams(
        depth_layers={k: int(v) for k, v in depth_layers.items()},
        include_final_norm=bool(body.get("include_final_norm", True)),
        filter_stopwords=bool(body.get("filter_stopwords", True)),
        project_through_o_delta=bool(body.get("project_through_o_delta", False)),
        min_tokens_per_cell=int(body.get("min_tokens_per_cell", 0)),
    )

    state.progress("generator", f"Generating probe set for template '{tmpl.name}'")

    def _do_generate():
        gen = EmbeddingGenerator(pipeline)
        ps = gen.generate(tmpl, params, progress=state.progress)
        state.probe_store.put(ps)
        return ps

    try:
        probe_set = await run_in_threadpool(_do_generate)
    except Exception as e:
        logger.exception("Probe generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": True,
        "set_id": probe_set.set_id,
        "template_id": probe_set.template_id,
        "template_name": probe_set.template_name,
        "capture_signature": probe_set.capture_signature,
        "n_probes": len(probe_set.probes),
        "depth_labels": list(probe_set.depth_labels),
    }


@app.delete("/api/probes/{set_id}")
async def api_probes_delete(set_id: str):
    ok = state.probe_store.delete(set_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Probe set not found")
    return {"ok": True}


# ─── Session I/O ────────────────────────────────────────────────────
@app.get("/api/session")
async def api_session():
    return state.session.to_dict()


@app.post("/api/session/reset")
async def api_session_reset():
    state.session = Session()
    state.selected_measurements = []
    if state.orchestrator is not None:
        state.orchestrator._selected = []
    return {"ok": True, "session_id": state.session.session_id}


@app.get("/api/session/export")
async def api_session_export():
    path = state.cache.session_path(state.session.session_id)
    await run_in_threadpool(export_session, state.session, path)
    return FileResponse(
        str(path),
        media_type="application/gzip",
        filename=f"{state.session.session_id}.json.gz",
    )


@app.post("/api/session/import")
async def api_session_import(file: UploadFile = File(...)):
    suffix = ".json.gz" if file.filename.endswith(".json.gz") else ".json"
    tmp = state.cache.layout.sessions / f"_import{suffix}"
    content = await file.read()

    def _do_import():
        with open(tmp, "wb") as out:
            out.write(content)
        try:
            return load_session(tmp)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    rec = await run_in_threadpool(_do_import)

    state.session = Session(session_id=rec.session_id)
    state.session.record = rec
    return {"ok": True, "session_id": rec.session_id,
            "n_prompts": len(rec.prompts)}


# ─── Presets ────────────────────────────────────────────────────────
@app.get("/api/presets")
async def api_presets_list():
    return {"presets": state.cache.list_presets()}


@app.post("/api/presets")
async def api_presets_save(request: Request):
    body = await request.json()
    name = body.get("name")
    config = body.get("config")
    if not name or not config:
        raise HTTPException(status_code=400, detail="name and config required")
    cfg = CaptureConfig.from_dict(config)
    path = state.cache.save_preset(name, cfg.to_json())
    return {"ok": True, "path": str(path)}


@app.get("/api/presets/{name}")
async def api_preset_load(name: str):
    try:
        js = state.cache.load_preset(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Preset not found")
    return json.loads(js)


@app.delete("/api/presets/{name}")
async def api_preset_delete(name: str):
    ok = state.cache.delete_preset(name)
    if not ok:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"ok": True}


# ─── Progress log ───────────────────────────────────────────────────
@app.get("/api/logs")
async def api_logs(since: float = 0):
    entries = [e for e in state.progress_log if e["ts"] > since]
    return {"now": time.time(), "entries": entries}


@app.get("/api/log")
async def api_log_alias(since: float = 0):
    """TASM alias for /api/logs."""
    return await api_logs(since=since)


@app.get("/api/progress")
async def api_progress():
    """TASM-shape progress endpoint. main.js polls this during loads
    and reads `data.log` (list of {stage, message, ts})."""
    return {
        "log": list(state.progress_log),
        "now": time.time(),
    }


# ─── TASM-compat /api/reset and /api/config ─────────────────────────
@app.post("/api/reset")
async def api_reset():
    """TASM-compat full reset: unload pipeline, clear session, reset
    selected measurements. main.js calls this for the 'Reset' button."""
    if state.pipeline is not None:
        try:
            state.pipeline.unload()
        except Exception:
            logger.exception("pipeline unload during reset failed")
    state.pipeline = None
    state.orchestrator = None
    state.selected_measurements = []
    state.session = Session()
    state.loading_state = {"active": False, "error": None}
    state.progress("reset", "Pipeline and session reset")
    return {"ok": True, "message": "Reset complete"}


@app.get("/api/config")
async def api_config_get():
    """TASM-compat UI-preferences endpoint.

    In TASM, `/api/config` was a simple persistent key-value store for
    frontend UI state: viz toggles, font sizes, card collapse states,
    the LTP layer-strategy dropdown value, etc. It had nothing to do
    with the analysis engine — the body shape was whatever the frontend
    chose to save, and GET returned it verbatim under a `config` key.

    main.js's `saveConfig()` and `loadConfig()` functions still expect
    exactly that contract (a `{ok, config: {...}}` response). An earlier
    pass through this file repurposed `/api/config` to mean capture +
    measurement selection, which silently broke every preference save
    in the Config tab. The capture/selections dispatcher is preserved
    under `/api/engine_setup` for anyone who wants it.
    """
    cfg_path = state.cache.layout.root / "ui_config.json"
    if cfg_path.exists():
        try:
            return {"ok": True, "config": json.loads(cfg_path.read_text())}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[/api/config GET] could not read {cfg_path}: {e}")
    return {"ok": True, "config": {}}


@app.post("/api/config")
async def api_config_post(request: Request):
    """TASM-compat UI-preferences save. Body is an opaque JSON blob
    that the frontend controls; we persist it verbatim and return
    `{ok: true}`. See `api_config_get` for the history on this endpoint."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Invalid JSON body: {e}")
    cfg_path = state.cache.layout.root / "ui_config.json"
    try:
        cfg_path.write_text(json.dumps(body, indent=2))
    except OSError as e:
        logger.error(f"[/api/config POST] could not write {cfg_path}: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": True}


@app.post("/api/engine_setup")
async def api_engine_setup(request: Request):
    """Unified capture + measurement-selection dispatcher.

    Previously mounted at POST /api/config — moved here because the
    TASM-derived frontend uses `/api/config` as a UI-preferences store
    (see api_config_get's docstring). This endpoint retains the original
    capture/selections behavior for any caller (including a future UI
    or scripted setup) that wants it.

    Body:
      {
        "capture": {... CaptureConfig dict ...},
        "selections": [{"name": ..., "params": {...}}, ...]
      }

    Either field is optional. If only `capture` is given, just sets
    capture; if only `selections`, just configures (will 409 if no
    capture is set yet)."""
    orch = state.require_orchestrator()
    body = await request.json()

    response: dict[str, Any] = {"ok": True}

    cap_dict = body.get("capture")
    if cap_dict:
        try:
            cap = CaptureConfig.from_dict(cap_dict)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(status_code=400,
                                 detail=f"Invalid capture: {e}")
        orch.set_capture_config(cap)
        orch._selected = []
        state.selected_measurements = []
        response["capture_set"] = True
        response["capture_signature"] = cap.signature()

    selections = body.get("selections")
    if selections is not None:
        if orch.capture_config is None:
            raise HTTPException(
                status_code=409,
                detail="Provide 'capture' first or set it via /api/capture.")
        parsed = [(s["name"], s.get("params") or {}) for s in selections]
        try:
            report = orch.configure_measurements(parsed)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        state.selected_measurements = [(n, p) for n, p in report["selected"]]
        response["configure_report"] = report
        response["ok"] = not report["errors"] and not report["violations"]

    return response


# ─── TASM session aliases ───────────────────────────────────────────
@app.post("/api/session/clear_all")
async def api_session_clear_all():
    """TASM alias for /api/session/reset."""
    return await api_session_reset()


@app.post("/api/session/clear_plots")
async def api_session_clear_plots():
    """TASM compat: no-op since TAGM doesn't render server-side plots
    (those come in session 5). Returns ok so the UI button works."""
    return {"ok": True, "message": "No server-side plot cache to clear (TAGM)",
            "freed_mb": 0, "cache_size_bytes": state.cache.disk_usage()}


# ─── Session results / dashboard / detail (TASM-compat) ─────────────
#
# TASM exposed three related read endpoints over the session:
#   /api/session/results — paginated, per-record TASM-shape (with _index, _plot_keys)
#   /api/dashboard       — summary aggregation, slim per-record fields
#   /api/results/detail  — start+count window, full TASM-shape records
#
# All three project TAGM's PromptRecord through the tasm_compat shaper
# so main.js gets the field names it expects.

def _tasm_shape_with_meta(prec_dict: dict, idx: int) -> dict:
    """Convert a TAGM PromptRecord dict to TASM shape and attach
    the `_index` and `_plot_keys` fields the frontend reads."""
    from tagm.service.tasm_compat import prompt_record_to_tasm_shape
    out = prompt_record_to_tasm_shape(prec_dict)
    out["_index"] = idx
    # Plot keys for this record. TASM's _plot_keys_for_result inspected
    # the record fields; here we match the semantics by listing keys
    # whose underlying data is present. Plots themselves come in session 5;
    # the keys list lets the frontend render plot card slots that lazy-load.
    plot_keys: list[str] = []
    if out.get("per_token_stress"):
        plot_keys.append("stress_per_token")
    if out.get("signed_attr"):
        plot_keys.append("signed_attribution")
    if out.get("amplitude_trajectory"):
        plot_keys.append("amplitude_trajectory")
    if out.get("ltp"):
        plot_keys.append("ltp_profiles")
    if out.get("sfd"):
        plot_keys.append("sfd_density")
    if out.get("rank_displacement"):
        plot_keys.append("rank_displacement")
    out["_plot_keys"] = plot_keys
    return out


def _slim_for_dashboard(tasm_record: dict) -> dict:
    """Strip per-token arrays and large objects; keep scalars + flags
    sufficient for the data table row. Mirrors TASM's slim projection."""
    slim: dict[str, Any] = {"_index": tasm_record.get("_index")}
    for k in ("prompt", "category", "seq_len", "stress_score",
              "net_correction", "entropy", "top2_share",
              "middle_share", "interior_cv", "kl_divergence",
              "n_negative_tokens", "has_negative_tokens"):
        if k in tasm_record:
            slim[k] = tasm_record[k]
    if "ltp" in tasm_record and isinstance(tasm_record["ltp"], dict):
        ltp = tasm_record["ltp"]
        slim["ltp"] = {k: ltp.get(k) for k in
                       ("mean_M", "mean_V", "max_prc", "n_directional")}
    if "sfd" in tasm_record and isinstance(tasm_record["sfd"], dict):
        slim["sfd"] = {k: tasm_record["sfd"].get(k) for k in
                       ("density_mean",)}
    if "rank_displacement" in tasm_record and isinstance(
            tasm_record["rank_displacement"], dict):
        rd = tasm_record["rank_displacement"]
        slim["rank_displacement"] = {k: rd.get(k) for k in
                                       ("mean_tau", "mean_overlap",
                                        "mean_replacement", "mean_disp_per_token",
                                        "total_displacement")}
    return slim


@app.get("/api/session/results")
async def api_session_results(page: int = 1, per_page: int = 10):
    """Paginated session results. Each record carries `_index` and
    `_plot_keys`. main.js calls this both for full-page detail (per_page=50)
    and for full-session bulk fetch (per_page=9999)."""
    prompts = state.session.record.prompts
    total = len(prompts)
    if total == 0:
        return {"ok": False, "results": [], "total": 0, "page": 1,
                "per_page": per_page, "total_pages": 0,
                "cache_size_bytes": state.cache.disk_usage()}

    per_page = max(1, min(int(per_page), 10000))
    total_pages = max(1, -(-total // per_page))   # ceil division
    page = max(1, min(int(page), total_pages))
    start = (page - 1) * per_page
    end = min(start + per_page, total)

    out_records = []
    for i in range(start, end):
        prec_dict = prompts[i].to_dict()
        out_records.append(_tasm_shape_with_meta(prec_dict, i))

    return {
        "ok": True,
        "results": out_records,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "cache_size_bytes": state.cache.disk_usage(),
    }


@app.get("/api/dashboard")
async def api_dashboard(force: bool = False):
    """Slim per-record list + session_info aggregate. Used by the data
    table. Same data as /api/session/results but stripped of large
    per-token arrays for transport efficiency."""
    prompts = state.session.record.prompts
    if len(prompts) == 0:
        return {"ok": False, "error": "No data in session.",
                "results": [], "session_info": {"n_results": 0},
                "cache_size_bytes": state.cache.disk_usage()}

    slim_results = []
    for i, p in enumerate(prompts):
        full = _tasm_shape_with_meta(p.to_dict(), i)
        slim_results.append(_slim_for_dashboard(full))

    session_info = {
        "n_results": len(prompts),
        "categories": state.session.categories(),
        "measurement_names": state.session.measurement_names(),
        "session_id": state.session.session_id,
    }
    if state.pipeline is not None and state.pipeline.loaded:
        desc = state.pipeline.describe()
        session_info["model"] = desc.get("model_pair", {}).get("instruct", "")

    return {
        "ok": True,
        "results": slim_results,
        "session_info": session_info,
        "cache_size_bytes": state.cache.disk_usage(),
    }


@app.get("/api/results/detail")
async def api_results_detail(start: int = 0, count: int = 50):
    """Start+count detail window. Used by terrain visualization which
    chunk-fetches full records to build a 3D scene."""
    prompts = state.session.record.prompts
    total = len(prompts)
    start = max(0, int(start))
    count = max(1, min(int(count), 1000))
    end = min(start + count, total)
    out_records = []
    for i in range(start, end):
        out_records.append(_tasm_shape_with_meta(prompts[i].to_dict(), i))
    return {
        "ok": True,
        "results": out_records,
        "start": start,
        "count": end - start,
        "total": total,
    }


# ─── Session mutation: remove / rerun / restore ────────────────────

@app.post("/api/session/remove")
async def api_session_remove(request: Request):
    """Remove specific prompt indices from the session. Reindexes
    remaining prompts so `_index` stays contiguous."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    indices = sorted(set(body.get("indices") or []), reverse=True)
    if not indices:
        raise HTTPException(status_code=400, detail="No indices provided.")

    prompts = state.session.record.prompts
    removed = 0
    for idx in indices:
        if 0 <= idx < len(prompts):
            prompts.pop(idx)
            removed += 1

    # Reassign prompt_ids to keep them contiguous (matches TASM's
    # behavior of renumbering _index after removal).
    for i, p in enumerate(prompts):
        p.prompt_id = f"p{i:04d}"

    logger.info(f"Session: removed {removed} prompts ({len(prompts)} remaining)")
    return {"ok": True, "removed": removed, "remaining": len(prompts)}


@app.post("/api/session/rerun")
async def api_session_rerun(request: Request):
    """Re-run analysis on specified prompt indices. Old results are
    removed from the session and fresh analyses are appended at the end."""
    if state.pipeline is None or not state.pipeline.loaded:
        raise HTTPException(status_code=400, detail="No model loaded.")
    orch = state.require_orchestrator()
    if orch.capture_config is None:
        raise HTTPException(status_code=409,
                             detail="No CaptureConfig set. POST /api/capture first.")
    if not orch._selected:
        raise HTTPException(status_code=409,
                             detail="No measurements selected.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    indices = sorted(set(body.get("indices") or []))
    if not indices:
        raise HTTPException(status_code=400, detail="No indices provided.")

    prompts = state.session.record.prompts
    to_rerun: list[dict] = []
    for idx in indices:
        if 0 <= idx < len(prompts):
            p = prompts[idx]
            to_rerun.append({"prompt": p.prompt, "category": p.category})

    if not to_rerun:
        raise HTTPException(status_code=400, detail="No valid indices.")

    # Remove old (reverse to keep indices stable during pop)
    for idx in sorted(indices, reverse=True):
        if 0 <= idx < len(prompts):
            prompts.pop(idx)
    for i, p in enumerate(prompts):
        p.prompt_id = f"p{i:04d}"

    state.progress("rerun", f"Rerunning {len(to_rerun)} prompts...")

    def _do_rerun():
        return orch.analyze_batch(to_rerun, session=state.session,
                                     progress=state.progress)

    try:
        records = await run_in_threadpool(_do_rerun)
    except Exception as e:
        logger.exception("Rerun failed")
        raise HTTPException(status_code=500, detail=str(e))

    await run_in_threadpool(_snapshot_session_to_disk)
    return {"ok": True, "rerun": len(records),
            "total": len(state.session.record.prompts)}


# ─── Session restore from disk ─────────────────────────────────────
#
# TASM persisted session state to disk continuously and offered an
# explicit restore. TAGM's session lives in memory; for restore semantics
# we snapshot the session JSON to the cache directory whenever a prompt
# is added (cheap), and restore from the snapshot on demand.

def _snapshot_session_to_disk() -> None:
    """Persist the current session state to disk so it can be restored
    after a server restart. Called after each add_prompt and after
    successful analyses."""
    try:
        snapshot_path = state.cache.layout.root / "last_session.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(state.session.to_dict(), f)
    except Exception:
        # Snapshot failure must not break analysis.
        logger.exception("Failed to snapshot session")


def _has_session_snapshot() -> bool:
    snapshot_path = state.cache.layout.root / "last_session.json"
    return snapshot_path.exists()


@app.post("/api/session/restore")
async def api_session_restore():
    """Restore session from the most recent disk snapshot. Returns
    {ok, n_results, model, cache_size_bytes} on success."""
    snapshot_path = state.cache.layout.root / "last_session.json"
    if not snapshot_path.exists():
        raise HTTPException(status_code=404,
                             detail="No session snapshot found on disk.")

    try:
        with open(snapshot_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500,
                             detail=f"Could not parse snapshot: {e}")

    # Reconstruct session from the snapshot.
    from tagm.service.session import Session, PromptRecord, SessionRecord
    new_session = Session()
    rec_dict = data
    # Rehydrate the session record. We accept both top-level dicts
    # (TAGM-native) and earlier snapshot shapes.
    new_session.record.session_id = rec_dict.get("session_id",
                                                   new_session.record.session_id)
    new_session.record.created_at = rec_dict.get("created_at",
                                                   new_session.record.created_at)
    new_session.record.model_pair = rec_dict.get("model_pair") or {}
    new_session.record.structure = rec_dict.get("structure") or {}
    new_session.record.capture_config = rec_dict.get("capture_config") or {}
    new_session.record.measurements_config = (
        rec_dict.get("measurements_config") or {})
    new_session.record.analyses = rec_dict.get("analyses") or {}
    new_session.record.probe_sets = rec_dict.get("probe_sets") or []
    # Prompts: each is a PromptRecord dict
    new_session.record.prompts = []
    for p in rec_dict.get("prompts") or []:
        prec = PromptRecord(
            prompt=p.get("prompt", ""),
            category=p.get("category", ""),
            tokens=list(p.get("tokens") or []),
            seq_len=int(p.get("seq_len", 0)),
            measurements=p.get("measurements") or {},
            metadata=p.get("metadata") or {},
            prompt_id=p.get("prompt_id", ""),
        )
        new_session.record.prompts.append(prec)

    state.session = new_session
    n = len(new_session.record.prompts)
    model = (new_session.record.model_pair or {}).get("instruct", "")

    logger.info(f"[SESSION] Restored {n} results from disk snapshot, "
                f"model: {model or 'unknown'}")
    return {
        "ok": True,
        "n_results": n,
        "categories": new_session.categories(),
        "model": model,
        "cache_size_bytes": state.cache.disk_usage(),
    }


# ─── User info ─────────────────────────────────────────────────────
#
# TASM stored a per-session user-info blob (probably for export labeling).
# main.js sends form data and ignores the response. We accept it and
# store the most recent submission on the AppState.

@app.post("/api/user_info")
async def api_user_info(request: Request):
    """Accept a user info form submission. Stored on app state for
    inclusion in exports. Failure is non-fatal — main.js wraps the
    call in try/catch and ignores errors."""
    form = await request.form()
    info = {k: form.get(k) for k in form.keys()
            if not isinstance(form.get(k), UploadFile)}
    state.user_info = info
    return {"ok": True, "user_info": info}


# ─── Prompts ────────────────────────────────────────────────────────
_PROMPTS_FILE = _PACKAGE_DIR.parent / "prompts.csv"


@app.get("/api/prompts")
async def api_prompts():
    """Return the built-in prompt library."""
    if not _PROMPTS_FILE.exists():
        return {"prompts": []}
    import csv as _csv
    out = []
    with open(_PROMPTS_FILE, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            out.append({"prompt": row.get("prompt", "").strip(),
                        "category": row.get("category", "").strip()})
    return {"prompts": out}


@app.post("/api/prompts")
async def api_prompts_add(request: Request):
    """Append a prompt to the built-in library CSV. Form data:
    prompt (required), category (default 'benign')."""
    form = await request.form()
    prompt = (form.get("prompt") or "").strip()
    category = (form.get("category") or "benign").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")

    import csv as _csv
    file_existed = _PROMPTS_FILE.exists()
    with open(_PROMPTS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f)
        if not file_existed:
            writer.writerow(["prompt", "category"])
        writer.writerow([prompt, category])

    # Return updated library (matches TASM)
    out = []
    with open(_PROMPTS_FILE, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            out.append({"prompt": row.get("prompt", "").strip(),
                        "category": row.get("category", "").strip()})
    logger.info(f"Prompt added to library: [{category}] {prompt[:60]}...")
    return {"ok": True, "prompts": out}


# ─── Modules: TASM-compat unified measurement+analysis interface ────
#
# TASM's frontend treated every measurement and analysis as a "module"
# with the same async-job interface: list, run (background), poll
# status, fetch results. The TAGM ModuleRunner wraps both kinds behind
# that interface. See tagm/service/modules_runner.py for details.

from tagm.service.modules_runner import runner as _module_runner


@app.get("/api/modules")
async def api_modules():
    """List all modules (measurements + analyses) with status."""
    return {"ok": True, "modules": _module_runner.list_modules()}


@app.post("/api/modules/upload_template")
async def api_modules_upload_template(file: UploadFile = File(...)):
    """Upload a template CSV. Saves to the templates directory so it
    becomes available to probe-using measurements."""
    templates_dir = _PACKAGE_DIR / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    dest = templates_dir / file.filename
    content = await file.read()
    try:
        with open(dest, "wb") as f:
            f.write(content)
    except OSError as e:
        raise HTTPException(status_code=500,
                             detail=f"Could not write template: {e}")
    return {"ok": True, "filename": file.filename}


@app.post("/api/modules/{module_name}/run")
async def api_modules_run(module_name: str, request: Request):
    """Start a module asynchronously. Body: JSON {params: {...}}."""
    body = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = {}
    params = body.get("params") or {}
    result = _module_runner.run(
        name=module_name,
        session=state.session,
        orchestrator=state.orchestrator,
        params=params,
        progress_fn=state.progress,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "run failed"))
    return result


@app.get("/api/modules/{module_name}/status")
async def api_modules_status(module_name: str):
    """Poll module execution status."""
    return _module_runner.get_status(module_name)


@app.get("/api/modules/{module_name}/results")
async def api_modules_results(module_name: str):
    """Fetch module results.

    First checks the ModuleRunner's stash (set by an explicit run);
    falls back to data already in the session (for measurements that
    were computed via /api/analyze rather than via a module run, or
    analyses already produced).

    For analyses, flattens AnalysisResult shape ({analysis_name,
    scalars, objects, per_prompt, parameters, warnings}) into the flat
    TASM wire format the frontend renderers read at top level. Ported
    analyses (correction_heatmap, correction_field_topology) stuff
    their TASM-shape outputs under .objects; hoisting them here is
    what makes renderCorrectionHeatmapResults and renderCFTResults
    find r.subjects, r.aggregate, r.displacement_stats, etc.
    """
    results = _module_runner.get_results(module_name)
    if results is None:
        # Session fallback: look for the data in the session record
        if module_name in state.session.record.analyses:
            results = state.session.record.analyses[module_name]
        else:
            # Look for the measurement across prompts
            per_prompt = []
            for p in state.session.record.prompts:
                m = p.measurements.get(module_name)
                if m:
                    per_prompt.append({
                        "prompt": p.prompt,
                        "category": p.category,
                        "result": m,
                    })
            if per_prompt:
                results = {
                    "ok": True,
                    "name": module_name,
                    "n_prompts": len(per_prompt),
                    "per_prompt": per_prompt,
                }
    if results is None:
        raise HTTPException(status_code=404,
                             detail=f"No results available for '{module_name}'.")

    results = _flatten_analysis_shape_for_ui(module_name, results)
    return {"ok": True, "results": results}


# Analyses that have been ported to TASM wire format. For these, the
# AnalysisResult.objects dict contains top-level fields the TASM-derived
# JS renderers read directly (e.g. r.subjects, r.aggregate for
# correction_heatmap); _flatten_analysis_shape_for_ui hoists objects.*
# to top level so renderCorrectionHeatmapResults and friends see what
# they expect. Analyses not in this set keep TAGM's nested
# AnalysisResult shape and are rendered by the TAGM-native JS
# fallback (renderTagmNativeResults in static/js/main.js).
_TASM_WIRE_FORMAT_ANALYSES = frozenset({
    "correction_heatmap",
    "correction_field_topology",
    "comparative_analysis",
    "token_variance",
    "correction_backscatter",
    "correction_manifold",
    "mi_readiness",
    "mi_instrumentation",
    "domain_surface",
})


def _flatten_analysis_shape_for_ui(name: str, results: dict) -> dict:
    """If `results` is AnalysisResult-shape AND the analysis has been
    ported to TASM wire format, promote .objects and .scalars to top
    level so the TASM-derived JS renderers can read fields directly.

    For any other shape, or for analyses not in the TASM-wire set,
    returns `results` unchanged — the TAGM-native JS renderer takes
    those through the nested shape.
    """
    if not isinstance(results, dict):
        return results
    if name not in _TASM_WIRE_FORMAT_ANALYSES:
        return results
    if ("analysis_name" not in results
            or "scalars" not in results
            or "objects" not in results):
        return results

    flat: dict = {}
    flat["analysis_name"] = results.get("analysis_name")
    flat["analysis_version"] = results.get("analysis_version")
    flat["warnings"] = results.get("warnings") or []
    flat["parameters"] = results.get("parameters") or {}
    flat["per_prompt"] = results.get("per_prompt") or {}

    for k, v in (results.get("scalars") or {}).items():
        flat.setdefault(k, v)
    for k, v in (results.get("objects") or {}).items():
        flat[k] = v

    flat["_tagm_analysis_native"] = {
        "scalars": results.get("scalars") or {},
        "objects": results.get("objects") or {},
    }
    return flat


@app.post("/api/modules/{module_name}/reset")
async def api_modules_reset(module_name: str):
    """Reset a module to idle state."""
    result = _module_runner.reset(module_name)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@app.get("/api/modules/{module_name}/download_log")
async def api_modules_download_log(module_name: str):
    """Download the standalone log for a completed module run."""
    log_path = _module_runner.get_log_path(module_name)
    if not log_path or not Path(log_path).exists():
        raise HTTPException(status_code=404,
                             detail="No log file available for this module.")
    return FileResponse(log_path, media_type="application/json",
                         filename=f"module_{module_name}_log.json")


# ─── Engine config (TASM-compat) ───────────────────────────────────
#
# TASM exposed engine-wide parameters through this endpoint with
# defaults that the UI presented as an "Advanced Parameters" panel.
# In TAGM, most of those parameters now live per-measurement (as
# scope parameters) and the orchestrator carries no engine-wide
# settings. We expose a small set of orchestrator-level defaults
# and accept arbitrary key/value updates that get echoed back.

_ENGINE_DEFAULTS = {
    # Default measurement parameters that apply across runs
    "ltp_k": 8,
    "ltp_layer_strategy": "signal",
    "ltp_svd_rank": 0,
    "sfd_svd_k": 16,
    "sfd_svd_seed": 42,
    "compute_kl": False,
    "compute_trajectory": True,
    "compute_ltp": False,
    "compute_sfd": False,
    "topk": 8,
    # Statistics
    "boundary_fraction": 0.1,
    "proof1_threshold": 1e-4,
    # Serialization
    "max_export_size_mb": 50,
}

_engine_config: dict = dict(_ENGINE_DEFAULTS)


@app.get("/api/engine_config")
async def api_engine_config_get():
    """Return current engine-wide config and defaults."""
    return {
        "ok": True,
        "config": dict(_engine_config),
        "defaults": dict(_ENGINE_DEFAULTS),
    }


@app.post("/api/engine_config")
async def api_engine_config_set(request: Request):
    """Update engine-wide config. Body: JSON {key: value, ...}.

    Unknown keys are accepted (forward compat); type-checked against
    defaults when present in the defaults dict."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    updates = body if isinstance(body, dict) else {}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    for k, v in updates.items():
        if k in _ENGINE_DEFAULTS:
            default = _ENGINE_DEFAULTS[k]
            # Coerce numeric/boolean types where defaults indicate
            if isinstance(default, bool):
                v = bool(v) if not isinstance(v, str) else v.lower() in (
                    "true", "1", "yes")
            elif isinstance(default, int) and not isinstance(default, bool):
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400,
                                         detail=f"'{k}' must be int")
            elif isinstance(default, float):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400,
                                         detail=f"'{k}' must be float")
        _engine_config[k] = v

    return {"ok": True, "config": dict(_engine_config)}


@app.post("/api/engine_config/reset")
async def api_engine_config_reset():
    """Reset engine config to defaults."""
    global _engine_config
    _engine_config = dict(_ENGINE_DEFAULTS)
    return {"ok": True, "config": dict(_engine_config)}


# ─── Probe set apply / status / clear (TASM-compat) ────────────────
#
# TASM let the user upload a probe CSV that became the "active" probe
# set used by probe-consuming measurements. TAGM has per-measurement
# probe references via (template_id, capture_signature). We bridge by
# tracking the most recently applied template as the implicit default
# for measurements whose params don't specify one.

_probe_apply_state: dict = {
    "active": False,
    "progress": "",
    "error": None,
    "result": None,
}
_probe_apply_lock = threading.Lock()
_active_probe_template: Optional[dict] = None  # {template_id, name, ...}


def _probe_apply_worker(file_bytes: bytes, filename: str) -> None:
    """Background: parse template CSV, register with template store,
    generate embeddings via the loaded pipeline, mark as active."""
    global _active_probe_template
    import tempfile
    try:
        _probe_apply_state["progress"] = "Parsing template..."
        from tagm.probes.template import parse_template_csv
        from tagm.probes.generator import EmbeddingGenerator, GenerationParams

        # parse_template_csv expects a path; write the upload to a temp file
        with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".csv", delete=False) as tf:
            tf.write(file_bytes)
            temp_path = Path(tf.name)
        try:
            template = parse_template_csv(temp_path,
                                            name=Path(filename).stem)
        finally:
            try:
                temp_path.unlink()
            except OSError:
                pass

        # Count probes (tokens across all cells) and subjects (rows)
        n_probes = sum(len(c.tokens) for c in template.cells)
        n_subjects = len(template.rows)
        n_levels = len(template.columns)
        level_names = list(template.columns)

        _probe_apply_state["progress"] = (
            f"Embedding {n_probes} probes...")
        if state.pipeline is None or not state.pipeline.loaded:
            raise RuntimeError("No model loaded; cannot embed probes.")

        # Pick layers — use L50/L75 of the model as TASM did
        n_layers = state.pipeline.adapter.n_layers(state.pipeline.instruct_model)
        layer_50 = max(0, n_layers // 2)
        layer_75 = max(0, (n_layers * 3) // 4)

        gen = EmbeddingGenerator(state.pipeline)
        gen_params = GenerationParams(
            depth_layers={"subject": layer_50, "escalation": layer_75},
            include_final_norm=True,
        )
        probe_set = gen.generate(
            template=template,
            params=gen_params,
            progress=lambda s, m: _probe_apply_state.update(
                {"progress": m}),
        )
        state.probe_store.put(probe_set)

        _active_probe_template = {
            "filename": filename,
            "template_id": template.template_id,
            "n_probes": n_probes,
            "n_subjects": n_subjects,
            "n_levels": n_levels,
            "levels": level_names,
            "layer_L50": layer_50,
            "layer_L75": layer_75,
            "set_id": probe_set.set_id,
        }
        _probe_apply_state["result"] = dict(_active_probe_template)
        _probe_apply_state["progress"] = "Complete."
        logger.info(f"[probe_set] applied {filename}: {n_probes} probes")
    except Exception as e:
        logger.exception("[probe_set] apply failed")
        _probe_apply_state["error"] = str(e)
    finally:
        _probe_apply_state["active"] = False


@app.post("/api/probe_set/apply")
async def api_probe_set_apply(file: UploadFile = File(...)):
    """Upload a probe template CSV; embed via the loaded pipeline."""
    if state.pipeline is None or not state.pipeline.loaded:
        raise HTTPException(status_code=400,
                             detail="No model loaded.")
    with _probe_apply_lock:
        if _probe_apply_state["active"]:
            raise HTTPException(
                status_code=409,
                detail="Probe embedding already in progress.")
        _probe_apply_state["active"] = True
        _probe_apply_state["progress"] = "Starting..."
        _probe_apply_state["error"] = None
        _probe_apply_state["result"] = None

    content = await file.read()
    threading.Thread(
        target=_probe_apply_worker,
        args=(content, file.filename),
        daemon=True).start()

    return {"ok": True, "status": "started", "filename": file.filename}


@app.get("/api/probe_set/apply_status")
async def api_probe_set_apply_status():
    """Poll probe-embedding progress."""
    return {
        "ok": True,
        "active": _probe_apply_state["active"],
        "progress": _probe_apply_state["progress"],
        "error": _probe_apply_state["error"],
        "result": _probe_apply_state["result"],
    }


@app.get("/api/probe_set/status")
async def api_probe_set_status():
    """Return info about the currently active probe set, if any."""
    if not _active_probe_template:
        return {"ok": True, "active": False}
    info = dict(_active_probe_template)
    info["ok"] = True
    info["active"] = True
    # Check whether the probe set is in the store
    try:
        sets = state.probe_store.list()
        info["cached"] = any(
            s.get("template_id") == info.get("template_id") for s in sets)
    except Exception:
        info["cached"] = False
    return info


@app.post("/api/probe_set/clear_caches")
async def api_probe_set_clear_caches():
    """Delete all probe-set caches and the SFD precompute cache."""
    deleted = 0
    try:
        from tagm.measurement.modules.spectral_field_density import (
            clear_sfd_caches)
        clear_sfd_caches()
        deleted += 1   # SFD cache cleared
    except Exception:
        pass

    # Clear stored probe sets
    try:
        for probe_set_meta in list(state.probe_store.list()):
            sid = probe_set_meta.get("set_id")
            if sid:
                state.probe_store.delete(sid)
                deleted += 1
    except Exception as e:
        logger.warning(f"[probe_set] partial cache clear: {e}")

    logger.info(f"[probe_set] cleared {deleted} cache items")
    return {"ok": True, "deleted": deleted}


# ─── Visualization HTML routes (TASM-compat) ───────────────────────
#
# TASM served these viz pages at fixed paths. The pages are static HTML
# in /static/ that fetch results from /api/modules/{name}/results.

@app.get("/chat", include_in_schema=False)
async def serve_chat():
    return FileResponse(str(_STATIC_DIR / "chat.html"))


@app.get("/domain_surface_viz", include_in_schema=False)
async def serve_domain_surface_viz():
    return FileResponse(str(_STATIC_DIR / "domain_surface_viz.html"))


@app.get("/correction_manifold_viz", include_in_schema=False)
async def serve_correction_manifold_viz():
    return FileResponse(str(_STATIC_DIR / "correction_manifold_viz.html"))


@app.get("/correction_heatmap_viz", include_in_schema=False)
async def serve_correction_heatmap_viz():
    return FileResponse(str(_STATIC_DIR / "correction_heatmap_viz.html"))


@app.get("/correction_backscatter_viz", include_in_schema=False)
async def serve_correction_backscatter_viz():
    return FileResponse(str(_STATIC_DIR / "correction_backscatter_viz.html"))


# ─── Chat (TASM-compat) ────────────────────────────────────────────

@app.post("/api/set_inference_model")
async def api_set_inference_model(request: Request):
    """Toggle the chat-side inference model between instruct and base.
    Both share the loaded pipeline; this just records the choice for
    /api/chat to consult."""
    form = await request.form()
    model_class = (form.get("model_class") or "").strip().lower()
    if model_class not in ("instruct", "base"):
        raise HTTPException(status_code=400,
                             detail="model_class must be 'instruct' or 'base'")
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "message": "No model loaded."}
    if model_class == "base" and state.pipeline.base_model is None:
        return {"ok": False, "message": "Base model not loaded into memory."}
    state.inference_class = model_class
    return {"ok": True, "active": model_class}


@app.post("/api/chat")
async def api_chat(request: Request):
    """Generate a chat response from the loaded model.

    Body:
      {
        "messages": [{"role": ..., "content": ...}, ...],
        "max_tokens": int (optional, default 256),
        "analyze": bool (optional) — also run TAGM measurements on the prompt
        "analyze_response": bool (optional) — also run on the response
        "category": str (optional)
      }
    """
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "error": "No model loaded"}

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages") or []
    max_tokens = int(body.get("max_tokens", 256))
    do_analyze = bool(body.get("analyze", False))
    do_analyze_response = bool(body.get("analyze_response", False))
    category = (body.get("category") or "").strip()

    if not messages:
        return {"ok": False, "error": "No messages"}

    use_base = (state.inference_class == "base")

    from tagm.service.chat import generate_chat_response

    def _do_generate():
        return generate_chat_response(
            pipeline=state.pipeline,
            messages=messages,
            max_tokens=max_tokens,
            use_base=use_base,
        )

    chat_result = await run_in_threadpool(_do_generate)
    if not chat_result.get("ok"):
        return chat_result

    response_text = chat_result["response"]
    out = {
        "ok": True,
        "response": response_text,
        "model_class": chat_result["model_class"],
    }

    # Optional analysis of the user prompt
    if do_analyze and state.orchestrator is not None and \
            state.orchestrator.capture_config is not None and \
            state.orchestrator._selected:
        try:
            prompt_text = messages[-1].get("content", "")

            def _analyze_user():
                p = state.orchestrator.analyze_prompt(
                    prompt_text, category=category or "chat_user",
                    session=state.session)
                p.metadata["role"] = "user"
                return p

            await run_in_threadpool(_analyze_user)
            await run_in_threadpool(_snapshot_session_to_disk)
            out["prompt_analyzed"] = True
        except Exception as e:
            logger.exception("[chat] prompt analysis failed")
            out["prompt_analysis_error"] = str(e)

    # Optional analysis of the model's response
    if do_analyze_response and state.orchestrator is not None and \
            state.orchestrator.capture_config is not None and \
            state.orchestrator._selected:
        try:
            def _analyze_response():
                p = state.orchestrator.analyze_prompt(
                    response_text, category="model_response",
                    session=state.session)
                p.metadata["role"] = "assistant"
                return p

            await run_in_threadpool(_analyze_response)
            await run_in_threadpool(_snapshot_session_to_disk)
            out["response_analyzed"] = True
        except Exception as e:
            logger.exception("[chat] response analysis failed")
            out["response_analysis_error"] = str(e)

    return out


# ─── Plot generation (TASM-compat) ─────────────────────────────────
#
# TASM rendered per-prompt plots server-side and served them as PNGs.
# /api/plots/individual/{idx}/{key} renders for a specific prompt;
# /api/plots/{key} would render aggregate views (TASM had this but
# main.js never invokes it — kept here as an alias for completeness).

@app.get("/api/plots/individual/{index}/{plot_key}")
async def api_plots_individual(index: int, plot_key: str):
    """Render a per-prompt plot. Returns image/png on success.
    Offloaded to a threadpool because matplotlib rendering is CPU-heavy
    and the browser fires many parallel img requests."""
    prompts = state.session.record.prompts
    if index < 0 or index >= len(prompts):
        raise HTTPException(status_code=404,
                             detail=f"Prompt index {index} out of range")

    from tagm.service.plots import render_plot
    from tagm.service.tasm_compat import prompt_record_to_tasm_shape

    prompt_dict = prompts[index].to_dict()

    def _do_render():
        shape = prompt_record_to_tasm_shape(prompt_dict)
        return render_plot(plot_key, shape)

    png = await run_in_threadpool(_do_render)
    if png is None:
        raise HTTPException(status_code=404,
                             detail=f"Unknown plot key '{plot_key}'")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/plots/{plot_key}")
async def api_plots_aggregate(plot_key: str):
    """Render an aggregate plot across all session prompts. Currently
    falls back to rendering against the first prompt (TASM's main.js
    doesn't call this path, but the route is preserved for parity)."""
    if not state.session.record.prompts:
        raise HTTPException(status_code=404,
                             detail="No prompts in session")

    from tagm.service.plots import render_plot
    from tagm.service.tasm_compat import prompt_record_to_tasm_shape

    prompt_dict = state.session.record.prompts[0].to_dict()

    def _do_render():
        shape = prompt_record_to_tasm_shape(prompt_dict)
        return render_plot(plot_key, shape)

    png = await run_in_threadpool(_do_render)
    if png is None:
        raise HTTPException(status_code=404,
                             detail=f"Unknown plot key '{plot_key}'")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


# ─── Export (TASM-compat two-step) ─────────────────────────────────
#
# TASM had POST /api/export to prepare an export, then GET
# /api/export/download to stream it. TAGM's native /api/session/export
# streams directly. We add the two-step alias for main.js.

_export_state = {"ready": False, "path": None}


@app.post("/api/export")
async def api_export_prepare(request: Request):
    """Prepare a session export. Body is JSON of options (csv, json,
    pdf, charts flags). All ignored except as informational — TAGM's
    export is a single .json.gz file regardless.

    Emits 'exporting' and 'done' progress events so main.js's
    pollExport() can detect completion. main.js waits for a progress
    entry with stage='done' and a message containing 'Export ready'
    before it hits /api/export/download — without these events, the
    UI polls indefinitely and the download never fires."""
    try:
        _ = await request.json()
    except Exception:
        pass

    state.progress("exporting", "Preparing session export...")

    # Generate the export and stash it to disk
    from tagm.service.export import export_session
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json.gz", prefix="tagm_export_")
    import os
    os.close(fd)

    def _do_export():
        export_session(state.session, Path(path))

    try:
        await run_in_threadpool(_do_export)
    except Exception as e:
        logger.exception("Export failed")
        state.progress("error", f"Export failed: {e}")
        return {"ok": False, "error": str(e)}

    _export_state["ready"] = True
    _export_state["path"] = path
    size_bytes = Path(path).stat().st_size if Path(path).exists() else 0
    state.progress("done",
        f"Export ready ({size_bytes // 1024} KB). Click Download.")
    return {"ok": True, "ready": True, "path": path,
            "size_bytes": size_bytes}


@app.get("/api/export/download")
async def api_export_download():
    """Stream the prepared export file. Falls back to generating fresh
    if /api/export hasn't been called first."""
    path = _export_state.get("path")
    if not path or not Path(path).exists():
        # Generate on the fly (no progress events — this is a direct
        # download path, not the polled two-step flow).
        from tagm.service.export import export_session
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".json.gz", prefix="tagm_export_")
        import os
        os.close(fd)

        def _do_export():
            export_session(state.session, Path(path))

        await run_in_threadpool(_do_export)
        _export_state["path"] = path

    return FileResponse(
        path, media_type="application/gzip",
        filename="tagm_session.json.gz",
    )
