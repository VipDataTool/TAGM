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
import time
import threading
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tagm.core.pipeline import Pipeline
from tagm.engine.analyzer import Analyzer
from tagm.engine.result import result_to_dict
from tagm.engine import config as engine_config
from tagm.engine.session import Session
from tagm.engine.app_core import (
    AppState, state,
    _load_model_registry,
    api_status_handler,
    api_config_get_handler, api_config_post_handler,
    api_load_model_handler,
    api_analyze_handler,
    api_analyze_batch_handler,
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
    MODELS_FILE, CONFIG_FILE,
)
from tagm.engine.modules import ModuleRunner
from tagm.probes.store import ProbeStore
from tagm.probes.generator import EmbeddingGenerator, GenerationParams
from tagm.probes.template import load_template, parse_template_csv
from tagm.core.cache import Cache

logger = logging.getLogger("tagm")

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
_module_runner = ModuleRunner()
_cache = Cache()
_probe_store = ProbeStore(root=_cache.layout.probes)

# When a model loads, propagate the pipeline to modules that need it
state.on_model_loaded(_module_runner.set_pipeline)

# ─── App ────────────────────────────────────────────────────────
app = FastAPI(title="TAGM", version="2.0.0")

# Static files
_static_dir = _PACKAGE_DIR.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


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

for _viz in ("chat", "domain_surface_viz", "correction_manifold_viz",
             "correction_heatmap_viz", "correction_backscatter_viz"):
    def _make_viz_route(name):
        async def handler():
            p = _static_dir / f"{name}.html"
            if p.exists():
                return HTMLResponse(p.read_text(encoding="utf-8"))
            raise HTTPException(status_code=404)
        return handler
    app.get(f"/{_viz}", include_in_schema=False)(_make_viz_route(_viz))


# ═══════════════════════════════════════════════════════════════
# Core API — delegated to engine.app_core
# ═══════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    return api_status_handler()

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
        return JSONResponse(status_code=400, content={"error": "All fields required."})
    pairs = _load_model_registry()
    found = False
    for p in pairs:
        if p.get("id") == id_clean:
            p.update({"name": name.strip(), "base": base.strip(), "instruct": instruct.strip()})
            found = True
            break
    if not found:
        pairs.append({"id": id_clean, "name": name.strip(), "base": base.strip(), "instruct": instruct.strip()})
    with open(MODELS_FILE, "w") as f:
        json.dump(pairs, f, indent=2)
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
    cls = (form.get("model_class") or "instruct").strip()
    state.inference_class = cls
    return {"ok": True, "inference_class": cls}

@app.post("/api/reset")
async def reset():
    return api_reset_handler()

@app.post("/api/analyze")
async def analyze(request: Request):
    return await api_analyze_handler(request)

@app.post("/api/analyze_batch")
async def analyze_batch(request: Request):
    return await api_analyze_batch_handler(request)

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

@app.post("/api/session/restore")
async def restore():
    return api_session_restore_handler()

@app.post("/api/session/clear_plots")
async def clear_plots():
    return {"ok": True, "message": "No server-side plot cache.", "freed_mb": 0}

@app.post("/api/session/clear_all")
async def clear_all():
    return api_reset_handler()

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
    rerun_results = []
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
                rd["_index"] = idx
                rd["_plot_keys"] = _plot_keys_for_result(rd)
                state.session.results[idx] = rd
                rerun_results.append(rd)
        except Exception as e:
            logger.exception(f"Rerun {idx} failed")
    state.session.save_to_disk()
    return {"ok": True, "n_rerun": len(rerun_results)}


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
    prompts = []
    if _PROMPTS_FILE.exists():
        with open(_PROMPTS_FILE, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                prompts.append({"prompt": row.get("prompt", ""), "category": row.get("category", "")})
    return {"prompts": prompts}

@app.post("/api/prompts")
async def add_prompt(prompt: str = Form(...), category: str = Form("")):
    exists = _PROMPTS_FILE.exists()
    with open(_PROMPTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["prompt", "category"])
        if not exists:
            writer.writeheader()
        writer.writerow({"prompt": prompt.strip(), "category": category.strip()})
    prompts = []
    with open(_PROMPTS_FILE, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            prompts.append({"prompt": row.get("prompt", ""), "category": row.get("category", "")})
    return {"ok": True, "prompts": prompts}


# ═══════════════════════════════════════════════════════════════
# Engine config
# ═══════════════════════════════════════════════════════════════

@app.get("/api/engine_config")
async def get_engine_config():
    return {"ok": True, "config": engine_config.as_dict(), "defaults": dict(engine_config.DEFAULTS)}

@app.post("/api/engine_config")
async def set_engine_config(request: Request):
    body = await request.json()
    engine_config.update(body)
    return {"ok": True, "config": engine_config.as_dict()}

@app.post("/api/engine_config/reset")
async def reset_engine_config():
    engine_config.reset()
    return {"ok": True, "config": engine_config.as_dict()}


# ═══════════════════════════════════════════════════════════════
# Modules (analysis modules)
# ═══════════════════════════════════════════════════════════════

@app.get("/api/modules")
async def list_modules():
    return {"ok": True, "modules": _module_runner.list_modules()}

@app.post("/api/modules/upload_template")
async def upload_template(file: UploadFile = File(...)):
    templates_dir = _PACKAGE_DIR / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    dest = templates_dir / file.filename
    with open(dest, "wb") as f:
        f.write(content)
    return {"ok": True, "filename": file.filename}

@app.post("/api/modules/{module_name}/run")
async def run_module(module_name: str, request: Request):
    body = {}
    ct = request.headers.get("content-type", "")
    if ct.startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = {}
    params = body.get("params") or {}
    result = _module_runner.run_module(
        name=module_name,
        session_results=state.session.results,
        params=params,
        session_dir=state.session.session_dir if hasattr(state.session, 'session_dir') else None,
    )
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Module failed to start.")}
    return result

@app.get("/api/modules/{module_name}/status")
async def module_status(module_name: str):
    return _module_runner.get_status(module_name)

@app.get("/api/modules/{module_name}/results")
async def module_results(module_name: str):
    results = _module_runner.get_results(module_name)
    if results is None:
        raise HTTPException(status_code=404, detail=f"No results for '{module_name}'.")
    return {"ok": True, "results": results}

@app.post("/api/modules/{module_name}/reset")
async def reset_module(module_name: str):
    return _module_runner.reset_module(module_name)

@app.get("/api/modules/{module_name}/download_log")
async def download_module_log(module_name: str):
    log_path = _module_runner.get_log_path(module_name)
    if not log_path or not Path(log_path).exists():
        raise HTTPException(status_code=404, detail="No log file.")
    return FileResponse(log_path, media_type="application/json")


# ═══════════════════════════════════════════════════════════════
# Probe sets
# ═══════════════════════════════════════════════════════════════

_probe_apply_state = {"active": False, "error": None, "progress": None, "result": None}

@app.post("/api/probe_set/apply")
async def probe_apply(request: Request):
    form = await request.form()
    file = form.get("file")
    if file is None:
        return {"ok": False, "error": "No file uploaded."}
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "error": "No model loaded."}

    # Save the CSV to project root
    _project_root = _PACKAGE_DIR.parent
    filename = file.filename
    dest = _project_root / filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # Start background embedding
    _probe_apply_state["active"] = True
    _probe_apply_state["error"] = None
    _probe_apply_state["progress"] = "Starting probe embedding..."
    _probe_apply_state["result"] = None

    def _embed_worker():
        try:
            from tagm.engine.modules.domain_surface import (
                embed_and_cache_probes, _load_probes, _detect_level_cols, _parse_meta)
            from tagm.engine import config as engine_config
            import json as _json

            pipeline = state.pipeline
            model = pipeline.instruct_model
            tokenizer = pipeline.tokenizer
            adapter = pipeline.adapter
            model_id = pipeline.instruct_model_id
            project_root = str(_project_root)

            # Determine depths
            csv_path = str(dest)
            meta = _parse_meta(csv_path)

            if "layer_low" in meta and "layer_high" in meta:
                subj_frac = max(0.0, min(1.0, float(meta["layer_low"])))
                esc_frac = max(0.0, min(1.0, float(meta["layer_high"])))
            else:
                subj_frac = max(0, min(1, engine_config.get("domain_embedding_layer_frac") or 0.50))
                esc_frac = max(0, min(1, engine_config.get("domain_escalation_layer_frac") or 0.75))

            depths = sorted(set([subj_frac, esc_frac]))
            n_layers = adapter.n_layers(model)

            for frac in depths:
                tag = f"L{int(frac * 100)}"
                _probe_apply_state["progress"] = f"Embedding at {tag}..."

                use_proj = engine_config.get("probe_projection_space")
                delta = None
                if use_proj:
                    target_layer = max(0, min(n_layers - 1, int(frac * n_layers)))
                    delta = pipeline.delta_store.o_delta_or_none(target_layer)

                embed_and_cache_probes(
                    model, tokenizer, adapter,
                    project_root, filename, model_id,
                    layer_frac=frac,
                    delta_matrix=delta if use_proj else None)

            # Activate
            config_path = _project_root / "probe_config.json"
            _json.dump({"active": [filename]}, open(config_path, "w"), indent=2)

            probes = _load_probes(csv_path)
            level_cols, level_names = _detect_level_cols(csv_path)
            subjects = sorted(set(p["subject"] for p in probes))

            _probe_apply_state["result"] = {
                "filename": filename,
                "n_probes": len(probes),
                "n_subjects": len(subjects),
                "n_levels": len(level_cols),
                "layer_L50": int(depths[0] * 100) if depths else 50,
                "layer_L75": int(depths[-1] * 100) if len(depths) > 1 else int(depths[0] * 100),
            }
            state.progress("done", f"Probe set applied: {filename} ({len(probes)} probes)")

        except Exception as e:
            logger.exception("Probe apply failed")
            _probe_apply_state["error"] = str(e)
        finally:
            _probe_apply_state["active"] = False

    import threading
    threading.Thread(target=_embed_worker, daemon=True).start()
    return {"ok": True}

@app.get("/api/probe_set/apply_status")
async def probe_apply_status():
    return {"ok": True, **_probe_apply_state}

@app.get("/api/probe_set/status")
async def probe_status():
    _project_root = _PACKAGE_DIR.parent
    config_path = _project_root / "probe_config.json"
    active = []
    if config_path.exists():
        try:
            active = json.loads(config_path.read_text()).get("active", [])
        except Exception:
            pass
    return {"ok": True, "active": active, "sets": []}

@app.post("/api/probe_set/clear_caches")
async def probe_clear_caches():
    _project_root = _PACKAGE_DIR.parent
    cache_dir = _project_root / "probe_cache"
    cleared = 0
    if cache_dir.exists():
        import shutil
        for f in cache_dir.iterdir():
            f.unlink()
            cleared += 1
    return {"ok": True, "message": f"Cleared {cleared} cache files."}


# ═══════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════

_export_ready = False
_export_path: Optional[Path] = None

@app.post("/api/export")
async def export_session(request: Request):
    global _export_ready, _export_path
    from tagm.service.export import export_session as _export

    if not state.session.results:
        return {"ok": False, "error": "No data to export."}

    def _do_export():
        global _export_ready, _export_path
        state.progress("exporting", "Preparing export...")
        p = _cache.layout.sessions / f"session_{state.session.session_id}.json.gz"
        _export(state.session, p)
        _export_path = p
        _export_ready = True
        state.progress("done", f"Export ready: {p.name}")

    _export_ready = False
    await run_in_threadpool(_do_export)
    return {"ok": True, "ready": True}

@app.get("/api/export/download")
async def export_download():
    if not _export_ready or _export_path is None or not _export_path.exists():
        raise HTTPException(status_code=404, detail="No export available.")
    return FileResponse(str(_export_path), media_type="application/gzip",
                        filename=_export_path.name)


# ═══════════════════════════════════════════════════════════════
# Plots (server-side matplotlib)
# ═══════════════════════════════════════════════════════════════

@app.get("/api/plots/{plot_key}")
async def get_plot(plot_key: str):
    """Batch-level plot. Renders from first result that has the data."""
    from tagm.service.plots import render_plot
    if not state.session.results:
        raise HTTPException(status_code=404, detail="No data in session.")
    # For batch-level plots, use the first result as representative
    # (true batch aggregation plots are a future enhancement)
    try:
        img = await run_in_threadpool(render_plot, plot_key, state.session.results[0])
        if img is None:
            raise HTTPException(status_code=404, detail=f"Unknown plot '{plot_key}'")
        return StreamingResponse(_io.BytesIO(img), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plots/individual/{index}/{plot_key}")
async def get_individual_plot(index: int, plot_key: str):
    from tagm.service.plots import render_plot
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

@app.post("/api/chat")
async def chat(request: Request):
    from tagm.service.chat import generate_chat_response
    if state.pipeline is None or not state.pipeline.loaded:
        raise HTTPException(status_code=409, detail="No model loaded.")
    body = await request.json()
    prompt = body.get("prompt") or body.get("message", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    use_base = (state.inference_class == "base")
    model = state.pipeline.base_model if use_base else state.pipeline.instruct_model
    if model is None:
        raise HTTPException(status_code=409, detail=f"{state.inference_class} model not loaded.")
    response = await run_in_threadpool(
        generate_chat_response, model, state.pipeline.tokenizer, prompt,
        temperature=engine_config.get("chat_temperature"),
        top_p=engine_config.get("chat_top_p"),
        max_tokens=engine_config.get("chat_max_tokens"),
    )
    return {"ok": True, "response": response}
