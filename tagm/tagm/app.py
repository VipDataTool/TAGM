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
             "correction_heatmap_viz", "correction_backscatter_viz",
             "probe_diagnostic_viz"):
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
_pg_embed_state = {"active": False, "error": None, "progress": None, "result": None}

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
            from tagm.probes.io import embed_and_activate_probe_set

            def _progress(msg):
                _probe_apply_state["progress"] = msg

            result = embed_and_activate_probe_set(
                state.pipeline, str(_project_root), filename,
                progress=_progress)

            if not result.get("applied"):
                _probe_apply_state["error"] = result.get(
                    "error", "Probe apply failed")
                return

            _probe_apply_state["result"] = {
                "filename": result["filename"],
                "n_probes": result["n_probes"],
                "n_subjects": result["n_subjects"],
                "n_levels": result["n_levels"],
                "layer_L50": result["depths"][0] if result["depths"] else 50,
                "layer_L75": (result["depths"][-1]
                              if len(result["depths"]) > 1
                              else result["depths"][0] if result["depths"] else 50),
            }
            state.progress(
                "done",
                f"Probe set applied: {result['filename']} "
                f"({result['n_probes']} probes)")

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

    if not active:
        return {"ok": True, "active": None}

    filename = active[0]
    csv_path = _project_root / filename
    n_probes = 0
    n_subjects = 0
    cached = False

    if csv_path.exists():
        try:
            from tagm.probes.io import load_probes
            probes = load_probes(str(csv_path))
            n_probes = len(probes)
            n_subjects = len(set(p["subject"] for p in probes))
        except Exception:
            pass

    cache_dir = _project_root / "probe_cache"
    if cache_dir.exists():
        stem = csv_path.stem
        cached = any(f.name.startswith(stem) and f.suffix == ".json"
                     for f in cache_dir.iterdir())

    return {
        "ok": True, "active": True, "filename": filename,
        "n_probes": n_probes, "n_subjects": n_subjects, "cached": cached,
    }

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
# Probe Generator: explicit embed action
# ═══════════════════════════════════════════════════════════════
#
# Runs against a probe CSV the user has already generated (or any CSV
# in the project root). Decoupled from generation; user inspects via
# Probe Diagnostic popout, then triggers this when ready to embed.
# Background-thread + polling pattern, identical to /api/probe_set/apply.

@app.post("/api/modules/probe_generator/embed_active")
async def pg_embed_active(request: Request):
    """Embed and activate a probe CSV that already exists in project root."""
    body = await request.json()
    filename = (body.get("filename") or "").strip()
    if not filename:
        return {"ok": False, "error": "No filename provided."}
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "error": "No model loaded."}

    _project_root = _PACKAGE_DIR.parent
    csv_path = _project_root / filename
    if not csv_path.exists():
        return {"ok": False, "error": f"Probe file not found: {filename}"}

    if _pg_embed_state["active"]:
        return {"ok": False, "error": "Embed already in progress."}

    _pg_embed_state["active"] = True
    _pg_embed_state["error"] = None
    _pg_embed_state["progress"] = "Starting probe embedding..."
    _pg_embed_state["result"] = None

    def _embed_worker():
        try:
            from tagm.probes.io import embed_and_activate_probe_set

            def _progress(msg):
                _pg_embed_state["progress"] = msg

            result = embed_and_activate_probe_set(
                state.pipeline, str(_project_root), filename,
                progress=_progress)

            if not result.get("applied"):
                _pg_embed_state["error"] = result.get("error", "Embed failed")
                return

            _pg_embed_state["result"] = result
            state.progress(
                "done",
                f"Probe set applied: {result['filename']} "
                f"({result['n_probes']} probes)")

        except Exception as e:
            logger.exception("PG embed failed")
            _pg_embed_state["error"] = str(e)
        finally:
            _pg_embed_state["active"] = False

    import threading
    threading.Thread(target=_embed_worker, daemon=True).start()
    return {"ok": True}


@app.get("/api/modules/probe_generator/embed_active_status")
async def pg_embed_active_status():
    return {"ok": True, **_pg_embed_state}


# ═══════════════════════════════════════════════════════════════
# Probe Diagnostic
# ═══════════════════════════════════════════════════════════════
#
# Reads from disk (active probe set or named file). Independent of
# any module's last-run output. Returns lattice properties: cell
# coverage, sample terms per cell, cross-class/cross-level
# collisions. Embedding-tier metrics added when probe cache exists
# for the active model.

@app.get("/api/probe_diagnostic")
async def probe_diagnostic(file: Optional[str] = None):
    """Compute lattice properties of a probe set on disk.

    Query params:
        file: optional CSV filename. If omitted, uses the active
              probe set from probe_config.json.
    """
    from tagm.probes.io import (
        get_active_probe, load_probes, detect_level_cols, parse_meta,
        probe_cache_path, load_probe_cache)
    from collections import Counter, defaultdict

    _project_root = str(_PACKAGE_DIR.parent)

    filename = file or get_active_probe(_project_root)
    if not filename:
        return {"ok": False, "error": "No active probe set."}

    csv_path = os.path.join(_project_root, filename)
    if not os.path.exists(csv_path):
        return {"ok": False, "error": f"Probe file not found: {filename}"}

    probes = load_probes(csv_path)
    if not probes:
        return {"ok": False, "error": "No probes loaded from CSV."}

    level_cols, level_names = detect_level_cols(csv_path)
    meta = parse_meta(csv_path)

    subjects = sorted(set(p["subject"] for p in probes))
    n_levels = len(level_names) if level_names else max(
        (p["level"] for p in probes), default=-1) + 1

    # ── Cell coverage: (subject, level) → list of probe texts ──
    cells = defaultdict(list)
    for p in probes:
        cells[(p["subject"], p["level"])].append(p["text"])

    cell_grid = []
    for s in subjects:
        row = []
        for l in range(n_levels):
            terms = cells.get((s, l), [])
            row.append({"count": len(terms), "sample": terms[:8]})
        cell_grid.append(row)

    counts_flat = [c["count"] for row in cell_grid for c in row]
    n_populated = sum(1 for c in counts_flat if c > 0)
    n_empty = sum(1 for c in counts_flat if c == 0)

    # ── Cross-class collisions: term appears in multiple subjects ──
    term_subjects = defaultdict(set)
    term_levels_per_subject = defaultdict(lambda: defaultdict(set))
    for p in probes:
        term_subjects[p["text"]].add(p["subject"])
        term_levels_per_subject[p["subject"]][p["text"]].add(p["level"])

    cross_class = []
    for term, subjs in term_subjects.items():
        if len(subjs) > 1:
            cross_class.append({
                "term": term,
                "subjects": sorted(subjs),
            })
    cross_class.sort(key=lambda r: (-len(r["subjects"]), r["term"]))

    # ── Cross-level collisions: term appears in multiple levels of same subject ──
    cross_level = []
    for s, term_lvls in term_levels_per_subject.items():
        for term, lvls in term_lvls.items():
            if len(lvls) > 1:
                cross_level.append({
                    "term": term,
                    "subject": s,
                    "levels": sorted(lvls),
                    "level_names": [level_names[l] for l in sorted(lvls)
                                    if l < len(level_names)],
                })
    cross_level.sort(key=lambda r: (-len(r["levels"]), r["subject"], r["term"]))

    # ── Embedding tier (best-effort): load cache for active model if present ──
    embedding_tier = None
    if state.pipeline is not None and state.pipeline.loaded:
        model_id = state.pipeline.instruct_model_id
        # Look for a cache at the subject-layer depth from CSV meta or default 0.50
        if "layer_low" in meta:
            try:
                frac = max(0.0, min(1.0, float(meta["layer_low"])))
            except Exception:
                frac = 0.50
        else:
            try:
                from tagm.engine import config as engine_config
                frac = max(0.0, min(1.0, float(engine_config.get(
                    "domain_embedding_layer_frac") or 0.50)))
            except Exception:
                frac = 0.50

        cache_path = probe_cache_path(_project_root, filename, model_id,
                                       frac, projected=False)
        cache = load_probe_cache(cache_path)
        if cache and cache.get("embeddings"):
            import numpy as np
            embs = np.array(cache["embeddings"], dtype=np.float32)
            # Index alignment: cache embeddings parallel the load_probes() order
            if len(embs) == len(probes):
                # Group embeddings by cell
                cell_embs = defaultdict(list)
                for i, p in enumerate(probes):
                    cell_embs[(p["subject"], p["level"])].append(embs[i])

                # Intra-cell cosine spread: 1 - mean pairwise cosine similarity
                # within each cell
                intra_grid = []
                centroids = {}
                for s in subjects:
                    row = []
                    for l in range(n_levels):
                        vecs = cell_embs.get((s, l), [])
                        if len(vecs) >= 2:
                            M = np.stack(vecs)
                            sims = M @ M.T
                            n = sims.shape[0]
                            mask = ~np.eye(n, dtype=bool)
                            mean_sim = float(sims[mask].mean())
                            spread = 1.0 - mean_sim
                            cent = M.mean(axis=0)
                            cent_norm = np.linalg.norm(cent)
                            if cent_norm > 1e-12:
                                centroids[(s, l)] = cent / cent_norm
                            row.append(round(spread, 4))
                        elif len(vecs) == 1:
                            centroids[(s, l)] = vecs[0]
                            row.append(None)
                        else:
                            row.append(None)
                    intra_grid.append(row)

                # Inter-cell separation: mean cosine distance between centroids
                cell_keys = list(centroids.keys())
                if len(cell_keys) >= 2:
                    C = np.stack([centroids[k] for k in cell_keys])
                    cs = C @ C.T
                    n = cs.shape[0]
                    mask = ~np.eye(n, dtype=bool)
                    inter_mean = 1.0 - float(cs[mask].mean())
                    inter_min = 1.0 - float(cs[mask].max())  # tightest pair
                else:
                    inter_mean = None
                    inter_min = None

                embedding_tier = {
                    "model_id": model_id,
                    "layer_frac": frac,
                    "intra_cell_spread": intra_grid,
                    "inter_cell_mean_distance": (
                        round(inter_mean, 4) if inter_mean is not None else None),
                    "inter_cell_min_distance": (
                        round(inter_min, 4) if inter_min is not None else None),
                    "n_cells_with_centroid": len(cell_keys),
                }

    return {
        "ok": True,
        "filename": filename,
        "n_probes": len(probes),
        "n_subjects": len(subjects),
        "n_levels": n_levels,
        "subjects": subjects,
        "level_names": level_names,
        "cell_grid": cell_grid,
        "summary": {
            "populated_cells": n_populated,
            "empty_cells": n_empty,
            "min_count": min(counts_flat) if counts_flat else 0,
            "max_count": max(counts_flat) if counts_flat else 0,
            "mean_count": (round(sum(counts_flat) / len(counts_flat), 1)
                           if counts_flat else 0),
        },
        "cross_class_collisions": cross_class,
        "cross_level_collisions": cross_level,
        "embedding_tier": embedding_tier,
    }


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

# Batch comparative plot dispatch table
_BATCH_PLOT_DISPATCH = {
    "exp_trajectory_overlay": "plot_trajectory_overlay",
    "exp_difference_from_benign": "plot_difference_from_benign",
    "exp_metric_scatters": "plot_metric_scatters",
    "exp_behavioral_comparison": "plot_behavioral_comparison",
    "exp_ltp_category_comparison": "plot_ltp_category_comparison",
    "exp_ltp_m_vs_stress": "plot_ltp_m_vs_stress",
    "exp_ltp_profile_shapes": "plot_ltp_profile_shape_distribution",
    "exp_sfd_category_comparison": "plot_sfd_category_comparison",
    "exp_sfd_vs_asm": "plot_sfd_vs_asm",
    "exp_rank_displacement": "plot_rank_displacement_by_category",
    "key_scatters": "plot_key_scatters",
    "discriminative_sublayers": "plot_discriminative_sublayers",
    "proof1_summary": "plot_proof1_summary",
}

def _render_batch_plot(plot_key: str, results: list) -> bytes | None:
    """Render a batch comparative plot. Returns PNG bytes or None."""
    import base64

    # Aggregate-based plots (need SimpleNamespace for statistics extractors)
    if plot_key in ("batch_summary", "separability"):
        from types import SimpleNamespace
        from tagm.engine.statistics import aggregate_batch
        from tagm.engine.visualizations import plot_batch_summary, plot_separability
        ns_results = [SimpleNamespace(**r) for r in results]
        agg = aggregate_batch(ns_results)
        if plot_key == "batch_summary":
            b64 = plot_batch_summary(agg)
        else:
            b64 = plot_separability(agg)
        return base64.b64decode(b64) if b64 else None

    # Comparative plots (use raw dicts — functions use r.get() style)
    func_name = _BATCH_PLOT_DISPATCH.get(plot_key)
    if func_name:
        import tagm.engine.comparative as comp
        func = getattr(comp, func_name, None)
        if func:
            b64 = func(results)
            return base64.b64decode(b64) if b64 else None

    return None

@app.get("/api/plots/{plot_key}")
async def get_plot(plot_key: str):
    """Batch or per-prompt plot."""
    if not state.session.results:
        raise HTTPException(status_code=404, detail="No data in session.")
    try:
        # Try batch plot first
        img = await run_in_threadpool(_render_batch_plot, plot_key, state.session.results)
        if img is not None:
            return StreamingResponse(_io.BytesIO(img), media_type="image/png")

        # Fall back to per-prompt plot (first result)
        from tagm.service.plots import render_plot
        img = await run_in_threadpool(render_plot, plot_key, state.session.results[0])
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

def _load_chat_config():
    """Load chat config from module settings or fall back to engine defaults."""
    config_path = _PACKAGE_DIR.parent / "chat_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except Exception:
            pass
    return {
        "temperature": engine_config.get("chat_temperature"),
        "top_p": engine_config.get("chat_top_p"),
        "max_tokens": engine_config.get("chat_max_tokens"),
        "analyze_prompts": True,
        "analyze_responses": False,
        "compute_ltp": True,
        "compute_sfd": True,
    }

@app.get("/api/chat/config")
async def get_chat_config():
    return {"ok": True, "config": _load_chat_config()}

@app.post("/api/chat")
async def chat(request: Request):
    from tagm.service.chat import generate_chat_response
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
    max_tokens = min(int(body.get("max_tokens", cfg["max_tokens"])),
                     engine_config.get("chat_max_tokens"))
    try:
        result = await run_in_threadpool(
            generate_chat_response, state.pipeline, messages,
            max_tokens=max_tokens,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )

        do_analyze = body.get("analyze", cfg.get("analyze_prompts", True))
        do_analyze_resp = body.get("analyze_response", cfg.get("analyze_responses", False))

        # Analyze user prompt into session if requested
        if result.get("ok") and do_analyze:
            prompt_text = messages[-1].get("content", "")
            if prompt_text and state.analyzer:
                try:
                    category = body.get("category", "chat")
                    compute_ltp = body.get("compute_ltp", cfg.get("compute_ltp", True))
                    compute_sfd = body.get("compute_sfd", cfg.get("compute_sfd", True))
                    rd = await run_in_threadpool(
                        _analyze_chat_turn, prompt_text, category,
                        compute_ltp, compute_sfd, "user")
                    result["prompt_analyzed"] = True
                except Exception as e:
                    logger.warning(f"Chat prompt analysis failed: {e}")

        # Analyze model response into session if requested
        if result.get("ok") and do_analyze_resp:
            response_text = result.get("response", "")
            if response_text and state.analyzer:
                try:
                    rd = await run_in_threadpool(
                        _analyze_chat_turn, response_text,
                        "model_response", False, False, "assistant")
                    result["response_analyzed"] = True
                except Exception as e:
                    logger.warning(f"Chat response analysis failed: {e}")

        return result
    except Exception as e:
        logger.exception("Chat endpoint failed")
        return {"ok": False, "error": str(e)}


def _analyze_chat_turn(text, category, compute_ltp, compute_sfd, role):
    """Analyze a chat turn and add it to the session with a role tag."""
    from tagm.engine.result import result_to_dict
    with _analysis_lock:
        result = state.analyzer.analyze_prompt(
            text, category=category,
            compute_kl=True, compute_full_trajectory=False,
            compute_ltp=compute_ltp, compute_sfd=compute_sfd,
        )
        rd = result_to_dict(result)
        rd["role"] = role
        state.session.add_result(rd)
        return rd
