"""Analysis-module endpoints, template store, and the liveness probe.

Route ordering note: the literal ``token_pair_coupling`` paths below all
use suffixes (`cache_status`, `reset_cache`, `export_cache`) that no
``/api/modules/{module_name}/...`` route claims, so they cannot be
shadowed by the parameterised routes regardless of registration order.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.core.cache import safe_filename
from src.engine.app_core import state

from src.api._state import PACKAGE_DIR, PROJECT_ROOT, module_runner

router = APIRouter(tags=["modules"])


@router.get("/api/modules")
async def list_modules():
    return {"ok": True, "modules": module_runner.list_modules()}


@router.post("/api/modules/upload_template")
async def upload_template(file: UploadFile = File(...)):
    templates_dir = PACKAGE_DIR / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    # Client-supplied filenames are untrusted: sanitize so a crafted name
    # ("../../...") cannot write outside templates_dir.
    filename = safe_filename(file.filename or "template.csv")
    dest = templates_dir / filename
    with open(dest, "wb") as f:
        f.write(content)
    # Return a path the module machinery can resolve from project root —
    # the bare filename resolved against root pointed at nothing.
    rel = os.path.relpath(dest, PROJECT_ROOT)
    return {"ok": True, "filename": rel}


@router.get("/api/health")
async def health():
    """Liveness probe. The port only opens after Python finishes
    importing this module (torch, transformers, sklearn — a 10-30s
    wall on cold starts), so anything that answers here is fully up.
    start.sh polls this to print an unambiguous READY banner."""
    return {"ok": True}


@router.get("/api/templates")
async def api_list_templates():
    from src.probes.io import list_templates
    root = str(PROJECT_ROOT)
    return {"ok": True, "templates": list_templates(root)}


@router.get("/api/templates/{name}")
async def api_get_template(name: str):
    from src.probes.io import load_template_raw, parse_template_echo
    root = str(PROJECT_ROOT)
    try:
        csv_text, axes, path = load_template_raw(root, os.path.basename(name))
    except FileNotFoundError:
        return {"ok": False, "error": f"Template not found: {name}"}
    try:
        parsed = parse_template_echo(root, os.path.basename(name))
    except Exception as e:
        parsed = {"error": str(e)}
    return {"ok": True, "name": os.path.basename(path), "csv": csv_text,
            "axes": axes, "parsed": parsed,
            "path": os.path.relpath(path, root)}


@router.post("/api/templates/save")
async def api_save_template(request: Request):
    from src.probes.io import save_template, parse_template_echo
    root = str(PROJECT_ROOT)
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON body."}
    name = body.get("name") or ""
    csv_text = body.get("csv") or ""
    axes = body.get("axes")
    if not csv_text.strip():
        return {"ok": False, "error": "Empty template."}
    try:
        rel = save_template(root, name, csv_text, axes)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    # The guarantee: immediately re-read through the production parsers
    # and echo back what the machinery will see.
    try:
        parsed = parse_template_echo(root, os.path.basename(rel))
    except Exception as e:
        return {"ok": False, "error": f"Saved, but the parser rejects it: {e}",
                "path": rel}
    return {"ok": True, "path": rel, "parsed": parsed}


@router.post("/api/modules/{module_name}/run")
async def run_module(module_name: str, request: Request):
    body = {}
    ct = request.headers.get("content-type", "")
    if ct.startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = {}
    params = body.get("params") or {}
    result = module_runner.run_module(
        name=module_name,
        session_results=state.session.results,
        params=params,
        session_dir=state.session.session_dir if hasattr(state.session, 'session_dir') else None,
    )
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Module failed to start.")}
    return result


@router.get("/api/modules/{module_name}/status")
async def module_status(module_name: str):
    return module_runner.get_status(module_name)


@router.get("/api/modules/{module_name}/results")
async def module_results(module_name: str):
    results = module_runner.get_results(module_name)
    if results is None:
        raise HTTPException(status_code=404, detail=f"No results for '{module_name}'.")
    return {"ok": True, "results": results}


@router.post("/api/modules/{module_name}/reset")
async def reset_module(module_name: str):
    return module_runner.reset_module(module_name)


@router.post("/api/modules/{module_name}/cancel")
async def cancel_module(module_name: str):
    """Request cancellation of a running module.

    Cooperative and therefore not instantaneous: the module stops at its next
    progress report. Returns {"ok": True, "cancelling": True} to say the
    request was accepted, not that the module has already stopped — the
    module_status SSE event reports the actual transition to "cancelled".
    """
    return module_runner.cancel_module(module_name)


@router.get("/api/modules/{module_name}/download_log")
async def download_module_log(module_name: str):
    log_path = module_runner.get_log_path(module_name)
    if not log_path or not Path(log_path).exists():
        raise HTTPException(status_code=404, detail="No log file.")
    return FileResponse(log_path, media_type="application/json")


# ─── Token Pair Coupling — cache management ─────────────────────

@router.get("/api/modules/token_pair_coupling/cache_status")
async def token_pair_cache_status():
    mod = module_runner.get_module("token_pair_coupling")
    if mod is None:
        return {"ok": False, "error": "Module not found."}
    return {"ok": True, **mod._get_cache_summary()}


@router.post("/api/modules/token_pair_coupling/reset_cache")
async def token_pair_reset_cache():
    from src.engine.modules.token_pair_coupling import TokenPairCoupling
    result = TokenPairCoupling.reset_cache()
    # Clear the in-memory cache reference on the live instance
    mod = module_runner.get_module("token_pair_coupling")
    if mod is not None:
        mod._cache = None
    return {"ok": True, **result}


@router.get("/api/modules/token_pair_coupling/export_cache")
async def token_pair_export_cache():
    cache_path = Path.home() / ".tagm" / "token_pair_cache.json"
    if not cache_path.exists():
        raise HTTPException(status_code=404, detail="No cache file.")
    return FileResponse(
        str(cache_path), media_type="application/json",
        filename="token_pair_cache.json")
