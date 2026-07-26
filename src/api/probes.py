"""Probe-set endpoints: apply, status, cache clearing, diagnostics."""
from __future__ import annotations

import logging
import os
import shutil
import threading

from fastapi import APIRouter, Request
from typing import Optional

from src.core.cache import safe_filename
from src.engine.app_core import state
from src.service.events import broker

from src.api._state import PROJECT_ROOT
from src.api._util import resolve_under

logger = logging.getLogger("src")

router = APIRouter(tags=["probes"])

_probe_apply_state = {"active": False, "error": None, "progress": None, "result": None}
_pg_embed_state = {"active": False, "error": None, "progress": None, "result": None}


def run_embed_job(state_dict: dict, event_name: str, filename: str) -> None:
    """Background worker shared by both probe-embedding endpoints.

    ``/api/probe_set/apply`` and
    ``/api/modules/probe_generator/embed_active`` ran byte-identical
    workers apart from their state dict and SSE event name. The published
    result is now the full ``embed_and_activate_probe_set`` record plus
    the derived ``layer_L50`` / ``layer_L75`` fields the Configuration tab
    reads — a superset of what either endpoint published before, so both
    clients keep working.

    Always clears ``state_dict["active"]`` and publishes ``event_name``,
    including on failure; otherwise the UI polls a job that never ends.
    """
    try:
        from src.probes.io import embed_and_activate_probe_set

        def _progress(msg):
            state_dict["progress"] = msg

        result = embed_and_activate_probe_set(
            state.pipeline, str(PROJECT_ROOT), filename, progress=_progress)

        if not result.get("applied"):
            state_dict["error"] = result.get("error", "Probe embed failed")
            return

        depths = result.get("depths") or []
        state_dict["result"] = {
            **result,
            "layer_L50": depths[0] if depths else 50,
            "layer_L75": (depths[-1] if len(depths) > 1
                          else depths[0] if depths else 50),
        }
        state.progress(
            "done",
            f"Probe set applied: {result['filename']} "
            f"({result['n_probes']} probes)")

    except Exception as e:
        logger.exception(f"[{event_name}] probe embed failed")
        state_dict["error"] = str(e)
    finally:
        state_dict["active"] = False
        broker.publish(event_name, {
            "active": False,
            "error": state_dict.get("error"),
            "result": state_dict.get("result"),
        })


@router.post("/api/probe_set/apply")
async def probe_apply(request: Request):
    form = await request.form()
    file = form.get("file")
    if file is None:
        return {"ok": False, "error": "No file uploaded."}
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "error": "No model loaded."}

    # Same in-progress guard as pg_embed: two concurrent applies raced on
    # _probe_apply_state and on the probe cache files they write. Claim the
    # slot *before* the first await — otherwise two requests both see
    # active=False while suspended on file.read().
    if _probe_apply_state["active"]:
        return {"ok": False, "error": "Probe apply already in progress."}
    _probe_apply_state["active"] = True
    _probe_apply_state["error"] = None
    _probe_apply_state["progress"] = "Starting probe embedding..."
    _probe_apply_state["result"] = None

    # Save the CSV to project root. Sanitize the client-supplied name —
    # path traversal otherwise writes anywhere the server can.
    filename = safe_filename(file.filename or "probes.csv")
    dest = PROJECT_ROOT / filename
    try:
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
    except Exception as e:
        _probe_apply_state["active"] = False
        logger.exception("Probe upload failed")
        return {"ok": False, "error": f"Upload failed: {e}"}

    threading.Thread(
        target=run_embed_job,
        args=(_probe_apply_state, "probe_status", filename),
        daemon=True,
    ).start()
    return {"ok": True}


@router.get("/api/probe_set/apply_status")
async def probe_apply_status():
    return {"ok": True, **_probe_apply_state}


@router.get("/api/probe_set/status")
async def probe_status():
    _project_root = PROJECT_ROOT
    from src.probes.io import get_active_probe_set, load_probes

    active = get_active_probe_set(str(_project_root))
    if active is None:
        return {"ok": True, "active": None}

    csv_path = _project_root / active.probe_file
    n_probes = active.n_probes
    n_subjects = 0

    if csv_path.exists():
        try:
            probes = load_probes(str(csv_path))
            # Re-derive from the live CSV — the active-record's count is
            # what was on disk at apply time; if the file has changed we
            # show the current count (and if it differs, the user can see
            # the drift in the status line).
            n_probes = len(probes)
            n_subjects = len(set(p["subject"] for p in probes))
        except Exception:
            # A malformed/renamed CSV silently fell back to the stale
            # counts from the active record with no trace anywhere.
            logger.exception(
                f"Probe status: failed to re-read {csv_path}")

    # Cache presence: check that the exact cache file the active record
    # points at actually exists on disk. This is a tighter check than
    # "any matching stem" — it confirms the binding is fulfilled.
    cached = False
    if not active.is_legacy() and active.depths:
        try:
            cache_path = active.cache_path(
                str(_project_root), active.subject_layer_frac())
            cached = os.path.exists(cache_path)
        except Exception:
            cached = False

    payload = {
        "ok": True,
        "active": True,
        # Core fields (kept for any older client code that still reads them).
        "filename": active.probe_file,
        "n_probes": n_probes,
        "n_subjects": n_subjects,
        "cached": cached,
        # Rich record for the new status line.
        "model_id": active.model_id,
        "depths": list(active.depths),
        "projected": active.projected,
        "applied_at": active.applied_at,
        "legacy": active.is_legacy(),
    }

    # If the active record was applied for a model other than the one
    # currently loaded, surface that as a structured warning the UI can
    # render alongside the green ✓.
    pipe = state.pipeline
    if pipe is not None and getattr(pipe, "loaded", False):
        payload["loaded_model_id"] = pipe.instruct_model_id
        if (active.model_id and
                active.model_id != pipe.instruct_model_id):
            payload["stale_for_loaded_model"] = True
        elif active.is_legacy():
            payload["stale_for_loaded_model"] = True
        else:
            payload["stale_for_loaded_model"] = False
    else:
        payload["loaded_model_id"] = None
        payload["stale_for_loaded_model"] = active.is_legacy()

    return payload


@router.post("/api/probe_set/clear_caches")
async def probe_clear_caches():
    cache_dir = PROJECT_ROOT / "probe_cache"
    cleared = 0
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_dir():
                cleared += sum(1 for p in f.rglob("*") if p.is_file())
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink()
                cleared += 1
    return {"ok": True, "message": f"Cleared {cleared} cache entries.",
            "deleted": cleared}


# ═══════════════════════════════════════════════════════════════
# Probe Generator: explicit embed action
# ═══════════════════════════════════════════════════════════════
#
# Runs against a probe CSV the user has already generated (or any CSV
# in the project root). Decoupled from generation; user inspects via
# Probe Diagnostic popout, then triggers this when ready to embed.
# Background-thread + polling pattern, identical to /api/probe_set/apply.

@router.post("/api/modules/probe_generator/embed_active")
async def pg_embed_active(request: Request):
    """Embed and activate a probe CSV that already exists in project root."""
    body = await request.json()
    filename = (body.get("filename") or "").strip()
    if not filename:
        return {"ok": False, "error": "No filename provided."}
    if state.pipeline is None or not state.pipeline.loaded:
        return {"ok": False, "error": "No model loaded."}

    # `filename` is request-supplied and may legitimately be a relative
    # subpath ("probe_cache/foo.csv"), so it is resolved and confirmed to
    # stay under the project root rather than sanitized flat. Unchecked,
    # "../../etc/passwd" was read (and embedded) by this endpoint.
    csv_path = resolve_under(PROJECT_ROOT, filename)
    if csv_path is None:
        return {"ok": False, "error": f"Invalid probe file path: {filename}"}
    if not csv_path.exists():
        return {"ok": False, "error": f"Probe file not found: {filename}"}

    if _pg_embed_state["active"]:
        return {"ok": False, "error": "Embed already in progress."}

    _pg_embed_state["active"] = True
    _pg_embed_state["error"] = None
    _pg_embed_state["progress"] = "Starting probe embedding..."
    _pg_embed_state["result"] = None

    threading.Thread(
        target=run_embed_job,
        args=(_pg_embed_state, "pg_embed_status", filename),
        daemon=True,
    ).start()
    return {"ok": True}


@router.get("/api/modules/probe_generator/embed_active_status")
async def pg_embed_active_status():
    return {"ok": True, **_pg_embed_state}


# ═══════════════════════════════════════════════════════════════
# Probe Diagnostic
# ═══════════════════════════════════════════════════════════════

@router.get("/api/probe_diagnostic")
async def probe_diagnostic(file: Optional[str] = None):
    """Compute lattice properties of a probe set on disk.

    Query params:
        file: optional CSV filename. If omitted, uses the active
              probe set from probe_config.json.
    """
    from src.probes.diagnostic import compute_diagnostic
    return compute_diagnostic(str(PROJECT_ROOT), file, state.pipeline)
