"""High-Efficiency Pipeline (HEP) endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from src.engine import config as engine_config
from src.engine.app_core import state, api_reset_handler
from src.service.events import broker

router = APIRouter(prefix="/api/hep", tags=["hep"])


@router.get("/status")
async def hep_status():
    """Return HEP state, disk/memory usage, and mmap info."""
    from src.core.cache import system_resources

    res = system_resources()
    mmap_dir = Path.home() / ".tagm" / "cache" / "deltas"
    mmap_files = list(mmap_dir.glob("*.tagm")) if mmap_dir.exists() else []
    mmap_size = sum(f.stat().st_size for f in mmap_files)

    return {
        "active": bool(engine_config.get("hep_active")),
        "delta_backend": engine_config.get("delta_backend"),
        "mmap_file": str(mmap_files[0]) if mmap_files else None,
        "mmap_size_bytes": mmap_size,
        "evict_base_cache": bool(engine_config.get("hep_evict_base_cache")),
        **res,
    }


@router.post("/initialize")
async def init_hep(request: Request):
    """Initialize the High-Efficiency Pipeline.

    Clears HF cache, removes old mmap files, resets pipeline,
    configures delta_backend to mmap.
    """
    from src.core.cache import clear_hf_cache, system_resources
    from src.core.db import get_db
    import gc

    # Reset pipeline first (thread-pooled: reset takes MODEL_LOCK)
    await run_in_threadpool(api_reset_handler)
    gc.collect()

    # Clear HF cache to free disk — but keep existing mmap delta files,
    # they're expensive to recompute and valid for cache reuse.
    hf_result = clear_hf_cache()

    # Configure HEP
    engine_config.update({
        "delta_backend": "mmap",
        "hep_active": True,
        "hep_evict_base_cache": True,
    })

    # Persist HEP state to DB so it survives restarts
    get_db().set_config("hep", {
        "active": True,
        "delta_backend": "mmap",
        "evict_base_cache": True,
    })

    res = system_resources()
    total_freed = hf_result["bytes_freed"]

    state.progress("ready", f"HEP initialized: freed {total_freed / 1e9:.1f} GB")
    broker.publish("progress", {
        "stage": "ready",
        "message": f"High-Efficiency Pipeline active. Freed {total_freed / 1e9:.1f} GB.",
    })

    return {
        "ok": True,
        "hf_freed": hf_result,
        "disk_free": res["disk_free"],
        "ram_available": res["ram_available"],
    }


@router.post("/deactivate")
async def deactivate_hep():
    """Deactivate HEP and return to standard memory mode."""
    from src.core.cache import clear_mmap_deltas
    from src.core.db import get_db

    await run_in_threadpool(api_reset_handler)
    clear_mmap_deltas()
    engine_config.update({
        "delta_backend": "memory",
        "hep_active": False,
        "hep_evict_base_cache": False,
    })

    # Persist deactivation
    get_db().set_config("hep", {
        "active": False,
        "delta_backend": "memory",
        "evict_base_cache": False,
    })

    state.progress("ready", "High-Efficiency Pipeline deactivated")
    return {"ok": True}
