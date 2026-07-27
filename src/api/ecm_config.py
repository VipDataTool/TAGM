"""Engine config endpoints, plus ECM config persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request

from src.engine import config as engine_config

from src.api._state import PROJECT_ROOT

logger = logging.getLogger("src")

# No prefix: two of the three routes are the bare /api/engine_config path.
router = APIRouter(tags=["engine_config"])

# Runtime state lives with the rest of the persistent data (~/.tagm, like
# tagm.db and cache/), not in the repo root where it dirties the checkout.
# The legacy repo-root location is migrated on first load below.
_ECM_CONFIG_FILE = Path.home() / ".tagm" / "ecm_config.json"
_LEGACY_ECM_CONFIG_FILE = PROJECT_ROOT / "ecm_config.json"
_ECM_KEYS = {"ecm_active", "ecm_n_scales", "ecm_gain", "ecm_floor",
             "ecm_deadband", "ecm_agreement", "ecm_no_repeat_ngram",
             "ecm_replay_warmup",
             # v4 (multi-channel) — load is key-presence guarded, so
             # pre-v4 config files lacking these simply keep defaults.
             "ecm_version", "ecm_channels", "ecm_entropy_weight",
             "ecm_density_weight", "ecm_fusion", "ecm_harvest_tokens",
             # Response Harvest generation params — persisted alongside ECM
             # config so temp/top-p/seed survive restart (load is key-presence
             # guarded, so older config files lacking these keep defaults).
             "harvest_temperature", "harvest_top_p", "harvest_seed",
             "harvest_seed_ecm"}
_ECM_CONFIG_VERSION = 2


def _load_ecm_config():
    """Load persisted ECM settings from disk into engine_config.

    Version-aware: v1 files (no _ecm_version field) predate the σ-unit
    signal, so their ecm_gain values are in raw nats and would massively
    over-tighten under v2 semantics. Drop gain from v1 files and keep
    the rest; the file is rewritten as v2 on the next save.
    """
    src = _ECM_CONFIG_FILE if _ECM_CONFIG_FILE.exists() else _LEGACY_ECM_CONFIG_FILE
    if src.exists():
        try:
            saved = json.loads(src.read_text())
            version = saved.get("_ecm_version", 1)
            keys = _ECM_KEYS if version >= 2 else (_ECM_KEYS - {"ecm_gain"})
            engine_config.update({k: v for k, v in saved.items() if k in keys})
            if version < 2:
                logger.info("[ECM] v1 config detected — ecm_gain reset to "
                            "v2 default (signal units changed to σ)")
            logger.info(f"[ECM] Loaded config from {src}")
        except Exception as e:
            logger.warning(f"[ECM] Failed to load config: {e}")
            return
        if src is _LEGACY_ECM_CONFIG_FILE:
            # One-time move out of the repo root, models.json-style:
            # rewrite at the new location, rename the original *.migrated.
            try:
                _save_ecm_config()
                src.rename(src.with_suffix(".json.migrated"))
                logger.info(f"[ECM] Migrated {src.name} -> {_ECM_CONFIG_FILE}")
            except Exception as e:
                logger.warning(f"[ECM] Config migration failed: {e}")


def _save_ecm_config():
    """Persist current ECM settings to disk."""
    try:
        vals = {k: engine_config.get(k) for k in _ECM_KEYS}
        vals["_ecm_version"] = _ECM_CONFIG_VERSION
        _ECM_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _ECM_CONFIG_FILE.write_text(json.dumps(vals, indent=1))
    except Exception as e:
        logger.warning(f"[ECM] Failed to save config: {e}")


# Load persisted ECM config at import time
_load_ecm_config()


@router.get("/api/engine_config")
async def get_engine_config():
    return {"ok": True, "config": engine_config.as_dict(),
            "defaults": dict(engine_config.DEFAULTS)}


@router.post("/api/engine_config")
async def set_engine_config(request: Request):
    body = await request.json()
    engine_config.update(body)
    # Persist ECM keys if any were changed
    if _ECM_KEYS & body.keys():
        _save_ecm_config()
    return {"ok": True, "config": engine_config.as_dict()}


@router.post("/api/engine_config/reset")
async def reset_engine_config():
    engine_config.reset()
    if _ECM_CONFIG_FILE.exists():
        _ECM_CONFIG_FILE.unlink()
    return {"ok": True, "config": engine_config.as_dict()}
