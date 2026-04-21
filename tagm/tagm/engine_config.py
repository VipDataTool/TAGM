"""Engine configuration registry — central registry of all
measurement-affecting parameters.

Ported from TASM's `engine/engine_config.py`. Every key is a knob the
user can change in the UI's Advanced Parameters panel. Modules read
values via `engine_config.get("key")`. Defaults reproduce TASM's original
hardcoded behavior exactly so a fresh TAGM install behaves identically
to the prototype.

The registry is persisted to `~/.tagm/cache/engine_config.json` so user
changes survive process restarts. The file is created on first write
(if absent at startup, defaults are used).

Usage
-----
    from tagm import engine_config

    n = engine_config.get("n_bootstrap")
    engine_config.update({"signal_layer_fraction": 0.25})
    engine_config.reset()
    snapshot = engine_config.as_dict()
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("tagm")


# ── Default values ──────────────────────────────────────────────────
#
# Every key here:
#   - has a default that reproduces TASM's original behavior exactly;
#   - is surfaced in the frontend's `ENGINE_PARAM_META` declaration so
#     the user can change it in the Advanced Parameters panel;
#   - is read by at least one measurement, analysis, or service path.
#
# DO NOT prune unused-looking keys without confirming no read site in
# `tagm/` (and no read site any analysis you may add later) consults
# them. The frontend renders a control for each one and a missing key
# would render an empty input.

DEFAULTS: dict[str, Any] = {
    # ── Signal Layer Selection ──
    # Each "third" of the model = n_layers * fraction. Middle third by
    # default for a 24-layer model: layers 8-15. Controls which layers
    # the stress / amplitude measurements aggregate over by default.
    "signal_layer_fraction": 0.333,

    # ── SFD Configuration ──
    # If True, SFD uses the same signal layers as ASM (driven by
    # signal_layer_fraction). If False, SFD uses the explicit
    # [sfd_layer_start, sfd_layer_end) range.
    "sfd_use_signal_layers": False,
    "sfd_layer_start": 9,
    "sfd_layer_end": 16,            # exclusive
    # Right singular vectors retained for per-token projection.
    "sfd_svd_k": 16,
    # torch.svd_lowrank is randomized; fixed seed = deterministic SFD.
    "sfd_svd_seed": 42,

    # ── Serialization ──
    # Decimal places retained when rounding measurement values for JSON
    # transport. Lower = smaller files; higher = preserved precision.
    "serialization_precision": 8,

    # ── Rank Displacement ──
    # Min shared candidates between base and instruct to compute Kendall
    # tau. Below this, tau = 0.
    "rd_min_shared": 2,

    # ── ASM Attribution ──
    # Fraction of tokens at each end counted as "boundary" for the
    # interior/boundary split (top2_share, middle_share). 0.1 = 10%
    # at each end, so 80% is "interior."
    "boundary_fraction": 0.1,
    # Number of next-token predictions captured from each model.
    "response_topk": 10,
    # Decomposition exactness threshold for Proof-1 checks.
    "proof1_threshold": 1e-4,

    # ── Weight Delta SVD ──
    # Max singular values computed for the spectral summary at model load.
    "delta_svd_k": 64,

    # ── Bootstrap Statistics ──
    "n_bootstrap": 5000,
    "ci_level": 0.95,
    "threshold_steps": 500,
    "min_valid_separability": 5,
    "min_samples_d": 2,

    # ── LTP Over-fetch ──
    # Extra candidates fetched to ensure k non-chosen alternatives.
    # First pass fetches k + ltp_overfetch_first; if that's not enough,
    # widens to k + ltp_overfetch_second.
    "ltp_overfetch_first": 1,
    "ltp_overfetch_second": 5,

    # ── Comparative Plots ──
    "disc_sublayers_top_n": 15,

    # ── Domain / Probe Embeddings ──
    # Hidden-state capture layer for domain embeddings, as a fraction of
    # model depth (0.0–1.0). 0.25 = syntax, 0.50 = domain+discourse
    # (recommended), 0.75 = task-specific. Changing requires re-analysis
    # of all prompts and regeneration of probe caches.
    "domain_embedding_layer_frac": 0.50,
    # Separate depth for escalation-level probe matching. When different
    # from domain_embedding_layer_frac, probes embed at both depths.
    "domain_escalation_layer_frac": 0.75,
    # Include position-0 token in per-token domain embeddings. Position
    # 0 is usually excluded (positional artifacts unrelated to semantics).
    # Enable if your tokenizer does not prepend a BOS/system token.
    "include_first_token": False,
    # Include per-token domain embeddings in session JSON export. Large
    # but enables offline re-analysis with different probe sets/depths.
    "export_domain_embeddings": False,
    # Project embeddings through the o_proj delta before probe matching.
    # Matches probes in the correction field's coordinate system instead
    # of raw hidden-state space.
    "probe_projection_space": False,
    # Use attention-weighted pooling for prompt-level domain embeddings
    # instead of uniform mean-pool. De-emphasizes function words.
    "attention_weighted_pool": False,
    # Persist probe embeddings across sessions. When off, probe caches
    # clear on startup and the user must re-apply a probe set each session.
    "persist_probe_caches": True,

    # ── Chat Generation ──
    # Sampling parameters for /api/chat. Control randomness and length.
    "chat_temperature": 0.7,
    "chat_top_p": 0.9,
    "chat_max_tokens": 512,

    # ── Auto-analyze defaults ──
    # When the user clicks Analyze without an explicit measurement
    # selection (the default UX of the TASM-derived frontend), these
    # flags decide which baseline measurements run automatically.
    # Mirror the `compute_*` checkboxes the TASM Configuration tab had.
    # Only measurements whose flag is True AND whose dependencies are
    # available are enabled; measurements that need the base model or a
    # probe set won't be auto-enabled if those aren't ready.
    "auto_enable_amplitude_trajectory": True,
    "auto_enable_amplitude_derived_metrics": True,
    "auto_enable_stress_score": True,
    "auto_enable_last_position_attribution": True,
    "auto_enable_lateral_tension_profile": False,
    "auto_enable_spectral_field_density": False,
    "auto_enable_rank_displacement": False,
    "auto_enable_probe_projection": False,
    "auto_enable_per_token_embedding": False,
    "auto_enable_backscatter_projection": False,
}


# ── Persistence path ────────────────────────────────────────────────
#
# Lives next to the rest of the on-disk cache so it travels with the
# user's TAGM workspace, not with the source tree. Override with
# TAGM_CACHE_DIR (same env var the Cache class uses) for testing or
# multi-instance setups.

def _config_path() -> Path:
    env = os.environ.get("TAGM_CACHE_DIR")
    root = Path(env) if env else Path.home() / ".tagm" / "cache"
    return root / "engine_config.json"


# ── Mutable runtime state ────────────────────────────────────────────

_lock = threading.Lock()
_config: dict[str, Any] = dict(DEFAULTS)


def _load_from_disk() -> None:
    """Read persisted overrides from disk and merge into _config.

    Defaults are seeded first; on-disk overrides win where present.
    Unknown keys on disk are kept (forward-compat) but not exposed via
    `get()` unless they're added to DEFAULTS.
    """
    path = _config_path()
    if not path.exists():
        return
    try:
        with open(path) as f:
            persisted = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"engine_config: could not read {path}: {e}")
        return
    if not isinstance(persisted, dict):
        return
    for k, v in persisted.items():
        if k in DEFAULTS:
            _config[k] = _coerce(k, v)


def _save_to_disk() -> None:
    """Write current _config to disk atomically (write tmp, rename)."""
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(_config, f, indent=2, sort_keys=True)
        tmp.replace(path)
    except OSError as e:
        logger.warning(f"engine_config: could not write {path}: {e}")


def _coerce(key: str, value: Any) -> Any:
    """Coerce `value` to the type of DEFAULTS[key].

    Booleans and numerics are coerced from strings ("true"/"1") so the
    same code path handles JSON, form data, and direct Python calls.
    Unknown keys pass through unchanged.
    """
    if key not in DEFAULTS:
        return value
    default = DEFAULTS[key]
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return value


# ── Public API ──────────────────────────────────────────────────────

def get(key: str) -> Any:
    """Return the current value for `key`, or its default, or None."""
    with _lock:
        if key in _config:
            return _config[key]
        return DEFAULTS.get(key)


def update(overrides: dict) -> dict:
    """Merge `overrides` into the live config and persist.

    Coerces values to declared types. Returns the post-update snapshot.
    Unknown keys (not in DEFAULTS) are ignored — they would not be read
    by anything in TAGM and silently storing them invites typos.
    """
    if not isinstance(overrides, dict):
        return as_dict()
    with _lock:
        for k, v in overrides.items():
            if k in DEFAULTS:
                _config[k] = _coerce(k, v)
            else:
                logger.warning(
                    f"engine_config.update: ignoring unknown key '{k}'")
        _save_to_disk()
        return dict(_config)


def reset() -> dict:
    """Reset all values to defaults and persist. Returns the snapshot."""
    with _lock:
        _config.clear()
        _config.update(DEFAULTS)
        _save_to_disk()
        return dict(_config)


def as_dict() -> dict:
    """Return a snapshot of the current config (defensive copy)."""
    with _lock:
        return dict(_config)


def defaults() -> dict:
    """Return a snapshot of the defaults (defensive copy)."""
    return dict(DEFAULTS)


# Load persisted overrides at import time so first reads see them.
_load_from_disk()
