"""Engine configuration: runtime parameters for all extraction functions.

Central registry — extraction functions read from here instead of using
hardcoded values. The frontend surfaces these in the Advanced Parameters
panel and updates them via /api/engine_config.

Every value has a default that reproduces TASM's original behavior.
"""

DEFAULTS = {
    # ── Signal Layer Selection ──
    # Fraction of layers at each end to exclude. Middle third is the default
    # signal region (e.g. for 24-layer Qwen 2.5-0.5B: layers 8–15).
    "signal_layer_fraction": 0.333,

    # ── SFD Configuration ──
    "sfd_use_signal_layers": False,
    "sfd_layer_start": 9,
    "sfd_layer_end": 16,  # exclusive
    "sfd_svd_k": 16,
    "sfd_svd_seed": 42,

    # ── Serialization ──
    "serialization_precision": 8,

    # ── Rank Displacement ──
    "rd_min_shared": 2,

    # ── ASM Attribution ──
    "boundary_fraction": 0.1,
    "response_topk": 10,
    "proof1_threshold": 1e-4,

    # ── Weight Delta SVD ──
    "delta_svd_k": 64,

    # ── Bootstrap Statistics ──
    "n_bootstrap": 5000,
    "ci_level": 0.95,
    "threshold_steps": 500,
    "min_valid_separability": 5,
    "min_samples_d": 2,

    # ── LTP ──
    "ltp_overfetch_first": 1,
    "ltp_overfetch_second": 5,

    # ── Comparative Plots ──
    "disc_sublayers_top_n": 15,

    # ── Domain Embedding ──
    "domain_embedding_layer_frac": 0.50,
    "domain_escalation_layer_frac": 0.75,
    "include_first_token": False,
    "export_domain_embeddings": False,
    "probe_projection_space": False,
    "attention_weighted_pool": False,
    "persist_probe_caches": True,

    # ── Chat Generation ──
    "chat_temperature": 0.7,
    "chat_top_p": 0.9,
    "chat_max_tokens": 256,
}

_config = dict(DEFAULTS)


def get(key):
    """Read a config value."""
    return _config.get(key, DEFAULTS.get(key))


def update(overrides: dict):
    """Update config from a dict (e.g. from frontend)."""
    for k, v in overrides.items():
        if k in DEFAULTS:
            default_type = type(DEFAULTS[k])
            try:
                _config[k] = default_type(v)
            except (TypeError, ValueError):
                _config[k] = v


def reset():
    """Reset all values to defaults."""
    _config.clear()
    _config.update(DEFAULTS)


def as_dict():
    """Return current config as a plain dict."""
    return dict(_config)
