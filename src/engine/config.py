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
    "sfd_svd_k_mode": "fixed",           # "fixed", "ratio", or "energy"
    "sfd_svd_ratio": 56,                  # hidden_dim / ratio = k (ratio mode)
    "sfd_svd_energy_threshold": 0.90,     # cumulative energy fraction (energy mode)
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
    # Consumed by the load worker (Pipeline.load svd_k). Takes effect on
    # the next model load.
    "delta_svd_k": 64,

    # ── Bootstrap Statistics ──
    "n_bootstrap": 5000,
    "ci_level": 0.95,
    "threshold_steps": 500,
    "min_valid_separability": 5,
    "min_samples_d": 2,

    # ── Comparative Plots ──
    "disc_sublayers_top_n": 15,

    # ── Domain Embedding ──
    "domain_embedding_layer_frac": 0.50,
    "domain_escalation_layer_frac": 0.75,
    "include_first_token": True,
    # ── Tokenization ──
    # Forwarded as the `add_special_tokens` kwarg at every tokenizer
    # call site. False = content-only tokenization across model
    # families: position 0 is the first content token whether the
    # underlying tokenizer would otherwise prepend a BOS (Llama 3's
    # `<|begin_of_text|>`) or not (Qwen 2.5). Keeps cross-family runs
    # 1:1 aligned at the position level. True = each family tokenizes
    # in its training-distribution-native form, but token sequences
    # differ across families by one position. Default False because
    # TAGM's primary use case is comparative analysis where empirical
    # comparability dominates.
    "add_special_tokens": False,
    "probe_projection_space": False,

    # ── Chat Generation ──
    "chat_temperature": 0.7,
    "chat_top_p": 0.9,
    "chat_max_tokens": 256,

    # ── High-Efficiency Pipeline ──
    "delta_backend": "memory",       # "memory" or "mmap"
    "hep_active": False,
    "hep_evict_base_cache": False,

    # ── Entropic Cascade Mitigation ──
    # Multi-scale entropy tracker that modulates sampling parameters
    # during chat generation to dampen cascade-prone trajectories.
    # Toggle via ecm_active; parameters tunable from the frontend.
    "ecm_active": False,              # Master toggle — off by default
    "ecm_n_scales": 5,                # EWMA scales (dyadic: 2,4,8,16,32 tokens)
    "ecm_gain": 0.5,                  # Temp reduction per σ of excess signal (v2 units —
                                      # v1 gains in raw nats do not transfer)
    "ecm_floor": 0.1,                 # ARBITRARY — no derivation. Prevents greedy collapse.
    "ecm_deadband": 0.75,              # σ-units of slope ignored as ordinary jitter
    "ecm_agreement": 2,               # Scales that must corroborate before signal fires
    "ecm_warmup": 8,                  # Live-detector cold start: tokens before any
                                      # signal can fire during generation. Default =
                                      # one variance window (1/VAR_LAMBDA). Hardcoded
                                      # pre-v5; explicit now so the live/replay pair
                                      # below is a visible, deliberate choice.
    "ecm_replay_warmup": 4,           # Replay-only cold-start guard (analysis mode).
                                      # Shorter than ecm_warmup: replay never actuates,
                                      # so it trades early-token sigma noise for
                                      # coverage of prompt-length traces. 0 disables.
    # ── ECM v4 (multi-channel) ──
    # ecm_version selects the processor: "v2" = entropy-only (ecm.py),
    # "v4" = pluggable channels (ecm_v4.py). Detector params above
    # (n_scales/deadband/agreement) are shared by both versions.
    "ecm_version": "v2",              # "v2" | "v4"
    "ecm_channels": "entropy,density",  # comma list; v4 only
    "ecm_entropy_weight": 1.0,        # >0 actuates
    "ecm_density_weight": 0.0,        # 0.0 = RECORD-ONLY (measure first,
                                      # couple to the actuator only after
                                      # benign/adversarial traces separate)
    "ecm_fusion": "max",              # "max" | "sum" over weighted signals
    "ecm_no_repeat_ngram": 4,         # no_repeat_ngram_size while ECM active (0 = off);
                                      # backstop against loops seeded during cooled steps
    "ecm_harvest_tokens": 64,         # When ECM is active during analysis, generate a
                                      # short response and analyze it as its own record.
                                      # 0 = off (prompt analysis only, current behavior).
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
