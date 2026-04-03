"""
TASM Engine Configuration

Central registry of all measurement-affecting parameters.
Engine modules read from this instead of using hardcoded values.
The frontend surfaces these in the Advanced Parameters panel.
app.py updates them when the user changes settings.

Every value here has a default that reproduces the original
hardcoded behavior exactly.
"""

DEFAULTS = {
    # ── Signal Layer Selection ──
    # Middle third of model depth. For 24-layer Qwen 2.5-0.5B: layers 8-15.
    # Controls which layers ASM aggregates stress scores from.
    "signal_layer_fraction": 0.333,  # each third = n_layers * fraction

    # ── SFD Configuration ──
    # SFD uses its own layer range, historically hardcoded to 9-15.
    # Setting this to True makes SFD use the same signal_layers as ASM.
    "sfd_use_signal_layers": False,
    # Fallback: explicit layer range when sfd_use_signal_layers is False.
    "sfd_layer_start": 9,
    "sfd_layer_end": 16,  # exclusive
    # Number of right singular vectors retained for per-token projection.
    "sfd_svd_k": 16,
    # Random seed for SVD computation (torch.svd_lowrank uses randomized
    # algorithm).  Fixed seed ensures deterministic SFD across sessions.
    "sfd_svd_seed": 42,

    # ── Serialization ──
    # Decimal places retained when rounding measurement values for JSON
    # transport.  Lower values reduce file size; higher values preserve
    # precision for downstream analysis.  Applies to all instrument
    # summary scalars and per-position arrays in SFD and RD.
    "serialization_precision": 8,

    # ── Rank Displacement ──
    # Minimum shared candidates between base and instruct to compute
    # Kendall tau. Below this, tau defaults to 0.0.
    "rd_min_shared": 2,

    # ── ASM Attribution ──
    # Boundary fraction: what fraction of tokens at each end counts as
    # "boundary" for the interior/boundary split (top2_share, middle_share).
    # 0.1 = 10% at each end, so 80% is "interior."
    "boundary_fraction": 0.1,
    # Number of next-token predictions captured from each model.
    "response_topk": 10,
    # Decomposition exactness threshold for Proof-1 checks.
    "proof1_threshold": 1e-4,

    # ── Weight Delta SVD ──
    # Max singular values computed for the spectral summary at model load.
    "delta_svd_k": 64,

    # ── Bootstrap Statistics ──
    # Number of bootstrap resamples for confidence intervals.
    "n_bootstrap": 5000,
    # Confidence interval width.
    "ci_level": 0.95,
    # Number of candidate thresholds tested for optimal classification.
    "threshold_steps": 500,
    # Minimum non-null values per metric to include in separability analysis.
    "min_valid_separability": 5,
    # Minimum samples per group to compute Cohen's d.
    "min_samples_d": 2,

    # ── LTP Over-fetch ──
    # Extra candidates fetched to ensure k non-chosen alternatives.
    # First pass fetches k + ltp_overfetch_first. If that's not enough,
    # widens to k + ltp_overfetch_second.
    "ltp_overfetch_first": 1,
    "ltp_overfetch_second": 5,

    # ── Comparative Plots ──
    # Number of sublayers shown in discriminative sublayers plot.
    "disc_sublayers_top_n": 15,

    # ── Modules ──
    # Pre-compute module caches at model load time (e.g. domain surface
    # probe embeddings).  Disable to speed up model loading if you don't
    # need modules that require pre-computation.
    "precompute_module_caches": False,

    # ── Domain Embedding ──
    # Hidden-state capture layer for domain embeddings, as a fraction of
    # model depth (0.0–1.0).  0.25=syntax, 0.50=domain+discourse
    # (recommended), 0.75=task-specific.  Changing this requires re-analysis
    # of all prompts and regeneration of probe caches.
    "domain_embedding_layer_frac": 0.50,
    # Separate depth for escalation-level probe matching.  When different
    # from domain_embedding_layer_frac, probes are embedded at both depths:
    # the first for subject (angular) assignment, the second for escalation
    # level (radial) assignment.  When equal, one pass is used for both.
    "domain_escalation_layer_frac": 0.75,
}

# Current runtime values — initialized from defaults.
# Updated by app.py when user changes settings.
_config = dict(DEFAULTS)


def get(key):
    """Read a config value."""
    return _config.get(key, DEFAULTS.get(key))


def update(overrides: dict):
    """Update config from a dict (e.g. from frontend)."""
    for k, v in overrides.items():
        if k in DEFAULTS:
            # Coerce to same type as default
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
