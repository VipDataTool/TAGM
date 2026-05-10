"""Plot generation.

TASM rendered per-prompt matplotlib plots server-side and served them
as PNGs via /api/plots/individual/{idx}/{key}. main.js embeds these
as <img> elements with onerror fallbacks. The frontend never composes
plots client-side; rendering is the backend's job.

This module mirrors TASM's plot semantics. Each plot key takes a
TASM-shape per-prompt result dict (the same shape /api/results/detail
returns) and produces a PNG bytes blob. matplotlib is imported lazily
so module load is cheap when no plots are requested.

Plot keys handled (matches `_plot_keys_for_result`):

  Always available (graceful empty-plot if data missing):
    - signed_attribution
    - stress_per_token
    - distribution_metrics

  Conditional (only emitted if the underlying data is present):
    - amplitude_trajectory
    - heatmap
    - ltp_profiles
    - ltp_tension_magnitudes
    - ltp_dual_trajectory
    - ltp_summary_stats
    - ltp_profile_heatmap
    - sfd_density
    - rank_displacement
"""
from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger("tagm")

_DPI = 100
_FIGSIZE_WIDE = (10, 4)
_FIGSIZE_SQUARE = (8, 6)
_BG = "#0f0f0f"
_FG = "#e8ecef"
_GRID = "#2d2d2d"
_BLUE = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f85149"
_ORANGE = "#d29922"
_PURPLE = "#cc79a7"


def _setup_matplotlib():
    """Lazy matplotlib setup. Returns the configured plt module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": _BG,
        "axes.facecolor": _BG,
        "axes.edgecolor": _GRID,
        "axes.labelcolor": _FG,
        "axes.titlecolor": _FG,
        "xtick.color": _FG,
        "ytick.color": _FG,
        "grid.color": _GRID,
        "text.color": _FG,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


def _render(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight",
                facecolor=_BG)
    buf.seek(0)
    data = buf.read()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return data


def _empty_plot(message: str) -> bytes:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color=_FG, fontsize=11, transform=ax.transAxes)
    ax.set_axis_off()
    return _render(fig)


# ── Plot key registry ──────────────────────────────────────────────

def render_plot(key: str, result: dict) -> Optional[bytes]:
    """Render the plot identified by `key` against `result` (TASM-shape).

    Returns PNG bytes on success, None if the plot key is unknown.
    Returns a placeholder PNG if the underlying data is missing.
    """
    handler = _PLOT_HANDLERS.get(key)
    if handler is None:
        return None
    try:
        return handler(result)
    except Exception as e:
        logger.exception(f"[plots] {key} failed")
        return _empty_plot(f"Render error: {e}")


# ── Per-prompt plot handlers ───────────────────────────────────────

def _plot_signed_attribution(r: dict) -> bytes:
    plt = _setup_matplotlib()
    signed = r.get("signed_attr") or []
    tokens = r.get("tokens") or []
    if not signed or not tokens:
        return _empty_plot("No signed attribution data")
    n = min(len(signed), len(tokens))
    signed = signed[:n]
    tokens = tokens[:n]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    colors = [_RED if v < 0 else _GREEN for v in signed]
    ax.bar(range(n), signed, color=colors, edgecolor=_GRID, linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color=_FG, linewidth=0.5)
    ax.set_ylabel("Signed attribution to last position")
    ax.set_title("Per-token contribution to last-position correction")
    fig.tight_layout()
    return _render(fig)


def _plot_stress_per_token(r: dict) -> bytes:
    plt = _setup_matplotlib()
    stress = r.get("per_token_stress") or []
    tokens = r.get("tokens") or []
    if not stress or not tokens:
        return _empty_plot("No per-token stress data")
    n = min(len(stress), len(tokens))
    stress = stress[:n]
    tokens = tokens[:n]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.bar(range(n), stress, color=_BLUE, edgecolor=_GRID, linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Stress")
    ax.set_title("Per-token attention-side correction stress")
    fig.tight_layout()
    return _render(fig)


def _plot_distribution_metrics(r: dict) -> bytes:
    plt = _setup_matplotlib()
    fields = [("Top-2 share", r.get("top2_share")),
              ("Middle share", r.get("middle_share")),
              ("Interior CV", r.get("interior_cv")),
              ("Entropy", r.get("entropy"))]
    fields = [(k, v) for k, v in fields if v is not None]
    if not fields:
        return _empty_plot("No distribution metrics")

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    labels, values = zip(*fields)
    ax.barh(labels, values, color=_PURPLE, edgecolor=_GRID, linewidth=0.5)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center", color=_FG, fontsize=9)
    ax.set_xlabel("Value")
    ax.set_title("Boundary / interior distribution metrics")
    fig.tight_layout()
    return _render(fig)


def _plot_amplitude_trajectory(r: dict) -> bytes:
    plt = _setup_matplotlib()
    at = r.get("amplitude_trajectory")
    # amplitude_trajectory is a flat list of raw values;
    # amplitude_normalized is a separate flat list.
    if isinstance(at, dict):
        raw = at.get("raw") or []
        norm = at.get("normalized") or []
        labels = at.get("sublayer_labels") or []
    else:
        raw = at if isinstance(at, list) else []
        norm = r.get("amplitude_normalized") or []
        labels = []
    if not raw and not norm:
        return _empty_plot("No amplitude trajectory data")

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    x = list(range(len(norm or raw)))
    if norm:
        ax.plot(x, norm, color=_BLUE, label="Normalized", linewidth=1.5)
    if raw:
        ax2 = ax.twinx()
        ax2.plot(x, raw, color=_ORANGE, label="Raw", linewidth=1.0,
                  linestyle="--", alpha=0.7)
        ax2.set_ylabel("Raw amplitude", color=_ORANGE)
        ax2.tick_params(axis="y", colors=_ORANGE)
    if labels and len(labels) == len(x):
        # Show every Nth label to avoid clutter
        step = max(1, len(labels) // 16)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Frobenius-normalized amplitude", color=_BLUE)
    ax.set_title("Amplitude trajectory across sublayers")
    if norm:
        ax.legend(loc="upper left")
    fig.tight_layout()
    return _render(fig)


def _plot_heatmap(r: dict) -> bytes:
    plt = _setup_matplotlib()
    hm = r.get("heatmap") or []
    if not hm:
        return _empty_plot("No heatmap data")

    import numpy as np
    arr = np.array(hm)
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    im = ax.imshow(arr, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Sublayer")
    ax.set_title("Per-token amplitude heatmap")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return _render(fig)


def _plot_ltp_profiles(r: dict) -> bytes:
    plt = _setup_matplotlib()
    ltp = r.get("ltp") or {}
    profiles = ltp.get("profiles") or []
    base = ltp.get("base_profiles") or []
    if not profiles:
        return _empty_plot("No LTP profile data")

    import numpy as np
    arr = np.array(profiles)
    base_arr = np.array(base) if base else None

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    # Heatmap of profiles (positions × k)
    im = ax.imshow(arr.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Counterfactual rank (k)")
    ax.set_title("Lateral tension profiles (instruct)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return _render(fig)


def _plot_ltp_tension_magnitudes(r: dict) -> bytes:
    plt = _setup_matplotlib()
    ltp = r.get("ltp") or {}
    mags = ltp.get("tension_magnitudes") or []
    tokens = r.get("tokens") or []
    if not mags:
        return _empty_plot("No LTP tension data")
    n = min(len(mags), len(tokens) if tokens else len(mags))
    mags = mags[:n]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(range(n), mags, color=_PURPLE, linewidth=1.5,
             marker="o", markersize=3)
    ax.fill_between(range(n), mags, alpha=0.2, color=_PURPLE)
    if tokens and len(tokens) >= n:
        step = max(1, n // 24)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels(tokens[:n:step], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tension magnitude")
    ax.set_title("Per-token lateral tension magnitude")
    fig.tight_layout()
    return _render(fig)


def _plot_ltp_dual_trajectory(r: dict) -> bytes:
    plt = _setup_matplotlib()
    ltp = r.get("ltp") or {}
    semantic = ltp.get("semantic_trajectory_2d") or []
    tension = ltp.get("tension_trajectory_2d") or []
    if not semantic or not tension:
        return _empty_plot("No dual-trajectory data")

    import numpy as np
    s = np.array(semantic)
    t = np.array(tension)
    if s.ndim != 2 or s.shape[1] < 2 or t.shape[1] < 2:
        return _empty_plot("Trajectory data has wrong shape")

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    ax.plot(s[:, 0], s[:, 1], color=_BLUE, marker="o", markersize=4,
            label="Semantic", linewidth=1.5)
    ax.plot(t[:, 0], t[:, 1], color=_ORANGE, marker="s", markersize=4,
            label="Tension", linewidth=1.5, alpha=0.7)
    # Connect each pair
    for i in range(min(len(s), len(t))):
        ax.plot([s[i, 0], t[i, 0]], [s[i, 1], t[i, 1]],
                 color=_GRID, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Dual trajectory: semantic vs tension")
    ax.legend()
    fig.tight_layout()
    return _render(fig)


def _plot_ltp_summary_stats(r: dict) -> bytes:
    plt = _setup_matplotlib()
    ltp = r.get("ltp") or {}
    fields = [("Mean M", ltp.get("mean_M")),
              ("Mean V", ltp.get("mean_V")),
              ("Mean L", ltp.get("mean_L")),
              ("Max PRC", ltp.get("max_prc")),
              ("N directional", ltp.get("n_directional"))]
    fields = [(k, v) for k, v in fields if v is not None]
    if not fields:
        return _empty_plot("No LTP summary data")

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    labels, values = zip(*fields)
    ax.barh(labels, values, color=_BLUE, edgecolor=_GRID, linewidth=0.5)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.4f}" if isinstance(v, float) else f" {v}",
                va="center", color=_FG, fontsize=9)
    ax.set_title("LTP summary statistics")
    fig.tight_layout()
    return _render(fig)


def _plot_ltp_profile_heatmap(r: dict) -> bytes:
    """Profile heatmap with base-instruct difference."""
    plt = _setup_matplotlib()
    ltp = r.get("ltp") or {}
    profiles = ltp.get("profiles") or []
    base = ltp.get("base_profiles") or []
    if not profiles or not base:
        return _empty_plot("Need both instruct and base profiles")

    import numpy as np
    a = np.array(profiles)
    b = np.array(base)
    n = min(a.shape[0], b.shape[0])
    diff = a[:n] - b[:n]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    vmax = float(np.abs(diff).max()) if diff.size else 1.0
    im = ax.imshow(diff.T, aspect="auto", cmap="RdBu_r", origin="lower",
                    vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Counterfactual rank (k)")
    ax.set_title("Profile difference: instruct − base")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return _render(fig)


def _plot_sfd_density(r: dict) -> bytes:
    plt = _setup_matplotlib()
    sfd = r.get("sfd") or {}
    density = sfd.get("per_token_density") or []
    tokens = r.get("tokens") or []
    if not density:
        return _empty_plot("No SFD density data")
    n = min(len(density), len(tokens) if tokens else len(density))
    density = density[:n]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.fill_between(range(n), density, alpha=0.4, color=_ORANGE)
    ax.plot(range(n), density, color=_ORANGE, linewidth=1.5)
    if tokens and len(tokens) >= n:
        step = max(1, n // 24)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels(tokens[:n:step], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Density (effective rank ratio)")
    ax.set_title("Per-token spectral field density")
    mean = sfd.get("density_mean")
    if mean is not None:
        ax.axhline(mean, color=_FG, linewidth=0.7, linestyle="--",
                    label=f"Mean = {mean:.3f}")
        ax.legend()
    fig.tight_layout()
    return _render(fig)


def _plot_rank_displacement(r: dict) -> bytes:
    plt = _setup_matplotlib()
    rd = r.get("rank_displacement") or {}

    # Engine produces per_position (list of dicts) with total_disp and replacement_ratio
    per_pos = rd.get("per_position") or []
    if per_pos and isinstance(per_pos[0], dict):
        disp = [p.get("total_disp", 0) for p in per_pos]
        repl = [p.get("replacement_ratio", 0) for p in per_pos]
    else:
        # Fallback: try legacy flat arrays
        disp = rd.get("per_token_disp") or []
        repl = rd.get("per_token_replacement") or []

    if not disp:
        return _empty_plot("No rank displacement data")

    tokens = r.get("tokens", [])
    x = range(len(disp))
    labels = tokens[:len(disp)] if tokens else [str(i) for i in x]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.bar(x, disp, color=_BLUE, edgecolor=_GRID, linewidth=0.5)
    ax1.set_ylabel("Total displacement")
    ax1.set_title(f"Per-position rank displacement (τ={rd.get('mean_tau', 0):.3f})")
    if repl:
        ax2.bar(x, repl, color=_RED, edgecolor=_GRID, linewidth=0.5)
        ax2.set_ylabel("Replacement ratio")
    ax2.set_xlabel("Token position")
    if len(labels) <= 20:
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return _render(fig)


_PLOT_HANDLERS = {
    "signed_attribution":         _plot_signed_attribution,
    "stress_per_token":           _plot_stress_per_token,
    "distribution_metrics":       _plot_distribution_metrics,
    "amplitude_trajectory":       _plot_amplitude_trajectory,
    "heatmap":                    _plot_heatmap,
    "ltp_profiles":               _plot_ltp_profiles,
    "ltp_tension_magnitudes":     _plot_ltp_tension_magnitudes,
    "ltp_dual_trajectory":        _plot_ltp_dual_trajectory,
    "ltp_summary_stats":          _plot_ltp_summary_stats,
    "ltp_profile_heatmap":        _plot_ltp_profile_heatmap,
    "sfd_density":                _plot_sfd_density,
    "rank_displacement":          _plot_rank_displacement,
}


def list_plot_keys() -> list[str]:
    """Return all known plot keys."""
    return list(_PLOT_HANDLERS.keys())
