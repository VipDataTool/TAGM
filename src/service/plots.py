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
import threading
from typing import Optional

logger = logging.getLogger("src")

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


_MPL_RC = {
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
}

_mpl_ready = False
_mpl_setup_lock = threading.Lock()


def _setup_matplotlib():
    """Configure matplotlib exactly once, process-wide.

    Previously this rewrote the global ``plt.rcParams`` on *every* plot
    call. Plots are dispatched through ``run_in_threadpool``, so
    concurrent renders were mutating shared global state; one-shot
    initialisation removes that race. rcParams are read by ``Figure``
    at construction time, so setting them once at first use is enough.
    """
    global _mpl_ready
    import matplotlib
    if not _mpl_ready:
        with _mpl_setup_lock:
            if not _mpl_ready:
                matplotlib.use("Agg")
                matplotlib.rcParams.update(_MPL_RC)
                _mpl_ready = True
    return matplotlib


def _new_fig(figsize=_FIGSIZE_WIDE):
    """Create a standalone Agg figure.

    Object-oriented API on purpose: ``plt.subplots()`` registers the
    figure in pyplot's global ``Gcf`` manager, so any exception between
    creation and ``plt.close()`` leaked it permanently (a growing set of
    live figures, each holding its rendered buffers). A bare ``Figure``
    is owned by the caller and collected normally on every path —
    success, handler exception, or client disconnect — and it is also
    safe to build from several threadpool workers at once.
    """
    _setup_matplotlib()
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize, dpi=_DPI)
    FigureCanvasAgg(fig)   # attach the Agg canvas so fig.savefig() works
    return fig


def _render(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight",
                facecolor=_BG)
    buf.seek(0)
    return buf.read()


def _empty_plot(message: str) -> bytes:
    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()
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
    signed = r.get("signed_attr") or []
    tokens = r.get("tokens") or []
    if not signed or not tokens:
        return _empty_plot("No signed attribution data")
    n = min(len(signed), len(tokens))
    signed = signed[:n]
    tokens = tokens[:n]

    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()
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
    stress = r.get("per_token_stress") or []
    tokens = r.get("tokens") or []
    if not stress or not tokens:
        return _empty_plot("No per-token stress data")
    n = min(len(stress), len(tokens))
    stress = stress[:n]
    tokens = tokens[:n]

    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()
    ax.bar(range(n), stress, color=_BLUE, edgecolor=_GRID, linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Stress")
    ax.set_title("Per-token attention-side correction stress")
    fig.tight_layout()
    return _render(fig)


def _plot_distribution_metrics(r: dict) -> bytes:
    fields = [("Top-2 share", r.get("top2_share")),
              ("Middle share", r.get("middle_share")),
              ("Interior CV", r.get("interior_cv")),
              ("Entropy", r.get("entropy"))]
    fields = [(k, v) for k, v in fields if v is not None]
    if not fields:
        return _empty_plot("No distribution metrics")

    fig = _new_fig(_FIGSIZE_SQUARE)
    ax = fig.subplots()
    labels, values = zip(*fields)
    ax.barh(labels, values, color=_PURPLE, edgecolor=_GRID, linewidth=0.5)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center", color=_FG, fontsize=9)
    ax.set_xlabel("Value")
    ax.set_title("Boundary / interior distribution metrics")
    fig.tight_layout()
    return _render(fig)


def _plot_amplitude_trajectory(r: dict) -> bytes:
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

    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()

    # The trajectory interleaves the two sublayer types:
    #     index = 2*layer + {0: attn, 1: mlp}
    # Plotting them as ONE series put alternating quantities on a shared axis
    # and produced a period-2 sawtooth that buried the actual depth trend.
    # Split by parity and plot against layer number instead.
    series = norm or raw
    n = len(series)
    interleaved = n >= 4 and n % 2 == 0 and not labels
    handles = []

    if interleaved:
        layers = list(range(n // 2))
        if norm:
            handles += ax.plot(layers, norm[0::2], color=_BLUE, linewidth=1.6,
                               marker="o", markersize=3, label="attn (normalized)")
            handles += ax.plot(layers, norm[1::2], color=_GREEN, linewidth=1.6,
                               marker="s", markersize=3, label="mlp (normalized)")
        if raw:
            ax2 = ax.twinx()
            handles += ax2.plot(layers, raw[0::2], color=_ORANGE, linewidth=1.0,
                                linestyle="--", alpha=0.7, label="attn (raw)")
            handles += ax2.plot(layers, raw[1::2], color=_RED, linewidth=1.0,
                                linestyle=":", alpha=0.7, label="mlp (raw)")
            ax2.set_ylabel("Raw amplitude", color=_ORANGE)
            ax2.tick_params(axis="y", colors=_ORANGE)
        ax.set_xlabel("Layer")
    else:
        x = list(range(n))
        if norm:
            handles += ax.plot(x, norm, color=_BLUE, label="Normalized",
                               linewidth=1.5)
        if raw:
            ax2 = ax.twinx()
            handles += ax2.plot(x, raw, color=_ORANGE, label="Raw",
                                linewidth=1.0, linestyle="--", alpha=0.7)
            ax2.set_ylabel("Raw amplitude", color=_ORANGE)
            ax2.tick_params(axis="y", colors=_ORANGE)
        if labels and len(labels) == len(x):
            # Show every Nth label to avoid clutter
            step = max(1, len(labels) // 16)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Sublayer index")

    ax.set_ylabel("Frobenius-normalized amplitude", color=_BLUE)
    ax.set_title("Amplitude trajectory across sublayers")
    # Collect handles from BOTH axes. ax.legend() alone only sees the primary
    # axis, so the twinx series was silently missing from the legend even
    # though it was given a label.
    if handles:
        ax.legend(handles, [h.get_label() for h in handles], loc="upper left",
                  fontsize=8)
    fig.tight_layout()
    return _render(fig)


def _plot_heatmap(r: dict) -> bytes:
    hm = r.get("heatmap") or []
    if not hm:
        return _empty_plot("No heatmap data")

    import numpy as np
    arr = np.array(hm)
    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()
    im = ax.imshow(arr, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Sublayer")
    ax.set_title("Per-token amplitude heatmap")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return _render(fig)


def _plot_ltp_profiles(r: dict) -> bytes:
    ltp = r.get("ltp") or {}
    profiles = ltp.get("profiles") or []
    base = ltp.get("base_profiles") or []
    if not profiles:
        return _empty_plot("No LTP profile data")

    import numpy as np
    arr = np.array(profiles)
    base_arr = np.array(base) if base else None

    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()
    # Heatmap of profiles (positions × k)
    im = ax.imshow(arr.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Counterfactual rank (k)")
    ax.set_title("Lateral tension profiles (instruct)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return _render(fig)


def _plot_ltp_tension_magnitudes(r: dict) -> bytes:
    ltp = r.get("ltp") or {}
    mags = ltp.get("tension_magnitudes") or []
    tokens = r.get("tokens") or []
    if not mags:
        return _empty_plot("No LTP tension data")
    n = min(len(mags), len(tokens) if tokens else len(mags))
    mags = mags[:n]

    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()
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

    fig = _new_fig(_FIGSIZE_SQUARE)
    ax = fig.subplots()
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
    ltp = r.get("ltp") or {}
    fields = [("Mean M", ltp.get("mean_M")),
              ("Mean V", ltp.get("mean_V")),
              ("Mean L", ltp.get("mean_L")),
              ("Max PRC", ltp.get("max_prc")),
              ("N directional", ltp.get("n_directional"))]
    fields = [(k, v) for k, v in fields if v is not None]
    if not fields:
        return _empty_plot("No LTP summary data")

    fig = _new_fig(_FIGSIZE_SQUARE)
    ax = fig.subplots()
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

    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()
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
    sfd = r.get("sfd") or {}
    density = sfd.get("per_token_density") or []
    tokens = r.get("tokens") or []
    if not density:
        return _empty_plot("No SFD density data")
    n = min(len(density), len(tokens) if tokens else len(density))
    density = density[:n]

    fig = _new_fig(_FIGSIZE_WIDE)
    ax = fig.subplots()
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

    fig = _new_fig((10, 6))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.bar(x, disp, color=_BLUE, edgecolor=_GRID, linewidth=0.5)
    ax1.set_ylabel("Total displacement")
    # mean_tau is None when no position had enough shared tokens to define it.
    _mt = rd.get("mean_tau")
    _mt_s = f"{_mt:.3f}" if _mt is not None else "n/a"
    ax1.set_title(f"Per-position rank displacement (τ={_mt_s})")
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


# ── Batch / comparative plots ──────────────────────────────────────
#
# Moved here from app.py so there is exactly one plot registry. These
# render from the whole session rather than a single result, and the
# functions live in engine.visualizations / engine.comparative.

# Comparative plots: plot key → function name in engine.comparative.
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

# Aggregate-based plots (need SimpleNamespace for statistics extractors).
_AGGREGATE_PLOT_KEYS = ("batch_summary", "separability")


def _render_batch_plot(plot_key: str, results: list) -> Optional[bytes]:
    """Render a batch comparative plot. Returns PNG bytes or None."""
    import base64

    if plot_key in _AGGREGATE_PLOT_KEYS:
        from types import SimpleNamespace
        from src.engine.statistics import aggregate_batch
        from src.engine.visualizations import plot_batch_summary, plot_separability
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
        import src.engine.comparative as comp
        func = getattr(comp, func_name, None)
        if func:
            b64 = func(results)
            return base64.b64decode(b64) if b64 else None

    return None


def render_session_plot(plot_key: str, results: list) -> Optional[bytes]:
    """One lookup for ``GET /api/plots/{plot_key}``.

    Dispatches on the key itself instead of the old "try the batch
    renderer, and if it returns None fall back to the per-prompt
    renderer" chain — which quietly rendered a per-prompt plot of
    result[0] whenever a genuine batch plot produced no image.

    Returns PNG bytes, or None if the key is unknown / has no data.
    """
    if plot_key in _AGGREGATE_PLOT_KEYS or plot_key in _BATCH_PLOT_DISPATCH:
        return _render_batch_plot(plot_key, results)
    if plot_key in _PLOT_HANDLERS:
        return render_plot(plot_key, results[0]) if results else None
    return None


def list_batch_plot_keys() -> list[str]:
    """Return all known batch/comparative plot keys."""
    return list(_AGGREGATE_PLOT_KEYS) + list(_BATCH_PLOT_DISPATCH.keys())
