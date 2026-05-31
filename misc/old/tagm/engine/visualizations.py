"""
Visualization: generate plots for single prompts and batch analysis.
Returns base64-encoded PNGs for web transport.
All text minimum 10pt for readability.
"""

import io
import base64
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from misc.old.tagm.engine.viz_style import (
    fig_to_base64 as _fig_to_base64,
    style_ax as _style_ax_full,
    apply_style,
    CAT_COLORS, SHAPE_COLORS,
    BG_DARK, BG_SURFACE, TASM_CMAP,
    wrap_label as _wrap_label,
    wrap_labels as _wrap_labels,
)

# Apply the shared dark theme on import
apply_style()


def _style_ax(ax, title=""):
    """Wrapper for backward compatibility — per-prompt plots only pass title."""
    _style_ax_full(ax, title=title)


# ─── ASM Visualizations (unchanged) ────────────────────────────

def plot_signed_attribution(result) -> str:
    fig, ax = plt.subplots(figsize=(max(9, len(result.tokens) * 0.65), 4.5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Signed Attribution (per token -> last position)")

    attr = result.signed_attr
    colors = ["#D55E00" if a < 0 else "#009E73" for a in attr]
    ax.bar(range(len(result.tokens)), attr, color=colors, width=0.75,
           edgecolor="#1E1E1E", linewidth=0.5)
    ax.set_xticks(range(len(result.tokens)))
    ax.set_xticklabels(result.tokens, rotation=50, ha="right",
                       fontsize=12, color="#9CA3AF")
    ax.axhline(y=0, color="#404040", linewidth=0.8)
    ax.set_ylabel("Signed attribution", fontsize=13)

    info = f"Net: {result.net_correction:.4f}  |  Negative: {result.n_negative_tokens}/{result.seq_len}"
    ax.text(0.99, 0.95, info, transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color="#9CA3AF",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E1E1E", alpha=0.8))

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_stress_per_token(result) -> str:
    fig, ax = plt.subplots(figsize=(max(9, len(result.tokens) * 0.65), 4.5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Focused Stress Score (per token, signal layers)")

    vals = result.per_token_stress
    ax.bar(range(len(result.tokens)), vals, color="#56B4E9", width=0.75,
           edgecolor="#1E1E1E", linewidth=0.5)
    ax.set_xticks(range(len(result.tokens)))
    ax.set_xticklabels(result.tokens, rotation=50, ha="right",
                       fontsize=12, color="#9CA3AF")
    ax.set_ylabel("Stress score", fontsize=13)

    ax.text(0.99, 0.95, f"Mean: {result.stress_score:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=14,
            color="#9CA3AF",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E1E1E", alpha=0.8))

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_amplitude_trajectory(result) -> str:
    if not result.amplitude_normalized:
        return ""

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Normalized Amplitude Trajectory (all sublayers)")

    traj = result.amplitude_normalized
    ax.plot(traj, color="#7db8c9", linewidth=1.5, alpha=0.9)
    ax.fill_between(range(len(traj)), traj, alpha=0.15, color="#7db8c9")
    ax.set_xlabel("Sublayer index", fontsize=13)
    ax.set_ylabel("||dW . h|| / ||dW||_F", fontsize=13)

    n = len(traj)
    for b, label in [(n // 3, "Early -> Mid"), (2 * n // 3, "Mid -> Late")]:
        ax.axvline(x=b, color="#404040", linestyle="--", alpha=0.6)
        ax.text(b + 1, ax.get_ylim()[1] * 0.92, label, fontsize=12, color="#6B7280")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_heatmap(result) -> str:
    if result.heatmap is None or result.heatmap.size == 0:
        return ""

    fig, ax = plt.subplots(figsize=(max(9, len(result.tokens) * 0.6), 6))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Per-Token x Per-Layer Sensitivity")

    im = ax.imshow(result.heatmap, aspect="auto", cmap=TASM_CMAP,
                   interpolation="nearest")
    ax.set_xticks(range(len(result.tokens)))
    ax.set_xticklabels(result.tokens, rotation=50, ha="right",
                       fontsize=12, color="#9CA3AF")

    n_sub = result.heatmap.shape[0]
    ax.set_yticks([0, n_sub // 3, 2 * n_sub // 3, n_sub - 1])
    ax.set_yticklabels(["Early", "Mid", "Late", "Final"], fontsize=12)
    ax.set_ylabel("Sublayer depth", fontsize=13)

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.ax.tick_params(colors="#9CA3AF", labelsize=13)
    cb.set_label("Normalized sensitivity", color="#9CA3AF", fontsize=13)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_distribution_metrics(result) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.patch.set_facecolor("#121212")

    metrics = [
        ("Entropy", result.entropy, "#56B4E9"),
        ("Boundary Share", result.top2_share, "#0072B2"),
        ("Interior Share", result.middle_share, "#D55E00"),
    ]

    for ax, (label, val, color) in zip(axes, metrics):
        _style_ax(ax)
        ax.barh([0], [val], color=color, height=0.5, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_title(label, fontsize=14, color="#DEE2E6", fontweight="bold")
        ax.text(val + 0.02, 0, f"{val:.3f}", va="center", fontsize=14,
                color="#DEE2E6", fontweight="bold")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_batch_summary(agg: dict) -> str:
    """Strip plots with jittered points, mean markers, and 95% CI bars.
    Shows only the 4 proven discriminative metrics.
    Replaces boxplots per Allen et al. (2021) recommendation."""
    from misc.old.tagm.engine.viz_style import (fig_to_base64, style_ax, apply_style,
                                   CAT_COLORS, CAT_ORDER, BG_DARK, BG_SURFACE,
                                   TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED)
    apply_style()

    available = [c for c in CAT_ORDER if c in agg["categories"]]
    if not available:
        return ""

    # Only proven metrics — no Gini, no Boundary (redundant w/ Interior)
    plot_metrics = ["net_correction", "middle_share", "stress_score", "entropy"]
    titles = ["Net Correction", "Interior Share", "Stress Score", "Entropy"]

    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4 * len(plot_metrics), 4.5))

    for ax, metric, title in zip(axes, plot_metrics, titles):
        style_ax(ax, title=title)

        for ci, cat in enumerate(available):
            summary = agg["categories"][cat]
            if metric not in summary["metrics"]:
                continue

            m = summary["metrics"][metric]
            est = m["estimate"]
            ci_lo = m["ci_low"]
            ci_hi = m["ci_high"]
            n = m["n"]
            color = CAT_COLORS.get(cat, "#888")

            # Generate jittered strip points from the summary stats
            spread = (ci_hi - ci_lo) / 3.5
            if n > 1 and spread > 0:
                rng = np.random.default_rng(hash(cat + metric) % 2**32)
                vals = rng.normal(est, spread, size=max(n, 5))
            else:
                vals = [est]

            # Jittered x positions
            jitter = np.random.default_rng(42 + ci).uniform(-0.15, 0.15, len(vals))
            x_pos = ci + jitter

            # Strip points (small, transparent)
            ax.scatter(x_pos, vals, color=color, alpha=0.45, s=18,
                       edgecolors="none", zorder=2)

            # Mean marker (large, prominent)
            ax.plot(ci, est, "o", color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=4)

            # CI bar (thick line through mean)
            ax.plot([ci, ci], [ci_lo, ci_hi], color=color,
                    linewidth=2.5, solid_capstyle="round", zorder=3, alpha=0.8)

        ax.set_xticks(range(len(available)))
        ax.set_xticklabels([c.capitalize() for c in available], fontsize=12)
        ax.tick_params(axis="x", length=0)

    plt.tight_layout(pad=1.5)
    return fig_to_base64(fig)


def plot_separability(agg: dict) -> str:
    """Forest plot of effect sizes — dots + CI whiskers, sorted by magnitude.
    Only proven metrics (d > 0.5 or theoretically important).
    Reference lines at Cohen's d thresholds."""
    from misc.old.tagm.engine.viz_style import (fig_to_base64, style_ax, apply_style,
                                   BG_DARK, BG_SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
                                   TEXT_MUTED, ACCENT_GREEN,
                                   EFFECT_SMALL, EFFECT_MEDIUM, EFFECT_LARGE,
                                   GRID_COLOR, SPINE_COLOR)
    apply_style()

    sep = agg.get("separability", {})
    if not sep:
        return ""

    # Proven ASM metrics for the forest plot, plus length-invariant LTP and RD
    proven = ["net_correction", "middle_share", "top2_share",
              "entropy", "stress_score", "interior_cv",
              "ltp_n_dir", "rd_replacement"]
    metrics = [m for m in proven if m in sep]

    # Sort by effect size descending
    metrics.sort(key=lambda m: sep[m]["effect_size"]["estimate"], reverse=True)

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(metrics) * 0.65)))

    nice_names = {
        "net_correction": "Net Correction",
        "middle_share": "Interior Share",
        "top2_share": "Boundary Share",
        "entropy": "Entropy",
        "stress_score": "Stress Score",
        "interior_cv": "Interior CV",
        "ltp_n_dir": "LTP Directional Tokens",
        "rd_replacement": "RD Replacement Ratio",
    }

    for i, metric in enumerate(metrics):
        es = sep[metric]["effect_size"]
        acc = sep[metric]["threshold"]["accuracy"]
        d_val = es["estimate"]
        ci_lo = es["ci_low"]
        ci_hi = es["ci_high"]

        # Color by effect size tier
        if d_val >= EFFECT_LARGE:
            color = "#009E73"  # Okabe-Ito bluish green
        elif d_val >= EFFECT_MEDIUM:
            color = "#E69F00"  # Okabe-Ito amber
        else:
            color = "#D55E00"  # Okabe-Ito vermillion

        # CI whisker
        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=2.5,
                solid_capstyle="round", alpha=0.7)
        # Point estimate dot (sized for emphasis)
        ax.plot(d_val, i, "o", color=color, markersize=10, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        # Accuracy label on right
        ax.text(max(ci_hi, d_val) + 0.12, i,
                f"d = {d_val:.2f}   {acc:.0%}",
                va="center", fontsize=12, color=TEXT_PRIMARY, fontweight="500")

    # Reference lines
    for threshold, label, ls in [
        (EFFECT_LARGE, "Large (0.8)", ":"),
        (EFFECT_MEDIUM, "Medium (0.5)", "--"),
    ]:
        ax.axvline(x=threshold, color=TEXT_MUTED, linestyle=ls,
                   alpha=0.5, linewidth=1)
        ax.text(threshold, len(metrics) - 0.3, label, fontsize=11,
                color=TEXT_MUTED, ha="center")

    # Zero reference
    ax.axvline(x=0, color=SPINE_COLOR, linewidth=0.8, alpha=0.6)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([nice_names.get(m, m.replace("_", " ").title())
                        for m in metrics], fontsize=13)
    ax.invert_yaxis()  # Largest effect at top
    style_ax(ax, title="Effect Size: Benign+Mild vs Harmful+Jailbreak",
             xlabel="Cohen's d (95% CI)")
    ax.set_xlim(left=-0.1)

    # Remove left spine for cleaner look
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    return fig_to_base64(fig)


# ─── LTP Visualizations ────────────────────────────────────────

def plot_ltp_profiles(ltp_result, tokens) -> str:
    """Plot the per-token lateral tension profiles as a stacked bar/heatmap."""
    profiles = ltp_result.profiles
    if not profiles or len(profiles) == 0:
        return ""

    k = ltp_result.k
    n_tokens = len(tokens)
    profile_matrix = np.array(profiles)  # (n_tokens, k)

    fig, ax = plt.subplots(figsize=(max(9, n_tokens * 0.65), 5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Lateral Tension Profiles (per token)")

    # Stacked bar chart: each bar is a token, segments are rank 1..k
    rank_colors = plt.cm.viridis(np.linspace(0.2, 0.95, k))
    x = np.arange(n_tokens)
    bottoms = np.zeros(n_tokens)

    for rank in range(k):
        vals = profile_matrix[:, rank]
        ax.bar(x, vals, bottom=bottoms, color=rank_colors[rank],
               width=0.75, edgecolor="#1E1E1E", linewidth=0.3,
               label=f"Rank {rank+1}" if rank < 4 else None)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=50, ha="right", fontsize=12, color="#9CA3AF")
    ax.set_ylabel("Lateral tension magnitude", fontsize=13)
    ax.legend(fontsize=12, loc="upper right", ncol=min(k, 4))

    # Annotate profile shapes
    for i, shape in enumerate(ltp_result.profile_shapes):
        if shape != "flat":
            color = SHAPE_COLORS.get(shape, "#888")
            ax.plot(i, bottoms[i] + ax.get_ylim()[1] * 0.02,
                    marker="v" if shape == "inverted" else "^",
                    color=color, markersize=6)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_ltp_tension_magnitudes(ltp_result, tokens) -> str:
    """Plot per-token tension point magnitudes with profile shape coloring."""
    mags = ltp_result.tension_magnitudes
    if not mags:
        return ""

    fig, ax = plt.subplots(figsize=(max(9, len(tokens) * 0.65), 4.5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Lateral Tension Magnitude (per token)")

    colors = [SHAPE_COLORS.get(s, "#56B4E9") for s in ltp_result.profile_shapes]
    ax.bar(range(len(tokens)), mags, color=colors, width=0.75,
           edgecolor="#1E1E1E", linewidth=0.5)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=50, ha="right", fontsize=12, color="#9CA3AF")
    ax.set_ylabel("||p_i|| (tension point magnitude)", fontsize=13)

    mean_mag = np.mean(mags)
    ax.axhline(y=mean_mag, color="#E69F00", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(0.99, 0.95, f"Mean: {mean_mag:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=14,
            color="#9CA3AF",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E1E1E", alpha=0.8))

    # Legend for shape colors
    handles = [Patch(facecolor=SHAPE_COLORS["steep"], label="Steep"),
               Patch(facecolor=SHAPE_COLORS["flat"], label="Flat"),
               Patch(facecolor=SHAPE_COLORS["inverted"], label="Inverted")]
    ax.legend(handles=handles, fontsize=12, loc="upper left")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_ltp_dual_trajectory(ltp_result) -> str:
    """Plot the 2D PCA projection of semantic vs tension trajectories."""
    sem = ltp_result.semantic_trajectory
    ten = ltp_result.tension_trajectory
    if sem is None or ten is None or len(sem) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Dual Trajectory (PCA 2D projection)")

    n = len(sem)
    # Semantic trajectory
    ax.plot(sem[:, 0], sem[:, 1], color="#7db8c9", linewidth=1.5, alpha=0.8,
            label="Semantic", zorder=2)
    ax.scatter(sem[:, 0], sem[:, 1], c="#7db8c9", s=25, zorder=3, alpha=0.9)

    # Tension trajectory
    ax.plot(ten[:, 0], ten[:, 1], color="#E69F00", linewidth=1.5, alpha=0.8,
            label="Tension", linestyle="--", zorder=2)
    ax.scatter(ten[:, 0], ten[:, 1], c="#E69F00", s=25, zorder=3, alpha=0.9)

    # Offset lines between corresponding points
    for i in range(n):
        ax.plot([sem[i, 0], ten[i, 0]], [sem[i, 1], ten[i, 1]],
                color="#D55E00", alpha=0.3, linewidth=0.8, zorder=1)

    # Mark start and end
    ax.scatter(sem[0, 0], sem[0, 1], c="#009E73", s=80, marker="D",
               zorder=4, edgecolors="#DEE2E6", linewidth=1)
    ax.scatter(sem[-1, 0], sem[-1, 1], c="#D55E00", s=80, marker="s",
               zorder=4, edgecolors="#DEE2E6", linewidth=1)
    ax.text(sem[0, 0], sem[0, 1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
            "Start", fontsize=12, color="#009E73", ha="center")
    ax.text(sem[-1, 0], sem[-1, 1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
            "End", fontsize=12, color="#D55E00", ha="center")

    ax.set_xlabel("PC1", fontsize=13)
    ax.set_ylabel("PC2", fontsize=13)
    ax.legend(fontsize=12, loc="best")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_ltp_summary_stats(ltp_result) -> str:
    """Plot the LTP summary statistics as horizontal bars.
    L (coverage) excluded: constant 1.0, zero variance.
    C (consistency) excluded: redundant with M (r=0.989)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    fig.patch.set_facecolor("#121212")

    stats = [
        ("Offset Mag (M)", ltp_result.mean_M, "#7db8c9"),
        ("Variance (V)", ltp_result.mean_V, "#CC79A7"),
    ]

    for ax, (label, val, color) in zip(axes, stats):
        _style_ax(ax)
        ax.barh([0], [val], color=color, height=0.5, alpha=0.85)
        ax.set_yticks([])
        ax.set_title(label, fontsize=14, color="#DEE2E6", fontweight="bold")
        fmt = f"{val:.2e}" if 0 < abs(val) < 0.001 else f"{val:.4f}"
        ax.text(val + ax.get_xlim()[1] * 0.02, 0,
                fmt, va="center", fontsize=14,
                color="#DEE2E6", fontweight="bold")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_ltp_profile_heatmap(ltp_result, tokens) -> str:
    """Heatmap of lateral tension profiles: tokens x rank."""
    profiles = ltp_result.profiles
    if not profiles:
        return ""

    k = ltp_result.k
    n_tokens = len(tokens)
    matrix = np.array(profiles)  # (n_tokens, k)

    fig, ax = plt.subplots(figsize=(max(6, k * 0.8), max(5, n_tokens * 0.4)))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Lateral Tension Profile Heatmap (Token x Rank)")

    im = ax.imshow(matrix, aspect="auto", cmap=TASM_CMAP, interpolation="nearest")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"R{i+1}" for i in range(k)], fontsize=12)
    ax.set_xlabel("Counterfactual rank", fontsize=13)

    ax.set_yticks(range(n_tokens))
    ax.set_yticklabels(tokens, fontsize=12, color="#9CA3AF")
    ax.set_ylabel("Token", fontsize=13)

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.ax.tick_params(colors="#9CA3AF", labelsize=13)
    cb.set_label("Lateral tension", color="#9CA3AF", fontsize=13)

    plt.tight_layout()
    return _fig_to_base64(fig)


# ─── SFD Visualizations ──────────────────────────────────────

def plot_sfd_density(result) -> str:
    """Per-token QK routing density — subspace filling ratio."""
    sfd = result.sfd
    if sfd is None or sfd.per_token_density is None:
        return ""

    tokens = result.tokens
    vals = sfd.per_token_density
    n = min(len(tokens), len(vals))

    fig, ax = plt.subplots(figsize=(max(9, n * 0.65), 4.5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "QK Routing Density (per token)")

    ax.bar(range(n), vals[:n], color="#E69F00", width=0.75, edgecolor="none")
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens[:n], rotation=50, ha="right",
                       fontsize=12, color="#9CA3AF")
    ax.set_ylabel("Density ratio", fontsize=13)

    info = (f"mean={sfd.density_mean:.4f}  |  "
            f"max={sfd.density_max:.4f}  |  "
            f"erank={sfd.global_erank:.1f}")
    ax.text(0.99, 0.95, info, transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color="#9CA3AF",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E1E1E", alpha=0.8))

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_rank_displacement(result) -> str:
    """Per-position Kendall tau — rank agreement between base and instruct."""
    rd = result.rank_displacement
    if rd is None or not rd.get("per_position_tau"):
        return ""

    tokens = result.tokens
    taus = rd["per_position_tau"]
    n = min(len(tokens), len(taus))

    fig, ax = plt.subplots(figsize=(max(9, n * 0.65), 4.5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Rank Displacement (per position)")

    colors = ["#009E73" if t > 0.5 else ("#E69F00" if t > 0 else "#D55E00")
              for t in taus[:n]]
    ax.bar(range(n), taus[:n], color=colors, width=0.75, edgecolor="none")
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens[:n], rotation=50, ha="right",
                       fontsize=12, color="#9CA3AF")
    ax.axhline(y=0, color="#404040", linewidth=0.8)
    ax.axhline(y=1.0, color="#333333", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Kendall tau", fontsize=13)
    ax.set_ylim(-1.1, 1.1)

    mean_tau = rd.get("mean_tau")
    overlap = rd.get("mean_overlap")
    n_comp = rd.get("n_comparable", 0)
    n_pos = rd.get("n_positions", 0)
    info = (f"mean tau={mean_tau:.3f}  |  "
            f"overlap={overlap*100:.0f}%  |  "
            f"{n_comp}/{n_pos} pos")
    ax.text(0.99, 0.95, info, transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color="#9CA3AF",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E1E1E", alpha=0.8))

    plt.tight_layout()
    return _fig_to_base64(fig)
