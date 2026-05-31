"""
Comparative Visualizations: cross-prompt analytics.
All text minimum 10pt for readability, with word-wrapped labels.
Extended with LTP comparative plots.
"""

import io
import base64
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from misc.old.tasm.engine import engine_config
from misc.old.tasm.engine.viz_style import (
    fig_to_base64 as _fig_to_base64,
    style_ax as _style_ax_full,
    apply_style,
    CAT_COLORS,
    wrap_label as _wrap,
)

# Apply the shared dark theme on import
apply_style()


def _style_ax(ax, title=""):
    """Wrapper for backward compatibility — comparative plots only pass title."""
    _style_ax_full(ax, title=title)


def _cat_legend(ax, results):
    used = set()
    handles = []
    for r in results:
        c = r.get("category", "unknown")
        if c not in used:
            used.add(c)
            handles.append(Patch(facecolor=CAT_COLORS.get(c, "#888"), label=c.title()))
    ax.legend(handles=handles, fontsize=12, loc="best")


def plot_trajectory_overlay(results: list) -> str:
    has_traj = [r for r in results if r.get("amplitude_normalized")]
    if not has_traj:
        return ""

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Normalized Amplitude Trajectories (all prompts)")

    for r in has_traj:
        color = CAT_COLORS.get(r.get("category", ""), "#888")
        ax.plot(r["amplitude_normalized"], color=color, alpha=0.5, linewidth=1.0)

    n = len(has_traj[0]["amplitude_normalized"])
    for b, t in [(n // 3, "Early -> Mid"), (2 * n // 3, "Mid -> Late")]:
        ax.axvline(x=b, color="#404040", linestyle="--", alpha=0.5)
        ax.text(b + 1, ax.get_ylim()[1] * 0.92, t, fontsize=12, color="#6B7280")

    ax.set_xlabel("Sublayer index", fontsize=13)
    ax.set_ylabel("Normalized sensitivity", fontsize=13)
    _cat_legend(ax, has_traj)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_difference_from_benign(results: list) -> str:
    has_traj = [r for r in results if r.get("amplitude_normalized")]
    if not has_traj:
        return ""

    benign = [r["amplitude_normalized"] for r in has_traj
              if r.get("category") in ("benign", "baseline")]
    if not benign:
        return ""

    baseline = np.mean(benign, axis=0)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Sensitivity Difference from Benign Baseline")

    cats = {}
    for r in has_traj:
        cat = r.get("category", "unknown")
        if cat in ("benign", "baseline"):
            continue
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(r["amplitude_normalized"])

    for cat, trajs in cats.items():
        mean_traj = np.mean(trajs, axis=0)
        diff = mean_traj - baseline
        color = CAT_COLORS.get(cat, "#888")
        ax.plot(diff, label=f"{cat.title()} - Benign", color=color, linewidth=1.8)
        ax.fill_between(range(len(diff)), diff, alpha=0.1, color=color)

    ax.axhline(y=0, color="#404040", linewidth=1)
    ax.set_xlabel("Sublayer index", fontsize=13)
    ax.set_ylabel("Delta normalized sensitivity", fontsize=13)
    ax.legend(fontsize=12)

    n = len(baseline)
    for b, t in [(n // 3, "Early -> Mid"), (2 * n // 3, "Mid -> Late")]:
        ax.axvline(x=b, color="#404040", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_discriminative_sublayers(results: list, top_n: int = None) -> str:
    if top_n is None:
        top_n = engine_config.get("disc_sublayers_top_n")
    has_traj = [r for r in results if r.get("amplitude_normalized")]
    if not has_traj:
        return ""

    benign = [r["amplitude_normalized"] for r in has_traj
              if r.get("category") in ("benign", "baseline")]
    adversarial = [r["amplitude_normalized"] for r in has_traj
                   if r.get("category") in ("jailbreak", "harmful", "adversarial")]

    if not benign or not adversarial:
        return ""

    benign_mean = np.mean(benign, axis=0)
    adv_mean = np.mean(adversarial, axis=0)
    diff = adv_mean - benign_mean

    n_sublayers = len(diff)
    n_layers = n_sublayers // 2
    sorted_idx = np.argsort(diff)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.45)))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, f"Top {top_n} Discriminative Sublayers\n(Adversarial - Benign)")

    labels = []
    values = []
    colors = []
    for rank, idx in enumerate(sorted_idx):
        layer_num = idx // 2
        sublayer = "Attn" if idx % 2 == 0 else "MLP"
        region = "Early" if layer_num < n_layers // 3 else (
            "Middle" if layer_num < 2 * n_layers // 3 else "Late")
        labels.append(f"L{layer_num} {sublayer} ({region})")
        values.append(diff[idx])
        colors.append("#CC79A7" if region == "Middle" else (
            "#56B4E9" if region == "Late" else "#009E73"))

    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.65, edgecolor="#1a1a1a")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, color="#9CA3AF")
    ax.set_xlabel("Delta sensitivity", fontsize=13)
    ax.invert_yaxis()

    handles = [Patch(facecolor="#009E73", label="Early"),
               Patch(facecolor="#CC79A7", label="Middle"),
               Patch(facecolor="#56B4E9", label="Late")]
    ax.legend(handles=handles, fontsize=12, loc="lower right")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_metric_scatters(results: list) -> str:
    if len(results) < 2:
        return ""

    scatter_defs = [
        ("stress_score", "kl_divergence", "Stress Score", "KL Divergence"),
        ("entropy", "net_correction", "Entropy", "Net Correction"),
        ("top2_share", "middle_share", "Boundary Share", "Interior Share"),
        ("stress_score", "net_correction", "Stress Score", "Net Correction"),
        ("entropy", "stress_score", "Entropy", "Stress Score"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#121212")

    for ax, (x_key, y_key, x_label, y_label) in zip(axes.flat, scatter_defs):
        _style_ax(ax, f"{x_label} vs {y_label}")

        for r in results:
            xv = r.get(x_key)
            yv = r.get(y_key)
            if xv is None or yv is None:
                continue
            color = CAT_COLORS.get(r.get("category", ""), "#888")
            ax.scatter(xv, yv, c=color, s=60, alpha=0.8,
                       edgecolors="#1E1E1E", linewidth=0.5)

        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)

    # Hide unused axes
    for ax in axes.flat[len(scatter_defs):]:
        ax.set_visible(False)

    _cat_legend(axes[0, 0], results)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_behavioral_comparison(results: list) -> str:
    has_both = [r for r in results
                if r.get("instruct_topk") and r.get("base_topk")]
    if not has_both:
        return ""

    fig, ax = plt.subplots(figsize=(max(11, len(has_both) * 1.0), 5.5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Behavioral Divergence: Instruct vs Base\nTop-1 Probability")

    x = np.arange(len(has_both))
    w = 0.35

    inst_probs = [r["instruct_topk"][0][1] for r in has_both]
    base_probs = [r["base_topk"][0][1] for r in has_both]
    labels = [_wrap(r["prompt"][:35], 18) for r in has_both]

    ax.bar(x - w/2, inst_probs, w, label="Instruct", color="#56B4E9", alpha=0.85)
    ax.bar(x + w/2, base_probs, w, label="Base", color="#D55E00", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12, color="#9CA3AF")
    ax.set_ylabel("Top-1 probability", fontsize=13)
    ax.legend(fontsize=13)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_proof1_summary(results: list) -> str:
    all_checks = []
    for r in results:
        for c in r.get("proof1_checks", []):
            all_checks.append(c)

    if not all_checks:
        return ""

    errors = [c["error"] for c in all_checks]
    exact_rate = sum(1 for c in all_checks if c["exact"]) / len(all_checks)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#121212")

    ax = axes[0]
    _style_ax(ax, "Proof 1 Exactness Errors\n(log scale)")
    ax.hist(np.log10(np.array(errors) + 1e-15), bins=30, color="#56B4E9", alpha=0.8,
            edgecolor="#1E1E1E")
    ax.set_xlabel("log10(error)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)

    ax = axes[1]
    _style_ax(ax, f"Exactness Rate: {exact_rate:.1%}")
    bars = ax.bar(["Exact\n(< 1e-4)", "Inexact"], [exact_rate, 1 - exact_rate],
                  color=["#009E73", "#D55E00"], alpha=0.85, width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction", fontsize=13)
    for bar, val in zip(bars, [exact_rate, 1 - exact_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", fontsize=13, color="#DEE2E6")

    plt.tight_layout()
    return _fig_to_base64(fig)


# ─── LTP Comparative Plots ──────────────────────────────────────

def plot_ltp_category_comparison(results: list) -> str:
    """Box plots of LTP summary statistics by category."""
    cats_order = ["benign", "mild", "harmful", "jailbreak"]
    has_ltp = [r for r in results if r.get("ltp")]
    if not has_ltp:
        return ""

    available = []
    for c in cats_order:
        if any(r.get("category") == c for r in has_ltp):
            available.append(c)
    if not available:
        return ""

    ltp_keys = [("mean_M", "Offset\nMagnitude"),
                ("mean_V", "Offset\nVariance")]

    fig, axes = plt.subplots(1, len(ltp_keys), figsize=(4.5 * len(ltp_keys), 5))
    fig.patch.set_facecolor("#121212")

    for ax, (key, title) in zip(axes, ltp_keys):
        _style_ax(ax, title)
        box_data = []
        box_labels = []
        box_colors = []

        for cat in available:
            vals = [r["ltp"][key] for r in has_ltp
                    if r.get("category") == cat and r.get("ltp") and r["ltp"].get(key) is not None]
            if vals:
                box_data.append(vals)
                box_labels.append(cat.title())
                box_colors.append(CAT_COLORS.get(cat, "#888"))

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor("#9CA3AF")
            for element in ["whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set_color("#9CA3AF")
            for item in bp["fliers"]:
                item.set_markeredgecolor("#9CA3AF")
            ax.tick_params(axis="x", labelsize=13, colors="#9CA3AF")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_ltp_m_vs_stress(results: list) -> str:
    """Scatter: LTP offset magnitude vs ASM stress score, colored by category."""
    has_ltp = [r for r in results if r.get("ltp") and r["ltp"].get("mean_M") is not None]
    if len(has_ltp) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Offset Magnitude (M) vs Stress Score")

    for r in has_ltp:
        color = CAT_COLORS.get(r.get("category", ""), "#888")
        ax.scatter(r.get("stress_score", 0), r["ltp"]["mean_M"],
                   c=color, s=60, alpha=0.8, edgecolors="#1E1E1E", linewidth=0.5)

    ax.set_xlabel("Stress Score (ASM)", fontsize=13)
    ax.set_ylabel("Offset Magnitude M (LTP)", fontsize=13)
    _cat_legend(ax, has_ltp)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_ltp_profile_shape_distribution(results: list) -> str:
    """Stacked bar chart of profile shape distribution by category."""
    has_ltp = [r for r in results if r.get("ltp") and r["ltp"].get("profile_shapes")]
    if not has_ltp:
        return ""

    cats_order = ["benign", "mild", "harmful", "jailbreak"]
    available = [c for c in cats_order if any(r.get("category") == c for r in has_ltp)]
    if not available:
        return ""

    fig, ax = plt.subplots(figsize=(max(8, len(available) * 2.5), 5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "Profile Shape Distribution by Category")

    shapes = ["steep", "flat", "inverted"]
    shape_colors = {"steep": "#E69F00", "flat": "#56B4E9", "inverted": "#D55E00"}
    x = np.arange(len(available))
    width = 0.25

    for j, shape in enumerate(shapes):
        fracs = []
        for cat in available:
            cat_shapes = []
            for r in has_ltp:
                if r.get("category") == cat:
                    cat_shapes.extend(r["ltp"]["profile_shapes"])
            total = len(cat_shapes) if cat_shapes else 1
            count = sum(1 for s in cat_shapes if s == shape)
            fracs.append(count / total)
        ax.bar(x + j * width, fracs, width, label=shape.title(),
               color=shape_colors[shape], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.title() for c in available], fontsize=13)
    ax.set_ylabel("Fraction of tokens", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_key_scatters(results: list) -> str:
    """Two-panel scatter: the two views that actually separate categories.
    Entropy vs Net Correction and Stress vs Net Correction.
    Includes 95% confidence ellipses per category."""
    from misc.old.tasm.engine.viz_style import (fig_to_base64, style_ax, apply_style,
                                   CAT_COLORS, CAT_ORDER, CAT_MARKERS,
                                   TEXT_PRIMARY, TEXT_SECONDARY)
    apply_style()

    panels = [
        ("entropy", "net_correction", "Entropy", "Net Correction"),
        ("stress_score", "net_correction", "Stress Score", "Net Correction"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, panels):
        style_ax(ax, xlabel=xlabel, ylabel=ylabel)

        for cat in CAT_ORDER:
            pts = [(r[xkey], r[ykey]) for r in results
                   if r.get("category") == cat and r.get(xkey) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            xs, ys = np.array(xs), np.array(ys)
            color = CAT_COLORS.get(cat, "#888")
            marker = CAT_MARKERS.get(cat, "o")

            # Scatter with redundant shape+color
            ax.scatter(xs, ys, c=color, marker=marker, s=40, alpha=0.7,
                       edgecolors="white", linewidths=0.4, zorder=3,
                       label=cat.capitalize())

            # 95% confidence ellipse
            if len(xs) >= 3:
                try:
                    from matplotlib.patches import Ellipse
                    cov = np.cov(xs, ys)
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    order = eigenvals.argsort()[::-1]
                    eigenvals = eigenvals[order]
                    eigenvecs = eigenvecs[:, order]
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    # 95% CI for 2D normal = chi2(2, 0.95) = 5.991
                    width = 2 * np.sqrt(5.991 * eigenvals[0])
                    height = 2 * np.sqrt(5.991 * eigenvals[1])
                    ell = Ellipse(xy=(xs.mean(), ys.mean()),
                                  width=width, height=height, angle=angle,
                                  facecolor=color, alpha=0.1,
                                  edgecolor=color, linewidth=1.5,
                                  linestyle="--", zorder=1)
                    ax.add_patch(ell)
                except Exception:
                    pass  # Skip ellipse on numerical issues

        ax.legend(fontsize=12, loc="best", framealpha=0.85)

    plt.tight_layout(pad=1.5)
    return fig_to_base64(fig)


def plot_sfd_category_comparison(results: list) -> str:
    """Box plot of SFD density by category."""
    cats_order = ["benign", "mild", "harmful", "jailbreak"]
    has_sfd = [r for r in results if r.get("sfd")]
    if not has_sfd:
        return ""

    available = [c for c in cats_order if any(r.get("category") == c for r in has_sfd)]
    if not available:
        return ""

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "QK Density by Category")

    box_data = []
    box_labels = []
    box_colors = []

    for cat in available:
        vals = [r["sfd"]["density_mean"] for r in has_sfd
                if r.get("category") == cat and r.get("sfd") and r["sfd"].get("density_mean") is not None]
        if vals:
            box_data.append(vals)
            box_labels.append(cat.title())
            box_colors.append(CAT_COLORS.get(cat, "#888"))

    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#9CA3AF")
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("#9CA3AF")
        for item in bp["fliers"]:
            item.set_markeredgecolor("#9CA3AF")
        ax.tick_params(axis="x", labelsize=13, colors="#9CA3AF")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_sfd_vs_asm(results: list) -> str:
    """Scatter: SFD density vs ASM middle share, colored by category."""
    has_sfd = [r for r in results if r.get("sfd") and r["sfd"].get("density_mean") is not None]
    if len(has_sfd) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#121212")
    _style_ax(ax, "QK Density (SFD) vs Interior Share (ASM)")

    for r in has_sfd:
        color = CAT_COLORS.get(r.get("category", ""), "#888")
        ax.scatter(r.get("middle_share", 0), r["sfd"]["density_mean"],
                   c=color, s=60, alpha=0.8, edgecolors="#1E1E1E", linewidth=0.5)

    ax.set_xlabel("Interior Share (ASM)", fontsize=13)
    ax.set_ylabel("QK Density (SFD)", fontsize=13)
    _cat_legend(ax, has_sfd)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_rank_displacement_by_category(results: list) -> str:
    """Box plots of rank displacement (Kendall tau and overlap) by category."""
    cats_order = ["benign", "mild", "harmful", "jailbreak"]
    has_rd = [r for r in results if r.get("rank_displacement") and
              r["rank_displacement"].get("mean_tau") is not None]
    if not has_rd:
        return ""

    available = [c for c in cats_order if any(r.get("category") == c for r in has_rd)]
    if not available:
        return ""

    rd_keys = [("mean_tau", "Kendall Tau"), ("mean_overlap", "Token Overlap")]

    fig, axes = plt.subplots(1, len(rd_keys), figsize=(4.5 * len(rd_keys), 5))
    fig.patch.set_facecolor("#121212")

    for ax, (key, title) in zip(axes, rd_keys):
        _style_ax(ax, title)
        box_data = []
        box_labels = []
        box_colors = []

        for cat in available:
            vals = [r["rank_displacement"][key] for r in has_rd
                    if r.get("category") == cat and r["rank_displacement"].get(key) is not None]
            if vals:
                box_data.append(vals)
                box_labels.append(cat.title())
                box_colors.append(CAT_COLORS.get(cat, "#888"))

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor("#9CA3AF")
            for element in ["whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set_color("#9CA3AF")
            for item in bp["fliers"]:
                item.set_markeredgecolor("#9CA3AF")
            ax.tick_params(axis="x", labelsize=13, colors="#9CA3AF")

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_all_comparative(results: list) -> dict:
    """Generate all comparative plots, organized into proven and experimental."""
    proven = {}
    experimental = {}

    # Proven (Analysis tab)
    proven["key_scatters"] = plot_key_scatters(results)
    proven["discriminative_sublayers"] = plot_discriminative_sublayers(results)
    proven["proof1_summary"] = plot_proof1_summary(results)

    # Experimental tab
    experimental["trajectory_overlay"] = plot_trajectory_overlay(results)
    experimental["difference_from_benign"] = plot_difference_from_benign(results)
    experimental["metric_scatters"] = plot_metric_scatters(results)
    experimental["behavioral_comparison"] = plot_behavioral_comparison(results)
    experimental["ltp_category_comparison"] = plot_ltp_category_comparison(results)
    experimental["ltp_m_vs_stress"] = plot_ltp_m_vs_stress(results)
    experimental["ltp_profile_shapes"] = plot_ltp_profile_shape_distribution(results)
    experimental["sfd_category_comparison"] = plot_sfd_category_comparison(results)
    experimental["sfd_vs_asm"] = plot_sfd_vs_asm(results)
    experimental["rank_displacement"] = plot_rank_displacement_by_category(results)

    # Flatten into single dict with prefix for experimental
    all_plots = {k: v for k, v in proven.items() if v}
    for k, v in experimental.items():
        if v:
            all_plots[f"exp_{k}"] = v

    return all_plots
