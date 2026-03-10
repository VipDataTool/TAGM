"""
Visualization: generate plots for single prompts and batch analysis.
Returns base64-encoded PNGs for web transport.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# Consistent color palette
CAT_COLORS = {
    "benign": "#2d936c",
    "baseline": "#2d936c",
    "user_baseline": "#78b89a",
    "mild": "#e0a458",
    "harmful": "#c44536",
    "jailbreak": "#7b2d8b",
    "adversarial": "#7b2d8b",
    "unknown": "#888888",
}

TASM_CMAP = LinearSegmentedColormap.from_list(
    "tasm", ["#0d1117", "#1a1e2e", "#2d3a6b", "#4a6fa5", "#7db8c9",
             "#c4e0a5", "#f0e68c", "#e8a838", "#c44536", "#8b1a1a"])


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _style_ax(ax, title=""):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#a0aec0")
    ax.xaxis.label.set_color("#a0aec0")
    ax.yaxis.label.set_color("#a0aec0")
    ax.title.set_color("#e2e8f0")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_color("#2d3748")
    ax.grid(True, alpha=0.15, color="#4a5568")


def plot_signed_attribution(result) -> str:
    """Per-token signed attribution bar chart."""
    fig, ax = plt.subplots(figsize=(max(8, len(result.tokens) * 0.55), 3.5))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Signed Attribution (per token → last position)")

    attr = result.signed_attr
    colors = ["#c44536" if a < 0 else "#2d936c" for a in attr]
    ax.bar(range(len(result.tokens)), attr, color=colors, width=0.75,
           edgecolor="#1a202c", linewidth=0.5)
    ax.set_xticks(range(len(result.tokens)))
    ax.set_xticklabels(result.tokens, rotation=50, ha="right",
                       fontsize=7, color="#a0aec0")
    ax.axhline(y=0, color="#4a5568", linewidth=0.8)
    ax.set_ylabel("Signed attribution", fontsize=9)

    # Annotate
    info = f"Net: {result.net_correction:.4f}  |  Negative tokens: {result.n_negative_tokens}/{result.seq_len}"
    ax.text(0.99, 0.95, info, transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#a0aec0",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a202c", alpha=0.8))

    return _fig_to_base64(fig)


def plot_stress_per_token(result) -> str:
    """Per-token stress score bar chart."""
    fig, ax = plt.subplots(figsize=(max(8, len(result.tokens) * 0.55), 3.5))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Focused Stress Score (per token, signal layers)")

    vals = result.per_token_stress
    ax.bar(range(len(result.tokens)), vals, color="#4a6fa5", width=0.75,
           edgecolor="#1a202c", linewidth=0.5)
    ax.set_xticks(range(len(result.tokens)))
    ax.set_xticklabels(result.tokens, rotation=50, ha="right",
                       fontsize=7, color="#a0aec0")
    ax.set_ylabel("Stress score", fontsize=9)

    ax.text(0.99, 0.95, f"Mean: {result.stress_score:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#a0aec0",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a202c", alpha=0.8))

    return _fig_to_base64(fig)


def plot_amplitude_trajectory(result) -> str:
    """Full amplitude trajectory across all sublayers."""
    if not result.amplitude_normalized:
        return ""

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Normalized Amplitude Trajectory (all sublayers)")

    traj = result.amplitude_normalized
    ax.plot(traj, color="#7db8c9", linewidth=1.2, alpha=0.9)
    ax.fill_between(range(len(traj)), traj, alpha=0.15, color="#7db8c9")
    ax.set_xlabel("Sublayer index")
    ax.set_ylabel("||ΔW · h|| / ||ΔW||_F")

    n = len(traj)
    for b, label in [(n // 3, "Early→Mid"), (2 * n // 3, "Mid→Late")]:
        ax.axvline(x=b, color="#4a5568", linestyle="--", alpha=0.6)
        ax.text(b + 1, ax.get_ylim()[1] * 0.92, label, fontsize=7, color="#718096")

    return _fig_to_base64(fig)


def plot_heatmap(result) -> str:
    """Token × layer sensitivity heatmap."""
    if result.heatmap is None or result.heatmap.size == 0:
        return ""

    fig, ax = plt.subplots(figsize=(max(8, len(result.tokens) * 0.5), 5))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Per-Token × Per-Layer Sensitivity")

    im = ax.imshow(result.heatmap, aspect="auto", cmap=TASM_CMAP,
                   interpolation="nearest")
    ax.set_xticks(range(len(result.tokens)))
    ax.set_xticklabels(result.tokens, rotation=50, ha="right",
                       fontsize=6, color="#a0aec0")

    n_sub = result.heatmap.shape[0]
    ax.set_yticks([0, n_sub // 3, 2 * n_sub // 3, n_sub - 1])
    ax.set_yticklabels(["Early", "→ Mid", "→ Late", "Final"], fontsize=8)
    ax.set_ylabel("Sublayer depth")

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.ax.tick_params(colors="#a0aec0", labelsize=7)
    cb.set_label("Normalized sensitivity", color="#a0aec0", fontsize=8)

    return _fig_to_base64(fig)


def plot_distribution_metrics(result) -> str:
    """Summary card of distribution metrics for a single prompt."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.5))
    fig.patch.set_facecolor("#0d1117")

    metrics = [
        ("Entropy", result.entropy, "#4a6fa5"),
        ("Gini", result.gini, "#7b2d8b"),
        ("Boundary share", result.top2_share, "#2d936c"),
        ("Interior share", result.middle_share, "#c44536"),
    ]

    for ax, (label, val, color) in zip(axes, metrics):
        _style_ax(ax)
        ax.barh([0], [val], color=color, height=0.5, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_title(label, fontsize=9, color="#e2e8f0", fontweight="bold")
        ax.text(val + 0.02, 0, f"{val:.3f}", va="center", fontsize=9,
                color="#e2e8f0", fontweight="bold")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_batch_summary(agg: dict) -> str:
    """Batch analysis summary: category box plots for key metrics."""
    cats_order = ["benign", "mild", "harmful", "jailbreak"]
    available = [c for c in cats_order if c in agg["categories"]]
    if not available:
        return ""

    plot_metrics = ["stress_score", "entropy", "top2_share",
                    "middle_share", "net_correction"]
    titles = ["Stress Score", "Entropy", "Boundary Share",
              "Interior Share", "Net Correction"]

    fig, axes = plt.subplots(1, len(plot_metrics),
                             figsize=(4 * len(plot_metrics), 4))
    fig.patch.set_facecolor("#0d1117")

    for ax, metric, title in zip(axes, plot_metrics, titles):
        _style_ax(ax, title)
        box_data = []
        box_labels = []
        box_colors = []

        for cat in available:
            summary = agg["categories"][cat]
            if metric in summary["metrics"]:
                m = summary["metrics"][metric]
                # Reconstruct approximate data from CI for visualization
                est = m["estimate"]
                n = m["n"]
                spread = (m["ci_high"] - m["ci_low"]) / 4
                if n > 1 and spread > 0:
                    rng = np.random.default_rng(hash(cat + metric) % 2**32)
                    vals = rng.normal(est, spread, size=max(n, 5))
                else:
                    vals = [est]
                box_data.append(vals)
                box_labels.append(cat.title())
                box_colors.append(CAT_COLORS.get(cat, "#888"))

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                           widths=0.5)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor("#a0aec0")
            for element in ["whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set_color("#a0aec0")
            for item in bp["fliers"]:
                item.set_markeredgecolor("#a0aec0")
            ax.tick_params(axis="x", labelsize=7, colors="#a0aec0")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_separability(agg: dict) -> str:
    """Effect sizes with bootstrap CIs for each metric."""
    sep = agg.get("separability", {})
    if not sep:
        return ""

    raw_metrics = [m for m in sep if not m.endswith("_ln")]

    fig, ax = plt.subplots(figsize=(8, max(3, len(raw_metrics) * 0.7)))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Effect Size: Benign vs Harmful (Cohen's d, 95% CI)")

    y_positions = []
    labels = []
    for i, metric in enumerate(raw_metrics):
        es = sep[metric]["effect_size"]
        y = i
        y_positions.append(y)
        labels.append(metric.replace("_", " ").title())

        color = "#2d936c" if es["estimate"] > 0.8 else (
            "#e0a458" if es["estimate"] > 0.5 else "#c44536")

        ax.barh(y, es["estimate"], color=color, height=0.5, alpha=0.8)
        ax.plot([es["ci_low"], es["ci_high"]], [y, y],
                color="#e2e8f0", linewidth=2, solid_capstyle="round")
        ax.plot([es["ci_low"]], [y], "|", color="#e2e8f0", markersize=10)
        ax.plot([es["ci_high"]], [y], "|", color="#e2e8f0", markersize=10)

        acc = sep[metric]["threshold"]["accuracy"]
        ax.text(max(es["ci_high"], es["estimate"]) + 0.1, y,
                f"d={es['estimate']:.2f}  acc={acc:.0%}",
                va="center", fontsize=8, color="#a0aec0")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0.8, color="#2d936c", linestyle=":", alpha=0.5, label="Large effect")
    ax.axvline(x=0.5, color="#e0a458", linestyle=":", alpha=0.5, label="Medium effect")
    ax.set_xlabel("Cohen's d")
    ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    return _fig_to_base64(fig)
