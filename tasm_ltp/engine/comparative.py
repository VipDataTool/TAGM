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

CAT_COLORS = {
    "benign": "#2d936c", "baseline": "#2d936c", "user_baseline": "#78b89a",
    "mild": "#e0a458", "harmful": "#c44536",
    "jailbreak": "#7b2d8b", "adversarial": "#7b2d8b", "unknown": "#888888",
}

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _style_ax(ax, title=""):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#a0aec0", labelsize=10)
    ax.xaxis.label.set_color("#a0aec0")
    ax.yaxis.label.set_color("#a0aec0")
    ax.title.set_color("#e2e8f0")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_color("#2d3748")
    ax.grid(True, alpha=0.15, color="#4a5568")

def _wrap(text, width=22):
    return "\n".join(textwrap.wrap(str(text), width))


def _cat_legend(ax, results):
    used = set()
    handles = []
    for r in results:
        c = r.get("category", "unknown")
        if c not in used:
            used.add(c)
            handles.append(Patch(facecolor=CAT_COLORS.get(c, "#888"), label=c.title()))
    ax.legend(handles=handles, fontsize=10, loc="best")


def plot_trajectory_overlay(results: list) -> str:
    has_traj = [r for r in results if r.get("amplitude_normalized")]
    if not has_traj:
        return ""

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Normalized Amplitude Trajectories (all prompts)")

    for r in has_traj:
        color = CAT_COLORS.get(r.get("category", ""), "#888")
        ax.plot(r["amplitude_normalized"], color=color, alpha=0.5, linewidth=1.0)

    n = len(has_traj[0]["amplitude_normalized"])
    for b, t in [(n // 3, "Early -> Mid"), (2 * n // 3, "Mid -> Late")]:
        ax.axvline(x=b, color="#4a5568", linestyle="--", alpha=0.5)
        ax.text(b + 1, ax.get_ylim()[1] * 0.92, t, fontsize=10, color="#718096")

    ax.set_xlabel("Sublayer index", fontsize=11)
    ax.set_ylabel("Normalized sensitivity", fontsize=11)
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
    fig.patch.set_facecolor("#0d1117")
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

    ax.axhline(y=0, color="#4a5568", linewidth=1)
    ax.set_xlabel("Sublayer index", fontsize=11)
    ax.set_ylabel("Delta normalized sensitivity", fontsize=11)
    ax.legend(fontsize=10)

    n = len(baseline)
    for b, t in [(n // 3, "Early -> Mid"), (2 * n // 3, "Mid -> Late")]:
        ax.axvline(x=b, color="#4a5568", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_discriminative_sublayers(results: list, top_n: int = 15) -> str:
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
    fig.patch.set_facecolor("#0d1117")
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
        colors.append("#7b2d8b" if region == "Middle" else (
            "#4a6fa5" if region == "Late" else "#2d936c"))

    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.65, edgecolor="#1a202c")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, color="#a0aec0")
    ax.set_xlabel("Delta sensitivity", fontsize=11)
    ax.invert_yaxis()

    handles = [Patch(facecolor="#2d936c", label="Early"),
               Patch(facecolor="#7b2d8b", label="Middle"),
               Patch(facecolor="#4a6fa5", label="Late")]
    ax.legend(handles=handles, fontsize=10, loc="lower right")

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
        ("gini", "net_correction", "Gini", "Net Correction"),
        ("entropy", "stress_score", "Entropy", "Stress Score"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#0d1117")

    for ax, (x_key, y_key, x_label, y_label) in zip(axes.flat, scatter_defs):
        _style_ax(ax, f"{x_label} vs {y_label}")

        for r in results:
            xv = r.get(x_key)
            yv = r.get(y_key)
            if xv is None or yv is None:
                continue
            color = CAT_COLORS.get(r.get("category", ""), "#888")
            ax.scatter(xv, yv, c=color, s=60, alpha=0.8,
                       edgecolors="#1a202c", linewidth=0.5)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)

    _cat_legend(axes[0, 0], results)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_behavioral_comparison(results: list) -> str:
    has_both = [r for r in results
                if r.get("instruct_topk") and r.get("base_topk")]
    if not has_both:
        return ""

    fig, ax = plt.subplots(figsize=(max(11, len(has_both) * 1.0), 5.5))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Behavioral Divergence: Instruct vs Base\nTop-1 Probability")

    x = np.arange(len(has_both))
    w = 0.35

    inst_probs = [r["instruct_topk"][0][1] for r in has_both]
    base_probs = [r["base_topk"][0][1] for r in has_both]
    labels = [_wrap(r["prompt"][:35], 18) for r in has_both]

    ax.bar(x - w/2, inst_probs, w, label="Instruct", color="#4a6fa5", alpha=0.85)
    ax.bar(x + w/2, base_probs, w, label="Base", color="#c44536", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, color="#a0aec0")
    ax.set_ylabel("Top-1 probability", fontsize=11)
    ax.legend(fontsize=11)

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
    fig.patch.set_facecolor("#0d1117")

    ax = axes[0]
    _style_ax(ax, "Proof 1 Exactness Errors\n(log scale)")
    ax.hist(np.log10(np.array(errors) + 1e-15), bins=30, color="#4a6fa5", alpha=0.8,
            edgecolor="#1a202c")
    ax.set_xlabel("log10(error)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

    ax = axes[1]
    _style_ax(ax, f"Exactness Rate: {exact_rate:.1%}")
    bars = ax.bar(["Exact\n(< 1e-4)", "Inexact"], [exact_rate, 1 - exact_rate],
                  color=["#2d936c", "#c44536"], alpha=0.85, width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction", fontsize=11)
    for bar, val in zip(bars, [exact_rate, 1 - exact_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", fontsize=11, color="#e2e8f0")

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

    ltp_keys = [("mean_M", "Offset\nMagnitude"), ("mean_C", "Offset\nConsistency"),
                ("mean_V", "Offset\nVariance"), ("mean_L", "Lateral\nCoverage")]

    fig, axes = plt.subplots(1, len(ltp_keys), figsize=(4.5 * len(ltp_keys), 5))
    fig.patch.set_facecolor("#0d1117")

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
                patch.set_edgecolor("#a0aec0")
            for element in ["whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set_color("#a0aec0")
            for item in bp["fliers"]:
                item.set_markeredgecolor("#a0aec0")
            ax.tick_params(axis="x", labelsize=10, colors="#a0aec0")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_ltp_m_vs_stress(results: list) -> str:
    """Scatter: LTP offset magnitude vs ASM stress score, colored by category."""
    has_ltp = [r for r in results if r.get("ltp") and r["ltp"].get("mean_M") is not None]
    if len(has_ltp) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Offset Magnitude (M) vs Stress Score")

    for r in has_ltp:
        color = CAT_COLORS.get(r.get("category", ""), "#888")
        ax.scatter(r.get("stress_score", 0), r["ltp"]["mean_M"],
                   c=color, s=60, alpha=0.8, edgecolors="#1a202c", linewidth=0.5)

    ax.set_xlabel("Stress Score (ASM)", fontsize=11)
    ax.set_ylabel("Offset Magnitude M (LTP)", fontsize=11)
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
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax, "Profile Shape Distribution by Category")

    shapes = ["steep", "flat", "inverted"]
    shape_colors = {"steep": "#e0a458", "flat": "#4a6fa5", "inverted": "#c44536"}
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
    ax.set_xticklabels([c.title() for c in available], fontsize=11)
    ax.set_ylabel("Fraction of tokens", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_all_comparative(results: list) -> dict:
    plots = {}
    plots["trajectory_overlay"] = plot_trajectory_overlay(results)
    plots["difference_from_benign"] = plot_difference_from_benign(results)
    plots["discriminative_sublayers"] = plot_discriminative_sublayers(results)
    plots["metric_scatters"] = plot_metric_scatters(results)
    plots["behavioral_comparison"] = plot_behavioral_comparison(results)
    plots["proof1_summary"] = plot_proof1_summary(results)
    # LTP comparative plots
    plots["ltp_category_comparison"] = plot_ltp_category_comparison(results)
    plots["ltp_m_vs_stress"] = plot_ltp_m_vs_stress(results)
    plots["ltp_profile_shapes"] = plot_ltp_profile_shape_distribution(results)
    return {k: v for k, v in plots.items() if v}
