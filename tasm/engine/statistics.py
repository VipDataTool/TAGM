"""
Statistics: bootstrap confidence intervals, effect sizes, and batch aggregation.
Designed for small-n situations where parametric assumptions are dubious.
"""

import numpy as np
from typing import List, Dict, Optional
from scipy import stats as sp_stats


def cohens_d(group_a: list, group_b: list) -> float:
    """Cohen's d effect size with pooled standard deviation."""
    a, b = np.array(group_a), np.array(group_b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return float(abs(a.mean() - b.mean()) / pooled_std)


def bootstrap_ci(values: list, n_boot: int = 5000,
                 ci: float = 0.95, statistic=np.mean) -> dict:
    """
    Bootstrap confidence interval for a statistic.
    Returns {"estimate": float, "ci_low": float, "ci_high": float, "n": int}
    """
    values = np.array(values)
    if len(values) < 2:
        val = float(statistic(values)) if len(values) > 0 else float("nan")
        return {"estimate": val, "ci_low": val, "ci_high": val, "n": len(values)}

    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(statistic(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    return {
        "estimate": float(statistic(values)),
        "ci_low": float(np.percentile(boot_stats, alpha * 100)),
        "ci_high": float(np.percentile(boot_stats, (1 - alpha) * 100)),
        "n": len(values),
    }


def bootstrap_effect_size(group_a: list, group_b: list,
                          n_boot: int = 5000, ci: float = 0.95) -> dict:
    """Bootstrap CI for Cohen's d."""
    a, b = np.array(group_a), np.array(group_b)
    if len(a) < 2 or len(b) < 2:
        d = cohens_d(group_a, group_b)
        return {"estimate": d, "ci_low": d, "ci_high": d,
                "n_a": len(a), "n_b": len(b)}

    rng = np.random.default_rng(42)
    boot_ds = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        ps = np.sqrt((sa.std(ddof=1)**2 + sb.std(ddof=1)**2) / 2)
        d = abs(sa.mean() - sb.mean()) / ps if ps > 0 else 0
        boot_ds.append(d)

    boot_ds = np.array(boot_ds)
    alpha = (1 - ci) / 2
    return {
        "estimate": cohens_d(group_a, group_b),
        "ci_low": float(np.percentile(boot_ds, alpha * 100)),
        "ci_high": float(np.percentile(boot_ds, (1 - alpha) * 100)),
        "n_a": len(a),
        "n_b": len(b),
    }


def best_threshold(benign_vals: list, harmful_vals: list) -> dict:
    """Find optimal classification threshold by brute-force sweep."""
    all_vals = [(v, "b") for v in benign_vals] + [(v, "h") for v in harmful_vals]
    if not all_vals:
        return {"threshold": 0, "accuracy": 0, "direction": ">="}

    vals = [v for v, _ in all_vals]
    best_acc = 0
    best_t = 0
    best_dir = ">="

    for t in np.linspace(min(vals), max(vals), 500):
        # harmful >= threshold
        c_high = sum(1 for v, c in all_vals
                     if (v >= t and c == "h") or (v < t and c == "b"))
        # harmful < threshold
        c_low = sum(1 for v, c in all_vals
                    if (v < t and c == "h") or (v >= t and c == "b"))
        if c_high >= c_low:
            acc = c_high / len(all_vals)
            d = ">="
        else:
            acc = c_low / len(all_vals)
            d = "<"
        if acc > best_acc:
            best_acc = acc
            best_t = t
            best_dir = d

    return {"threshold": float(best_t), "accuracy": float(best_acc),
            "direction": best_dir}


def aggregate_batch(results: list) -> dict:
    """
    Aggregate a batch of PromptResult objects into summary statistics.
    Groups by category and computes effect sizes between benign and harmful.
    """
    # Group by category
    groups: Dict[str, list] = {}
    for r in results:
        cat = r.category or "unknown"
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(r)

    # Per-category summary
    cat_summaries = {}
    metrics = ["stress_score", "entropy", "gini", "top2_share",
               "middle_share", "interior_cv", "net_correction"]

    for cat, group in groups.items():
        summary = {"n": len(group), "metrics": {}}
        for m in metrics:
            vals = [float(getattr(r, m)) for r in group if getattr(r, m) is not None]
            if vals:
                summary["metrics"][m] = bootstrap_ci(vals)
            # Length-normalized version
            ln_attr = m + "_ln"
            ln_vals = [float(getattr(r, ln_attr)) for r in group
                       if hasattr(r, ln_attr) and getattr(r, ln_attr) is not None]
            if ln_vals:
                summary["metrics"][m + "_ln"] = bootstrap_ci(ln_vals)

        # KL if available
        kl_vals = [float(r.kl_divergence) for r in group if r.kl_divergence is not None]
        if kl_vals:
            summary["metrics"]["kl_divergence"] = bootstrap_ci(kl_vals)

        # Mean seq length
        summary["mean_seq_len"] = float(np.mean([r.seq_len for r in group]))

        # Negative token frequency
        n_with_neg = sum(1 for r in group if r.has_negative_tokens)
        summary["negative_token_rate"] = n_with_neg / len(group) if group else 0

        cat_summaries[cat] = summary

    # Effect sizes: benign-ish vs harmful-ish
    benign_cats = {"benign", "baseline", "mild", "user_baseline"}
    harmful_cats = {"harmful", "jailbreak", "adversarial"}

    benign_results = [r for r in results if r.category in benign_cats]
    harmful_results = [r for r in results if r.category in harmful_cats]

    separability = {}
    if benign_results and harmful_results:
        for m in metrics:
            b_vals = [float(getattr(r, m)) for r in benign_results if getattr(r, m) is not None]
            h_vals = [float(getattr(r, m)) for r in harmful_results if getattr(r, m) is not None]
            if b_vals and h_vals:
                separability[m] = {
                    "effect_size": bootstrap_effect_size(b_vals, h_vals),
                    "threshold": best_threshold(b_vals, h_vals),
                    "benign_mean": float(np.mean(b_vals)),
                    "harmful_mean": float(np.mean(h_vals)),
                }

                # Length-normalized version
                ln_attr = m + "_ln"
                b_ln = [float(getattr(r, ln_attr)) for r in benign_results
                        if hasattr(r, ln_attr) and getattr(r, ln_attr) is not None]
                h_ln = [float(getattr(r, ln_attr)) for r in harmful_results
                        if hasattr(r, ln_attr) and getattr(r, ln_attr) is not None]
                if b_ln and h_ln:
                    separability[m + "_ln"] = {
                        "effect_size": bootstrap_effect_size(b_ln, h_ln),
                        "threshold": best_threshold(b_ln, h_ln),
                    }

    # Correlation between stress_score and kl_divergence
    correlations = {}
    ss_vals = [float(r.stress_score) for r in results if r.kl_divergence is not None]
    kl_vals = [float(r.kl_divergence) for r in results if r.kl_divergence is not None]
    if len(ss_vals) >= 3:
        r_val, p_val = sp_stats.pearsonr(ss_vals, kl_vals)
        correlations["stress_vs_kl"] = {"r": float(r_val), "p": float(p_val),
                                         "n": len(ss_vals)}

    return {
        "n_total": len(results),
        "categories": cat_summaries,
        "separability": separability,
        "correlations": correlations,
    }
