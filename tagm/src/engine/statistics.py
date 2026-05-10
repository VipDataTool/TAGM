"""
Statistics: bootstrap confidence intervals, effect sizes, and batch aggregation.
Designed for small-n situations where parametric assumptions are dubious.
Extended with LTP summary statistics (M, C, V).
"""

import numpy as np
from typing import List, Dict
from scipy import stats as sp_stats
import math
from src.engine import config as engine_config


def _safe_float(v):
    """Convert to float, replacing NaN/Inf with 0."""
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def _clean_values(vals):
    """Filter out NaN/Inf from a list of floats."""
    return [v for v in vals if not (math.isnan(v) or math.isinf(v))]


def cohens_d(group_a: list, group_b: list) -> float:
    """Cohen's d effect size with pooled standard deviation."""
    a, b = np.array(_clean_values(group_a)), np.array(_clean_values(group_b))
    if len(a) < engine_config.get("min_samples_d") or len(b) < engine_config.get("min_samples_d"):
        return 0.0
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return _safe_float(abs(a.mean() - b.mean()) / pooled_std)


def bootstrap_ci(values: list, n_boot: int = None,
                 ci: float = None, statistic=np.mean) -> dict:
    """
    Bootstrap confidence interval for a statistic.
    Returns {"estimate": float, "ci_low": float, "ci_high": float, "n": int}
    """
    if n_boot is None:
        n_boot = engine_config.get("n_bootstrap")
    if ci is None:
        ci = engine_config.get("ci_level")
    values = np.array(_clean_values(list(values)))
    if len(values) < 1:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    if len(values) < 2:
        val = _safe_float(statistic(values))
        return {"estimate": val, "ci_low": val, "ci_high": val, "n": len(values)}

    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(statistic(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    return {
        "estimate": _safe_float(statistic(values)),
        "ci_low": _safe_float(np.percentile(boot_stats, alpha * 100)),
        "ci_high": _safe_float(np.percentile(boot_stats, (1 - alpha) * 100)),
        "n": len(values),
    }


def bootstrap_effect_size(group_a: list, group_b: list,
                          n_boot: int = None, ci: float = None) -> dict:
    """Bootstrap CI for Cohen's d."""
    if n_boot is None:
        n_boot = engine_config.get("n_bootstrap")
    if ci is None:
        ci = engine_config.get("ci_level")
    a, b = np.array(_clean_values(group_a)), np.array(_clean_values(group_b))
    min_d = engine_config.get("min_samples_d")
    if len(a) < min_d or len(b) < min_d:
        d = cohens_d(list(a), list(b))
        return {"estimate": _safe_float(d), "ci_low": _safe_float(d),
                "ci_high": _safe_float(d), "n_a": len(a), "n_b": len(b)}

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
        "estimate": _safe_float(cohens_d(list(a), list(b))),
        "ci_low": _safe_float(np.percentile(boot_ds, alpha * 100)),
        "ci_high": _safe_float(np.percentile(boot_ds, (1 - alpha) * 100)),
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

    for t in np.linspace(min(vals), max(vals), engine_config.get("threshold_steps")):
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


# ─── Metric extraction registry ──────────────────────────────────

# All metrics tracked by the statistics system.
# Each entry: (stat_key, extractor) where extractor(PromptResult) -> float | None.

def _get_ltp_val_generic(r, ltp_key):
    """Extract an LTP scalar from a PromptResult (handles both live and reconstituted)."""
    if hasattr(r, 'ltp') and r.ltp is not None:
        obj = r.ltp
        if isinstance(obj, dict):
            return obj.get(ltp_key)
        return getattr(obj, ltp_key, None)
    if hasattr(r, ltp_key):
        return getattr(r, ltp_key, None)
    return None


def _get_sfd_val_generic(r, sfd_key):
    """Extract an SFD scalar from a PromptResult."""
    if hasattr(r, 'sfd') and r.sfd is not None:
        obj = r.sfd
        if isinstance(obj, dict):
            return obj.get(sfd_key)
        return getattr(obj, sfd_key, None)
    return None


def _get_rd_val(r, rd_key):
    """Extract a rank-displacement scalar from a PromptResult."""
    rd = getattr(r, 'rank_displacement', None)
    if isinstance(rd, dict):
        return rd.get(rd_key)
    return None


# Each entry: (stat_key, extractor) where extractor(PromptResult) -> float | None.
ALL_METRICS_REGISTRY = [
    # ASM
    ("stress_score",   lambda r: getattr(r, 'stress_score', None)),
    ("entropy",        lambda r: getattr(r, 'entropy', None)),
    ("top2_share",     lambda r: getattr(r, 'top2_share', None)),
    ("middle_share",   lambda r: getattr(r, 'middle_share', None)),
    ("interior_cv",    lambda r: getattr(r, 'interior_cv', None)),
    ("net_correction", lambda r: getattr(r, 'net_correction', None)),
    # KL
    ("kl_divergence",  lambda r: getattr(r, 'kl_divergence', None)),
    # LTP (ltp_mean_L excluded: constant 1.0, zero variance)
    ("ltp_mean_M",     lambda r: _get_ltp_val_generic(r, 'mean_M')),
    ("ltp_mean_V",     lambda r: _get_ltp_val_generic(r, 'mean_V')),
    ("ltp_n_dir",      lambda r: _get_ltp_val_generic(r, 'n_directional')),
    # SFD
    ("sfd_density_mean",  lambda r: _get_sfd_val_generic(r, 'density_mean')),
    # Rank displacement
    ("rank_displacement_tau",     lambda r: _get_rd_val(r, 'mean_tau')),
    ("rank_displacement_overlap", lambda r: _get_rd_val(r, 'mean_overlap')),
    ("rd_replacement",            lambda r: _get_rd_val(r, 'mean_replacement')),
    ("rd_disp_per_tok",           lambda r: _get_rd_val(r, 'mean_disp_per_token')),
]


# ─── Length correlation diagnostics ──────────────────────────────

def length_correlations(results: list) -> dict:
    """Compute Pearson r between each metric and seq_len.

    This is a diagnostic: it validates that length-invariant metrics
    are actually indifferent to sequence length.  Any significant
    correlation after the formula fixes indicates genuine behavioral
    signal, not a measurement defect.

    Returns:
        dict mapping stat_key -> {"r": float, "r_sq": float, "n": int}
    """
    lengths = np.array([float(r.seq_len) for r in results])
    out = {}

    for stat_key, extractor in ALL_METRICS_REGISTRY:
        raw = [extractor(r) for r in results]
        valid = [(i, float(v)) for i, v in enumerate(raw)
                 if v is not None and not (math.isnan(float(v)) or math.isinf(float(v)))]
        if len(valid) < engine_config.get("min_valid_separability"):
            continue

        valid_y = np.array([v for _, v in valid])
        if valid_y.std() < 1e-12:
            continue

        valid_x = np.array([lengths[i] for i, _ in valid])
        if valid_x.std() == 0:
            continue

        r_val = float(np.corrcoef(valid_x, valid_y)[0, 1])
        out[stat_key] = {
            "r": _safe_float(r_val),
            "r_sq": _safe_float(r_val ** 2),
            "n": len(valid),
        }

    return out


def aggregate_batch(results: list) -> dict:
    """Aggregate a batch of PromptResult objects into summary statistics.

    Groups by category, computes bootstrap CIs for every metric,
    effect sizes between category pairs, and length correlation
    diagnostics.
    """
    # ── Build result-index lookup ─────────────────────────────────
    result_indices: Dict[str, List[int]] = {}
    for i, r in enumerate(results):
        cat = r.category or "unknown"
        if cat not in result_indices:
            result_indices[cat] = []
        result_indices[cat].append(i)

    # ── Per-category summary ──────────────────────────────────────
    cat_summaries = {}

    for cat, indices in result_indices.items():
        group = [results[i] for i in indices]
        summary = {"n": len(group), "metrics": {},
                   "mean_seq_len": float(np.mean([r.seq_len for r in group]))}

        n_with_neg = sum(1 for r in group if r.has_negative_tokens)
        summary["negative_token_rate"] = n_with_neg / len(group) if group else 0

        for stat_key, extractor in ALL_METRICS_REGISTRY:
            vals = [float(v) for r in group
                    for v in [extractor(r)] if v is not None
                    and not (math.isnan(float(v)) or math.isinf(float(v)))]
            if vals:
                summary["metrics"][stat_key] = bootstrap_ci(vals)

        cat_summaries[cat] = summary

    # ── Pairwise separability ─────────────────────────────────────
    benign_cats = {"benign", "baseline", "mild", "user_baseline"}
    target_cats = {"harmful", "adversarial", "jailbreak", "dual-use"}

    benign_idx = [i for i, r in enumerate(results) if r.category in benign_cats]

    separability = {}
    for target_cat in target_cats:
        target_idx = result_indices.get(target_cat, [])
        if not benign_idx or not target_idx:
            continue

        pair_key = f"benign_vs_{target_cat}"
        separability[pair_key] = {}

        for stat_key, extractor in ALL_METRICS_REGISTRY:
            b_vals = [float(v) for i in benign_idx
                      for v in [extractor(results[i])] if v is not None
                      and not (math.isnan(float(v)) or math.isinf(float(v)))]
            t_vals = [float(v) for i in target_idx
                      for v in [extractor(results[i])] if v is not None
                      and not (math.isnan(float(v)) or math.isinf(float(v)))]

            if b_vals and t_vals:
                separability[pair_key][stat_key] = {
                    "effect_size": bootstrap_effect_size(b_vals, t_vals),
                    "threshold": best_threshold(b_vals, t_vals),
                    "benign_mean": float(np.mean(b_vals)),
                    "target_mean": float(np.mean(t_vals)),
                }

    # ── Legacy flat separability (benign-ish vs harmful-ish pooled) ──
    harmful_cats_all = {"harmful", "jailbreak", "adversarial"}
    harmful_idx = [i for i, r in enumerate(results) if r.category in harmful_cats_all]

    separability_legacy = {}
    if benign_idx and harmful_idx:
        for stat_key, extractor in ALL_METRICS_REGISTRY:
            b_vals = [float(v) for i in benign_idx
                      for v in [extractor(results[i])] if v is not None
                      and not (math.isnan(float(v)) or math.isinf(float(v)))]
            h_vals = [float(v) for i in harmful_idx
                      for v in [extractor(results[i])] if v is not None
                      and not (math.isnan(float(v)) or math.isinf(float(v)))]
            if b_vals and h_vals:
                separability_legacy[stat_key] = {
                    "effect_size": bootstrap_effect_size(b_vals, h_vals),
                    "threshold": best_threshold(b_vals, h_vals),
                    "benign_mean": float(np.mean(b_vals)),
                    "harmful_mean": float(np.mean(h_vals)),
                }

    # ── Correlations ──────────────────────────────────────────────
    correlations = {}
    corr_pairs = [
        (float(r.stress_score), float(r.kl_divergence))
        for r in results
        if r.kl_divergence is not None
        and not (math.isnan(float(r.stress_score)) or math.isinf(float(r.stress_score)))
        and not (math.isnan(float(r.kl_divergence)) or math.isinf(float(r.kl_divergence)))
    ]
    if len(corr_pairs) >= 3:
        ss_vals, kl_vals = zip(*corr_pairs)
        r_val, p_val = sp_stats.pearsonr(ss_vals, kl_vals)
        correlations["stress_vs_kl"] = {"r": _safe_float(r_val), "p": _safe_float(p_val),
                                         "n": len(corr_pairs)}

    # ── Length correlation diagnostics ────────────────────────────
    len_corr = length_correlations(results)

    return {
        "n_total": len(results),
        "categories": cat_summaries,
        "separability": separability_legacy,
        "separability_pairwise": separability,
        "length_correlations": len_corr,
        "correlations": correlations,
    }
