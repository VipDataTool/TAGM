"""
Statistics: bootstrap confidence intervals, effect sizes, and batch aggregation.
Designed for small-n situations where parametric assumptions are dubious.
Extended with LTP summary statistics (M, C, V, L).
"""

import numpy as np
from typing import List, Dict, Optional
from scipy import stats as sp_stats
import math


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
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return _safe_float(abs(a.mean() - b.mean()) / pooled_std)


def bootstrap_ci(values: list, n_boot: int = 5000,
                 ci: float = 0.95, statistic=np.mean) -> dict:
    """
    Bootstrap confidence interval for a statistic.
    Returns {"estimate": float, "ci_low": float, "ci_high": float, "n": int}
    """
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
                          n_boot: int = 5000, ci: float = 0.95) -> dict:
    """Bootstrap CI for Cohen's d."""
    a, b = np.array(_clean_values(group_a)), np.array(_clean_values(group_b))
    if len(a) < 2 or len(b) < 2:
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


# ─── Metric extraction helpers ────────────────────────────────────

# Every metric we want to residualize: (stat_key, extractor_function)
# Extractors take a PromptResult and return float | None.

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


# Registry of all metrics eligible for length residualization.
# Each entry: (stat_key, extractor) where extractor(r) -> float | None.
ALL_METRICS_REGISTRY = [
    # ASM
    ("stress_score",   lambda r: getattr(r, 'stress_score', None)),
    ("entropy",        lambda r: getattr(r, 'entropy', None)),
    ("gini",           lambda r: getattr(r, 'gini', None)),
    ("top2_share",     lambda r: getattr(r, 'top2_share', None)),
    ("middle_share",   lambda r: getattr(r, 'middle_share', None)),
    ("interior_cv",    lambda r: getattr(r, 'interior_cv', None)),
    ("net_correction", lambda r: getattr(r, 'net_correction', None)),
    # KL
    ("kl_divergence",  lambda r: getattr(r, 'kl_divergence', None)),
    # LTP
    ("ltp_mean_M",     lambda r: _get_ltp_val_generic(r, 'mean_M')),
    ("ltp_mean_C",     lambda r: _get_ltp_val_generic(r, 'mean_C')),
    ("ltp_mean_V",     lambda r: _get_ltp_val_generic(r, 'mean_V')),
    ("ltp_mean_L",     lambda r: _get_ltp_val_generic(r, 'mean_L')),
    ("ltp_n_dir",      lambda r: _get_ltp_val_generic(r, 'n_directional')),
    # SFD (match existing key names used in category summaries and frontend)
    ("sfd_density_mean",  lambda r: _get_sfd_val_generic(r, 'density_mean')),
    ("sfd_entropy_mean",  lambda r: _get_sfd_val_generic(r, 'entropy_mean')),
    ("sfd_energy_mean",   lambda r: _get_sfd_val_generic(r, 'energy_mean')),
    # Rank displacement (match existing key names)
    ("rank_displacement_tau",     lambda r: _get_rd_val(r, 'mean_tau')),
    ("rank_displacement_overlap", lambda r: _get_rd_val(r, 'mean_overlap')),
    ("rd_replacement",            lambda r: _get_rd_val(r, 'mean_replacement')),
    ("rd_disp_per_tok",           lambda r: _get_rd_val(r, 'mean_disp_per_token')),
]


# ─── Regression-based length residualization ──────────────────────

def length_residualize(results: list, fit_categories: Optional[set] = None) -> dict:
    """Fit OLS regression of each metric against seq_len, return residuals.

    Args:
        results: list of PromptResult objects.
        fit_categories: if provided, fit the regression only on results in these
            categories (e.g. {"benign"} to use the benign trendline).  Residuals
            are still computed for ALL results.  If None, fits on the full batch.

    Returns:
        dict mapping stat_key -> {
            "residuals": list of float|None (one per result, None if metric missing),
            "slope": float,
            "intercept": float,
            "r": float (Pearson r of metric vs length),
            "r_sq": float,
        }
    """
    lengths = np.array([float(r.seq_len) for r in results])
    out = {}

    for stat_key, extractor in ALL_METRICS_REGISTRY:
        # Extract values (None where missing)
        raw = [extractor(r) for r in results]

        # Build mask of valid values
        valid = [(i, float(v)) for i, v in enumerate(raw)
                 if v is not None and not (math.isnan(float(v)) or math.isinf(float(v)))]
        if len(valid) < 5:
            continue

        # Skip degenerate metrics (constant or near-constant values)
        all_valid_y_check = np.array([v for _, v in valid])
        if all_valid_y_check.std() < 1e-12:
            continue

        # Determine fit subset
        if fit_categories is not None:
            fit_idx = [i for i, v in valid
                       if (results[i].category or "unknown") in fit_categories]
        else:
            fit_idx = [i for i, _ in valid]

        if len(fit_idx) < 3:
            # Not enough data in fit subset, fall back to full
            fit_idx = [i for i, _ in valid]

        valid_map = {i: v for i, v in valid}
        fit_x = np.array([lengths[i] for i in fit_idx])
        fit_y = np.array([valid_map[i] for i in fit_idx])

        # OLS: y = slope * x + intercept
        if fit_x.std() == 0 or fit_y.std() == 0:
            continue
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        predicted_all = slope * lengths + intercept

        # Pearson r on full valid set
        all_valid_x = np.array([lengths[i] for i, _ in valid])
        all_valid_y = np.array([v for _, v in valid])
        if len(all_valid_x) >= 3 and all_valid_x.std() > 0 and all_valid_y.std() > 0:
            r_val = float(np.corrcoef(all_valid_x, all_valid_y)[0, 1])
        else:
            r_val = 0.0

        # Compute residuals for all results
        residuals = [None] * len(results)
        for i, v in valid:
            residuals[i] = float(v) - float(predicted_all[i])

        out[stat_key] = {
            "residuals": residuals,
            "slope": _safe_float(slope),
            "intercept": _safe_float(intercept),
            "r": _safe_float(r_val),
            "r_sq": _safe_float(r_val ** 2),
        }

    return out


def aggregate_batch(results: list) -> dict:
    """
    Aggregate a batch of PromptResult objects into summary statistics.
    Groups by category and computes effect sizes between categories.
    Includes ALL metrics (ASM, LTP, SFD, rank displacement).

    Length residualization: fits OLS regression of each metric against
    seq_len across the full batch, then computes Cohen's d on residuals.
    Both raw and residualized effect sizes are reported.
    """
    # ── Compute length residuals for all metrics ──────────────────
    resid_data = length_residualize(results)

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

        # Negative token frequency
        n_with_neg = sum(1 for r in group if r.has_negative_tokens)
        summary["negative_token_rate"] = n_with_neg / len(group) if group else 0

        # Raw metric summaries
        for stat_key, extractor in ALL_METRICS_REGISTRY:
            vals = [float(v) for r in group
                    for v in [extractor(r)] if v is not None
                    and not (math.isnan(float(v)) or math.isinf(float(v)))]
            if vals:
                summary["metrics"][stat_key] = bootstrap_ci(vals)

            # Length-residualized summary
            if stat_key in resid_data:
                resid_vals = [resid_data[stat_key]["residuals"][i]
                              for i in indices
                              if resid_data[stat_key]["residuals"][i] is not None]
                if resid_vals:
                    summary["metrics"][stat_key + "_resid"] = bootstrap_ci(resid_vals)

        cat_summaries[cat] = summary

    # ── Pairwise separability ─────────────────────────────────────
    # Compute effect sizes for ALL category pairs, not just benign-vs-harmful.
    # Primary comparisons: benign vs each other category.
    benign_cats = {"benign", "baseline", "mild", "user_baseline"}
    target_cats = {"harmful", "adversarial", "jailbreak", "dual-use"}

    benign_idx = [i for i, r in enumerate(results) if r.category in benign_cats]

    separability = {}
    separability_resid = {}

    for target_cat in target_cats:
        target_idx = result_indices.get(target_cat, [])
        if not benign_idx or not target_idx:
            continue

        pair_key = f"benign_vs_{target_cat}"
        separability[pair_key] = {}
        separability_resid[pair_key] = {}

        for stat_key, extractor in ALL_METRICS_REGISTRY:
            # Raw values
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

            # Residualized values
            if stat_key in resid_data:
                rd = resid_data[stat_key]["residuals"]
                b_resid = [rd[i] for i in benign_idx if rd[i] is not None]
                t_resid = [rd[i] for i in target_idx if rd[i] is not None]
                if b_resid and t_resid:
                    separability_resid[pair_key][stat_key] = {
                        "effect_size": bootstrap_effect_size(b_resid, t_resid),
                        "threshold": best_threshold(b_resid, t_resid),
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
                entry = {
                    "effect_size": bootstrap_effect_size(b_vals, h_vals),
                    "threshold": best_threshold(b_vals, h_vals),
                    "benign_mean": float(np.mean(b_vals)),
                    "harmful_mean": float(np.mean(h_vals)),
                }
                separability_legacy[stat_key] = entry

                # Residualized
                if stat_key in resid_data:
                    rd = resid_data[stat_key]["residuals"]
                    b_r = [rd[i] for i in benign_idx if rd[i] is not None]
                    h_r = [rd[i] for i in harmful_idx if rd[i] is not None]
                    if b_r and h_r:
                        separability_legacy[stat_key + "_resid"] = {
                            "effect_size": bootstrap_effect_size(b_r, h_r),
                            "threshold": best_threshold(b_r, h_r),
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

    # ── Length regression diagnostics ─────────────────────────────
    length_regressions = {}
    for stat_key, info in resid_data.items():
        length_regressions[stat_key] = {
            "slope": info["slope"],
            "intercept": info["intercept"],
            "r": info["r"],
            "r_sq": info["r_sq"],
        }

    return {
        "n_total": len(results),
        "categories": cat_summaries,
        "separability": separability_legacy,          # backward compat
        "separability_pairwise": separability,        # raw, per-pair
        "separability_residualized": separability_resid,  # length-controlled
        "length_regressions": length_regressions,     # regression diagnostics
        "correlations": correlations,
    }
