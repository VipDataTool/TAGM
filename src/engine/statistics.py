"""
Statistics: bootstrap confidence intervals, effect sizes, and batch aggregation.
Designed for small-n situations where parametric assumptions are dubious.
Extended with LTP summary statistics (M, C, V).
"""

import numpy as np
from typing import List, Dict
from scipy import stats as sp_stats
import math
import zlib
from src.engine import config as engine_config

# Base seed for every resampling procedure here.  Each call derives its actual
# seed from this plus a hash of its own input, so results stay reproducible
# while different metrics no longer share an identical resample sequence.
_BASE_SEED = 42


def _seeded_rng(*arrays) -> np.random.Generator:
    """Deterministic RNG whose stream depends on the data it will resample.

    Previously every call did `default_rng(42)`, so two metrics with the same
    n were resampled with the *identical* index sequence and their CI widths
    were deterministically coupled rather than independent.
    """
    h = _BASE_SEED
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h = zlib.crc32(a.tobytes(), h & 0xFFFFFFFF)
    return np.random.default_rng(h & 0xFFFFFFFF)


def _safe_float(v):
    """Convert to float, replacing NaN/Inf with 0.

    Prefer `_opt_float` for reported statistics: collapsing a NaN statistic to
    an exact 0.0 makes "undefined" indistinguishable from "no effect".
    """
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def _opt_float(v):
    """Convert to float, returning None (not 0.0) for NaN/Inf."""
    if v is None:
        return None
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _clean_values(vals):
    """Filter out NaN/Inf from a list of floats."""
    return [v for v in vals if not (math.isnan(v) or math.isinf(v))]


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees-of-freedom weighted pooled SD.

    The previous form, sqrt((s_a^2 + s_b^2)/2), is the equal-n special case.
    Callers here routinely pool four benign categories against a single target
    category, so n is unequal and the two forms differ materially.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    num = (na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)
    return float(np.sqrt(num / (na + nb - 2)))


def cohens_d(group_a: list, group_b: list) -> float:
    """Signed Cohen's d (group_b - group_a) with df-weighted pooled SD.

    Signed, not absolute.  Taking abs() here discarded the direction of every
    effect ("benign > harmful" and "harmful > benign" became identical) and,
    when bootstrapped, produced a folded distribution whose lower CI bound was
    almost always > 0 even for two samples from the same distribution.
    """
    a, b = np.array(_clean_values(group_a)), np.array(_clean_values(group_b))
    if len(a) < engine_config.get("min_samples_d") or len(b) < engine_config.get("min_samples_d"):
        return 0.0
    pooled_std = _pooled_sd(a, b)
    if pooled_std == 0:
        return 0.0
    return _safe_float((b.mean() - a.mean()) / pooled_std)


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
    raw = list(values)
    values = np.array(_clean_values(raw))
    # Non-finite values are dropped, which silently shrinks n.  Report how many
    # so a metric that NaNs on half the corpus is not mistaken for a clean one.
    n_dropped = len(raw) - len(values)
    if len(values) < 1:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0,
                "n_dropped": n_dropped}
    if len(values) < 2:
        val = _safe_float(statistic(values))
        return {"estimate": val, "ci_low": val, "ci_high": val,
                "n": len(values), "n_dropped": n_dropped}

    rng = _seeded_rng(values)
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
        "n_dropped": n_dropped,
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
        # Not enough data to bootstrap.  Say so explicitly rather than emitting
        # a zero-width CI that reads like a precise estimate.
        return {"estimate": _safe_float(d), "ci_low": _safe_float(d),
                "ci_high": _safe_float(d), "n_a": len(a), "n_b": len(b),
                "insufficient_n": True, "excludes_zero": False}

    rng = _seeded_rng(a, b)
    boot_ds = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        ps = _pooled_sd(sa, sb)
        # Signed, matching cohens_d.  Bootstrapping |d| gives a folded
        # distribution bounded below by 0, so ci_low could never straddle zero
        # and the interval was structurally unable to report "no effect".
        d = (sb.mean() - sa.mean()) / ps if ps > 0 else 0.0
        boot_ds.append(d)

    boot_ds = np.array(boot_ds)
    alpha = (1 - ci) / 2
    lo = _safe_float(np.percentile(boot_ds, alpha * 100))
    hi = _safe_float(np.percentile(boot_ds, (1 - alpha) * 100))
    return {
        "estimate": _safe_float(cohens_d(list(a), list(b))),
        "ci_low": lo,
        "ci_high": hi,
        "n_a": len(a),
        "n_b": len(b),
        "insufficient_n": False,
        # Now a meaningful test: with a signed bootstrap an interval that
        # straddles zero genuinely indicates no detectable effect.
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def _sweep_best_accuracy(sorted_vals: np.ndarray, labels_sorted: np.ndarray):
    """Exact best split accuracy over all thresholds and both directions.

    `labels_sorted` is 1 for harmful, 0 for benign, ordered by ascending value.
    Returns (best_accuracy, best_threshold, best_direction).

    This replaces the old fixed `threshold_steps` linspace scan: it is exact
    (every distinct split is considered, not a 500-point approximation) and it
    is O(n log n) rather than O(steps * n), which is what makes the permutation
    null below affordable.
    """
    n = len(labels_sorted)
    if n == 0:
        return 0.0, 0.0, ">="
    cum_h = np.concatenate(([0], np.cumsum(labels_sorted)))          # (n+1,)
    cum_b = np.arange(n + 1) - cum_h                                  # (n+1,)
    total_h = cum_h[-1]
    # Predict harmful when v >= t, with t placed at split index i.
    acc_high = (total_h - cum_h + cum_b) / n
    # The complementary rule ("harmful when v < t") is exactly 1 - acc_high.
    acc_low = 1.0 - acc_high

    # Only split indices that a real threshold can actually realize: the start,
    # the end, and each boundary between DISTINCT values.  Splitting inside a
    # run of tied values would put equal numbers on opposite sides of a single
    # threshold, which is impossible — allowing it overstates the achievable
    # accuracy on any metric with ties.
    boundaries = np.ones(n + 1, dtype=bool)
    if n > 1:
        boundaries[1:n] = sorted_vals[1:] != sorted_vals[:-1]

    masked_high = np.where(boundaries, acc_high, -np.inf)
    masked_low = np.where(boundaries, acc_low, -np.inf)
    i_high = int(np.argmax(masked_high))
    i_low = int(np.argmax(masked_low))
    if masked_high[i_high] >= masked_low[i_low]:
        best_acc, i, direction = float(masked_high[i_high]), i_high, ">="
    else:
        best_acc, i, direction = float(masked_low[i_low]), i_low, "<"
    if i < n:
        best_t = float(sorted_vals[i])
    else:
        best_t = float(sorted_vals[-1]) + 1.0
    return best_acc, best_t, direction


def best_threshold(benign_vals: list, harmful_vals: list,
                   n_perm: int = None) -> dict:
    """Best in-sample split accuracy, with a label-permutation null.

    IMPORTANT: `accuracy` is optimized and evaluated on the same data.  It is
    the maximum over every candidate threshold and both directions, so it is
    biased upward and is NOT an estimate of out-of-sample performance.  With
    the small n this tool routinely runs (min_samples_d defaults to 2), a
    perfect in-sample split is the *expected* outcome under pure noise.

    `p_value` is what makes the number interpretable: it is the fraction of
    label permutations that achieve at least this accuracy.  Read the pair
    together, never `accuracy` alone.
    """
    b = np.array(_clean_values(list(benign_vals)), dtype=np.float64)
    h = np.array(_clean_values(list(harmful_vals)), dtype=np.float64)
    n = len(b) + len(h)
    if n == 0:
        return {"threshold": 0.0, "accuracy": 0.0, "direction": ">=",
                "n": 0, "p_value": None, "null_mean": None,
                "in_sample": True}

    vals = np.concatenate([b, h])
    labels = np.concatenate([np.zeros(len(b)), np.ones(len(h))])
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    sorted_labels = labels[order]

    best_acc, best_t, direction = _sweep_best_accuracy(sorted_vals, sorted_labels)

    # Permutation null: shuffle the labels, re-run the same maximizing sweep.
    # Because the sweep is the same procedure, the null absorbs the selection
    # bias exactly.
    if n_perm is None:
        n_perm = int(engine_config.get("n_permutations") or 0) or 200
    p_value = None
    null_mean = None
    null_p95 = None
    if len(b) > 0 and len(h) > 0 and n_perm > 0:
        rng = _seeded_rng(b, h)
        null_accs = np.empty(n_perm, dtype=np.float64)
        perm_labels = sorted_labels.copy()
        for j in range(n_perm):
            rng.shuffle(perm_labels)
            null_accs[j] = _sweep_best_accuracy(sorted_vals, perm_labels)[0]
        # +1 in numerator and denominator: the observed labelling is itself one
        # draw from the null, which keeps the test valid at small n_perm.
        p_value = float((1 + np.sum(null_accs >= best_acc)) / (1 + n_perm))
        null_mean = float(null_accs.mean())
        null_p95 = float(np.percentile(null_accs, 95))

    return {"threshold": float(best_t), "accuracy": float(best_acc),
            "direction": direction, "n": int(n),
            "n_benign": int(len(b)), "n_harmful": int(len(h)),
            "p_value": p_value, "null_mean": null_mean, "null_p95": null_p95,
            "in_sample": True}


def benjamini_hochberg(p_values: List[float], fdr: float = 0.05) -> dict:
    """Benjamini-Hochberg step-up procedure.

    Returns {"threshold": float|None, "n_significant": int, "fdr": float}.
    `threshold` is the largest p that survives; compare each raw p against it.

    aggregate_batch runs len(ALL_METRICS_REGISTRY) metrics x 4 target
    categories plus a pooled pass,
    so raw p-values there are not directly interpretable.
    """
    ps = sorted(p for p in p_values if p is not None)
    m = len(ps)
    if m == 0:
        return {"threshold": None, "n_significant": 0, "fdr": fdr, "n_tests": 0}
    thresh = None
    for i, p in enumerate(ps, start=1):
        if p <= (i / m) * fdr:
            thresh = p
    n_sig = sum(1 for p in ps if thresh is not None and p <= thresh)
    return {"threshold": thresh, "n_significant": n_sig,
            "fdr": fdr, "n_tests": m}


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


def _attr_val(r, key):
    """Attribution-derived scalar, or None if attribution never ran.

    When signed attribution is unavailable (typically because attention
    weights were not captured — full_capture defaults to False) these fields
    keep their dataclass default of 0.0.  Feeding those zeros to bootstrap_ci
    as if they were observations is the difference between "we measured no
    correction" and "we did not measure".
    """
    if getattr(r, 'attribution_unavailable', None):
        return None
    return getattr(r, key, None)


# Each entry: (stat_key, extractor) where extractor(PromptResult) -> float | None.
ALL_METRICS_REGISTRY = [
    # ASM
    ("stress_score",   lambda r: getattr(r, 'stress_score', None)),
    # LENGTH-SENSITIVE. entropy is normalized by log(seq_len), which only
    # corrects the uniform case: under a fixed per-token law its expectation
    # still drifts with n (0.83 -> 0.90 from n=10 to n=160), and under a
    # fixed-sparsity law it drifts the other way (0.52 -> 0.37). The sign of
    # the bias depends on the underlying sparsity, so no fixed rescaling
    # repairs it. interior_cv drifts likewise (0.88 -> 1.23). Both produce
    # ~30% false-positive rates when the two groups differ in typical length.
    # Retained because they are established outputs, but ALWAYS read them
    # against length_correlations before treating a difference as behavioural.
    ("entropy",        lambda r: _attr_val(r, 'entropy')),
    ("top2_share",     lambda r: _attr_val(r, 'top2_share')),
    # middle_share is EXACTLY 1 - top2_share (attr_dist sums to 1 and the two
    # partition it), verified to 10 decimal places. Including both double-
    # counted a single degree of freedom in the Benjamini-Hochberg n_tests
    # and in the MI module's PCA. Dropped from cross-prompt comparison; the
    # field is still computed, stored and displayed.
    ("interior_cv",    lambda r: _attr_val(r, 'interior_cv')),
    ("net_correction", lambda r: _attr_val(r, 'net_correction')),
    # KL
    ("kl_divergence",  lambda r: getattr(r, 'kl_divergence', None)),
    # LTP (ltp_mean_L excluded: constant 1.0, zero variance)
    ("ltp_mean_M",     lambda r: _get_ltp_val_generic(r, 'mean_M')),
    ("ltp_mean_V",     lambda r: _get_ltp_val_generic(r, 'mean_V')),
    # RATE, not the raw count. n_directional is extensive (it grows linearly
    # with sequence length), so comparing it across prompts of differing
    # length produced a spurious effect of |d| ~ 3.7 that excluded zero in
    # 100% of null replications. directional_frac is the length-invariant form.
    ("ltp_directional_frac",
     lambda r: _get_ltp_val_generic(r, 'directional_frac')),
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

    A diagnostic: every metric in ALL_METRICS_REGISTRY is intended to be
    INTENSIVE (a mean or rate), so a strong correlation with seq_len is
    evidence of either genuine length-related behaviour or a residual
    measurement defect.

    Do NOT read a correlation here as automatically meaning "genuine
    behavioral signal" — the previous wording of this docstring said exactly
    that, and it was wrong.  An EXTENSIVE metric (a sum, a count, or a max
    over token positions) correlates with length BY CONSTRUCTION, and no
    amount of downstream statistics can separate that from behaviour.
    `n_directional` was such a metric and was replaced here by
    `directional_frac` for precisely this reason.

    Before adding a metric to the registry, check it is intensive.

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


def _length_confound(results: list, idx_a: list, idx_b: list,
                     stat_key: str) -> dict:
    """Flag a separability result that prompt length could explain.

    Several metrics are not length-invariant even though they look like pure
    shape statistics.  `entropy` is divided by log(seq_len), which only
    corrects the uniform case: with a fixed per-token law its expectation
    still climbs (0.84 at n=10 to 0.90 at n=160), and `interior_cv` climbs
    harder (0.87 to 1.23).  Under a null where the two groups differ ONLY in
    prompt length, they report a significant effect in 55% and 28% of
    replications respectively against a nominal 5%.

    Rescaling cannot fix this — under a fixed-sparsity law the bias runs the
    other way, so the sign depends on the data.  What CAN be done is refuse to
    report the comparison silently: this returns the group length difference
    and the metric's own correlation with length, so a reader can see when
    "harmful differs from benign" might just be "harmful prompts are longer".
    """
    la = np.array([float(results[i].seq_len) for i in idx_a])
    lb = np.array([float(results[i].seq_len) for i in idx_b])
    if len(la) < 2 or len(lb) < 2:
        return {"checked": False}

    pooled = np.sqrt((la.var(ddof=1) + lb.var(ddof=1)) / 2)
    len_d = float((lb.mean() - la.mean()) / pooled) if pooled > 0 else 0.0

    # Correlation of this metric with length, across both groups pooled.
    extractor = dict(ALL_METRICS_REGISTRY).get(stat_key)
    r = None
    if extractor is not None:
        xs, ys = [], []
        for i in list(idx_a) + list(idx_b):
            v = extractor(results[i])
            if v is None:
                continue
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                continue
            xs.append(float(results[i].seq_len))
            ys.append(v)
        if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
            r = _safe_float(np.corrcoef(xs, ys)[0, 1])

    # Fire on the LENGTH DIFFERENCE alone.  Requiring a strong metric-length
    # correlation as well was too strict and missed the very case this exists
    # for: with benign ~40 tokens and harmful ~90, entropy false-positives in
    # 55% of null replications while its pooled r is only ~0.28, because
    # mixing two groups at different lengths dilutes the within-sample
    # correlation. Once the groups are separated in length by this much, no
    # metric comparison between them can cleanly attribute a difference to
    # behaviour rather than length; r is reported as supporting evidence, not
    # as a precondition.
    suspect = bool(abs(len_d) > 0.8)
    return {
        "checked": True,
        "length_effect_size": len_d,
        "metric_length_r": r,
        "mean_len_a": float(la.mean()),
        "mean_len_b": float(lb.mean()),
        "suspect": suspect,
    }


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
                entry = {
                    "effect_size": bootstrap_effect_size(b_vals, t_vals),
                    "threshold": best_threshold(b_vals, t_vals),
                    "benign_mean": float(np.mean(b_vals)),
                    "target_mean": float(np.mean(t_vals)),
                }
                entry["length_confound"] = _length_confound(
                    results, benign_idx, target_idx, stat_key)
                separability[pair_key][stat_key] = entry

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

    # ── Multiple-comparison correction ────────────────────────────
    # ~14 metrics x 4 target categories, plus a pooled pass, all reported from
    # the same session.  Without this the raw threshold p-values are not
    # interpretable: at FDR 0.05 you expect ~3 "hits" from noise alone.
    all_ps = []
    for pair in separability.values():
        for entry in pair.values():
            all_ps.append(entry.get("threshold", {}).get("p_value"))
    for entry in separability_legacy.values():
        all_ps.append(entry.get("threshold", {}).get("p_value"))
    fdr_level = engine_config.get("fdr_level") or 0.05
    correction = benjamini_hochberg(all_ps, fdr=fdr_level)

    # Annotate each test with its own verdict so the UI cannot show a bare p.
    def _mark(entry):
        p = entry.get("threshold", {}).get("p_value")
        thr = correction["threshold"]
        entry["threshold"]["significant_fdr"] = bool(
            p is not None and thr is not None and p <= thr)

    for pair in separability.values():
        for entry in pair.values():
            _mark(entry)
    for entry in separability_legacy.values():
        _mark(entry)

    return {
        "n_total": len(results),
        "categories": cat_summaries,
        "separability": separability_legacy,
        "separability_pairwise": separability,
        "length_correlations": len_corr,
        "correlations": correlations,
        "multiple_comparisons": correction,
    }
