"""Mechanistic Interpretability Readiness (ported from TASM).

Direct port of TASM's `engine/modules/mechanistic_interpretability.py`.
Computes AUROC (full features, length-only, length-residualized),
per-metric AUROC, PCA dimensionality, metric redundancy, random
projection baseline, and readiness scorecard. Emits the JSON shape
`renderMIResults` in static/js/main.js reads.

Operates on TAGM's scalar measurements. Required scalars per prompt
(collected across measurement objects — see `_extract_features`):
  stress_score.stress_mean / stress_score.stress (legacy)
  last_position_attribution.net_correction_to_last
  last_position_attribution.entropy
  last_position_attribution.top2_share
  last_position_attribution.middle_share
  last_position_attribution.interior_cv
  last_position_attribution.kl_divergence (optional)
  lateral_tension_profile.mean_M (optional)
  spectral_field_density.density_mean (optional)
  rank_displacement.mean_tau (optional)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.analysis.statistics import extract_scalar
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


# Metric list: (display_key, measurement_name, scalar_field)
_CORE_METRICS = [
    ("stress_score",        "stress_score",              "stress_mean"),
    ("net_correction",      "last_position_attribution", "net_correction_to_last"),
    ("entropy",             "last_position_attribution", "entropy"),
    ("top2_share",          "last_position_attribution", "top2_share"),
    ("middle_share",        "last_position_attribution", "middle_share"),
    ("interior_cv",         "last_position_attribution", "interior_cv"),
    ("kl_divergence",       "last_position_attribution", "kl_divergence"),
]
_EXTRA_METRICS = [
    ("ltp_mean_M",          "lateral_tension_profile",   "mean_M"),
    ("ltp_n_directional",   "lateral_tension_profile",   "n_directional"),
    ("sfd_density_mean",    "spectral_field_density",    "density_mean"),
    ("rd_mean_tau",         "rank_displacement",         "mean_tau"),
]

_PCA_AXIS_NAMES = [
    "Correction Magnitude", "Correction Distribution",
    "Spectral Engagement",  "Directional Tension",
    "Output Divergence",    "Field Concentration",
    "Rank Stability",       "Spectral Entropy",
]


# ── Helpers (ported from TASM) ─────────────────────────────────

def _auroc(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    total = 0.0
    for p in pos:
        total += float(np.sum(neg < p)) + 0.5 * float(np.sum(neg == p))
    return total / (len(pos) * len(neg))


def _logistic_predict(X, w, b):
    z = X @ w + b
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic(X, y, lr=0.1, n_iter=500, reg=0.01):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        p = _logistic_predict(X, w, b)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        grad_w = (X.T @ (p - y)) / n + reg * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _cv_auroc(X, y, n_folds=5, seed=42, lr=0.1, n_iter=500, reg=0.01):
    rng = np.random.RandomState(seed)
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    folds = np.zeros(n, dtype=int)
    for i, idx in enumerate(pos_idx):
        folds[idx] = i % n_folds
    for i, idx in enumerate(neg_idx):
        folds[idx] = i % n_folds

    aucs = []
    for f in range(n_folds):
        test_mask = folds == f
        train_mask = ~test_mask
        if np.sum(y[test_mask] == 1) == 0 or np.sum(y[test_mask] == 0) == 0:
            continue
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0)
        sd[sd < 1e-10] = 1.0
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        w, b = _fit_logistic(X_tr, y_tr, lr=lr, n_iter=n_iter, reg=reg)
        pred = _logistic_predict(X_te, w, b)
        aucs.append(_auroc(y_te, pred))
    if not aucs:
        return 0.5, 0.0
    return float(np.mean(aucs)), float(np.std(aucs))


def _residualize_length(X, seq_lens):
    X_res = np.copy(X)
    sl = seq_lens - seq_lens.mean()
    denom = float(np.dot(sl, sl))
    if denom < 1e-10:
        return X_res
    for col in range(X.shape[1]):
        a = float(np.dot(sl, X[:, col] - X[:, col].mean()) / denom)
        X_res[:, col] = X[:, col] - a * seq_lens
    return X_res


def _length_correlations(X, seq_lens, names):
    out = []
    for col in range(X.shape[1]):
        r = np.corrcoef(X[:, col], seq_lens)[0, 1]
        if not np.isfinite(r):
            r = 0.0
        r2 = r ** 2
        if r2 > 0.6:
            sev = "heavily confounded"
        elif r2 > 0.3:
            sev = "moderately confounded"
        elif r2 > 0.1:
            sev = "mildly confounded"
        else:
            sev = "minimal"
        out.append({
            "metric": names[col],
            "r": round(float(r), 4),
            "r_squared": round(float(r2), 4),
            "severity": sev,
        })
    out.sort(key=lambda x: abs(x["r"]), reverse=True)
    return out


def _pca_analysis(X, names, variance_target=0.95):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-10] = 1.0
    Z = (X - mu) / sd
    C = np.cov(Z.T) if Z.shape[1] > 1 else np.array([[np.var(Z)]])
    if C.ndim == 0:
        C = np.array([[C]])
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    total = float(eigvals.sum())
    if total < 1e-10:
        return [], 0, 0
    var_explained = eigvals / total
    cumulative = np.cumsum(var_explained)
    n_target = int(np.searchsorted(cumulative, variance_target) + 1)
    n_components = min(len(var_explained), 8)
    comps = []
    for i in range(n_components):
        axis_name = (_PCA_AXIS_NAMES[i] if i < len(_PCA_AXIS_NAMES)
                     else f"PC{i+1}")
        loadings = list(zip(names, eigvecs[:, i].tolist()))
        loadings.sort(key=lambda t: abs(t[1]), reverse=True)
        comps.append({
            "index": i + 1,
            "name": axis_name,
            "variance_explained": round(float(var_explained[i]) * 100, 1),
            "cumulative": round(float(cumulative[i]) * 100, 1),
            "top_loadings": [
                {"metric": m, "loading": round(float(l), 4)}
                for m, l in loadings[:5]
            ],
        })
    return comps, n_target, n_components


def _metric_redundancy(X, names, threshold=0.8):
    if X.shape[1] < 2:
        return []
    C = np.corrcoef(X.T)
    pairs = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            r = C[i, j]
            if not np.isfinite(r):
                continue
            if abs(r) > threshold:
                if abs(r) > 0.95:
                    impl = "functionally identical"
                elif abs(r) > 0.9:
                    impl = "highly redundant"
                else:
                    impl = "redundant"
                pairs.append({
                    "metric_a": names[i],
                    "metric_b": names[j],
                    "r": round(float(r), 4),
                    "implication": impl,
                })
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return pairs


def _per_metric_auroc(X, y, names):
    out = []
    for col in range(X.shape[1]):
        a = _auroc(y, X[:, col])
        a_inv = _auroc(y, -X[:, col])
        best = max(a, a_inv)
        direction = "higher → risk" if a >= a_inv else "lower → risk"
        out.append({
            "metric": names[col],
            "auroc": round(float(best), 4),
            "direction": direction,
        })
    out.sort(key=lambda x: x["auroc"], reverse=True)
    return out


def _random_projection_baseline(X, y, n_trials=10, seed=42, n_folds=5,
                                 lr=0.1, n_iter=500, reg=0.01):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    frob = float(np.linalg.norm(X, "fro"))
    aucs = []
    for t in range(n_trials):
        R = rng.randn(n, d)
        rfrob = float(np.linalg.norm(R, "fro"))
        if rfrob > 1e-10:
            R = R * (frob / rfrob)
        auc, _ = _cv_auroc(R, y, n_folds=n_folds, seed=seed + t,
                            lr=lr, n_iter=n_iter, reg=reg)
        aucs.append(auc)
    return {
        "mean_auc": round(float(np.mean(aucs)), 4),
        "std_auc": round(float(np.std(aucs)), 4),
        "min_auc": round(float(np.min(aucs)), 4),
        "max_auc": round(float(np.max(aucs)), 4),
        "n_trials": n_trials,
    }


# ── Feature extraction from TAGM session ─────────────────────

def _extract_features(prompts, safe_cats):
    feature_names = [k for k, _, _ in (_CORE_METRICS + _EXTRA_METRICS)]
    rows, labels, lengths, cats, valid = [], [], [], [], []
    risk_cats = {"harmful", "jailbreak", "adversarial", "dual-use"} - safe_cats

    for p in prompts:
        cat = (p.get("category") or "").lower().strip()
        if cat in ("model_response", "unknown", ""):
            valid.append(False)
            continue
        if cat not in safe_cats and cat not in risk_cats:
            valid.append(False)
            continue

        core_vals = []
        ok = True
        for _, mname, fname in _CORE_METRICS:
            v = extract_scalar(p, mname, fname)
            if v is None:
                ok = False
                break
            core_vals.append(float(v))
        if not ok:
            valid.append(False)
            continue

        extra_vals = []
        for _, mname, fname in _EXTRA_METRICS:
            v = extract_scalar(p, mname, fname)
            extra_vals.append(float(v) if v is not None else np.nan)

        rows.append(core_vals + extra_vals)
        labels.append(0 if cat in safe_cats else 1)
        lengths.append(float(p.get("seq_len") or len(p.get("tokens") or [])))
        cats.append(cat)
        valid.append(True)

    if not rows:
        return None, None, None, None, None

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=np.int32)
    seq_lens = np.array(lengths, dtype=float)

    nan_cols = np.all(np.isnan(X), axis=0)
    keep = ~nan_cols
    X = X[:, keep]
    feature_names = [n for n, k in zip(feature_names, keep) if k]
    for col in range(X.shape[1]):
        m = np.isnan(X[:, col])
        if m.any() and not m.all():
            X[m, col] = float(np.nanmean(X[:, col]))

    return X, y, seq_lens, cats, feature_names


# ── Module ─────────────────────────────────────────────────

@register_analysis
class MIReadiness(AnalysisModule):
    name = "mi_readiness"
    display_name = "MI Readiness Analysis"
    description = (
        "Evaluates session data against mechanistic interpretability "
        "community evaluation standards: AUROC discrimination, length "
        "confound analysis, PCA metric consolidation, random "
        "projection baselines, and metric redundancy detection."
    )
    version = "1.0.0"

    parameters = [
        ModuleParameter(
            name="n_folds", display_name="CV folds", kind="int",
            default=5, min_value=2, max_value=20, advanced=True,
            description="Stratified k-fold splits for AUROC."),
        ModuleParameter(
            name="n_random_trials", display_name="Random baseline trials",
            kind="int", default=10, min_value=1, max_value=100,
            advanced=True,
            description="Number of random projection baseline trials."),
        ModuleParameter(
            name="pca_variance_target", display_name="PCA variance target",
            kind="float", default=0.95, min_value=0.5, max_value=0.99,
            advanced=True,
            description="Fraction of variance for effective dim count."),
        ModuleParameter(
            name="redundancy_threshold",
            display_name="Redundancy |r| threshold",
            kind="float", default=0.80, min_value=0.5, max_value=0.99,
            advanced=True,
            description="|r| above this flags a metric pair as redundant."),
        ModuleParameter(
            name="safe_categories",
            display_name="Safe categories",
            kind="select", default="benign,mild",
            options=("benign,mild", "benign", "benign,mild,dual-use"),
            description="Which category labels count as the safe class."),
        ModuleParameter(
            name="random_seed",
            display_name="Random seed",
            kind="int", default=42, min_value=0, max_value=2**31-1,
            advanced=True,
            description="Seed for CV splits and random baseline trials."),
    ]

    def run(self, session, params, probes=None, context=None):
        n_folds = int(params.get("n_folds", 5))
        n_random = int(params.get("n_random_trials", 10))
        pca_target = float(params.get("pca_variance_target", 0.95))
        red_thresh = float(params.get("redundancy_threshold", 0.80))
        seed = int(params.get("random_seed", 42))
        safe_cats = set(c.strip().lower() for c in
                         str(params.get("safe_categories",
                                         "benign,mild")).split(","))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"n_folds": n_folds, "n_random_trials": n_random,
                        "pca_variance_target": pca_target,
                        "redundancy_threshold": red_thresh,
                        "random_seed": seed,
                        "safe_categories": sorted(safe_cats)},
        )

        prompts = session.get("prompts") or []
        X, y, seq_lens, cats, names = _extract_features(prompts, safe_cats)

        if X is None or len(X) < 10:
            err = ("Insufficient labeled data. Need ≥10 prompts with "
                   "safe (benign/mild) and risk (harmful/jailbreak) "
                   "categories.")
            result.warnings.append(err)
            result.objects["error"] = err
            result.objects["n_valid"] = 0 if X is None else len(X)
            return result

        n_safe = int(np.sum(y == 0))
        n_risk = int(np.sum(y == 1))
        if n_safe < 3 or n_risk < 3:
            err = (f"Need ≥3 safe and ≥3 risk prompts; have "
                   f"{n_safe} safe / {n_risk} risk.")
            result.warnings.append(err)
            result.objects["error"] = err
            result.objects["n_safe"] = n_safe
            result.objects["n_risk"] = n_risk
            return result

        out: dict = {
            "n_prompts": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_safe": n_safe, "n_risk": n_risk,
            "feature_names": names,
            "categories": sorted(set(cats)),
            "safe_categories": sorted(safe_cats),
            "config": {
                "n_folds": n_folds, "random_seed": seed,
                "pca_variance_target": pca_target,
                "redundancy_threshold": red_thresh,
            },
        }

        # AUROC full
        auc_full, auc_std = _cv_auroc(X, y, n_folds=n_folds, seed=seed)
        out["auroc_full"] = {
            "auc": round(auc_full, 4),
            "std": round(auc_std, 4),
            "n_folds": n_folds,
        }

        # AUROC length only
        auc_len, auc_len_std = _cv_auroc(
            seq_lens.reshape(-1, 1), y, n_folds=n_folds, seed=seed)
        out["auroc_length_only"] = {
            "auc": round(auc_len, 4),
            "std": round(auc_len_std, 4),
        }
        out["mean_seq_len_safe"] = round(float(seq_lens[y == 0].mean()), 1)
        out["mean_seq_len_risk"] = round(float(seq_lens[y == 1].mean()), 1)

        # AUROC residualized
        X_res = _residualize_length(X, seq_lens)
        auc_res, auc_res_std = _cv_auroc(X_res, y, n_folds=n_folds, seed=seed)
        out["auroc_residualized"] = {
            "auc": round(auc_res, 4),
            "std": round(auc_res_std, 4),
        }

        out["length_confounds"] = _length_correlations(X, seq_lens, names)
        out["per_metric_auroc"] = _per_metric_auroc(X, y, names)

        components, n_target, n_components = _pca_analysis(
            X, names, variance_target=pca_target)
        out["pca"] = {
            "components": components,
            "n_for_95pct": n_target,
            "effective_dimensionality": n_target,
            "n_features_original": int(X.shape[1]),
            "variance_target": pca_target,
        }

        out["redundancy"] = {
            "pairs": _metric_redundancy(X, names, threshold=red_thresh),
            "threshold": red_thresh,
        }
        out["redundancy"]["n_redundant_pairs"] = len(
            out["redundancy"]["pairs"])

        out["random_baseline"] = _random_projection_baseline(
            X, y, n_trials=n_random, seed=seed, n_folds=n_folds)
        out["random_baseline"]["delta_above_random"] = round(
            auc_full - out["random_baseline"]["mean_auc"], 4)

        # Scorecard
        scorecard = []
        if auc_full >= 0.95:
            scorecard.append({"item": "Discrimination (AUROC)",
                              "status": "strong",
                              "detail": f"AUC {auc_full:.3f} — excellent"})
        elif auc_full >= 0.80:
            scorecard.append({"item": "Discrimination (AUROC)",
                              "status": "adequate",
                              "detail": f"AUC {auc_full:.3f} — good"})
        else:
            scorecard.append({"item": "Discrimination (AUROC)",
                              "status": "weak",
                              "detail": f"AUC {auc_full:.3f} — weak"})

        if auc_res >= 0.90:
            scorecard.append({"item": "Length Confound",
                              "status": "strong",
                              "detail": f"Residualized AUC {auc_res:.3f}"})
        elif auc_res >= 0.75:
            scorecard.append({"item": "Length Confound",
                              "status": "adequate",
                              "detail": f"Residualized AUC {auc_res:.3f}"})
        else:
            scorecard.append({"item": "Length Confound",
                              "status": "weak",
                              "detail": f"Residualized AUC {auc_res:.3f}"})

        delta = out["random_baseline"]["delta_above_random"]
        if delta >= 0.15:
            scorecard.append({"item": "Weight-Delta Specificity",
                              "status": "strong",
                              "detail": f"Δ +{delta:.3f} over random"})
        elif delta >= 0.05:
            scorecard.append({"item": "Weight-Delta Specificity",
                              "status": "adequate",
                              "detail": f"Δ +{delta:.3f} over random"})
        else:
            scorecard.append({"item": "Weight-Delta Specificity",
                              "status": "weak",
                              "detail": f"Δ +{delta:.3f} — random "
                                          f"projections match"})

        red_n = out["redundancy"]["n_redundant_pairs"]
        if red_n <= 2:
            scorecard.append({"item": "Metric Efficiency",
                              "status": "strong",
                              "detail": f"{red_n} redundant pairs"})
        elif red_n <= 5:
            scorecard.append({"item": "Metric Efficiency",
                              "status": "adequate",
                              "detail": f"{red_n} redundant pairs"})
        else:
            scorecard.append({"item": "Metric Efficiency",
                              "status": "weak",
                              "detail": f"{red_n} redundant pairs — "
                                          f"consolidate"})

        min_class = min(n_safe, n_risk)
        if min_class >= 25:
            scorecard.append({"item": "Sample Size",
                              "status": "strong",
                              "detail": f"Min class size {min_class}"})
        elif min_class >= 10:
            scorecard.append({"item": "Sample Size",
                              "status": "adequate",
                              "detail": f"Min class size {min_class}"})
        else:
            scorecard.append({"item": "Sample Size",
                              "status": "weak",
                              "detail": f"Min class size {min_class}"})

        out["scorecard"] = scorecard
        statuses = [s["status"] for s in scorecard]
        if all(s == "strong" for s in statuses):
            out["overall_readiness"] = "ready"
            out["readiness_summary"] = (
                "Data meets MI community evaluation standards.")
        elif any(s == "weak" for s in statuses):
            n_weak = statuses.count("weak")
            out["overall_readiness"] = "gaps"
            out["readiness_summary"] = (
                f"{n_weak} critical gap(s) identified. See scorecard.")
        else:
            out["overall_readiness"] = "near_ready"
            out["readiness_summary"] = (
                "Close to MI standards. Minor improvements recommended.")

        result.objects.update(out)
        result.scalars["auroc_full"] = round(auc_full, 4)
        result.scalars["auroc_residualized"] = round(auc_res, 4)
        result.scalars["delta_over_random"] = delta
        result.scalars["effective_dimensionality"] = n_target
        return result
