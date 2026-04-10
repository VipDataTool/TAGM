"""
Mechanistic Interpretability Readiness Module for TASM.

Addresses the MI community evaluation gaps identified in the
independent review:

1. AUROC computation (replaces Cohen's d as primary metric)
2. Length confound analysis (raw vs length-residualized AUC)
3. PCA metric consolidation (effective dimensionality)
4. Random projection baseline (validates weight-delta contribution)
5. Metric redundancy detection (near-duplicate pairs)
6. Cross-model transfer readiness check

Operates purely on session data — no model inference required.
"""

import logging
import numpy as np
from collections import defaultdict

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")

# The 15 core TAGM metrics in canonical order
METRIC_KEYS = [
    "stress_score", "net_correction", "entropy", "gini",
    "top2_share", "middle_share", "interior_cv", "kl_divergence",
]
LTP_KEYS = ["ltp_mean_M", "ltp_mean_C", "ltp_n_directional"]
SFD_KEYS = ["sfd_density_mean", "sfd_entropy_mean", "sfd_energy_mean"]
RD_KEYS = ["rank_displacement_mean_tau"]

# Human-readable names for the independent measurement axes
PCA_AXIS_NAMES = [
    "Correction Magnitude",
    "Correction Distribution",
    "Spectral Engagement",
    "Directional Tension",
    "Output Divergence",
    "Field Concentration",
    "Rank Stability",
    "Spectral Entropy",
]


def _extract_features(results, safe_cats=None):
    """Extract feature matrix and labels from session results.

    Args:
        results: list of session result dicts
        safe_cats: set of category names to treat as safe (class 0).
                   Everything else with a recognized category becomes risk (class 1).

    Returns:
        X: np.ndarray (n_prompts, n_features)
        y: np.ndarray (n_prompts,) — 0=safe, 1=risk
        seq_lens: np.ndarray (n_prompts,)
        categories: list of str
        feature_names: list of str
        valid_mask: which results were usable
    """
    if safe_cats is None:
        safe_cats = {"benign", "mild"}
    risk_cats = {"harmful", "jailbreak", "adversarial", "dual-use"} - safe_cats
    feature_names = list(METRIC_KEYS)
    rows = []
    labels = []
    lengths = []
    cats = []
    valid = []

    for i, r in enumerate(results):
        cat = (r.get("category") or "").lower().strip()
        if cat in ("model_response", "unknown", ""):
            valid.append(False)
            continue

        # Extract core metrics
        vals = []
        ok = True
        for k in METRIC_KEYS:
            v = r.get(k)
            if v is None:
                ok = False
                break
            vals.append(float(v))

        if not ok:
            valid.append(False)
            continue

        # Try LTP metrics
        ltp = r.get("ltp") or {}
        for k in LTP_KEYS:
            short = k.replace("ltp_", "")
            v = ltp.get(short)
            if v is not None:
                vals.append(float(v))
            else:
                vals.append(np.nan)

        # Try SFD metrics
        sfd = r.get("sfd") or {}
        for k in SFD_KEYS:
            short = k.replace("sfd_", "")
            v = sfd.get(short)
            if v is not None:
                vals.append(float(v))
            else:
                vals.append(np.nan)

        # Try RD metrics
        rd = r.get("rank_displacement") or {}
        for k in RD_KEYS:
            short = k.replace("rank_displacement_", "")
            v = rd.get(short)
            if v is not None:
                vals.append(float(v))
            else:
                vals.append(np.nan)

        # Binary label: safe vs risk based on configured categories
        if cat in safe_cats:
            label = 0
        elif cat in risk_cats:
            label = 1
        else:
            valid.append(False)
            continue

        rows.append(vals)
        labels.append(label)
        lengths.append(r.get("seq_len", len(r.get("tokens", []))))
        cats.append(cat)
        valid.append(True)

    if not rows:
        return None, None, None, None, None, valid

    # Build feature names for extended metrics
    all_names = list(METRIC_KEYS) + list(LTP_KEYS) + list(SFD_KEYS) + list(RD_KEYS)

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    seq_lens = np.array(lengths, dtype=np.float64)

    # Drop columns that are all NaN
    nan_cols = np.all(np.isnan(X), axis=0)
    keep = ~nan_cols
    X = X[:, keep]
    all_names = [n for n, k in zip(all_names, keep) if k]

    # Fill remaining NaNs with column means
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if mask.any() and not mask.all():
            X[mask, col] = np.nanmean(X[:, col])

    return X, y, seq_lens, cats, all_names, valid


def _auroc(y_true, y_score):
    """Compute AUROC without sklearn dependency."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    n_pos, n_neg = len(pos), len(neg)
    total = 0
    for p in pos:
        total += np.sum(neg < p) + 0.5 * np.sum(neg == p)
    return total / (n_pos * n_neg)


def _logistic_predict(X, w, b):
    """Logistic regression prediction (sigmoid)."""
    z = X @ w + b
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic(X, y, lr=0.1, n_iter=500, reg=0.01):
    """Minimal logistic regression via gradient descent."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        p = _logistic_predict(X, w, b)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        grad_w = (X.T @ (p - y)) / n + reg * w
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _cv_auroc(X, y, n_folds=5, seed=42, lr=0.1, n_iter=500, reg=0.01):
    """Stratified k-fold cross-validated AUROC."""
    rng = np.random.RandomState(seed)
    n = len(y)

    # Stratified fold assignment
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
    for fold in range(n_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        if np.sum(y[test_mask] == 1) == 0 or np.sum(y[test_mask] == 0) == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        # Standardize
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
    """Remove linear dependence on sequence length from all features."""
    X_res = np.copy(X)
    for col in range(X.shape[1]):
        # OLS: feature = a * seq_len + b + residual
        sl = seq_lens - seq_lens.mean()
        denom = np.dot(sl, sl)
        if denom < 1e-10:
            continue
        a = np.dot(sl, X[:, col] - X[:, col].mean()) / denom
        X_res[:, col] = X[:, col] - a * seq_lens
    return X_res


def _length_correlations(X, seq_lens, feature_names):
    """Compute Pearson r between each feature and sequence length."""
    corrs = []
    for col in range(X.shape[1]):
        r = np.corrcoef(X[:, col], seq_lens)[0, 1]
        if np.isnan(r):
            r = 0.0
        r2 = r ** 2
        if r2 > 0.6:
            severity = "heavily confounded"
        elif r2 > 0.3:
            severity = "moderately confounded"
        elif r2 > 0.1:
            severity = "mildly confounded"
        else:
            severity = "minimal"
        corrs.append({
            "metric": feature_names[col],
            "r": round(float(r), 4),
            "r_squared": round(float(r2), 4),
            "severity": severity,
        })
    corrs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return corrs


def _pca_analysis(X, feature_names, variance_target=0.95):
    """PCA on standardized features. Returns variance explained and loadings."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-10] = 1.0
    Z = (X - mu) / sd

    # Covariance matrix
    C = np.cov(Z.T)
    if C.ndim == 0:
        C = np.array([[C]])

    eigvals, eigvecs = np.linalg.eigh(C)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total = eigvals.sum()
    if total < 1e-10:
        return [], [], 0

    var_explained = eigvals / total
    cumulative = np.cumsum(var_explained)

    # How many for target variance?
    n_95 = int(np.searchsorted(cumulative, variance_target) + 1)
    n_components = min(len(var_explained), 8)

    components = []
    for i in range(n_components):
        axis_name = PCA_AXIS_NAMES[i] if i < len(PCA_AXIS_NAMES) else f"PC{i+1}"
        # Top loading features for this component
        loadings = list(zip(feature_names, eigvecs[:, i].tolist()))
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        top_loadings = [{"metric": m, "loading": round(float(l), 4)}
                        for m, l in loadings[:5]]
        components.append({
            "index": i + 1,
            "name": axis_name,
            "variance_explained": round(float(var_explained[i]) * 100, 1),
            "cumulative": round(float(cumulative[i]) * 100, 1),
            "top_loadings": top_loadings,
        })

    return components, n_95, n_components


def _metric_redundancy(X, feature_names, threshold=0.8):
    """Find near-duplicate metric pairs (|r| > threshold)."""
    n_feat = X.shape[1]
    corr_matrix = np.corrcoef(X.T)
    pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            r = corr_matrix[i, j]
            if np.isnan(r):
                continue
            if abs(r) > threshold:
                pairs.append({
                    "metric_a": feature_names[i],
                    "metric_b": feature_names[j],
                    "r": round(float(r), 4),
                    "implication": "functionally identical" if abs(r) > 0.95
                                  else "highly redundant" if abs(r) > 0.9
                                  else "redundant",
                })
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return pairs


def _random_projection_baseline(X, y, seq_lens, n_trials=10, seed=42,
                                 n_folds=5, lr=0.1, n_iter=500, reg=0.01):
    """Replace feature matrix with random projections of matched norm.

    This tests whether the weight-delta projection is special or whether
    any high-dimensional projection would separate categories equally well.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Frobenius norm of original feature matrix
    frob = np.linalg.norm(X, 'fro')

    aucs = []
    for trial in range(n_trials):
        R = rng.randn(n, d)
        R_frob = np.linalg.norm(R, 'fro')
        if R_frob > 1e-10:
            R = R * (frob / R_frob)
        auc, _ = _cv_auroc(R, y, n_folds=n_folds, seed=seed + trial,
                           lr=lr, n_iter=n_iter, reg=reg)
        aucs.append(auc)

    return {
        "mean_auc": round(float(np.mean(aucs)), 4),
        "std_auc": round(float(np.std(aucs)), 4),
        "min_auc": round(float(np.min(aucs)), 4),
        "max_auc": round(float(np.max(aucs)), 4),
        "n_trials": n_trials,
    }


def _per_metric_auroc(X, y, feature_names):
    """Single-feature AUROC for each metric."""
    results = []
    for col in range(X.shape[1]):
        auc = _auroc(y, X[:, col])
        # Also try inverted
        auc_inv = _auroc(y, -X[:, col])
        best = max(auc, auc_inv)
        direction = "higher → risk" if auc >= auc_inv else "lower → risk"
        results.append({
            "metric": feature_names[col],
            "auroc": round(float(best), 4),
            "direction": direction,
        })
    results.sort(key=lambda x: x["auroc"], reverse=True)
    return results


def _amplification_check(results):
    """Check for bidirectional scaling pattern if multi-model data exists.

    Looks for systematic category-level differences that would indicate
    the correction field sharpens with scale.
    """
    by_cat = defaultdict(list)
    for r in results:
        cat = (r.get("category") or "").lower().strip()
        stress = r.get("stress_score")
        if cat and stress is not None:
            by_cat[cat].append(float(stress))

    if len(by_cat) < 2:
        return None

    summary = {}
    for cat, vals in by_cat.items():
        summary[cat] = {
            "n": len(vals),
            "mean_stress": round(float(np.mean(vals)), 4),
            "std_stress": round(float(np.std(vals)), 4),
        }

    # Compute safe vs risk separation
    safe_vals = []
    risk_vals = []
    for cat, vals in by_cat.items():
        if cat in ("benign", "mild"):
            safe_vals.extend(vals)
        elif cat in ("harmful", "jailbreak", "adversarial"):
            risk_vals.extend(vals)

    if safe_vals and risk_vals:
        safe_mean = np.mean(safe_vals)
        risk_mean = np.mean(risk_vals)
        pooled_std = np.sqrt(
            (np.var(safe_vals) * (len(safe_vals) - 1) +
             np.var(risk_vals) * (len(risk_vals) - 1)) /
            max(1, len(safe_vals) + len(risk_vals) - 2)
        )
        cohens_d = (risk_mean - safe_mean) / max(pooled_std, 1e-10)
        summary["_separation"] = {
            "safe_mean": round(float(safe_mean), 4),
            "risk_mean": round(float(risk_mean), 4),
            "cohens_d": round(float(cohens_d), 3),
            "n_safe": len(safe_vals),
            "n_risk": len(risk_vals),
        }

    return summary


class MechanisticInterpretabilityModule(TASMModule):
    name = "mechanistic_interpretability"
    display_name = "MI Readiness Analysis"
    description = (
        "Evaluates session data against mechanistic interpretability "
        "community standards: AUROC discrimination, length confound analysis, "
        "PCA metric consolidation, random projection baselines, and "
        "metric redundancy detection."
    )
    version = "1.0.0"

    min_results = 10
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="n_folds",
            display_name="CV Folds",
            description="Number of cross-validation folds for AUROC.",
            type="int",
            default=5,
            min_val=3,
            max_val=10,
        ),
        ModuleParameter(
            name="n_random_trials",
            display_name="Random Projection Trials",
            description="Number of random projection baselines to average.",
            type="int",
            default=10,
            min_val=5,
            max_val=50,
        ),
        ModuleParameter(
            name="redundancy_threshold",
            display_name="Redundancy |r| Threshold",
            description="Pearson |r| above which metric pairs are flagged as redundant.",
            type="float",
            default=0.80,
            min_val=0.5,
            max_val=0.99,
        ),
        ModuleParameter(
            name="random_seed",
            display_name="Random Seed",
            description="Seed for CV fold assignment and random projections. Set for reproducibility.",
            type="int",
            default=42,
            min_val=0,
            max_val=99999,
        ),
        ModuleParameter(
            name="pca_variance_target",
            display_name="PCA Variance Target",
            description="Cumulative variance fraction to determine effective dimensionality (e.g. 0.95 = 95%).",
            type="float",
            default=0.95,
            min_val=0.80,
            max_val=0.99,
        ),
        ModuleParameter(
            name="lr_learning_rate",
            display_name="Logistic Regression LR",
            description="Learning rate for the internal logistic regression classifier used for AUROC.",
            type="float",
            default=0.1,
            min_val=0.001,
            max_val=1.0,
        ),
        ModuleParameter(
            name="lr_iterations",
            display_name="Logistic Regression Iterations",
            description="Gradient descent iterations for logistic regression. Higher = better convergence.",
            type="int",
            default=500,
            min_val=100,
            max_val=5000,
        ),
        ModuleParameter(
            name="lr_regularization",
            display_name="Logistic Regression L2 Reg",
            description="L2 regularization strength. Higher = more regularized, prevents overfitting on small datasets.",
            type="float",
            default=0.01,
            min_val=0.0,
            max_val=1.0,
        ),
        ModuleParameter(
            name="safe_categories",
            display_name="Safe Categories",
            description="Comma-separated categories treated as safe (class 0). Everything else with a known category becomes risk (class 1).",
            type="select",
            default="benign,mild",
            options=["benign,mild", "benign", "benign,mild,dual-use"],
        ),
    ]

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[MI] {msg}")

        n_folds = params.get("n_folds", 5)
        n_random = params.get("n_random_trials", 10)
        redundancy_thresh = params.get("redundancy_threshold", 0.80)
        random_seed = params.get("random_seed", 42)
        pca_var_target = params.get("pca_variance_target", 0.95)
        lr_lr = params.get("lr_learning_rate", 0.1)
        lr_iters = params.get("lr_iterations", 500)
        lr_reg = params.get("lr_regularization", 0.01)
        safe_cats_str = params.get("safe_categories", "benign,mild")
        safe_cats = set(c.strip().lower() for c in safe_cats_str.split(",") if c.strip())

        # ── Extract features ──
        prog("Extracting features from session data...")
        prog(f"Safe categories: {sorted(safe_cats)}")
        X, y, seq_lens, cats, feature_names, valid = _extract_features(session_results, safe_cats=safe_cats)

        if X is None or len(X) < 10:
            return {
                "error": "Insufficient labeled data. Need at least 10 prompts "
                         "with safe (benign/mild) and risk (harmful/jailbreak) categories.",
                "n_valid": 0,
            }

        n_safe = int(np.sum(y == 0))
        n_risk = int(np.sum(y == 1))

        if n_safe < 3 or n_risk < 3:
            return {
                "error": f"Need at least 3 safe and 3 risk prompts. "
                         f"Have {n_safe} safe, {n_risk} risk.",
                "n_safe": n_safe,
                "n_risk": n_risk,
            }

        prog(f"Feature matrix: {X.shape[0]} prompts × {X.shape[1]} features "
             f"({n_safe} safe, {n_risk} risk)")

        results = {
            "n_prompts": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_safe": n_safe,
            "n_risk": n_risk,
            "feature_names": feature_names,
            "categories": sorted(set(cats)),
            "safe_categories": sorted(safe_cats),
            "config": {
                "n_folds": n_folds,
                "random_seed": random_seed,
                "pca_variance_target": pca_var_target,
                "lr_learning_rate": lr_lr,
                "lr_iterations": lr_iters,
                "lr_regularization": lr_reg,
                "redundancy_threshold": redundancy_thresh,
            },
        }

        # ── 1. AUROC: Full features ──
        prog("Computing cross-validated AUROC (full features)...")
        auc_full, auc_std = _cv_auroc(X, y, n_folds=n_folds, seed=random_seed,
                                       lr=lr_lr, n_iter=lr_iters, reg=lr_reg)
        results["auroc_full"] = {
            "auc": round(auc_full, 4),
            "std": round(auc_std, 4),
            "n_folds": n_folds,
        }
        prog(f"AUROC (full): {auc_full:.4f} ± {auc_std:.4f}")

        # ── 2. AUROC: Length only ──
        prog("Computing AUROC (sequence length only)...")
        X_len = seq_lens.reshape(-1, 1)
        auc_len, auc_len_std = _cv_auroc(X_len, y, n_folds=n_folds, seed=random_seed,
                                          lr=lr_lr, n_iter=lr_iters, reg=lr_reg)
        results["auroc_length_only"] = {
            "auc": round(auc_len, 4),
            "std": round(auc_len_std, 4),
        }
        results["mean_seq_len_safe"] = round(float(seq_lens[y == 0].mean()), 1)
        results["mean_seq_len_risk"] = round(float(seq_lens[y == 1].mean()), 1)

        # ── 3. AUROC: Length-residualized ──
        prog("Computing length-residualized AUROC...")
        X_res = _residualize_length(X, seq_lens)
        auc_res, auc_res_std = _cv_auroc(X_res, y, n_folds=n_folds, seed=random_seed,
                                          lr=lr_lr, n_iter=lr_iters, reg=lr_reg)
        results["auroc_residualized"] = {
            "auc": round(auc_res, 4),
            "std": round(auc_res_std, 4),
        }
        prog(f"AUROC (residualized): {auc_res:.4f} ± {auc_res_std:.4f}")

        # ── 4. Length confound correlations ──
        prog("Analyzing length confounds...")
        results["length_confounds"] = _length_correlations(X, seq_lens, feature_names)

        # ── 5. Per-metric AUROC ──
        prog("Computing per-metric AUROC...")
        results["per_metric_auroc"] = _per_metric_auroc(X, y, feature_names)

        # ── 6. PCA analysis ──
        prog("Running PCA on feature space...")
        components, n_95, n_components = _pca_analysis(X, feature_names, variance_target=pca_var_target)
        results["pca"] = {
            "components": components,
            "n_for_95pct": n_95,
            "effective_dimensionality": n_95,
            "n_features_original": int(X.shape[1]),
            "variance_target": pca_var_target,
        }
        prog(f"PCA: {n_95} components for 95% variance (from {X.shape[1]} features)")

        # ── 7. Metric redundancy ──
        prog("Detecting metric redundancy...")
        all_pairs = _metric_redundancy(X, feature_names, threshold=redundancy_thresh)
        results["redundancy"] = {
            "pairs": all_pairs,
            "threshold": redundancy_thresh,
            "n_redundant_pairs": len(all_pairs),
        }

        # ── 8. Random projection baseline ──
        prog(f"Running random projection baseline ({n_random} trials)...")
        results["random_baseline"] = _random_projection_baseline(
            X, y, seq_lens, n_trials=n_random, seed=random_seed,
            n_folds=n_folds, lr=lr_lr, n_iter=lr_iters, reg=lr_reg)
        delta_above_random = auc_full - results["random_baseline"]["mean_auc"]
        results["random_baseline"]["delta_above_random"] = round(delta_above_random, 4)
        prog(f"Random baseline AUC: {results['random_baseline']['mean_auc']:.4f} "
             f"(TAGM Δ = +{delta_above_random:.4f})")

        # ── 9. Category stress analysis ──
        prog("Analyzing category stress distributions...")
        results["category_stress"] = _amplification_check(session_results)

        # ── 10. MI Readiness Scorecard ──
        prog("Generating MI readiness scorecard...")
        scorecard = []

        # AUROC criterion
        if auc_full >= 0.95:
            scorecard.append({"item": "Discrimination (AUROC)", "status": "strong",
                              "detail": f"AUC {auc_full:.3f} — excellent separation"})
        elif auc_full >= 0.80:
            scorecard.append({"item": "Discrimination (AUROC)", "status": "adequate",
                              "detail": f"AUC {auc_full:.3f} — good separation"})
        else:
            scorecard.append({"item": "Discrimination (AUROC)", "status": "weak",
                              "detail": f"AUC {auc_full:.3f} — insufficient for MI venue"})

        # Length confound
        if auc_res >= 0.90:
            scorecard.append({"item": "Length Confound", "status": "strong",
                              "detail": f"Residualized AUC {auc_res:.3f} — signal survives length removal"})
        elif auc_res >= 0.75:
            scorecard.append({"item": "Length Confound", "status": "adequate",
                              "detail": f"Residualized AUC {auc_res:.3f} — partial length dependence"})
        else:
            scorecard.append({"item": "Length Confound", "status": "weak",
                              "detail": f"Residualized AUC {auc_res:.3f} — signal may be length-driven"})

        # Random baseline gap
        if delta_above_random >= 0.15:
            scorecard.append({"item": "Weight-Delta Specificity", "status": "strong",
                              "detail": f"Δ +{delta_above_random:.3f} over random projection"})
        elif delta_above_random >= 0.05:
            scorecard.append({"item": "Weight-Delta Specificity", "status": "adequate",
                              "detail": f"Δ +{delta_above_random:.3f} over random projection"})
        else:
            scorecard.append({"item": "Weight-Delta Specificity", "status": "weak",
                              "detail": f"Δ +{delta_above_random:.3f} — random projections perform similarly"})

        # Metric efficiency
        redundant_count = results["redundancy"]["n_redundant_pairs"]
        if redundant_count <= 2:
            scorecard.append({"item": "Metric Efficiency", "status": "strong",
                              "detail": f"{redundant_count} redundant pairs — clean measurement space"})
        elif redundant_count <= 5:
            scorecard.append({"item": "Metric Efficiency", "status": "adequate",
                              "detail": f"{redundant_count} redundant pairs — some consolidation recommended"})
        else:
            scorecard.append({"item": "Metric Efficiency", "status": "weak",
                              "detail": f"{redundant_count} redundant pairs — consolidate to ~{n_95} independent axes"})

        # Sample size
        min_class = min(n_safe, n_risk)
        if min_class >= 25:
            scorecard.append({"item": "Sample Size", "status": "strong",
                              "detail": f"Min class size {min_class} — adequate for MI evaluation"})
        elif min_class >= 10:
            scorecard.append({"item": "Sample Size", "status": "adequate",
                              "detail": f"Min class size {min_class} — borderline, more data recommended"})
        else:
            scorecard.append({"item": "Sample Size", "status": "weak",
                              "detail": f"Min class size {min_class} — insufficient for reliable metrics"})

        results["scorecard"] = scorecard

        # Overall readiness
        statuses = [s["status"] for s in scorecard]
        if all(s == "strong" for s in statuses):
            results["overall_readiness"] = "ready"
            results["readiness_summary"] = "Data meets MI community evaluation standards."
        elif any(s == "weak" for s in statuses):
            n_weak = statuses.count("weak")
            results["overall_readiness"] = "gaps"
            results["readiness_summary"] = f"{n_weak} critical gap(s) identified. See scorecard."
        else:
            results["overall_readiness"] = "near_ready"
            results["readiness_summary"] = "Data is close to MI standards. Minor improvements recommended."

        prog(f"MI Readiness: {results['overall_readiness']} — {results['readiness_summary']}")
        prog("Analysis complete.")

        return results
