"""Shared metric primitives and the canonical prompt-category taxonomy.

Why this module exists
──────────────────────
Before it, the analysis modules each carried their own copy of the same few
primitives, and the copies had DIVERGED:

* Five separate Wilcoxon-Mann-Whitney AUROC implementations.  Two of them
  (``mechanistic_interpretability._auroc`` and ``routing_ablation._auroc``)
  had no small-n guard, and one of those two also took its arguments in the
  opposite order.  The practical consequence was in ``routing_ablation``,
  where a 1-vs-1 holdout produced a ``holdout_auroc`` of exactly 0.0 or 1.0
  and that value drove a boolean scientific claim
  (``"harm_probe_survived": holdout_auroc > 0.70``).  Under the other three
  implementations the identical input returned 0.5.

* Six different harm/safe category vocabularies.  ``unknown`` was treated as
  harmful in two modules and dropped in four; ``mild`` was safe in three and
  silently dropped in another.  Two modules running against the same session
  would report contradictory ``n_harm``.

Everything here is intentionally small and dependency-light so that every
module can import it without circularity.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

# ── Category taxonomy ───────────────────────────────────────────

# The union of every category string the six previous per-module vocabularies
# recognised, resolved into ONE assignment.  Anything not listed is UNKNOWN and
# is excluded from harm/safe contrasts rather than being guessed at.
HARM_CATEGORIES = frozenset({
    "harmful", "jailbreak", "adversarial", "dual-use", "dual_use",
    "risk", "dangerous",
})

SAFE_CATEGORIES = frozenset({
    "benign", "safe", "mild", "neutral", "baseline", "user_baseline",
})

# Deliberately neither: "unknown" was harmful in two modules and dropped in
# four.  Treating an unlabelled prompt as harmful silently inflates n_harm, so
# it is excluded here and callers that want it must say so explicitly.
AMBIGUOUS_CATEGORIES = frozenset({"unknown", "", None})


def category_class(category: Optional[str]) -> Optional[str]:
    """Map a raw category string to ``"harm"``, ``"safe"``, or ``None``.

    ``None`` means "do not use this prompt in a harm/safe contrast".
    """
    if category is None:
        return None
    c = category.strip().lower()
    if c in HARM_CATEGORIES:
        return "harm"
    if c in SAFE_CATEGORIES:
        return "safe"
    return None


def split_by_class(items: Iterable, key=lambda r: getattr(r, "category", None)):
    """Partition an iterable into (harm, safe, unclassified) lists."""
    harm, safe, other = [], [], []
    for it in items:
        cls = category_class(key(it))
        (harm if cls == "harm" else safe if cls == "safe" else other).append(it)
    return harm, safe, other


# ── AUROC ───────────────────────────────────────────────────────

# Below this many samples total, a rank statistic is not interpretable.
MIN_AUROC_SAMPLES = 4
# ...and below this many in EITHER class independently.  A 1-vs-n comparison
# clears the total guard but can still only return 0.0 or 1.0.
MIN_CLASS_SAMPLES = 2


def auroc(scores: Sequence[float], labels: Sequence[int],
          min_samples: int = MIN_AUROC_SAMPLES) -> float:
    """Wilcoxon-Mann-Whitney AUROC. ``labels`` is 1 for the positive class.

    Argument order is (scores, labels) — note that one of the implementations
    this replaces used (y_true, y_score), so check call sites when migrating.

    Returns 0.5 (chance) when the sample is too small to support a rank
    statistic, rather than a spuriously perfect 0.0/1.0.  The guard is applied
    BOTH to the total and PER CLASS: a 1-vs-3 split passes a total-only
    check yet can still only return 0.0 or 1.0, which is precisely the
    degenerate case this function exists to prevent.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    if len(s) < min_samples:
        return 0.5
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) < MIN_CLASS_SAMPLES or len(neg) < MIN_CLASS_SAMPLES:
        return 0.5
    # Rank-based form, ties handled by average ranks.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_with_n(scores: Sequence[float], labels: Sequence[int],
                 min_samples: int = MIN_AUROC_SAMPLES) -> dict:
    """AUROC plus the sample sizes and an explicit under-powered flag.

    Prefer this over the bare float wherever the value feeds a verdict: a
    reader cannot tell an AUROC of 1.0 computed on n=2 from one computed on
    n=200, and the former is what a 1-vs-1 holdout produces.
    """
    y = np.asarray(labels)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "auroc": auroc(scores, labels, min_samples=min_samples),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "underpowered": bool(n_pos + n_neg < min_samples
                             or n_pos < 2 or n_neg < 2),
    }


def permutation_auroc_p(scores: Sequence[float], labels: Sequence[int],
                        n_perm: int = 200, seed: int = 42) -> Optional[float]:
    """One-sided permutation p-value for an observed AUROC.

    Shuffles the labels and recomputes, so it absorbs whatever selection the
    scoring procedure performed. Returns None when the sample is too small.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if len(s) < MIN_AUROC_SAMPLES or len(set(y.tolist())) < 2:
        return None
    observed = auroc(s, y)
    rng = np.random.default_rng(seed)
    perm = y.copy()
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(perm)
        if auroc(s, perm) >= observed:
            hits += 1
    return float((1 + hits) / (1 + n_perm))


# ── Vector helpers ──────────────────────────────────────────────

NORM_EPS = 1e-12


def unit(v: np.ndarray, eps: float = NORM_EPS) -> np.ndarray:
    """L2-normalize, guarding a zero vector (returns it unchanged)."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n < eps else v / n


def cosine(a: np.ndarray, b: np.ndarray, eps: float = NORM_EPS) -> float:
    """Cosine similarity with a norm guard. Returns 0.0 if either side is null."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    val = float(np.dot(a, b) / (na * nb))
    if math.isnan(val):
        return 0.0
    return max(-1.0, min(1.0, val))


def mean_difference_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Unit difference-of-means direction (the 'Arditi' refusal direction).

    NOTE: a direction fitted this way must be EVALUATED ON HELD-OUT DATA.
    Scoring the same rows it was fitted to measures the fit, not the effect —
    and a random-projection control does not correct for it, because a random
    direction was never fitted to the labels at all.  Use
    ``fit_direction_holdout`` below, or permute the labels and refit.
    """
    return unit(np.asarray(pos).mean(axis=0) - np.asarray(neg).mean(axis=0))


def fit_direction_holdout(pos: np.ndarray, neg: np.ndarray,
                          n_folds: int = 5, seed: int = 42) -> dict:
    """Cross-validated difference-of-means direction with an honest AUROC.

    The direction is refitted inside each fold on the training rows only, then
    scored on the held-out rows of BOTH classes.  Also runs a label-permutation
    null using the identical fit-and-score procedure, which is the correct
    control for a fitted direction.

    Returns
    -------
    {"direction": ndarray,      # fitted on all data, for downstream use
     "cv_auroc": float,         # honest, out-of-fold
     "train_auroc": float,      # in-sample, reported for comparison only
     "p_value": float | None,   # permutation null on cv_auroc
     "n_pos": int, "n_neg": int, "underpowered": bool}
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    n_pos, n_neg = len(pos), len(neg)

    full_dir = mean_difference_direction(pos, neg)
    train_scores = np.concatenate([pos @ full_dir, neg @ full_dir])
    train_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    train_auroc = auroc(train_scores, train_labels)

    result = {
        "direction": full_dir,
        "train_auroc": train_auroc,
        "cv_auroc": 0.5,
        "p_value": None,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "underpowered": bool(n_pos < 2 or n_neg < 2
                             or n_pos + n_neg < MIN_AUROC_SAMPLES),
    }
    if result["underpowered"]:
        return result

    def _cv_auroc(labels_pos_first: np.ndarray) -> float:
        X = np.vstack([pos, neg])
        y = labels_pos_first
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        folds = np.array_split(idx, min(n_folds, len(X)))
        oof_scores = np.zeros(len(X))
        ok = True
        for fold in folds:
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[fold] = False
            ytr = y[train_mask]
            if ytr.sum() == 0 or ytr.sum() == len(ytr):
                ok = False
                break
            d = mean_difference_direction(X[train_mask][ytr == 1],
                                          X[train_mask][ytr == 0])
            oof_scores[fold] = X[fold] @ d
        if not ok:
            return 0.5
        return auroc(oof_scores, y)

    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    result["cv_auroc"] = _cv_auroc(y_true)

    # Permutation null: same fit-and-score pipeline, shuffled labels.
    rng = np.random.default_rng(seed + 1)
    perm = y_true.copy()
    n_perm = 100
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(perm)
        if _cv_auroc(perm) >= result["cv_auroc"]:
            hits += 1
    result["p_value"] = float((1 + hits) / (1 + n_perm))
    return result
