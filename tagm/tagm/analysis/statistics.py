"""Shared statistical utilities for TAGM analyses.

Bootstrap confidence intervals, Cohen's d, and small-n friendly
aggregation patterns. Translated from TASM's `engine/statistics.py`.
Uses numpy only; scipy is optional (Kendall's tau imports locally where used).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import numpy as np


def _safe_float(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if (math.isnan(v) or math.isinf(v)) else v


def _clean(vals: Iterable) -> list[float]:
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not (math.isnan(f) or math.isinf(f)):
            out.append(f)
    return out


def cohens_d(group_a: Iterable, group_b: Iterable,
             min_samples: int = 2) -> float:
    """Cohen's d effect size with pooled SD. Returns 0.0 if groups are too small."""
    a = np.array(_clean(group_a))
    b = np.array(_clean(group_b))
    if len(a) < min_samples or len(b) < min_samples:
        return 0.0
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled == 0:
        return 0.0
    return _safe_float(abs(a.mean() - b.mean()) / pooled)


def bootstrap_ci(values: Iterable,
                 n_boot: int = 5000,
                 ci: float = 0.95,
                 statistic: Callable = np.mean,
                 seed: int = 42) -> dict:
    """Bootstrap confidence interval for a statistic over `values`.

    Returns a dict: {'estimate', 'ci_low', 'ci_high', 'n'}.
    Handles n<2 gracefully by returning a degenerate interval at the
    value itself.
    """
    v = np.array(_clean(list(values)))
    if len(v) < 1:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    if len(v) < 2:
        val = _safe_float(statistic(v))
        return {"estimate": val, "ci_low": val, "ci_high": val, "n": len(v)}

    rng = np.random.default_rng(seed)
    boot = np.array([statistic(rng.choice(v, size=len(v), replace=True))
                     for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return {
        "estimate": _safe_float(statistic(v)),
        "ci_low": _safe_float(np.percentile(boot, alpha * 100)),
        "ci_high": _safe_float(np.percentile(boot, (1 - alpha) * 100)),
        "n": int(len(v)),
    }


def bootstrap_effect_size(group_a: Iterable, group_b: Iterable,
                           n_boot: int = 5000, ci: float = 0.95,
                           min_samples: int = 2,
                           seed: int = 42) -> dict:
    """Bootstrap CI for Cohen's d effect size."""
    a = np.array(_clean(group_a))
    b = np.array(_clean(group_b))
    if len(a) < min_samples or len(b) < min_samples:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_a": len(a), "n_b": len(b)}

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        pooled = np.sqrt((sa.std(ddof=1)**2 + sb.std(ddof=1)**2) / 2)
        if pooled == 0:
            boot.append(0.0)
        else:
            boot.append(abs(sa.mean() - sb.mean()) / pooled)
    boot = np.array(boot)
    alpha = (1 - ci) / 2
    return {
        "estimate": cohens_d(a, b, min_samples=min_samples),
        "ci_low": _safe_float(np.percentile(boot, alpha * 100)),
        "ci_high": _safe_float(np.percentile(boot, (1 - alpha) * 100)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def optimal_threshold(group_a: Iterable, group_b: Iterable,
                       n_steps: int = 500) -> dict:
    """Find the threshold that maximally separates group_a from group_b.

    Scans n_steps thresholds between min and max; returns the threshold
    that maximizes correct classification (both groups weighted equally).
    """
    a = np.array(_clean(group_a))
    b = np.array(_clean(group_b))
    if len(a) < 1 or len(b) < 1:
        return {"threshold": 0.0, "accuracy": 0.5, "direction": "none"}

    combined = np.concatenate([a, b])
    lo = float(combined.min())
    hi = float(combined.max())
    if lo == hi:
        return {"threshold": lo, "accuracy": 0.5, "direction": "none"}

    # Try both directions: a > threshold and a < threshold
    best = {"threshold": lo, "accuracy": 0.0, "direction": "gt"}
    for t in np.linspace(lo, hi, n_steps):
        acc_gt = (0.5 * ((a > t).sum() / len(a)) +
                  0.5 * ((b <= t).sum() / len(b)))
        acc_lt = (0.5 * ((a < t).sum() / len(a)) +
                  0.5 * ((b >= t).sum() / len(b)))
        if acc_gt > best["accuracy"]:
            best = {"threshold": float(t), "accuracy": float(acc_gt), "direction": "gt"}
        if acc_lt > best["accuracy"]:
            best = {"threshold": float(t), "accuracy": float(acc_lt), "direction": "lt"}
    return best


# ── Session traversal helpers ───────────────────────────────────────

def extract_scalar(prompt_record: dict, measurement_name: str,
                    field_name: str) -> Optional[float]:
    """Fetch a scalar value from a prompt's measurement result.

    Traverses: prompt['measurements'][measurement_name]['scalars'][field_name].
    Returns None if any part of the path is absent or the value is None.
    """
    ms = (prompt_record.get("measurements") or {}).get(measurement_name)
    if not ms:
        return None
    scalars = ms.get("scalars") or {}
    v = scalars.get(field_name)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def group_by_category(session: dict) -> dict[str, list[dict]]:
    """Group session prompts by their 'category' field."""
    out: dict[str, list[dict]] = {}
    for p in session.get("prompts") or []:
        cat = p.get("category") or "uncategorized"
        out.setdefault(cat, []).append(p)
    return out
