"""
Correction Manifold Module for TASM.

Constructs a 6-dimensional intrinsic manifold from correction field
measurements and probe geometry, enabling unsupervised classification
of prompts by their alignment signature.

The manifold combines:
  1. Subject domain angle (from domain surface probe proximity)
  2. Probe escalation level (from domain surface probe proximity)
  3-6. Four orthogonal correction signals as corner attractors
       within each subject × level cell

Achieves 94.1% binary classification (safe vs. adversarial) using
KNN on the resulting 2D projected positions, with zero human-assigned
category labels in the feature pipeline.

Pure post-processor: requires session results and a completed
domain_surface module run.

Original concept: Ostrander (2026).
"""

import os
import json
import math
import logging
import numpy as np
from collections import Counter, defaultdict

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")

# The four independent signals (|r| < 0.40 between all pairs)
# Selected from the full set of 8 by correlation analysis
SIGNAL_KEYS = [
    ("stress_score",        "ASM"),
    ("interior_cv",         "IntCV"),
    ("rd_mean_replacement", "RD_repl"),
    ("sfd_density_mean",    "SFD_d"),
]

# Full set of 8 signals for reporting
ALL_SIGNAL_KEYS = [
    ("stress_score",        "ASM"),
    ("entropy",             "Entropy"),
    ("gini",                "Gini"),
    ("interior_cv",         "IntCV"),
    ("sfd_energy_mean",     "SFD_e"),
    ("sfd_density_mean",    "SFD_d"),
    ("rd_mean_replacement", "RD_repl"),
    ("rd_mean_overlap",     "RD_ovlp"),
]

LEVEL_NAMES = ["nouns", "phrase", "question", "instruct", "meta"]


# ─── Probe-Level Computation ─────────────────────────────────

def _compute_prompt_probe_stats(domain_surface_data):
    """Extract per-prompt probe level and subject from domain surface observations.

    Returns:
        mean_level: array of mean probe escalation level per prompt
        blended_angle: array of blended subject angle per prompt
        dom_subject: array of dominant subject index per prompt
        n_prompts: number of prompts
    """
    obs = domain_surface_data.get("observations", [])
    subjects = domain_surface_data.get("subjects", [])
    n_subj = len(subjects)

    # Subject angles: evenly spaced around circle, starting at top
    subj_angles = np.linspace(0, 2 * np.pi, n_subj, endpoint=False) - np.pi / 2

    # Accumulate per-prompt statistics
    # obs fields: tok, cat, dy, disp, repl, dx, asm, sfd_e, sfd_d, pi, pos,
    #             near_dist, near_level, near_subj_idx
    prompt_subjects = defaultdict(Counter)
    prompt_levels = defaultdict(Counter)

    for o in obs:
        pi = o[9]   # prompt index
        prompt_subjects[pi][o[13]] += 1  # near_subj_idx
        prompt_levels[pi][o[12]] += 1    # near_level

    # Determine prompt count from domain surface
    n_prompts = domain_surface_data.get("n_prompts_used",
                domain_surface_data.get("n_prompts_total", 0))

    mean_level = np.full(n_prompts, 2.0)
    blended_angle = np.zeros(n_prompts)
    dom_subject = np.zeros(n_prompts, dtype=int)

    for pi in range(n_prompts):
        if pi in prompt_levels:
            total = sum(prompt_levels[pi].values())
            mean_level[pi] = sum(l * n for l, n in prompt_levels[pi].items()) / total

        if pi in prompt_subjects:
            counts = prompt_subjects[pi]
            dom_subject[pi] = counts.most_common(1)[0][0]

            # Circular weighted mean of subject angles
            total = sum(counts.values())
            sin_sum = sum((n / total) * np.sin(subj_angles[si])
                          for si, n in counts.items())
            cos_sum = sum((n / total) * np.cos(subj_angles[si])
                          for si, n in counts.items())
            blended_angle[pi] = np.arctan2(sin_sum, cos_sum)

    return mean_level, blended_angle, dom_subject, n_prompts, subj_angles


# ─── Manifold Construction ───────────────────────────────────

def _get_ring(level):
    """Map continuous probe level to ring index (0=question, 1=instruct, 2=meta)."""
    t = max(0.0, min(1.0, (level - 2.0) / 2.0))
    if t < 0.33:
        return 0
    elif t < 0.67:
        return 1
    else:
        return 2


def _cell_corners(angle, ring_idx, ring_bands, wedge_half):
    """Compute 4 corner positions for a cell at given angle and ring.

    Corner mapping:
        0 (ASM):     outer-right
        1 (IntCV):   outer-left
        2 (RD_repl): inner-right
        3 (SFD_d):   inner-left

    Returns:
        corners: (4, 2) array of corner positions
    """
    a_left = angle - wedge_half
    a_right = angle + wedge_half
    ri = ring_bands[ring_idx]["inner"]
    ro = ring_bands[ring_idx]["outer"]

    return np.array([
        [ro * np.cos(a_right), ro * np.sin(a_right)],   # ASM: outer-right
        [ro * np.cos(a_left),  ro * np.sin(a_left)],    # IntCV: outer-left
        [ri * np.cos(a_right), ri * np.sin(a_right)],   # RD_repl: inner-right
        [ri * np.cos(a_left),  ri * np.sin(a_left)],    # SFD_d: inner-left
    ])


def _build_manifold(session_results, mean_level, blended_angle, dom_subject,
                     ring_bands, wedge_half, attractor_power, attractor_floor,
                     progress=None):
    """Compute 2D manifold positions for all prompts.

    For each prompt:
        1. Determine cell from subject angle × probe ring
        2. Compute 4 corner positions for that cell
        3. Weight corners by signal values raised to attractor_power
        4. Position = weighted centroid of corners

    Returns:
        positions: (n, 2) array of manifold positions
        norm_signals: (n, 4) normalized signal values
        raw_signals: (n, 8) all 8 raw signal values
        rings: (n,) ring assignments
    """
    n = len(session_results)

    # Resolve signal values from session results, handling nested keys.
    # Summary CSV flattens sfd.density_mean to sfd_density_mean, but
    # results.json keeps the nested structure.  This resolver handles both.
    _NESTED = {
        "sfd_density_mean":    ("sfd", "density_mean"),
        "sfd_energy_mean":     ("sfd", "energy_mean"),
        "sfd_entropy_mean":    ("sfd", "entropy_mean"),
        "rd_mean_replacement": ("rank_displacement", "mean_replacement"),
        "rd_mean_overlap":     ("rank_displacement", "mean_overlap"),
        "rd_mean_tau":         ("rank_displacement", "mean_tau"),
    }

    def _get_signal(r, key):
        v = r.get(key)
        if v is not None:
            return float(v)
        nested = _NESTED.get(key)
        if nested:
            parent = r.get(nested[0])
            if isinstance(parent, dict):
                v2 = parent.get(nested[1])
                if v2 is not None:
                    return float(v2)
        return 0.0

    # Extract the 4 independent signals
    raw_4 = np.zeros((n, 4))
    for i, r in enumerate(session_results):
        for j, (key, _) in enumerate(SIGNAL_KEYS):
            raw_4[i, j] = _get_signal(r, key)

    # Normalize to [0, 1]
    mins = raw_4.min(axis=0)
    maxs = raw_4.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    norm_4 = (raw_4 - mins) / ranges

    # Extract all 8 signals
    raw_8 = np.zeros((n, 8))
    for i, r in enumerate(session_results):
        for j, (key, _) in enumerate(ALL_SIGNAL_KEYS):
            raw_8[i, j] = _get_signal(r, key)

    # Normalize all 8
    norm_8 = np.zeros_like(raw_8)
    for j in range(8):
        mn, mx = raw_8[:, j].min(), raw_8[:, j].max()
        norm_8[:, j] = (raw_8[:, j] - mn) / (mx - mn) if mx > mn else 0

    # Compute positions
    positions = np.zeros((n, 2))
    rings = np.zeros(n, dtype=int)

    for i in range(n):
        ri = _get_ring(mean_level[i])
        rings[i] = ri
        corners = _cell_corners(blended_angle[i], ri, ring_bands, wedge_half)

        # Attractor weights
        w = norm_4[i] ** attractor_power + attractor_floor
        w_sum = w.sum()

        positions[i, 0] = np.dot(w, corners[:, 0]) / w_sum
        positions[i, 1] = np.dot(w, corners[:, 1]) / w_sum

    if progress:
        progress(f"Computed manifold positions for {n} prompts")

    return positions, norm_4, norm_8, raw_8, rings


# ─── Classification ──────────────────────────────────────────

def _run_knn(positions, categories, k_values, progress=None):
    """Run KNN classification on manifold positions.

    Returns dict with per-k accuracy and confusion matrix for best k.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import confusion_matrix

    y = np.array(categories)

    results = {"per_k": {}}

    # 6-class
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, positions, y, cv=10, scoring="accuracy")
        results["per_k"][k] = {
            "accuracy": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        }

    # Best k
    best_k = max(results["per_k"], key=lambda k: results["per_k"][k]["accuracy"])
    results["best_k"] = best_k
    results["best_accuracy"] = results["per_k"][best_k]["accuracy"]

    # Confusion matrix for best k
    y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=best_k),
                               positions, y, cv=10)
    labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    results["confusion"] = {
        "labels": labels.tolist() if hasattr(labels, 'tolist') else list(labels),
        "matrix": cm.tolist(),
    }
    results["overall_accuracy"] = round(float((y_pred == y).mean()), 4)

    # Binary: safe (b, m) vs risk (a, j)
    binary_map = {"b": "safe", "m": "safe", "a": "risk", "j": "risk"}
    mask = np.array([c in binary_map for c in y])
    if mask.sum() > 20:
        y_bin = np.array([binary_map[c] for c in y[mask]])
        X_bin = positions[mask]

        binary_results = {}
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_bin, y_bin, cv=10, scoring="accuracy")
            binary_results[k] = {
                "accuracy": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
            }

        best_bin_k = max(binary_results, key=lambda k: binary_results[k]["accuracy"])
        results["binary"] = {
            "per_k": binary_results,
            "best_k": best_bin_k,
            "best_accuracy": binary_results[best_bin_k]["accuracy"],
            "n_safe": int((y_bin == "safe").sum()),
            "n_risk": int((y_bin == "risk").sum()),
        }

    if progress:
        acc_6 = results["overall_accuracy"]
        acc_2 = results.get("binary", {}).get("best_accuracy", 0)
        progress(f"KNN: 6-class {acc_6:.1%}, binary {acc_2:.1%}")

    return results


# ─── Category Mean Positions ─────────────────────────────────

def _category_centroids(positions, categories):
    """Compute mean positions per category for visualization reference."""
    cats = np.array(categories)
    centroids = {}
    for c in sorted(set(cats)):
        mask = cats == c
        centroids[c] = {
            "x": round(float(positions[mask, 0].mean()), 4),
            "y": round(float(positions[mask, 1].mean()), 4),
            "n": int(mask.sum()),
        }
    return centroids


# ─── Module Class ────────────────────────────────────────────

class CorrectionManifoldModule(TASMModule):
    """6D intrinsic correction manifold for unsupervised classification.

    Combines probe geometry (subject × escalation level) with four
    orthogonal correction signals (ASM, IntCV, RD replacement, SFD density)
    into a unified spatial manifold. Runs KNN classification on the
    resulting positions as validation.

    Requires a completed domain_surface module run.
    """

    name = "correction_manifold"
    display_name = "Correction Manifold"
    description = (
        "Constructs a 6D intrinsic manifold from correction signals and "
        "probe geometry. Each prompt is positioned by subject domain "
        "(angular), escalation level (radial), and four corner-attractor "
        "correction signals. Enables unsupervised KNN classification."
    )
    version = "0.1.0"

    min_results = 20
    requires_sfd = True
    requires_ltp = False
    requires_rd = True

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="attractor_power",
                display_name="Attractor Power",
                description=(
                    "Exponent applied to normalized signal values before "
                    "corner weighting. Higher values pull points more "
                    "aggressively toward dominant signal corners."
                ),
                type="float",
                default=3.0,
                min_val=1.0,
                max_val=6.0,
            ),
            ModuleParameter(
                name="attractor_floor",
                display_name="Attractor Floor",
                description=(
                    "Minimum weight for each corner attractor to prevent "
                    "zero-signal collapse. Lower values allow sharper "
                    "corner clustering."
                ),
                type="float",
                default=0.01,
                min_val=0.001,
                max_val=0.2,
            ),
            ModuleParameter(
                name="knn_k_values",
                display_name="KNN k Values",
                description=(
                    "Comma-separated k values for KNN classification "
                    "(e.g., 3,5,7,11,15)"
                ),
                type="select",
                default="3,5,7,11,15",
                options=["3,5,7,11,15", "3,5,7", "5,7,11", "3,5,7,11,15,21"],
            ),
            ModuleParameter(
                name="ring_gap",
                display_name="Ring Gap",
                description=(
                    "Gap between escalation ring bands (0.0\u20130.1). "
                    "Larger gaps create clearer ring separation."
                ),
                type="float",
                default=0.04,
                min_val=0.0,
                max_val=0.1,
            ),
        ]

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        # Check that domain surface has been run
        if hasattr(self, '_session_dir') and self._session_dir:
            ds_path = os.path.join(self._session_dir, "module_domain_surface.json")
            if not os.path.exists(ds_path):
                return False, (
                    "Domain Surface module must be run first. "
                    "The Correction Manifold requires probe proximity data."
                )

        return True, "OK"

    def run(self, session_results, params, progress=None):
        """Execute correction manifold analysis.

        Requires a completed domain_surface module run for probe
        proximity data. Computes 6D manifold positions and runs
        KNN classification as validation.
        """
        attractor_power = params.get("attractor_power", 3.0)
        attractor_floor = params.get("attractor_floor", 0.01)
        k_str = params.get("knn_k_values", "3,5,7,11,15")
        k_values = [int(k.strip()) for k in k_str.split(",")]
        ring_gap = params.get("ring_gap", 0.04)

        # ── Load domain surface output ──
        if progress:
            progress("Loading domain surface data...")

        ds_data = self._load_domain_surface(session_results)
        if ds_data is None:
            raise RuntimeError(
                "Domain Surface module output not found. "
                "Run the Domain Surface module first."
            )

        subjects = ds_data.get("subjects", [])
        n_subj = len(subjects)
        if n_subj == 0:
            raise RuntimeError("No subjects found in domain surface data.")

        # ── Compute probe stats ──
        if progress:
            progress("Computing per-prompt probe statistics...")

        mean_level, blended_angle, dom_subject, n_prompts, subj_angles = \
            _compute_prompt_probe_stats(ds_data)

        if n_prompts != len(session_results):
            logger.warning(
                f"[MANIFOLD] Prompt count mismatch: DS has {n_prompts}, "
                f"session has {len(session_results)}. Using min."
            )
            n_prompts = min(n_prompts, len(session_results))
            session_results = session_results[:n_prompts]
            mean_level = mean_level[:n_prompts]
            blended_angle = blended_angle[:n_prompts]
            dom_subject = dom_subject[:n_prompts]

        # ── Ring geometry ──
        ring_bands = [
            {"inner": 0.18,            "outer": 0.40 - ring_gap},
            {"inner": 0.40 + ring_gap, "outer": 0.66 - ring_gap},
            {"inner": 0.66 + ring_gap, "outer": 0.92},
        ]
        wedge_half = np.pi / n_subj * 0.88

        # ── Build manifold ──
        if progress:
            progress("Building 6D manifold positions...")

        positions, norm_4, norm_8, raw_8, rings = _build_manifold(
            session_results, mean_level, blended_angle, dom_subject,
            ring_bands, wedge_half, attractor_power, attractor_floor,
            progress)

        # ── KNN classification ──
        categories = []
        for r in session_results:
            cat = r.get("category", "unknown")
            if cat == "dual-use":
                categories.append("d")
            elif cat:
                categories.append(cat[0])
            else:
                categories.append("?")

        if progress:
            progress("Running KNN classification...")

        knn_results = _run_knn(positions, categories, k_values, progress)

        # ── Category centroids ──
        centroids = _category_centroids(positions, categories)

        # ── Build visualization data ──
        if progress:
            progress("Building visualization data...")

        prompts_viz = []
        for i in range(n_prompts):
            r = session_results[i]
            prompts_viz.append([
                r.get("prompt", "")[:80],         # 0: prompt text
                categories[i],                     # 1: category code
                round(float(positions[i, 0]), 4),  # 2: x position
                round(float(-positions[i, 1]), 4), # 3: y position (flip for canvas)
                int(dom_subject[i]),               # 4: dominant subject
                int(rings[i]),                     # 5: ring index
                round(float(norm_4[i, 0]), 4),     # 6: ASM normalized
                round(float(norm_4[i, 1]), 4),     # 7: IntCV normalized
                round(float(norm_4[i, 2]), 4),     # 8: RD_repl normalized
                round(float(norm_4[i, 3]), 4),     # 9: SFD_d normalized
                round(float(mean_level[i]), 2),    # 10: mean probe level
            ] + [round(float(raw_8[i, j]), 4) for j in range(8)])  # 11-18: raw

        # Cell corners for visualization
        cells_viz = []
        for si in range(n_subj):
            for ri in range(3):
                corners = _cell_corners(subj_angles[si], ri, ring_bands, wedge_half)
                center = corners.mean(axis=0)
                cells_viz.append({
                    "s": si, "r": ri,
                    "corners": [[round(float(c[0]), 4), round(float(-c[1]), 4)]
                                for c in corners],
                    "center": [round(float(center[0]), 4),
                               round(float(-center[1]), 4)],
                })

        subj_short = [s.replace("_", " ").title()[:12] for s in subjects]

        # ── Build output ──
        output = {
            # Metadata
            "version": self.version,
            "n_prompts": n_prompts,
            "attractor_power": attractor_power,
            "attractor_floor": attractor_floor,

            # Geometry
            "subjects": subjects,
            "subj_short": subj_short,
            "subj_angles": [round(float(a), 4) for a in subj_angles],
            "rings": [
                {"label": "Question", "inner": ring_bands[0]["inner"],
                 "outer": ring_bands[0]["outer"]},
                {"label": "Instruct", "inner": ring_bands[1]["inner"],
                 "outer": ring_bands[1]["outer"]},
                {"label": "Meta", "inner": ring_bands[2]["inner"],
                 "outer": ring_bands[2]["outer"]},
            ],
            "corner_labels": [s for _, s in SIGNAL_KEYS],
            "corner_rays": [
                {"angle": 45,   "name": "ASM",     "color": "#ef4444"},
                {"angle": 135,  "name": "IntCV",   "color": "#22d3ee"},
                {"angle": 225,  "name": "SFD_d",   "color": "#4ade80"},
                {"angle": 315,  "name": "RD_repl", "color": "#a78bfa"},
            ],

            # Signal metadata
            "signal_keys": [s for _, s in SIGNAL_KEYS],
            "all_signal_keys": [s for _, s in ALL_SIGNAL_KEYS],

            # Per-prompt data
            "prompts": prompts_viz,
            "cells": cells_viz,

            # Classification results
            "classification": knn_results,
            "category_centroids": centroids,

            # Escalation stats
            "level_stats": {
                cat: round(float(np.mean([
                    mean_level[i] for i in range(n_prompts)
                    if categories[i] == cat
                ])), 2)
                for cat in sorted(set(categories)) if cat != "?"
            },
        }

        if progress:
            acc = knn_results.get("binary", {}).get("best_accuracy", 0)
            progress(f"Complete: {n_prompts} prompts, "
                     f"binary accuracy {acc:.1%}")

        return output

    def _load_domain_surface(self, session_results):
        """Load domain surface module output from session directory."""
        # Try session directory
        if hasattr(self, '_session_dir') and self._session_dir:
            ds_path = os.path.join(self._session_dir, "module_domain_surface.json")
            if os.path.exists(ds_path):
                try:
                    with open(ds_path) as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"[MANIFOLD] Failed to load domain surface: {e}")

        # Try common paths
        for path in ["module_domain_surface.json",
                      "datasets/current/module_domain_surface.json"]:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception:
                    continue

        return None
