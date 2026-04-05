"""
Correction Manifold Module for TASM — The Witness Plate.

Constructs a manifold from correction field measurements and probe
geometry using anchor-repulsor design, enabling unsupervised
classification of prompts by their alignment signature.

The manifold combines:
  1. Subject domain angle (from domain surface probe proximity)
  2. Probe escalation level (ring, from domain surface probe proximity)
  3. RD replacement ratio (radial position within ring)
  4-5. Signal-driven repulsion from anchor points in locally-oriented
       coordinate frames: Entropy/SFD_e (radial), KL/ASM (tangential)

Each probe anchor (subject × level intersection) pushes tokens outward
via its own rotated coordinate frame. "North" = radially outward from
center. Clusters self-organize from signal patterns without global
attractor geometry.

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
    ("entropy",             "Entropy"),      # prompt radial repulsor
    ("kl_divergence",       "KL"),           # prompt tangential repulsor
    ("rd_mean_replacement", "RD_repl"),      # radial position within ring
    ("sfd_energy_mean",     "SFD_e"),        # token radial repulsor (proxy for Entropy)
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

def _get_ring(level, n_levels):
    """Map continuous probe level to ring index for n_levels rings.

    level is a continuous value in [0, n_levels-1].
    Maps linearly to ring indices [0, n_levels-1].
    """
    if n_levels <= 1:
        return 0
    t = max(0.0, min(1.0, level / (n_levels - 1)))
    ring = int(t * (n_levels - 0.01))
    return min(ring, n_levels - 1)


def _make_ring_bands(n_rings, ring_gap=0.04, r_inner=0.18, r_outer=0.92):
    """Generate evenly-spaced ring bands for n_rings rings.

    Returns list of {"inner": float, "outer": float} dicts.
    """
    if n_rings <= 0:
        return []
    total_gap = ring_gap * (n_rings - 1)
    usable = r_outer - r_inner - total_gap
    band_width = usable / n_rings

    bands = []
    cursor = r_inner
    for i in range(n_rings):
        bands.append({"inner": round(cursor, 4), "outer": round(cursor + band_width, 4)})
        cursor += band_width + ring_gap
    return bands


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
                     ring_bands, wedge_half, push_strength,
                     n_levels=5, progress=None):
    """Compute 2D manifold positions for all prompts.

    Anchor-repulsor geometry:
        - Subject angle determines angular direction (wedge)
        - Probe level determines ring band (escalation)
        - RD_repl controls radial position within ring
        - Entropy pushes radially from anchor (outward when high)
        - KL pushes tangentially from anchor (clockwise when high)
        Each anchor has a local coordinate frame: "north" = radially
        outward, "east" = tangential clockwise.

    Returns:
        positions: (n, 2) array of manifold positions
        norm_signals: (n, 4) normalized signal values [Entropy, KL, RD_repl, SFD_e]
        raw_signals: (n, 8) all 8 raw signal values
        rings: (n,) ring assignments
    """
    n = len(session_results)

    # Resolve signal values from session results
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

    # Extract the 4 signals: Entropy[0], KL[1], RD_repl[2], SFD_e[3]
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

    # ── Anchor-repulsor position computation ──
    positions = np.zeros((n, 2))
    rings = np.zeros(n, dtype=int)

    for i in range(n):
        ri = _get_ring(mean_level[i], n_levels)
        rings[i] = ri
        inner = ring_bands[ri]["inner"]
        outer = ring_bands[ri]["outer"]

        # RD controls radial position within ring
        rd_frac = norm_4[i, 2]  # RD_repl normalized
        anchor_r = inner + (outer - inner) * 0.5  # anchor at ring midpoint

        # Subject angle
        angle = blended_angle[i]

        # Anchor position
        ax = anchor_r * np.cos(angle)
        ay = anchor_r * np.sin(angle)

        # Local coordinate frame at this anchor
        rad_x, rad_y = np.cos(angle), np.sin(angle)  # radial outward
        tan_x, tan_y = np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)  # tangential

        # Signal-driven push from anchor
        max_push = (outer - inner) * push_strength
        e_push = (norm_4[i, 0] - 0.5) * 2 * max_push   # Entropy → radial
        k_push = (norm_4[i, 1] - 0.5) * 2 * max_push   # KL → tangential
        rd_push = (rd_frac - 0.5) * max_push * 0.3       # RD → secondary radial

        positions[i, 0] = ax + rad_x * (e_push + rd_push) + tan_x * k_push
        positions[i, 1] = ay + rad_y * (e_push + rd_push) + tan_y * k_push

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
        "Constructs the witness plate from correction signals and "
        "probe geometry. Anchor-repulsor design: each probe anchor "
        "pushes tokens outward via locally-oriented signal axes "
        "(Entropy/SFD_e radial, KL/ASM tangential). Enables "
        "unsupervised KNN classification."
    )
    version = "0.2.0"

    min_results = 20
    requires_sfd = True
    requires_ltp = False
    requires_rd = True

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="push_strength",
                display_name="Push Strength",
                description=(
                    "How strongly signals push tokens away from their "
                    "anchor points. Higher values create more spread "
                    "within each cell, revealing local structure."
                ),
                type="float",
                default=0.55,
                min_val=0.1,
                max_val=1.5,
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
        push_strength = params.get("push_strength", 0.55)
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

        # ── Level names from domain surface ──
        level_names = ds_data.get("level_names", ["nouns", "phrase", "question", "instruct", "meta"])
        n_levels = len(level_names)

        # ── Ring geometry (one ring per escalation level) ──
        ring_bands = _make_ring_bands(n_levels, ring_gap=ring_gap, r_inner=0.10)
        wedge_half = np.pi / n_subj * 0.88

        # ── Build manifold ──
        if progress:
            progress(f"Building manifold: {n_levels} rings × {n_subj} subjects...")

        positions, norm_4, norm_8, raw_8, rings = _build_manifold(
            session_results, mean_level, blended_angle, dom_subject,
            ring_bands, wedge_half, push_strength,
            n_levels=n_levels, progress=progress)

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
                round(float(norm_4[i, 0]), 4),     # 6: Entropy normalized (radial)
                round(float(norm_4[i, 1]), 4),     # 7: KL normalized (tangential)
                round(float(norm_4[i, 2]), 4),     # 8: RD_repl normalized
                round(float(norm_4[i, 3]), 4),     # 9: SFD_e normalized
                round(float(mean_level[i]), 2),    # 10: mean probe level
            ] + [round(float(raw_8[i, j]), 4) for j in range(8)])  # 11-18: raw

        # ── Token-level positions from domain surface observations ──
        tokens_viz = []
        obs = ds_data.get("observations", [])
        if obs:
            # Collect per-token signal ranges for normalization
            tok_asm = [o[6] for o in obs]
            tok_sfd_e = [o[7] for o in obs]
            tok_repl = [o[4] for o in obs]

            def _norm(vals):
                mn, mx = min(vals), max(vals)
                rng = mx - mn if mx > mn else 1
                return mn, rng

            asm_mn, asm_rng = _norm(tok_asm)
            sfd_e_mn, sfd_e_rng = _norm(tok_sfd_e)
            repl_mn, repl_rng = _norm(tok_repl)

            for o in obs:
                # Token signals normalized to [0,1]
                n_asm = (o[6] - asm_mn) / asm_rng
                n_sfd_e = (o[7] - sfd_e_mn) / sfd_e_rng
                n_repl = (o[4] - repl_mn) / repl_rng

                # Ring from probe level
                t_level = o[12]  # near_level
                t_ri = _get_ring(t_level, n_levels)
                t_inner = ring_bands[t_ri]["inner"]
                t_outer = ring_bands[t_ri]["outer"]

                # Subject angle from nearest probe
                t_si = int(o[13])  # near_subj_idx
                t_angle = subj_angles[t_si] if t_si < len(subj_angles) else 0

                # Anchor at ring midpoint on subject centerline
                anchor_r = (t_inner + t_outer) / 2

                # Local coordinate frame
                rad_x, rad_y = np.cos(t_angle), np.sin(t_angle)
                tan_x, tan_y = np.cos(t_angle + np.pi / 2), np.sin(t_angle + np.pi / 2)

                # Signal-driven push from anchor
                max_push = (t_outer - t_inner) * push_strength
                e_push = (n_sfd_e - 0.5) * 2 * max_push   # SFD_e → radial
                a_push = (n_asm - 0.5) * 2 * max_push      # ASM → tangential
                rd_push = (n_repl - 0.5) * max_push * 0.3   # RD → secondary radial

                t_x = anchor_r * np.cos(t_angle) + rad_x * (e_push + rd_push) + tan_x * a_push
                t_y = anchor_r * np.sin(t_angle) + rad_y * (e_push + rd_push) + tan_y * a_push

                pi = o[9]  # prompt index
                cat = categories[pi] if pi < len(categories) else "?"
                tokens_viz.append([
                    o[0],                          # 0: token text
                    cat,                           # 1: category code
                    round(float(t_x), 4),          # 2: x
                    round(float(-t_y), 4),         # 3: y (flip)
                    t_si,                          # 4: subject index
                    t_ri,                          # 5: ring index
                    round(float(n_repl), 3),       # 6: RD normalized
                    round(float(n_sfd_e), 3),      # 7: SFD_e normalized (radial)
                    round(float(n_asm), 3),        # 8: ASM normalized (tangential)
                    pi,                            # 9: prompt index
                ])

        # Cell corners for visualization
        cells_viz = []
        for si in range(n_subj):
            for ri in range(len(ring_bands)):
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
            "push_strength": push_strength,

            # Geometry
            "subjects": subjects,
            "subj_short": subj_short,
            "subj_angles": [round(float(a), 4) for a in subj_angles],
            "rings": [
                {"label": level_names[i].title() if i < len(level_names) else f"Ring {i}",
                 "inner": ring_bands[i]["inner"],
                 "outer": ring_bands[i]["outer"]}
                for i in range(len(ring_bands))
            ],
            "repulsor_axes": {
                "prompt": {
                    "radial": {"signal": "Entropy", "color": "#ff6347"},
                    "tangential": {"signal": "KL", "color": "#4fc3f7"},
                },
                "token": {
                    "radial": {"signal": "SFD_e", "color": "#ff6347"},
                    "tangential": {"signal": "ASM", "color": "#4fc3f7"},
                },
            },
            "center_signal": {"name": "RD", "color": "#a78bfa"},

            # Signal metadata
            "signal_keys": [s for _, s in SIGNAL_KEYS],
            "all_signal_keys": [s for _, s in ALL_SIGNAL_KEYS],

            # Per-prompt data
            "prompts": prompts_viz,
            "tokens": tokens_viz,
            "n_tokens": len(tokens_viz),
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
