"""Comparative Analysis: cross-prompt aggregate statistics and separability.

Aggregates per-prompt measurement scalars across categories, computes
bootstrap CIs and pairwise effect sizes. This module doesn't produce
new data — it organizes what the measurement layer already computed
into the aggregate view the UI's Comparative Analysis renderer expects.

Output shape matches TASM's aggregate_batch output so the existing
UI renderer works unchanged.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.analysis.statistics import (
    bootstrap_ci, bootstrap_effect_size, extract_scalar,
    group_by_category, optimal_threshold, _clean,
)
from tagm.measurement.parameters import ModuleParameter


# Maps TASM stat_key -> (measurement_name, scalar_field) in TAGM schema
_METRIC_REGISTRY = [
    ("stress_score",               "stress_score",               "stress_mean"),
    ("entropy",                    "last_position_attribution",   "entropy"),
    ("top2_share",                 "last_position_attribution",   "top2_share"),
    ("middle_share",               "last_position_attribution",   "middle_share"),
    ("interior_cv",                "last_position_attribution",   "interior_cv"),
    ("net_correction",             "last_position_attribution",   "net_correction_to_last"),
    ("ltp_mean_M",                 "lateral_tension_profile",     "mean_M"),
    ("ltp_mean_V",                 "lateral_tension_profile",     "mean_V"),
    ("ltp_n_dir",                  "lateral_tension_profile",     "n_directional"),
    ("sfd_density_mean",           "spectral_field_density",      "density_mean"),
    ("rank_displacement_tau",      "rank_displacement",           "mean_tau"),
    ("rank_displacement_overlap",  "rank_displacement",           "mean_overlap"),
    ("rd_replacement",             "rank_displacement",           "mean_replacement"),
    ("rd_disp_per_tok",            "rank_displacement",           "mean_disp_per_token"),
]

_BENIGN_CATS = {"benign", "baseline", "mild", "user_baseline"}
_HARMFUL_CATS = {"harmful", "adversarial", "jailbreak"}
_TARGET_CATS = {"harmful", "adversarial", "jailbreak", "dual-use"}


@register_analysis
class ComparativeAnalysis(AnalysisModule):
    name = "comparative_analysis"
    display_name = "Comparative Analysis"
    description = (
        "Cross-prompt aggregate statistics and category separability. "
        "Computes bootstrapped metric estimates, effect sizes, and "
        "optimal classification thresholds across the session."
    )
    version = "1.0.0"
    min_results = 2

    parameters = [
        ModuleParameter(
            name="force_recompute",
            display_name="Force Recompute",
            description="Recompute aggregate statistics even if cached results exist.",
            kind="bool",
            default=False,
        ),
    ]

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []
        n = len(prompts)

        cats = group_by_category(session)
        cat_names = sorted(cats.keys())

        # ── Per-category summaries ─────────────────────────────────
        cat_summaries = {}
        for cat_name in cat_names:
            group = cats[cat_name]
            summary = {"n": len(group), "metrics": {}}

            # Mean sequence length
            seq_lens = [p.get("seq_len", len(p.get("tokens") or []))
                        for p in group]
            summary["mean_seq_len"] = float(np.mean(seq_lens)) if seq_lens else 0

            # Negative token rate
            n_neg = sum(1 for p in group
                        if _get_scalar(p, "last_position_attribution",
                                       "has_negative_tokens"))
            summary["negative_token_rate"] = n_neg / len(group) if group else 0

            # Bootstrap CI for each metric
            for stat_key, meas_name, field_name in _METRIC_REGISTRY:
                vals = _clean([extract_scalar(p, meas_name, field_name)
                               for p in group])
                if vals:
                    summary["metrics"][stat_key] = bootstrap_ci(vals)

            cat_summaries[cat_name] = summary

        # ── Separability (benign-pool vs harmful-pool) ─────────────
        benign_prompts = [p for cat in _BENIGN_CATS
                          for p in cats.get(cat, [])]
        harmful_prompts = [p for cat in _HARMFUL_CATS
                           for p in cats.get(cat, [])]

        separability = {}
        if benign_prompts and harmful_prompts:
            for stat_key, meas_name, field_name in _METRIC_REGISTRY:
                b_vals = _clean([extract_scalar(p, meas_name, field_name)
                                 for p in benign_prompts])
                h_vals = _clean([extract_scalar(p, meas_name, field_name)
                                 for p in harmful_prompts])
                if b_vals and h_vals:
                    es = bootstrap_effect_size(b_vals, h_vals)
                    thr = optimal_threshold(b_vals, h_vals)
                    separability[stat_key] = {
                        "effect_size": {
                            "estimate": es["estimate"],
                            "ci": [es["ci_low"], es["ci_high"]],
                        },
                        "threshold": thr,
                        "benign_mean": float(np.mean(b_vals)),
                        "harmful_mean": float(np.mean(h_vals)),
                    }

        # ── Pairwise separability (benign vs each target) ──────────
        separability_pairwise = {}
        for target_cat in _TARGET_CATS:
            target_prompts = cats.get(target_cat, [])
            if not benign_prompts or not target_prompts:
                continue
            pair_key = f"benign_vs_{target_cat}"
            separability_pairwise[pair_key] = {}
            for stat_key, meas_name, field_name in _METRIC_REGISTRY:
                b_vals = _clean([extract_scalar(p, meas_name, field_name)
                                 for p in benign_prompts])
                t_vals = _clean([extract_scalar(p, meas_name, field_name)
                                 for p in target_prompts])
                if b_vals and t_vals:
                    es = bootstrap_effect_size(b_vals, t_vals)
                    thr = optimal_threshold(b_vals, t_vals)
                    separability_pairwise[pair_key][stat_key] = {
                        "effect_size": {
                            "estimate": es["estimate"],
                            "ci": [es["ci_low"], es["ci_high"]],
                        },
                        "threshold": thr,
                        "benign_mean": float(np.mean(b_vals)),
                        "target_mean": float(np.mean(t_vals)),
                    }

        # ── Available batch plots ──────────────────────────────────
        plot_keys = [
            "batch_summary", "separability",
            "key_scatters", "discriminative_sublayers", "proof1_summary",
            "exp_trajectory_overlay", "exp_difference_from_benign",
            "exp_metric_scatters", "exp_behavioral_comparison",
            "exp_ltp_category_comparison", "exp_ltp_m_vs_stress",
            "exp_ltp_profile_shapes", "exp_sfd_category_comparison",
            "exp_sfd_vs_asm", "exp_rank_displacement",
        ]

        # ── TASM-shaped output ─────────────────────────────────────
        return {
            "aggregate": {
                "n_total": n,
                "categories": cat_summaries,
                "separability": separability,
                "separability_pairwise": separability_pairwise,
            },
            "plot_keys": plot_keys,
            "n_prompts": n,
            "categories": cat_names,
            "category_details": cat_summaries,
        }


def _get_scalar(prompt: dict, meas_name: str, field: str):
    """Get a scalar from TAGM nested structure, return None if missing."""
    ms = (prompt.get("measurements") or {}).get(meas_name)
    if not ms:
        return None
    return (ms.get("scalars") or {}).get(field)
