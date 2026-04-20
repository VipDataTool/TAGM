"""ComparativeAnalysis: cross-prompt aggregate statistics and category separability.

Ported from TASM's `engine/modules/comparative_analysis.py` plus the core of
`engine/statistics.py::aggregate_batch`. Emits the JSON shape the frontend's
`renderComparativeResults` function reads directly:

    {
      "n_prompts": int,
      "categories": [str, ...],
      "plot_keys": [str, ...],
      "aggregate": {
        "categories": {
          cat: {
            "n": int,
            "metrics": {
              "stress_score":   {"estimate": float, "ci": [lo, hi]},
              "entropy":        {...},
              "middle_share":   {...},
              "net_correction": {...},
              "interior_cv":    {...},
              "sfd_density_mean": {...},
              "rank_displacement_tau": {...},
            },
          },
        },
        "separability": {
          metric_key: {
            "effect_size": {"estimate": float, "ci": [lo, hi]},
            "threshold": {"threshold": float, "accuracy": float},
          },
        },
      },
    }

Uses TAGM's own bootstrap / threshold helpers in tagm.analysis.statistics.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.analysis.statistics import (
    bootstrap_ci, bootstrap_effect_size, extract_scalar,
    group_by_category, optimal_threshold,
)
from tagm.measurement.parameters import ModuleParameter


# Metric keys rendered by the UI in the Category Summary table
# (column ordering matches the JS renderer). Each entry is
# (column_key, measurement_name, scalar_field).
_UI_METRICS = [
    ("stress_score",           "stress_score",               "stress_mean"),
    ("entropy",                "last_position_attribution",  "entropy"),
    ("middle_share",           "last_position_attribution",  "middle_share"),
    ("net_correction",         "last_position_attribution",  "net_correction_to_last"),
    ("interior_cv",            "last_position_attribution",  "interior_cv"),
    ("sfd_density_mean",       "spectral_field_density",     "density_mean"),
    ("rank_displacement_tau",  "rank_displacement",          "mean_tau"),
    # Extra columns carried along for the Separability table. The UI
    # doesn't render them as their own columns but they populate
    # separability entries below.
    ("top2_share",             "last_position_attribution",  "top2_share"),
    ("ltp_mean_M",             "lateral_tension_profile",    "mean_M"),
    ("ltp_max_prc",            "lateral_tension_profile",    "max_prc"),
]


# Plot keys the batch-viz panel can show. TASM's Comparative analysis
# returned this list unconditionally — the VIZ_REGISTRY in the
# frontend filters by `scope==='batch'` and by availability.
_DEFAULT_PLOT_KEYS = [
    "batch_summary", "separability",
    "key_scatters", "discriminative_sublayers", "proof1_summary",
    "exp_trajectory_overlay", "exp_difference_from_benign",
    "exp_metric_scatters", "exp_behavioral_comparison",
    "exp_ltp_category_comparison", "exp_ltp_m_vs_stress",
    "exp_ltp_profile_shapes", "exp_sfd_category_comparison",
    "exp_sfd_vs_asm", "exp_rank_displacement",
]


@register_analysis
class ComparativeAnalysis(AnalysisModule):
    name = "comparative_analysis"
    display_name = "Comparative Analysis"
    description = (
        "Per-category aggregate statistics with bootstrap CIs and pairwise "
        "effect sizes. Produces the session-level summary the UI Category "
        "Summary table and Separability panel read."
    )
    version = "1.0.0"

    parameters = [
        ModuleParameter(
            name="n_bootstrap",
            display_name="Bootstrap resamples",
            description="Number of bootstrap resamples for CIs.",
            kind="int", default=5000, min_value=100, max_value=50000,
            advanced=True,
        ),
        ModuleParameter(
            name="ci_level",
            display_name="CI level",
            description="Confidence level for intervals (0.95 = 95%).",
            kind="float", default=0.95, min_value=0.5, max_value=0.999,
            advanced=True,
        ),
        ModuleParameter(
            name="threshold_steps",
            display_name="Threshold scan steps",
            description="Candidate thresholds evaluated per metric.",
            kind="int", default=500, min_value=10, max_value=10000,
            advanced=True,
        ),
    ]

    def run(self, session, params, probes=None, context=None):
        n_boot = int(params.get("n_bootstrap", 5000))
        ci = float(params.get("ci_level", 0.95))
        n_steps = int(params.get("threshold_steps", 500))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"n_bootstrap": n_boot, "ci_level": ci,
                        "threshold_steps": n_steps},
        )

        prompts = session.get("prompts") or []
        if not prompts:
            result.warnings.append("No prompts in session")
            result.objects.update(_empty_aggregate())
            return result

        # Group by category
        cats = group_by_category(session)
        cat_names = sorted(cats.keys())

        # ── Build per-category metrics in the shape the UI reads ──
        ui_categories: dict[str, dict] = {}
        # Also collect per-metric per-category value arrays for the
        # separability computation below.
        metric_values: dict[str, dict[str, list[float]]] = {
            key: {} for key, _, _ in _UI_METRICS
        }

        for cat_name in cat_names:
            cat_prompts = cats[cat_name]
            metrics_entry: dict[str, dict] = {}
            for col_key, meas_name, field_name in _UI_METRICS:
                vals: list[float] = []
                for p in cat_prompts:
                    v = extract_scalar(p, meas_name, field_name)
                    if v is not None:
                        vals.append(v)
                metric_values[col_key][cat_name] = vals
                if not vals:
                    continue
                estimate_ci = bootstrap_ci(vals, n_boot=n_boot, ci=ci)
                metrics_entry[col_key] = {
                    "estimate": estimate_ci["estimate"],
                    "ci": [estimate_ci["ci_low"], estimate_ci["ci_high"]],
                    "n": len(vals),
                }
            ui_categories[cat_name] = {
                "n": len(cat_prompts),
                "metrics": metrics_entry,
            }

        # ── Separability: Cohen's d between safe and risk partitions ──
        # TASM distinguished "safe" (benign/mild) vs "risk" (harmful/
        # jailbreak/etc) for the Cohen's d table. We follow the same
        # convention so the UI's AUROC-style colour coding lines up.
        safe_cats = {"benign", "baseline", "mild"}
        risk_cats = set(cat_names) - safe_cats

        separability: dict[str, dict] = {}
        for col_key, meas_name, field_name in _UI_METRICS:
            safe_vals: list[float] = []
            risk_vals: list[float] = []
            for cname, vals in metric_values[col_key].items():
                if cname in safe_cats:
                    safe_vals.extend(vals)
                else:
                    risk_vals.extend(vals)
            if len(safe_vals) < 2 or len(risk_vals) < 2:
                continue
            es = bootstrap_effect_size(safe_vals, risk_vals,
                                        n_boot=n_boot, ci=ci)
            thr = optimal_threshold(safe_vals, risk_vals, n_steps=n_steps)
            separability[col_key] = {
                "effect_size": {
                    "estimate": es["estimate"],
                    "ci": [es["ci_low"], es["ci_high"]],
                },
                "threshold": {
                    "threshold": thr["threshold"],
                    "accuracy": thr["accuracy"],
                    "direction": thr["direction"],
                },
                "n_safe": len(safe_vals),
                "n_risk": len(risk_vals),
            }

        # ── Pairwise per-category effects (kept from TAGM native) ──
        pairwise: dict[str, list[dict]] = {}
        for col_key in metric_values:
            entries = []
            for a, b in combinations(cat_names, 2):
                va, vb = metric_values[col_key][a], metric_values[col_key][b]
                if len(va) < 2 or len(vb) < 2:
                    continue
                es = bootstrap_effect_size(va, vb, n_boot=n_boot, ci=ci)
                entries.append({
                    "a": a, "b": b,
                    "cohens_d": es["estimate"],
                    "cohens_d_ci": [es["ci_low"], es["ci_high"]],
                })
            if entries:
                pairwise[col_key] = entries

        aggregate = {
            "categories": ui_categories,
            "separability": separability,
            "pairwise_effects": pairwise,
            "n_total": len(prompts),
        }

        # Emit TASM wire format. The flattener in app.py hoists these
        # into top-level keys on the /api/modules/.../results response.
        result.objects["aggregate"] = aggregate
        result.objects["categories"] = list(cat_names)
        result.objects["plot_keys"] = list(_DEFAULT_PLOT_KEYS)
        result.objects["n_prompts"] = len(prompts)
        result.scalars["n_prompts"] = len(prompts)
        result.scalars["n_categories"] = len(cat_names)
        result.scalars["n_metrics"] = len(_UI_METRICS)

        return result


def _empty_aggregate() -> dict:
    return {
        "aggregate": {
            "categories": {},
            "separability": {},
            "pairwise_effects": {},
            "n_total": 0,
        },
        "categories": [],
        "plot_keys": list(_DEFAULT_PLOT_KEYS),
        "n_prompts": 0,
    }
