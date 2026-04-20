"""ComparativeAnalysis: cross-prompt aggregate statistics and category separability.

For each declared metric, computes per-category means with bootstrap
confidence intervals, computes Cohen's d between category pairs, and
identifies the optimal classification threshold. Produces the
session-level summary used for publishable plots.

Translated from TASM's `engine/modules/comparative_analysis.py` plus the
core of `engine/statistics.py::aggregate_batch`.

Dependencies: any measurement whose scalars the user wants aggregated.
The default metrics below cover what TASM's comparative pipeline produced;
users can pass custom metrics in `params.metrics`.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.analysis.statistics import (
    bootstrap_ci, bootstrap_effect_size, cohens_d, extract_scalar,
    group_by_category, optimal_threshold,
)
from tagm.measurement.parameters import ModuleParameter


# Default metric list: (measurement_name, scalar_field, display_name)
_DEFAULT_METRICS = [
    ("stress_score", "stress_mean", "Stress Score"),
    ("last_position_attribution", "net_correction_to_last", "Net Correction"),
    ("last_position_attribution", "entropy", "Entropy"),
    ("last_position_attribution", "top2_share", "Top-2 Share"),
    ("last_position_attribution", "middle_share", "Middle Share"),
    ("last_position_attribution", "interior_cv", "Interior CV"),
    ("lateral_tension_profile", "mean_M", "LTP Mean M"),
    ("lateral_tension_profile", "mean_V", "LTP Mean V"),
    ("lateral_tension_profile", "mean_L", "LTP Mean L"),
    ("lateral_tension_profile", "max_prc", "LTP Max PRC"),
    ("spectral_field_density", "density_mean", "SFD Density Mean"),
    ("spectral_field_density", "density_max", "SFD Density Max"),
    ("rank_displacement", "mean_disp_per_token", "RD Mean Disp"),
    ("rank_displacement", "mean_replacement", "RD Mean Replacement"),
]


@register_analysis
class ComparativeAnalysis(AnalysisModule):
    name = "comparative_analysis"
    display_name = "Comparative Analysis"
    description = (
        "Per-category aggregate statistics with bootstrap CIs and pairwise "
        "effect sizes. Produces the session-level summary for comparison plots."
    )
    version = "0.1.0"

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

    def run(self, session, params, probes=None):
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
            return result

        cats = group_by_category(session)
        cat_names = sorted(cats.keys())

        per_category: dict = {}
        per_metric: dict = {}

        for measurement_name, field_name, display in _DEFAULT_METRICS:
            all_values = []
            cat_values: dict[str, list] = {}
            for cat_name in cat_names:
                vals = []
                for p in cats[cat_name]:
                    v = extract_scalar(p, measurement_name, field_name)
                    if v is not None:
                        vals.append(v)
                cat_values[cat_name] = vals
                all_values.extend(vals)

            if not all_values:
                continue

            # Per-category bootstrap CIs
            cat_ci = {
                cname: bootstrap_ci(vals, n_boot=n_boot, ci=ci)
                for cname, vals in cat_values.items()
            }

            # Pairwise effect sizes
            pairwise = []
            for a, b in combinations(cat_names, 2):
                if not cat_values[a] or not cat_values[b]:
                    continue
                es = bootstrap_effect_size(cat_values[a], cat_values[b],
                                            n_boot=n_boot, ci=ci)
                thr = optimal_threshold(cat_values[a], cat_values[b], n_steps=n_steps)
                pairwise.append({
                    "category_a": a, "category_b": b,
                    "cohens_d": es["estimate"],
                    "cohens_d_ci": [es["ci_low"], es["ci_high"]],
                    "threshold": thr["threshold"],
                    "accuracy": thr["accuracy"],
                    "direction": thr["direction"],
                })

            metric_key = f"{measurement_name}.{field_name}"
            per_metric[metric_key] = {
                "measurement": measurement_name,
                "field": field_name,
                "display_name": display,
                "per_category": cat_ci,
                "pairwise_effects": pairwise,
                "overall_n": len(all_values),
            }

        for cat_name in cat_names:
            per_category[cat_name] = {
                "n_prompts": len(cats[cat_name]),
            }

        result.objects["per_metric"] = per_metric
        result.objects["per_category"] = per_category
        result.objects["categories"] = list(cat_names)
        result.scalars["n_prompts"] = len(prompts)
        result.scalars["n_categories"] = len(cat_names)
        result.scalars["n_metrics"] = len(per_metric)

        return result
