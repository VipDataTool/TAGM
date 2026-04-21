"""MIReadiness: diagnostics for mechanistic-interpretability readiness of a session.

Checks that the captured measurements provide what MI-style analyses need:
  - Per-layer coverage of signal metrics
  - Proof-1 exactness rates
  - Signal-to-noise ratios across prompts
  - Class-balance of categories for separability work

Translated from TASM's `engine/modules/mechanistic_interpretability.py`;
the 774-line original combined readiness diagnostics with per-token
instrumentation. Here the readiness portion lives in this module and the
per-token instrumentation lives in MIInstrumentation.
"""
from __future__ import annotations

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.analysis.statistics import extract_scalar, group_by_category
from tagm.measurement.parameters import ModuleParameter


@register_analysis
class MIReadiness(AnalysisModule):
    name = "mi_readiness"
    display_name = "MI Readiness"
    description = (
        "Diagnostics assessing whether a session has enough measurement "
        "coverage, category balance, and proof-1 exactness to support "
        "downstream mechanistic-interpretability analyses."
    )
    version = "0.1.0"

    depends_on_measurements = ("last_position_attribution", "stress_score")

    parameters = [
        ModuleParameter(
            name="min_prompts_per_category",
            display_name="Minimum prompts per category",
            description="Below this, per-category statistics are flagged as under-powered.",
            kind="int", default=5, min_value=1,
        ),
    ]

    def run(self, session, params, probes=None):
        min_n = int(params.get("min_prompts_per_category", 5))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"min_prompts_per_category": min_n},
        )

        prompts = session.get("prompts") or []
        if not prompts:
            result.warnings.append("No prompts in session")
            return result

        cats = group_by_category(session)

        # Category balance
        cat_balance = {c: len(ps) for c, ps in cats.items()}
        underpowered = [c for c, n in cat_balance.items() if n < min_n]

        # Proof-1 exactness rate
        n_proof = 0
        n_exact = 0
        for p in prompts:
            lpa = (p.get("measurements") or {}).get("last_position_attribution") or {}
            checks = (lpa.get("objects") or {}).get("proof1_checks") or []
            for c in checks:
                n_proof += 1
                if c.get("exact"):
                    n_exact += 1

        # Coverage: how many prompts have each expected measurement
        expected = [
            "last_position_attribution", "stress_score", "amplitude_trajectory",
            "lateral_tension_profile", "spectral_field_density", "rank_displacement",
        ]
        coverage = {}
        for m in expected:
            present = sum(1 for p in prompts
                          if m in (p.get("measurements") or {}))
            coverage[m] = {"present": present, "total": len(prompts),
                           "fraction": present / len(prompts)}

        # Signal-to-noise: std(stress_score) / mean(stress_score) across prompts
        stress_vals = []
        for p in prompts:
            v = extract_scalar(p, "stress_score", "stress_mean")
            if v is not None:
                stress_vals.append(v)

        snr = None
        if stress_vals:
            import numpy as np
            m = float(np.mean(stress_vals))
            s = float(np.std(stress_vals))
            snr = (s / m) if m > 0 else None

        result.scalars["n_prompts"] = len(prompts)
        result.scalars["n_categories"] = len(cats)
        result.scalars["n_underpowered_categories"] = len(underpowered)
        result.scalars["proof1_exactness_rate"] = (
            n_exact / n_proof if n_proof > 0 else float("nan"))
        result.scalars["stress_cv"] = snr if snr is not None else float("nan")

        result.objects["category_balance"] = cat_balance
        result.objects["underpowered_categories"] = list(underpowered)
        result.objects["measurement_coverage"] = coverage

        if underpowered:
            result.warnings.append(
                f"Under-powered categories (< {min_n} prompts): {underpowered}")
        if n_proof > 0 and n_exact / n_proof < 0.8:
            result.warnings.append(
                f"Proof-1 exactness rate is low ({n_exact}/{n_proof}); "
                "decompositions may be approximate.")

        return result
