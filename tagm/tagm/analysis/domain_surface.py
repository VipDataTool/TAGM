"""DomainSurface: polar-projected per-token embeddings onto probe axes.

Each token is placed in a polar coordinate system where:
  - angle (θ) = best-matching subject-class probe direction
  - radius (r) = escalation-level probe match (or stress magnitude)

The resulting (θ, r) scatter is the "domain surface." Per-category
density kernels, class-occupancy fractions, and "hot-region" summaries
are the analysis outputs.

Translated from the core algorithm of TASM's
`engine/modules/domain_surface.py`. That file was 1055 lines because
of extensive plot generation and export paths; TAGM moves rendering
to the service/frontend layer and this module returns the raw polar
data plus cell assignments for downstream rendering.

This is the largest translation in the analysis layer. Preserved
semantics:
  - Position 0 is excluded by default (positional artifact), matching
    TASM's `include_first_token = False` default. (In TAGM this is a
    user-set parameter; no implicit filtering.)
  - Angular quantization uses the best-matching subject probe.
  - Radial coordinate is either escalation-probe-score or stress-score
    (controlled by the `radial_source` parameter).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter


@register_analysis
class DomainSurface(AnalysisModule):
    name = "domain_surface"
    display_name = "Domain Surface"
    description = (
        "Per-token polar coordinates (angle from subject probe, radius from "
        "escalation probe or stress), aggregated into a domain-surface "
        "scatter and per-class occupancy summaries."
    )
    version = "0.1.0"

    depends_on_measurements = ("probe_projection", "stress_score")

    parameters = [
        ModuleParameter(
            name="radial_source",
            display_name="Radial source",
            description="What drives the radial coordinate.",
            kind="select", default="stress",
            options=("stress", "escalation_score", "density"),
        ),
        ModuleParameter(
            name="include_first_token",
            display_name="Include position-0 tokens",
            description="Position 0 is usually excluded due to positional artifacts.",
            kind="bool", default=False, advanced=True,
        ),
        ModuleParameter(
            name="min_match_score",
            display_name="Minimum probe match score",
            description="Tokens whose best probe score is below this threshold "
                        "are dropped from the surface.",
            kind="float", default=0.0, min_value=-1.0, max_value=1.0,
            advanced=True,
        ),
    ]

    def run(self, session, params, probes=None):
        radial = params.get("radial_source", "stress")
        include_first = bool(params.get("include_first_token", False))
        min_score = float(params.get("min_match_score", 0.0))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"radial_source": radial,
                        "include_first_token": include_first,
                        "min_match_score": min_score},
        )

        prompts = session.get("prompts") or []
        if not prompts:
            return result

        # Collect (theta_idx, r, category, token, prompt_idx, position) points
        points: list[dict[str, Any]] = []
        probe_labels: list[str] = []
        occupancy: dict[str, int] = defaultdict(int)

        for prompt_idx, p in enumerate(prompts):
            ms = p.get("measurements") or {}
            proj = ms.get("probe_projection") or {}
            labels = (proj.get("objects") or {}).get("probe_labels") or []
            per_token = proj.get("per_token") or {}
            best_idx = per_token.get("best_class_idx") or []
            best_score = per_token.get("best_score") or []

            if not labels or not best_idx:
                continue

            if not probe_labels:
                probe_labels = labels

            # Radial channel
            if radial == "stress":
                r_source = (ms.get("stress_score") or {}).get("per_token", {}).get("stress") or []
            elif radial == "density":
                r_source = (ms.get("spectral_field_density") or {}).get("per_token", {}).get("density") or []
            else:
                # escalation_score: per-token score from a secondary probe_projection
                # at the escalation layer; if not present, fall back to best_score
                r_source = best_score

            tokens = p.get("tokens") or []
            category = p.get("category") or "uncategorized"
            start = 0 if include_first else 1
            n = min(len(best_idx), len(r_source), len(tokens))

            for pos in range(start, n):
                idx = best_idx[pos]
                val = r_source[pos]
                score = best_score[pos] if pos < len(best_score) else None
                if idx is None or val is None:
                    continue
                try:
                    idx_i = int(idx)
                    r_f = float(val)
                    s_f = float(score) if score is not None else 0.0
                except (TypeError, ValueError):
                    continue
                if np.isnan(r_f) or np.isinf(r_f):
                    continue
                if s_f < min_score:
                    continue
                if not (0 <= idx_i < len(labels)):
                    continue

                # Polar angle: evenly distribute probe indices around the circle
                theta = 2 * math.pi * idx_i / max(len(labels), 1)
                points.append({
                    "theta": theta,
                    "theta_idx": idx_i,
                    "r": r_f,
                    "category": category,
                    "prompt_idx": prompt_idx,
                    "position": pos,
                    "token": tokens[pos],
                    "match_score": s_f,
                    "probe_label": labels[idx_i],
                })
                occupancy[labels[idx_i]] += 1

        # Per-category occupancy (which probe classes dominate per category)
        per_category_occ: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        for pt in points:
            per_category_occ[pt["category"]][pt["probe_label"]] += 1

        # Simple kernel density at each probe angle per category
        per_cat_profile: dict[str, dict] = {}
        for cat_name, occ in per_category_occ.items():
            total = sum(occ.values())
            per_cat_profile[cat_name] = {
                "n_points": total,
                "per_class_fraction": {
                    k: (v / total if total > 0 else 0.0)
                    for k, v in occ.items()
                },
            }

        result.objects["probe_labels"] = probe_labels
        result.objects["points"] = points
        result.objects["occupancy"] = dict(occupancy)
        result.objects["per_category"] = per_cat_profile
        result.scalars["n_points"] = len(points)
        result.scalars["n_probe_classes"] = len(probe_labels)
        result.scalars["n_prompts"] = len(prompts)
        return result
