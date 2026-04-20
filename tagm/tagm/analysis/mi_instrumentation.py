"""MIInstrumentation: per-token, per-layer instrumentation aggregated across prompts.

Builds the tables MI-style analysis wants:
  - (category, prompt_index, layer, token_position) → stress, signed_attr, density
  - Per-category mean per-layer profiles
  - Per-position variance across prompts within a category

Translated from TASM's `engine/modules/mi_instrumentation.py`. TASM had to
work around its per-layer fields being stripped from the export; TAGM's
schema guarantees that per_layer_per_token fields survive serialization,
so this analysis has cleaner inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.analysis.statistics import group_by_category
from tagm.measurement.parameters import ModuleParameter


@register_analysis
class MIInstrumentation(AnalysisModule):
    name = "mi_instrumentation"
    display_name = "MI Instrumentation"
    description = (
        "Per-token, per-layer aggregations across the session. Produces "
        "per-category layer profiles and per-position variance for tokens "
        "that are informative for MI work."
    )
    version = "0.1.0"

    depends_on_measurements = ("last_position_attribution",)

    parameters = [
        ModuleParameter(
            name="max_prompt_length",
            display_name="Max prompt length (for fixed-width aggregation)",
            description="Prompts longer than this are truncated for aggregation.",
            kind="int", default=64, min_value=4, max_value=512,
        ),
    ]

    def run(self, session, params, probes=None):
        max_len = int(params.get("max_prompt_length", 64))
        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"max_prompt_length": max_len},
        )

        prompts = session.get("prompts") or []
        if not prompts:
            return result

        cats = group_by_category(session)
        per_category: dict[str, Any] = {}

        for cat_name, cat_prompts in cats.items():
            # Per-position signed attribution (padded to max_len)
            attr_stack: list[np.ndarray] = []
            stress_stack: list[np.ndarray] = []

            for p in cat_prompts:
                ms = p.get("measurements") or {}

                lpa = ms.get("last_position_attribution") or {}
                arr = (lpa.get("per_token") or {}).get("signed_attribution_to_last")
                if arr:
                    pa = _pad_or_truncate(arr, max_len)
                    attr_stack.append(pa)

                ss = ms.get("stress_score") or {}
                sarr = (ss.get("per_token") or {}).get("stress")
                if sarr:
                    pa = _pad_or_truncate(sarr, max_len)
                    stress_stack.append(pa)

            def _agg(stack):
                if not stack:
                    return {"mean": [], "std": [], "n": 0}
                arr = np.array(stack, dtype=float)
                with np.errstate(invalid="ignore"):
                    return {
                        "mean": np.nanmean(arr, axis=0).tolist(),
                        "std": np.nanstd(arr, axis=0).tolist(),
                        "n": int(arr.shape[0]),
                    }

            per_category[cat_name] = {
                "n_prompts": len(cat_prompts),
                "signed_attribution": _agg(attr_stack),
                "stress": _agg(stress_stack),
            }

        result.objects["per_category_profiles"] = per_category
        result.scalars["n_prompts"] = len(prompts)
        result.scalars["n_categories"] = len(cats)
        return result


def _pad_or_truncate(arr, max_len: int) -> np.ndarray:
    """Normalize a variable-length per-token array to fixed width with NaN padding."""
    a = np.array([x if x is not None else np.nan for x in arr], dtype=float)
    if len(a) >= max_len:
        return a[:max_len]
    padded = np.full(max_len, np.nan, dtype=float)
    padded[:len(a)] = a
    return padded
