"""CorrectionBackscatter: aggregate per-prompt backscatter projections.

Averages the (n_probes, n_sublayers) backscatter magnitude matrices
across prompts (or per-category), producing session-level views of
which (layer, role) sublayers the correction field uses for which
semantic probe classes.

Translated from TASM's `engine/modules/correction_backscatter.py` core
algorithm (the TASM file was 632 lines, most of which built plots and
exports that now live in the service/frontend layer).
"""
from __future__ import annotations

import numpy as np

from misc.old.tagm.analysis.base import AnalysisModule, AnalysisResult
from misc.old.tagm.analysis.registry import register_analysis
from misc.old.tagm.analysis.statistics import group_by_category
from misc.old.tagm.measurement.parameters import ModuleParameter


@register_analysis
class CorrectionBackscatter(AnalysisModule):
    name = "correction_backscatter"
    display_name = "Correction Backscatter"
    description = (
        "Session-level aggregation of per-prompt backscatter projections. "
        "Produces mean magnitude matrices per category and overall, plus "
        "per-sublayer discriminative rankings."
    )
    version = "0.1.0"

    depends_on_measurements = ("backscatter_projection",)

    parameters = [
        ModuleParameter(
            name="per_category",
            display_name="Aggregate per category",
            description="Also produce per-category mean matrices.",
            kind="bool", default=True,
        ),
        ModuleParameter(
            name="top_discriminative",
            display_name="Top-N discriminative sublayers",
            description="Number of highest-between-category-variance sublayers to flag.",
            kind="int", default=15, min_value=3, max_value=200,
        ),
    ]

    def run(self, session, params, probes=None):
        per_category = bool(params.get("per_category", True))
        top_n = int(params.get("top_discriminative", 15))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"per_category": per_category,
                        "top_discriminative": top_n},
        )

        prompts = session.get("prompts") or []
        if not prompts:
            return result

        # Gather all backscatter matrices + labels (assume consistent shape
        # across the session; warn and skip if not)
        matrices: list[np.ndarray] = []
        categories: list[str] = []
        probe_labels: list[str] = []
        sublayer_labels: list[str] = []

        for p in prompts:
            bs = (p.get("measurements") or {}).get("backscatter_projection") or {}
            objs = bs.get("objects") or {}
            M = objs.get("magnitude_matrix") or []
            pl = objs.get("probe_labels") or []
            sl = objs.get("sublayer_labels") or []
            if not M or not pl or not sl:
                continue
            arr = np.array(M, dtype=float)
            if arr.shape[0] != len(pl) or arr.shape[1] != len(sl):
                continue
            if not probe_labels:
                probe_labels = pl
                sublayer_labels = sl
            elif probe_labels != pl or sublayer_labels != sl:
                # Skip this prompt; shape/labels mismatch (probably a different
                # probe set used mid-session)
                continue
            matrices.append(arr)
            categories.append(p.get("category") or "uncategorized")

        if not matrices:
            result.warnings.append("No compatible backscatter matrices in session")
            return result

        stack = np.stack(matrices, axis=0)  # (n_prompts, n_probes, n_sublayers)

        overall_mean = stack.mean(axis=0)      # (n_probes, n_sublayers)
        overall_std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 \
            else np.zeros_like(overall_mean)

        result.objects["probe_labels"] = probe_labels
        result.objects["sublayer_labels"] = sublayer_labels
        result.objects["overall_mean"] = overall_mean.tolist()
        result.objects["overall_std"] = overall_std.tolist()
        result.scalars["n_prompts"] = int(stack.shape[0])
        result.scalars["n_probes"] = int(stack.shape[1])
        result.scalars["n_sublayers"] = int(stack.shape[2])

        if per_category:
            by_cat: dict[str, list[int]] = {}
            for i, c in enumerate(categories):
                by_cat.setdefault(c, []).append(i)
            per_cat_out: dict[str, dict] = {}
            for cat_name, idxs in by_cat.items():
                sub = stack[idxs]
                per_cat_out[cat_name] = {
                    "n_prompts": len(idxs),
                    "mean": sub.mean(axis=0).tolist(),
                    "std": (sub.std(axis=0, ddof=1).tolist()
                             if len(idxs) > 1
                             else np.zeros_like(sub[0]).tolist()),
                }
            result.objects["per_category"] = per_cat_out

            # Per-sublayer between-category variance (max - min of cat means)
            if len(by_cat) >= 2:
                cat_mean_stack = np.stack(
                    [np.array(per_cat_out[c]["mean"]) for c in by_cat], axis=0)
                # Variance across categories per (probe, sublayer) cell;
                # summarize per sublayer by summing across probes
                between_var = cat_mean_stack.var(axis=0, ddof=1).sum(axis=0)
                order = np.argsort(-between_var)
                result.objects["top_discriminative_sublayers"] = [
                    {"sublayer": sublayer_labels[int(i)],
                     "between_category_variance_sum": float(between_var[int(i)])}
                    for i in order[:top_n]
                ]

        return result
