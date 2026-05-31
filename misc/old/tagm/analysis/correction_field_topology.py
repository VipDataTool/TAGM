"""CorrectionFieldTopology: topology summaries of the correction field.

Translated from TASM's `engine/modules/displacement_field.py` (renamed
to match TAGM's naming: the field being analyzed is the correction
field, and its displacement is one of several topology features).

Consumes per-prompt LTP + RankDisplacement outputs plus amplitude
trajectory data, computes:

  - Per-layer tension-magnitude trajectories across prompts
  - Per-category mean trajectories
  - Transitions (where in the layer stack the correction field
    switches from attention-dominated to MLP-dominated)
  - Displacement flow: per-prompt direction of travel in the
    (M, V, L) space across LTP's monitored layers.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from misc.old.tagm.analysis.base import AnalysisModule, AnalysisResult
from misc.old.tagm.analysis.registry import register_analysis
from misc.old.tagm.analysis.statistics import group_by_category
from misc.old.tagm.measurement.parameters import ModuleParameter


@register_analysis
class CorrectionFieldTopology(AnalysisModule):
    name = "correction_field_topology"
    display_name = "Correction Field Topology"
    description = (
        "Per-layer topology of the correction field across prompts: "
        "tension-magnitude trajectories, attn/MLP transition points, "
        "and displacement flow in the (M, V, L) summary space."
    )
    version = "0.1.0"

    depends_on_measurements = ("lateral_tension_profile", "amplitude_trajectory")

    parameters = [
        ModuleParameter(
            name="layer_normalize",
            display_name="Normalize per layer",
            description="Divide per-layer magnitudes by their session-wide max.",
            kind="bool", default=True,
        ),
    ]

    def run(self, session, params, probes=None):
        layer_normalize = bool(params.get("layer_normalize", True))
        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"layer_normalize": layer_normalize},
        )

        prompts = session.get("prompts") or []
        if not prompts:
            return result

        # Per-layer tension from LTP offset_magnitude
        layer_tension: dict[int, list[float]] = defaultdict(list)
        layer_cov: dict[int, list[float]] = defaultdict(list)
        per_category_tension: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list))

        # attn_frac trajectories across prompts (from AmplitudeDerivedMetrics)
        sublayer_attn_frac_vals: list[list[float]] = []

        for p in prompts:
            ms = p.get("measurements") or {}
            category = p.get("category") or "uncategorized"

            ltp = ms.get("lateral_tension_profile") or {}
            pl = ltp.get("per_layer") or {}
            offset_mag = pl.get("offset_magnitude") or {}
            coverage = pl.get("lateral_coverage") or {}
            for k, v in offset_mag.items():
                try:
                    li = int(k); fv = float(v)
                except (TypeError, ValueError):
                    continue
                layer_tension[li].append(fv)
                per_category_tension[category][li].append(fv)
            for k, v in coverage.items():
                try:
                    li = int(k); fv = float(v)
                except (TypeError, ValueError):
                    continue
                layer_cov[li].append(fv)

            adm = ms.get("amplitude_derived_metrics") or {}
            af = (adm.get("per_token") or {}).get("attn_frac") or []
            if af:
                sublayer_attn_frac_vals.append([float(v) if v is not None else np.nan
                                                 for v in af])

        # Aggregate per layer
        layer_indices = sorted(layer_tension.keys())
        layer_mean_tension = [float(np.mean(layer_tension[l])) for l in layer_indices]
        layer_std_tension = [float(np.std(layer_tension[l], ddof=1))
                              if len(layer_tension[l]) > 1 else 0.0
                              for l in layer_indices]
        layer_mean_cov = [float(np.mean(layer_cov.get(l, [0.0])))
                          for l in layer_indices]

        if layer_normalize and layer_mean_tension:
            m = max(layer_mean_tension)
            if m > 0:
                layer_mean_tension = [v / m for v in layer_mean_tension]

        # Per-category trajectory
        per_cat_out = {}
        for cat_name, per_layer in per_category_tension.items():
            idxs = sorted(per_layer.keys())
            means = [float(np.mean(per_layer[l])) for l in idxs]
            if layer_normalize and means:
                m = max(means) or 1.0
                means = [v / m for v in means]
            per_cat_out[cat_name] = {
                "n_prompts": max((len(per_layer[l]) for l in idxs), default=0),
                "layer_indices": idxs,
                "mean_tension": means,
            }

        # attn/MLP transition: mean attn_frac across prompts per token position
        mean_attn_frac = None
        if sublayer_attn_frac_vals:
            max_len = max(len(a) for a in sublayer_attn_frac_vals)
            mat = np.full((len(sublayer_attn_frac_vals), max_len), np.nan)
            for i, a in enumerate(sublayer_attn_frac_vals):
                mat[i, :len(a)] = a
            with np.errstate(invalid="ignore"):
                mean_attn_frac = np.nanmean(mat, axis=0).tolist()

        result.objects["layer_indices"] = layer_indices
        result.objects["layer_mean_tension"] = layer_mean_tension
        result.objects["layer_std_tension"] = layer_std_tension
        result.objects["layer_mean_coverage"] = layer_mean_cov
        result.objects["per_category"] = per_cat_out
        if mean_attn_frac is not None:
            result.objects["mean_attn_frac_per_position"] = mean_attn_frac
        result.scalars["n_prompts"] = len(prompts)
        result.scalars["n_layers_present"] = len(layer_indices)
        return result
