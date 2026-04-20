"""AmplitudeTrajectory: per-sublayer correction-field amplitude across layers.

For each captured layer and each sublayer type (attn or mlp), computes the
mean correction-norm across tokens. Produces the "trajectory" of RLHF's
active regions through the model and a (n_sublayers, seq_len) per-token
heatmap used by AmplitudeDerivedMetrics.

Translated from TASM's `Analyzer._extract_amplitude_trajectory`.

Capture expectation:
  Needs pre_attn_norm and post_attn_norm hidden states at one or more
  layers. The user's CaptureConfig decides which layers; this measurement
  operates on layers captured at BOTH hook points.
"""
from __future__ import annotations

import numpy as np
import torch

from tagm.measurement.base import MeasurementModule
from tagm.measurement.parameters import ModuleParameter
from tagm.measurement.registry import register_measurement
from tagm.measurement.requirements import CaptureExpectation
from tagm.measurement.result import FieldSpec, MeasurementResult
from tagm.measurement.scope import describe_scope_resolution


@register_measurement
class AmplitudeTrajectory(MeasurementModule):
    name = "amplitude_trajectory"
    display_name = "Amplitude Trajectory"
    description = (
        "Per-sublayer correction-field amplitude across captured layers, "
        "with per-token breakdowns. Foundation for AmplitudeDerivedMetrics."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="layers",
            display_name="Scope layers",
            description=(
                "Optional subset of captured layers to include in the "
                "trajectory. Empty (default) means all layers captured at "
                "both pre_attn_norm and post_attn_norm. Scope parameter; "
                "not a capture parameter."
            ),
            kind="layer_list", default=[],
        ),
    ]

    def capture_expectation(self, params):
        return CaptureExpectation(
            hook_points_required=("pre_attn_norm", "post_attn_norm"),
            capture_types_required=frozenset({"hidden"}),
            min_layers_captured=1,
        )

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        seq_len = run_result.seq_len
        store = run_result.activations
        scope = list(params.get("layers") or [])

        pre_layers = set(store.layers_for("pre_attn_norm", "hidden"))
        post_layers = set(store.layers_for("post_attn_norm", "hidden"))
        jointly_captured = sorted(pre_layers & post_layers)

        if scope:
            scope_set = set(int(x) for x in scope)
            layers = [l for l in jointly_captured if l in scope_set]
        else:
            layers = jointly_captured

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={
                "layers_requested": list(scope),
                "layers_used": list(layers),
                "scope_resolution": describe_scope_resolution(
                    scope, layers, "pre_attn_norm ∩ post_attn_norm"),
            },
        )

        raw_traj: list[float] = []
        norm_traj: list[float] = []
        heatmap_rows: list[np.ndarray] = []
        sublayer_labels: list[str] = []

        for layer_idx in layers:
            for sublayer_type in ("attn", "mlp"):
                hook_point = ("pre_attn_norm" if sublayer_type == "attn"
                              else "post_attn_norm")
                sublayer_labels.append(f"L{layer_idx}_{sublayer_type}")

                h = store.get(layer_idx, hook_point, "hidden")
                roles = (("q", "k", "v") if sublayer_type == "attn"
                         else ("gate", "up"))

                raw_sum = 0.0
                norm_sum = 0.0
                per_tok = torch.zeros(seq_len)

                for role in roles:
                    dw = delta_store.get_or_none(layer_idx, role)
                    if dw is None:
                        continue
                    fnorm = delta_store.frob_norm(layer_idx, role)
                    if dw.shape[1] != h.shape[2] or fnorm <= 0:
                        continue
                    h_slice = h[0, :seq_len].float()
                    projected = torch.matmul(h_slice, dw.float().T)
                    pn = projected.norm(dim=-1)
                    raw_sum += pn.mean().item()
                    norm_sum += (pn / fnorm).mean().item()
                    per_tok += pn / fnorm

                raw_traj.append(raw_sum)
                norm_traj.append(norm_sum)
                heatmap_rows.append(per_tok.float().numpy().astype(float))

        heatmap = (np.array(heatmap_rows) if heatmap_rows
                   else np.zeros((0, seq_len)))

        result.scalars["trajectory_mean_raw"] = (
            float(np.mean(raw_traj)) if raw_traj else 0.0)
        result.scalars["trajectory_mean_normalized"] = (
            float(np.mean(norm_traj)) if norm_traj else 0.0)
        result.scalars["n_layers_used"] = int(len(layers))
        result.objects["amplitude_raw"] = list(raw_traj)
        result.objects["amplitude_normalized"] = list(norm_traj)
        result.objects["heatmap"] = heatmap.tolist()
        result.objects["heatmap_shape"] = list(heatmap.shape)
        result.objects["sublayer_labels"] = sublayer_labels

        self._annotate(result)
        return result

    def _annotate(self, result: MeasurementResult) -> None:
        result.field_specs["amplitude_raw"] = FieldSpec(
            name="amplitude_raw", kind="object",
            description="Per-sublayer raw amplitude (list, length 2*n_layers_used).",
            length_invariant=True,
        )
        result.field_specs["amplitude_normalized"] = FieldSpec(
            name="amplitude_normalized", kind="object",
            description="Per-sublayer Frobenius-normalized amplitude.",
            length_invariant=True,
        )
        result.field_specs["heatmap"] = FieldSpec(
            name="heatmap", kind="object",
            description="(n_sublayers, seq_len) per-token normalized correction "
                        "amplitudes; rows alternate attn/mlp per layer.",
            length_invariant=False,
        )
        result.field_specs["sublayer_labels"] = FieldSpec(
            name="sublayer_labels", kind="object",
            description="Human-readable labels for each heatmap row.",
            length_invariant=True,
        )
        result.field_specs["n_layers_used"] = FieldSpec(
            name="n_layers_used", kind="scalar",
            description="Count of layers actually aggregated.",
            length_invariant=True,
        )
