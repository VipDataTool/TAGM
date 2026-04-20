"""StressScore measurement.

Per-layer, per-token sum of ||h @ dW_p^T|| / ||dW_p||_F across p ∈ {q, k, v},
averaged across layers. Measures how much each token engages the alignment
delta subspace on attention side, normalized by delta magnitude so layers
are comparable.

Translated from TASM's `Analyzer._extract_stress_score`.

Capture expectation:
  Needs pre_attn_norm hidden states at one or more layers. The user's
  CaptureConfig decides which layers; this measurement operates on
  whatever layers are captured at pre_attn_norm *and* have q/k/v deltas
  available, narrowed by the optional `layers` scope parameter.
"""
from __future__ import annotations

import numpy as np
import torch

from tagm.measurement.base import MeasurementModule
from tagm.measurement.parameters import ModuleParameter
from tagm.measurement.registry import register_measurement
from tagm.measurement.requirements import CaptureExpectation
from tagm.measurement.result import FieldSpec, MeasurementResult, padded_per_token
from tagm.measurement.scope import describe_scope_resolution, resolve_scope_layers


@register_measurement
class StressScore(MeasurementModule):
    name = "stress_score"
    display_name = "Stress Score"
    description = (
        "Per-token attention-side correction stress: normalized sum of "
        "||h @ dW_p^T|| / ||dW_p||_F over p ∈ {q, k, v}, averaged across "
        "the layers this measurement operates on. Intensity axis of the "
        "measurement triple."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="layers",
            display_name="Scope layers",
            description=(
                "Optional subset of captured layers to aggregate over. "
                "Empty (default) means 'all captured layers that also "
                "have q/k/v deltas available'. This is a scope parameter, "
                "not a capture parameter — the user's CaptureConfig "
                "chooses which layers are recorded."
            ),
            kind="layer_list", default=[],
        ),
    ]

    def capture_expectation(self, params):
        return CaptureExpectation(
            hook_points_required=("pre_attn_norm",),
            capture_types_required=frozenset({"hidden"}),
            min_layers_captured=1,
        )

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        seq_len = run_result.seq_len
        store = run_result.activations
        scope = params.get("layers") or []

        # Resolve which layers to aggregate over. Empty scope → all available.
        layers = resolve_scope_layers(
            activation_store=store,
            hook_point="pre_attn_norm",
            capture_type="hidden",
            scope=scope,
            required_delta_roles=("q", "k", "v"),
            delta_store=delta_store,
        )

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={
                "layers_requested": list(scope),
                "layers_used": list(layers),
                "scope_resolution": describe_scope_resolution(
                    scope, layers, "pre_attn_norm"),
            },
        )

        if not layers:
            result.per_token["stress"] = padded_per_token(seq_len)
            result.scalars["stress_mean"] = float("nan")
            self._annotate(result)
            return result

        per_token_total = torch.zeros(seq_len)
        n_layers_used = 0

        for layer_idx in layers:
            h = store.get(layer_idx, "pre_attn_norm", "hidden")
            layer_contribution = torch.zeros(seq_len)
            layer_had_any = False

            for role in ("q", "k", "v"):
                dw = delta_store.get_or_none(layer_idx, role)
                if dw is None:
                    continue
                fnorm = delta_store.frob_norm(layer_idx, role)
                if dw.shape[1] != h.shape[2] or fnorm <= 0:
                    continue
                projected = torch.matmul(h[0, :seq_len].float(), dw.float().T)
                layer_contribution += projected.norm(dim=-1) / fnorm
                layer_had_any = True

            if layer_had_any:
                per_token_total += layer_contribution
                n_layers_used += 1

        if n_layers_used > 0:
            per_token_total /= n_layers_used

        per_token_arr = per_token_total.numpy().astype(float)
        result.per_token["stress"] = per_token_arr
        result.scalars["stress_mean"] = (
            float(per_token_arr.mean()) if n_layers_used > 0 else float("nan")
        )
        result.scalars["n_layers_used"] = int(n_layers_used)

        self._annotate(result)
        return result

    def _annotate(self, result: MeasurementResult) -> None:
        result.field_specs["stress"] = FieldSpec(
            name="stress", kind="per_token",
            description=("Per-token attention-side correction stress, "
                          "layer-averaged."),
            length_invariant=False,
            semantic_note=("Index i is the stress at token i itself — this is "
                           "a genuine per-token measure, unlike "
                           "last_position_attribution."),
        )
        result.field_specs["stress_mean"] = FieldSpec(
            name="stress_mean", kind="scalar",
            description="Mean of per-token stress over the prompt.",
            length_invariant=True,
        )
        result.field_specs["n_layers_used"] = FieldSpec(
            name="n_layers_used", kind="scalar",
            description="Count of layers actually aggregated "
                        "(intersection of scope, capture, deltas).",
            length_invariant=True,
        )
