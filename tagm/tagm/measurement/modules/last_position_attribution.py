"""LastPositionAttribution (formerly TASM's "signed attribution").

Per the translation doc and TASM forensic audit finding #1: TASM's
`signed_attr[i]` is the contribution of token `i` to the *last position's*
correction, averaged over the specified layers — not a per-token correction.
TAGM names it explicitly and records the semantic meaning in field metadata
so downstream consumers can't mistake what the numbers mean.

Math is preserved verbatim from TASM's `Analyzer._extract_signed_attribution`.

Capture expectation:
  Needs pre_attn_norm hidden states AND attention_weights at attn_output,
  at the same layers. The user's CaptureConfig decides which layers;
  this measurement operates on layers where both are captured *and*
  the v delta is available, narrowed by the optional `layers` scope parameter.
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
class LastPositionAttribution(MeasurementModule):
    name = "last_position_attribution"
    display_name = "Last-Position Attribution"
    description = (
        "Per-token contribution to the last position's correction field at "
        "each operated-on layer, projected through dW_V and attention weights. "
        "Replaces TASM's 'signed attribution' with a name that matches what "
        "the values actually index."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="layers",
            display_name="Scope layers",
            description=(
                "Optional subset of captured layers to aggregate over. "
                "Empty means all captured layers that have both "
                "pre_attn_norm hidden states AND attn_output attention "
                "weights AND v deltas available. Scope parameter; "
                "not a capture parameter."
            ),
            kind="layer_list", default=[],
        ),
        ModuleParameter(
            name="boundary_fraction",
            display_name="Boundary fraction",
            description=(
                "Fraction of tokens at each end counted as 'boundary' for the "
                "interior/boundary distribution split (top2_share, middle_share)."
            ),
            kind="float",
            default=0.1, min_value=0.0, max_value=0.5,
            engine_config_key="boundary_fraction",
        ),
        ModuleParameter(
            name="proof1_threshold",
            display_name="Proof-1 exactness threshold",
            description=(
                "Maximum acceptable error in the Proof-1 decomposition check "
                "(sum of signed attribution should equal ||delta||)."
            ),
            kind="float",
            default=1e-4, min_value=0.0,
            advanced=True,
            engine_config_key="proof1_threshold",
        ),
    ]

    def capture_expectation(self, params):
        return CaptureExpectation(
            hook_points_required=("pre_attn_norm",),
            capture_types_required=frozenset({"hidden"}),
            needs_attention_weights=True,
            min_layers_captured=1,
        )

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        scope = list(params.get("layers") or [])
        boundary_fraction = float(params.get("boundary_fraction", 0.1))
        proof1_threshold = float(params.get("proof1_threshold", 1e-4))

        seq_len = run_result.seq_len
        store = run_result.activations

        # Resolve which layers to aggregate over. This measurement needs
        # BOTH pre_attn_norm hidden AND attn_output attention_weights at
        # the same layer. Use the intersection of what's captured at
        # both hook points.
        pre_layers = set(store.layers_for("pre_attn_norm", "hidden"))
        attn_layers = set(store.layers_for("attn_output", "attention_weights"))
        jointly_captured = pre_layers & attn_layers

        # Apply scope and v-delta filter
        candidates = resolve_scope_layers(
            activation_store=store,
            hook_point="pre_attn_norm",
            capture_type="hidden",
            scope=scope,
            required_delta_roles=("v",),
            delta_store=delta_store,
        )
        layers = [l for l in candidates if l in jointly_captured]

        # Structure info from the adapter
        n_heads = run_result.structure.n_attention_heads
        n_kv_heads = run_result.structure.n_kv_heads
        head_dim = run_result.structure.head_dim
        heads_per_kv = n_heads // n_kv_heads

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={
                "layers_requested": list(scope),
                "layers_used": list(layers),
                "boundary_fraction": boundary_fraction,
                "proof1_threshold": proof1_threshold,
                "scope_resolution": describe_scope_resolution(
                    scope, layers, "pre_attn_norm+attn_output"),
            },
        )

        per_layer_signed: dict[int, np.ndarray] = {}
        per_layer_amplitude: dict[int, float] = {}
        proof1_checks: list[dict] = []
        layer_attrs: list[torch.Tensor] = []

        for layer_idx in layers:
            h = store.get(layer_idx, "pre_attn_norm", "hidden")[0, :seq_len].float()
            alpha = store.get(
                layer_idx, "attn_output", "attention_weights"
            )[0, :, :seq_len, :seq_len].float()

            dw_v = delta_store.get(layer_idx, "v")

            v = torch.matmul(h, dw_v.float().T)
            v_heads = v.view(seq_len, n_kv_heads, head_dim)

            alpha_grouped = alpha.view(n_kv_heads, heads_per_kv, seq_len, seq_len)
            alpha_kv = alpha_grouped.mean(dim=1)

            head_attrs = []
            layer_amp = 0.0
            for kv_head in range(n_kv_heads):
                v_h = v_heads[:, kv_head, :]
                a_h = alpha_kv[kv_head]
                delta = torch.matmul(a_h, v_h)
                d_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                u_hat = delta / d_norm
                proj = torch.matmul(u_hat, v_h.T)
                signed = a_h * proj
                head_attrs.append(signed[-1, :])
                layer_amp += d_norm[-1].item()

                attr_sum = signed[-1, :].sum().item()
                delta_norm_val = d_norm[-1].item()
                error = abs(attr_sum - delta_norm_val)
                proof1_checks.append({
                    "layer": int(layer_idx),
                    "head": int(kv_head),
                    "attr_sum": round(attr_sum, 8),
                    "delta_norm": round(delta_norm_val, 8),
                    "error": float(f"{error:.2e}"),
                    "exact": error < proof1_threshold,
                })

            layer_attr = torch.stack(head_attrs).mean(dim=0)
            layer_attrs.append(layer_attr)
            per_layer_amplitude[int(layer_idx)] = layer_amp / n_kv_heads
            per_layer_signed[int(layer_idx)] = layer_attr.float().numpy()

        if not layer_attrs:
            result.per_token["signed_attribution_to_last"] = padded_per_token(seq_len)
            result.scalars["net_correction_to_last"] = float("nan")
            result.scalars["n_negative_tokens"] = 0
            result.scalars["has_negative_tokens"] = 0
            result.scalars["entropy"] = float("nan")
            result.scalars["top2_share"] = float("nan")
            result.scalars["middle_share"] = float("nan")
            result.scalars["interior_cv"] = float("nan")
            result.scalars["n_layers_used"] = 0
            result.objects["proof1_checks"] = proof1_checks
            self._annotate(result)
            return result

        avg_attr = torch.stack(layer_attrs).mean(dim=0).float().numpy()

        result.per_token["signed_attribution_to_last"] = avg_attr.astype(float)
        result.scalars["net_correction_to_last"] = float(avg_attr.sum())
        result.scalars["n_negative_tokens"] = int((avg_attr < 0).sum())
        result.scalars["has_negative_tokens"] = int(bool((avg_attr < 0).any()))
        result.scalars["n_layers_used"] = int(len(layers))

        # Distribution metrics over |signed_attr|
        attr_abs = np.abs(avg_attr)
        total = attr_abs.sum()
        attr_dist = (attr_abs / total) if total > 0 \
            else np.ones_like(attr_abs) / len(attr_abs)

        ent = float(-np.sum(attr_dist * np.log(attr_dist + 1e-10)))
        max_ent = float(np.log(seq_len)) if seq_len > 1 else 1.0
        result.scalars["entropy"] = float(ent / max_ent) if max_ent > 0 else 0.0

        n = len(attr_dist)
        boundary = max(1, round(boundary_fraction * n))
        if n >= 2:
            top2 = float(attr_dist[:boundary].sum() + attr_dist[-boundary:].sum())
            interior = (attr_dist[boundary:-boundary] if n > 2 * boundary
                        else np.array([]))
            if len(interior) > 0:
                middle = float(interior.sum())
                imean = float(interior.mean())
                icv = float(interior.std() / imean) if imean > 0 else 0.0
            else:
                middle = float(1.0 - top2)
                icv = 0.0
        else:
            top2 = 1.0
            middle = 0.0
            icv = 0.0
        result.scalars["top2_share"] = top2
        result.scalars["middle_share"] = middle
        result.scalars["interior_cv"] = icv

        result.per_layer_per_token["signed_attribution_to_last_by_layer"] = {
            li: arr.astype(float) for li, arr in per_layer_signed.items()
        }
        result.per_layer["amplitude"] = dict(per_layer_amplitude)

        result.objects["proof1_checks"] = proof1_checks

        self._annotate(result)
        return result

    def _annotate(self, result: MeasurementResult) -> None:
        result.field_specs["signed_attribution_to_last"] = FieldSpec(
            name="signed_attribution_to_last",
            kind="per_token",
            description=("Per-token contribution to the last position's "
                         "correction field (signed)."),
            units="correction-norm units",
            semantic_note=(
                "Index i is the contribution of token i to the LAST position's "
                "correction, averaged over the layers this measurement "
                "operated on. This is NOT a per-token correction; it "
                "measures which earlier tokens drove the correction at "
                "the final position."),
            length_invariant=False,
        )
        result.field_specs["net_correction_to_last"] = FieldSpec(
            name="net_correction_to_last", kind="scalar",
            description="Sum of signed_attribution_to_last across tokens.",
            length_invariant=False,
        )
        result.field_specs["n_negative_tokens"] = FieldSpec(
            name="n_negative_tokens", kind="scalar",
            description="Count of positions with negative signed attribution.",
            length_invariant=False,
        )
        result.field_specs["entropy"] = FieldSpec(
            name="entropy", kind="scalar",
            description="Shannon entropy of |signed_attr| distribution, "
                        "normalized by log(seq_len).",
            units="normalized bits", length_invariant=True,
        )
        for name in ("top2_share", "middle_share", "interior_cv"):
            result.field_specs[name] = FieldSpec(
                name=name, kind="scalar",
                description=f"Boundary-interior distribution statistic: {name}.",
                length_invariant=True,
            )
        result.field_specs["signed_attribution_to_last_by_layer"] = FieldSpec(
            name="signed_attribution_to_last_by_layer",
            kind="per_layer_per_token",
            description="Per-layer breakdown of signed attribution to the last "
                        "position (before cross-layer averaging).",
            length_invariant=False,
        )
        result.field_specs["amplitude"] = FieldSpec(
            name="amplitude", kind="per_layer",
            description="Per-layer correction-field amplitude at the last position.",
            length_invariant=True,
        )
        result.field_specs["proof1_checks"] = FieldSpec(
            name="proof1_checks", kind="object",
            description="Per-(layer, head) decomposition-exactness diagnostics "
                        "(sum of signed attr vs ||delta||).",
            length_invariant=True,
        )
        result.field_specs["n_layers_used"] = FieldSpec(
            name="n_layers_used", kind="scalar",
            description="Count of layers actually aggregated.",
            length_invariant=True,
        )
