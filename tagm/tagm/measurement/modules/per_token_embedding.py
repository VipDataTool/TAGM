"""PerTokenEmbedding: captures per-token residual-stream embeddings at multiple depths.

Feeds downstream analysis modules that need raw embedding vectors
(domain_surface, correction_manifold, etc.) rather than derived metrics.
Exported flag controls whether the (large) per-token vectors are
serialized to the session record.

Capture expectation:
  Needs residual_post_block hidden states captured at one or more layers.
  The user's CaptureConfig picks which layers. This measurement operates
  on layers listed in the `depths` parameter that are actually present
  in the store; layers named in `depths` but absent from the store are
  skipped with a recorded note.

Output:
  per_token_embeddings: dict[depth_label -> (seq_len, hidden_size) array]
  L2-normalized along the hidden dimension.
"""
from __future__ import annotations

import numpy as np

from tagm.measurement.base import MeasurementModule
from tagm.measurement.parameters import ModuleParameter
from tagm.measurement.registry import register_measurement
from tagm.measurement.requirements import CaptureExpectation
from tagm.measurement.result import FieldSpec, MeasurementResult


@register_measurement
class PerTokenEmbedding(MeasurementModule):
    name = "per_token_embedding"
    display_name = "Per-Token Embedding"
    description = (
        "Captures per-token residual-stream embeddings at user-specified "
        "depth-labeled layers. Feeds analysis modules that need the raw "
        "embedding vectors."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="depths",
            display_name="Depth labels",
            description=(
                "Comma-separated label:layer pairs (e.g. "
                "'subject:12,escalation:18'). Each label names a depth; "
                "the layer index identifies which captured layer to read "
                "for that depth. Labels whose layers are not in the "
                "CaptureConfig will be skipped (noted in parameters.skipped_depths)."
            ),
            kind="string", default="subject:12,escalation:18",
        ),
        ModuleParameter(
            name="include_final_norm",
            display_name="Include final norm",
            description="Also read the final_norm output as depth 'final'.",
            kind="bool", default=True,
        ),
        ModuleParameter(
            name="include_in_export",
            display_name="Include in JSON export",
            description=(
                "If true, these (large) arrays are serialized in the session "
                "export. Off by default to keep exports compact."
            ),
            kind="bool", default=False, advanced=True,
        ),
    ]

    def capture_expectation(self, params):
        depths = _parse_depths(params.get("depths", ""))
        include_final = bool(params.get("include_final_norm", True))

        # We need residual_post_block captured (at some layer) if the
        # user named any depth layers, and/or final_norm if requested.
        hook_points = []
        if depths:
            hook_points.append("residual_post_block")
        return CaptureExpectation(
            hook_points_required=tuple(hook_points),
            capture_types_required=frozenset({"hidden"}),
            needs_final_norm=include_final,
            min_layers_captured=1 if depths else 0,
        )

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        seq_len = run_result.seq_len
        store = run_result.activations
        depths = _parse_depths(params.get("depths", ""))
        include_final = bool(params.get("include_final_norm", True))
        export = bool(params.get("include_in_export", False))

        # Determine which requested depths are actually available
        available_layers = set(store.layers_for("residual_post_block", "hidden"))
        used_depths: list[tuple[str, int]] = []
        skipped_depths: list[dict] = []
        for label, layer in depths:
            if layer in available_layers:
                used_depths.append((label, layer))
            else:
                skipped_depths.append({
                    "label": label, "layer": layer,
                    "reason": "layer not captured at residual_post_block",
                })

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={
                "depths_requested": [{"label": l, "layer": ly} for l, ly in depths],
                "depths_used": [{"label": l, "layer": ly} for l, ly in used_depths],
                "skipped_depths": skipped_depths,
                "include_final_norm": include_final,
                "include_in_export": export,
            },
        )

        per_token_embeddings: dict[str, list[list[float]]] = {}

        for label, layer in used_depths:
            h = store.get(layer, "residual_post_block", "hidden")
            arr = h[0, :seq_len].float().cpu().numpy()
            arr = _normalize_rows(arr)
            if export:
                per_token_embeddings[label] = arr.astype(float).tolist()
            result.scalars[f"{label}_hidden_size"] = (
                int(arr.shape[1]) if arr.size else 0)

        if include_final and store.has(None, "final_norm", "hidden"):
            h = store.get(None, "final_norm", "hidden")
            arr = h[0, :seq_len].float().cpu().numpy()
            arr = _normalize_rows(arr)
            if export:
                per_token_embeddings["final"] = arr.astype(float).tolist()
            result.scalars["final_hidden_size"] = (
                int(arr.shape[1]) if arr.size else 0)

        if export:
            result.objects["per_token_embeddings"] = per_token_embeddings

        result.field_specs["per_token_embeddings"] = FieldSpec(
            name="per_token_embeddings", kind="object",
            description=(
                "Per-depth (seq_len, hidden_size) arrays of L2-normalized "
                "per-token residual embeddings. Only present when "
                "include_in_export is True."
            ),
            length_invariant=False,
        )
        return result


def _parse_depths(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for entry in (spec or "").split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        label, layer = entry.split(":", 1)
        try:
            out.append((label.strip(), int(layer.strip())))
        except ValueError:
            continue
    return out


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return arr / norms
