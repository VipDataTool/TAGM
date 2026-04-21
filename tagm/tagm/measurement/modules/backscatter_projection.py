"""BackscatterProjection: per-prompt ΔW projections per (layer, role).

For each probe in the active probe set and each (layer, role) in the
user's selection, computes the projection of the probe's embedding through
the corresponding delta. Produces per-probe-per-sublayer magnitudes
that feed the correction_backscatter analysis.

Capture expectation:
  Empty — this is a pure weight-space measurement. No forward-pass
  capture is needed; it only reads probe embeddings and delta tensors.
"""
from __future__ import annotations

import numpy as np
import torch

from tagm.measurement.base import MeasurementModule
from tagm.measurement.parameters import ModuleParameter
from tagm.measurement.registry import register_measurement
from tagm.measurement.requirements import CaptureExpectation, ProbeRequirement
from tagm.measurement.result import FieldSpec, MeasurementResult


@register_measurement
class BackscatterProjection(MeasurementModule):
    name = "backscatter_projection"
    display_name = "Backscatter Projection"
    description = (
        "Projects probe embeddings through each selected (layer, role) ΔW, "
        "producing a (n_probes, n_sublayers) magnitude matrix. Feeds the "
        "correction_backscatter analysis."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="template_id",
            display_name="Probe template ID",
            kind="string", default="",
            description="Content hash of the probe template.",
        ),
        ModuleParameter(
            name="capture_signature",
            display_name="Probe capture signature",
            kind="string", default="",
            description="Capture-config signature used for the probe set.",
        ),
        ModuleParameter(
            name="depth_label",
            display_name="Probe depth",
            kind="select", default="subject",
            options=("subject", "escalation", "final"),
            description="Which probe depth to use as the query.",
        ),
        ModuleParameter(
            name="layers",
            display_name="Scope layers",
            description=(
                "Layer indices to project through. Empty means all layers "
                "for which the delta store has the required roles. "
                "Scope parameter."
            ),
            kind="layer_list", default=[],
        ),
        ModuleParameter(
            name="roles",
            display_name="Roles",
            description="Projection roles to include.",
            kind="multi_select",
            default=["q", "k", "v", "o", "gate", "up", "down"],
            options=("q", "k", "v", "o", "gate", "up", "down"),
        ),
    ]

    def capture_expectation(self, params):
        # Pure weight-space measurement — no forward capture required
        return CaptureExpectation.empty()

    def probe_requirements(self, params):
        tid = params.get("template_id") or ""
        sig = params.get("capture_signature") or ""
        if not tid or not sig:
            return None
        return ProbeRequirement(template_id=tid, capture_signature=sig)

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        depth_label = params.get("depth_label", "subject")
        scope = list(params.get("layers") or [])
        roles = list(params.get("roles") or [])

        # Resolve layers: intersection of scope (if given) and layers
        # present in the delta store
        delta_layers = set(delta_store.layers())
        if scope:
            layers = sorted(delta_layers & set(int(x) for x in scope))
        else:
            layers = sorted(delta_layers)

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={
                "template_id": params.get("template_id"),
                "capture_signature": params.get("capture_signature"),
                "depth_label": depth_label,
                "layers_requested": list(scope),
                "layers_used": list(layers),
                "roles": list(roles),
            },
        )

        probe_set = probes.get("probe_set") if isinstance(probes, dict) else probes
        if probe_set is None:
            result.objects["magnitude_matrix"] = []
            result.objects["probe_labels"] = []
            result.objects["sublayer_labels"] = []
            self._annotate(result)
            return result

        probe_matrix, probe_labels = probe_set.embeddings_matrix(depth_label)
        if probe_matrix.shape[0] == 0 or not layers or not roles:
            result.objects["magnitude_matrix"] = []
            result.objects["probe_labels"] = list(probe_labels)
            result.objects["sublayer_labels"] = []
            self._annotate(result)
            return result

        sublayer_labels: list[str] = []
        magnitudes: list[np.ndarray] = []
        probe_tensor = torch.from_numpy(probe_matrix.astype(np.float32))

        for layer_idx in layers:
            for role in roles:
                dw = delta_store.get_or_none(layer_idx, role)
                if dw is None:
                    continue
                if dw.shape[1] != probe_tensor.shape[1]:
                    sublayer_labels.append(f"L{layer_idx}_{role}")
                    magnitudes.append(np.zeros(probe_tensor.shape[0]))
                    continue
                projected = torch.matmul(probe_tensor, dw.float().T)
                mag = projected.norm(dim=-1).cpu().numpy()
                fnorm = delta_store.frob_norm(layer_idx, role)
                if fnorm > 0:
                    mag = mag / fnorm
                sublayer_labels.append(f"L{layer_idx}_{role}")
                magnitudes.append(mag.astype(float))

        if magnitudes:
            M = np.stack(magnitudes, axis=1)
        else:
            M = np.zeros((probe_tensor.shape[0], 0))

        result.objects["magnitude_matrix"] = M.astype(float).tolist()
        result.objects["probe_labels"] = list(probe_labels)
        result.objects["sublayer_labels"] = sublayer_labels
        result.scalars["n_probes"] = int(M.shape[0])
        result.scalars["n_sublayers"] = int(M.shape[1])
        result.scalars["mean_magnitude"] = float(M.mean()) if M.size else 0.0
        self._annotate(result)
        return result

    def _annotate(self, result: MeasurementResult) -> None:
        result.field_specs["magnitude_matrix"] = FieldSpec(
            name="magnitude_matrix", kind="object",
            description="(n_probes, n_sublayers) matrix of "
                        "||probe @ dW_p^T||/||dW_p||_F.",
            length_invariant=True,
        )
        result.field_specs["probe_labels"] = FieldSpec(
            name="probe_labels", kind="object",
            description="Labels for rows of magnitude_matrix.",
            length_invariant=True,
        )
        result.field_specs["sublayer_labels"] = FieldSpec(
            name="sublayer_labels", kind="object",
            description="Labels (e.g. 'L12_q') for columns of magnitude_matrix.",
            length_invariant=True,
        )
