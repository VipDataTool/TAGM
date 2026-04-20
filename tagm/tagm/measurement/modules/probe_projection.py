"""ProbeProjection: per-prompt projections onto probe-delta directions.

For each token, projects its residual-stream embedding at a specified
layer onto the subject axis of a probe set. Output is a per-token
(class, subclass, score) assignment that downstream analyses like
correction_heatmap and correction_manifold consume.

Capture expectation:
  Needs residual_post_block hidden states. The layer read is the
  `subject_layer` parameter; if that layer is not captured, the
  measurement fails loudly via the expectation validation step.

Probe requirement:
  Looks up a ProbeSet by (template_id, capture_signature, model_pair_id).
"""
from __future__ import annotations

import numpy as np

from tagm.measurement.base import MeasurementModule
from tagm.measurement.parameters import ModuleParameter
from tagm.measurement.registry import register_measurement
from tagm.measurement.requirements import CaptureExpectation, ProbeRequirement
from tagm.measurement.result import FieldSpec, MeasurementResult, padded_per_token


@register_measurement
class ProbeProjection(MeasurementModule):
    name = "probe_projection"
    display_name = "Probe Projection"
    description = (
        "Per-token projection of residual-stream embeddings onto probe-delta "
        "directions, yielding a (class, subclass, score) assignment per token."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="template_id",
            display_name="Probe template ID",
            description="Content hash of the probe template this projection uses.",
            kind="string", default="",
        ),
        ModuleParameter(
            name="capture_signature",
            display_name="Probe capture signature",
            description="Capture-config signature the probe set was generated with.",
            kind="string", default="",
        ),
        ModuleParameter(
            name="depth_label",
            display_name="Probe depth",
            description="Which probe depth to project against (e.g. 'subject').",
            kind="select", default="subject",
            options=("subject", "escalation", "final"),
        ),
        ModuleParameter(
            name="subject_layer",
            display_name="Subject layer",
            description=(
                "Captured layer whose residual_post_block hidden states "
                "are projected against probe directions. Must be present "
                "in the CaptureConfig."
            ),
            kind="int", default=12, min_value=0,
        ),
    ]

    def capture_expectation(self, params):
        return CaptureExpectation(
            hook_points_required=("residual_post_block",),
            capture_types_required=frozenset({"hidden"}),
            min_layers_captured=1,
        )

    def probe_requirements(self, params):
        tid = params.get("template_id") or ""
        sig = params.get("capture_signature") or ""
        if not tid or not sig:
            return None
        return ProbeRequirement(template_id=tid, capture_signature=sig)

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        seq_len = run_result.seq_len
        layer = int(params.get("subject_layer", 12))
        depth_label = params.get("depth_label", "subject")
        store = run_result.activations

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={k: params.get(k) for k in
                        ("template_id", "capture_signature",
                         "depth_label", "subject_layer")},
        )

        probe_set = probes.get("probe_set") if isinstance(probes, dict) else probes

        if probe_set is None:
            result.per_token["best_class_idx"] = padded_per_token(seq_len)
            result.per_token["best_score"] = padded_per_token(seq_len)
            result.objects["per_token_assignment"] = []
            self._annotate(result)
            return result

        probe_matrix, probe_labels = probe_set.embeddings_matrix(depth_label)
        if probe_matrix.shape[0] == 0:
            result.per_token["best_class_idx"] = padded_per_token(seq_len)
            result.per_token["best_score"] = padded_per_token(seq_len)
            result.objects["probe_labels"] = []
            self._annotate(result)
            return result

        if not store.has(layer, "residual_post_block", "hidden"):
            result.per_token["best_class_idx"] = padded_per_token(seq_len)
            result.per_token["best_score"] = padded_per_token(seq_len)
            result.objects["probe_labels"] = list(probe_labels)
            result.parameters["note"] = (
                f"subject_layer={layer} not captured at residual_post_block; "
                "projection skipped")
            self._annotate(result)
            return result

        h = store.get(layer, "residual_post_block", "hidden")[0, :seq_len]\
            .float().cpu().numpy()

        norms = np.linalg.norm(h, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        h_normed = h / norms

        scores = h_normed @ probe_matrix.T
        best_idx = np.argmax(scores, axis=1).astype(float)
        best_score = np.max(scores, axis=1).astype(float)

        result.per_token["best_class_idx"] = best_idx
        result.per_token["best_score"] = best_score
        result.objects["score_matrix"] = scores.astype(float).tolist()
        result.objects["probe_labels"] = list(probe_labels)
        result.objects["per_token_assignment"] = [
            {
                "token_index": int(i),
                "best_probe": probe_labels[int(best_idx[i])],
                "best_score": float(best_score[i]),
            }
            for i in range(seq_len)
        ]

        self._annotate(result)
        return result

    def _annotate(self, result: MeasurementResult) -> None:
        result.field_specs["best_class_idx"] = FieldSpec(
            name="best_class_idx", kind="per_token",
            description="Per-token index (into probe_labels) of the best-matching probe.",
            length_invariant=False,
        )
        result.field_specs["best_score"] = FieldSpec(
            name="best_score", kind="per_token",
            description="Per-token cosine score with the best-matching probe.",
            units="cosine", length_invariant=False,
        )
        result.field_specs["score_matrix"] = FieldSpec(
            name="score_matrix", kind="object",
            description="(seq_len, n_probes) per-token-per-probe cosine scores.",
            length_invariant=False,
        )
