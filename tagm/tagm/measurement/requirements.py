"""Measurement expectations and probe requirements.

An earlier draft had measurements *declare* capture requirements that
the orchestrator unioned into a CaptureConfig. That design conflated
two concerns:

  - *Capture selection* — which layers, hook points, and capture types
    are recorded during forward passes. This is a pipeline-level choice
    the user makes once.

  - *Measurement scope* — which of the captured layers a given
    measurement aggregates over. This is a per-measurement parameter.

The corrected contract: the user sets a CaptureConfig at the pipeline
level. Measurements read from whatever is in the ActivationStore. A
measurement declares a `CaptureExpectation` — the minimum guarantees
it needs to run at all (e.g. "I need attention_weights somewhere", "I
need the final_norm hidden state") — and the orchestrator validates
that the active CaptureConfig satisfies those expectations before
dispatching. Measurements do not pick layers; the user does, and the
measurement operates on whatever is present in the store.

ProbeRequirement is unchanged from the earlier draft: it says "I need
the probe set generated from template X with capture-signature Y." The
framework looks it up in the ProbeStore; if missing, it raises
ProbeNotAvailableError.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CaptureExpectation:
    """What a measurement needs the pipeline's CaptureConfig to provide.

    This is a *validation declaration*, not a *capture request*. The
    orchestrator uses it to check that the user-set CaptureConfig has
    enough in it to satisfy this measurement, and to reject a
    measurement selection that could not possibly run against the
    current capture.

    Fields:
      hook_points_required:    Hook points the measurement needs at
                               least one layer of (e.g. ("pre_attn_norm",)).
                               The measurement does not pick *which*
                               layers; the user's CaptureConfig does.
                               The measurement just needs at least one
                               layer to exist at each required hook point.
      capture_types_required:  Capture types required across every
                               entry in hook_points_required. Usually
                               frozenset({"hidden"}); override if a
                               measurement also needs attention weights
                               at the same hook point.
      needs_attention_weights: True if the measurement needs attention
                               weights captured at the attn_output hook
                               point. Separate because attention weights
                               are a capture type with special
                               orchestration implications
                               (output_attentions flag on forward).
      needs_final_norm:        True if the measurement needs the layer-
                               independent final_norm hidden state.
      needs_base_logits:       True if the measurement needs the base
                               model's logits (for KL divergence, base-
                               counterfactual probing, etc.).
      min_layers_captured:     Minimum number of distinct layer indices
                               that must be present in the store at the
                               required hook points. Default 1.
    """
    hook_points_required: tuple[str, ...] = ()
    capture_types_required: frozenset[str] = frozenset({"hidden"})
    needs_attention_weights: bool = False
    needs_final_norm: bool = False
    needs_base_logits: bool = False
    min_layers_captured: int = 1

    @classmethod
    def empty(cls) -> "CaptureExpectation":
        return cls()

    @property
    def needs_base_model(self) -> bool:
        """Whether satisfying this expectation requires the base model."""
        return self.needs_base_logits


@dataclass
class ExpectationViolation:
    """A specific reason a measurement can't run against the current capture.

    Carried in the orchestrator's validation output so users see a
    targeted "to run StressScore, enable pre_attn_norm capture at at
    least one layer" rather than a generic "measurement failed."
    """
    measurement_name: str
    reason: str


@dataclass(frozen=True)
class ProbeRequirement:
    """A measurement's declaration of the probe set it needs.

    A ProbeSet is looked up in the ProbeStore by
    (model_pair_id, template_id, capture_signature). The model_pair_id
    is provided by the framework from the currently loaded Pipeline;
    the module supplies template_id and capture_signature.

    Optional filters narrow the loaded probe set to a subset (e.g. a
    measurement that only uses one class's worth of probe embeddings).
    """
    template_id: str
    capture_signature: str
    subclass_filter: Optional[tuple[str, ...]] = None
    class_filter: Optional[tuple[str, ...]] = None


class ProbeNotAvailableError(Exception):
    """Raised by the framework when a ProbeRequirement cannot be satisfied.

    Surfaced to the user with an actionable message: generate the required
    probe set first, or remove the probe-using measurement from the session.
    """
    def __init__(self, requirement: ProbeRequirement,
                 available_sets: Optional[list[str]] = None):
        self.requirement = requirement
        self.available_sets = available_sets or []
        super().__init__(
            f"Probe set for template '{requirement.template_id}' with capture "
            f"signature '{requirement.capture_signature}' not found. "
            f"Available probe sets: {self.available_sets}. "
            f"Generate the required probe set before running this measurement."
        )


def validate_expectation(expectation: CaptureExpectation,
                          capture_config) -> list[str]:
    """Validate that a CaptureConfig satisfies a measurement's expectation.

    Returns a list of human-readable violation reasons. Empty list means
    the expectation is satisfied.

    Accepts a CaptureConfig (not imported at module top level to avoid
    coupling the requirements module to capture internals).
    """
    # Build an index of what's in the config
    points_by_hook: dict[str, set[int]] = {}
    capture_types_by_hook: dict[str, set[str]] = {}
    for p in capture_config.points:
        if p.layer is not None:
            points_by_hook.setdefault(p.hook_point, set()).add(p.layer)
        capture_types_by_hook.setdefault(p.hook_point, set()).update(p.capture)

    errors: list[str] = []

    # Required hook points with enough layers
    for hp in expectation.hook_points_required:
        if hp not in points_by_hook and hp != "final_norm":
            errors.append(
                f"capture config has no '{hp}' hook points; at least "
                f"{expectation.min_layers_captured} layer(s) required")
            continue
        n_layers = len(points_by_hook.get(hp, set()))
        if hp != "final_norm" and n_layers < expectation.min_layers_captured:
            errors.append(
                f"capture config has {hp} at {n_layers} layer(s); "
                f"{expectation.min_layers_captured} required")
        # Required capture types at that hook point
        have_types = capture_types_by_hook.get(hp, set())
        missing_types = expectation.capture_types_required - have_types
        if missing_types:
            errors.append(
                f"capture config at '{hp}' does not include "
                f"capture types {sorted(missing_types)}; has "
                f"{sorted(have_types)}")

    # Attention weights
    if expectation.needs_attention_weights:
        attn_types = capture_types_by_hook.get("attn_output", set())
        if "attention_weights" not in attn_types:
            errors.append(
                "measurement needs attention_weights captured at "
                "'attn_output'; current capture config does not include it")

    # Final norm
    if expectation.needs_final_norm:
        if "final_norm" not in capture_types_by_hook:
            errors.append(
                "measurement needs final_norm hidden state captured; "
                "current capture config does not include final_norm")

    return errors
