"""MeasurementModule abstract base class.

One subclass per measurement. The framework:
  1. User sets a CaptureConfig at the pipeline level (one choice, up front).
  2. User selects a set of measurements and their parameters.
  3. Framework asks each measurement for its CaptureExpectation and validates
     that the user-set CaptureConfig satisfies it; rejects the selection if not.
  4. Pipeline runs one forward pass per the user-set CaptureConfig.
  5. Framework calls each module's compute() with the RunResult.
  6. compute() reads from the ActivationStore opportunistically — taking
     whatever layers are present at the hook points it cares about, narrowed
     by any scope parameters the user set.
  7. Framework validates the returned MeasurementResult against the per-token
     alignment contract and merges into the session record.

The contract distinction is deliberate:
  - The Pipeline (via the user's CaptureConfig) controls *what* is captured.
  - Measurements declare *what they need to exist* to run (an expectation),
    not *what to capture* (a request).
  - Measurement parameters control *scope of aggregation* over captured
    data, not the capture itself.

Modules are stateless — any per-model caching (SVD precomputes, probe
cache) is owned by a separate Cache/DeltaStore object, not held on the
module instance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from tagm.measurement.requirements import CaptureExpectation, ProbeRequirement
from tagm.measurement.result import MeasurementResult

if TYPE_CHECKING:
    from tagm.core.adapter.base import ModelAdapter
    from tagm.core.deltas.store import DeltaStore
    from tagm.core.pipeline import RunResult


class MeasurementModule(ABC):
    """Abstract base for per-prompt measurement modules."""

    # ── Identity ────────────────────────────────────────────────────
    name: str = ""              # short identifier, used as dict key in session record
    display_name: str = ""      # human-readable name for UI
    description: str = ""
    version: str = "0.1.0"

    # ── Parameter declarations ──────────────────────────────────────
    # List[ModuleParameter]. These are scope/aggregation parameters,
    # NOT capture parameters. Resolved (user-or-default) values are
    # passed to compute() via the params dict and recorded on the result.
    parameters: list = []

    # ── Inter-measurement dependencies ──────────────────────────────
    # Optional tuple of measurement names this measurement's compute()
    # reads from (via params["_dependencies"]). The orchestrator runs
    # dependencies first via topological sort.
    depends_on: tuple[str, ...] = ()

    # ── Capture expectation (what needs to exist, not what to capture) ──
    @abstractmethod
    def capture_expectation(self, params: dict) -> CaptureExpectation:
        """Declare what this measurement needs the capture config to provide.

        The orchestrator validates the user-set CaptureConfig against this
        expectation before dispatching. Violations are reported back to
        the user with actionable hints (e.g. "enable attention_weights
        at attn_output").

        This is NOT a request for capture. The measurement does not pick
        layers; the user's CaptureConfig does. The measurement just asks
        "does the capture include what I need to function at all?"

        The params dict is provided in case the answer depends on user-set
        parameters (rare — most expectations are measurement-intrinsic).
        """
        ...

    # ── Probe requirements ──────────────────────────────────────────
    def probe_requirements(self, params: dict) -> Optional[ProbeRequirement]:
        """Declare any probe embeddings this measurement needs from the ProbeStore.

        Default: no probe requirements. Override for probe-using
        measurements (correction-heatmap, correction-manifold, etc.).
        """
        return None

    # ── Base-side logit extraction (batch mode) ─────────────────────
    def base_extract(self, pipeline, prompt: str,
                      base_logits, params: dict) -> dict:
        """Extract base-side data needed by this measurement from base logits.

        Called once per prompt during the base phase of Pipeline.run_pair_batch.
        Returns a dict whose keys are fields on BatchBaseCache (e.g.
        'per_position_base_alts', 'base_counterfactual_tokens').

        Only invoked if capture_expectation(params).needs_base_logits is True.
        Default: empty dict.
        """
        return {}

    # ── Computation ─────────────────────────────────────────────────
    @abstractmethod
    def compute(
        self,
        run_result: "RunResult",
        adapter: "ModelAdapter",
        delta_store: "DeltaStore",
        params: dict,
        probes: Optional[dict] = None,
        base_cache: Optional[dict] = None,
    ) -> MeasurementResult:
        """Compute the measurement from captured data.

        Framework guarantees:
          - The user-set CaptureConfig has satisfied this measurement's
            capture_expectation(); at least the minimum hook points and
            capture types it asked for are present in run_result.activations.
          - delta_store contains deltas for whichever layers the capture
            covered (or returns None for layers outside any active filter).
          - probes contains probe embeddings if probe_requirements() returned
            a requirement.
          - base_cache contains base-side extracted data if this is a paired run.
          - params is fully resolved (defaults filled in, validated). Scope
            parameters (like 'layers') narrow aggregation over captured data;
            they do not change what was captured.

        The measurement reads from run_result.activations opportunistically,
        intersecting (a) what the user chose to capture, (b) what delta_store
        has deltas for, and (c) any user-set scope parameter. It operates on
        that intersection and records what it actually used in the
        MeasurementResult's parameters dict.
        """
        ...

    # ── Metadata (for UI and exports) ───────────────────────────────
    def metadata(self) -> dict:
        """Self-describing metadata for UI population and export."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "depends_on": list(self.depends_on),
            "parameters": [p.to_dict() for p in self.parameters],
        }
