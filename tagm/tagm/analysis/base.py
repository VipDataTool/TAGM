"""AnalysisModule base class + ModuleOutput dataclass.

An analysis module is a post-session consumer: it reads primary data
(per-prompt MeasurementResults on the session record) and optional
secondary data (other analyses' outputs), plus any resource handles
it declares a need for, and returns a ModuleOutput with three
compartments.

The module never writes anywhere except its own return value. The
framework (service/modules_runner.py) wraps that return value in a
four-compartment mailbox and stores it in session.record.analyses[name].
See TAGM_analysis_layer_interface.md for the full contract.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Avoid runtime import cycles; these are only for type annotations.
    from tagm.core.deltas.store import DeltaStore
    from tagm.core.pipeline import Pipeline
    from tagm.probes.artifact import ProbeSet


# ── JSON sanitation (shared with MeasurementResult) ────────────────

def _sanitize(v: Any) -> Any:
    """Coerce numpy/NaN/inf to JSON-safe values."""
    if v is None:
        return None
    if isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, np.ndarray):
        return [_sanitize(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize(x) for x in v]
    return v


# ── ModuleOutput ───────────────────────────────────────────────────

@dataclass
class ModuleOutput:
    """The only thing a module is allowed to produce.

    Three compartments:
      - scalars:    summary numbers the UI shows in the card header strip.
      - objects:    everything else (tables, arrays, nested dicts).
      - per_prompt: values keyed by prompt_id that per-prompt viewers pick out.

    No name, no version, no timestamps, no warnings, no parameters.
    Those are framework-owned and live in the mailbox's other three
    compartments (module / run / sources) that wrap this one.
    """
    scalars: dict[str, float] = field(default_factory=dict)
    objects: dict[str, Any] = field(default_factory=dict)
    per_prompt: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scalars": _sanitize(self.scalars),
            "objects": _sanitize(self.objects),
            "per_prompt": _sanitize(self.per_prompt),
        }


# ── AnalysisModule ─────────────────────────────────────────────────

class AnalysisModule(ABC):
    """Abstract base for post-session analysis modules.

    Subclasses declare identity, data dependencies, resource needs,
    and a parameter schema as class attributes, then implement run().
    The framework handles registration, discovery, validation,
    dispatch, failure isolation, and storage.

    See TAGM_analysis_layer_interface.md §5 for the declaration
    template and §9 for the per-module port checklist.
    """

    # ── Identity ───────────────────────────────────────────────────
    name: str = ""
    display_name: str = ""
    description: str = ""
    version: str = "0.1.0"

    # ── Data dependencies ──────────────────────────────────────────
    # Measurement names this analysis reads from each prompt's
    # `.measurements` dict. The framework verifies presence before
    # dispatch.
    #
    # Semantics (enforced by check_dependencies):
    #   - Empty tuple: no dependency.
    #   - One entry: every prompt must have that measurement.
    #   - Multiple entries: every prompt must have AT LEAST ONE.
    #     This supports modules that degrade gracefully between
    #     interchangeable sources (e.g. CFT accepts LTP or RD).
    depends_on_measurements: tuple[str, ...] = ()

    # ── Resource requirements ──────────────────────────────────────
    # Boolean flags. If True, the framework resolves the resource and
    # passes it as a kwarg to run(). If the resource is absent, the
    # framework raises before dispatch; the module never sees None
    # when a required resource is unavailable.
    requires_probe_set: bool = False
    requires_delta_store: bool = False
    requires_pipeline: bool = False

    # ── Prompt-count gate ──────────────────────────────────────────
    min_prompts: int = 1

    # ── User-settable parameters ───────────────────────────────────
    # List of ModuleParameter (from tagm.measurement.parameters).
    parameters: list = []

    # ── Execution ──────────────────────────────────────────────────
    @abstractmethod
    def run(
        self,
        session: dict,
        params: dict,
        *,
        progress: Callable[[str], None],
        probes: Optional["ProbeSet"] = None,
        delta_store: Optional["DeltaStore"] = None,
        pipeline: Optional["Pipeline"] = None,
    ) -> ModuleOutput:
        """Compute the analysis.

        Args:
          session: SessionRecord.to_dict() — read-only. Contains
            `prompts` (list of per-prompt dicts each with a
            `measurements` dict), `analyses` (prior analyses'
            mailboxes), plus `model_pair`, `structure`,
            `capture_config`, `measurements_config`, `probe_sets`.
          params: resolved + validated parameter values. No missing
            keys, no out-of-range values.
          progress: progress("message") publishes to the UI status
            line. No-op if unused.
          probes: present iff requires_probe_set = True.
          delta_store: present iff requires_delta_store = True.
          pipeline: present iff requires_pipeline = True.

        Returns:
          ModuleOutput. The framework wraps it in the mailbox.

        Raises:
          On unrecoverable error. The framework catches and records
          the error in the mailbox's run.error field. The module
          does not return a partial result on failure.
        """
        ...

    # ── Metadata (for UI /api/modules) ─────────────────────────────
    def metadata(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "depends_on_measurements": list(self.depends_on_measurements),
            "requires_probe_set": self.requires_probe_set,
            "requires_delta_store": self.requires_delta_store,
            "requires_pipeline": self.requires_pipeline,
            "min_prompts": self.min_prompts,
            "parameters": [p.to_dict() for p in self.parameters],
        }

    # ── Dependency check (framework calls before dispatch) ─────────
    def check_dependencies(self, session: dict) -> list[str]:
        """Return a list of error strings for missing measurement deps.

        Semantics (see depends_on_measurements above):
          - Empty: always passes.
          - Single entry: strict — every prompt must have it.
          - Multiple entries: disjunctive — every prompt must have
            at least one. Partial coverage is not an error; modules
            opt into graceful degradation by declaring multiple deps.
        """
        prompts = session.get("prompts") or []
        if not prompts:
            return [f"Analysis '{self.name}' needs at least one prompt "
                    f"with measurements"]
        if not self.depends_on_measurements:
            return []

        errors: list[str] = []
        required = tuple(self.depends_on_measurements)

        if len(required) == 1:
            name = required[0]
            missing = [i for i, p in enumerate(prompts)
                       if name not in (p.get("measurements") or {})]
            if missing:
                if len(missing) == len(prompts):
                    errors.append(
                        f"Analysis '{self.name}' requires measurement "
                        f"'{name}'; no prompts have it")
                else:
                    errors.append(
                        f"Analysis '{self.name}' requires measurement "
                        f"'{name}'; {len(missing)}/{len(prompts)} prompts "
                        f"are missing it")
        else:
            none_with_any = True
            for p in prompts:
                ms = p.get("measurements") or {}
                if any(r in ms for r in required):
                    none_with_any = False
                    break
            if none_with_any:
                errors.append(
                    f"Analysis '{self.name}' requires at least one of "
                    f"{list(required)}; no prompts have any of them")

        return errors
