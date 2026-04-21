"""AnalysisModule abstract base.

Analysis modules aggregate across many prompts' measurement results. They
declare which measurements they depend on (by measurement name); the
framework checks the session record for those dependencies before running
the analysis and raises a clear error if any are absent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AnalysisResult:
    """Output of an analysis module.

    Loose schema — analyses produce varied outputs (clusters, plots,
    tables), so the shape is `objects`-centric with scalars for summary
    statistics. Serialization uses the same NaN→None sanitization as
    MeasurementResult.
    """
    analysis_name: str
    analysis_version: str

    scalars: dict[str, float] = field(default_factory=dict)
    objects: dict[str, Any] = field(default_factory=dict)
    per_prompt: dict[str, Any] = field(default_factory=dict)
    # per_prompt[field][prompt_id] = ... — used for per-prompt derived
    # values that the analysis writes back into the session.

    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        import math
        def sanitize(v):
            if v is None: return None
            if isinstance(v, (bool, int, str)): return v
            if isinstance(v, float):
                return None if (math.isnan(v) or math.isinf(v)) else v
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (np.floating,)):
                f = float(v)
                return None if (math.isnan(f) or math.isinf(f)) else f
            if isinstance(v, np.ndarray):
                return [sanitize(x) for x in v.tolist()]
            if isinstance(v, dict):
                return {str(k): sanitize(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [sanitize(x) for x in v]
            return v
        return {
            "analysis_name": self.analysis_name,
            "analysis_version": self.analysis_version,
            "scalars": sanitize(self.scalars),
            "objects": sanitize(self.objects),
            "per_prompt": sanitize(self.per_prompt),
            "parameters": sanitize(self.parameters),
            "warnings": list(self.warnings),
        }


class AnalysisModule(ABC):
    """Abstract base for post-session analysis modules."""

    name: str = ""
    display_name: str = ""
    description: str = ""
    version: str = "0.1.0"

    # Measurement dependencies (names from the measurement registry).
    # The framework validates these are present in the session before
    # running the analysis. A missing dependency is a loud error, not
    # silent degradation.
    depends_on_measurements: tuple[str, ...] = ()

    # Parameter declarations (same ModuleParameter class as measurements).
    parameters: list = []

    @abstractmethod
    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> AnalysisResult:
        """Compute the analysis from a session record.

        session: dict in the canonical session shape. Keys include:
          - 'prompts': list of per-prompt dicts, each containing
            'prompt', 'category', and keyed measurement results.
          - 'model_pair': {instruct, base, adapter_family, ...}
          - 'capture_config': the unioned CaptureConfig used.
          - 'structure': model structure snapshot (n_layers, etc.).

        params: resolved parameter values for this analysis.

        probes: optional probe data (for analyses that need probe sets
        directly, like correction_manifold).

        Returns an AnalysisResult. The framework merges it into the
        session record under result.analysis_name.
        """
        ...

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "depends_on_measurements": list(self.depends_on_measurements),
            "parameters": [p.to_dict() for p in self.parameters],
        }

    def check_dependencies(self, session: dict) -> list[str]:
        """Return a list of error strings for missing measurement dependencies.

        Reads the flat record shape: a measurement is "present on a prompt"
        if any of the root-level keys it publishes is on that prompt's dict.
        """
        prompts = session.get("prompts") or []
        if not prompts:
            return [f"Analysis '{self.name}' needs at least one prompt"]
        errors: list[str] = []
        for required in self.depends_on_measurements:
            keys = _MEASUREMENT_OUTPUT_KEYS.get(required, (required,))
            missing = [i for i, p in enumerate(prompts)
                       if not any(k in p for k in keys)]
            if missing:
                if len(missing) == len(prompts):
                    errors.append(
                        f"Analysis '{self.name}' requires measurement "
                        f"'{required}'; no prompts have it")
                else:
                    errors.append(
                        f"Analysis '{self.name}' requires measurement "
                        f"'{required}'; {len(missing)}/{len(prompts)} "
                        f"prompts are missing it")
        return errors


# Map each measurement name to the root-level keys it publishes on a
# prompt record. Used by check_dependencies; mirrors the orchestrator's
# `_record_measurement` mapping (one-way: name → keys it lands).
_MEASUREMENT_OUTPUT_KEYS = {
    "stress_score":              ("stress_score", "per_token_stress"),
    "last_position_attribution": ("signed_attr", "net_correction", "entropy"),
    "amplitude_trajectory":      ("amplitude_trajectory",
                                  "amplitude_normalized"),
    "amplitude_derived_metrics": ("per_token_attn_frac",
                                  "per_token_coherence",
                                  "per_token_sublayer_rank",
                                  "token_similarity"),
    "lateral_tension_profile":   ("ltp",),
    "spectral_field_density":    ("sfd",),
    "rank_displacement":         ("rank_displacement",),
    "probe_projection":          ("probe_projection",),
    "per_token_embedding":       ("per_token_embeddings",
                                  "per_token_final_emb"),
    "backscatter_projection":    ("backscatter",),
}
