"""CaptureConfig and CapturePoint.

A CapturePoint is a single capture instruction: "record capture type X at
hook point Y at layer Z with reduction R and precision P." A CaptureConfig
is a named collection of capture points.

CaptureConfigs are user-constructible (from UI form controls) and
saveable as named presets. They are validated against an adapter before
use. Under TAGM's contract, capture is a pipeline-level, user-set
concern — it is NOT derived from the set of selected measurements.
Measurements declare only `CaptureExpectation`s (what must exist to run),
which the orchestrator validates against the active CaptureConfig.

Nothing about a CaptureConfig depends on a specific model — only on the
family, via the adapter's declared HOOK_POINTS. The same CaptureConfig
can be used across model pairs within a family.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

from tagm.core.types import ValidationResult

if TYPE_CHECKING:
    from tagm.core.adapter.base import ModelAdapter


VALID_CAPTURE_TYPES = frozenset({"hidden", "attention_weights"})
VALID_PRECISIONS = frozenset({"model_dtype", "float32", "float16", "bfloat16"})
VALID_REDUCTIONS = frozenset({None, "mean", "last", "first"})


@dataclass(frozen=True)
class CapturePoint:
    """A single capture instruction.

    Fields:
      layer:       Layer index, or None for layer-independent hook points
                   (e.g. final_norm).
      hook_point:  Name from the adapter's HOOK_POINTS dict.
      capture:     frozenset of capture types to record at this point
                   (subset of VALID_CAPTURE_TYPES and of what the hook
                   point supports per adapter declaration).
      reduction:   None for full per-token tensor, or one of
                   "mean" | "last" | "first" to reduce across the
                   sequence dimension before storing.
      precision:   Storage precision for the captured tensor. "model_dtype"
                   keeps the model's native dtype; others force-cast.

    Frozen + hashable, so two points comparing equal can be deduplicated
    when unioning requirements across measurements.
    """
    layer: Optional[int]
    hook_point: str
    capture: frozenset[str]
    reduction: Optional[str] = None
    precision: str = "model_dtype"

    def __post_init__(self):
        if not self.capture:
            raise ValueError("CapturePoint.capture must be a non-empty frozenset")
        if not isinstance(self.capture, frozenset):
            # Friendly coercion; keeps the dataclass hashable
            object.__setattr__(self, "capture", frozenset(self.capture))
        invalid = self.capture - VALID_CAPTURE_TYPES
        if invalid:
            raise ValueError(
                f"Invalid capture types in CapturePoint: {sorted(invalid)}. "
                f"Valid: {sorted(VALID_CAPTURE_TYPES)}")
        if self.precision not in VALID_PRECISIONS:
            raise ValueError(
                f"Invalid precision '{self.precision}'. "
                f"Valid: {sorted(VALID_PRECISIONS)}")
        if self.reduction not in VALID_REDUCTIONS:
            raise ValueError(
                f"Invalid reduction '{self.reduction}'. "
                f"Valid: {sorted(v for v in VALID_REDUCTIONS if v is not None)} "
                f"or None")

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "hook_point": self.hook_point,
            "capture": sorted(self.capture),
            "reduction": self.reduction,
            "precision": self.precision,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CapturePoint":
        return cls(
            layer=d["layer"],
            hook_point=d["hook_point"],
            capture=frozenset(d["capture"]),
            reduction=d.get("reduction"),
            precision=d.get("precision", "model_dtype"),
        )


@dataclass(frozen=True)
class CaptureConfig:
    """A named, complete capture specification.

    Constructed by the UI (from form controls), programmatically (by
    measurement modules declaring their requirements, then unioned by
    the framework), or loaded from a saved preset.

    The `points` tuple is order-irrelevant for correctness; the hook
    installer groups them by (layer, hook_point) internally. Duplicates
    across requirements are deduplicated at union time.
    """
    name: str
    points: tuple[CapturePoint, ...]
    description: str = ""

    # ── Serialization ───────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaptureConfig":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            points=tuple(CapturePoint.from_dict(p) for p in d["points"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "CaptureConfig":
        return cls.from_dict(json.loads(s))

    # ── Union (for merging measurement requirements) ────────────────
    def union(self, other: "CaptureConfig") -> "CaptureConfig":
        """Return a new config containing the union of points from self and other.

        Points that are structurally identical (same layer, hook_point,
        capture, reduction, precision) are deduplicated. Points that
        differ in reduction/precision at the same (layer, hook_point)
        are both kept — the installer resolves by taking the primary
        and logging a warning.

        Name and description are taken from self; caller is responsible
        for renaming the unioned config if desired.
        """
        seen = set(self.points)
        merged = list(self.points)
        for p in other.points:
            if p not in seen:
                merged.append(p)
                seen.add(p)
        return CaptureConfig(
            name=self.name,
            description=self.description or other.description,
            points=tuple(merged),
        )

    # ── Validation against adapter ──────────────────────────────────
    def validate(self, adapter: "ModelAdapter") -> ValidationResult:
        """Check that every capture point is satisfiable by the given adapter."""
        errors: list[str] = []
        warnings: list[str] = []
        for i, p in enumerate(self.points):
            if p.hook_point not in adapter.HOOK_POINTS:
                errors.append(
                    f"Point {i}: hook point '{p.hook_point}' not declared by "
                    f"adapter '{adapter.family_id}'. Available: "
                    f"{list(adapter.HOOK_POINTS)}")
                continue
            spec = adapter.HOOK_POINTS[p.hook_point]
            if spec.layer_independent and p.layer is not None:
                warnings.append(
                    f"Point {i}: hook point '{p.hook_point}' is "
                    f"layer-independent but layer={p.layer} was specified "
                    f"(will be ignored)")
            if not spec.layer_independent and p.layer is None:
                errors.append(
                    f"Point {i}: hook point '{p.hook_point}' requires a "
                    f"layer index")
            unsupported = p.capture - spec.captures
            if unsupported:
                errors.append(
                    f"Point {i}: hook point '{p.hook_point}' does not "
                    f"support capture types {sorted(unsupported)}; "
                    f"this point supports {sorted(spec.captures)}")
        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)

    # ── Signature (for content-addressed caching) ───────────────────
    def signature(self) -> str:
        """Stable hash over the structural content of the config.

        Used by the probe subsystem to decide whether a cached probe set
        was generated with the same capture config. Name and description
        are excluded; only the point set matters.
        """
        import hashlib
        canonical = sorted(p.to_dict() for p in self.points)
        blob = json.dumps(canonical, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    # ── Empty / combinator helpers ──────────────────────────────────
    @classmethod
    def empty(cls, name: str = "empty") -> "CaptureConfig":
        """Empty config — valid against any adapter, captures nothing."""
        return cls(name=name, points=(), description="No capture")

    def with_name(self, name: str, description: str = "") -> "CaptureConfig":
        return CaptureConfig(
            name=name,
            description=description or self.description,
            points=self.points,
        )
