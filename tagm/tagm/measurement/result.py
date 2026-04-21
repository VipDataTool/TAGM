"""MeasurementResult: the standard output of a measurement module.

Enforces TAGM's per-token alignment contract: every per-token array has
length `seq_len`, indexed by raw token position. Positions where the
measurement is undefined carry NaN (for floats) or None (for objects).
No per-token array ever has length `seq_len - 1` with an offset. The
framework validates this before merging a result into the session record.

FieldSpec carries self-describing metadata so exports don't require
out-of-band knowledge of what each field means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ── Field specs (carried with results) ──────────────────────────────

@dataclass(frozen=True)
class FieldSpec:
    """Describes a single field in a MeasurementResult.

    Self-describing metadata so that exports carry enough context for
    a consumer to interpret each number without reference to source
    code. The export format includes field_specs for every measurement.
    """
    name: str
    kind: str                          # "scalar" | "per_token" | "per_layer"
                                        # | "per_layer_per_token" | "object"
    description: str
    length_invariant: bool = False     # True if comparable across prompts of different lengths
    units: Optional[str] = None        # e.g. "bits", "cosine", "probability"
    deterministic: bool = True          # True if this value is deterministic given the model + inputs
    semantic_note: Optional[str] = None
    # Human-readable clarification of the exact semantics, specifically to
    # avoid TASM's class of bugs where `signed_attr[i]` meant "contribution
    # to last position's correction" despite the per-token-looking name.
    # Measurements should use this to state what their per-token arrays
    # are actually indexed on.


# ── The result dataclass ────────────────────────────────────────────

@dataclass
class MeasurementResult:
    """Standard output of a measurement module for one prompt.

    Fields are grouped by shape:
      - scalars:             single floats
      - per_token:           length-seq_len float arrays (NaN where undefined)
      - per_layer:           dicts keyed by layer_idx, values are scalars
      - per_layer_per_token: dicts keyed by layer_idx, values are length-seq_len arrays
      - objects:             non-numeric results (strings, nested dicts, tuple lists)

    The `field_specs` dict carries one FieldSpec per field name across all
    groups — consumers look up a field name in `field_specs` to understand
    what it means, regardless of which group it lives in.

    The `parameters` dict records the exact parameter values used to
    compute this result. Recorded here (not inferred from defaults) so
    the "no hidden parameters" contract survives export/reload.
    """
    measurement_name: str
    measurement_version: str

    scalars: dict[str, float] = field(default_factory=dict)
    per_token: dict[str, np.ndarray] = field(default_factory=dict)
    per_layer: dict[str, dict[int, float]] = field(default_factory=dict)
    per_layer_per_token: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    objects: dict[str, Any] = field(default_factory=dict)

    field_specs: dict[str, FieldSpec] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    # ── Validation ──────────────────────────────────────────────────
    def validate(self, seq_len: int) -> list[str]:
        """Check that per-token arrays have the correct length.

        Returns a list of human-readable error strings; empty list means
        valid. The framework treats a non-empty list as a contract
        violation and does not merge the result.
        """
        errors: list[str] = []

        for name, arr in self.per_token.items():
            if not isinstance(arr, np.ndarray):
                errors.append(
                    f"per_token['{name}'] is not a numpy array "
                    f"(got {type(arr).__name__})")
                continue
            if arr.shape[0] != seq_len:
                errors.append(
                    f"per_token['{name}'] has length {arr.shape[0]}, "
                    f"expected {seq_len} (seq_len contract)")

        for name, by_layer in self.per_layer_per_token.items():
            if not isinstance(by_layer, dict):
                errors.append(
                    f"per_layer_per_token['{name}'] is not a dict "
                    f"(got {type(by_layer).__name__})")
                continue
            for layer_idx, arr in by_layer.items():
                if not isinstance(arr, np.ndarray):
                    errors.append(
                        f"per_layer_per_token['{name}'][{layer_idx}] "
                        f"is not a numpy array")
                    continue
                if arr.shape[0] != seq_len:
                    errors.append(
                        f"per_layer_per_token['{name}'][{layer_idx}] "
                        f"has length {arr.shape[0]}, expected {seq_len}")

        return errors

    # ── Serialization ───────────────────────────────────────────────
    def to_dict(self) -> dict:
        """JSON-safe dict representation with NaN → None coercion."""
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
                return {str(k): sanitize(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [sanitize(x) for x in v]
            if hasattr(v, "to_dict"):
                return sanitize(v.to_dict())
            return v

        return {
            "measurement_name": self.measurement_name,
            "measurement_version": self.measurement_version,
            "scalars": sanitize(self.scalars),
            "per_token": {k: sanitize(v) for k, v in self.per_token.items()},
            "per_layer": {k: sanitize(v) for k, v in self.per_layer.items()},
            "per_layer_per_token": {
                k: {str(lk): sanitize(la) for lk, la in v.items()}
                for k, v in self.per_layer_per_token.items()
            },
            "objects": sanitize(self.objects),
            "field_specs": {
                k: {
                    "name": fs.name,
                    "kind": fs.kind,
                    "description": fs.description,
                    "length_invariant": fs.length_invariant,
                    "units": fs.units,
                    "deterministic": fs.deterministic,
                    "semantic_note": fs.semantic_note,
                }
                for k, fs in self.field_specs.items()
            },
            "parameters": sanitize(self.parameters),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MeasurementResult":
        """Reconstruct from a JSON-loaded dict."""
        def np_array(v):
            return np.array(v, dtype=float) if v is not None else None

        specs = {}
        for k, sd in (d.get("field_specs") or {}).items():
            specs[k] = FieldSpec(
                name=sd.get("name", k),
                kind=sd.get("kind", "object"),
                description=sd.get("description", ""),
                length_invariant=sd.get("length_invariant", False),
                units=sd.get("units"),
                deterministic=sd.get("deterministic", True),
                semantic_note=sd.get("semantic_note"),
            )

        pt = {k: np.array([x if x is not None else np.nan for x in v], dtype=float)
              for k, v in (d.get("per_token") or {}).items()}

        plpt = {}
        for name, by_layer in (d.get("per_layer_per_token") or {}).items():
            plpt[name] = {
                int(lk): np.array(
                    [x if x is not None else np.nan for x in la], dtype=float)
                for lk, la in by_layer.items()
            }

        pl = {}
        for name, by_layer in (d.get("per_layer") or {}).items():
            pl[name] = {int(lk): (float(v) if v is not None else float("nan"))
                        for lk, v in by_layer.items()}

        return cls(
            measurement_name=d["measurement_name"],
            measurement_version=d.get("measurement_version", "0.0.0"),
            scalars=d.get("scalars") or {},
            per_token=pt,
            per_layer=pl,
            per_layer_per_token=plpt,
            objects=d.get("objects") or {},
            field_specs=specs,
            parameters=d.get("parameters") or {},
        )


# ── Helpers for constructing per-token arrays that obey the contract ──

def padded_per_token(seq_len: int, fill: float = float("nan")) -> np.ndarray:
    """Return a length-seq_len float array initialized to `fill` (default NaN).

    Measurements build per-token outputs by allocating a padded array,
    writing valid positions, and returning the array. NaN sentinel values
    at positions the measurement doesn't define keep the per-token
    alignment contract loud rather than silent.
    """
    return np.full(seq_len, fill, dtype=float)


def place_per_token(arr: np.ndarray, start: int, values: np.ndarray) -> None:
    """Write `values` into `arr` starting at index `start`.

    Bounds-checked: raises ValueError if the write would exceed arr's length.
    """
    end = start + len(values)
    if end > len(arr):
        raise ValueError(
            f"place_per_token: write of length {len(values)} at start={start} "
            f"exceeds array length {len(arr)}")
    arr[start:end] = values
