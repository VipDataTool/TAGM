"""Shared type definitions and protocols for the TAGM instrument layer.

Small module so that mutual imports between adapter, capture, and pipeline
layers can reference these types without creating cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


# ── Validation result (shared between CaptureConfig, DeltaStore, etc.) ──

@dataclass
class ValidationResult:
    """Generic validator output: ok flag + structured errors and warnings."""
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def extend(self, other: "ValidationResult") -> "ValidationResult":
        return ValidationResult(
            ok=self.ok and other.ok,
            errors=list(self.errors) + list(other.errors),
            warnings=list(self.warnings) + list(other.warnings),
        )


# ── Progress callback protocol ──────────────────────────────────────

class ProgressCallback(Protocol):
    """Callable signature for stage/message progress reporting.

    stage:   short identifier (e.g. "loading", "deltas", "spectral", "ready")
    message: human-readable progress string
    """
    def __call__(self, stage: str, message: str) -> None: ...


def noop_progress(stage: str, message: str) -> None:
    """No-op progress callback; safe default when caller doesn't want logs."""
    return None
