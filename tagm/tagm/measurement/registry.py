"""Measurement module registry.

Simple dict keyed by the module's `name` attribute. Populated by
`register_measurement()` calls, typically at import time in
`tagm/measurement/modules/__init__.py`.
"""
from __future__ import annotations

from tagm.measurement.base import MeasurementModule


_MEASUREMENTS: dict[str, type[MeasurementModule]] = {}


def register_measurement(cls: type[MeasurementModule]) -> type[MeasurementModule]:
    """Register a MeasurementModule subclass. Usable as a decorator.

    Raises ValueError if another measurement with the same name is
    already registered — name collisions are silent correctness hazards
    (two modules writing to the same key in the session record).
    """
    if not cls.name:
        raise ValueError(
            f"{cls.__name__}.name must be non-empty to register")
    if cls.name in _MEASUREMENTS and _MEASUREMENTS[cls.name] is not cls:
        raise ValueError(
            f"Measurement name '{cls.name}' already registered by "
            f"{_MEASUREMENTS[cls.name].__name__}")
    _MEASUREMENTS[cls.name] = cls
    return cls


def find_measurement(name: str) -> type[MeasurementModule]:
    """Look up a measurement class by name. Raises KeyError if not found."""
    if name not in _MEASUREMENTS:
        raise KeyError(
            f"No measurement registered with name '{name}'. "
            f"Registered: {sorted(_MEASUREMENTS)}")
    return _MEASUREMENTS[name]


def list_measurements() -> list[dict]:
    """Return metadata for all registered measurements (for UI population)."""
    return [cls().metadata() for cls in _MEASUREMENTS.values()]


def clear_registry() -> None:
    """Clear the registry. Used by tests."""
    _MEASUREMENTS.clear()
