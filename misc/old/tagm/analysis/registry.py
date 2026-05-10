"""Analysis module registry. Same structure as measurement registry."""
from __future__ import annotations

from misc.old.tagm.analysis.base import AnalysisModule


_ANALYSES: dict[str, type[AnalysisModule]] = {}


def register_analysis(cls: type[AnalysisModule]) -> type[AnalysisModule]:
    """Register an AnalysisModule subclass. Usable as a decorator."""
    if not cls.name:
        raise ValueError(f"{cls.__name__}.name must be non-empty to register")
    if cls.name in _ANALYSES and _ANALYSES[cls.name] is not cls:
        raise ValueError(
            f"Analysis name '{cls.name}' already registered by "
            f"{_ANALYSES[cls.name].__name__}")
    _ANALYSES[cls.name] = cls
    return cls


def find_analysis(name: str) -> type[AnalysisModule]:
    if name not in _ANALYSES:
        raise KeyError(
            f"No analysis registered with name '{name}'. "
            f"Registered: {sorted(_ANALYSES)}")
    return _ANALYSES[name]


def list_analyses() -> list[dict]:
    return [cls().metadata() for cls in _ANALYSES.values()]


def clear_registry() -> None:
    _ANALYSES.clear()
