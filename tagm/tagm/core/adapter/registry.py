"""Adapter registry: discovery, registration, auto-detection.

Simple ordered list of registered adapter classes. `find_adapter` walks
the list in registration order and returns an instance of the first one
whose `matches()` classmethod returns True for the given model.

Registration order matters only when multiple adapters could match the same
model (which should not happen in practice — family detection via `model_type`
is unambiguous). Newly registered adapters go to the end of the list.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from tagm.core.adapter.base import ModelAdapter
from tagm.core.adapter.qwen2 import Qwen2Adapter
from tagm.core.adapter.llama3 import Llama3Adapter

if TYPE_CHECKING:
    from transformers import PreTrainedModel


_ADAPTERS: list[type[ModelAdapter]] = [
    Qwen2Adapter,
    Llama3Adapter,
]


def register_adapter(adapter_cls: type[ModelAdapter]) -> None:
    """Register a new adapter class for auto-detection.

    Idempotent: registering the same class twice is a no-op. No check
    that family_id is unique — if two registered adapters declare the
    same family_id, that's a caller bug surfaced at introspection time.
    """
    if adapter_cls not in _ADAPTERS:
        _ADAPTERS.append(adapter_cls)


def find_adapter(model: "PreTrainedModel") -> ModelAdapter:
    """Return an adapter instance for the given loaded model.

    Raises ValueError if no registered adapter matches. The error message
    includes the model's declared model_type and the list of registered
    families, so the caller can either add an adapter or correct their
    model selection.
    """
    for cls in _ADAPTERS:
        if cls.matches(model):
            return cls()
    model_type = getattr(model.config, "model_type", "unknown")
    registered = [a.family_id for a in _ADAPTERS]
    raise ValueError(
        f"No registered TAGM adapter matches model type '{model_type}'. "
        f"Registered families: {registered}. "
        f"To add a new family, implement a ModelAdapter subclass and call "
        f"tagm.core.adapter.register_adapter(YourAdapter)."
    )


def list_families() -> list[dict]:
    """Return metadata for all registered adapter families (for UI population).

    Includes per-family hook point declarations so the UI can render a
    capture-config form against the adapter's actual capabilities rather
    than hardcoding hook points client-side.
    """
    out = []
    for cls in _ADAPTERS:
        # cls.HOOK_POINTS is a class attribute, accessible without instantiating
        hook_points = []
        for name, spec in (cls.HOOK_POINTS or {}).items():
            hook_points.append({
                "name": spec.name,
                "description": spec.description,
                "captures": sorted(spec.captures),
                "layer_independent": spec.layer_independent,
            })
        out.append({
            "family_id": cls.family_id,
            "display_name": cls.family_display_name,
            "hook_points": hook_points,
        })
    return out
