"""Model family adapters.

Each adapter describes one model family's structural contract (hook points,
projection roles, attribute paths, tokenizer policy, memory profile). One
adapter per family; many specific model pairs within a family share it.

Adding a new family is one new adapter class, registered in `registry.py`.
Nothing else in TAGM changes.
"""
from tagm.core.adapter.base import (
    ModelAdapter,
    HookPointSpec,
    ProjectionRole,
    MemoryProfile,
)
from tagm.core.adapter.registry import find_adapter, register_adapter, list_families

__all__ = [
    "ModelAdapter",
    "HookPointSpec",
    "ProjectionRole",
    "MemoryProfile",
    "find_adapter",
    "register_adapter",
    "list_families",
]
