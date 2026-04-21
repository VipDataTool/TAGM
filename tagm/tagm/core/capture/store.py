"""ActivationStore: canonical per-prompt container for captured activations.

Addressed as `store.get(layer_idx, hook_point, capture_type) -> tensor`.
Populated by hooks during the forward pass; read by measurement modules
after the pass completes. Per-prompt lifetime; caller explicitly clears
between prompts (the Pipeline does this by constructing a fresh store
per run).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from tagm.core.adapter.base import ModelAdapter
    from tagm.core.capture.config import CaptureConfig


# Address triple: (layer_or_None, hook_point_name, capture_type)
_AddrKey = tuple[Optional[int], str, str]


class ActivationStore:
    """Store activations captured during one forward pass.

    The schema is family-agnostic: (layer_idx, hook_point_name, capture_type)
    regardless of the model family. layer_idx of None addresses
    layer-independent hook points (e.g. final_norm).

    The store carries its originating CaptureConfig and adapter so consumers
    can introspect what was requested vs what was captured without
    out-of-band context.
    """

    def __init__(self, capture_config: "CaptureConfig", adapter: "ModelAdapter"):
        self._capture_config = capture_config
        self._adapter = adapter
        self._data: dict[_AddrKey, torch.Tensor] = {}

    # ── Writing ─────────────────────────────────────────────────────
    def put(self, layer: Optional[int], hook_point: str, capture_type: str,
            tensor: torch.Tensor) -> None:
        """Store a captured tensor under the given address.

        Overwrites any existing entry at the same address. Called by the
        hook installer's forward hooks; callers outside the instrument
        layer should not need to write to the store.
        """
        self._data[(layer, hook_point, capture_type)] = tensor

    # ── Reading ─────────────────────────────────────────────────────
    def get(self, layer: Optional[int], hook_point: str, capture_type: str
            ) -> torch.Tensor:
        """Retrieve a captured tensor.

        Raises KeyError with a descriptive message (including available
        addresses) if the requested tensor is not present — this is a
        common debugging moment, so the error needs to be actionable.
        """
        key = (layer, hook_point, capture_type)
        if key not in self._data:
            raise KeyError(
                f"No capture at layer={layer}, hook_point='{hook_point}', "
                f"capture_type='{capture_type}'. Available: "
                f"{[self._fmt_key(k) for k in self.keys()]}"
            )
        return self._data[key]

    def has(self, layer: Optional[int], hook_point: str, capture_type: str
            ) -> bool:
        return (layer, hook_point, capture_type) in self._data

    def get_or_none(self, layer: Optional[int], hook_point: str,
                     capture_type: str) -> Optional[torch.Tensor]:
        """Non-raising variant of get(); returns None if not present."""
        return self._data.get((layer, hook_point, capture_type))

    # ── Bulk access ─────────────────────────────────────────────────
    def layers_for(self, hook_point: str, capture_type: str) -> list[int]:
        """Return the sorted list of layer indices present at a given
        (hook_point, capture_type). Skips None (layer-independent) entries.
        """
        return sorted(
            k[0] for k in self._data
            if k[0] is not None and k[1] == hook_point and k[2] == capture_type
        )

    def keys(self) -> list[_AddrKey]:
        """All address triples present in the store, sorted."""
        return sorted(
            self._data.keys(),
            key=lambda k: (k[0] if k[0] is not None else -1, k[1], k[2]),
        )

    # ── Diagnostics ─────────────────────────────────────────────────
    def coverage(self) -> dict:
        """Structured summary: {hook_point: {capture_type: [layer_idx, ...]}}.

        Useful for debugging "my measurement said it needed X but there's
        nothing in the store at X" situations.
        """
        out: dict[str, dict[str, list]] = {}
        for layer, hp, ct in self.keys():
            out.setdefault(hp, {}).setdefault(ct, []).append(layer)
        return out

    def nbytes(self) -> int:
        """Approximate total bytes held in captured tensors.

        Useful for memory diagnostics after a large capture. Uses each
        tensor's `element_size() * numel()`; ignores Python-object overhead.
        """
        return sum(t.element_size() * t.numel() for t in self._data.values())

    # ── Lifecycle ───────────────────────────────────────────────────
    def clear(self) -> None:
        """Drop all captured tensors. Cheap; does not free underlying memory
        until Python garbage-collects the tensors themselves."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: _AddrKey) -> bool:
        return key in self._data

    # ── Introspection ───────────────────────────────────────────────
    @property
    def capture_config(self) -> "CaptureConfig":
        return self._capture_config

    @property
    def adapter(self) -> "ModelAdapter":
        return self._adapter

    @staticmethod
    def _fmt_key(key: _AddrKey) -> str:
        layer, hp, ct = key
        if layer is None:
            return f"{hp}/{ct}"
        return f"layer{layer}/{hp}/{ct}"
