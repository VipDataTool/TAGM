"""Hook installer: translate a CaptureConfig into registered torch hooks.

Given an adapter, a loaded model, a CaptureConfig, and an empty
ActivationStore, `install_hooks` registers forward hooks that populate
the store during the next forward pass. Returns a list of handles for
the caller to remove once capture is complete.

Design note: one hook is installed per (layer, hook_point) pair, even
if multiple capture types are requested at that location. The hook
closure internally decides which capture types to extract from the
output. This keeps hook count proportional to distinct capture locations,
not to requirements.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from torch.nn import Module
    from tagm.core.adapter.base import ModelAdapter
    from tagm.core.capture.config import CaptureConfig, CapturePoint
    from tagm.core.capture.store import ActivationStore


# ── Public API ──────────────────────────────────────────────────────

def install_hooks(model, adapter: "ModelAdapter",
                   capture_config: "CaptureConfig",
                   store: "ActivationStore") -> list:
    """Install all hooks required by `capture_config` and return their handles.

    Caller is responsible for calling `remove_hooks(handles)` after the
    forward pass completes, typically in a try/finally around the forward
    call (the Pipeline does this).

    Raises RuntimeError if any hook point cannot be resolved on the model.
    """
    handles: list = []

    # Group capture points by (layer, hook_point) so we install one hook
    # per module regardless of how many capture types were requested.
    grouped: dict[tuple[Optional[int], str], list["CapturePoint"]] = {}
    for p in capture_config.points:
        grouped.setdefault((p.layer, p.hook_point), []).append(p)

    for (layer, hook_point), points in grouped.items():
        try:
            target = adapter.resolve_hook_target(model, hook_point, layer)
        except (KeyError, IndexError, AttributeError, ValueError) as e:
            # Roll back partial installation
            remove_hooks(handles)
            raise RuntimeError(
                f"Failed to resolve hook target for layer={layer}, "
                f"hook_point='{hook_point}' on adapter "
                f"'{adapter.family_id}': {type(e).__name__}: {e}"
            ) from e

        capture_types: set[str] = set()
        for p in points:
            capture_types.update(p.capture)

        # Reduction and precision: use the first point at this location.
        # If multiple points at the same location specify different
        # reductions/precisions, that's a config-union error; we take
        # the primary silently (no good answer for competing requirements).
        primary = points[0]

        hook_fn = _make_hook(
            layer=layer,
            hook_point=hook_point,
            capture_types=frozenset(capture_types),
            reduction=primary.reduction,
            precision=primary.precision,
            store=store,
        )
        handles.append(target.register_forward_hook(hook_fn))

    return handles


def remove_hooks(handles: list) -> None:
    """Remove all hook handles in the list. Idempotent."""
    for h in handles:
        try:
            h.remove()
        except Exception:
            # Best-effort; a handle may already be removed if the model
            # was rebuilt mid-run.
            pass
    handles.clear()


# ── Internals ───────────────────────────────────────────────────────

def _make_hook(layer, hook_point, capture_types, reduction, precision, store):
    """Build a forward hook closure that writes to the given store.

    The hook understands two output shapes:
      - Tensor: treated as hidden-state output (shape [..., hidden]).
      - Tuple: first element treated as hidden state; second element,
               if present and non-None, treated as attention weights
               (shape [batch, n_heads, seq_len, seq_len]).
    """
    want_hidden = "hidden" in capture_types
    want_attn = "attention_weights" in capture_types

    def hook(module, inputs, output):
        # Normalize output into (hidden_tensor, attn_tensor_or_None)
        if isinstance(output, tuple):
            hidden = output[0] if len(output) > 0 else None
            attn = output[1] if len(output) > 1 else None
        else:
            hidden = output
            attn = None

        if want_hidden and hidden is not None:
            t = hidden.detach()
            t = _apply_reduction(t, reduction)
            t = _apply_precision(t, precision)
            store.put(layer, hook_point, "hidden", t)

        if want_attn and attn is not None:
            # Attention weights always stored full; reduction doesn't apply
            # to a 4D tensor in a meaningful way, and precision casting
            # attention weights below float16 loses information.
            store.put(layer, hook_point, "attention_weights",
                      _apply_precision(attn.detach(), precision))

    return hook


def _apply_reduction(tensor: torch.Tensor, reduction: Optional[str]) -> torch.Tensor:
    """Reduce across the sequence dimension.

    Assumes the sequence dimension is axis 1 (standard HF output shape
    [batch, seq_len, hidden]). Returns the tensor unchanged if reduction
    is None or the tensor doesn't have a sequence axis.
    """
    if reduction is None or tensor.dim() < 2:
        return tensor
    if reduction == "mean":
        return tensor.mean(dim=1, keepdim=True)
    if reduction == "last":
        return tensor[:, -1:, ...]
    if reduction == "first":
        return tensor[:, :1, ...]
    return tensor  # defensive: validated upstream, but don't crash


def _apply_precision(tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """Cast to the requested storage precision.

    "model_dtype" returns the tensor unchanged. Other values force-cast
    to the corresponding torch dtype. Casting is done with .to(); no
    quantization beyond native dtype conversion.
    """
    if precision == "model_dtype":
        return tensor
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target = dtype_map.get(precision)
    if target is None:
        return tensor
    if tensor.dtype == target:
        return tensor
    return tensor.to(target)
