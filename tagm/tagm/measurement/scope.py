"""Shared helpers for measurement modules.

Measurements read opportunistically from the ActivationStore, narrowed
by the user's scope parameter (if any) and what's available in the
delta store. This module provides the common pattern as a single
helper so every measurement resolves scope consistently.
"""
from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tagm.core.capture.store import ActivationStore
    from tagm.core.deltas.store import DeltaStore


def resolve_scope_layers(
    activation_store: "ActivationStore",
    hook_point: str,
    capture_type: str = "hidden",
    scope: Optional[Iterable[int]] = None,
    required_delta_roles: Optional[Iterable[str]] = None,
    delta_store: Optional["DeltaStore"] = None,
) -> list[int]:
    """Resolve the effective layers a measurement will aggregate over.

    The rule is:
      available   = layers present in activation_store at (hook_point, capture_type)
      in_scope    = scope if scope is non-empty, else all available
      with_deltas = in_scope intersected with layers that have all
                    required_delta_roles available in delta_store
                    (if required_delta_roles is given)

    This single entry point replaces ad-hoc `for layer_idx in layers: if
    store.has(...) and delta_store.get_or_none(...)` loops that every
    measurement had to write. It also turns an empty scope into "all
    available" consistently, removing the layers=[] ambiguity.

    Args:
      activation_store:       store to check for captured hidden states.
      hook_point:             hook point name.
      capture_type:           "hidden" | "attention_weights".
      scope:                  user-set scope parameter, or None/empty for
                              "use whatever's available."
      required_delta_roles:   roles the measurement needs deltas for at
                              each layer. Layers missing any required
                              role are filtered out.
      delta_store:            required if required_delta_roles is given.

    Returns:
      Sorted list of layer indices the measurement should operate on.
      Empty list is a legitimate return — the caller should handle it by
      returning a NaN-padded/empty-but-valid MeasurementResult.
    """
    available = set(activation_store.layers_for(hook_point, capture_type))

    if scope:
        scope_set = set(int(x) for x in scope)
        in_scope = available & scope_set
    else:
        in_scope = available

    if required_delta_roles and delta_store is not None:
        delta_filtered = set()
        for layer_idx in in_scope:
            if all(delta_store.has(layer_idx, role)
                   for role in required_delta_roles):
                delta_filtered.add(layer_idx)
        return sorted(delta_filtered)

    return sorted(in_scope)


def describe_scope_resolution(
    requested_scope: Optional[Iterable[int]],
    resolved_layers: list[int],
    hook_point: str,
) -> str:
    """Human-readable diagnostic for what a scope resolution resolved to.

    Recorded in MeasurementResult.parameters so users can see "you asked
    for layers [5,6,7] at pre_attn_norm; capture only had [5,6]; deltas
    only available for [5]; used [5]."
    """
    req = (sorted(int(x) for x in requested_scope)
           if requested_scope else "all available")
    return (f"requested={req}, hook_point={hook_point}, "
            f"resolved={resolved_layers}")
