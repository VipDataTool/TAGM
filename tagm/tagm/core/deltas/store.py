"""DeltaStore: canonical weight-delta container for a loaded model pair.

Addressed as `store.get(layer_idx, role) -> tensor`. Role is one of
"q" | "k" | "v" | "o" | "gate" | "up" | "down" (subset depends on the
adapter family, but Qwen2/Llama3 both expose all seven).

Persists for the model-pair's lifetime in the Pipeline. A DeltaStore
records the `layer_filter` it was built under; a consumer asking for
a layer outside the filter gets a descriptive `LayerNotComputedError`
rather than a bare KeyError — so "I need layer 4 and it wasn't loaded"
is distinguishable from "layer 4 doesn't exist on this model."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from tagm.core.adapter.base import ModelAdapter


class LayerNotComputedError(KeyError):
    """Raised when a consumer asks for a layer that exists on the model
    but was excluded by this DeltaStore's `layer_filter`.

    Distinct from a plain KeyError so callers can surface it as
    "re-load with a wider layer filter" rather than "this role/layer
    doesn't exist on this family."
    """
    def __init__(self, layer: int, role: str, filter_: Optional[list[int]],
                 n_layers: int):
        self.layer = layer
        self.role = role
        self.layer_filter = filter_
        self.n_layers = n_layers
        msg = (f"Delta for layer={layer}, role='{role}' was not computed. "
               f"Model has {n_layers} layers; ")
        if filter_ is None:
            msg += "DeltaStore was built over all layers, so this role "\
                   "may not be declared by the adapter."
        else:
            msg += (f"DeltaStore was built with layer_filter={sorted(filter_)}. "
                    f"Reload with a filter that includes layer {layer}.")
        super().__init__(msg)


@dataclass
class DeltaSpectralSummary:
    """Per-delta spectral structure, computed once at load time.

    Stored on the DeltaStore alongside each delta tensor. Consumed by
    SFD (which builds its own QK cache on top) and by the frontend's
    spectral-summary display. All fields except singular_values are
    scalar summaries cheap to serialize.
    """
    eff_rank: float              # exp(Shannon entropy of normalized singular values)
    top1_share: float            # fraction of spectral energy in the top singular value
    top5_share: float            # fraction of spectral energy in the top 5
    stable_rank: float           # ||delta||_F^2 / sigma_1^2
    frob_norm: float             # Frobenius norm of the delta
    singular_values: Optional[torch.Tensor] = None
    # Full top-k singular values (as returned by torch.svd_lowrank with q=k).
    # Kept optionally so higher-k measurements can reuse; may be None to
    # save memory if only scalars were requested.


@dataclass
class DeltaStoreMetadata:
    """Record of how this DeltaStore was built, carried for introspection."""
    base_model_id: str
    instruct_model_id: str
    adapter_family: str
    dtype: str
    layer_filter: Optional[list[int]]
    n_layers: int
    n_deltas: int


class DeltaStore:
    """Canonical storage for weight deltas.

    Address: store.get(layer_idx, role) -> tensor

    Internally holds a dict keyed by (layer_idx, role). Also holds
    per-delta Frobenius norms (needed as normalizers by most measurements
    that use deltas) and per-delta spectral summaries (computed once at
    load time, read many times).
    """

    def __init__(self, adapter: "ModelAdapter", metadata: DeltaStoreMetadata):
        self._adapter = adapter
        self._metadata = metadata
        self._data: dict[tuple[int, str], torch.Tensor] = {}
        self._frob_norms: dict[tuple[int, str], float] = {}
        self._spectral: dict[tuple[int, str], DeltaSpectralSummary] = {}

    # ── Writing (used by compute.py) ────────────────────────────────
    def put(self, layer_idx: int, role: str, tensor: torch.Tensor,
            frob_norm: Optional[float] = None) -> None:
        """Store a delta tensor and (optionally pre-computed) Frobenius norm.

        If frob_norm is None it is computed from the tensor. Adapter
        role validation is delegated to compute.py which uses the
        adapter to derive roles; the store itself doesn't re-validate.
        """
        self._data[(layer_idx, role)] = tensor
        self._frob_norms[(layer_idx, role)] = (
            float(frob_norm) if frob_norm is not None
            else float(tensor.norm().item())
        )

    def put_spectral(self, layer_idx: int, role: str,
                      summary: DeltaSpectralSummary) -> None:
        self._spectral[(layer_idx, role)] = summary

    # ── Reading ─────────────────────────────────────────────────────
    def get(self, layer_idx: int, role: str) -> torch.Tensor:
        """Retrieve a delta tensor. Raises LayerNotComputedError if the
        requested (layer, role) was excluded by this store's filter."""
        key = (layer_idx, role)
        if key not in self._data:
            if role not in self._adapter.PROJECTION_ROLES:
                raise KeyError(
                    f"Role '{role}' not declared by adapter "
                    f"'{self._adapter.family_id}'. Available: "
                    f"{list(self._adapter.PROJECTION_ROLES)}")
            raise LayerNotComputedError(
                layer=layer_idx, role=role,
                filter_=self._metadata.layer_filter,
                n_layers=self._metadata.n_layers,
            )
        return self._data[key]

    def get_or_none(self, layer_idx: int, role: str) -> Optional[torch.Tensor]:
        """Non-raising variant. Returns None for any missing delta."""
        return self._data.get((layer_idx, role))

    def has(self, layer_idx: int, role: str) -> bool:
        return (layer_idx, role) in self._data

    def frob_norm(self, layer_idx: int, role: str) -> float:
        """Frobenius norm of delta at (layer, role). Raises like get()."""
        key = (layer_idx, role)
        if key not in self._frob_norms:
            # Force an error via get() for consistent messaging
            self.get(layer_idx, role)
        return self._frob_norms[key]

    def spectral(self, layer_idx: int, role: str
                  ) -> Optional[DeltaSpectralSummary]:
        return self._spectral.get((layer_idx, role))

    # ── Role/layer iteration ────────────────────────────────────────
    def layers(self) -> list[int]:
        """Sorted unique layer indices present in the store."""
        return sorted({layer for layer, _ in self._data})

    def roles_at(self, layer_idx: int) -> list[str]:
        """Roles present at a given layer."""
        return sorted(role for layer, role in self._data if layer == layer_idx)

    def items(self):
        """Iterate over ((layer, role), tensor) pairs. For bulk readers."""
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: tuple[int, str]) -> bool:
        return key in self._data

    # ── Adapter convenience (named accessors) ───────────────────────
    def v_delta(self, layer_idx: int) -> torch.Tensor:
        """Shortcut for store.get(layer_idx, 'v')."""
        return self.get(layer_idx, "v")

    def o_delta(self, layer_idx: int) -> torch.Tensor:
        """Shortcut for store.get(layer_idx, 'o')."""
        return self.get(layer_idx, "o")

    def v_delta_or_none(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.get_or_none(layer_idx, "v")

    def o_delta_or_none(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.get_or_none(layer_idx, "o")

    # ── Metadata ────────────────────────────────────────────────────
    @property
    def metadata(self) -> DeltaStoreMetadata:
        return self._metadata

    @property
    def adapter(self) -> "ModelAdapter":
        return self._adapter

    @property
    def layer_filter(self) -> Optional[list[int]]:
        return self._metadata.layer_filter

    @property
    def full_deltas_available(self) -> bool:
        """True if this store was built over all layers (no filter)."""
        return self._metadata.layer_filter is None

    # ── Aggregate summaries ─────────────────────────────────────────
    def aggregate_spectral_summary(self) -> dict:
        """Average spectral properties across all held deltas.

        Used by the frontend's model-info panel and by exports for
        reproducibility. Returns zeros-dict if no spectral data is
        present (e.g. spectral computation was skipped).
        """
        if not self._spectral:
            return {
                "mean_eff_rank": 0.0,
                "std_eff_rank": 0.0,
                "mean_top1_share": 0.0,
                "attn_mean_rank": 0.0,
                "mlp_mean_rank": 0.0,
                "n_sublayers": 0,
            }

        import statistics

        eff_ranks = [s.eff_rank for s in self._spectral.values()]
        top1 = [s.top1_share for s in self._spectral.values()]
        attn_roles = {"q", "k", "v", "o"}
        attn_ranks = [s.eff_rank for (l, r), s in self._spectral.items()
                      if r in attn_roles]
        mlp_ranks = [s.eff_rank for (l, r), s in self._spectral.items()
                     if r not in attn_roles]

        def _mean(xs): return float(statistics.mean(xs)) if xs else 0.0
        def _std(xs): return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0

        return {
            "mean_eff_rank": _mean(eff_ranks),
            "std_eff_rank": _std(eff_ranks),
            "mean_top1_share": _mean(top1),
            "attn_mean_rank": _mean(attn_ranks),
            "mlp_mean_rank": _mean(mlp_ranks),
            "n_sublayers": len(eff_ranks),
        }

    def total_bytes(self) -> int:
        """Approximate bytes held in delta tensors. For memory diagnostics."""
        return sum(t.element_size() * t.numel() for t in self._data.values())
