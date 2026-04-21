"""Weight-delta subsystem: compute, storage, spectral analysis.

The DeltaStore holds per-layer, per-role weight deltas (instruct minus base)
for the currently loaded model pair. Populated once at Pipeline.load() time
via `compute_deltas_from_disk`, which reads base-model weights directly from
safetensors files without ever instantiating the full base model — the
memory discipline that lets TAGM run on a free Codespace.

Spectral precomputes (effective rank, top-k shares, singular values) are
computed once at load time and cached on the store; downstream measurements
like SFD and RD read them without triggering any SVD work per prompt.
"""
from tagm.core.deltas.store import DeltaStore, DeltaSpectralSummary
from tagm.core.deltas.compute import compute_deltas_from_disk
from tagm.core.deltas.spectral import compute_spectral_profile

__all__ = [
    "DeltaStore",
    "DeltaSpectralSummary",
    "compute_deltas_from_disk",
    "compute_spectral_profile",
]
