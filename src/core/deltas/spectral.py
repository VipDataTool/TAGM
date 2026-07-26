"""Spectral precomputes for weight deltas.

Computes effective rank, top-k energy shares, stable rank, and (optionally)
the top-k singular values for each delta in a DeltaStore. Called once by
the Pipeline after delta computation; results live on the DeltaStore for
downstream measurements (SFD in particular) to read without per-prompt SVD.

Effective rank is `exp(Shannon entropy of normalized singular values)` —
a smooth measure of "how many directions does this delta use." A low
eff_rank means RLHF made a surgical correction; a high eff_rank means
it reshaped the whole subspace.

Translated from TASM's `engine/model_manager.py::_compute_spectral_profile`;
uses numpy for the entropy/share math to match TASM's outputs bit-for-bit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from src.core.deltas.store import DeltaSpectralSummary
from src.core.types import ProgressCallback, noop_progress

if TYPE_CHECKING:
    from src.core.deltas.store import DeltaStore


def compute_spectral_profile(
    store: "DeltaStore",
    svd_k: int = 64,
    keep_singular_values: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> None:
    """Compute spectral summary for every delta in `store`, in place.

    Args:
      store:                 DeltaStore to annotate. Mutated in place via
                             store.put_spectral().
      svd_k:                 Number of singular values to compute per delta.
                             Truncated SVD via torch.svd_lowrank.
      keep_singular_values:  If True, store the top-k singular values on
                             the summary for downstream reuse (SFD). If False,
                             only scalar summaries are kept (lower memory).
      progress:              Optional callback for progress reporting.

    The math is computed in float32 for numerical stability regardless of the
    delta's storage dtype. A random seed is fixed within the function to make
    torch.svd_lowrank deterministic across runs.
    """
    log = progress or noop_progress

    n = len(store)
    if n == 0:
        log("spectral", "No deltas in store; skipping spectral profile")
        return

    # Deterministic low-rank SVD.
    #
    # fork_rng, not a bare torch.manual_seed(42): the latter reseeds the
    # PROCESS-wide generator at model-load time, so every later sampling
    # operation (chat generation, module random baselines) silently inherited
    # a fixed seed as a side effect of profiling the deltas.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        _profile_all(store, svd_k, keep_singular_values, log, n)


def _profile_all(store, svd_k, keep_singular_values, log, n):
    """Body of compute_spectral_profile; split out only so the caller can
    wrap it in fork_rng."""
    for i, ((layer_idx, role), delta) in enumerate(store.items()):
        try:
            d = delta.float().cpu()
            k = min(svd_k, min(d.shape))
            # Oversample by 8 and slice back to k.  With q == k and the
            # default niter=2, randomized SVD systematically underestimates
            # the leading singular values for slowly-decaying spectra, which
            # biases eff_rank and both energy shares.
            q_over = min(k + 8, min(d.shape))
            U, S, Vh = torch.svd_lowrank(d, q=q_over, niter=2)
            S = S[:k]
            s = S.float().numpy().astype(np.float64)

            # Effective rank: exp(Shannon entropy of normalized s-distribution).
            # Computed over the top-k spectrum — a documented approximation.
            s_norm = s / (s.sum() + 1e-10)
            s_nonzero = s_norm[s_norm > 1e-10]
            if len(s_nonzero) > 0:
                ent = -np.sum(s_nonzero * np.log(s_nonzero))
                eff_rank = float(np.exp(ent))
            else:
                eff_rank = 1.0

            # Exact Frobenius norm from the full tensor — NOT the truncated
            # spectrum. Energy shares and stable rank are fractions of the
            # true total energy, so they no longer overstate concentration
            # for deltas whose rank exceeds svd_k.
            frob = float(torch.linalg.vector_norm(d).item())
            total_energy = frob * frob
            if total_energy > 0:
                top1_share = float(s[0] ** 2) / total_energy
                top5_share = float((s[:5] ** 2).sum()) / total_energy
            else:
                top1_share = 0.0
                top5_share = 0.0

            # Stable rank: ||A||_F^2 / sigma_1^2 (exact numerator)
            stable = total_energy / (s[0] ** 2) if s[0] > 0 else 1.0

            summary = DeltaSpectralSummary(
                eff_rank=round(eff_rank, 8),
                top1_share=round(top1_share, 8),
                top5_share=round(top5_share, 8),
                stable_rank=round(stable, 8),
                frob_norm=frob,
                singular_values=(torch.from_numpy(s.astype(np.float32))
                                 if keep_singular_values else None),
            )
            store.put_spectral(layer_idx, role, summary)

            if (i + 1) % 20 == 0 or (i + 1) == n:
                log("spectral", f"Spectral profile: {i + 1}/{n} deltas")

        except Exception as e:
            log("spectral", f"Layer {layer_idx} role {role}: SVD failed ({e}); "
                            f"skipping this delta")
            continue

    agg = store.aggregate_spectral_summary()
    log("spectral", f"Spectral profile complete: {agg['n_sublayers']} sublayers, "
                    f"mean eff_rank={agg['mean_eff_rank']:.1f}, "
                    f"attn={agg['attn_mean_rank']:.1f}, "
                    f"mlp={agg['mlp_mean_rank']:.1f}")
