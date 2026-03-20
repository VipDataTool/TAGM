"""
Spectral Field Density (SFD): per-token subspace engagement measurement.

Measures how many dimensions of the QK routing subspace each token's
activation engages, providing the 'dimensionality' axis alongside
ASM's 'intensity' and LTP's 'selectivity'.

Architecture:
  Precompute (model load): SVD of concatenated [ΔW_Q; ΔW_K] per layer.
    Cache V_k (right singular vectors) and S (singular values).
    Compute global measures: effective rank, spectral entropy, log volume.
  Per token (forward pass): project h through V_k, weight by S,
    compute energy, spectral entropy, and density ratio.
  Aggregate (post-analysis): mean/max/var/p90 of per-token measures.

The QK subspace is the 'biome' — the routing topology RLHF reshaped.
SFD measures how broadly each token's activation engages that topology.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ═══ Data structures ═══

@dataclass
class SFDLayerCache:
    """Cached SVD factors for one layer's QK delta."""
    V_k: np.ndarray          # (k, d_in) right singular vectors
    S: np.ndarray             # (k,) singular values
    erank: float              # effective rank of the full spectrum
    spectral_entropy: float   # H = -sum p_i log p_i
    norm_entropy: float       # H / log(r), in [0,1]
    log_volume: float         # sum log(s_i) over top k
    stable_rank: float        # ||delta||_F^2 / sigma_1^2
    frob_norm: float          # Frobenius norm of the concatenated delta


@dataclass
class SFDCache:
    """Complete SFD cache for all monitored layers."""
    layers: Dict[int, SFDLayerCache] = field(default_factory=dict)
    k: int = 16               # truncation rank
    n_layers: int = 0
    # Aggregate model-level stats
    mean_erank: float = 0.0
    mean_norm_entropy: float = 0.0


@dataclass
class SFDTokenResult:
    """Per-token SFD measures."""
    energy: float = 0.0           # ||Sigma * c_t||^2
    spectral_entropy: float = 0.0 # H_t of weighted projection
    erank_t: float = 0.0          # exp(H_t)
    density: float = 0.0          # erank_t / erank_global


@dataclass
class SFDResult:
    """Prompt-level SFD result."""
    # Per-token arrays
    per_token_energy: Optional[np.ndarray] = None
    per_token_entropy: Optional[np.ndarray] = None
    per_token_density: Optional[np.ndarray] = None

    # Prompt-level aggregates (12 scalars)
    energy_mean: float = 0.0
    energy_max: float = 0.0
    energy_var: float = 0.0
    energy_p90: float = 0.0

    entropy_mean: float = 0.0
    entropy_max: float = 0.0
    entropy_var: float = 0.0
    entropy_p90: float = 0.0

    density_mean: float = 0.0
    density_max: float = 0.0
    density_var: float = 0.0
    density_p90: float = 0.0

    # Model-level context (for interpretation)
    global_erank: float = 0.0
    n_layers_monitored: int = 0
    k: int = 16

    def to_dict(self):
        """Serialize for JSON transport."""
        return {
            'per_token_energy': self.per_token_energy.tolist() if self.per_token_energy is not None else None,
            'per_token_entropy': self.per_token_entropy.tolist() if self.per_token_entropy is not None else None,
            'per_token_density': self.per_token_density.tolist() if self.per_token_density is not None else None,
            'energy_mean': round(self.energy_mean, 6),
            'energy_max': round(self.energy_max, 6),
            'energy_var': round(self.energy_var, 8),
            'energy_p90': round(self.energy_p90, 6),
            'entropy_mean': round(self.entropy_mean, 4),
            'entropy_max': round(self.entropy_max, 4),
            'entropy_var': round(self.entropy_var, 6),
            'entropy_p90': round(self.entropy_p90, 4),
            'density_mean': round(self.density_mean, 4),
            'density_max': round(self.density_max, 4),
            'density_var': round(self.density_var, 6),
            'density_p90': round(self.density_p90, 4),
            'global_erank': round(self.global_erank, 2),
            'n_layers_monitored': self.n_layers_monitored,
            'k': self.k,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstitute from stored dict."""
        if d is None:
            return None
        r = cls()
        for key in ['energy_mean', 'energy_max', 'energy_var', 'energy_p90',
                     'entropy_mean', 'entropy_max', 'entropy_var', 'entropy_p90',
                     'density_mean', 'density_max', 'density_var', 'density_p90',
                     'global_erank', 'n_layers_monitored', 'k']:
            if key in d:
                setattr(r, key, d[key])
        for arr_key in ['per_token_energy', 'per_token_entropy', 'per_token_density']:
            if d.get(arr_key) is not None:
                setattr(r, arr_key, np.array(d[arr_key], dtype=float))
        return r


# ═══ Precomputation (model load) ═══

def precompute_sfd_cache(state, layer_indices: List[int] = None,
                         k: int = 16) -> SFDCache:
    """Compute SVD of concatenated [ΔW_Q; ΔW_K] per layer.

    Args:
        state: ModelState with deltas dict.
        layer_indices: which layers to compute (default: signal layers 9-15).
        k: truncation rank for SVD.

    Returns:
        SFDCache with V_k, S, and global measures per layer.
    """
    if layer_indices is None:
        layer_indices = list(range(min(9, state.n_layers), min(16, state.n_layers)))

    cache = SFDCache(k=k)
    eranks = []

    for layer_idx in layer_indices:
        # Get Q and K deltas for this layer
        q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"

        dw_q = state.deltas.get(q_key)
        dw_k = state.deltas.get(k_key)

        if dw_q is None or dw_k is None:
            logger.warning(f"[SFD] Layer {layer_idx}: missing Q or K delta, skipping")
            continue

        try:
            # Concatenate [ΔW_Q; ΔW_K] along rows (they share d_in columns)
            dw_qk = torch.cat([dw_q.float().cpu(), dw_k.float().cpu()], dim=0)

            # Truncated SVD
            actual_k = min(k, min(dw_qk.shape))
            U, S, Vh = torch.svd_lowrank(dw_qk, q=actual_k)

            s = S.numpy().astype(np.float64)
            v_k = Vh.numpy().astype(np.float32)  # (k, d_in)

            # Global measures from singular value spectrum
            s_pos = s[s > 1e-10]
            if len(s_pos) == 0:
                continue

            # Spectral entropy (magnitude distribution)
            p = s_pos / s_pos.sum()
            H = -np.sum(p * np.log(p))
            erank = float(np.exp(H))
            norm_H = H / np.log(len(s_pos)) if len(s_pos) > 1 else 0.0

            # Log volume
            log_vol = float(np.sum(np.log(s_pos)))

            # Stable rank
            frob2 = float(np.sum(s ** 2))
            stable_r = frob2 / (s[0] ** 2) if s[0] > 0 else 1.0

            # Frobenius norm
            frob_norm = float(np.sqrt(frob2))

            layer_cache = SFDLayerCache(
                V_k=v_k,
                S=s_pos.astype(np.float32),
                erank=erank,
                spectral_entropy=float(H),
                norm_entropy=float(norm_H),
                log_volume=log_vol,
                stable_rank=stable_r,
                frob_norm=frob_norm,
            )
            cache.layers[layer_idx] = layer_cache
            eranks.append(erank)

            logger.info(f"[SFD] Layer {layer_idx}: QK delta "
                        f"{dw_qk.shape[0]}x{dw_qk.shape[1]}, "
                        f"erank={erank:.1f}, H_norm={norm_H:.3f}, "
                        f"stable_rank={stable_r:.1f}")

        except Exception as e:
            logger.error(f"[SFD] Layer {layer_idx} SVD failed: {e}")
            continue

    cache.n_layers = len(cache.layers)
    if eranks:
        cache.mean_erank = float(np.mean(eranks))
        cache.mean_norm_entropy = float(np.mean(
            [lc.norm_entropy for lc in cache.layers.values()]
        ))

    logger.info(f"[SFD] Cache built: {cache.n_layers} layers, k={k}, "
                f"mean erank={cache.mean_erank:.1f}")
    return cache


# ═══ Per-token computation (forward pass) ═══

def compute_sfd_token(h: np.ndarray, layer_cache: SFDLayerCache) -> SFDTokenResult:
    """Compute SFD measures for a single token at a single layer.

    Args:
        h: activation vector (d_in,) as numpy array.
        layer_cache: cached SVD factors for this layer.

    Returns:
        SFDTokenResult with energy, spectral entropy, erank, density.
    """
    V_k = layer_cache.V_k   # (k, d_in)
    S = layer_cache.S        # (k,)

    # Project: c_t = V_k @ h
    c = V_k @ h              # (k,)

    # Weight: w_t = S * c_t (elementwise)
    w = S[:len(c)] * c       # (k,)

    # Energy: ||w_t||^2
    w2 = w * w
    energy = float(w2.sum())

    if energy < 1e-20:
        return SFDTokenResult(energy=energy)

    # Spectral entropy of the weighted projection
    q = w2 / energy          # normalized energy distribution
    q_pos = q[q > 1e-10]
    H_t = float(-np.sum(q_pos * np.log(q_pos)))
    erank_t = float(np.exp(H_t))

    # Density ratio
    density = erank_t / layer_cache.erank if layer_cache.erank > 0 else 0.0

    return SFDTokenResult(
        energy=energy,
        spectral_entropy=H_t,
        erank_t=erank_t,
        density=density,
    )


def compute_sfd_sequence(activations: np.ndarray, cache: SFDCache) -> SFDResult:
    """Compute SFD for a full token sequence across all cached layers.

    Args:
        activations: dict mapping layer_idx -> (n_tokens, d_in) numpy array,
                     OR a single (n_tokens, d_in) array (averaged across layers).
        cache: precomputed SFDCache.

    Returns:
        SFDResult with per-token and prompt-level measures.
    """
    if cache.n_layers == 0:
        return SFDResult()

    # If activations is a dict of per-layer arrays, process each layer
    # and average across layers. If it's a single array, use all layers.
    if isinstance(activations, dict):
        layer_acts = activations
    else:
        # Use the same activation for all layers (pre-layernorm capture)
        layer_acts = {li: activations for li in cache.layers}

    n_tokens = None
    accum_energy = None
    accum_entropy = None
    accum_density = None
    n_layers_used = 0

    for layer_idx, layer_cache in cache.layers.items():
        acts = layer_acts.get(layer_idx)
        if acts is None:
            continue

        if len(acts.shape) == 1:
            acts = acts.reshape(1, -1)

        if n_tokens is None:
            n_tokens = acts.shape[0]
            accum_energy = np.zeros(n_tokens)
            accum_entropy = np.zeros(n_tokens)
            accum_density = np.zeros(n_tokens)

        for t in range(min(n_tokens, acts.shape[0])):
            tr = compute_sfd_token(acts[t], layer_cache)
            accum_energy[t] += tr.energy
            accum_entropy[t] += tr.spectral_entropy
            accum_density[t] += tr.density

        n_layers_used += 1

    if n_layers_used == 0 or n_tokens is None:
        return SFDResult()

    # Average across layers
    per_energy = accum_energy / n_layers_used
    per_entropy = accum_entropy / n_layers_used
    per_density = accum_density / n_layers_used

    result = SFDResult(
        per_token_energy=per_energy,
        per_token_entropy=per_entropy,
        per_token_density=per_density,
        global_erank=cache.mean_erank,
        n_layers_monitored=n_layers_used,
        k=cache.k,
    )

    # Prompt-level aggregates
    if n_tokens > 0:
        result.energy_mean = float(np.mean(per_energy))
        result.energy_max = float(np.max(per_energy))
        result.energy_var = float(np.var(per_energy))
        result.energy_p90 = float(np.percentile(per_energy, 90))

        result.entropy_mean = float(np.mean(per_entropy))
        result.entropy_max = float(np.max(per_entropy))
        result.entropy_var = float(np.var(per_entropy))
        result.entropy_p90 = float(np.percentile(per_entropy, 90))

        result.density_mean = float(np.mean(per_density))
        result.density_max = float(np.max(per_density))
        result.density_var = float(np.var(per_density))
        result.density_p90 = float(np.percentile(per_density, 90))

    return result


# ═══ Rank displacement (LTP companion measure) ═══

def compute_rank_displacement(instruct_cf, base_cf):
    """Compute Kendall's tau rank correlation between instruct and base
    counterfactual token rankings at each position.

    Args:
        instruct_cf: list of [(token, prob), ...] per position (from LTP).
        base_cf: list of [(token, prob), ...] per position (from base pass).

    Returns:
        dict with mean_tau, mean_overlap, per_position_tau, per_position_overlap.
    """
    from scipy.stats import kendalltau

    if not instruct_cf or not base_cf:
        return None

    n_pos = min(len(instruct_cf), len(base_cf))
    taus = []
    overlaps = []

    for pos in range(n_pos):
        i_alts = instruct_cf[pos]
        b_alts = base_cf[pos]

        if not i_alts or not b_alts:
            overlaps.append(0.0)
            continue

        i_tokens = [tok for tok, prob in i_alts]
        b_tokens = [tok for tok, prob in b_alts]

        # Token overlap
        shared = [tok for tok in i_tokens if tok in b_tokens]
        all_tokens = set(i_tokens) | set(b_tokens)
        overlap = len(shared) / len(all_tokens) if all_tokens else 0.0
        overlaps.append(overlap)

        # Rank correlation on shared tokens
        if len(shared) >= 3:
            i_ranks = [i_tokens.index(tok) for tok in shared]
            b_ranks = [b_tokens.index(tok) for tok in shared]
            tau, _ = kendalltau(i_ranks, b_ranks)
            if not np.isnan(tau):
                taus.append(tau)

    return {
        'mean_tau': float(np.mean(taus)) if taus else None,
        'mean_overlap': float(np.mean(overlaps)) if overlaps else None,
        'per_position_tau': [round(t, 4) for t in taus] if taus else [],
        'per_position_overlap': [round(o, 4) for o in overlaps],
        'n_comparable': len(taus),
        'n_positions': n_pos,
    }
