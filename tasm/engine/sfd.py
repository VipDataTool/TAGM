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

from engine import engine_config

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
    per_token_density: Optional[np.ndarray] = None

    # Prompt-level aggregates
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
        p = engine_config.get("serialization_precision")
        return {
            'per_token_density': self.per_token_density.tolist() if self.per_token_density is not None else None,
            'density_mean': round(self.density_mean, p),
            'density_max': round(self.density_max, p),
            'density_var': round(self.density_var, p),
            'density_p90': round(self.density_p90, p),
            'global_erank': round(self.global_erank, p),
            'n_layers_monitored': self.n_layers_monitored,
            'k': self.k,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstitute from stored dict."""
        if d is None:
            return None
        r = cls()
        for key in ['density_mean', 'density_max', 'density_var', 'density_p90',
                     'global_erank', 'n_layers_monitored', 'k']:
            if key in d:
                setattr(r, key, d[key])
        if d.get('per_token_density') is not None:
            r.per_token_density = np.array(d['per_token_density'], dtype=float)
        return r


# ═══ Precomputation (model load) ═══

def precompute_sfd_cache(state, layer_indices: List[int] = None,
                         k: int = None) -> SFDCache:
    """Compute SVD of concatenated [ΔW_Q; ΔW_K] per layer.

    Args:
        state: ModelState with deltas dict.
        layer_indices: which layers to compute. Default from config.
        k: truncation rank for SVD. Default from config.

    Returns:
        SFDCache with V_k, S, and global measures per layer.
    """
    if k is None:
        k = engine_config.get("sfd_svd_k")
    if layer_indices is None:
        if engine_config.get("sfd_use_signal_layers"):
            layer_indices = list(state.signal_layers) if hasattr(state, 'signal_layers') else None
        if layer_indices is None:
            start = engine_config.get("sfd_layer_start")
            end = engine_config.get("sfd_layer_end")
            layer_indices = list(range(min(start, state.n_layers), min(end, state.n_layers)))

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

            # Truncated SVD — seed for determinism (randomized algorithm)
            svd_seed = engine_config.get("sfd_svd_seed")
            if svd_seed is not None:
                torch.manual_seed(svd_seed)
            # torch.svd_lowrank returns (U, S, V) where A ≈ U @ diag(S) @ V.T
            # V is (d_in, k) — we need V.T = (k, d_in) for projection c = V_k^T @ h
            actual_k = min(k, min(dw_qk.shape))
            U, S, V = torch.svd_lowrank(dw_qk, q=actual_k)

            s = S.float().numpy().astype(np.float64)
            v_k = V.float().numpy().astype(np.float32).T  # (k, d_in) — transposed for V_k^T @ h

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
        per_token_density=per_density,
        global_erank=cache.mean_erank,
        n_layers_monitored=n_layers_used,
        k=cache.k,
    )

    # Prompt-level aggregates
    if n_tokens > 0:
        result.density_mean = float(np.mean(per_density))
        result.density_max = float(np.max(per_density))
        result.density_var = float(np.var(per_density))
        result.density_p90 = float(np.percentile(per_density, 90))

    return result


# ═══ Rank displacement (LTP companion measure) ═══

def compute_rank_displacement(instruct_cf, base_cf):
    """Compute vocabulary-space displacement between instruct and base model
    counterfactual candidate sets at each token position.

    Decomposes displacement into three pools per position:
      - Matched: candidates in both top-k. Displacement = |Δprob|.
      - Promoted: instruct-only candidates. Displacement = full inst prob.
      - Demoted: base-only candidates. Displacement = full base prob.

    Also produces per-position displacement profiles (8 values per bank)
    suitable for the terrain viewer, where each height is a candidate's
    contribution to the total displacement.

    Args:
        instruct_cf: list of [(token, prob), ...] per position (from LTP).
        base_cf: list of [(token, prob), ...] per position (from base pass).

    Returns:
        dict with per-position and aggregate match-concentration metrics,
        displacement profiles for terrain rendering, and legacy tau/overlap.
    """
    from scipy.stats import kendalltau

    if not instruct_cf or not base_cf:
        return None

    n_pos = min(len(instruct_cf), len(base_cf))
    k = 8  # expected candidates per bank

    # Per-position accumulators
    per_pos = []
    taus = []          # legacy: Kendall's tau
    overlaps = []      # legacy: Jaccard overlap

    # Terrain-compatible displacement profiles: 8 values per bank per position
    instruct_disp_profiles = []  # list of 8-element lists
    base_disp_profiles = []      # list of 8-element lists

    for pos in range(n_pos):
        i_alts = instruct_cf[pos]
        b_alts = base_cf[pos]

        if not i_alts or not b_alts:
            per_pos.append({
                'n_matched': 0, 'n_promoted': 0, 'n_demoted': 0,
                'matched_disp': 0.0, 'promoted_mass': 0.0, 'demoted_mass': 0.0,
                'total_disp': 0.0, 'replacement_ratio': 0.0, 'concentration': 0.0,
            })
            instruct_disp_profiles.append([0.0] * k)
            base_disp_profiles.append([0.0] * k)
            overlaps.append(0.0)
            taus.append(0.0)
            continue

        # Build token -> (rank, prob) lookups
        inst = {t: (rank, p) for rank, (t, p) in enumerate(i_alts)}
        base = {t: (rank, p) for rank, (t, p) in enumerate(b_alts)}

        matched_tokens = set(inst.keys()) & set(base.keys())
        promoted_tokens = set(inst.keys()) - set(base.keys())
        demoted_tokens = set(base.keys()) - set(inst.keys())

        n_matched = len(matched_tokens)
        n_promoted = len(promoted_tokens)
        n_demoted = len(demoted_tokens)

        matched_disp = sum(abs(inst[t][1] - base[t][1]) for t in matched_tokens)
        promoted_mass = sum(inst[t][1] for t in promoted_tokens)
        demoted_mass = sum(base[t][1] for t in demoted_tokens)
        total_disp = matched_disp + promoted_mass + demoted_mass

        replacement_ratio = ((promoted_mass + demoted_mass) / total_disp
                             if total_disp > 0 else 0.0)
        concentration = total_disp / n_matched if n_matched > 0 else 0.0

        p = engine_config.get("serialization_precision")
        per_pos.append({
            'n_matched': n_matched,
            'n_promoted': n_promoted,
            'n_demoted': n_demoted,
            'matched_disp': round(matched_disp, p),
            'promoted_mass': round(promoted_mass, p),
            'demoted_mass': round(demoted_mass, p),
            'total_disp': round(total_disp, p),
            'replacement_ratio': round(replacement_ratio, p),
            'concentration': round(concentration, p),
        })

        # ── Terrain displacement profiles ──
        # Instruct bank: each instruct candidate's displacement contribution
        i_profile = []
        for t, p in i_alts[:k]:
            if t in base:
                i_profile.append(abs(p - base[t][1]))  # matched: |Δp|
            else:
                i_profile.append(p)  # promoted: full prob
        while len(i_profile) < k:
            i_profile.append(0.0)
        instruct_disp_profiles.append(i_profile)

        # Base bank: each base candidate's displacement contribution
        b_profile = []
        for t, p in b_alts[:k]:
            if t in inst:
                b_profile.append(abs(p - inst[t][1]))  # matched: |Δp|
            else:
                b_profile.append(p)  # demoted: full prob
        while len(b_profile) < k:
            b_profile.append(0.0)
        base_disp_profiles.append(b_profile)

        # ── Legacy metrics ──
        i_tokens = [tok for tok, prob in i_alts]
        b_tokens = [tok for tok, prob in b_alts]
        all_tokens = set(i_tokens) | set(b_tokens)
        overlap = len(matched_tokens) / len(all_tokens) if all_tokens else 0.0
        overlaps.append(overlap)

        shared = [tok for tok in i_tokens if tok in b_tokens]
        if len(shared) >= engine_config.get("rd_min_shared"):
            i_ranks = [i_tokens.index(tok) for tok in shared]
            b_ranks = [b_tokens.index(tok) for tok in shared]
            tau, _ = kendalltau(i_ranks, b_ranks)
            taus.append(tau if not np.isnan(tau) else 0.0)
        else:
            # 0 or 1 shared candidates — nothing to correlate
            taus.append(0.0)

    # ── Aggregate statistics ──
    n = len(per_pos)
    mean_matched = sum(p['n_matched'] for p in per_pos) / n if n else 0
    mean_replacement = sum(p['replacement_ratio'] for p in per_pos) / n if n else 0
    mean_concentration = sum(p['concentration'] for p in per_pos) / n if n else 0
    mean_disp_per_token = sum(p['total_disp'] for p in per_pos) / n if n else 0
    total_displacement = sum(p['total_disp'] for p in per_pos)
    high_replacement_frac = (sum(1 for p in per_pos if p['replacement_ratio'] > 0.5) / n
                             if n else 0)
    low_match_frac = (sum(1 for p in per_pos if p['n_matched'] < 5) / n
                      if n else 0)

    p = engine_config.get("serialization_precision")
    return {
        # ── New match-concentration metrics ──
        'mean_matched': round(mean_matched, p),
        'mean_replacement': round(mean_replacement, p),
        'mean_concentration': round(mean_concentration, p),
        'mean_disp_per_token': round(mean_disp_per_token, p),
        'total_displacement': round(total_displacement, p),
        'high_replacement_frac': round(high_replacement_frac, p),
        'low_match_frac': round(low_match_frac, p),
        'per_position': per_pos,

        # ── Terrain displacement profiles (8 per bank per position) ──
        'instruct_disp_profiles': instruct_disp_profiles,
        'base_disp_profiles': base_disp_profiles,

        # ── Legacy (backward compat) ──
        'mean_tau': float(np.mean(taus)) if taus else 0.0,
        'mean_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
        'per_position_tau': [round(t, p) for t in taus],
        'per_position_overlap': [round(o, p) for o in overlaps],
        'n_comparable': len(taus),
        'n_positions': n_pos,
    }
