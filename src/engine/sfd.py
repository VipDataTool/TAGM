"""Spectral Field Density and Rank Displacement using TAGM's delta store.

Precompute function uses adapter-mediated delta access. Per-token SFD
computation and rank displacement are pure numpy — no model access needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from src.engine import config as engine_config

if TYPE_CHECKING:
    from src.engine.analyzer import Analyzer

logger = logging.getLogger("src")


# ── Data structures ─────────────────────────────────────────────

@dataclass
class SFDLayerCache:
    V_k: np.ndarray          # (k, d_in) right singular vectors
    S: np.ndarray             # (k,) singular values
    erank: float
    spectral_entropy: float
    norm_entropy: float
    log_volume: float
    stable_rank: float
    frob_norm: float


# ── Precomputation ──────────────────────────────────────────────

def precompute_sfd_cache(analyzer: "Analyzer", k: int = None) -> dict:
    """Compute SVD of concatenated [ΔW_Q; ΔW_K] per layer.

    Returns a dict with 'layers' (dict of layer_idx → SFDLayerCache),
    'k', 'mean_erank', etc.
    """
    if k is None:
        k = engine_config.get("sfd_svd_k")

    ds = analyzer.delta_store
    n_layers = analyzer.n_layers

    if engine_config.get("sfd_use_signal_layers"):
        layer_indices = list(analyzer.signal_layers)
    else:
        start = engine_config.get("sfd_layer_start")
        end = engine_config.get("sfd_layer_end")
        layer_indices = list(range(min(start, n_layers), min(end, n_layers)))

    layers_cache: Dict[int, SFDLayerCache] = {}
    eranks = []

    for layer_idx in layer_indices:
        dw_q = ds.get_or_none(layer_idx, "q")
        dw_k = ds.get_or_none(layer_idx, "k")
        if dw_q is None or dw_k is None:
            continue

        try:
            dw_qk = torch.cat([dw_q.float().cpu(), dw_k.float().cpu()], dim=0)

            svd_seed = engine_config.get("sfd_svd_seed")
            if svd_seed is not None:
                torch.manual_seed(svd_seed)

            actual_k = min(k, min(dw_qk.shape))
            U, S, V = torch.svd_lowrank(dw_qk, q=actual_k)

            s = S.float().numpy().astype(np.float64)
            v_k = V.float().numpy().astype(np.float32).T  # (k, d_in)

            # Filter degenerate singular values — and keep the right
            # singular vectors aligned with the surviving values. Filtering
            # only ``s`` previously left V_k with more rows than S, which
            # made ``S * c`` a shape-mismatch crash in compute_sfd.
            keep = s > 1e-10
            s_pos = s[keep]
            if len(s_pos) == 0:
                continue
            v_k = v_k[keep]

            p = s_pos / s_pos.sum()
            H = -np.sum(p * np.log(p))
            erank = float(np.exp(H))
            norm_H = H / np.log(len(s_pos)) if len(s_pos) > 1 else 0.0
            log_vol = float(np.sum(np.log(s_pos)))
            frob2 = float(np.sum(s ** 2))
            stable_r = frob2 / (s[0] ** 2) if s[0] > 0 else 1.0

            layers_cache[layer_idx] = SFDLayerCache(
                V_k=v_k, S=s_pos.astype(np.float32),
                erank=erank, spectral_entropy=float(H),
                norm_entropy=float(norm_H), log_volume=log_vol,
                stable_rank=stable_r, frob_norm=float(np.sqrt(frob2)),
            )
            eranks.append(erank)

        except Exception as e:
            logger.error(f"[SFD] Layer {layer_idx} SVD failed: {e}")

    mean_erank = float(np.mean(eranks)) if eranks else 0.0

    return {
        "layers": layers_cache,
        "k": k,
        "n_layers": len(layers_cache),
        "mean_erank": mean_erank,
    }


# ── Per-token computation ───────────────────────────────────────

def compute_sfd(layer_acts: dict, cache: dict):
    """Compute SFD for a full token sequence across all cached layers.

    Args:
        layer_acts: {layer_idx: (n_tokens, d_in) numpy array}
        cache: precomputed SFD cache from precompute_sfd_cache().

    Returns:
        SFDResult-compatible object (uses engine.result.SFDResult).
    """
    from src.engine.result import SFDResult

    layers_cache = cache.get("layers", {})
    if not layers_cache:
        return SFDResult()

    n_tokens = None
    accum_density = None
    n_layers_used = 0

    for layer_idx, layer_cache in layers_cache.items():
        acts = layer_acts.get(layer_idx)
        if acts is None:
            continue
        if len(acts.shape) == 1:
            acts = acts.reshape(1, -1)

        if n_tokens is None:
            n_tokens = acts.shape[0]
            accum_density = np.zeros(n_tokens)
            accum_directions = [None] * n_tokens

        V_k = layer_cache.V_k
        S = layer_cache.S

        for t in range(min(n_tokens, acts.shape[0])):
            c = V_k @ acts[t]
            w = S[:len(c)] * c
            w2 = w * w
            energy = w2.sum()
            if energy < 1e-20:
                continue
            q = w2 / energy
            q_pos = q[q > 1e-10]
            H_t = float(-np.sum(q_pos * np.log(q_pos)))
            erank_t = float(np.exp(H_t))
            accum_density[t] += erank_t / layer_cache.erank if layer_cache.erank > 0 else 0.0

            # Accumulate weighted direction (layer-averaged)
            w_normed = w / (np.linalg.norm(w) + 1e-10)
            if accum_directions[t] is None:
                accum_directions[t] = w_normed.astype(np.float32)
            else:
                accum_directions[t] = accum_directions[t] + w_normed.astype(np.float32)

        n_layers_used += 1

    if n_layers_used == 0 or n_tokens is None:
        return SFDResult()

    per_density = accum_density / n_layers_used

    # Normalize accumulated directions across layers and serialize
    per_directions = []
    for t in range(n_tokens):
        if accum_directions[t] is not None:
            d = accum_directions[t] / n_layers_used
            per_directions.append([round(float(x), 6) for x in d])
        else:
            per_directions.append([])

    return SFDResult(
        per_token_density=list(per_density),
        per_token_directions=per_directions,
        density_mean=float(np.mean(per_density)),
        density_max=float(np.max(per_density)),
        density_var=float(np.var(per_density)),
        density_p90=float(np.percentile(per_density, 90)),
        global_erank=cache.get("mean_erank", 0.0),
        n_layers_used=n_layers_used,
    )


# ── Rank displacement ──────────────────────────────────────────

def compute_rank_displacement(instruct_cf, base_cf):
    """Displacement between instruct and base counterfactual candidate sets.

    Pure computation — no model access. Identical to TASM's implementation.
    """
    from scipy.stats import kendalltau

    if not instruct_cf or not base_cf:
        return None

    n_pos = min(len(instruct_cf), len(base_cf))
    k = 8
    per_pos = []
    taus = []
    overlaps = []
    instruct_disp_profiles = []
    base_disp_profiles = []
    p = engine_config.get("serialization_precision")

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

        # Keyed by surface string; distinct token ids can decode+strip to
        # the same string (" the" vs "the"), so disambiguate duplicates by
        # occurrence index — keeping key, rank, and prob mutually consistent
        # (the old dict kept the LAST duplicate while the tau path used the
        # FIRST, silently mixing ranks).
        def _keyed(alts):
            out, seen = {}, {}
            for rank, (tok, prob) in enumerate(alts):
                n = seen.get(tok, 0)
                seen[tok] = n + 1
                key = tok if n == 0 else f"{tok}\u0000{n}"
                out[key] = (rank, prob)
            return out

        inst = _keyed(i_alts)
        base = _keyed(b_alts)

        matched_tokens = set(inst) & set(base)
        promoted_tokens = set(inst) - set(base)
        demoted_tokens = set(base) - set(inst)

        matched_disp = sum(abs(inst[t][1] - base[t][1]) for t in matched_tokens)
        promoted_mass = sum(inst[t][1] for t in promoted_tokens)
        demoted_mass = sum(base[t][1] for t in demoted_tokens)
        total_disp = matched_disp + promoted_mass + demoted_mass
        replacement_ratio = (promoted_mass + demoted_mass) / total_disp if total_disp > 0 else 0.0
        concentration = total_disp / len(matched_tokens) if matched_tokens else 0.0

        per_pos.append({
            'n_matched': len(matched_tokens), 'n_promoted': len(promoted_tokens),
            'n_demoted': len(demoted_tokens),
            'matched_disp': round(matched_disp, p), 'promoted_mass': round(promoted_mass, p),
            'demoted_mass': round(demoted_mass, p), 'total_disp': round(total_disp, p),
            'replacement_ratio': round(replacement_ratio, p),
            'concentration': round(concentration, p),
        })

        # Terrain profiles
        i_profile = []
        for t, prob in i_alts[:k]:
            i_profile.append(abs(prob - base[t][1]) if t in base else prob)
        while len(i_profile) < k:
            i_profile.append(0.0)
        instruct_disp_profiles.append(i_profile)

        b_profile = []
        for t, prob in b_alts[:k]:
            b_profile.append(abs(prob - inst[t][1]) if t in inst else prob)
        while len(b_profile) < k:
            b_profile.append(0.0)
        base_disp_profiles.append(b_profile)

        # Legacy tau/overlap — same disambiguated keys as the mass metrics
        i_tokens = list(inst.keys())
        b_tokens = list(base.keys())
        all_tokens = set(i_tokens) | set(b_tokens)
        overlaps.append(len(matched_tokens) / len(all_tokens) if all_tokens else 0.0)

        shared = [tok for tok in i_tokens if tok in set(b_tokens)]
        if len(shared) >= engine_config.get("rd_min_shared"):
            i_ranks = [i_tokens.index(tok) for tok in shared]
            b_ranks = [b_tokens.index(tok) for tok in shared]
            tau, _ = kendalltau(i_ranks, b_ranks)
            taus.append(tau if not np.isnan(tau) else 0.0)
        else:
            taus.append(0.0)

    n = len(per_pos)
    return {
        'mean_matched': round(sum(x['n_matched'] for x in per_pos) / n, p) if n else 0,
        'mean_replacement': round(sum(x['replacement_ratio'] for x in per_pos) / n, p) if n else 0,
        'mean_concentration': round(sum(x['concentration'] for x in per_pos) / n, p) if n else 0,
        'mean_disp_per_token': round(sum(x['total_disp'] for x in per_pos) / n, p) if n else 0,
        'total_displacement': round(sum(x['total_disp'] for x in per_pos), p),
        'high_replacement_frac': round(sum(1 for x in per_pos if x['replacement_ratio'] > 0.5) / n, p) if n else 0,
        'low_match_frac': round(sum(1 for x in per_pos if x['n_matched'] < 5) / n, p) if n else 0,
        'per_position': per_pos,
        'instruct_disp_profiles': instruct_disp_profiles,
        'base_disp_profiles': base_disp_profiles,
        'mean_tau': float(np.mean(taus)) if taus else 0.0,
        'mean_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
        'per_position_tau': [round(t, p) for t in taus],
        'per_position_overlap': [round(o, p) for o in overlaps],
        'n_comparable': len(taus),
        'n_positions': n_pos,
    }
