"""Lateral Tension Profile computation using TAGM's adapter layer.

Core computation is identical to TASM's engine/ltp.py. Only the
model-access patterns are adapted: delta_store instead of state.deltas,
adapter.unembedding_weight instead of model.lm_head.weight, etc.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from src.engine import config as engine_config
from src.engine.result import LTPResult

if TYPE_CHECKING:
    from src.engine.analyzer import Analyzer

logger = logging.getLogger("src")


# ── SVD precomputation ──────────────────────────────────────────

def precompute_svd_cache(analyzer: "Analyzer", rank: int = 5) -> Dict[int, torch.Tensor]:
    """SVD-truncate dW_V at each layer, retaining top-k singular directions."""
    cache = {}
    for layer_idx in range(analyzer.n_layers):
        dw_v = analyzer.delta_store.v_delta_or_none(layer_idx)
        if dw_v is None:
            continue
        U, S, Vt = torch.linalg.svd(dw_v.float(), full_matrices=False)
        k = min(rank, len(S))
        truncated = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(analyzer.pipeline.dtype)
        cache[layer_idx] = truncated
    logger.info(f"[SVD] Cache built: {len(cache)} layers, rank={rank}")
    return cache


# ── Core computation ────────────────────────────────────────────

def compute_ltp(analyzer: "Analyzer", logits, tokens, input_ids,
                k: int = 8, layer_strategy: str = "signal",
                svd_cache: Dict[int, torch.Tensor] = None,
                svd_rank: int = 0,
                base_logits=None,
                precomputed_base_alts=None) -> LTPResult:
    """Compute the Lateral Tension Profile from a completed forward pass.

    Uses analyzer.delta_store for delta access and analyzer._capture.activations
    for cached hidden states. Core math is identical to TASM.
    """
    result = LTPResult(k=k, layer_strategy=layer_strategy, svd_rank=svd_rank)
    seq_len = len(tokens)
    device = analyzer.device
    dtype = analyzer.pipeline.dtype
    ds = analyzer.delta_store

    if layer_strategy == "late":
        late_start = 2 * analyzer.n_layers // 3
        monitored = list(range(late_start, analyzer.n_layers))
    else:
        monitored = list(analyzer.signal_layers)

    # Only keep layers that have v_proj deltas
    monitored = [l for l in monitored if ds.has(l, "v")]
    result.monitored_layers = monitored
    if not monitored:
        return result

    # Unembedding matrix
    try:
        W_u = analyzer.adapter.unembedding_weight(analyzer.instruct_model)
    except AttributeError:
        return result

    log_probs = logits[0]
    token_ids = input_ids[0]

    # Per-position instruct alternatives
    per_position_alts = []
    per_position_chosen = []
    for i in range(seq_len):
        chosen_id = token_ids[i].item()
        per_position_chosen.append(chosen_id)
        topk_result = torch.topk(log_probs[i], k + engine_config.get("ltp_overfetch_first"))
        probs = torch.softmax(topk_result.values, dim=-1)
        alts = [(tid, probs[j].item())
                for j, tid in enumerate(topk_result.indices.tolist())
                if tid != chosen_id and len([(t,p) for t,p in [] ]) < k]
        # Rebuild properly
        alts = []
        for j, tid in enumerate(topk_result.indices.tolist()):
            if tid != chosen_id and len(alts) < k:
                alts.append((tid, probs[j].item()))
        if len(alts) < k:
            topk2 = torch.topk(log_probs[i], k + engine_config.get("ltp_overfetch_second"))
            probs2 = torch.softmax(topk2.values, dim=-1)
            existing_ids = {a[0] for a in alts}
            for j, tid in enumerate(topk2.indices.tolist()):
                if tid != chosen_id and tid not in existing_ids and len(alts) < k:
                    alts.append((tid, probs2[j].item()))
        per_position_alts.append(alts)
        cf_tokens = [(analyzer.tokenizer.decode(aid).strip(), prob) for aid, prob in alts]
        result.counterfactual_tokens.append(cf_tokens)

    # Base model alternatives
    per_position_base_alts = []
    if precomputed_base_alts is not None:
        per_position_base_alts = precomputed_base_alts
    elif base_logits is not None:
        base_log_probs = base_logits[0]
        for i in range(seq_len):
            chosen_id = per_position_chosen[i]
            topk_result = torch.topk(base_log_probs[i], k + engine_config.get("ltp_overfetch_second"))
            probs = torch.softmax(topk_result.values, dim=-1)
            alts = []
            for j, tid in enumerate(topk_result.indices.tolist()):
                if tid != chosen_id and len(alts) < k:
                    alts.append((tid, probs[j].item()))
            per_position_base_alts.append(alts)

    # Per-layer, per-position computation
    activations = analyzer._capture.activations
    n_kv_heads = analyzer.n_kv_heads
    n_heads = analyzer.n_heads
    head_dim = analyzer.head_dim
    hidden_size = analyzer.hidden_size
    heads_per_kv = n_heads // n_kv_heads
    model = analyzer.instruct_model

    all_layer_tension_points = {l: [] for l in monitored}
    all_layer_profiles = {l: [] for l in monitored}
    all_layer_base_profiles = {l: [] for l in monitored}

    for layer_idx in monitored:
        h_key = f"layer_{layer_idx}_h"
        if h_key not in activations:
            h_key = f"layer_{layer_idx}_traj_attn"
            if h_key not in activations:
                continue

        h = activations[h_key][0]
        if h.shape[0] < seq_len:
            continue

        if svd_cache is not None and layer_idx in svd_cache:
            dw_v_full = svd_cache[layer_idx]
        else:
            dw_v_full = ds.v_delta_or_none(layer_idx)
        if dw_v_full is None:
            continue
        dw_v_half = dw_v_full * 0.5

        W_O = analyzer.adapter.o_proj_weight(model, layer_idx)

        def _project_alts(alts_list, chosen_id, tau, return_laterals=False):
            magnitudes = []
            laterals = [] if return_laterals else None
            for alt_id, alt_prob in alts_list:
                d_ic = W_u[alt_id] - W_u[chosen_id]
                delta_val = torch.matmul(dw_v_half, d_ic)
                expanded = delta_val.view(n_kv_heads, head_dim) \
                    .repeat_interleave(heads_per_kv, dim=0).reshape(-1)
                proj = torch.matmul(W_O, expanded)
                fwd = torch.dot(proj, tau) * tau
                lateral = proj - fwd
                magnitudes.append(lateral.norm().item())
                if return_laterals:
                    laterals.append((alt_prob, lateral))
            while len(magnitudes) < k:
                magnitudes.append(0.0)
            return magnitudes, laterals

        for i in range(seq_len):
            inst_alts = per_position_alts[i]
            base_alts = per_position_base_alts[i] if i < len(per_position_base_alts) else inst_alts
            chosen_id = per_position_chosen[i]

            if not inst_alts:
                all_layer_tension_points[layer_idx].append(
                    torch.zeros(hidden_size, device=device, dtype=dtype))
                all_layer_profiles[layer_idx].append(np.zeros(k))
                all_layer_base_profiles[layer_idx].append(np.zeros(k))
                continue

            if i > 0:
                diff = h[i] - h[i - 1]
                diff_norm = diff.norm()
                tau = diff / diff_norm if diff_norm > 1e-8 else torch.zeros_like(h[i])
            else:
                if seq_len > 1:
                    diff = h[1] - h[0]
                    diff_norm = diff.norm()
                    tau = diff / diff_norm if diff_norm > 1e-8 else torch.zeros_like(h[0])
                else:
                    tau = torch.zeros_like(h[0])

            instruct_magnitudes, inst_laterals = _project_alts(
                inst_alts, chosen_id, tau, return_laterals=True)
            base_magnitudes, _ = _project_alts(base_alts, chosen_id, tau)

            weighted_tension = torch.zeros(hidden_size, device=device, dtype=dtype)
            prob_sum = 0.0
            for alt_prob, lateral in inst_laterals:
                weighted_tension += alt_prob * lateral
                prob_sum += alt_prob
            if prob_sum > 0:
                weighted_tension /= prob_sum

            all_layer_tension_points[layer_idx].append(weighted_tension)
            all_layer_profiles[layer_idx].append(np.array(instruct_magnitudes[:k]))
            all_layer_base_profiles[layer_idx].append(np.array(base_magnitudes[:k]))

    # Aggregate across layers
    for i in range(seq_len):
        layer_profiles = [all_layer_profiles[l][i]
                          for l in monitored if i < len(all_layer_profiles[l])]
        layer_base_profiles = [all_layer_base_profiles[l][i]
                               for l in monitored if i < len(all_layer_base_profiles[l])]
        layer_tensions = [all_layer_tension_points[l][i]
                          for l in monitored if i < len(all_layer_tension_points[l])]

        avg_profile = np.mean(layer_profiles, axis=0) if layer_profiles else np.zeros(k)
        avg_base_profile = np.mean(layer_base_profiles, axis=0) if layer_base_profiles else np.zeros(k)
        result.profiles.append(avg_profile)
        result.base_profiles.append(avg_base_profile)

        if layer_tensions:
            avg_tension = torch.stack(layer_tensions).mean(dim=0)
        else:
            avg_tension = torch.zeros(hidden_size, device=device, dtype=dtype)
        result.tension_points.append(avg_tension.float().cpu().numpy())
        result.tension_magnitudes.append(float(avg_tension.norm().item()))
        result.profile_shapes.append(_classify_profile(avg_profile))

    # Per-layer statistics
    for layer_idx in monitored:
        points = all_layer_tension_points.get(layer_idx, [])
        if not points:
            continue
        magnitudes = [p.norm().item() for p in points]
        non_zero = [i for i, m in enumerate(magnitudes) if m > 1e-10]
        result.lateral_coverage[layer_idx] = len(non_zero) / seq_len if seq_len > 0 else 0.0
        if not non_zero:
            result.offset_magnitude[layer_idx] = 0.0
            result.offset_variance[layer_idx] = 0.0
            continue
        active_points = torch.stack([points[i] for i in non_zero])
        mean_offset = active_points.mean(dim=0)
        result.offset_magnitude[layer_idx] = mean_offset.norm().item()
        result.offset_variance[layer_idx] = float(np.var([magnitudes[i] for i in non_zero]))

    if monitored:
        result.mean_M = np.mean([result.offset_magnitude.get(l, 0.0) for l in monitored])
        result.mean_V = np.mean([result.offset_variance.get(l, 0.0) for l in monitored])
        result.mean_L = np.mean([result.lateral_coverage.get(l, 0.0) for l in monitored])

    # PRC (Peak Rank Concentration)
    PRC_THRESHOLD = 0.02
    prc_values = []
    for profile in result.profiles:
        if len(profile) >= 2:
            total = np.sum(profile)
            prc = float(np.max(profile / total) - 1.0 / k) if total > 0 else 0.0
        else:
            prc = 0.0
        prc_values.append(prc)
    result.prc_per_token = prc_values
    result.max_prc = max(prc_values) if prc_values else 0.0
    result.n_directional = sum(1 for p in prc_values if p > PRC_THRESHOLD)

    # Dual trajectory
    _compute_dual_trajectory(result, activations, monitored, seq_len, hidden_size)

    return result


def _classify_profile(profile: np.ndarray) -> str:
    if len(profile) < 2 or np.sum(profile) < 1e-10:
        return "flat"
    total = np.sum(profile)
    if total <= 0:
        return "flat"
    normed = profile / total
    if normed[0] > 0.4 and (len(normed) < 2 or normed[0] > 2 * normed[1]):
        return "steep"
    first_half = np.mean(normed[:len(normed)//2]) if len(normed) >= 2 else normed[0]
    second_half = np.mean(normed[len(normed)//2:]) if len(normed) >= 2 else 0
    if second_half > first_half * 1.3:
        return "inverted"
    return "flat"


def _compute_dual_trajectory(result, activations, monitored, seq_len, hidden_size):
    if seq_len < 2 or not monitored:
        return
    layer_idx = monitored[0]
    h_key = f"layer_{layer_idx}_h"
    if h_key not in activations:
        h_key = f"layer_{layer_idx}_traj_attn"
        if h_key not in activations:
            return
    h = activations[h_key][0].cpu().float().numpy()
    if h.shape[0] < seq_len:
        return
    tension_traj = np.zeros_like(h)
    for i in range(seq_len):
        if i < len(result.tension_points):
            tp = result.tension_points[i]
            if len(tp) == h.shape[1]:
                tension_traj[i] = h[i] + tp
            else:
                tension_traj[i] = h[i]
        else:
            tension_traj[i] = h[i]
    combined = np.vstack([h, tension_traj])
    mean = combined.mean(axis=0)
    centered = combined - mean
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T
        result.semantic_trajectory = proj[:seq_len]
        result.tension_trajectory = proj[seq_len:]
    except (np.linalg.LinAlgError, TypeError, ValueError):
        pass
