"""
Lateral Tension Profile (LTP) computation engine.

Extends the ASM framework with directional information from the alignment field
surrounding the generation path. Computes per-token lateral tension profiles,
tension trajectories, and summary statistics (M, C, V, L).

Reference: "Geometric Alignment Signals in Language Model Representations:
The Lateral Tension Profile" (Ostrander, 2026)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class LTPResult:
    """Per-prompt LTP computation results."""
    # Per-token ordered scalar profiles: list of k-dim vectors, one per token
    profiles: List[np.ndarray] = field(default_factory=list)

    # Per-token tension point vectors (d-dimensional, in normal plane)
    tension_points: List[np.ndarray] = field(default_factory=list)

    # Per-token tension point magnitudes
    tension_magnitudes: List[float] = field(default_factory=list)

    # Per-token profile shape classification: "steep", "flat", "inverted"
    profile_shapes: List[str] = field(default_factory=list)

    # Counterfactual tokens at each position: list of [(token_str, logit_prob)]
    counterfactual_tokens: List[List[Tuple[str, float]]] = field(default_factory=list)

    # Summary statistics (per monitored layer, keyed by layer index)
    offset_magnitude: Dict[int, float] = field(default_factory=dict)   # M
    offset_consistency: Dict[int, float] = field(default_factory=dict)  # C
    offset_variance: Dict[int, float] = field(default_factory=dict)    # V
    lateral_coverage: Dict[int, float] = field(default_factory=dict)   # L

    # Aggregate summary (averaged across monitored layers)
    mean_M: float = 0.0
    mean_C: float = 0.0
    mean_V: float = 0.0
    mean_L: float = 0.0

    # Dual trajectory offset vectors (for visualization)
    semantic_trajectory: Optional[np.ndarray] = None  # (n, d) or (n, 2) after PCA
    tension_trajectory: Optional[np.ndarray] = None

    # Layer strategy used
    layer_strategy: str = "signal"  # "signal" or "late"
    monitored_layers: List[int] = field(default_factory=list)
    k: int = 8


def compute_ltp(model_manager, logits, tokens, input_ids,
                k: int = 8, layer_strategy: str = "signal") -> LTPResult:
    """
    Compute the Lateral Tension Profile for a completed forward pass.

    Args:
        model_manager: ModelManager with loaded state, activations, and deltas
        logits: model output logits tensor (1, seq_len, vocab_size)
        tokens: list of token strings
        input_ids: token ID tensor (1, seq_len)
        k: counterfactual neighborhood size
        layer_strategy: "signal" (middle third) or "late" (final third)

    Returns:
        LTPResult with per-token profiles, tension points, and summary stats
    """
    state = model_manager.state
    result = LTPResult(k=k, layer_strategy=layer_strategy)
    seq_len = len(tokens)
    device = state.device
    dtype = state.dtype

    # Determine monitored layers
    if layer_strategy == "late":
        late_start = 2 * state.n_layers // 3
        monitored = list(range(late_start, state.n_layers))
    else:
        monitored = list(state.signal_layers)

    # Filter to layers with available deltas
    monitored = [l for l in monitored
                 if f"model.layers.{l}.self_attn.v_proj.weight" in state.deltas]
    result.monitored_layers = monitored

    if not monitored:
        return result

    # Get the unembedding matrix
    model = state.model_instruct
    if hasattr(model, 'lm_head'):
        W_u = model.lm_head.weight.detach()  # (vocab_size, hidden_dim)
    elif hasattr(model.model, 'embed_tokens'):
        # Tied embeddings
        W_u = model.model.embed_tokens.weight.detach()
    else:
        return result

    # Extract top-k+1 alternatives at each position from logits
    # logits shape: (1, seq_len, vocab_size)
    log_probs = logits[0]  # (seq_len, vocab_size)
    token_ids = input_ids[0]  # (seq_len,)

    # For each position, get the chosen token and top-k alternatives
    per_position_alts = []   # list of [(alt_id, alt_prob), ...]
    per_position_chosen = []  # list of chosen token id

    for i in range(seq_len):
        chosen_id = token_ids[i].item()
        per_position_chosen.append(chosen_id)

        # Get top-(k+1) tokens by logit
        topk_result = torch.topk(log_probs[i], k + 1)
        topk_ids = topk_result.indices.tolist()
        topk_logits = topk_result.values

        # Exclude the chosen token, keep top-k alternatives
        alts = []
        probs = torch.softmax(topk_logits, dim=-1)
        for j, tid in enumerate(topk_ids):
            if tid != chosen_id and len(alts) < k:
                alts.append((tid, probs[j].item()))

        # If chosen wasn't in top-k+1, we already have k; otherwise pad
        if len(alts) < k:
            # Get more alternatives
            topk2 = torch.topk(log_probs[i], k + 5)
            probs2 = torch.softmax(topk2.values, dim=-1)
            existing_ids = {a[0] for a in alts}
            for j, tid in enumerate(topk2.indices.tolist()):
                if tid != chosen_id and tid not in existing_ids and len(alts) < k:
                    alts.append((tid, probs2[j].item()))

        per_position_alts.append(alts)

        # Store counterfactual token strings
        cf_tokens = [(state.tokenizer.decode(aid).strip(), prob)
                     for aid, prob in alts]
        result.counterfactual_tokens.append(cf_tokens)

    # Compute LTP across monitored layers, accumulate
    all_layer_tension_points = {l: [] for l in monitored}
    all_layer_profiles = {l: [] for l in monitored}

    for layer_idx in monitored:
        h_key = f"layer_{layer_idx}_h"
        if h_key not in model_manager.activations:
            # Try trajectory hook key
            h_key = f"layer_{layer_idx}_traj_attn"
            if h_key not in model_manager.activations:
                continue

        h = model_manager.activations[h_key][0]  # (seq_len, hidden_dim)
        dw_v = state.v_delta(layer_idx)
        if dw_v is None:
            continue

        for i in range(seq_len):
            alts = per_position_alts[i]
            chosen_id = per_position_chosen[i]

            if not alts:
                all_layer_tension_points[layer_idx].append(
                    torch.zeros(state.hidden_size, device=device, dtype=dtype))
                all_layer_profiles[layer_idx].append(
                    np.zeros(k))
                continue

            # Compute semantic trajectory direction τ
            if i > 0:
                diff = h[i] - h[i - 1]
                diff_norm = diff.norm()
                if diff_norm > 1e-8:
                    tau = diff / diff_norm
                else:
                    tau = torch.zeros_like(h[i])
            else:
                if seq_len > 1:
                    diff = h[1] - h[0]
                    diff_norm = diff.norm()
                    tau = diff / diff_norm if diff_norm > 1e-8 else torch.zeros_like(h[0])
                else:
                    tau = torch.zeros_like(h[0])

            # Compute counterfactual directions and lateral tensions
            profile_magnitudes = []
            weighted_tension = torch.zeros(state.hidden_size, device=device, dtype=dtype)
            prob_sum = 0.0

            for alt_id, alt_prob in alts:
                # Counterfactual direction: d_ic = W_u[c] - W_u[chosen]
                d_ic = W_u[alt_id] - W_u[chosen_id]

                # Project through weight delta: ΔW_V · d_ic
                delta_proj = torch.matmul(dw_v, d_ic)

                # Project onto normal plane (remove forward component)
                forward_component = torch.dot(delta_proj, tau) * tau
                lateral = delta_proj - forward_component

                lat_mag = lateral.norm().item()
                profile_magnitudes.append(lat_mag)

                # Accumulate weighted tension point
                weighted_tension += alt_prob * lateral
                prob_sum += alt_prob

            # Normalize weights
            if prob_sum > 0:
                weighted_tension /= prob_sum

            # Pad profile if fewer than k alternatives
            while len(profile_magnitudes) < k:
                profile_magnitudes.append(0.0)

            all_layer_tension_points[layer_idx].append(weighted_tension)
            all_layer_profiles[layer_idx].append(
                np.array(profile_magnitudes[:k]))

    # Aggregate across layers
    if not monitored:
        return result

    # Average profiles and tension points across monitored layers
    for i in range(seq_len):
        layer_profiles = []
        layer_tensions = []
        for l in monitored:
            if i < len(all_layer_profiles[l]):
                layer_profiles.append(all_layer_profiles[l][i])
            if i < len(all_layer_tension_points[l]):
                layer_tensions.append(all_layer_tension_points[l][i])

        if layer_profiles:
            avg_profile = np.mean(layer_profiles, axis=0)
        else:
            avg_profile = np.zeros(k)
        result.profiles.append(avg_profile)

        if layer_tensions:
            avg_tension = torch.stack(layer_tensions).mean(dim=0)
        else:
            avg_tension = torch.zeros(state.hidden_size, device=device, dtype=dtype)
        result.tension_points.append(avg_tension.cpu().numpy())
        result.tension_magnitudes.append(float(avg_tension.norm().item()))

        # Classify profile shape
        result.profile_shapes.append(_classify_profile(avg_profile))

    # Compute summary statistics per layer
    for layer_idx in monitored:
        points = all_layer_tension_points.get(layer_idx, [])
        if not points:
            continue

        magnitudes = [p.norm().item() for p in points]
        non_zero = [i for i, m in enumerate(magnitudes) if m > 1e-10]

        # L: lateral coverage
        result.lateral_coverage[layer_idx] = len(non_zero) / seq_len if seq_len > 0 else 0.0

        if not non_zero:
            result.offset_magnitude[layer_idx] = 0.0
            result.offset_consistency[layer_idx] = 0.0
            result.offset_variance[layer_idx] = 0.0
            continue

        # Mean offset vector
        active_points = torch.stack([points[i] for i in non_zero])
        mean_offset = active_points.mean(dim=0)

        # M: offset magnitude
        M = mean_offset.norm().item()
        result.offset_magnitude[layer_idx] = M

        # C: offset consistency
        mean_mag = np.mean([magnitudes[i] for i in non_zero])
        result.offset_consistency[layer_idx] = M / mean_mag if mean_mag > 0 else 0.0

        # V: offset magnitude variance
        active_mags = [magnitudes[i] for i in non_zero]
        result.offset_variance[layer_idx] = float(np.var(active_mags))

    # Aggregate summary stats across layers
    if monitored:
        result.mean_M = np.mean([result.offset_magnitude.get(l, 0.0) for l in monitored])
        result.mean_C = np.mean([result.offset_consistency.get(l, 0.0) for l in monitored])
        result.mean_V = np.mean([result.offset_variance.get(l, 0.0) for l in monitored])
        result.mean_L = np.mean([result.lateral_coverage.get(l, 0.0) for l in monitored])

    # Build dual trajectory for visualization (PCA to 2D)
    _compute_dual_trajectory(result, model_manager, monitored, seq_len)

    return result


def _classify_profile(profile: np.ndarray) -> str:
    """Classify a lateral tension profile as steep, flat, or inverted."""
    if len(profile) < 2 or np.sum(profile) < 1e-10:
        return "flat"

    # Normalize
    total = np.sum(profile)
    if total <= 0:
        return "flat"
    normed = profile / total

    # Steep: first entry dominates (> 40% of total and > 2x second)
    if normed[0] > 0.4 and (len(normed) < 2 or normed[0] > 2 * normed[1]):
        return "steep"

    # Inverted: later entries larger than earlier ones
    first_half = np.mean(normed[:len(normed)//2]) if len(normed) >= 2 else normed[0]
    second_half = np.mean(normed[len(normed)//2:]) if len(normed) >= 2 else 0
    if second_half > first_half * 1.3:
        return "inverted"

    return "flat"


def _compute_dual_trajectory(result: LTPResult, model_manager, monitored, seq_len):
    """Compute 2D projections of semantic and tension trajectories for visualization."""
    if seq_len < 2 or not monitored:
        return

    # Use the first monitored layer's activations
    layer_idx = monitored[0]
    h_key = f"layer_{layer_idx}_h"
    if h_key not in model_manager.activations:
        h_key = f"layer_{layer_idx}_traj_attn"
        if h_key not in model_manager.activations:
            return

    h = model_manager.activations[h_key][0].cpu().numpy()  # (seq_len, hidden_dim)

    # Semantic trajectory: raw positions
    # Tension trajectory: positions displaced by tension points
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

    # PCA to 2D for visualization
    combined = np.vstack([h, tension_traj])  # (2*seq_len, hidden_dim)
    mean = combined.mean(axis=0)
    centered = combined - mean

    # Truncated SVD for efficiency
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T  # (2*seq_len, 2)
        result.semantic_trajectory = proj[:seq_len]
        result.tension_trajectory = proj[seq_len:]
    except np.linalg.LinAlgError:
        pass


def ltp_result_to_dict(r: LTPResult) -> dict:
    """Serialize an LTPResult for JSON transport."""
    def _safe(v):
        if isinstance(v, (np.floating, np.float32, np.float64)):
            v = float(v)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(v, np.ndarray):
            return [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
                    else float(x) for x in v.tolist()]
        return v

    return {
        "profiles": [_safe(p) for p in r.profiles],
        "tension_magnitudes": [_safe(m) for m in r.tension_magnitudes],
        "profile_shapes": r.profile_shapes,
        "counterfactual_tokens": r.counterfactual_tokens,
        "offset_magnitude": {str(k): _safe(v) for k, v in r.offset_magnitude.items()},
        "offset_consistency": {str(k): _safe(v) for k, v in r.offset_consistency.items()},
        "offset_variance": {str(k): _safe(v) for k, v in r.offset_variance.items()},
        "lateral_coverage": {str(k): _safe(v) for k, v in r.lateral_coverage.items()},
        "mean_M": _safe(r.mean_M),
        "mean_C": _safe(r.mean_C),
        "mean_V": _safe(r.mean_V),
        "mean_L": _safe(r.mean_L),
        "layer_strategy": r.layer_strategy,
        "monitored_layers": r.monitored_layers,
        "k": r.k,
        "semantic_trajectory_2d": r.semantic_trajectory.tolist() if r.semantic_trajectory is not None else [],
        "tension_trajectory_2d": r.tension_trajectory.tolist() if r.tension_trajectory is not None else [],
    }
