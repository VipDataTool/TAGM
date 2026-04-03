"""
Lateral Tension Profile (LTP) computation engine.

Extends the ASM framework with directional information from the alignment field
surrounding the generation path. Computes per-token lateral tension profiles,
tension trajectories, and summary statistics (M, C, V, L).

Two optional signal-sharpening enhancements (independently togglable):
  - SVD truncation: projects through the dominant safety subspace of dW_V
  - Tuned-lens correction: calibrates unembedding probe directions per layer

Reference: "Geometric Alignment Signals in Language Model Representations:
The Lateral Tension Profile" (Ostrander, 2026)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import logging
from engine import engine_config

logger = logging.getLogger("tasm")


# ─── Precomputation: SVD truncation ─────────────────────────────

def precompute_svd_cache(state, rank: int = 5) -> Dict[int, torch.Tensor]:
    """SVD-truncate dW_V at each layer, retaining only the top-k singular
    directions where alignment corrections concentrate. Returns dict mapping
    layer_idx -> truncated dW_V tensor (same shape as original, rank <= k)."""
    cache = {}
    for layer_idx in range(state.n_layers):
        dw_v = state.v_delta(layer_idx)
        if dw_v is None:
            continue
        U, S, Vt = torch.linalg.svd(dw_v.float(), full_matrices=False)
        k = min(rank, len(S))
        truncated = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(state.dtype)
        cache[layer_idx] = truncated
        total_energy = (S ** 2).sum().item()
        kept_energy = (S[:k] ** 2).sum().item()
        frac = kept_energy / total_energy if total_energy > 0 else 0
        logger.info(f"[SVD] Layer {layer_idx}: rank-{k} captures {frac:.1%} of energy")
    logger.info(f"[SVD] Cache built: {len(cache)} layers, rank={rank}")
    return cache


# ─── Precomputation: tuned-lens calibration ──────────────────────

def precompute_tuned_lens_cache(model_manager, calibration_prompts: List[str],
                                 layer_indices: List[int] = None
                                 ) -> Dict[int, torch.Tensor]:
    """Fit per-layer affine transforms mapping intermediate hidden states to
    final hidden states. Returns dict mapping layer_idx -> A_l (d x d).
    To correct unembedding direction d for layer l: d_corrected = A_l @ d."""
    state = model_manager.state
    model = state.model_instruct
    device = state.device
    dtype = state.dtype
    if layer_indices is None:
        layer_indices = list(state.signal_layers)

    collected = {}
    hooks = []

    def make_hook(key):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                collected.setdefault(key, []).append(output[0].detach())
            else:
                collected.setdefault(key, []).append(output.detach())
        return hook

    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"h_{layer_idx}")))
    hooks.append(model.model.norm.register_forward_hook(make_hook("h_final")))

    logger.info(f"[TunedLens] Calibrating with {len(calibration_prompts)} prompts "
                f"across {len(layer_indices)} layers...")
    with torch.no_grad():
        for prompt in calibration_prompts:
            inputs = state.tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)

    for h in hooks:
        h.remove()

    H_final = torch.cat(collected.get("h_final", []), dim=1).squeeze(0).float()
    cache = {}
    for layer_idx in layer_indices:
        key = f"h_{layer_idx}"
        if key not in collected:
            continue
        H_l = torch.cat(collected[key], dim=1).squeeze(0).float()
        if H_l.shape[0] != H_final.shape[0]:
            logger.warning(f"[TunedLens] Shape mismatch at layer {layer_idx}, skipping")
            continue
        try:
            result = torch.linalg.lstsq(H_l, H_final)
            A_l = result.solution.to(dtype)
            cache[layer_idx] = A_l
            pred = H_l @ A_l.float()
            residual = (pred - H_final).norm() / H_final.norm()
            logger.info(f"[TunedLens] Layer {layer_idx}: residual = {residual:.4f}")
        except Exception as e:
            logger.warning(f"[TunedLens] Layer {layer_idx} fit failed: {e}")

    del collected
    import gc; gc.collect()
    logger.info(f"[TunedLens] Cache built: {len(cache)} layers, "
                f"{H_final.shape[0]} tokens")
    return cache


# ─── LTP Result ──────────────────────────────────────────────────

@dataclass
class LTPResult:
    """Per-prompt LTP computation results."""
    profiles: List[np.ndarray] = field(default_factory=list)
    base_profiles: List[np.ndarray] = field(default_factory=list)  # Base bank: ΔW/2 with base counterfactuals
    tension_points: List[np.ndarray] = field(default_factory=list)
    tension_magnitudes: List[float] = field(default_factory=list)
    profile_shapes: List[str] = field(default_factory=list)
    counterfactual_tokens: List[List[Tuple[str, float]]] = field(default_factory=list)
    offset_magnitude: Dict[int, float] = field(default_factory=dict)
    offset_consistency: Dict[int, float] = field(default_factory=dict)
    offset_variance: Dict[int, float] = field(default_factory=dict)
    lateral_coverage: Dict[int, float] = field(default_factory=dict)
    mean_M: float = 0.0
    mean_C: float = 0.0
    mean_V: float = 0.0
    mean_L: float = 0.0
    max_prc: float = 0.0          # Peak Rank Concentration of hottest token
    n_directional: int = 0         # Tokens with PRC > threshold
    prc_per_token: List[float] = field(default_factory=list)  # Per-token PRC values
    semantic_trajectory: Optional[np.ndarray] = None
    tension_trajectory: Optional[np.ndarray] = None
    layer_strategy: str = "signal"
    monitored_layers: List[int] = field(default_factory=list)
    k: int = 8
    svd_rank: int = 0
    tuned_lens: bool = False


# ─── Core computation ────────────────────────────────────────────

def compute_ltp(model_manager, logits, tokens, input_ids,
                k: int = 8, layer_strategy: str = "signal",
                svd_cache: Dict[int, torch.Tensor] = None,
                svd_rank: int = 0,
                tuned_lens_cache: Dict[int, torch.Tensor] = None,
                base_logits=None,
                precomputed_base_alts=None) -> LTPResult:
    """Compute the Lateral Tension Profile for a completed forward pass.
    svd_cache: precomputed truncated dW_V per layer (None = use raw delta).
    svd_rank: the truncation rank used (for recording in results).
    tuned_lens_cache: precomputed per-layer affine transforms (None = raw probes).
    precomputed_base_alts: list of [(token_id, prob), ...] per position from a
        prior base-model pass. If provided, base_logits is not needed — enables
        sequential (non-concurrent) model loading."""
    state = model_manager.state
    result = LTPResult(
        k=k, layer_strategy=layer_strategy,
        svd_rank=svd_rank,
        tuned_lens=tuned_lens_cache is not None and len(tuned_lens_cache) > 0,
    )
    seq_len = len(tokens)
    device = state.device
    dtype = state.dtype

    if layer_strategy == "late":
        late_start = 2 * state.n_layers // 3
        monitored = list(range(late_start, state.n_layers))
    else:
        monitored = list(state.signal_layers)

    monitored = [l for l in monitored
                 if f"model.layers.{l}.self_attn.v_proj.weight" in state.deltas]
    result.monitored_layers = monitored
    if not monitored:
        return result

    model = state.model_instruct
    if hasattr(model, 'lm_head'):
        W_u = model.lm_head.weight.detach()
    elif hasattr(model.model, 'embed_tokens'):
        W_u = model.model.embed_tokens.weight.detach()
    else:
        return result

    log_probs = logits[0]
    token_ids = input_ids[0]

    per_position_alts = []
    per_position_chosen = []

    for i in range(seq_len):
        chosen_id = token_ids[i].item()
        per_position_chosen.append(chosen_id)
        topk_result = torch.topk(log_probs[i], k + engine_config.get("ltp_overfetch_first"))
        topk_ids = topk_result.indices.tolist()
        topk_logits = topk_result.values
        alts = []
        probs = torch.softmax(topk_logits, dim=-1)
        for j, tid in enumerate(topk_ids):
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
        cf_tokens = [(state.tokenizer.decode(aid).strip(), prob) for aid, prob in alts]
        result.counterfactual_tokens.append(cf_tokens)

    # ── Base model counterfactual alternatives (for base bank probe directions) ──
    per_position_base_alts = []
    if precomputed_base_alts is not None:
        # Sequential mode: base alts were pre-computed in a prior base-model phase
        per_position_base_alts = precomputed_base_alts
        logger.info(f"[LTP] Using {len(per_position_base_alts)} pre-computed base alts (sequential mode)")
    elif base_logits is not None:
        base_log_probs = base_logits[0]
        for i in range(seq_len):
            chosen_id = per_position_chosen[i]
            topk_result = torch.topk(base_log_probs[i], k + engine_config.get("ltp_overfetch_second"))
            topk_ids = topk_result.indices.tolist()
            probs = torch.softmax(topk_result.values, dim=-1)
            alts = []
            for j, tid in enumerate(topk_ids):
                if tid != chosen_id and len(alts) < k:
                    alts.append((tid, probs[j].item()))
            per_position_base_alts.append(alts)

    all_layer_tension_points = {l: [] for l in monitored}
    all_layer_profiles = {l: [] for l in monitored}
    all_layer_base_profiles = {l: [] for l in monitored}

    # Log which activation keys are available for LTP layers
    avail_keys = [k for k in model_manager.activations.keys() if 'layer_' in k]
    ltp_keys = [f"layer_{l}_h" for l in monitored]
    found = [k for k in ltp_keys if k in model_manager.activations]
    if not found:
        logger.warning(f"[LTP] No activations found for monitored layers {monitored}. "
                      f"Available keys: {avail_keys[:10]}...")

    for layer_idx in monitored:
        h_key = f"layer_{layer_idx}_h"
        if h_key not in model_manager.activations:
            h_key = f"layer_{layer_idx}_traj_attn"
            if h_key not in model_manager.activations:
                continue

        h = model_manager.activations[h_key][0]

        # Guard: activation length must match seq_len. Mismatch can occur
        # if a prior failed analysis left stale activations.
        if h.shape[0] < seq_len:
            logger.warning(f"[LTP] Activation size mismatch at layer {layer_idx}: "
                          f"h={h.shape[0]} vs seq_len={seq_len}. Skipping layer.")
            continue

        # ── ΔW_V / 2 ──
        if svd_cache is not None and layer_idx in svd_cache:
            dw_v_full = svd_cache[layer_idx]
        else:
            dw_v_full = state.v_delta(layer_idx)
        if dw_v_full is None:
            continue
        dw_v_half = dw_v_full * 0.5

        W_O = model.model.layers[layer_idx].self_attn.o_proj.weight.detach()
        n_kv_heads = state.n_kv_heads
        n_heads = state.n_heads
        head_dim = state.head_dim
        heads_per_kv = n_heads // n_kv_heads

        has_tl = (tuned_lens_cache is not None and layer_idx in tuned_lens_cache)
        if has_tl:
            A_l = tuned_lens_cache[layer_idx]

        def _project_alts(alts_list, chosen_id, tau, return_laterals=False):
            """Project a set of counterfactual alternatives through ΔW/2 → W_O → lateral.
            Same pipeline for both banks — only the input alternatives differ.
            When return_laterals=True, also returns the full lateral vectors
            (needed for weighted tension computation on the instruct bank)."""
            magnitudes = []
            laterals = [] if return_laterals else None
            for alt_id, alt_prob in alts_list:
                d_ic = W_u[alt_id] - W_u[chosen_id]
                if has_tl:
                    d_ic = torch.matmul(A_l, d_ic)
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
                    torch.zeros(state.hidden_size, device=device, dtype=dtype))
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

            # Instruct bank: get magnitudes AND lateral vectors in one pass
            instruct_magnitudes, inst_laterals = _project_alts(
                inst_alts, chosen_id, tau, return_laterals=True)
            # Base bank: magnitudes only (no weighted tension needed)
            base_magnitudes, _ = _project_alts(base_alts, chosen_id, tau)

            # Weighted tension from the instruct bank's already-computed laterals
            # (reuses projections instead of recomputing them)
            weighted_tension = torch.zeros(state.hidden_size, device=device, dtype=dtype)
            prob_sum = 0.0
            for alt_prob, lateral in inst_laterals:
                weighted_tension += alt_prob * lateral
                prob_sum += alt_prob
            if prob_sum > 0:
                weighted_tension /= prob_sum

            all_layer_tension_points[layer_idx].append(weighted_tension)
            all_layer_profiles[layer_idx].append(np.array(instruct_magnitudes[:k]))
            all_layer_base_profiles[layer_idx].append(np.array(base_magnitudes[:k]))

    if not monitored:
        return result

    for i in range(seq_len):
        layer_profiles = []
        layer_base_profiles = []
        layer_tensions = []
        for l in monitored:
            if i < len(all_layer_profiles[l]):
                layer_profiles.append(all_layer_profiles[l][i])
            if i < len(all_layer_base_profiles[l]):
                layer_base_profiles.append(all_layer_base_profiles[l][i])
            if i < len(all_layer_tension_points[l]):
                layer_tensions.append(all_layer_tension_points[l][i])
        avg_profile = np.mean(layer_profiles, axis=0) if layer_profiles else np.zeros(k)
        avg_base_profile = np.mean(layer_base_profiles, axis=0) if layer_base_profiles else np.zeros(k)
        result.profiles.append(avg_profile)
        result.base_profiles.append(avg_base_profile)
        if layer_tensions:
            avg_tension = torch.stack(layer_tensions).mean(dim=0)
        else:
            avg_tension = torch.zeros(state.hidden_size, device=device, dtype=dtype)
        result.tension_points.append(avg_tension.float().cpu().numpy())
        result.tension_magnitudes.append(float(avg_tension.norm().item()))
        result.profile_shapes.append(_classify_profile(avg_profile))

    for layer_idx in monitored:
        points = all_layer_tension_points.get(layer_idx, [])
        if not points:
            continue
        magnitudes = [p.norm().item() for p in points]
        non_zero = [i for i, m in enumerate(magnitudes) if m > 1e-10]
        result.lateral_coverage[layer_idx] = len(non_zero) / seq_len if seq_len > 0 else 0.0
        if not non_zero:
            result.offset_magnitude[layer_idx] = 0.0
            result.offset_consistency[layer_idx] = 0.0
            result.offset_variance[layer_idx] = 0.0
            continue
        active_points = torch.stack([points[i] for i in non_zero])
        mean_offset = active_points.mean(dim=0)
        M = mean_offset.norm().item()
        result.offset_magnitude[layer_idx] = M
        mean_mag = np.mean([magnitudes[i] for i in non_zero])
        result.offset_consistency[layer_idx] = M / mean_mag if mean_mag > 0 else 0.0
        result.offset_variance[layer_idx] = float(np.var([magnitudes[i] for i in non_zero]))

    if monitored:
        result.mean_M = np.mean([result.offset_magnitude.get(l, 0.0) for l in monitored])
        result.mean_C = np.mean([result.offset_consistency.get(l, 0.0) for l in monitored])
        result.mean_V = np.mean([result.offset_variance.get(l, 0.0) for l in monitored])
        result.mean_L = np.mean([result.lateral_coverage.get(l, 0.0) for l in monitored])

    # Per-token Peak Rank Concentration (PRC)
    # PRC = max(normalized_profile) - 1/k for each token
    # Measures directional structure: 0 = flat, >0 = directional preference
    PRC_THRESHOLD = 0.02  # ~1.6pp above uniform for k=8
    k = result.k or 8
    prc_values = []
    for profile in result.profiles:
        if len(profile) >= 2:
            total = np.sum(profile)
            if total > 0:
                normed = profile / total
                prc = float(np.max(normed) - 1.0 / k)
            else:
                prc = 0.0
        else:
            prc = 0.0
        prc_values.append(prc)
    result.prc_per_token = prc_values
    result.max_prc = max(prc_values) if prc_values else 0.0
    result.n_directional = sum(1 for p in prc_values if p > PRC_THRESHOLD)

    # Diagnostic: confirm base_profiles differ from instruct profiles
    if result.base_profiles and result.profiles:
        n_bp = len(result.base_profiles)
        n_ip = len(result.profiles)
        if n_bp > 0 and n_ip > 0:
            diff = np.mean([np.sum(np.abs(result.profiles[i] - result.base_profiles[i]))
                           for i in range(min(n_bp, n_ip))])
            logger.info(f"[LTP] base_profiles: {n_bp} entries, mean |instruct - base| = {diff:.8f}"
                       f" {'(IDENTICAL — o_proj delta missing?)' if diff < 1e-10 else '(asymmetric ✓)'}")
        else:
            logger.warning(f"[LTP] base_profiles: {n_bp} entries, profiles: {n_ip} entries")
    else:
        logger.warning(f"[LTP] base_profiles EMPTY")

    _compute_dual_trajectory(result, model_manager, monitored, seq_len)
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


def _compute_dual_trajectory(result, model_manager, monitored, seq_len):
    if seq_len < 2 or not monitored:
        return
    layer_idx = monitored[0]
    h_key = f"layer_{layer_idx}_h"
    if h_key not in model_manager.activations:
        h_key = f"layer_{layer_idx}_traj_attn"
        if h_key not in model_manager.activations:
            return
    h = model_manager.activations[h_key][0].cpu().float().numpy()
    if h.shape[0] < seq_len:
        return  # Activation mismatch — skip trajectory
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


def ltp_result_to_dict(r: LTPResult) -> dict:
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
        "base_profiles": [_safe(p) for p in r.base_profiles],
        "tension_magnitudes": [_safe(m) for m in r.tension_magnitudes],
        "profile_shapes": r.profile_shapes,
        "counterfactual_tokens": r.counterfactual_tokens,
        "offset_magnitude": {str(k): _safe(v) for k, v in r.offset_magnitude.items()},
        "offset_consistency": {str(k): _safe(v) for k, v in r.offset_consistency.items()},
        "offset_variance": {str(k): _safe(v) for k, v in r.offset_variance.items()},
        "lateral_coverage": {str(k): _safe(v) for k, v in r.lateral_coverage.items()},
        "mean_M": _safe(r.mean_M), "mean_C": _safe(r.mean_C),
        "mean_V": _safe(r.mean_V), "mean_L": _safe(r.mean_L),
        "max_prc": _safe(r.max_prc), "n_directional": r.n_directional,
        "prc_per_token": [_safe(p) for p in r.prc_per_token],
        "layer_strategy": r.layer_strategy,
        "monitored_layers": r.monitored_layers,
        "k": r.k, "svd_rank": r.svd_rank, "tuned_lens": r.tuned_lens,
        "semantic_trajectory_2d": r.semantic_trajectory.tolist() if r.semantic_trajectory is not None else [],
        "tension_trajectory_2d": r.tension_trajectory.tolist() if r.tension_trajectory is not None else [],
    }
