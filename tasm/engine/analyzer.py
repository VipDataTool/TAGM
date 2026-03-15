"""
Core ASM Analyzer: single-pass computation of amplitude trajectories,
signed attribution, distribution metrics, behavioral divergence,
model response capture, and Lateral Tension Profile (LTP).
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from engine.ltp import (LTPResult, compute_ltp, ltp_result_to_dict,
                        precompute_svd_cache, precompute_tuned_lens_cache)


@dataclass
class PromptResult:
    """Complete analysis result for a single prompt."""
    prompt: str = ""
    category: str = ""
    tokens: list = field(default_factory=list)
    seq_len: int = 0

    # Stage 1: Full amplitude trajectory (all sublayers)
    amplitude_trajectory: list = field(default_factory=list)
    amplitude_normalized: list = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None

    # Stage 2: Focused stress score (signal layers only)
    stress_score: float = 0.0
    per_token_stress: Optional[np.ndarray] = None

    # Stage 3: Signed attribution
    signed_attr: Optional[np.ndarray] = None
    net_correction: float = 0.0
    n_negative_tokens: int = 0
    has_negative_tokens: bool = False

    # Stage 3 per-layer detail (layer_idx -> np.array of per-token attr)
    per_layer_signed_attr: dict = field(default_factory=dict)

    # Stage 3b: Distribution metrics
    entropy: float = 0.0
    gini: float = 0.0
    top2_share: float = 0.0
    middle_share: float = 0.0
    interior_cv: float = 0.0

    # Length-normalized metrics
    entropy_ln: Optional[float] = None
    top2_share_ln: Optional[float] = None
    middle_share_ln: Optional[float] = None
    stress_score_ln: Optional[float] = None

    # Behavioral divergence
    kl_divergence: Optional[float] = None

    # Model responses (top-k next token predictions)
    instruct_topk: list = field(default_factory=list)  # [(token, prob), ...]
    base_topk: list = field(default_factory=list)

    # Proof 1 exactness checks
    proof1_checks: list = field(default_factory=list)  # [{layer, head, sum, norm, error}]

    # Signal layer breakdown
    signal_layer_indices: list = field(default_factory=list)
    per_layer_amplitude: dict = field(default_factory=dict)

    # LTP results
    ltp: Optional[LTPResult] = None


class Analyzer:
    def __init__(self, model_manager):
        self.mm = model_manager
        # LTP enhancement caches (lazily initialized, persist for model lifetime)
        self._svd_caches = {}    # {rank: {layer_idx: tensor}}
        self._tuned_lens_cache = None  # {layer_idx: tensor} or None

    def _get_svd_cache(self, rank: int):
        """Get or build SVD cache for the given rank."""
        if rank not in self._svd_caches:
            self._svd_caches[rank] = precompute_svd_cache(self.mm.state, rank=rank)
        return self._svd_caches[rank]

    def _get_tuned_lens_cache(self, layer_indices=None):
        """Get or build tuned-lens cache (calibrated once, reused)."""
        if self._tuned_lens_cache is None:
            from engine.baselines import get_baseline_prompts
            cal_prompts = get_baseline_prompts()
            if not cal_prompts:
                cal_prompts = ["The quick brown fox jumps over the lazy dog.",
                               "What is the capital of France?",
                               "How does photosynthesis work?"]
            self._tuned_lens_cache = precompute_tuned_lens_cache(
                self.mm, cal_prompts, layer_indices=layer_indices)
        return self._tuned_lens_cache

    def clear_ltp_caches(self):
        """Clear LTP caches (e.g. on model reload)."""
        self._svd_caches.clear()
        self._tuned_lens_cache = None

    def analyze_prompt(self, prompt: str, category: str = "",
                       compute_kl: bool = False,
                       compute_full_trajectory: bool = True,
                       capture_responses: bool = False,
                       compute_ltp: bool = False,
                       ltp_k: int = 8,
                       ltp_layer_strategy: str = "signal",
                       ltp_svd_rank: int = 0,
                       ltp_tuned_lens: bool = False,
                       response_topk: int = 10) -> PromptResult:
        """Run the full analysis pipeline in a SINGLE forward pass."""
        state = self.mm.state
        result = PromptResult(prompt=prompt, category=category)
        result.signal_layer_indices = list(state.signal_layers)

        # Install all hooks, run one forward pass
        # Determine which extra layers LTP needs (if any)
        ltp_layers = None
        if compute_ltp and ltp_layer_strategy == "late":
            late_start = 2 * state.n_layers // 3
            ltp_layers = list(range(late_start, state.n_layers))
        self.mm.install_analysis_hooks(full_trajectory=compute_full_trajectory,
                                       ltp_layers=ltp_layers)
        tokens, inputs, model_out = self.mm.forward(prompt, output_attentions=True)

        result.tokens = [t.replace("\u0120", " ").replace("\u010a", "\\n") for t in tokens]
        result.seq_len = len(tokens)
        seq_len = result.seq_len

        # Capture instruct model top-k predictions from this forward pass
        if capture_responses or compute_kl:
            logits = model_out.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, min(response_topk, probs.shape[0]))
            result.instruct_topk = [
                (state.tokenizer.decode(idx.item()).strip(), round(p.item(), 4))
                for idx, p in zip(topk.indices, topk.values)
            ]

        # Extract all metrics from cached activations
        self._extract_signed_attribution(result, seq_len, state)
        self._extract_stress_score(result, seq_len, state)
        if compute_full_trajectory:
            self._extract_amplitude_trajectory(result, seq_len, state)

        # LTP computation (uses cached activations + logits from the same pass)
        if compute_ltp:
            result.ltp = self._compute_ltp(
                model_out.logits, tokens, inputs["input_ids"],
                k=ltp_k, layer_strategy=ltp_layer_strategy,
                svd_rank=ltp_svd_rank, use_tuned_lens=ltp_tuned_lens)

        # Free activations and hooks before KL/response pass
        self.mm.clear_activations()
        self.mm._remove_hooks()

        # KL divergence + base model responses (separate pass)
        if compute_kl or capture_responses:
            self._compute_behavioral_comparison(
                result, state, compute_kl=compute_kl,
                capture_base=(capture_responses and state.model_base is not None),
                topk=response_topk)

        return result

    def _compute_ltp(self, logits, tokens, input_ids,
                     k: int = 8, layer_strategy: str = "signal",
                     svd_rank: int = 0, use_tuned_lens: bool = False) -> LTPResult:
        """Compute LTP using cached activations from the forward pass."""
        from engine.ltp import compute_ltp as _compute_ltp

        # Build caches on demand (lazy, persist for model lifetime)
        svd_cache = self._get_svd_cache(svd_rank) if svd_rank > 0 else None
        tl_cache = self._get_tuned_lens_cache() if use_tuned_lens else None

        return _compute_ltp(
            self.mm, logits, tokens, input_ids,
            k=k, layer_strategy=layer_strategy,
            svd_cache=svd_cache, svd_rank=svd_rank,
            tuned_lens_cache=tl_cache)

    def _extract_signed_attribution(self, result, seq_len, state):
        """Extract signed attribution with per-layer detail and proof checks."""
        n_kv_heads = state.n_kv_heads
        head_dim = state.head_dim
        n_heads = state.n_heads
        heads_per_kv = n_heads // n_kv_heads

        layer_attrs = []

        for layer_idx in state.signal_layers:
            h_key = f"layer_{layer_idx}_h"
            a_key = f"layer_{layer_idx}_attn"

            if h_key not in self.mm.activations or a_key not in self.mm.attn_weights:
                continue

            h = self.mm.activations[h_key][0]
            alpha = self.mm.attn_weights[a_key][0]
            dw_v = state.v_delta(layer_idx)
            if dw_v is None:
                continue

            v = torch.matmul(h, dw_v.T)
            v_heads = v.view(seq_len, n_kv_heads, head_dim)

            alpha_grouped = alpha.view(n_kv_heads, heads_per_kv, seq_len, seq_len)
            alpha_kv = alpha_grouped.mean(dim=1)

            head_attrs = []
            layer_amp = 0.0
            for kv_head in range(n_kv_heads):
                v_h = v_heads[:, kv_head, :]
                a_h = alpha_kv[kv_head]
                delta = torch.matmul(a_h, v_h)
                d_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                u_hat = delta / d_norm
                proj = torch.matmul(u_hat, v_h.T)
                signed = a_h * proj
                head_attrs.append(signed[-1, :])
                layer_amp += d_norm[-1].item()

                # Proof 1 exactness: sum of signed attr should equal ||delta||
                attr_sum = signed[-1, :].sum().item()
                delta_norm_val = d_norm[-1].item()
                error = abs(attr_sum - delta_norm_val)
                result.proof1_checks.append({
                    "layer": layer_idx, "head": kv_head,
                    "attr_sum": round(attr_sum, 8),
                    "delta_norm": round(delta_norm_val, 8),
                    "error": float(f"{error:.2e}"),
                    "exact": error < 1e-4,
                })

            layer_attr = torch.stack(head_attrs).mean(dim=0)
            layer_attrs.append(layer_attr)
            result.per_layer_amplitude[layer_idx] = layer_amp / n_kv_heads
            result.per_layer_signed_attr[layer_idx] = layer_attr.numpy().tolist()

        if not layer_attrs:
            return

        avg_attr = torch.stack(layer_attrs).mean(dim=0).numpy()
        result.signed_attr = avg_attr
        result.net_correction = float(avg_attr.sum())
        result.n_negative_tokens = int(sum(1 for a in avg_attr if a < 0))
        result.has_negative_tokens = result.n_negative_tokens > 0

        # Distribution metrics
        attr_abs = np.abs(avg_attr)
        total = attr_abs.sum()
        attr_dist = attr_abs / total if total > 0 else np.ones_like(attr_abs) / len(attr_abs)

        ent = -np.sum(attr_dist * np.log(attr_dist + 1e-10))
        max_ent = np.log(seq_len) if seq_len > 1 else 1.0
        result.entropy = float(ent / max_ent)

        sorted_d = np.sort(attr_dist)
        n = len(sorted_d)
        cum = np.cumsum(sorted_d)
        result.gini = float(1 - 2 * cum.sum() / (n * sorted_d.sum())) if sorted_d.sum() > 0 else 0.0

        if seq_len >= 2:
            result.top2_share = float(attr_dist[0] + attr_dist[-1])
        else:
            result.top2_share = 1.0

        if seq_len > 2:
            result.middle_share = float(attr_dist[1:-1].sum())
            interior = attr_dist[1:-1]
            imean = interior.mean()
            result.interior_cv = float(interior.std() / imean) if imean > 0 else 0.0
        else:
            result.middle_share = 0.0
            result.interior_cv = 0.0

    def _extract_stress_score(self, result, seq_len, state):
        per_token_total = torch.zeros(seq_len)
        n_layers = 0

        for layer_idx in state.signal_layers:
            key = f"layer_{layer_idx}_h"
            if key not in self.mm.activations:
                continue
            h = self.mm.activations[key]

            for p in ["q", "k", "v"]:
                dname = f"model.layers.{layer_idx}.self_attn.{p}_proj.weight"
                if dname in state.deltas:
                    dw = state.deltas[dname]
                    fnorm = state.delta_frob_norms[dname]
                    if dw.shape[1] == h.shape[2] and fnorm > 0:
                        projected = torch.matmul(h[0], dw.T)
                        per_token_total += projected.norm(dim=-1) / fnorm
            n_layers += 1

        if n_layers > 0:
            per_token_total /= n_layers

        result.stress_score = float(per_token_total.mean().item())
        result.per_token_stress = per_token_total.numpy()

    def _extract_amplitude_trajectory(self, result, seq_len, state):
        raw_traj = []
        norm_traj = []
        heatmap_rows = []

        for layer_idx in range(state.n_layers):
            for sublayer_type in ["attn", "mlp"]:
                key = f"layer_{layer_idx}_traj_{sublayer_type}"
                if key not in self.mm.activations:
                    raw_traj.append(0.0)
                    norm_traj.append(0.0)
                    heatmap_rows.append(np.zeros(seq_len))
                    continue

                h = self.mm.activations[key]
                if sublayer_type == "attn":
                    dnames = [f"model.layers.{layer_idx}.self_attn.{p}_proj.weight"
                              for p in ["q", "k", "v"]]
                else:
                    dnames = [f"model.layers.{layer_idx}.mlp.{p}_proj.weight"
                              for p in ["gate", "up"]]

                raw_sum = 0.0
                norm_sum = 0.0
                per_tok = torch.zeros(seq_len)

                for dname in dnames:
                    if dname in state.deltas:
                        dw = state.deltas[dname]
                        fnorm = state.delta_frob_norms[dname]
                        if dw.shape[1] == h.shape[2] and fnorm > 0:
                            projected = torch.matmul(h[0], dw.T)
                            pn = projected.norm(dim=-1)
                            raw_sum += pn.mean().item()
                            norm_sum += (pn / fnorm).mean().item()
                            per_tok += pn / fnorm

                raw_traj.append(raw_sum)
                norm_traj.append(norm_sum)
                heatmap_rows.append(per_tok.numpy())

        result.amplitude_trajectory = raw_traj
        result.amplitude_normalized = norm_traj
        result.heatmap = np.array(heatmap_rows)

    def _compute_behavioral_comparison(self, result, state,
                                        compute_kl=False,
                                        capture_base=False,
                                        topk=10):
        """KL divergence and base model predictions in a single base-model pass."""
        inputs = state.tokenizer(result.prompt, return_tensors="pt").to(state.device)
        with torch.no_grad():
            logits_inst = state.model_instruct(**inputs).logits[0, -1, :]

            if state.model_base is not None:
                logits_base = state.model_base(**inputs).logits[0, -1, :]

                if compute_kl:
                    log_p_inst = torch.log_softmax(logits_inst, dim=-1)
                    log_p_base = torch.log_softmax(logits_base, dim=-1)
                    p_inst = torch.softmax(logits_inst, dim=-1)
                    result.kl_divergence = float(
                        (p_inst * (log_p_inst - log_p_base)).sum().item())

                if capture_base:
                    probs_base = torch.softmax(logits_base, dim=-1)
                    tk = torch.topk(probs_base, min(topk, probs_base.shape[0]))
                    result.base_topk = [
                        (state.tokenizer.decode(idx.item()).strip(), round(p.item(), 4))
                        for idx, p in zip(tk.indices, tk.values)
                    ]


def result_to_dict(r: PromptResult) -> dict:
    """Serialize a PromptResult for JSON transport."""
    def _native(v):
        if v is None:
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
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
            return [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x
                    for x in v.tolist()]
        if hasattr(v, 'item'):
            val = v.item()
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            return val
        return v

    d = {
        "prompt": r.prompt,
        "category": r.category,
        "tokens": r.tokens,
        "seq_len": r.seq_len,
        "stress_score": _native(r.stress_score),
        "per_token_stress": r.per_token_stress.tolist() if r.per_token_stress is not None else [],
        "signed_attr": r.signed_attr.tolist() if r.signed_attr is not None else [],
        "net_correction": _native(r.net_correction),
        "n_negative_tokens": _native(r.n_negative_tokens),
        "has_negative_tokens": r.has_negative_tokens,
        "entropy": _native(r.entropy),
        "gini": _native(r.gini),
        "top2_share": _native(r.top2_share),
        "middle_share": _native(r.middle_share),
        "interior_cv": _native(r.interior_cv),
        "entropy_ln": _native(r.entropy_ln),
        "top2_share_ln": _native(r.top2_share_ln),
        "middle_share_ln": _native(r.middle_share_ln),
        "stress_score_ln": _native(r.stress_score_ln),
        "kl_divergence": _native(r.kl_divergence),
        "instruct_topk": r.instruct_topk,
        "base_topk": r.base_topk,
        "proof1_checks": r.proof1_checks,
        "per_layer_signed_attr": {str(k): v for k, v in r.per_layer_signed_attr.items()},
        "amplitude_trajectory": [_native(v) for v in r.amplitude_trajectory],
        "amplitude_normalized": [_native(v) for v in r.amplitude_normalized],
        "heatmap": r.heatmap.tolist() if r.heatmap is not None else [],
        "signal_layer_indices": r.signal_layer_indices,
        "per_layer_amplitude": {str(k): _native(v) for k, v in r.per_layer_amplitude.items()},
    }

    # LTP data
    if r.ltp is not None:
        d["ltp"] = ltp_result_to_dict(r.ltp)
    else:
        d["ltp"] = None

    # Classification
    try:
        from engine.classifier import classify_to_dict
        d["classification"] = classify_to_dict(d)
    except Exception:
        d["classification"] = None

    return d
