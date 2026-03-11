"""
Core ASM Analyzer: single-pass computation of amplitude trajectories,
signed attribution, distribution metrics, and behavioral divergence.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


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

    # Stage 1d: Per-token per-layer heatmap (normalized)
    heatmap: Optional[np.ndarray] = None

    # Stage 2: Focused stress score (signal layers only)
    stress_score: float = 0.0
    per_token_stress: Optional[np.ndarray] = None

    # Stage 3: Signed attribution
    signed_attr: Optional[np.ndarray] = None
    net_correction: float = 0.0
    n_negative_tokens: int = 0
    has_negative_tokens: bool = False

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

    # Signal layer breakdown
    signal_layer_indices: list = field(default_factory=list)
    per_layer_amplitude: dict = field(default_factory=dict)


class Analyzer:
    def __init__(self, model_manager):
        self.mm = model_manager

    def analyze_prompt(self, prompt: str, category: str = "",
                       compute_kl: bool = False,
                       compute_full_trajectory: bool = True) -> PromptResult:
        """
        Run the full analysis pipeline in a SINGLE forward pass.
        Installs all required hooks, runs once, extracts everything.
        """
        state = self.mm.state
        result = PromptResult(prompt=prompt, category=category)
        result.signal_layer_indices = list(state.signal_layers)

        # Install all hooks for this analysis, run one forward pass
        self.mm.install_analysis_hooks(full_trajectory=compute_full_trajectory)
        tokens, inputs, _ = self.mm.forward(prompt, output_attentions=True)

        result.tokens = [t.replace("\u0120", " ").replace("\u010a", "\\n") for t in tokens]
        result.seq_len = len(tokens)
        seq_len = result.seq_len

        # --- Extract signed attribution from signal layers ---
        self._extract_signed_attribution(result, seq_len, state)

        # --- Extract stress score from signal layers ---
        self._extract_stress_score(result, seq_len, state)

        # --- Extract full trajectory if requested ---
        if compute_full_trajectory:
            self._extract_amplitude_trajectory(result, seq_len, state)

        # --- Free activation memory and remove hooks ---
        self.mm.clear_activations()
        self.mm._remove_hooks()

        # --- KL divergence (separate pass with base model) ---
        if compute_kl:
            self._compute_kl_divergence(result, state)

        return result

    def _extract_signed_attribution(self, result, seq_len, state):
        """Extract signed attribution from already-computed activations."""
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

            layer_attr = torch.stack(head_attrs).mean(dim=0)
            layer_attrs.append(layer_attr)
            result.per_layer_amplitude[layer_idx] = layer_amp / n_kv_heads

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
        """Extract stress score from already-computed signal layer activations."""
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
        """Extract full trajectory from already-computed all-layer activations."""
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
                    dnames = [
                        f"model.layers.{layer_idx}.self_attn.{p}_proj.weight"
                        for p in ["q", "k", "v"]
                    ]
                else:
                    dnames = [
                        f"model.layers.{layer_idx}.mlp.{p}_proj.weight"
                        for p in ["gate", "up"]
                    ]

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

    def _compute_kl_divergence(self, result, state):
        """Compute KL(instruct || base) at last token position."""
        if state.model_base is None:
            return

        inputs = state.tokenizer(result.prompt, return_tensors="pt").to(state.device)
        with torch.no_grad():
            logits_inst = state.model_instruct(**inputs).logits[0, -1, :]
            logits_base = state.model_base(**inputs).logits[0, -1, :]

        log_p_inst = torch.log_softmax(logits_inst, dim=-1)
        log_p_base = torch.log_softmax(logits_base, dim=-1)
        p_inst = torch.softmax(logits_inst, dim=-1)

        kl = (p_inst * (log_p_inst - log_p_base)).sum().item()
        result.kl_divergence = kl


def result_to_dict(r: PromptResult) -> dict:
    """Serialize a PromptResult for JSON transport."""
    def _native(v):
        if v is None:
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if hasattr(v, 'item'):
            return v.item()
        return v

    return {
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
        "amplitude_trajectory": [_native(v) for v in r.amplitude_trajectory],
        "amplitude_normalized": [_native(v) for v in r.amplitude_normalized],
        "heatmap": r.heatmap.tolist() if r.heatmap is not None else [],
        "signal_layer_indices": r.signal_layer_indices,
        "per_layer_amplitude": {str(k): _native(v) for k, v in r.per_layer_amplitude.items()},
    }
