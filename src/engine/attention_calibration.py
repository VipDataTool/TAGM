"""Attention calibration: focus/tail token selection, Q/K collection,
key-difference subspace construction, and SKOP risk scoring.

Reusable infrastructure for per-head attention analysis. Not specific
to routing ablation. Any module needing per-head attention data can
import from here.

Based on SKOP (Luo et al., arXiv:2605.06342). Installs its own hooks
(separate from ActivationCapture) for Q/K tensor capture.

Usage:
    cal = AttentionCalibrator(model, adapter, tokenizer)
    cal.calibrate(prompts, labels, progress=progress_fn)

    directions = cal.fit_directions()
    projectors, scores = cal.build_subspace_and_score(directions)
    projected = apply_skop_projection(directions, projectors)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch

from src.engine import config as engine_config
from src.core.locks import MODEL_LOCK

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from src.core.adapter.base import ModelAdapter

logger = logging.getLogger("src")


# ── Focus/tail computation ─────────────────────────────────────────

def compute_focus_tail(
    attn_weights: np.ndarray,
    tau: float = 0.80,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per-head focus and tail index sets from attention weights.

    For each head, computes the total incoming attention each token
    receives (summed across query positions), sorts by descending
    mass, and takes the minimal set exceeding tau as the focus set.
    The complement is the tail set.

    Args:
        attn_weights: [n_heads, seq_len, seq_len] attention matrix.
            Entry [h, q, k] is the attention weight from query q to key k.
        tau: attention mass threshold for focus set (default 0.80).

    Returns:
        (focus_by_head, tail_by_head) where each maps
        head_index -> array of token indices.
    """
    n_heads, seq_len, _ = attn_weights.shape
    focus_by_head = {}
    tail_by_head = {}

    for h in range(n_heads):
        # Total incoming attention per key position (summed over queries)
        incoming = attn_weights[h].sum(axis=0)  # [seq_len]
        total = incoming.sum()
        if total < 1e-12:
            focus_by_head[h] = np.arange(seq_len)
            tail_by_head[h] = np.array([], dtype=np.int64)
            continue

        # Sort by descending attention mass
        order = np.argsort(-incoming)
        cumulative = np.cumsum(incoming[order]) / total

        # Minimal set exceeding tau
        n_focus = int(np.searchsorted(cumulative, tau)) + 1
        n_focus = min(n_focus, seq_len)

        focus_by_head[h] = order[:n_focus].copy()
        tail_by_head[h] = order[n_focus:].copy()

    return focus_by_head, tail_by_head


# ── Key-difference subspace ────────────────────────────────────────

def build_key_difference_subspace(
    k_focus: np.ndarray,
    k_tail: np.ndarray,
    gamma: float = 0.90,
    max_pairs: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the SKOP key-difference second-moment matrix and projector.

    Samples (focus, tail) pairs, computes delta_k = k_focus - k_tail,
    builds the second-moment matrix (not centered, per SKOP convention),
    and selects the top-p eigenvectors by cumulative energy >= gamma.

    Args:
        k_focus: [n_focus, d_head] key vectors at focus positions.
        k_tail: [n_tail, d_head] key vectors at tail positions.
        gamma: energy coverage threshold (default 0.90).
        max_pairs: maximum number of (focus, tail) pairs to sample.
        seed: random seed for pair sampling.

    Returns:
        projector: [d_head, d_head] matrix P = I - U_p U_p^T
        sigma: [d_head, d_head] second-moment matrix (for Rayleigh)
        p: number of retained eigenvectors
    """
    if len(k_focus) == 0 or len(k_tail) == 0:
        d = k_focus.shape[1] if len(k_focus) > 0 else k_tail.shape[1]
        return np.eye(d, dtype=np.float32), np.zeros((d, d), dtype=np.float32), 0

    d_head = k_focus.shape[1]
    rng = np.random.default_rng(seed)

    n_pairs = min(max_pairs, len(k_focus) * len(k_tail))
    focus_idx = rng.choice(len(k_focus), size=n_pairs, replace=True)
    tail_idx = rng.choice(len(k_tail), size=n_pairs, replace=True)

    deltas = k_focus[focus_idx].astype(np.float64) - k_tail[tail_idx].astype(np.float64)

    # Second-moment matrix (not centered per SKOP convention:
    # "the mean key-difference is itself a high-energy direction
    # that centring would discard")
    sigma = (deltas.T @ deltas) / len(deltas)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    # Sort descending
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Smallest p such that cumulative energy >= gamma
    total_energy = eigenvalues.sum()
    if total_energy < 1e-12:
        return np.eye(d_head, dtype=np.float32), sigma.astype(np.float32), 0

    cumulative = np.cumsum(eigenvalues) / total_energy
    above = np.where(cumulative >= gamma)[0]
    p = int(above[0]) + 1 if len(above) > 0 else len(eigenvalues)
    p = max(1, min(p, d_head))

    # Projector: P = I - U_p U_p^T
    U_p = eigenvectors[:, :p]  # [d_head, p]
    projector = np.eye(d_head, dtype=np.float64) - U_p @ U_p.T

    return projector.astype(np.float32), sigma.astype(np.float32), p


# ── Rayleigh quotient ──────────────────────────────────────────────

def rayleigh_quotient(
    direction: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """SKOP Rayleigh quotient: R = (r^T Sigma r) / (||r||^2 + eps).

    High R means the direction strongly overlaps with the key-difference
    subspace, meaning projecting it out would significantly change
    attention routing. These are the heads to intervene on.
    """
    v = direction.astype(np.float64)
    norm_sq = float(np.dot(v, v))
    return float(v @ sigma.astype(np.float64) @ v) / (norm_sq + eps)


# ── SKOP projection ───────────────────────────────────────────────

def apply_skop_projection(
    per_head_directions: dict[tuple[int, int], np.ndarray],
    subspace_projectors: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """Project each per-head direction through the SKOP key-difference
    subspace projector, removing components that would reroute
    attention away from utility-critical tokens.

    Args:
        per_head_directions: (layer, head) -> [d_head] direction vector.
        subspace_projectors: (layer, head) -> [d_head, d_head] projector.

    Returns:
        New dict of projected and renormalized direction vectors.
    """
    projected = {}
    for key, direction in per_head_directions.items():
        projector = subspace_projectors.get(key)
        if projector is None:
            projected[key] = direction
            continue

        d = projector.astype(np.float64) @ direction.astype(np.float64)
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            projected[key] = (d / norm).astype(np.float32)
        else:
            # Direction lies entirely in the key-difference subspace;
            # projecting it out leaves nothing. Skip this head.
            logger.debug(f"[SKOP] Head {key}: direction collapsed after "
                         f"projection, skipping.")

    return projected


# ── Focus-to-tail mass shift ───────────────────────────────────────

def compute_delta_m(
    attn_before: np.ndarray,
    attn_after: np.ndarray,
    focus_sets: dict[int, np.ndarray],
) -> dict[int, float]:
    """Per-head focus-to-tail attention mass shift.

    Positive delta_m means attention mass moved from focus to tail
    (routing was disrupted). Near zero on benign prompts means the
    intervention is appropriately targeted.

    Args:
        attn_before: [n_heads, seq, seq] attention weights before intervention.
        attn_after: [n_heads, seq, seq] attention weights after intervention.
        focus_sets: head_index -> array of focus token indices.

    Returns:
        head_index -> delta_m (float). Positive = mass left focus set.
    """
    n_heads = attn_before.shape[0]
    result = {}

    for h in range(n_heads):
        focus = focus_sets.get(h)
        if focus is None or len(focus) == 0:
            result[h] = 0.0
            continue

        # Mean attention mass on focus tokens (summed over query positions)
        focus_mass_before = attn_before[h, :, focus].sum() / attn_before[h].sum().clip(1e-12)
        focus_mass_after = attn_after[h, :, focus].sum() / attn_after[h].sum().clip(1e-12)

        result[h] = float(focus_mass_before - focus_mass_after)

    return result


# ── Main calibrator ────────────────────────────────────────────────

class AttentionCalibrator:
    """Collects per-head Q, K, and attention data for SKOP-style analysis.

    Installs its own hooks on q_proj and k_proj (separate from
    ActivationCapture). Uses the adapter for model-family-agnostic
    hook resolution.

    Workflow:
        1. calibrate() - forward passes on labeled prompts
        2. fit_directions() - difference-of-means from focus-restricted Q
        3. build_subspace_and_score() - key-diff subspace + Rayleigh scores
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        adapter: "ModelAdapter",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer

        self.n_layers = adapter.n_layers(model)
        self.n_q_heads, self.n_kv_heads = adapter.attention_heads(model)
        self.d_head = adapter.head_dim(model)
        self.gqa_ratio = self.n_q_heads // self.n_kv_heads

        # Accumulated data (populated by calibrate())
        self._focus_sets: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        self._tail_sets: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

        # Running sums for Q direction fitting (focus-restricted)
        self._q_harm_sums: dict[tuple[int, int], np.ndarray] = {}
        self._q_safe_sums: dict[tuple[int, int], np.ndarray] = {}
        self._q_harm_counts: dict[tuple[int, int], int] = defaultdict(int)
        self._q_safe_counts: dict[tuple[int, int], int] = defaultdict(int)

        # Key activations at focus/tail for subspace construction
        self._k_focus_harm: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        self._k_focus_safe: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        self._k_tail_all: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

        # Aggregated focus sets (computed after calibration)
        self._agg_focus: dict[tuple[int, int], np.ndarray] = {}
        self._agg_tail: dict[tuple[int, int], np.ndarray] = {}

        self._calibrated = False
        self._focus_tau = 0.80

    def calibrate(
        self,
        prompts: list[str],
        labels: list[str],
        tau: float = 0.80,
        progress: Optional[Callable] = None,
    ) -> None:
        """Run calibration forward passes.

        Args:
            prompts: calibration prompts.
            labels: per-prompt label, either "harm" or "safe".
            tau: attention mass threshold for focus sets (default 0.80).
            progress: optional progress callback.
        """
        self._focus_tau = tau
        device = next(self.model.parameters()).device

        # Initialize running sums
        for li in range(self.n_layers):
            for hi in range(self.n_q_heads):
                self._q_harm_sums[(li, hi)] = np.zeros(self.d_head, dtype=np.float64)
                self._q_safe_sums[(li, hi)] = np.zeros(self.d_head, dtype=np.float64)

        with MODEL_LOCK:
            for pi, (prompt, label) in enumerate(zip(prompts, labels)):
                if progress and pi % 10 == 0:
                    progress(f"Calibration: {pi+1}/{len(prompts)}")

                captured_q = {}
                captured_k = {}

                # Install hooks on q_proj and k_proj per layer
                hooks = []
                for li in range(self.n_layers):
                    layer_module = self.model.model.layers[li].self_attn

                    def _make_q_hook(layer_idx):
                        def hook(module, input, output):
                            captured_q[layer_idx] = output.detach().float().cpu()
                        return hook

                    def _make_k_hook(layer_idx):
                        def hook(module, input, output):
                            captured_k[layer_idx] = output.detach().float().cpu()
                        return hook

                    hooks.append(layer_module.q_proj.register_forward_hook(
                        _make_q_hook(li)))
                    hooks.append(layer_module.k_proj.register_forward_hook(
                        _make_k_hook(li)))

                # Forward pass with attention weights
                inputs = self.tokenizer(
                    prompt, return_tensors="pt",
                    add_special_tokens=engine_config.get("add_special_tokens"),
                ).to(device)

                with torch.no_grad():
                    output = self.model(
                        **inputs, output_attentions=True,
                    )

                # Remove hooks immediately
                for h in hooks:
                    h.remove()

                # Process each layer
                for li in range(self.n_layers):
                    q_tensor = captured_q.get(li)
                    k_tensor = captured_k.get(li)
                    attn = None

                    # Get attention weights from model output
                    if (hasattr(output, 'attentions') and
                            output.attentions is not None and
                            li < len(output.attentions) and
                            output.attentions[li] is not None):
                        attn = output.attentions[li][0].cpu().numpy()
                        # attn shape: [n_q_heads, seq, seq]

                    if q_tensor is None or attn is None:
                        continue

                    seq_len = q_tensor.shape[1]

                    # Reshape Q: [1, seq, n_q_heads * d_head] -> [seq, n_q_heads, d_head]
                    q = q_tensor[0].numpy().reshape(seq_len, self.n_q_heads, self.d_head)

                    # Reshape K if available: [1, seq, n_kv_heads * d_head] -> [seq, n_kv_heads, d_head]
                    k = None
                    if k_tensor is not None:
                        k = k_tensor[0].numpy().reshape(seq_len, self.n_kv_heads, self.d_head)

                    # Compute focus/tail sets from attention weights
                    focus_by_head, tail_by_head = compute_focus_tail(attn, tau=tau)

                    # Process each Q head
                    for hi in range(self.n_q_heads):
                        focus_idx = focus_by_head.get(hi, np.arange(seq_len))
                        tail_idx = tail_by_head.get(hi, np.array([], dtype=np.int64))

                        # Store focus/tail sets for aggregation
                        self._focus_sets[(li, hi)].append(focus_idx)
                        self._tail_sets[(li, hi)].append(tail_idx)

                        # Focus-restricted Q activation (mean over focus tokens)
                        if len(focus_idx) > 0:
                            q_focus_mean = q[focus_idx, hi, :].mean(axis=0)
                        else:
                            q_focus_mean = q[:, hi, :].mean(axis=0)

                        if label == "harm":
                            self._q_harm_sums[(li, hi)] += q_focus_mean
                            self._q_harm_counts[(li, hi)] += 1
                        else:
                            self._q_safe_sums[(li, hi)] += q_focus_mean
                            self._q_safe_counts[(li, hi)] += 1

                        # Collect K at focus/tail positions for subspace
                        # Map Q head to its KV head for GQA
                        kv_hi = hi // self.gqa_ratio
                        if k is not None:
                            if len(focus_idx) > 0:
                                k_f = k[focus_idx, kv_hi, :].astype(np.float32)
                                if label == "harm":
                                    self._k_focus_harm[(li, hi)].append(k_f)
                                else:
                                    self._k_focus_safe[(li, hi)].append(k_f)

                            if len(tail_idx) > 0:
                                k_t = k[tail_idx, kv_hi, :].astype(np.float32)
                                self._k_tail_all[(li, hi)].append(k_t)

                # Free tensors
                del captured_q, captured_k

        # Aggregate focus sets across prompts (union of per-prompt focus sets)
        for (li, hi), focus_lists in self._focus_sets.items():
            if focus_lists:
                all_focus = np.unique(np.concatenate(focus_lists))
                self._agg_focus[(li, hi)] = all_focus

        self._calibrated = True
        if progress:
            progress(f"Calibration complete: {len(prompts)} prompts, "
                     f"{self.n_layers} layers, {self.n_q_heads} Q heads")

    def fit_directions(self) -> dict[tuple[int, int], np.ndarray]:
        """Fit per-head routing directions from focus-restricted Q activations.

        Returns:
            (layer, head) -> unit direction vector [d_head].
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() first.")

        directions = {}
        for li in range(self.n_layers):
            for hi in range(self.n_q_heads):
                nh = self._q_harm_counts[(li, hi)]
                ns = self._q_safe_counts[(li, hi)]
                if nh == 0 or ns == 0:
                    continue

                harm_mean = self._q_harm_sums[(li, hi)] / nh
                safe_mean = self._q_safe_sums[(li, hi)] / ns
                direction = harm_mean - safe_mean
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    directions[(li, hi)] = (direction / norm).astype(np.float32)

        logger.info(f"[Calibration] Fitted {len(directions)} per-head "
                    f"routing directions")
        return directions

    def build_subspace_and_score(
        self,
        directions: dict[tuple[int, int], np.ndarray],
        gamma: float = 0.90,
        max_pairs: int = 500,
    ) -> tuple[dict[tuple[int, int], np.ndarray],
               dict[tuple[int, int], float]]:
        """Build per-head key-difference subspace projectors and compute
        Rayleigh quotient risk scores.

        Args:
            directions: per-head directions from fit_directions().
            gamma: energy coverage threshold for subspace (default 0.90).
            max_pairs: max sampled (focus, tail) pairs per head.

        Returns:
            (projectors, risk_scores) where:
            - projectors: (layer, head) -> [d_head, d_head] projector P
            - risk_scores: (layer, head) -> Rayleigh quotient R
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() first.")

        projectors = {}
        risk_scores = {}

        for (li, hi), direction in directions.items():
            # Concatenate collected K vectors
            k_focus_parts = (self._k_focus_harm.get((li, hi), []) +
                             self._k_focus_safe.get((li, hi), []))
            k_tail_parts = self._k_tail_all.get((li, hi), [])

            if not k_focus_parts or not k_tail_parts:
                projectors[(li, hi)] = np.eye(self.d_head, dtype=np.float32)
                risk_scores[(li, hi)] = 0.0
                continue

            k_focus = np.concatenate(k_focus_parts, axis=0)
            k_tail = np.concatenate(k_tail_parts, axis=0)

            projector, sigma, p = build_key_difference_subspace(
                k_focus, k_tail, gamma=gamma, max_pairs=max_pairs,
            )
            projectors[(li, hi)] = projector
            risk_scores[(li, hi)] = rayleigh_quotient(direction, sigma)

        logger.info(f"[Calibration] Built subspace projectors for "
                    f"{len(projectors)} heads (gamma={gamma})")
        return projectors, risk_scores

    def get_focus_sets_for_layer(
        self, layer_idx: int,
    ) -> dict[int, np.ndarray]:
        """Return aggregated focus sets for all heads at a given layer.

        Useful for delta_m measurement after intervention.
        """
        result = {}
        for (li, hi), focus in self._agg_focus.items():
            if li == layer_idx:
                result[hi] = focus
        return result


# ── Thin SFD context for persistence checks ───────────────────────

class SFDContext:
    """Minimal interface satisfying precompute_sfd_cache(analyzer=...).

    Avoids importing the full Analyzer. Delegates to pipeline/adapter
    for the four properties precompute_sfd_cache needs.
    """

    def __init__(self, pipeline):
        model = pipeline.instruct_model
        self.delta_store = pipeline.delta_store
        self.n_layers = pipeline.adapter.n_layers(model)
        self.hidden_size = pipeline.adapter.hidden_size(model)
        # Use config layer range (sfd_use_signal_layers=False is default)
        self.signal_layers = []


def compute_sfd_persistence(
    pipeline,
    pre_sfd_cache: dict,
    progress: Optional[Callable] = None,
) -> dict:
    """Recompute SFD cache on the current model state and measure
    cosine similarity with the pre-intervention SFD directions.

    Args:
        pipeline: TAGM pipeline (provides model, adapter, delta_store).
        pre_sfd_cache: SFD cache from before intervention.
        progress: optional progress callback.

    Returns:
        {
            "mean_cosine": float,
            "per_layer_cosine": {layer_idx: float},
            "post_k": int,
            "signal_survived": bool,  # mean_cosine > 0.80
        }
    """
    from src.engine.sfd import precompute_sfd_cache

    if progress:
        progress("Recomputing SFD cache for persistence check...")

    ctx = SFDContext(pipeline)
    post_cache = precompute_sfd_cache(ctx)

    pre_layers = pre_sfd_cache.get("layers", {})
    post_layers = post_cache.get("layers", {})

    cosines = {}
    for li in pre_layers:
        if li not in post_layers:
            continue
        pre_v = pre_layers[li].V_k  # [k, d_model]
        post_v = post_layers[li].V_k

        # Compare the top singular vectors (use the first one as
        # the primary direction proxy)
        if len(pre_v) > 0 and len(post_v) > 0:
            # Cosine between the first (dominant) right singular vector
            v1 = pre_v[0] / (np.linalg.norm(pre_v[0]) + 1e-12)
            v2 = post_v[0] / (np.linalg.norm(post_v[0]) + 1e-12)
            cos = float(np.dot(v1, v2))
            cosines[li] = abs(cos)  # sign is arbitrary for SVD vectors

    mean_cos = float(np.mean(list(cosines.values()))) if cosines else 0.0

    return {
        "mean_cosine": round(mean_cos, 4),
        "per_layer_cosine": {str(k): round(v, 4) for k, v in cosines.items()},
        "post_k": post_cache.get("k", 0),
        "signal_survived": mean_cos > 0.80,
    }
