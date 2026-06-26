"""QK routing intervention: project directions out of query vectors per head.

Extension of engine/interventions.py for attention-routing ablation.
Operates on the query tensor during the attention forward pass, projecting
out a per-head direction to remove routing-level harm signals without
touching the residual stream.

Design constraints (from literature):
  - Symmetric Q/K rotation is a no-op on attention scores (AQUA, arXiv:2509.11155).
    All interventions must be asymmetric. We operate on Q only.
  - Per-head Python hooks are 32× slower than one vectorized hook per layer.
    We precompute a [n_heads, d_head, d_head] projector tensor and apply
    it in a single batched einsum.
  - Must respect GQA: Llama 3.2-1B has 32 Q heads but 8 KV heads.
    We operate on Q heads only, so GQA is handled naturally.
  - All projection math in fp32 to avoid bf16 accumulation error;
    cast back to model dtype before returning.

Integration:
  - Hooks attach to `model.model.layers[i].self_attn` via register_forward_pre_hook
    to intercept the hidden states before Q/K/V projection, OR via a custom
    post-projection hook if the attention module exposes Q/K tensors.
  - For HuggingFace models with `attn_implementation="eager"`, we can hook
    after the Q projection and before the attention score computation.

Usage:
    qk_intv = QKRoutingIntervention()
    qk_intv.install(model, adapter, per_head_directions, risk_head_mask)
    out = model.generate(**inputs)  # Q vectors modified at risk heads
    qk_intv.remove()
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from src.core.adapter.base import ModelAdapter

logger = logging.getLogger("src")


class QKRoutingIntervention:
    """Manages query-space directional ablation at selected attention heads.

    For each risk head (ℓ, h), projects a fitted direction out of the
    query vector:  q_h ← q_h − (q_h · r̂_h) r̂_h

    Non-risk heads pass through unchanged. The projection is applied
    after Q projection but before attention score computation, using a
    precomputed per-head projector tensor for efficiency.
    """

    def __init__(self):
        self._hooks: list = []
        self._projectors: dict[int, torch.Tensor] = {}  # layer_idx -> [n_heads, d_head, d_head]
        self._installed = False

    def install(
        self,
        model: "PreTrainedModel",
        adapter: "ModelAdapter",
        per_head_directions: dict[tuple[int, int], np.ndarray],
        risk_head_mask: Optional[dict[int, list[int]]] = None,
    ) -> None:
        """Install Q-projection hooks on the model.

        Args:
            model: The loaded HuggingFace model.
            adapter: TAGM model adapter for structure introspection.
            per_head_directions: Mapping (layer_idx, head_idx) -> direction
                vector of shape [d_head], unit-normalized. Only heads with
                entries here are intervened on.
            risk_head_mask: Optional mapping layer_idx -> list of head indices
                to intervene on. If None, all heads with directions are used.
        """
        self.remove()

        n_layers = adapter.n_layers(model)
        n_heads, _ = adapter.attention_heads(model)
        d_head = adapter.head_dim(model)
        first_param = next(model.parameters())
        device, dtype = first_param.device, first_param.dtype

        # Group directions by layer
        layer_dirs: dict[int, dict[int, np.ndarray]] = {}
        for (li, hi), direction in per_head_directions.items():
            if risk_head_mask is not None and hi not in risk_head_mask.get(li, []):
                continue
            if li not in layer_dirs:
                layer_dirs[li] = {}
            layer_dirs[li][hi] = direction

        if not layer_dirs:
            logger.warning("[QK-ROUTING] No risk heads selected; no hooks installed.")
            return

        # Build per-layer projector tensors
        for li, head_dirs in layer_dirs.items():
            # Start with identity (no intervention) for all heads
            projector = torch.eye(d_head, dtype=torch.float32).unsqueeze(0).expand(
                n_heads, -1, -1).clone()

            for hi, direction in head_dirs.items():
                v = torch.tensor(direction, dtype=torch.float32)
                v = v / (v.norm() + 1e-12)
                # P = I - v v^T  (project out v)
                projector[hi] = torch.eye(d_head, dtype=torch.float32) - torch.outer(v, v)

            self._projectors[li] = projector.to(device=device)

            # Hook into the self_attn module at this layer
            attn_module = model.model.layers[li].self_attn
            hook = attn_module.register_forward_hook(
                self._make_q_projection_hook(li, n_heads, d_head, dtype)
            )
            self._hooks.append(hook)

        n_risk = sum(len(hd) for hd in layer_dirs.values())
        logger.info(
            f"[QK-ROUTING] Installed Q-projection hooks: "
            f"{len(layer_dirs)} layers, {n_risk} risk heads"
        )
        self._installed = True

    def _make_q_projection_hook(self, layer_idx: int, n_heads: int,
                                 d_head: int, model_dtype: torch.dtype):
        """Create a forward hook that modifies the attention output.

        For eager attention, the self_attn module receives hidden_states
        and produces (attn_output, attn_weights, past_key_value). We
        intercept the query computation by hooking into the module's
        internals.

        Implementation note: HuggingFace's eager attention computes
        Q = self.q_proj(hidden_states) inside forward(). We cannot
        intercept Q mid-computation with a module-level hook. Instead,
        we hook the q_proj Linear directly with a post-hook that
        modifies Q after projection but before it enters the attention
        score computation.
        """
        projector = self._projectors[layer_idx]

        def hook(module, input, output):
            """Hook on q_proj: output is the raw Q tensor [batch, seq, n_heads*d_head].

            We reshape to [batch, seq, n_heads, d_head], apply per-head
            projection, and reshape back.
            """
            if not isinstance(output, torch.Tensor):
                return output

            orig_shape = output.shape
            batch, seq_len = orig_shape[0], orig_shape[1]

            # Reshape: [batch, seq, n_heads * d_head] -> [batch, seq, n_heads, d_head]
            q = output.view(batch, seq_len, n_heads, d_head)

            # Project in fp32 for numerical stability
            q_f32 = q.float()
            # Einsum: for each head, apply its projector matrix
            # q_projected[b, s, h, :] = projector[h, :, :] @ q[b, s, h, :]
            q_projected = torch.einsum('hij,bshj->bshi', projector.float(), q_f32)

            # Cast back and reshape
            return q_projected.to(dtype=output.dtype).view(orig_shape)

        return hook

    def install_on_q_proj(
        self,
        model: "PreTrainedModel",
        adapter: "ModelAdapter",
        per_head_directions: dict[tuple[int, int], np.ndarray],
        risk_head_mask: Optional[dict[int, list[int]]] = None,
    ) -> None:
        """Install hooks directly on q_proj Linear modules.

        This is the preferred method: hooks fire on the q_proj output
        (the raw Q tensor before reshape/RoPE), giving us clean access
        to modify Q per-head before attention scores are computed.
        """
        self.remove()

        n_layers = adapter.n_layers(model)
        n_heads, _ = adapter.attention_heads(model)
        d_head = adapter.head_dim(model)
        first_param = next(model.parameters())
        device, dtype = first_param.device, first_param.dtype

        # Group directions by layer
        layer_dirs: dict[int, dict[int, np.ndarray]] = {}
        for (li, hi), direction in per_head_directions.items():
            if risk_head_mask is not None and hi not in risk_head_mask.get(li, []):
                continue
            layer_dirs.setdefault(li, {})[hi] = direction

        if not layer_dirs:
            logger.warning("[QK-ROUTING] No risk heads selected.")
            return

        for li, head_dirs in layer_dirs.items():
            projector = torch.eye(d_head, dtype=torch.float32).unsqueeze(0).expand(
                n_heads, -1, -1).clone()

            for hi, direction in head_dirs.items():
                v = torch.tensor(direction, dtype=torch.float32)
                v = v / (v.norm() + 1e-12)
                projector[hi] = torch.eye(d_head, dtype=torch.float32) - torch.outer(v, v)

            proj_device = projector.to(device=device)
            self._projectors[li] = proj_device

            # Hook onto q_proj directly
            q_proj_module = model.model.layers[li].self_attn.q_proj

            def _make_hook(proj_tensor, nh, dh, dt):
                def hook_fn(module, input, output):
                    if not isinstance(output, torch.Tensor):
                        return output
                    orig_shape = output.shape
                    batch, seq_len = orig_shape[0], orig_shape[1]
                    q = output.view(batch, seq_len, nh, dh)
                    q_f32 = q.float()
                    q_proj = torch.einsum('hij,bshj->bshi', proj_tensor.float(), q_f32)
                    return q_proj.to(dtype=dt).view(orig_shape)
                return hook_fn

            hook = q_proj_module.register_forward_hook(
                _make_hook(proj_device, n_heads, d_head, dtype)
            )
            self._hooks.append(hook)

        n_risk = sum(len(hd) for hd in layer_dirs.values())
        logger.info(
            f"[QK-ROUTING] Installed q_proj hooks: "
            f"{len(layer_dirs)} layers, {n_risk} risk heads"
        )
        self._installed = True

    def remove(self) -> None:
        """Remove all hooks and clear projectors."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._projectors.clear()
        self._installed = False

    @property
    def is_installed(self) -> bool:
        return self._installed


# ── Utility: Risk-head selection via Rayleigh quotient ──────────

def compute_risk_scores(
    model: "PreTrainedModel",
    adapter: "ModelAdapter",
    per_head_directions: dict[tuple[int, int], np.ndarray],
    key_activations_harm: dict[tuple[int, int], np.ndarray],
    key_activations_safe: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], float]:
    """Score each head by how much its routing direction overlaps with
    the key-difference subspace (SKOP Rayleigh quotient).

    High score = the direction would strongly reroute attention if
    projected out. These are the heads to intervene on.

    Args:
        per_head_directions: (layer, head) -> d_head direction vector
        key_activations_harm: (layer, head) -> [n_tokens, d_head] key
            activations from harmful prompts
        key_activations_safe: (layer, head) -> [n_tokens, d_head] key
            activations from safe prompts

    Returns:
        (layer, head) -> risk score (Rayleigh quotient)
    """
    scores = {}
    for (li, hi), direction in per_head_directions.items():
        harm_keys = key_activations_harm.get((li, hi))
        safe_keys = key_activations_safe.get((li, hi))
        if harm_keys is None or safe_keys is None:
            scores[(li, hi)] = 0.0
            continue

        # Build key-difference second-moment matrix
        # Use random pairs to keep computation tractable
        n_pairs = min(500, len(harm_keys) * len(safe_keys))
        rng = np.random.default_rng(42)
        harm_idx = rng.choice(len(harm_keys), size=n_pairs, replace=True)
        safe_idx = rng.choice(len(safe_keys), size=n_pairs, replace=True)
        deltas = harm_keys[harm_idx] - safe_keys[safe_idx]

        # Second-moment matrix (not centered — SKOP convention)
        sigma = (deltas.T @ deltas) / len(deltas)

        # Rayleigh quotient
        v = direction / (np.linalg.norm(direction) + 1e-12)
        r = float(v @ sigma @ v) / (float(np.dot(v, v)) + 1e-12)
        scores[(li, hi)] = r

    return scores


def select_risk_heads(
    scores: dict[tuple[int, int], float],
    fraction: float = 0.20,
) -> dict[int, list[int]]:
    """Select the top fraction of heads by risk score.

    Returns:
        layer_idx -> list of head indices to intervene on
    """
    if not scores:
        return {}

    sorted_heads = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n_select = max(1, int(len(sorted_heads) * fraction))
    selected = sorted_heads[:n_select]

    result: dict[int, list[int]] = {}
    for (li, hi), score in selected:
        result.setdefault(li, []).append(hi)

    return result
