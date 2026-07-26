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
  - Hooks attach to `model.model.layers[i].self_attn.q_proj` with a forward
    post-hook, so they see the raw Q tensor before reshape/RoPE and can
    modify it per head before attention scores are computed.
  - Hooking `self_attn` itself does NOT work: its forward returns a tuple,
    not a tensor, so a hook written against a q_proj output silently
    no-ops.  That was the bug in the old `install()`.

Usage:
    qk_intv = QKRoutingIntervention()
    try:
        qk_intv.install_on_q_proj(model, adapter, per_head_directions,
                                  risk_head_mask)
        out = model.generate(**inputs)  # Q vectors modified at risk heads
    finally:
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
        """Deprecated alias for install_on_q_proj().

        This method used to register its hook on `self_attn` while the hook
        body was written for a `q_proj` output.  `self_attn.forward` returns a
        tuple, so the hook's `isinstance(output, torch.Tensor)` guard was
        always False and the hook returned the output unmodified — while
        logging "Installed Q-projection hooks" and setting `_installed = True`.
        Any experiment routed through it silently measured NO intervention and
        would have been read as a genuine null result.

        It now delegates to the correct implementation.  Call
        install_on_q_proj() directly in new code.
        """
        logger.warning(
            "[QK-ROUTING] install() is deprecated and previously applied NO "
            "intervention; delegating to install_on_q_proj()."
        )
        return self.install_on_q_proj(
            model, adapter, per_head_directions, risk_head_mask)

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
