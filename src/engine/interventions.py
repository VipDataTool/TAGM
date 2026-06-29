"""Adapter-based forward hooks for activation *intervention*.

Mirror image of engine/hooks.py:ActivationCapture. Where ActivationCapture
reads activations and stores them off, ActivationIntervention *writes* to
activations in-flight by returning modified tensors from forward hooks.

The adapter still resolves where to hook; this module only chooses what
to do once the hook fires. Two primitive operations are supported:

    ablate : h' = h - alpha * (h · v) * v      (project v out)
    add    : h' = h + alpha * v                (steer along v)

Applied to hidden states at a declared hook point (typically
``residual_post_block`` at one or more layers). These are the two
operations used in Arditi et al. 2024 ("Refusal in LLMs is mediated by
a single direction"): ablation removes a candidate behavior direction
and tests whether the behavior disappears; addition adds the direction
and tests whether the behavior appears on inputs that would not
normally elicit it.

Design notes
------------
* No adapter changes required. ``ModelAdapter.resolve_hook_target``
  returns an ``nn.Module``; PyTorch's forward-hook API is symmetric
  about reads and writes. We use the same resolution path as
  ActivationCapture.
* Intervenable hook points are whitelisted here rather than tagged on
  ``HookPointSpec`` — this keeps the intervention module adapter-free.
  If ``HookPointSpec.captures`` later grows an ``intervenable`` tag,
  the whitelist can be replaced with a capability check.
* Hooks persist across forward passes, so generation ``model.generate(...)``
  runs the intervention on every newly-generated token automatically.
  No separate generate-time plumbing is needed.
* Direction tensors are unit-normalized, cast to the model's dtype, and
  moved to the model's device at install time. Callers can pass
  fp32 CPU numpy-derived tensors without worrying about placement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from src.core.adapter.base import ModelAdapter


# Hook points where returning a replacement hidden-state tensor has
# well-defined semantics for the subsequent forward pass. Attention
# weights are deliberately excluded — overwriting them produces
# unnormalized attention distributions that silently corrupt the block.
INTERVENABLE_HOOK_POINTS = frozenset({
    "pre_attn_norm",
    "post_attn_norm",
    "mlp_output",
    "residual_post_block",
    "final_norm",
})


# ── Intervention specification ──────────────────────────────────────

@dataclass
class InterventionSpec:
    """One intervention at one module location.

    Attributes
    ----------
    mode:
        ``"ablate"`` projects ``direction`` out of the hidden state;
        ``"add"`` adds ``alpha * direction`` to it.
    direction:
        1-D tensor of shape ``(hidden,)``. Normalized at install time.
    layer_idx:
        Layer index for layer-dependent hook points. ``None`` for
        layer-independent points (e.g. ``final_norm``).
    hook_point:
        Name declared in ``adapter.HOOK_POINTS``. Must be in
        ``INTERVENABLE_HOOK_POINTS``. Default is ``residual_post_block``
        — the canonical Arditi-style intervention point.
    alpha:
        For ``ablate``: projection scaling (1.0 = full removal,
        <1.0 = partial ablation, >1.0 = over-projection).
        For ``add``: coefficient on the added vector.
    """
    mode: str
    direction: torch.Tensor
    layer_idx: Optional[int] = None
    hook_point: str = "residual_post_block"
    alpha: float = 1.0

    def __post_init__(self):
        if self.mode not in ("ablate", "add"):
            raise ValueError(
                f"mode must be 'ablate' or 'add', got {self.mode!r}")
        if self.direction.ndim != 1:
            raise ValueError(
                f"direction must be 1-D, got shape "
                f"{tuple(self.direction.shape)}")


# ── Intervention manager ────────────────────────────────────────────

class ActivationIntervention:
    """Manages causal interventions on a model via forward hooks.

    Usage — Arditi-style directional ablation::

        intv = ActivationIntervention()
        intv.install(model, adapter, [
            InterventionSpec(mode="ablate", direction=v, layer_idx=19),
        ])
        out = model.generate(**inputs)   # v is projected out at L19
        intv.remove()

    Usage — directional steering (addition)::

        intv.install_addition(model, adapter, v, layer_idx=15, alpha=3.0)

    Multiple specs compose — hooks fire in the order they were installed,
    at their respective modules. To ablate at a range of layers (Arditi's
    preferred approach), pass one spec per layer.
    """

    def __init__(self):
        self._hooks: list = []
        self._specs: list[InterventionSpec] = []

    # ── Install ─────────────────────────────────────────────────────

    def install(
        self,
        model: "PreTrainedModel",
        adapter: "ModelAdapter",
        specs: list[InterventionSpec],
    ) -> None:
        """Install one or more interventions. Replaces any existing ones."""
        self.remove()

        first_param = next(model.parameters())
        device, dtype = first_param.device, first_param.dtype
        hidden = adapter.hidden_size(model)

        for spec in specs:
            if spec.hook_point not in INTERVENABLE_HOOK_POINTS:
                raise ValueError(
                    f"Hook point {spec.hook_point!r} is not intervenable. "
                    f"Allowed: {sorted(INTERVENABLE_HOOK_POINTS)}")
            if spec.hook_point not in adapter.HOOK_POINTS:
                raise KeyError(
                    f"Adapter {adapter.family_id!r} does not declare hook "
                    f"point {spec.hook_point!r}. Declared: "
                    f"{list(adapter.HOOK_POINTS)}")
            if spec.direction.shape[0] != hidden:
                raise ValueError(
                    f"direction has dim {spec.direction.shape[0]} but model "
                    f"hidden size is {hidden}")

            # Normalize and place on model's device/dtype.
            v = spec.direction.detach().to(device=device, dtype=dtype)
            v = v / (v.norm() + 1e-12)

            target = adapter.resolve_hook_target(
                model, spec.hook_point, spec.layer_idx)

            if spec.mode == "ablate":
                hook_fn = _make_ablate_hook(v, spec.alpha)
            else:  # "add"
                hook_fn = _make_add_hook(v, spec.alpha)

            self._hooks.append(target.register_forward_hook(hook_fn))
            self._specs.append(spec)

    # ── Convenience constructors ────────────────────────────────────

    def install_ablation(
        self,
        model: "PreTrainedModel",
        adapter: "ModelAdapter",
        direction: torch.Tensor,
        layer_idx: int,
        hook_point: str = "residual_post_block",
        alpha: float = 1.0,
    ) -> None:
        """Install a single directional ablation at one layer."""
        self.install(model, adapter, [InterventionSpec(
            mode="ablate", direction=direction, layer_idx=layer_idx,
            hook_point=hook_point, alpha=alpha,
        )])

    def install_addition(
        self,
        model: "PreTrainedModel",
        adapter: "ModelAdapter",
        direction: torch.Tensor,
        layer_idx: int,
        hook_point: str = "residual_post_block",
        alpha: float = 1.0,
    ) -> None:
        """Install a single directional addition at one layer."""
        self.install(model, adapter, [InterventionSpec(
            mode="add", direction=direction, layer_idx=layer_idx,
            hook_point=hook_point, alpha=alpha,
        )])

    def install_multilayer_ablation(
        self,
        model: "PreTrainedModel",
        adapter: "ModelAdapter",
        direction: torch.Tensor,
        layer_indices: list[int],
        hook_point: str = "residual_post_block",
        alpha: float = 1.0,
    ) -> None:
        """Install the same ablation at multiple layers (Arditi's default)."""
        self.install(model, adapter, [
            InterventionSpec(
                mode="ablate", direction=direction, layer_idx=li,
                hook_point=hook_point, alpha=alpha,
            ) for li in layer_indices
        ])

    # ── Teardown / introspection ────────────────────────────────────

    def remove(self) -> None:
        """Remove all installed intervention hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._specs.clear()

    def is_active(self) -> bool:
        return bool(self._hooks)

    def describe(self) -> str:
        """Human-readable summary of currently-installed interventions."""
        if not self._specs:
            return "ActivationIntervention(no hooks installed)"
        lines = [f"ActivationIntervention ({len(self._specs)} hooks):"]
        for i, s in enumerate(self._specs):
            lines.append(
                f"  [{i}] {s.mode} @ {s.hook_point}"
                f"{'' if s.layer_idx is None else f' L{s.layer_idx}'}"
                f"  alpha={s.alpha}"
            )
        return "\n".join(lines)

    # ── Context manager sugar ───────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


# ── Hook factories (module-private) ─────────────────────────────────

def _extract_hidden(output):
    """Split a module output into (hidden_tensor, rebuild_fn).

    Decoder-block modules return a tuple ``(hidden, present_kv, attn_weights)``;
    norms and MLPs return a plain tensor. The rebuild function reinserts a
    new hidden tensor into the original output shape so downstream consumers
    see the structure they expect.
    """
    if isinstance(output, tuple):
        h = output[0]
        rest = output[1:]
        return h, lambda new_h: (new_h,) + rest
    return output, lambda new_h: new_h


def _make_ablate_hook(v: torch.Tensor, alpha: float):
    """Project v out of the hidden state: h' = h - alpha * (h·v) * v.

    Broadcasts over any leading dimensions (batch, sequence, etc.) as long
    as the final dim matches ``v.shape[0]``.
    """
    def hook(module, inp, output):
        h, rebuild = _extract_hidden(output)
        # (h · v) along the last dim, keeping a trailing 1 for broadcast.
        proj = (h * v).sum(dim=-1, keepdim=True)
        h_new = h - alpha * proj * v
        return rebuild(h_new)
    return hook


def _make_add_hook(v: torch.Tensor, alpha: float):
    """Add alpha * v to the hidden state at every position."""
    def hook(module, inp, output):
        h, rebuild = _extract_hidden(output)
        h_new = h + alpha * v
        return rebuild(h_new)
    return hook
