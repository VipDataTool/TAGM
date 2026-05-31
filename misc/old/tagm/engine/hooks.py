"""Adapter-based hook installation for activation capture.

The adapter resolves model-family-specific module paths. Activations
land in flat dicts keyed by TASM's conventions:

    activations["layer_5_h"]       — hidden state at layer 5 (pre-attn norm)
    attn_weights["layer_5_attn"]   — attention weights at layer 5
    activations["layer_5_traj_attn"] — trajectory capture (attn sublayer)
    activations["layer_5_traj_mlp"]  — trajectory capture (mlp sublayer)
    activations["final_norm_h"]    — final norm output

No ActivationStore, no CaptureConfig. The caller decides which layers
to hook based on signal_layers, LTP layers, domain layers, etc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from misc.old.tagm.engine import config as engine_config

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from misc.old.tagm.core.adapter.base import ModelAdapter


class ActivationCapture:
    """Manages hooks on a model and stores captured activations.

    Usage:
        cap = ActivationCapture()
        cap.install(model, adapter, signal_layers=[8,9,10],
                    full_trajectory=True, domain_layer=12)
        tokens, inputs, output = cap.forward(model, tokenizer, prompt)
        h = cap.activations["layer_9_h"]
        cap.remove()
    """

    def __init__(self):
        self.activations: dict = {}
        self.attn_weights: dict = {}
        self._hooks: list = []

    def install(
        self,
        model: "PreTrainedModel",
        adapter: "ModelAdapter",
        signal_layers: list[int],
        full_trajectory: bool = False,
        ltp_layers: list[int] | None = None,
        domain_layer: int | None = None,
        escalation_layer: int | None = None,
        output_attentions: bool = False,
    ) -> None:
        """Install forward hooks on the model.

        Uses the adapter to resolve model-family-specific module paths,
        but stores results in flat keys matching TASM's conventions.

        Args:
            signal_layers: layers for stress/attribution (always hooked)
            full_trajectory: hook ALL layers for amplitude trajectory
            ltp_layers: additional layers for LTP computation
            domain_layer: layer for domain embedding capture
            escalation_layer: layer for escalation embedding capture
            output_attentions: whether forward will request attention outputs
        """
        self.remove()
        self.activations.clear()
        self.attn_weights.clear()

        n_layers = adapter.n_layers(model)
        hooked: set[int] = set()

        def _make_hidden_hook(key: str):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    self.activations[key] = output[0].detach()
                else:
                    self.activations[key] = output.detach()
            return hook

        def _make_attn_hook(key: str):
            def hook(module, inp, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    self.attn_weights[key] = output[1].detach()
            return hook

        # Signal layers: hidden state + attention weights
        for li in signal_layers:
            if li >= n_layers:
                continue
            target = adapter.resolve_hook_target(model, "pre_attn_norm", li)
            self._hooks.append(
                target.register_forward_hook(_make_hidden_hook(f"layer_{li}_h")))

            attn_target = adapter.resolve_hook_target(model, "attn_output", li)
            self._hooks.append(
                attn_target.register_forward_hook(_make_attn_hook(f"layer_{li}_attn")))
            hooked.add(li)

        # Domain/escalation layers (if outside signal range)
        for dl in [domain_layer, escalation_layer]:
            if dl is not None and dl not in hooked and dl < n_layers:
                target = adapter.resolve_hook_target(model, "pre_attn_norm", dl)
                self._hooks.append(
                    target.register_forward_hook(_make_hidden_hook(f"layer_{dl}_h")))
                attn_target = adapter.resolve_hook_target(model, "attn_output", dl)
                self._hooks.append(
                    attn_target.register_forward_hook(_make_attn_hook(f"layer_{dl}_attn")))
                hooked.add(dl)

        # LTP layers (additional layers outside signal range)
        if ltp_layers:
            for li in ltp_layers:
                if li not in hooked and li < n_layers:
                    target = adapter.resolve_hook_target(model, "pre_attn_norm", li)
                    self._hooks.append(
                        target.register_forward_hook(_make_hidden_hook(f"layer_{li}_h")))
                    hooked.add(li)

        # Final norm (always — used for domain embeddings + correction heatmap)
        try:
            final_target = adapter.resolve_hook_target(model, "final_norm")
            self._hooks.append(
                final_target.register_forward_hook(_make_hidden_hook("final_norm_h")))
        except (KeyError, AttributeError):
            # Adapter may not declare final_norm; fall back to direct access
            if hasattr(model, "model") and hasattr(model.model, "norm"):
                self._hooks.append(
                    model.model.norm.register_forward_hook(
                        _make_hidden_hook("final_norm_h")))

        # Full trajectory: hook every layer at both sublayer points
        if full_trajectory:
            for li in range(n_layers):
                # Attention sublayer: input_layernorm output
                try:
                    attn_traj = adapter.resolve_hook_target(model, "pre_attn_norm", li)
                    self._hooks.append(
                        attn_traj.register_forward_hook(
                            _make_hidden_hook(f"layer_{li}_traj_attn")))
                except (KeyError, AttributeError):
                    pass

                # MLP sublayer: post_attention_layernorm output
                try:
                    mlp_traj = adapter.resolve_hook_target(model, "post_attn_norm", li)
                    self._hooks.append(
                        mlp_traj.register_forward_hook(
                            _make_hidden_hook(f"layer_{li}_traj_mlp")))
                except (KeyError, AttributeError):
                    pass

    def forward(self, model, tokenizer, prompt: str, output_attentions: bool = False):
        """Run a forward pass with hooks active.

        Returns (tokens, inputs, model_output) — same signature as TASM's
        ModelManager.forward().
        """
        import torch

        self.activations.clear()
        self.attn_weights.clear()

        inputs = tokenizer(
            prompt, return_tensors="pt",
            add_special_tokens=engine_config.get("add_special_tokens"),
        ).to(next(model.parameters()).device)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokens = [_clean_token(t) for t in tokens]

        with torch.no_grad():
            output = model(**inputs, output_attentions=output_attentions)

        return tokens, inputs, output

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        """Clear captured data without removing hooks."""
        self.activations.clear()
        self.attn_weights.clear()


def _clean_token(token: str) -> str:
    """Normalize tokenizer-specific whitespace encodings for display."""
    return (token
            .replace("\u0120", " ")    # BPE leading space
            .replace("\u010a", "\\n")   # BPE newline
            .replace("\u2581", " "))    # SentencePiece leading space
