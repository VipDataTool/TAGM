"""Qwen 2.x adapter.

Covers all Qwen2ForCausalLM models (0.5B, 1.5B, 3B, 7B, ...) with a single
adapter instance. The adapter knows nothing about specific model pairs —
the Pipeline wires it up to whichever instruct/base pair the user selected.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from misc.old.tagm.core.adapter.base import (
    ModelAdapter,
    HookPointSpec,
    ProjectionRole,
    MemoryProfile,
    walk_module_path,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from torch.nn import Module


class Qwen2Adapter(ModelAdapter):
    """Adapter for the Qwen 2.x model family (model_type == 'qwen2')."""

    family_id = "qwen2"
    family_display_name = "Qwen 2.x"

    HOOK_POINTS = {
        "pre_attn_norm": HookPointSpec(
            name="pre_attn_norm",
            description="Output of input_layernorm (input to self-attention)",
            module_path="model.layers.{layer}.input_layernorm",
            captures=frozenset({"hidden"}),
        ),
        "attn_output": HookPointSpec(
            name="attn_output",
            description="Output of self-attention block (with attention weights)",
            module_path="model.layers.{layer}.self_attn",
            captures=frozenset({"hidden", "attention_weights"}),
        ),
        "post_attn_norm": HookPointSpec(
            name="post_attn_norm",
            description="Output of post_attention_layernorm (input to MLP)",
            module_path="model.layers.{layer}.post_attention_layernorm",
            captures=frozenset({"hidden"}),
        ),
        "mlp_output": HookPointSpec(
            name="mlp_output",
            description="Output of MLP block",
            module_path="model.layers.{layer}.mlp",
            captures=frozenset({"hidden"}),
        ),
        "residual_post_block": HookPointSpec(
            name="residual_post_block",
            description="Residual stream after the full transformer block",
            module_path="model.layers.{layer}",
            captures=frozenset({"hidden"}),
        ),
        "final_norm": HookPointSpec(
            name="final_norm",
            description="Output of final RMSNorm before lm_head",
            module_path="model.norm",
            captures=frozenset({"hidden"}),
            layer_independent=True,
        ),
    }

    PROJECTION_ROLES = {
        "q":    ProjectionRole("q",    "model.layers.{layer}.self_attn.q_proj.weight",    "attention"),
        "k":    ProjectionRole("k",    "model.layers.{layer}.self_attn.k_proj.weight",    "attention"),
        "v":    ProjectionRole("v",    "model.layers.{layer}.self_attn.v_proj.weight",    "attention"),
        "o":    ProjectionRole("o",    "model.layers.{layer}.self_attn.o_proj.weight",    "attention"),
        "gate": ProjectionRole("gate", "model.layers.{layer}.mlp.gate_proj.weight",       "mlp"),
        "up":   ProjectionRole("up",   "model.layers.{layer}.mlp.up_proj.weight",         "mlp"),
        "down": ProjectionRole("down", "model.layers.{layer}.mlp.down_proj.weight",       "mlp"),
    }

    # Precompiled pattern for key parsing
    _PROJ_PATTERN = re.compile(
        r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)_proj\.weight$"
    )
    _ROLE_FROM_KEY = {
        ("self_attn", "q"): "q",
        ("self_attn", "k"): "k",
        ("self_attn", "v"): "v",
        ("self_attn", "o"): "o",
        ("mlp", "gate"): "gate",
        ("mlp", "up"): "up",
        ("mlp", "down"): "down",
    }

    # ── Detection ───────────────────────────────────────────────────
    @classmethod
    def matches(cls, model: "PreTrainedModel") -> bool:
        return getattr(model.config, "model_type", "") == "qwen2"

    # ── Structure introspection ─────────────────────────────────────
    def n_layers(self, model): return model.config.num_hidden_layers
    def hidden_size(self, model): return model.config.hidden_size

    def attention_heads(self, model):
        n_heads = model.config.num_attention_heads
        n_kv = getattr(model.config, "num_key_value_heads", n_heads)
        return n_heads, n_kv

    def head_dim(self, model):
        return self.hidden_size(model) // self.attention_heads(model)[0]

    def vocab_size(self, model): return model.config.vocab_size

    # ── Hook resolution ─────────────────────────────────────────────
    def resolve_hook_target(self, model, hook_point, layer_idx=None):
        if hook_point not in self.HOOK_POINTS:
            raise KeyError(
                f"Hook point '{hook_point}' not declared by adapter "
                f"'{self.family_id}'. Available: {list(self.HOOK_POINTS)}"
            )
        spec = self.HOOK_POINTS[hook_point]
        if spec.layer_independent:
            return walk_module_path(model, spec.module_path)
        if layer_idx is None:
            raise ValueError(
                f"Hook point '{hook_point}' requires layer_idx")
        n = self.n_layers(model)
        if not (0 <= layer_idx < n):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for model with {n} layers")
        path = spec.module_path.format(layer=layer_idx)
        return walk_module_path(model, path)

    # ── Projection key resolution ───────────────────────────────────
    def projection_weight_key(self, role, layer_idx):
        if role not in self.PROJECTION_ROLES:
            raise KeyError(
                f"Projection role '{role}' not declared by adapter "
                f"'{self.family_id}'. Available: {list(self.PROJECTION_ROLES)}"
            )
        return self.PROJECTION_ROLES[role].weight_key_template.format(layer=layer_idx)

    def is_projection_key(self, key):
        return self._PROJ_PATTERN.match(key) is not None

    def parse_projection_key(self, key):
        m = self._PROJ_PATTERN.match(key)
        if not m:
            return None
        layer = int(m.group(1))
        block = m.group(2)
        proj_name = m.group(3)
        role = self._ROLE_FROM_KEY.get((block, proj_name))
        if role is None:
            return None
        return (role, layer)

    # ── Memory profiling ────────────────────────────────────────────
    def estimate_memory_profile(self, model_id: str) -> MemoryProfile:
        """Heuristic table for known Qwen 2 sizes.

        Entries are tagged by substring match on the model_id (e.g. "0.5B",
        "1.5B"). A conservative fallback is returned if no tag matches.
        """
        size_table = {
            "0.5B": MemoryProfile(1.0, 1.0, 0.6, 0.2),
            "1.5B": MemoryProfile(3.0, 3.0, 1.8, 0.6),
            "3B":   MemoryProfile(6.0, 6.0, 3.6, 1.2),
            "7B":   MemoryProfile(14.0, 14.0, 8.4, 2.8),
            "14B":  MemoryProfile(28.0, 28.0, 16.8, 5.6),
            "32B":  MemoryProfile(64.0, 64.0, 38.0, 13.0),
        }
        # Check largest tags first so "14B" isn't shadowed by "4B"
        for tag in sorted(size_table, key=lambda t: len(t), reverse=True):
            if tag in model_id:
                return size_table[tag]
        return MemoryProfile(8.0, 8.0, 4.8, 1.6)  # conservative fallback
