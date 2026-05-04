"""Llama 3.x adapter.

Covers the Llama 3 model family. Structurally very similar to Qwen2 (same
decoder-block layout, same projection names under self_attn/mlp), which
is intentional — having two adapters that follow the same pattern is a
sanity check that the adapter abstraction captures the family-varying
structure in the right places.

For models that use tied embeddings (smaller Llama3 variants), the default
`unembedding_weight` implementation falls back to `model.embed_tokens`.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from tagm.core.adapter.base import (
    ModelAdapter,
    HookPointSpec,
    ProjectionRole,
    MemoryProfile,
    walk_module_path,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class Llama3Adapter(ModelAdapter):
    """Adapter for the Llama 3.x family (model_type == 'llama')."""

    family_id = "llama3"
    family_display_name = "Llama 3.x"

    HOOK_POINTS = {
        "pre_attn_norm": HookPointSpec(
            name="pre_attn_norm",
            description="Output of input_layernorm (input to self-attention)",
            module_path="model.layers.{layer}.input_layernorm",
            captures=frozenset({"hidden"}),
        ),
        "attn_output": HookPointSpec(
            name="attn_output",
            description="Output of self-attention block",
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

    _PROJ_PATTERN = re.compile(
        r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)_proj\.weight$"
    )
    _ROLE_FROM_KEY = {
        ("self_attn", "q"): "q", ("self_attn", "k"): "k",
        ("self_attn", "v"): "v", ("self_attn", "o"): "o",
        ("mlp", "gate"): "gate", ("mlp", "up"): "up", ("mlp", "down"): "down",
    }

    @classmethod
    def matches(cls, model):
        return getattr(model.config, "model_type", "") == "llama"

    # ── Tokenizer policy ────────────────────────────────────────────
    # Suppress the leading <|begin_of_text|> (Llama 3) / <s> (Llama 2-arch
    # variants like TinyLlama) that the HF tokenizer prepends by default.
    # This brings position 0 in line with Qwen 2.5, whose tokenizer does
    # not add a BOS, so cross-family analyses share the same token-axis
    # contract: position 0 is the first content token, not a special
    # marker.
    #
    # Why we strip it here rather than skip it post-hoc:
    #   - Avoids a position-0 attention-sink + unconditional-prior spike
    #     that dominates the correction-field topology view and obscures
    #     real signal at content tokens.
    #   - Keeps result.tokens, per_token_kl, ltp profiles, and rank
    #     displacement bank profiles all aligned to a single content-only
    #     token axis with no per-call-site special-casing.
    #   - The model still sees every input symmetrically (base and
    #     instruct both run without BOS), so KL and tension comparisons
    #     remain valid; only the absolute calibration to training
    #     distribution shifts slightly, which we don't depend on.
    #
    # Two-step suppression because Llama-family tokenizers come in three
    # shapes and the canonical lever differs:
    #
    #   1. LlamaTokenizer (slow, sentencepiece): add_bos_token is a
    #      class attribute checked in build_inputs_with_special_tokens.
    #      A plain attribute set is sufficient.
    #
    #   2. LlamaTokenizerFast (sentencepiece-based fast; TinyLlama):
    #      add_bos_token is a property whose setter calls
    #      update_post_processor() and rebuilds the Rust template.
    #      Setting it works.
    #
    #   3. PreTrainedTokenizerFast (Llama 3, tiktoken BPE): no property
    #      setter — assigning add_bos_token is inert. BOS is enforced
    #      by a TemplateProcessing post-processor on the underlying
    #      Rust tokenizer, and that post-processor must be rewritten
    #      directly to take effect.
    #
    # We do both: set the attribute (handles cases 1 and 2), then rewrite
    # the post-processor (handles case 3, and is a safe no-op for the
    # others because Llama-family tokenizers don't use the post-processor
    # to add EOS or pair-separator tokens — EOS is generated, not
    # appended, and TAGM never tokenizes pair sequences).
    #
    # To restore default Llama BOS behavior, delete this override.
    def load_tokenizer(self, model_id: str, hf_token: Optional[str] = None
                        ) -> "PreTrainedTokenizer":
        tok = super().load_tokenizer(model_id, hf_token=hf_token)

        # Step 1: the canonical Python-level lever.
        if hasattr(tok, "add_bos_token"):
            tok.add_bos_token = False

        # Step 2: rewrite the Rust-side post-processor for fast tokenizers
        # whose BOS injection is enforced there independently of any
        # Python attribute. The replacement template emits the input
        # sequence verbatim with no special tokens.
        inner = getattr(tok, "_tokenizer", None)
        if inner is not None and hasattr(inner, "post_processor"):
            from tokenizers import processors
            inner.post_processor = processors.TemplateProcessing(
                single="$A:0",
                pair="$A:0 $B:1",
                special_tokens=[],
            )

        return tok

    def n_layers(self, model): return model.config.num_hidden_layers
    def hidden_size(self, model): return model.config.hidden_size

    def attention_heads(self, model):
        n_heads = model.config.num_attention_heads
        n_kv = getattr(model.config, "num_key_value_heads", n_heads)
        return n_heads, n_kv

    def head_dim(self, model):
        return self.hidden_size(model) // self.attention_heads(model)[0]

    def vocab_size(self, model): return model.config.vocab_size

    def resolve_hook_target(self, model, hook_point, layer_idx=None):
        if hook_point not in self.HOOK_POINTS:
            raise KeyError(f"Hook point '{hook_point}' not declared by "
                           f"adapter '{self.family_id}'")
        spec = self.HOOK_POINTS[hook_point]
        if spec.layer_independent:
            return walk_module_path(model, spec.module_path)
        if layer_idx is None:
            raise ValueError(f"Hook point '{hook_point}' requires layer_idx")
        n = self.n_layers(model)
        if not (0 <= layer_idx < n):
            raise IndexError(f"layer_idx {layer_idx} out of range for model "
                             f"with {n} layers")
        return walk_module_path(model, spec.module_path.format(layer=layer_idx))

    def projection_weight_key(self, role, layer_idx):
        if role not in self.PROJECTION_ROLES:
            raise KeyError(f"Projection role '{role}' not declared")
        return self.PROJECTION_ROLES[role].weight_key_template.format(layer=layer_idx)

    def is_projection_key(self, key):
        return self._PROJ_PATTERN.match(key) is not None

    def parse_projection_key(self, key):
        m = self._PROJ_PATTERN.match(key)
        if not m:
            return None
        role = self._ROLE_FROM_KEY.get((m.group(2), m.group(3)))
        if role is None:
            return None
        return (role, int(m.group(1)))

    def estimate_memory_profile(self, model_id: str) -> MemoryProfile:
        """Heuristic table for known Llama 3 sizes."""
        size_table = {
            "1B":   MemoryProfile(2.5, 2.5, 1.5, 0.5),
            "3B":   MemoryProfile(6.0, 6.0, 3.6, 1.2),
            "8B":   MemoryProfile(16.0, 16.0, 9.6, 3.2),
            "70B":  MemoryProfile(140.0, 140.0, 84.0, 28.0),
        }
        for tag in sorted(size_table, key=lambda t: len(t), reverse=True):
            if tag in model_id:
                return size_table[tag]
        return MemoryProfile(16.0, 16.0, 9.6, 3.2)  # conservative fallback
