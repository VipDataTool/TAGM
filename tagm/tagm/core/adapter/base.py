"""ModelAdapter abstract base class.

An adapter describes a model family's structural contract. It is the *only*
place model-family-specific knowledge lives. Pipeline, capture, measurement,
and analysis layers all read from this contract; they never inspect model
internals directly.

One adapter per family (qwen2, llama3, ...); many specific model IDs within
a family share an adapter instance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from torch.nn import Module

if TYPE_CHECKING:
    # Only imported for typing; the adapter module itself doesn't need
    # transformers installed until a concrete adapter is instantiated.
    from transformers import PreTrainedModel, PreTrainedTokenizer


# ── Hook points ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class HookPointSpec:
    """Description of a single hookable location within a layer (or model).

    Adapters declare HOOK_POINTS as a mapping from name → HookPointSpec.
    Names should describe what the point *is* (e.g. "residual_post_block"),
    not how it's implemented (e.g. "input_layernorm_output").
    """
    name: str
    description: str
    module_path: str            # e.g. "model.layers.{layer}.input_layernorm" or "model.norm"
    captures: frozenset[str]    # which capture types this hook point supports
    layer_independent: bool = False  # True for model-level points like final_norm


# ── Projection roles ────────────────────────────────────────────────

@dataclass(frozen=True)
class ProjectionRole:
    """Description of a projection role within an attention or MLP block.

    role:                "q", "k", "v", "o", "gate", "up", "down"
    weight_key_template: template for the state_dict key, with {layer} placeholder
    block_type:          "attention" or "mlp"
    """
    role: str
    weight_key_template: str
    block_type: str


# ── Memory profile ──────────────────────────────────────────────────

@dataclass
class MemoryProfile:
    """Adapter's estimate of memory characteristics for a model pair.

    Used by the Pipeline to decide whether both models can live in RAM
    concurrently, or whether sequential loading (base-phase then instruct-phase)
    is required. All estimates in GB.
    """
    instruct_model_gb: float
    base_model_gb: float
    full_deltas_gb: float
    signal_only_deltas_gb: float

    def can_run_concurrent(self, available_gb: float) -> bool:
        """Whether base and instruct can both be loaded plus deltas, with 20% margin."""
        return available_gb >= (self.instruct_model_gb + self.base_model_gb +
                                 self.full_deltas_gb) * 1.2

    def can_load_full_deltas(self, available_gb: float) -> bool:
        """Whether full deltas fit after instruct is loaded, with 50% margin."""
        headroom = available_gb - self.instruct_model_gb
        return headroom >= self.full_deltas_gb * 1.5


# ── Adapter base class ──────────────────────────────────────────────

class ModelAdapter(ABC):
    """Abstract base for model family adapters.

    Subclasses declare structural knowledge for a model family without
    inheriting any pipeline, measurement, or service code. Adapters are
    stateless; one instance per family is registered globally and reused
    for any model pair within the family.
    """

    # ── Identity ────────────────────────────────────────────────────
    family_id: str = ""              # short identifier (e.g. "qwen2")
    family_display_name: str = ""    # human-readable (e.g. "Qwen 2.x")

    # ── Declared contract ───────────────────────────────────────────
    HOOK_POINTS: dict[str, HookPointSpec] = {}
    PROJECTION_ROLES: dict[str, ProjectionRole] = {}

    # ── Detection ───────────────────────────────────────────────────
    @classmethod
    @abstractmethod
    def matches(cls, model: "PreTrainedModel") -> bool:
        """True if this adapter can handle the given loaded model.

        Typically checks `model.config.model_type` or architecture-specific
        properties. Used by the registry for auto-detection.
        """
        ...

    # ── Structure introspection ─────────────────────────────────────
    @abstractmethod
    def n_layers(self, model: "PreTrainedModel") -> int: ...

    @abstractmethod
    def hidden_size(self, model: "PreTrainedModel") -> int: ...

    @abstractmethod
    def attention_heads(self, model: "PreTrainedModel") -> tuple[int, int]:
        """Return (n_attention_heads, n_kv_heads). For non-GQA models the two
        values are equal."""
        ...

    @abstractmethod
    def head_dim(self, model: "PreTrainedModel") -> int: ...

    @abstractmethod
    def vocab_size(self, model: "PreTrainedModel") -> int: ...

    # ── Hook resolution ─────────────────────────────────────────────
    @abstractmethod
    def resolve_hook_target(self, model: "PreTrainedModel",
                             hook_point: str,
                             layer_idx: Optional[int] = None) -> Module:
        """Walk the model to the nn.Module at hook_point.

        For layer-dependent hook points, layer_idx must be provided.
        For layer-independent hook points (e.g. final_norm), layer_idx is ignored.

        Raises KeyError if hook_point is not declared in HOOK_POINTS.
        Raises IndexError if layer_idx is out of range.
        """
        ...

    # ── Projection key resolution ───────────────────────────────────
    @abstractmethod
    def projection_weight_key(self, role: str, layer_idx: int) -> str:
        """Return the state_dict key for a given projection role at a layer.

        e.g. ("v", 5) → "model.layers.5.self_attn.v_proj.weight"
        Raises KeyError if role is not declared in PROJECTION_ROLES.
        """
        ...

    @abstractmethod
    def is_projection_key(self, key: str) -> bool:
        """True if a state_dict key matches any declared projection role."""
        ...

    @abstractmethod
    def parse_projection_key(self, key: str) -> Optional[tuple[str, int]]:
        """Parse a state_dict key into (role, layer_idx), or None if not recognized."""
        ...

    # ── Unembedding weight access ───────────────────────────────────
    def unembedding_weight(self, model: "PreTrainedModel"):
        """Return the model's unembedding matrix (for LTP-style probe directions).

        Default implementation checks `lm_head` then `model.embed_tokens` (for
        tied-embedding models). Override if the family has a different structure.
        """
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            return model.lm_head.weight.detach()
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens.weight.detach()
        raise AttributeError(f"Cannot locate unembedding weight on model of "
                             f"family '{self.family_id}'")

    # ── O-projection weight access (base weights, not deltas) ───────
    def o_proj_weight(self, model: "PreTrainedModel", layer_idx: int):
        """Return the o_proj weight tensor at a layer (base weight, not delta).

        Used by LTP to project lateral tension directions back into
        residual-stream coordinates. Default implementation uses the
        standard HuggingFace attribute path.
        """
        return model.model.layers[layer_idx].self_attn.o_proj.weight.detach()

    # ── Tokenizer policy ────────────────────────────────────────────
    def shared_tokenizer(self) -> bool:
        """True if base and instruct models in this family share a tokenizer.

        Default True. Override if the family has known tokenizer divergence
        between base and instruct variants.
        """
        return True

    def load_tokenizer(self, model_id: str, hf_token: Optional[str] = None
                        ) -> "PreTrainedTokenizer":
        """Load the tokenizer for a model in this family.

        Default uses AutoTokenizer; override for family-specific handling.
        """
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    # ── Memory profiling ────────────────────────────────────────────
    @abstractmethod
    def estimate_memory_profile(self, model_id: str) -> MemoryProfile:
        """Estimate memory footprint for a specific model in this family.

        Used by the pipeline to decide concurrent vs sequential base/instruct
        loading. Estimates are heuristic; the pipeline treats them as advisory.
        """
        ...

    # ── Capture defaults ────────────────────────────────────────────
    def default_capture_config(self, model: "PreTrainedModel"):
        """Return a sensible default CaptureConfig for this family.

        Typically captures hidden states at residual_post_block at every
        layer, plus final_norm. Override for family-specific defaults.
        Note: this is an *adapter default*, not a measurement default —
        measurements declare their own CaptureExpectations independently.
        """
        from tagm.core.capture.config import CaptureConfig, CapturePoint
        n = self.n_layers(model)
        points = [
            CapturePoint(
                layer=i,
                hook_point="residual_post_block",
                capture=frozenset({"hidden"}),
                reduction=None,
                precision="model_dtype",
            )
            for i in range(n)
        ]
        points.append(CapturePoint(
            layer=None,
            hook_point="final_norm",
            capture=frozenset({"hidden"}),
            reduction=None,
            precision="model_dtype",
        ))
        return CaptureConfig(
            name=f"{self.family_id}-default",
            description=f"Adapter default for {self.family_display_name}: "
                        f"residual_post_block at every layer + final_norm",
            points=tuple(points),
        )


# ── Internal helper: dotted path walker ─────────────────────────────

def walk_module_path(root, dotted_path: str):
    """Walk a dotted attribute path with `[N]`-style list indexing.

    Handles patterns like "model.layers.5.self_attn" by treating a numeric
    path segment after a list-ish attribute as an index rather than an
    attribute lookup.

    Shared by all adapters; lives in base.py so adapters don't reimplement.
    """
    obj = root
    parts = dotted_path.split(".")
    i = 0
    while i < len(parts):
        part = parts[i]
        obj = getattr(obj, part)
        i += 1
        # If the next part is numeric, treat it as a list index into `obj`
        if i < len(parts) and parts[i].isdigit():
            obj = obj[int(parts[i])]
            i += 1
    return obj
