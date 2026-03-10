"""
Model Manager: loads base/instruct model pairs, computes weight deltas.
Supports any HuggingFace transformer pair with identical architecture.
"""

import torch
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import Optional

# Known model pairs: (base, instruct, display_name)
KNOWN_PAIRS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen 2.5 0.5B"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen 2.5 1.5B"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5 3B"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B"),
}


@dataclass
class ModelState:
    """Holds loaded model, tokenizer, deltas, and metadata."""
    model_instruct: object = None
    model_base: object = None          # only loaded when needed for KL
    tokenizer: object = None
    deltas: dict = field(default_factory=dict)
    delta_frob_norms: dict = field(default_factory=dict)
    v_deltas: dict = field(default_factory=dict)  # V-projection deltas for signed attribution
    config: object = None
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    signal_layers: list = field(default_factory=list)
    pair_id: str = ""
    base_model_id: str = ""
    instruct_model_id: str = ""
    device: str = "cpu"
    dtype: object = torch.float32
    loaded: bool = False


class ModelManager:
    def __init__(self):
        self.state: Optional[ModelState] = None
        self._hooks = []
        self.activations = {}
        self.attn_weights = {}

    def get_available_pairs(self):
        return {k: v[2] for k, v in KNOWN_PAIRS.items()}

    def is_loaded(self, pair_id: str = None) -> bool:
        if self.state is None or not self.state.loaded:
            return False
        if pair_id and self.state.pair_id != pair_id:
            return False
        return True

    def load_pair(self, pair_id: str = None, base_id: str = None,
                  instruct_id: str = None, device: str = "cpu",
                  callback=None):
        """
        Load a model pair and compute deltas.
        callback(stage, message) for progress updates.
        """
        def log(stage, msg):
            if callback:
                callback(stage, msg)

        # Resolve model IDs
        if pair_id and pair_id in KNOWN_PAIRS:
            base_id, instruct_id, display = KNOWN_PAIRS[pair_id]
        elif base_id and instruct_id:
            pair_id = f"custom:{base_id}|{instruct_id}"
            display = f"{base_id} / {instruct_id}"
        else:
            raise ValueError("Provide pair_id or both base_id and instruct_id")

        # Skip if already loaded
        if self.is_loaded(pair_id):
            log("ready", f"{display} already loaded")
            return self.state

        # Unload previous
        self._unload()

        dtype = torch.float32
        state = ModelState(
            pair_id=pair_id,
            base_model_id=base_id,
            instruct_model_id=instruct_id,
            device=device,
            dtype=dtype,
        )

        t0 = time.time()

        # Load instruct model
        log("loading", f"Loading instruct model: {instruct_id}")
        state.model_instruct = AutoModelForCausalLM.from_pretrained(
            instruct_id, torch_dtype=dtype, device_map=device,
            attn_implementation="eager",
        )
        state.tokenizer = AutoTokenizer.from_pretrained(instruct_id)
        if state.tokenizer.pad_token is None:
            state.tokenizer.pad_token = state.tokenizer.eos_token

        state.config = state.model_instruct.config
        state.n_layers = state.config.num_hidden_layers
        state.n_heads = state.config.num_attention_heads
        state.n_kv_heads = getattr(state.config, "num_key_value_heads", state.n_heads)
        state.hidden_size = state.config.hidden_size
        state.head_dim = state.hidden_size // state.n_heads

        # Default signal layers: middle third of attention layers
        mid_start = state.n_layers // 3
        mid_end = 2 * state.n_layers // 3
        state.signal_layers = list(range(mid_start, mid_end))

        log("loading", f"Loading base model: {base_id}")
        model_base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=dtype, device_map=device,
        )

        # Compute deltas
        log("computing", "Computing weight deltas...")
        base_sd = model_base.state_dict()
        inst_sd = state.model_instruct.state_dict()

        proj_keys = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight",
            "gate_proj.weight", "up_proj.weight",
        ]

        for name in base_sd:
            if any(k in name for k in proj_keys):
                d = inst_sd[name] - base_sd[name]
                state.deltas[name] = d
                state.delta_frob_norms[name] = d.norm().item()

        # V-projection deltas for signed attribution (all layers)
        for layer_idx in range(state.n_layers):
            vname = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            if vname in state.deltas:
                state.v_deltas[layer_idx] = state.deltas[vname]

        del model_base, base_sd, inst_sd
        gc.collect()

        elapsed = time.time() - t0
        log("ready", f"Loaded {display} in {elapsed:.1f}s "
            f"({state.n_layers}L, {state.hidden_size}d, "
            f"{len(state.deltas)} deltas)")

        state.loaded = True
        self.state = state
        return state

    def load_base_for_kl(self, callback=None):
        """Load base model for KL divergence computation (heavier on RAM)."""
        if self.state.model_base is not None:
            return
        if callback:
            callback("loading", "Loading base model for behavioral comparison...")
        self.state.model_base = AutoModelForCausalLM.from_pretrained(
            self.state.base_model_id,
            torch_dtype=self.state.dtype,
            device_map=self.state.device,
        )

    def unload_base(self):
        """Free base model RAM after KL computation."""
        if self.state and self.state.model_base is not None:
            del self.state.model_base
            self.state.model_base = None
            gc.collect()

    def install_hooks(self, layers=None):
        """Install forward hooks on specified layers (or signal layers)."""
        self._remove_hooks()
        self.activations.clear()
        self.attn_weights.clear()

        if layers is None:
            layers = self.state.signal_layers

        model = self.state.model_instruct

        def make_output_hook(name):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook

        def make_attn_hook(name):
            def hook(module, inp, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    self.attn_weights[name] = output[1].detach()
            return hook

        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            h = layer.input_layernorm.register_forward_hook(
                make_output_hook(f"layer_{layer_idx}_h"))
            self._hooks.append(h)
            h = layer.self_attn.register_forward_hook(
                make_attn_hook(f"layer_{layer_idx}_attn"))
            self._hooks.append(h)

    def install_full_hooks(self):
        """Install hooks on ALL layers for full trajectory computation."""
        self._remove_hooks()
        self.activations.clear()
        self.attn_weights.clear()

        model = self.state.model_instruct

        def make_output_hook(name):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook

        for layer_idx, layer in enumerate(model.model.layers):
            h = layer.input_layernorm.register_forward_hook(
                make_output_hook(f"layer_{layer_idx}_attn"))
            self._hooks.append(h)
            h = layer.post_attention_layernorm.register_forward_hook(
                make_output_hook(f"layer_{layer_idx}_mlp"))
            self._hooks.append(h)

    def forward(self, prompt: str, output_attentions=False):
        """Run forward pass, populating hooked activations."""
        self.activations.clear()
        self.attn_weights.clear()
        inputs = self.state.tokenizer(prompt, return_tensors="pt").to(self.state.device)
        tokens = self.state.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        with torch.no_grad():
            out = self.state.model_instruct(**inputs, output_attentions=output_attentions)
        return tokens, inputs, out

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _unload(self):
        self._remove_hooks()
        if self.state:
            del self.state.model_instruct
            if self.state.model_base:
                del self.state.model_base
            self.state = None
        gc.collect()
