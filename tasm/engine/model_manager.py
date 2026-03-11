"""
Model Manager: loads base/instruct model pairs, computes weight deltas.
Supports any HuggingFace transformer pair with identical architecture.
"""

import torch
import gc
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import Optional

# HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")

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
    model_base: object = None
    tokenizer: object = None
    deltas: dict = field(default_factory=dict)
    delta_frob_norms: dict = field(default_factory=dict)
    config: object = None
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    signal_layers: list = field(default_factory=list)
    pair_id: str = ""
    display_name: str = ""
    base_model_id: str = ""
    instruct_model_id: str = ""
    device: str = "cpu"
    dtype: object = torch.float32
    loaded: bool = False

    def v_delta(self, layer_idx: int):
        """Reference into deltas dict - no duplication."""
        return self.deltas.get(
            f"model.layers.{layer_idx}.self_attn.v_proj.weight")


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
        def log(stage, msg):
            if callback:
                callback(stage, msg)

        if pair_id and pair_id in KNOWN_PAIRS:
            base_id, instruct_id, display = KNOWN_PAIRS[pair_id]
        elif base_id and instruct_id:
            pair_id = f"custom:{base_id}|{instruct_id}"
            display = f"{base_id} / {instruct_id}"
        else:
            raise ValueError("Provide pair_id or both base_id and instruct_id")

        if self.is_loaded(pair_id):
            log("ready", f"{display} already loaded")
            return self.state

        # Aggressive unload: free everything before loading new pair
        self._unload()
        gc.collect()
        gc.collect()  # second pass catches reference cycles

        dtype = torch.float32
        state = ModelState(
            pair_id=pair_id, display_name=display,
            base_model_id=base_id, instruct_model_id=instruct_id,
            device=device, dtype=dtype,
        )

        t0 = time.time()

        log("loading", f"Loading instruct model: {instruct_id}")
        state.model_instruct = AutoModelForCausalLM.from_pretrained(
            instruct_id, dtype=dtype, device_map=device,
            attn_implementation="eager", token=HF_TOKEN,
        )
        state.tokenizer = AutoTokenizer.from_pretrained(instruct_id, token=HF_TOKEN)
        if state.tokenizer.pad_token is None:
            state.tokenizer.pad_token = state.tokenizer.eos_token

        state.config = state.model_instruct.config
        state.n_layers = state.config.num_hidden_layers
        state.n_heads = state.config.num_attention_heads
        state.n_kv_heads = getattr(state.config, "num_key_value_heads", state.n_heads)
        state.hidden_size = state.config.hidden_size
        state.head_dim = state.hidden_size // state.n_heads

        mid_start = state.n_layers // 3
        mid_end = 2 * state.n_layers // 3
        state.signal_layers = list(range(mid_start, mid_end))

        # Load base model, extract ONLY projection weights, then free immediately.
        # This avoids holding two full model objects in RAM simultaneously.
        log("loading", f"Loading base model weights: {base_id}")
        model_base = AutoModelForCausalLM.from_pretrained(
            base_id, dtype=dtype, device_map=device, token=HF_TOKEN,
        )

        proj_keys = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight",
            "gate_proj.weight", "up_proj.weight",
        ]

        # Extract only the projection weights we need, as cloned tensors
        # so they survive after we delete the model
        log("computing", "Extracting base projection weights...")
        base_proj_weights = {}
        for name, param in model_base.state_dict().items():
            if any(k in name for k in proj_keys):
                base_proj_weights[name] = param.clone()

        # Free the entire base model before computing deltas
        del model_base
        gc.collect()
        gc.collect()

        # Compute deltas from instruct model (still loaded) and extracted base weights
        log("computing", "Computing weight deltas...")
        inst_sd = state.model_instruct.state_dict()

        for name in base_proj_weights:
            if name in inst_sd:
                d = inst_sd[name] - base_proj_weights[name]
                state.deltas[name] = d
                state.delta_frob_norms[name] = d.norm().item()

        del base_proj_weights, inst_sd
        gc.collect()

        elapsed = time.time() - t0
        log("ready", f"Loaded {display} in {elapsed:.1f}s "
            f"({state.n_layers}L, {state.hidden_size}d, "
            f"{len(state.deltas)} deltas)")

        state.loaded = True
        self.state = state
        return state

    def load_base_for_kl(self, callback=None):
        if self.state.model_base is not None:
            return
        if callback:
            callback("loading", "Loading base model for behavioral comparison...")
        self.state.model_base = AutoModelForCausalLM.from_pretrained(
            self.state.base_model_id,
            dtype=self.state.dtype,
            device_map=self.state.device,
            token=HF_TOKEN,
        )

    def unload_base(self):
        if self.state and self.state.model_base is not None:
            del self.state.model_base
            self.state.model_base = None
            gc.collect()

    def install_analysis_hooks(self, full_trajectory: bool = False):
        """
        Install all hooks for single-pass analysis.

        Always: signal layer layernorm outputs + attention weights.
        If full_trajectory: all-layer layernorm outputs for both sublayers.
        """
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

        def make_attn_hook(name):
            def hook(module, inp, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    self.attn_weights[name] = output[1].detach()
            return hook

        # Signal layers: layernorm output + attention weights
        for layer_idx in self.state.signal_layers:
            layer = model.model.layers[layer_idx]
            self._hooks.append(
                layer.input_layernorm.register_forward_hook(
                    make_output_hook(f"layer_{layer_idx}_h")))
            self._hooks.append(
                layer.self_attn.register_forward_hook(
                    make_attn_hook(f"layer_{layer_idx}_attn")))

        # Full trajectory: all layers, both sublayer inputs
        if full_trajectory:
            for layer_idx, layer in enumerate(model.model.layers):
                self._hooks.append(
                    layer.input_layernorm.register_forward_hook(
                        make_output_hook(f"layer_{layer_idx}_traj_attn")))
                self._hooks.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        make_output_hook(f"layer_{layer_idx}_traj_mlp")))

    def forward(self, prompt: str, output_attentions=False):
        self.activations.clear()
        self.attn_weights.clear()
        inputs = self.state.tokenizer(prompt, return_tensors="pt").to(self.state.device)
        tokens = self.state.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        with torch.no_grad():
            out = self.state.model_instruct(**inputs, output_attentions=output_attentions)
        return tokens, inputs, out

    def clear_activations(self):
        self.activations.clear()
        self.attn_weights.clear()

    def reset(self):
        self._unload()

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _unload(self):
        self._remove_hooks()
        self.activations.clear()
        self.attn_weights.clear()
        if self.state:
            if self.state.model_instruct is not None:
                del self.state.model_instruct
            if self.state.model_base is not None:
                del self.state.model_base
            self.state.deltas.clear()
            self.state.delta_frob_norms.clear()
            self.state = None
        gc.collect()
