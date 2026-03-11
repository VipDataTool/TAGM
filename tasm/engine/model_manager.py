"""
Model Manager: loads base/instruct model pairs, computes weight deltas.
Supports any HuggingFace transformer pair with identical architecture.

Memory-optimized: base model weights are loaded directly from safetensors
files on disk without ever instantiating the full model. This halves
peak RAM during delta computation.
"""

import torch
import gc
import os
import json
import time
import traceback
import psutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open
from dataclasses import dataclass, field
from typing import Optional, Dict

# HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")


def _mem_gb():
    """Current process RSS in GB."""
    return psutil.Process().memory_info().rss / (1024**3)


def _sys_mem():
    """Return (used_gb, total_gb, pct) for system memory."""
    m = psutil.virtual_memory()
    return m.used / (1024**3), m.total / (1024**3), m.percent

# Known model pairs: (base, instruct, display_name)
KNOWN_PAIRS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen 2.5 0.5B"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen 2.5 1.5B"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5 3B"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B"),
}

PROJ_KEYS = ["q_proj.weight", "k_proj.weight", "v_proj.weight",
             "gate_proj.weight", "up_proj.weight"]


@dataclass
class ModelState:
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
    dtype: object = torch.float16
    loaded: bool = False

    def v_delta(self, layer_idx: int):
        return self.deltas.get(
            f"model.layers.{layer_idx}.self_attn.v_proj.weight")


def _compute_deltas_from_disk(model_id: str, instruct_model, dtype, log_fn=None):
    """
    Compute weight deltas by reading base model projection weights directly
    from safetensors files on disk, ONE TENSOR AT A TIME.

    Uses safe_open which memory-maps the file and reads individual tensors
    without loading the entire file. Peak additional memory = accumulated
    deltas only (~60% of one model copy).
    """
    if log_fn:
        log_fn("loading", f"Downloading/caching base model files: {model_id}")

    try:
        local_dir = snapshot_download(
            model_id, token=HF_TOKEN,
            allow_patterns=["*.safetensors", "*.json"],
        )
    except Exception as e:
        if log_fn:
            log_fn("error", f"Download failed: {e}")
        raise
    local_path = Path(local_dir)

    if log_fn:
        used, total, pct = _sys_mem()
        log_fn("memory", f"After download: {_mem_gb():.1f} GB process, "
               f"{used:.1f}/{total:.1f} GB system ({pct:.0f}%)")

    index_path = local_path / "model.safetensors.index.json"
    single_path = local_path / "model.safetensors"

    inst_sd = instruct_model.state_dict()

    deltas = {}
    delta_frob_norms = {}

    def _process_safetensor_file(fpath, keys_to_extract):
        """Read specific tensors from a safetensors file one at a time."""
        with safe_open(str(fpath), framework="pt") as f:
            available = set(f.keys())
            for key in keys_to_extract:
                if key in available and key in inst_sd:
                    base_tensor = f.get_tensor(key).to(dtype=dtype)
                    d = inst_sd[key] - base_tensor
                    deltas[key] = d
                    delta_frob_norms[key] = d.norm().item()
                    del base_tensor

    if index_path.exists():
        # Sharded model
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        shards_needed = {}
        for key, shard_file in weight_map.items():
            if any(pk in key for pk in PROJ_KEYS):
                if shard_file not in shards_needed:
                    shards_needed[shard_file] = []
                shards_needed[shard_file].append(key)

        if log_fn:
            total_keys = sum(len(v) for v in shards_needed.values())
            log_fn("computing", f"Computing {total_keys} deltas from "
                   f"{len(shards_needed)} shards...")

        for i, (shard_file, keys) in enumerate(shards_needed.items()):
            _process_safetensor_file(local_path / shard_file, keys)
            if log_fn:
                log_fn("computing",
                       f"  Shard {i+1}/{len(shards_needed)}: "
                       f"{len(deltas)} deltas, {_mem_gb():.1f} GB RSS")

    elif single_path.exists():
        # Single file — still reads one tensor at a time via safe_open
        if log_fn:
            log_fn("computing", "Computing deltas from model.safetensors...")

        with safe_open(str(single_path), framework="pt") as f:
            all_keys = [k for k in f.keys() if any(pk in k for pk in PROJ_KEYS)]

        if log_fn:
            log_fn("computing", f"  {len(all_keys)} projection weights to process")

        _process_safetensor_file(single_path, all_keys)

    else:
        raise FileNotFoundError(
            f"No safetensors files found in {local_path}. "
            f"Model may use pytorch .bin format (not yet supported).")

    del inst_sd
    gc.collect()

    if log_fn:
        total_mb = sum(t.numel() * t.element_size() for t in deltas.values()) / 1e6
        used, total, pct = _sys_mem()
        log_fn("memory", f"After deltas: {_mem_gb():.1f} GB process, "
               f"{used:.1f}/{total:.1f} GB system ({pct:.0f}%), "
               f"{len(deltas)} deltas ({total_mb:.0f} MB)")

    return deltas, delta_frob_norms


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

        self._unload()
        gc.collect()
        gc.collect()

        used, total, pct = _sys_mem()
        log("memory", f"After unload: {_mem_gb():.1f} GB process, "
            f"{used:.1f}/{total:.1f} GB system ({pct:.0f}%)")

        dtype = torch.float16
        state = ModelState(
            pair_id=pair_id, display_name=display,
            base_model_id=base_id, instruct_model_id=instruct_id,
            device=device, dtype=dtype,
        )

        t0 = time.time()

        # Step 1: Load instruct model (the only full model we keep in RAM)
        log("loading", f"Loading instruct model ({dtype}): {instruct_id}")
        state.model_instruct = AutoModelForCausalLM.from_pretrained(
            instruct_id, dtype=dtype, device_map=device,
            attn_implementation="eager", token=HF_TOKEN,
            low_cpu_mem_usage=True,
        )
        state.tokenizer = AutoTokenizer.from_pretrained(instruct_id, token=HF_TOKEN)
        if state.tokenizer.pad_token is None:
            state.tokenizer.pad_token = state.tokenizer.eos_token

        used, total, pct = _sys_mem()
        log("memory", f"After instruct load: {_mem_gb():.1f} GB process, "
            f"{used:.1f}/{total:.1f} GB system ({pct:.0f}%)")

        state.config = state.model_instruct.config
        state.n_layers = state.config.num_hidden_layers
        state.n_heads = state.config.num_attention_heads
        state.n_kv_heads = getattr(state.config, "num_key_value_heads", state.n_heads)
        state.hidden_size = state.config.hidden_size
        state.head_dim = state.hidden_size // state.n_heads

        mid_start = state.n_layers // 3
        mid_end = 2 * state.n_layers // 3
        state.signal_layers = list(range(mid_start, mid_end))

        # Step 2: Compute deltas directly from base model safetensors files.
        # Base model is NEVER loaded as a model object. Deltas computed
        # one shard at a time to minimize peak memory.
        deltas, frob_norms = _compute_deltas_from_disk(
            base_id, state.model_instruct, dtype=dtype, log_fn=log)

        state.deltas = deltas
        state.delta_frob_norms = frob_norms

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
            low_cpu_mem_usage=True,
        )

    def unload_base(self):
        if self.state and self.state.model_base is not None:
            del self.state.model_base
            self.state.model_base = None
            gc.collect()

    def install_analysis_hooks(self, full_trajectory: bool = False):
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

        for layer_idx in self.state.signal_layers:
            layer = model.model.layers[layer_idx]
            self._hooks.append(
                layer.input_layernorm.register_forward_hook(
                    make_output_hook(f"layer_{layer_idx}_h")))
            self._hooks.append(
                layer.self_attn.register_forward_hook(
                    make_attn_hook(f"layer_{layer_idx}_attn")))

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
        gc.collect()
