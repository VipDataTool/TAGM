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


def _memlog(msg, log_fn=None):
    """Print memory status to console AND progress log."""
    used, total, pct = _sys_mem()
    line = f"[MEM] {msg} | process={_mem_gb():.1f}GB, system={used:.1f}/{total:.1f}GB ({pct:.0f}%)"
    print(line, flush=True)
    if log_fn:
        log_fn("memory", line)

# ─── Model registry (loaded from models.json) ────────────────────
MODELS_FILE = Path(__file__).parent.parent / "models.json"

def _load_model_registry() -> dict:
    """Load model pairs from models.json. Returns {id: (base, instruct, name)}."""
    if not MODELS_FILE.exists():
        return {}
    with open(MODELS_FILE) as f:
        entries = json.load(f)
    return {e["id"]: (e["base"], e["instruct"], e["name"]) for e in entries}

def _save_model_registry(pairs: dict):
    """Save model pairs back to models.json."""
    entries = [{"id": k, "base": v[0], "instruct": v[1], "name": v[2]}
               for k, v in pairs.items()]
    with open(MODELS_FILE, "w") as f:
        json.dump(entries, f, indent=2)

KNOWN_PAIRS = _load_model_registry()

PROJ_KEYS = ["q_proj.weight", "k_proj.weight", "v_proj.weight",
             "o_proj.weight", "gate_proj.weight", "up_proj.weight"]


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
    base_safetensors_path: str = ""  # for on-demand delta computation
    device: str = "cpu"
    dtype: object = torch.float16
    loaded: bool = False
    full_deltas_available: bool = False  # True if deltas for ALL layers are loaded

    # Delta spectral structure (computed once at load time)
    delta_spectral: dict = field(default_factory=dict)  # {key: {eff_rank, top1_share, top5_share}}
    spectral_summary: dict = field(default_factory=dict)  # Aggregate stats

    def v_delta(self, layer_idx: int):
        return self.deltas.get(
            f"model.layers.{layer_idx}.self_attn.v_proj.weight")

    def o_delta(self, layer_idx: int):
        return self.deltas.get(
            f"model.layers.{layer_idx}.self_attn.o_proj.weight")


def _compute_deltas_from_disk(model_id: str, instruct_model, dtype,
                               layer_filter=None, log_fn=None):
    """
    Compute weight deltas by reading base model projection weights directly
    from safetensors files on disk, ONE TENSOR AT A TIME.

    layer_filter: if set, only compute deltas for these layer indices.
                  Dramatically reduces memory for large models.
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

    _memlog("After base model download", log_fn)

    index_path = local_path / "model.safetensors.index.json"
    single_path = local_path / "model.safetensors"

    inst_sd = instruct_model.state_dict()

    _memlog("After state_dict()", log_fn)

    deltas = {}
    delta_frob_norms = {}

    def _key_wanted(key):
        """Check if this weight key should be included."""
        if not any(pk in key for pk in PROJ_KEYS):
            return False
        if layer_filter is not None:
            # Extract layer index from key like "model.layers.5.self_attn.q_proj.weight"
            for li in layer_filter:
                if f"model.layers.{li}." in key:
                    return True
            return False
        return True

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
            if _key_wanted(key):
                if shard_file not in shards_needed:
                    shards_needed[shard_file] = []
                shards_needed[shard_file].append(key)

        print(f"[DELTA] Computing deltas from {len(shards_needed)} shards...", flush=True)

        for i, (shard_file, keys) in enumerate(shards_needed.items()):
            _process_safetensor_file(local_path / shard_file, keys)
            _memlog(f"Shard {i+1}/{len(shards_needed)}: {len(deltas)} deltas", log_fn)

    elif single_path.exists():
        print("[DELTA] Computing deltas from single safetensors file...", flush=True)

        with safe_open(str(single_path), framework="pt") as f:
            all_keys = [k for k in f.keys() if _key_wanted(k)]

        print(f"[DELTA] {len(all_keys)} projection weights to process", flush=True)
        _process_safetensor_file(single_path, all_keys)
        _memlog(f"After all deltas: {len(deltas)} total", log_fn)

    else:
        raise FileNotFoundError(
            f"No safetensors files found in {local_path}. "
            f"Model may use pytorch .bin format (not yet supported).")

    del inst_sd
    gc.collect()

    _memlog(f"Final: {len(deltas)} deltas", log_fn)

    return deltas, delta_frob_norms


def _compute_spectral_profile(state, log_fn=print):
    """Compute effective rank and spectral structure of each delta matrix.
    
    Effective rank = exp(entropy of normalized singular values).
    Low rank = RLHF made a surgical correction (few directions modified).
    High rank = RLHF reshaped the entire subspace.
    
    Computed once at load time. Zero per-prompt cost.
    """
    import numpy as np
    
    spectral = {}
    eff_ranks = []
    top1_shares = []
    
    for key, delta in state.deltas.items():
        try:
            # SVD on float32 for numerical stability
            d = delta.float().cpu()
            # For large matrices, use truncated SVD (top 64 singular values is enough)
            k = min(64, min(d.shape))
            U, S, Vh = torch.svd_lowrank(d, q=k)
            s = S.numpy()
            
            # Normalize singular values to a probability distribution
            s_norm = s / (s.sum() + 1e-10)
            s_nonzero = s_norm[s_norm > 1e-10]
            
            # Effective rank: exp(Shannon entropy of singular value distribution)
            ent = -np.sum(s_nonzero * np.log(s_nonzero))
            eff_rank = float(np.exp(ent))
            
            # Top-1 share: fraction of total spectral energy in the first singular value
            total_energy = float((s ** 2).sum())
            top1_energy = float(s[0] ** 2) / total_energy if total_energy > 0 else 0
            top5_energy = float((s[:5] ** 2).sum()) / total_energy if total_energy > 0 else 0
            
            spectral[key] = {
                'eff_rank': round(eff_rank, 2),
                'top1_share': round(top1_energy, 4),
                'top5_share': round(top5_energy, 4),
            }
            eff_ranks.append(eff_rank)
            top1_shares.append(top1_energy)
        except Exception:
            continue
    
    state.delta_spectral = spectral
    
    if eff_ranks:
        # Aggregate summary
        attn_ranks = [v['eff_rank'] for k, v in spectral.items() if 'self_attn' in k]
        mlp_ranks = [v['eff_rank'] for k, v in spectral.items() if 'mlp' in k]
        
        state.spectral_summary = {
            'mean_eff_rank': round(float(np.mean(eff_ranks)), 2),
            'std_eff_rank': round(float(np.std(eff_ranks)), 2),
            'mean_top1_share': round(float(np.mean(top1_shares)), 4),
            'attn_mean_rank': round(float(np.mean(attn_ranks)), 2) if attn_ranks else 0,
            'mlp_mean_rank': round(float(np.mean(mlp_ranks)), 2) if mlp_ranks else 0,
            'n_sublayers': len(eff_ranks),
        }
        log_fn("info", f"Spectral profile: {len(eff_ranks)} sublayers, "
               f"mean rank={state.spectral_summary['mean_eff_rank']:.1f}, "
               f"attn={state.spectral_summary['attn_mean_rank']:.1f}, "
               f"mlp={state.spectral_summary['mlp_mean_rank']:.1f}")


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

        _memlog("After unload", log)

        dtype = torch.float16
        state = ModelState(
            pair_id=pair_id, display_name=display,
            base_model_id=base_id, instruct_model_id=instruct_id,
            device=device, dtype=dtype,
        )

        t0 = time.time()

        # Step 1: Load instruct model (the only full model we keep in RAM)
        print(f"[LOAD] Loading instruct model ({dtype}): {instruct_id}", flush=True)
        log("loading", f"Loading instruct model ({dtype}): {instruct_id}")
        state.model_instruct = AutoModelForCausalLM.from_pretrained(
            instruct_id, dtype=dtype, device_map=device,
            attn_implementation="eager", token=HF_TOKEN,
            low_cpu_mem_usage=True,
        )
        state.tokenizer = AutoTokenizer.from_pretrained(instruct_id, token=HF_TOKEN)
        if state.tokenizer.pad_token is None:
            state.tokenizer.pad_token = state.tokenizer.eos_token

        _memlog("After instruct model load", log)

        state.config = state.model_instruct.config
        state.n_layers = state.config.num_hidden_layers
        state.n_heads = state.config.num_attention_heads
        state.n_kv_heads = getattr(state.config, "num_key_value_heads", state.n_heads)
        state.hidden_size = state.config.hidden_size
        state.head_dim = state.hidden_size // state.n_heads

        mid_start = state.n_layers // 3
        mid_end = 2 * state.n_layers // 3
        state.signal_layers = list(range(mid_start, mid_end))

        # Check available memory to decide whether to compute all-layer
        # or signal-layer-only deltas
        _, sys_total, sys_pct = _sys_mem()
        free_gb = sys_total * (1 - sys_pct / 100)
        # Rough estimate: each layer has ~5 projection deltas, each ~(hidden^2 * 2 bytes)
        est_all_deltas_gb = state.n_layers * 5 * (state.hidden_size ** 2) * 2 / 1e9
        est_signal_deltas_gb = len(state.signal_layers) * 5 * (state.hidden_size ** 2) * 2 / 1e9

        if free_gb > est_all_deltas_gb * 1.5:
            layer_filter = None  # enough RAM for all layers
            print(f"[LOAD] {free_gb:.1f}GB free, loading ALL layer deltas "
                  f"(est {est_all_deltas_gb:.1f}GB)", flush=True)
        else:
            layer_filter = state.signal_layers
            print(f"[LOAD] {free_gb:.1f}GB free, loading SIGNAL LAYER deltas only "
                  f"(est {est_signal_deltas_gb:.1f}GB, layers {state.signal_layers})",
                  flush=True)

        # Step 2: Compute deltas from base model safetensors files.
        deltas, frob_norms = _compute_deltas_from_disk(
            base_id, state.model_instruct, dtype=dtype,
            layer_filter=layer_filter, log_fn=log)

        state.deltas = deltas
        state.delta_frob_norms = frob_norms
        state.full_deltas_available = (layer_filter is None)

        # Compute delta spectral structure (effective rank per sublayer)
        try:
            _compute_spectral_profile(state, log)
        except Exception as e:
            log("warning", f"Spectral profile computation failed: {e}")

        # Save the base path for on-demand delta computation
        try:
            state.base_safetensors_path = snapshot_download(
                base_id, token=HF_TOKEN,
                allow_patterns=["*.safetensors", "*.json"],
                local_files_only=True,  # already cached
            )
        except Exception:
            state.base_safetensors_path = ""

        elapsed = time.time() - t0
        n_o_proj = sum(1 for k in state.deltas if 'o_proj' in k)
        log("ready", f"Loaded {display} in {elapsed:.1f}s "
            f"({state.n_layers}L, {state.hidden_size}d, "
            f"{len(state.deltas)} deltas, {n_o_proj} o_proj)")

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

    def install_analysis_hooks(self, full_trajectory: bool = False,
                               ltp_layers: list = None):
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

        # Always hook signal layers
        hooked = set()
        for layer_idx in self.state.signal_layers:
            layer = model.model.layers[layer_idx]
            self._hooks.append(
                layer.input_layernorm.register_forward_hook(
                    make_output_hook(f"layer_{layer_idx}_h")))
            self._hooks.append(
                layer.self_attn.register_forward_hook(
                    make_attn_hook(f"layer_{layer_idx}_attn")))
            hooked.add(layer_idx)

        # Hook additional LTP layers (e.g. late layers) if requested
        if ltp_layers:
            for layer_idx in ltp_layers:
                if layer_idx not in hooked and layer_idx < len(model.model.layers):
                    layer = model.model.layers[layer_idx]
                    self._hooks.append(
                        layer.input_layernorm.register_forward_hook(
                            make_output_hook(f"layer_{layer_idx}_h")))
                    hooked.add(layer_idx)
            print(f"[HOOKS] LTP late-layer hooks installed: {sorted(ltp_layers)} "
                  f"(total hooked: {len(hooked)})", flush=True)
        else:
            print(f"[HOOKS] Signal-only hooks: {sorted(hooked)} (no LTP layers)", flush=True)

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
