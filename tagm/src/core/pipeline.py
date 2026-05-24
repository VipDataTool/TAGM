"""Pipeline: model loading, delta computation, and lifecycle management.

Owns a loaded model pair (instruct + base), a DeltaStore, a tokenizer,
and an adapter. The engine's Analyzer uses these directly for inference;
the Pipeline itself is the model-infrastructure coordinator.

Lifecycle:
    1. Construct with model IDs.
    2. load(): load instruct model, auto-detect adapter, compute deltas.
    3. Pass the Pipeline to Analyzer(pipeline) for computation.
    4. unload() when done.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from src.core.adapter.registry import find_adapter
from src.core.deltas.compute import compute_deltas_from_disk
from src.core.deltas.spectral import compute_spectral_profile
from src.core.types import ProgressCallback, noop_progress

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from src.core.adapter.base import ModelAdapter
    from src.core.deltas.store import DeltaStore


class Pipeline:
    """Owns a loaded model pair and provides model infrastructure.

    All model-family knowledge is delegated to the adapter. All delta
    storage goes through DeltaStore. The engine's Analyzer handles
    forward passes and activation capture.
    """

    def __init__(
        self,
        instruct_model_id: str,
        base_model_id: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        hf_token: Optional[str] = None,
        adapter: Optional["ModelAdapter"] = None,
    ):
        self.instruct_model_id = instruct_model_id
        self.base_model_id = base_model_id
        self.device = device
        self.dtype = dtype
        self.hf_token = hf_token
        self._explicit_adapter = adapter

        self.adapter: Optional["ModelAdapter"] = None
        self.instruct_model: Optional["PreTrainedModel"] = None
        self.base_model: Optional["PreTrainedModel"] = None
        self.tokenizer: Optional["PreTrainedTokenizer"] = None
        self.delta_store: Optional["DeltaStore"] = None
        self.inference_class: str = "instruct"

        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def active_model(self):
        """Return the model selected by inference_class."""
        if self.inference_class == "base" and self.base_model is not None:
            return self.base_model
        return self.instruct_model

    def load(
        self,
        layer_filter: Optional[list[int]] = None,
        compute_spectral: bool = True,
        svd_k: int = 64,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        """Load the instruct model, detect adapter, compute deltas."""
        from transformers import AutoModelForCausalLM

        log = progress or noop_progress

        log("loading", f"Loading instruct model: {self.instruct_model_id}")
        self.instruct_model = AutoModelForCausalLM.from_pretrained(
            self.instruct_model_id,
            dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
            token=self.hf_token,
            low_cpu_mem_usage=True,
        )

        if self._explicit_adapter is not None:
            self.adapter = self._explicit_adapter
        else:
            self.adapter = find_adapter(self.instruct_model)
        log("loading", f"Adapter: {self.adapter.family_display_name} "
                       f"({self.adapter.family_id})")

        self.tokenizer = self.adapter.load_tokenizer(
            self.instruct_model_id, hf_token=self.hf_token)

        log("deltas", "Computing weight deltas from base model")

        # Select delta store backend
        from src.engine import config as engine_config
        delta_backend = engine_config.get("delta_backend")
        pre_store = None

        if delta_backend == "mmap":
            from src.core.deltas.store import MmapDeltaStore, DeltaStoreMetadata
            safe_id = self.instruct_model_id.replace("/", "__").replace("\\", "__")
            cache_dir = Path.home() / ".tagm" / "cache" / "deltas"
            cache_dir.mkdir(parents=True, exist_ok=True)
            mmap_path = cache_dir / f"{safe_id}.tagm"
            # Remove stale file from a previous load
            if mmap_path.exists():
                mmap_path.unlink()
            placeholder_meta = DeltaStoreMetadata(
                base_model_id=self.base_model_id,
                instruct_model_id=self.instruct_model_id,
                adapter_family=self.adapter.family_id,
                dtype=str(self.dtype).replace("torch.", ""),
                layer_filter=None, n_layers=0, n_deltas=0,
            )
            pre_store = MmapDeltaStore(
                self.adapter, placeholder_meta, mmap_path, dtype=self.dtype)
            log("deltas", f"Using mmap delta store: {mmap_path.name}")

        self.delta_store = compute_deltas_from_disk(
            base_model_id=self.base_model_id,
            instruct_model=self.instruct_model,
            adapter=self.adapter,
            dtype=self.dtype,
            layer_filter=layer_filter,
            hf_token=self.hf_token,
            progress=progress,
            store=pre_store,
            streaming=(delta_backend == "mmap"),
        )

        # If mmap, finalize the file and reopen in read mode
        if delta_backend == "mmap" and hasattr(self.delta_store, 'reopen_readonly'):
            self.delta_store.reopen_readonly()

        # HEP: evict base model from HF cache after deltas are computed.
        # The base weights were only needed for subtraction; the mmap file
        # now holds the deltas. Evicting frees ~6GB of disk for 3B models.
        if delta_backend == "mmap" and engine_config.get("hep_evict_base_cache"):
            from src.core.cache import evict_hf_model
            evict_result = evict_hf_model(self.base_model_id)
            if evict_result["removed"]:
                freed_gb = evict_result["bytes_freed"] / 1e9
                log("deltas", f"HEP: evicted base model cache "
                              f"({self.base_model_id}), freed {freed_gb:.1f} GB")
            gc.collect()

        if compute_spectral:
            log("spectral", "Computing delta spectral profile")
            compute_spectral_profile(
                self.delta_store,
                svd_k=svd_k,
                keep_singular_values=True,
                progress=progress,
            )

        self._loaded = True
        log("ready", "Pipeline ready")

    def load_base(self, progress: Optional[ProgressCallback] = None) -> None:
        """Load the base model for live base inference."""
        from transformers import AutoModelForCausalLM

        log = progress or noop_progress
        if self.base_model is not None:
            return
        log("loading", f"Loading base model: {self.base_model_id}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
            token=self.hf_token,
            low_cpu_mem_usage=True,
        )

    def unload_base(self) -> None:
        """Free the base model."""
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload(self) -> None:
        """Release all model state."""
        if self.instruct_model is not None:
            del self.instruct_model
            self.instruct_model = None
        self.unload_base()
        # Close mmap delta store if applicable
        if self.delta_store is not None and hasattr(self.delta_store, 'close'):
            try:
                self.delta_store.close()
            except Exception:
                pass
        self.delta_store = None
        self.tokenizer = None
        self.adapter = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self) -> dict:
        """Structured summary of loaded-model state."""
        if not self._loaded:
            return {"loaded": False}
        n_heads, n_kv = self.adapter.attention_heads(self.instruct_model)
        return {
            "loaded": True,
            "adapter": {
                "family_id": self.adapter.family_id,
                "display_name": self.adapter.family_display_name,
            },
            "model_pair": {
                "instruct": self.instruct_model_id,
                "base": self.base_model_id,
                "device": self.device,
                "dtype": str(self.dtype).replace("torch.", ""),
            },
            "structure": {
                "n_layers": self.adapter.n_layers(self.instruct_model),
                "hidden_size": self.adapter.hidden_size(self.instruct_model),
                "n_attention_heads": n_heads,
                "n_kv_heads": n_kv,
                "head_dim": self.adapter.head_dim(self.instruct_model),
                "vocab_size": self.adapter.vocab_size(self.instruct_model),
            },
            "deltas": {
                "n_deltas": len(self.delta_store),
                "layer_filter": self.delta_store.layer_filter,
                "full_deltas_available": self.delta_store.full_deltas_available,
                "total_bytes": self.delta_store.total_bytes(),
                "spectral": self.delta_store.aggregate_spectral_summary(),
            },
            "base_model_loaded": self.base_model is not None,
        }
