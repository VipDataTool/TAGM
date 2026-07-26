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
import hashlib
import json
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
        # use_safetensors=True refuses .bin checkpoints, which go through
        # torch's pickle loader.  Model ids arrive unvalidated from the HTTP
        # API, so this keeps an arbitrary repo id from reaching a deserializer.
        self.instruct_model = AutoModelForCausalLM.from_pretrained(
            self.instruct_model_id,
            dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
            token=self.hf_token,
            low_cpu_mem_usage=True,
            use_safetensors=True,
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
        skip_delta_computation = False

        if delta_backend == "mmap":
            from src.core.deltas.store import MmapDeltaStore, DeltaStoreMetadata
            # The cache key must cover EVERY input that changes the stored
            # deltas.  It previously used only instruct_model_id, so re-running
            # the same instruct model against a DIFFERENT BASE silently reused
            # the old deltas — and the file format carries no model identity,
            # so nothing downstream could detect it.  The integrity spot-check
            # below only verifies a tensor is finite and matches its own stored
            # norm; it cannot tell right-base data from wrong-base data.
            safe_id = self.instruct_model_id.replace("/", "__").replace("\\", "__")
            cache_key = "|".join([
                self.instruct_model_id,
                self.base_model_id or "",
                self.adapter.family_id,
                str(self.dtype),
                repr(sorted(layer_filter) if layer_filter else None),
            ])
            key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
            cache_dir = Path.home() / ".tagm" / "cache" / "deltas"
            cache_dir.mkdir(parents=True, exist_ok=True)
            mmap_path = cache_dir / f"{safe_id}.{key_hash}.tagm"

            # Reclaim caches written by the old naming scheme ({safe_id}.tagm,
            # keyed on the instruct model alone).  They can never be reused —
            # the key now includes a hash — so leaving them would silently leak
            # multiple GB per model pair.
            legacy_path = cache_dir / f"{safe_id}.tagm"
            if legacy_path.exists() and legacy_path != mmap_path:
                try:
                    freed = legacy_path.stat().st_size
                    legacy_path.unlink()
                    log("deltas", f"Removed legacy delta cache "
                                  f"{legacy_path.name} ({freed / 1e9:.1f} GB) — "
                                  f"it was keyed on the instruct model only")
                except OSError as e:
                    log("deltas", f"Could not remove legacy delta cache "
                                  f"{legacy_path.name}: {e}")

            # A sidecar recording exactly what produced this file, so a stale
            # or hand-copied cache can be rejected rather than trusted.
            sidecar_path = mmap_path.with_name(mmap_path.name + ".json")
            expected_sidecar = {
                "instruct_model_id": self.instruct_model_id,
                "base_model_id": self.base_model_id,
                "adapter_family": self.adapter.family_id,
                "dtype": str(self.dtype),
                "layer_filter": sorted(layer_filter) if layer_filter else None,
                "format_version": 1,
            }

            placeholder_meta = DeltaStoreMetadata(
                base_model_id=self.base_model_id,
                instruct_model_id=self.instruct_model_id,
                adapter_family=self.adapter.family_id,
                dtype=str(self.dtype).replace("torch.", ""),
                layer_filter=None, n_layers=0, n_deltas=0,
            )

            # Check for cached mmap file from a previous load.
            # Reject it unless the sidecar matches this exact pair/dtype/filter.
            sidecar_ok = False
            if sidecar_path.exists():
                try:
                    sidecar_ok = (
                        json.loads(sidecar_path.read_text()) == expected_sidecar)
                except Exception as e:
                    log("deltas", f"HEP: unreadable cache sidecar ({e}); recomputing")
            elif mmap_path.exists():
                log("deltas", "HEP: cached mmap has no sidecar (written by an "
                              "older build); recomputing to guarantee the "
                              "deltas match this model pair")

            if sidecar_ok and mmap_path.exists() and mmap_path.stat().st_size > 256:
                try:
                    cached_store = MmapDeltaStore(
                        self.adapter, placeholder_meta, mmap_path, dtype=self.dtype)
                    if cached_store._mode == "read" and len(cached_store) > 0:
                        # Spot-check: read one tensor to verify file integrity
                        test_key = next(iter(cached_store._index))
                        test_tensor = cached_store.get(*test_key)
                        n_entries = len(cached_store)
                        test_shape = tuple(test_tensor.shape)
                        test_norm = float(test_tensor.norm().item())
                        stored_norm = cached_store._frob_norms.get(test_key, 0)
                        del test_tensor

                        # Validate: shape should be 2D, norm should be finite
                        # and roughly match the stored norm
                        if (len(test_shape) != 2 or test_norm != test_norm or
                                test_norm == 0 or
                                (stored_norm > 0 and
                                 abs(test_norm - stored_norm) / stored_norm > 0.01)):
                            log("deltas", f"HEP: cached mmap failed validation "
                                          f"(shape={test_shape}, norm={test_norm:.4f}, "
                                          f"stored_norm={stored_norm:.4f})")
                            cached_store.close()
                            mmap_path.unlink(missing_ok=True)
                        else:
                            # Replace the placeholder metadata with the real
                            # thing.  Leaving layer_filter=None / n_layers=0 in
                            # place made full_deltas_available() return True for
                            # a filtered store and made LayerNotComputedError
                            # report "n_layers=0", which points at the wrong
                            # cause.
                            cached_store._metadata = DeltaStoreMetadata(
                                base_model_id=self.base_model_id,
                                instruct_model_id=self.instruct_model_id,
                                adapter_family=self.adapter.family_id,
                                dtype=str(self.dtype).replace("torch.", ""),
                                layer_filter=(sorted(layer_filter)
                                              if layer_filter else None),
                                n_layers=self.adapter.n_layers(self.instruct_model),
                                n_deltas=n_entries,
                            )
                            self.delta_store = cached_store
                            skip_delta_computation = True
                            log("deltas", f"HEP: reusing cached mmap — {n_entries} "
                                          f"deltas, {mmap_path.stat().st_size / 1e6:.0f} MB, "
                                          f"validated L{test_key[0]}.{test_key[1]} "
                                          f"shape={test_shape} norm={test_norm:.4f}")
                    else:
                        log("deltas", f"HEP: cached mmap empty or not readable, "
                                      f"recomputing")
                        cached_store.close()
                        mmap_path.unlink(missing_ok=True)
                except Exception as e:
                    log("deltas", f"HEP: cached mmap failed ({type(e).__name__}: {e}), "
                                  f"recomputing")
                    mmap_path.unlink(missing_ok=True)

            if not skip_delta_computation:
                # Remove any partial file and create fresh.  The sidecar is
                # written only after the store is fully flushed (below), so an
                # interrupted run leaves no sidecar and is never reused.
                mmap_path.unlink(missing_ok=True)
                sidecar_path.unlink(missing_ok=True)
                pre_store = MmapDeltaStore(
                    self.adapter, placeholder_meta, mmap_path, dtype=self.dtype)
                self._pending_delta_sidecar = (sidecar_path, expected_sidecar)
                log("deltas", f"Using mmap delta store: {mmap_path.name}")

        if not skip_delta_computation:
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
                # Only now is the file complete.  Writing the sidecar here (and
                # not at creation time) means an interrupted or failed delta
                # computation leaves a file with no sidecar, which the check
                # above refuses to reuse.
                pending = getattr(self, "_pending_delta_sidecar", None)
                if pending is not None:
                    sc_path, sc_data = pending
                    try:
                        sc_path.write_text(json.dumps(sc_data, indent=2))
                    except Exception as e:
                        log("deltas", f"HEP: could not write cache sidecar ({e}); "
                                      f"this cache will be recomputed next load")
                    self._pending_delta_sidecar = None

            # HEP: evict base model from HF cache after deltas are computed.
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
        # See the note on the instruct load above.
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
            token=self.hf_token,
            low_cpu_mem_usage=True,
            use_safetensors=True,
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
