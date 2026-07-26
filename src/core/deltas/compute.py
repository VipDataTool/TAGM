"""Compute weight deltas by reading base model weights from safetensors, one
tensor at a time, without ever instantiating the full base model.

This is the single largest memory-discipline decision in TAGM (inherited
from TASM). For a 7B model pair, instantiating both models concurrently
would require ~28GB just to diff their weights; streaming from disk
requires ~28GB peak only during the brief window where a single delta
is being computed.

Translated from TASM's `engine/model_manager.py::_compute_deltas_from_disk`,
with two changes: (1) projection key discovery is delegated to the adapter
(`adapter.is_projection_key`, `adapter.parse_projection_key`) rather than
hardcoded PROJ_KEYS list; (2) output is written into a DeltaStore addressed
by (layer, role) rather than into a string-keyed dict.
"""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from src.core.deltas.store import DeltaStore, DeltaStoreMetadata
from src.core.types import ProgressCallback, noop_progress

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from src.core.adapter.base import ModelAdapter

logger = logging.getLogger("src")


def compute_deltas_from_disk(
    base_model_id: str,
    instruct_model: "PreTrainedModel",
    adapter: "ModelAdapter",
    dtype: torch.dtype,
    layer_filter: Optional[list[int]] = None,
    hf_token: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    store: Optional["DeltaStore"] = None,
    streaming: bool = False,
) -> "DeltaStore":
    """Compute weight deltas (instruct - base) for projection weights.

    Args:
      base_model_id:   HuggingFace model ID for the base model.
      instruct_model:  Already-loaded instruct model. Its state_dict is the
                       source of instruct weights; no base model is instantiated.
      adapter:         Adapter instance for the model family. Used to decide
                       which weight keys are "projection" weights we care about.
      dtype:           Torch dtype for delta tensors. Base tensors are cast
                       to this dtype before subtraction; deltas inherit it.
      layer_filter:    If set, only compute deltas for these layer indices.
                       If None, compute for all layers declared by the model.
      hf_token:        Optional HuggingFace token for downloading gated models.
      progress:        Optional callback for loading/memory reports.
      streaming:       If True (HEP mode), download and delete base model
                       shards one at a time to minimize peak disk usage.

    Returns:
      A populated DeltaStore addressed by (layer_idx, role).

    Raises:
      FileNotFoundError:   if the base model ships in .bin format (not supported;
                           only safetensors is handled by this streaming path).
      Exception:           downloads failures propagate from huggingface_hub.
    """
    from huggingface_hub import snapshot_download, hf_hub_download
    from safetensors.torch import safe_open

    log = progress or noop_progress

    # ── Build metadata ──────────────────────────────────────────────
    n_layers = adapter.n_layers(instruct_model)
    instruct_id = getattr(instruct_model, "name_or_path", "unknown")
    metadata = DeltaStoreMetadata(
        base_model_id=base_model_id,
        instruct_model_id=instruct_id,
        adapter_family=adapter.family_id,
        dtype=str(dtype).replace("torch.", ""),
        layer_filter=sorted(layer_filter) if layer_filter is not None else None,
        n_layers=n_layers,
        n_deltas=0,
    )
    if store is None:
        store = DeltaStore(adapter, metadata)
    else:
        store._metadata = metadata

    inst_sd = instruct_model.state_dict()

    def _key_wanted(key: str) -> bool:
        if not adapter.is_projection_key(key):
            return False
        if layer_filter is None:
            return True
        parsed = adapter.parse_projection_key(key)
        if parsed is None:
            return False
        _, layer_idx = parsed
        return layer_idx in layer_filter

    _delta_count = [0]  # mutable counter for closure
    # Keys present in the base checkpoint but absent from the instruct model
    # (or vice versa).  Skipping these silently is how a MISMATCHED PAIR ends
    # up producing a partially-populated store, which surfaces later as a
    # LayerNotComputedError that blames the layer filter — the wrong cause.
    _skipped_keys = []
    _dtype_narrowed = [False]

    def _process_file(fpath: Path, keys_to_extract: list[str]) -> None:
        n_keys = len(keys_to_extract)
        with safe_open(str(fpath), framework="pt") as f:
            available = set(f.keys())
            for ki, key in enumerate(keys_to_extract):
                if key not in available or key not in inst_sd:
                    _skipped_keys.append(key)
                    continue
                base_raw = f.get_tensor(key)
                inst_tensor = inst_sd[key]
                if inst_tensor.shape != base_raw.shape:
                    # Different hidden size / vocab size / head count.
                    _skipped_keys.append(f"{key}(shape {tuple(base_raw.shape)}"
                                         f" vs {tuple(inst_tensor.shape)})")
                    del base_raw
                    continue
                # Casting the base tensor to `dtype` BEFORE the subtraction
                # truncates it.  When the checkpoints ship fp32/fp16 and dtype
                # is bf16 (8 mantissa bits) that truncation error is of the
                # same order as the delta itself, so difference first and
                # narrow after.
                if base_raw.dtype != dtype or inst_tensor.dtype != dtype:
                    _dtype_narrowed[0] = True
                    common = torch.promote_types(base_raw.dtype, inst_tensor.dtype)
                    delta = (inst_tensor.to(common) - base_raw.to(common)).to(dtype)
                else:
                    delta = inst_tensor - base_raw
                base_tensor = base_raw
                parsed = adapter.parse_projection_key(key)
                if parsed is None:
                    del base_tensor, delta
                    continue
                role, layer_idx = parsed
                store.put(layer_idx, role, delta)
                del base_tensor, delta
                _delta_count[0] += 1
                if (ki + 1) % 10 == 0 or ki == n_keys - 1:
                    log("deltas", f"  delta {_delta_count[0]}: "
                                  f"L{layer_idx}.{role} [{ki+1}/{n_keys}]")

    if streaming:
        # ── HEP streaming: download one shard at a time ─────────────
        # Downloads the index.json first to discover the shard layout,
        # then fetches, processes, and deletes each shard individually.
        # Peak disk = instruct cache + ONE shard + growing mmap file,
        # instead of instruct cache + ALL shards + mmap file.
        import tempfile, shutil

        log("loading", f"HEP streaming: downloading index for {base_model_id}")

        # Download just the index file (tiny) to discover shard layout
        try:
            index_file = hf_hub_download(
                base_model_id,
                "model.safetensors.index.json",
                token=hf_token,
            )
            with open(index_file) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            sharded = True
        except Exception:
            # Single-file model (no index) — download the whole thing
            sharded = False

        if sharded:
            shards_needed: dict[str, list[str]] = {}
            for key, shard_file in weight_map.items():
                if _key_wanted(key):
                    shards_needed.setdefault(shard_file, []).append(key)

            log("deltas", f"HEP streaming: {len(shards_needed)} shard(s) to process")
            scratch = Path(tempfile.mkdtemp(prefix="tagm_hep_"))

            try:
                for i, (shard_file, keys) in enumerate(shards_needed.items()):
                    log("deltas", f"Downloading shard {i+1}/{len(shards_needed)}: "
                                  f"{shard_file}")
                    shard_path = Path(hf_hub_download(
                        base_model_id, shard_file,
                        token=hf_token, local_dir=scratch,
                    ))
                    _process_file(shard_path, keys)
                    log("deltas", f"Shard {i+1}/{len(shards_needed)}: "
                                  f"{len(store)} deltas accumulated")
                    # Delete the shard immediately to free disk
                    try:
                        shard_path.unlink()
                    except Exception:
                        pass
                    gc.collect()
            finally:
                # Clean up scratch directory
                try:
                    shutil.rmtree(scratch, ignore_errors=True)
                except Exception:
                    pass

        else:
            # Single-file model — download, process, delete
            log("deltas", "HEP streaming: single safetensors file")
            scratch = Path(tempfile.mkdtemp(prefix="tagm_hep_"))
            try:
                sf_path = Path(hf_hub_download(
                    base_model_id, "model.safetensors",
                    token=hf_token, local_dir=scratch,
                ))
                with safe_open(str(sf_path), framework="pt") as f:
                    all_keys = [k for k in f.keys() if _key_wanted(k)]
                log("deltas", f"Computing deltas from single file: "
                              f"{len(all_keys)} projection weight(s)")
                _process_file(sf_path, all_keys)
                try:
                    sf_path.unlink()
                except Exception:
                    pass
            finally:
                try:
                    shutil.rmtree(scratch, ignore_errors=True)
                except Exception:
                    pass

    else:
        # ── Standard mode: snapshot_download (all at once) ──────────
        log("loading", f"Downloading/caching base model files: {base_model_id}")
        try:
            local_dir = snapshot_download(
                base_model_id,
                token=hf_token,
                allow_patterns=["*.safetensors", "*.json"],
            )
        except Exception as e:
            log("error", f"Base model download failed: {e}")
            raise
        local_path = Path(local_dir)
        log("deltas", "Base model files cached on disk")

        index_path = local_path / "model.safetensors.index.json"
        single_path = local_path / "model.safetensors"

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            shards_needed: dict[str, list[str]] = {}
            for key, shard_file in weight_map.items():
                if _key_wanted(key):
                    shards_needed.setdefault(shard_file, []).append(key)
            log("deltas", f"Computing deltas from {len(shards_needed)} shard(s)")
            for i, (shard_file, keys) in enumerate(shards_needed.items()):
                _process_file(local_path / shard_file, keys)
                log("deltas", f"Shard {i+1}/{len(shards_needed)}: "
                              f"{len(store)} deltas accumulated")
        elif single_path.exists():
            with safe_open(str(single_path), framework="pt") as f:
                all_keys = [k for k in f.keys() if _key_wanted(k)]
            log("deltas", f"Computing deltas from single safetensors file: "
                          f"{len(all_keys)} projection weight(s) to process")
            _process_file(single_path, all_keys)
        else:
            raise FileNotFoundError(
                f"No safetensors files found in {local_path}. "
                f"TAGM does not support pytorch .bin-format models — convert "
                f"the base model to safetensors or use a different model pair."
            )

    # ── Cleanup ─────────────────────────────────────────────────────
    del inst_sd
    gc.collect()

    store.metadata.n_deltas = len(store)
    log("deltas", f"Delta computation complete: {len(store)} deltas, "
                   f"{store.total_bytes() / 1e9:.1f}GB")

    # ── Pair-compatibility report ───────────────────────────────────
    # Nothing else compares the two checkpoints, so an incompatible pair is
    # otherwise invisible until a much later, misleading error.
    if _dtype_narrowed[0]:
        log("deltas", f"note: checkpoint dtype differs from the requested "
                      f"{str(dtype).replace('torch.', '')}; deltas were "
                      f"differenced at the wider dtype and narrowed after")
    if _skipped_keys:
        preview = ", ".join(_skipped_keys[:5])
        more = f" (+{len(_skipped_keys) - 5} more)" if len(_skipped_keys) > 5 else ""
        logger.warning(
            "[deltas] %d projection weight(s) present in one checkpoint but "
            "not the other, or shape-mismatched: %s%s — this usually means "
            "'%s' and '%s' are not a matched base/instruct pair",
            len(_skipped_keys), preview, more, base_model_id, instruct_id)
        log("deltas", f"WARNING: {len(_skipped_keys)} weight(s) skipped — "
                      f"'{base_model_id}' may not be the base of "
                      f"'{instruct_id}'")

    expected = len(layer_filter or range(n_layers)) * len(adapter.PROJECTION_ROLES)
    if len(store) != expected:
        logger.warning(
            "[deltas] store holds %d deltas but %d were expected "
            "(%d layers x %d roles); downstream layer lookups will raise "
            "LayerNotComputedError for the gaps",
            len(store), expected,
            len(layer_filter or range(n_layers)), len(adapter.PROJECTION_ROLES))

    return store
