"""DeltaStore: canonical weight-delta container for a loaded model pair.

Addressed as `store.get(layer_idx, role) -> tensor`. Role is one of
"q" | "k" | "v" | "o" | "gate" | "up" | "down" (subset depends on the
adapter family, but Qwen2/Llama3 both expose all seven).

Persists for the model-pair's lifetime in the Pipeline. A DeltaStore
records the `layer_filter` it was built under; a consumer asking for
a layer outside the filter gets a descriptive `LayerNotComputedError`
rather than a bare KeyError — so "I need layer 4 and it wasn't loaded"
is distinguishable from "layer 4 doesn't exist on this model."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from src.core.adapter.base import ModelAdapter


class LayerNotComputedError(KeyError):
    """Raised when a consumer asks for a layer that exists on the model
    but was excluded by this DeltaStore's `layer_filter`.

    Distinct from a plain KeyError so callers can surface it as
    "re-load with a wider layer filter" rather than "this role/layer
    doesn't exist on this family."
    """
    def __init__(self, layer: int, role: str, filter_: Optional[list[int]],
                 n_layers: int):
        self.layer = layer
        self.role = role
        self.layer_filter = filter_
        self.n_layers = n_layers
        msg = (f"Delta for layer={layer}, role='{role}' was not computed. "
               f"Model has {n_layers} layers; ")
        if filter_ is None:
            msg += "DeltaStore was built over all layers, so this role "\
                   "may not be declared by the adapter."
        else:
            msg += (f"DeltaStore was built with layer_filter={sorted(filter_)}. "
                    f"Reload with a filter that includes layer {layer}.")
        super().__init__(msg)


@dataclass
class DeltaSpectralSummary:
    """Per-delta spectral structure, computed once at load time.

    Stored on the DeltaStore alongside each delta tensor. Consumed by
    SFD (which builds its own QK cache on top) and by the frontend's
    spectral-summary display. All fields except singular_values are
    scalar summaries cheap to serialize.
    """
    eff_rank: float              # exp(Shannon entropy of normalized singular values)
    top1_share: float            # fraction of spectral energy in the top singular value
    top5_share: float            # fraction of spectral energy in the top 5
    stable_rank: float           # ||delta||_F^2 / sigma_1^2
    frob_norm: float             # Frobenius norm of the delta
    singular_values: Optional[torch.Tensor] = None
    # Full top-k singular values (as returned by torch.svd_lowrank with q=k).
    # Kept optionally so higher-k measurements can reuse; may be None to
    # save memory if only scalars were requested.


@dataclass
class DeltaStoreMetadata:
    """Record of how this DeltaStore was built, carried for introspection."""
    base_model_id: str
    instruct_model_id: str
    adapter_family: str
    dtype: str
    layer_filter: Optional[list[int]]
    n_layers: int
    n_deltas: int


class DeltaStore:
    """Canonical storage for weight deltas.

    Address: store.get(layer_idx, role) -> tensor

    Internally holds a dict keyed by (layer_idx, role). Also holds
    per-delta Frobenius norms (needed as normalizers by most measurements
    that use deltas) and per-delta spectral summaries (computed once at
    load time, read many times).
    """

    def __init__(self, adapter: "ModelAdapter", metadata: DeltaStoreMetadata):
        self._adapter = adapter
        self._metadata = metadata
        self._data: dict[tuple[int, str], torch.Tensor] = {}
        self._frob_norms: dict[tuple[int, str], float] = {}
        self._spectral: dict[tuple[int, str], DeltaSpectralSummary] = {}

    # ── Writing (used by compute.py) ────────────────────────────────
    def put(self, layer_idx: int, role: str, tensor: torch.Tensor,
            frob_norm: Optional[float] = None) -> None:
        """Store a delta tensor and (optionally pre-computed) Frobenius norm.

        If frob_norm is None it is computed from the tensor. Adapter
        role validation is delegated to compute.py which uses the
        adapter to derive roles; the store itself doesn't re-validate.
        """
        self._data[(layer_idx, role)] = tensor
        self._frob_norms[(layer_idx, role)] = (
            float(frob_norm) if frob_norm is not None
            else float(tensor.norm().item())
        )

    def put_spectral(self, layer_idx: int, role: str,
                      summary: DeltaSpectralSummary) -> None:
        self._spectral[(layer_idx, role)] = summary

    # ── Reading ─────────────────────────────────────────────────────
    def get(self, layer_idx: int, role: str) -> torch.Tensor:
        """Retrieve a delta tensor. Raises LayerNotComputedError if the
        requested (layer, role) was excluded by this store's filter."""
        key = (layer_idx, role)
        if key not in self._data:
            if role not in self._adapter.PROJECTION_ROLES:
                raise KeyError(
                    f"Role '{role}' not declared by adapter "
                    f"'{self._adapter.family_id}'. Available: "
                    f"{list(self._adapter.PROJECTION_ROLES)}")
            raise LayerNotComputedError(
                layer=layer_idx, role=role,
                filter_=self._metadata.layer_filter,
                n_layers=self._metadata.n_layers,
            )
        return self._data[key]

    def get_or_none(self, layer_idx: int, role: str) -> Optional[torch.Tensor]:
        """Non-raising variant. Returns None for any missing delta."""
        return self._data.get((layer_idx, role))

    def has(self, layer_idx: int, role: str) -> bool:
        return (layer_idx, role) in self._data

    def frob_norm(self, layer_idx: int, role: str) -> float:
        """Frobenius norm of delta at (layer, role). Raises like get()."""
        key = (layer_idx, role)
        if key not in self._frob_norms:
            # Force an error via get() for consistent messaging
            self.get(layer_idx, role)
        return self._frob_norms[key]

    def spectral(self, layer_idx: int, role: str
                  ) -> Optional[DeltaSpectralSummary]:
        return self._spectral.get((layer_idx, role))

    # ── Role/layer iteration ────────────────────────────────────────
    def layers(self) -> list[int]:
        """Sorted unique layer indices present in the store."""
        return sorted({layer for layer, _ in self._data})

    def roles_at(self, layer_idx: int) -> list[str]:
        """Roles present at a given layer."""
        return sorted(role for layer, role in self._data if layer == layer_idx)

    def items(self):
        """Iterate over ((layer, role), tensor) pairs. For bulk readers."""
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: tuple[int, str]) -> bool:
        return key in self._data

    # ── Adapter convenience (named accessors) ───────────────────────
    def v_delta(self, layer_idx: int) -> torch.Tensor:
        """Shortcut for store.get(layer_idx, 'v')."""
        return self.get(layer_idx, "v")

    def o_delta(self, layer_idx: int) -> torch.Tensor:
        """Shortcut for store.get(layer_idx, 'o')."""
        return self.get(layer_idx, "o")

    def v_delta_or_none(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.get_or_none(layer_idx, "v")

    def o_delta_or_none(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.get_or_none(layer_idx, "o")

    # ── Metadata ────────────────────────────────────────────────────
    @property
    def metadata(self) -> DeltaStoreMetadata:
        return self._metadata

    @property
    def adapter(self) -> "ModelAdapter":
        return self._adapter

    @property
    def layer_filter(self) -> Optional[list[int]]:
        return self._metadata.layer_filter

    @property
    def full_deltas_available(self) -> bool:
        """True if this store was built over all layers (no filter)."""
        return self._metadata.layer_filter is None

    # ── Aggregate summaries ─────────────────────────────────────────
    def aggregate_spectral_summary(self) -> dict:
        """Average spectral properties across all held deltas.

        Used by the frontend's model-info panel and by exports for
        reproducibility. Returns zeros-dict if no spectral data is
        present (e.g. spectral computation was skipped).
        """
        if not self._spectral:
            return {
                "mean_eff_rank": 0.0,
                "std_eff_rank": 0.0,
                "mean_top1_share": 0.0,
                "attn_mean_rank": 0.0,
                "mlp_mean_rank": 0.0,
                "n_sublayers": 0,
            }

        import statistics

        eff_ranks = [s.eff_rank for s in self._spectral.values()]
        top1 = [s.top1_share for s in self._spectral.values()]
        attn_roles = {"q", "k", "v", "o"}
        attn_ranks = [s.eff_rank for (l, r), s in self._spectral.items()
                      if r in attn_roles]
        mlp_ranks = [s.eff_rank for (l, r), s in self._spectral.items()
                     if r not in attn_roles]

        def _mean(xs): return float(statistics.mean(xs)) if xs else 0.0
        def _std(xs): return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0

        return {
            "mean_eff_rank": _mean(eff_ranks),
            "std_eff_rank": _std(eff_ranks),
            "mean_top1_share": _mean(top1),
            "attn_mean_rank": _mean(attn_ranks),
            "mlp_mean_rank": _mean(mlp_ranks),
            "n_sublayers": len(eff_ranks),
        }

    def total_bytes(self) -> int:
        """Approximate bytes held in delta tensors. For memory diagnostics."""
        return sum(t.element_size() * t.numel() for t in self._data.values())


# ═══════════════════════════════════════════════════════════════════
# MmapDeltaStore — disk-backed variant for large models (HEP mode)
# ═══════════════════════════════════════════════════════════════════

import mmap
import os
import struct
import logging
from pathlib import Path

logger = logging.getLogger("src")

# File format constants
_MAGIC = b"TAGM_DELTA_MMAP_V1"
_HEADER_SIZE = 256
_INDEX_ENTRY_SIZE = 48
_DTYPE_MAP = {0: torch.float32, 1: torch.bfloat16, 2: torch.float16}
_DTYPE_REV = {v: k for k, v in _DTYPE_MAP.items()}


class MmapDeltaStore:
    """DeltaStore backed by a memory-mapped flat file.

    Same public interface as DeltaStore. Consumers don't know which
    backend they're using — store.get(layer, role) returns a torch.Tensor
    in both cases.

    The file layout is:
        [HEADER: 256 bytes]  magic, n_entries, dtype code
        [INDEX: n × 48 bytes]  per-entry: layer, role, offset, shape, frob_norm
        [DATA: concatenated raw tensor bytes]

    The index is loaded fully into RAM (~8 KB for 168 deltas). The data
    region is mmap'd. On get(), a zero-copy torch tensor is constructed
    from the mmap. The OS pages in only the requested region.

    Write path: put() appends tensor bytes to the file and records the
    index entry. The header and index are finalized on flush()/close().

    Read path: the constructor detects an existing file (has valid magic)
    and opens it in read-only mmap mode. New files are opened in write mode.
    """

    def __init__(self, adapter: "ModelAdapter", metadata: DeltaStoreMetadata,
                 mmap_path: Path, dtype: torch.dtype = torch.bfloat16):
        self._adapter = adapter
        self._metadata = metadata
        self._path = Path(mmap_path)
        self._dtype = dtype
        self._dtype_code = _DTYPE_REV.get(dtype, 1)  # default bfloat16
        self._element_size = torch.tensor([], dtype=dtype).element_size()

        # In-memory index: maps (layer_idx, role) -> dict with offset, rows, cols, frob_norm
        self._index: dict[tuple[int, str], dict] = {}
        self._frob_norms: dict[tuple[int, str], float] = {}
        self._spectral: dict[tuple[int, str], DeltaSpectralSummary] = {}

        # File handles
        self._file = None
        self._mm = None
        self._data_start = 0  # byte offset where tensor data begins
        self._write_pos = 0   # current write position in data region
        self._mode = None     # "write" or "read"

        if self._path.exists() and self._path.stat().st_size >= _HEADER_SIZE:
            self._open_read()
        else:
            self._open_write()

    # ── File I/O ───────────────────────────────────────────────────

    def _open_write(self):
        """Create a new file for writing deltas."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w+b")
        self._mode = "write"
        # Write placeholder header (updated on flush)
        self._file.write(b"\x00" * _HEADER_SIZE)
        # Index region starts after header; grows as entries are added.
        # Data region starts after the index. Since we don't know n_entries
        # in advance, we buffer index entries in RAM and write them at flush.
        self._index_entries_raw = []  # list of 48-byte packed entries
        self._data_start = 0  # set at flush time
        self._write_pos = 0
        # Write tensor data to a temp region starting right after the header;
        # at flush time, we'll rewrite the file with header + index + data.
        self._data_buf_start = _HEADER_SIZE

    def _open_read(self):
        """Open an existing mmap file for reading."""
        self._file = open(self._path, "rb")
        self._mode = "read"

        # Read and validate header
        header = self._file.read(_HEADER_SIZE)
        magic = header[:len(_MAGIC)]
        if magic != _MAGIC:
            raise ValueError(f"Not a TAGM delta mmap file: {self._path}")

        n_entries = struct.unpack_from("<I", header, 32)[0]
        self._dtype_code = header[36]
        self._dtype = _DTYPE_MAP.get(self._dtype_code, torch.bfloat16)
        self._element_size = torch.tensor([], dtype=self._dtype).element_size()

        # Read index
        index_size = n_entries * _INDEX_ENTRY_SIZE
        index_raw = self._file.read(index_size)

        self._data_start = _HEADER_SIZE + index_size

        for i in range(n_entries):
            off = i * _INDEX_ENTRY_SIZE
            entry = index_raw[off:off + _INDEX_ENTRY_SIZE]
            layer_idx = struct.unpack_from("<H", entry, 0)[0]
            role = entry[2:10].rstrip(b"\x00").decode("utf-8")
            data_offset = struct.unpack_from("<Q", entry, 10)[0]
            rows = struct.unpack_from("<I", entry, 18)[0]
            cols = struct.unpack_from("<I", entry, 22)[0]
            frob_norm = struct.unpack_from("<d", entry, 26)[0]

            key = (layer_idx, role)
            self._index[key] = {
                "offset": data_offset,
                "rows": rows,
                "cols": cols,
            }
            self._frob_norms[key] = frob_norm

        # Mmap the entire file for data access
        self._file.seek(0)
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        self._metadata.n_deltas = n_entries
        logger.info(f"[MMAP] Opened {self._path.name}: {n_entries} deltas, "
                     f"dtype={self._dtype}")

    def flush(self):
        """Finalize the file: write header + index, ensure data is synced."""
        if self._mode != "write" or self._file is None:
            return

        n_entries = len(self._index)
        index_size = n_entries * _INDEX_ENTRY_SIZE

        # We wrote data bytes starting at _data_buf_start (right after the
        # header).  The index has to be inserted between header and data, so
        # the data region must shift right by index_size.
        #
        # This used to be `data_bytes = self._file.read()` — the ENTIRE delta
        # payload into one Python bytes object, then written back.  For a 7B
        # pair that is an ~8 GB allocation plus an 8 GB rewrite at close,
        # which is precisely the peak-memory spike the mmap backend exists to
        # avoid.  Shift in fixed-size chunks instead, working backwards so a
        # forward-overlapping move never overwrites bytes it has yet to read.
        self._file.flush()
        self._file.seek(0, os.SEEK_END)
        data_end = self._file.tell()
        data_len = data_end - self._data_buf_start
        new_data_start = _HEADER_SIZE + index_size
        shift = new_data_start - self._data_buf_start

        if shift > 0 and data_len > 0:
            self._file.truncate(data_end + shift)
            chunk = 32 * 1024 * 1024
            remaining = data_len
            while remaining > 0:
                n = min(chunk, remaining)
                src = self._data_buf_start + remaining - n
                self._file.seek(src)
                buf = self._file.read(n)
                self._file.seek(src + shift)
                self._file.write(buf)
                remaining -= n
                del buf
        elif shift < 0:
            # Index is smaller than the gap reserved for it: move data left,
            # front to back, then truncate the leftover tail.
            chunk = 32 * 1024 * 1024
            moved = 0
            while moved < data_len:
                n = min(chunk, data_len - moved)
                self._file.seek(self._data_buf_start + moved)
                buf = self._file.read(n)
                self._file.seek(new_data_start + moved)
                self._file.write(buf)
                moved += n
                del buf
            self._file.truncate(new_data_start + data_len)

        # Header
        self._file.seek(0)
        hdr = bytearray(_HEADER_SIZE)
        hdr[:len(_MAGIC)] = _MAGIC
        struct.pack_into("<I", hdr, 32, n_entries)
        hdr[36] = self._dtype_code
        self._file.write(bytes(hdr))

        # Index (in insertion order, which matches data order)
        for entry_bytes in self._index_entries_raw:
            self._file.write(entry_bytes)

        self._data_start = new_data_start
        self._file.flush()

        self._metadata.n_deltas = n_entries

    def close(self):
        """Flush, unmap, and close the file."""
        if self._mode == "write":
            self.flush()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def reopen_readonly(self):
        """After writing, close and reopen in read mode for mmap access."""
        self.close()
        self._index.clear()
        self._frob_norms.clear()
        self._open_read()

    # ── Writing (used by compute.py) ────────────────────────────────

    def put(self, layer_idx: int, role: str, tensor: torch.Tensor,
            frob_norm: Optional[float] = None) -> None:
        """Append a delta tensor to the file and record its index entry."""
        if self._mode != "write":
            raise RuntimeError("MmapDeltaStore is in read mode; cannot put()")

        key = (layer_idx, role)
        fn = float(frob_norm) if frob_norm is not None else float(tensor.norm().item())

        # Ensure target dtype, contiguous, on CPU
        t = tensor.to(self._dtype).cpu().contiguous()
        rows, cols = t.shape[0], t.shape[1] if t.dim() > 1 else 1

        # Extract raw bytes (works for all dtypes including bfloat16)
        n_bytes = t.numel() * t.element_size()
        raw_bytes = bytes(t.untyped_storage()[:n_bytes])

        # Data offset relative to start of data region
        data_offset = self._write_pos

        # Write tensor bytes
        self._file.seek(self._data_buf_start + self._write_pos)
        self._file.write(raw_bytes)
        self._write_pos += len(raw_bytes)

        # Build index entry (48 bytes)
        entry = bytearray(_INDEX_ENTRY_SIZE)
        struct.pack_into("<H", entry, 0, layer_idx)
        role_bytes = role.encode("utf-8")[:8]
        entry[2:2 + len(role_bytes)] = role_bytes
        struct.pack_into("<Q", entry, 10, data_offset)
        struct.pack_into("<I", entry, 18, rows)
        struct.pack_into("<I", entry, 22, cols)
        struct.pack_into("<d", entry, 26, fn)
        self._index_entries_raw.append(bytes(entry))

        # Update in-memory index
        self._index[key] = {"offset": data_offset, "rows": rows, "cols": cols}
        self._frob_norms[key] = fn

    def put_spectral(self, layer_idx: int, role: str,
                      summary: DeltaSpectralSummary) -> None:
        """Store spectral summary in memory (small, not worth mmap'ing)."""
        self._spectral[(layer_idx, role)] = summary

    # ── Reading ─────────────────────────────────────────────────────

    def get(self, layer_idx: int, role: str) -> torch.Tensor:
        """Retrieve a delta tensor from the mmap.

        Returns a torch.Tensor. For bfloat16, the tensor is constructed
        from raw bytes via torch.frombuffer(). The returned tensor shares
        memory with the mmap — .float() creates a copy, which is what
        every consumer does.
        """
        key = (layer_idx, role)
        if key not in self._index:
            if role not in self._adapter.PROJECTION_ROLES:
                raise KeyError(
                    f"Role '{role}' not declared by adapter "
                    f"'{self._adapter.family_id}'. Available: "
                    f"{list(self._adapter.PROJECTION_ROLES)}")
            raise LayerNotComputedError(
                layer=layer_idx, role=role,
                filter_=self._metadata.layer_filter,
                n_layers=self._metadata.n_layers,
            )

        if self._mm is None:
            raise RuntimeError(
                "MmapDeltaStore: file not memory-mapped. Call "
                "reopen_readonly() after writing, or open an existing file.")

        info = self._index[key]
        rows, cols = info["rows"], info["cols"]
        n_elements = rows * cols
        n_bytes = n_elements * self._element_size
        file_offset = self._data_start + info["offset"]

        # Read raw bytes from mmap and construct tensor
        raw = self._mm[file_offset:file_offset + n_bytes]
        t = torch.frombuffer(bytearray(raw), dtype=self._dtype).reshape(rows, cols)
        return t

    def get_or_none(self, layer_idx: int, role: str) -> Optional[torch.Tensor]:
        """Non-raising variant. Returns None for any missing delta."""
        if (layer_idx, role) not in self._index:
            return None
        return self.get(layer_idx, role)

    def has(self, layer_idx: int, role: str) -> bool:
        return (layer_idx, role) in self._index

    def frob_norm(self, layer_idx: int, role: str) -> float:
        """Frobenius norm from the in-memory index. Zero I/O."""
        key = (layer_idx, role)
        if key not in self._frob_norms:
            self.get(layer_idx, role)  # raises appropriate error
        return self._frob_norms[key]

    def spectral(self, layer_idx: int, role: str
                  ) -> Optional[DeltaSpectralSummary]:
        return self._spectral.get((layer_idx, role))

    # ── Role/layer iteration ────────────────────────────────────────

    def layers(self) -> list[int]:
        """Sorted unique layer indices present in the store."""
        return sorted({layer for layer, _ in self._index})

    def roles_at(self, layer_idx: int) -> list[str]:
        """Roles present at a given layer."""
        return sorted(role for layer, role in self._index if layer == layer_idx)

    def items(self):
        """Iterate over ((layer, role), tensor) pairs.

        Each tensor is constructed from the mmap on demand. Consumers
        should process and release each tensor before requesting the
        next — accumulating references defeats the memory savings.
        """
        for key in self._index:
            yield key, self.get(*key)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: tuple[int, str]) -> bool:
        return key in self._index

    # ── Adapter convenience ─────────────────────────────────────────

    def v_delta(self, layer_idx: int) -> torch.Tensor:
        return self.get(layer_idx, "v")

    def o_delta(self, layer_idx: int) -> torch.Tensor:
        return self.get(layer_idx, "o")

    def v_delta_or_none(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.get_or_none(layer_idx, "v")

    def o_delta_or_none(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.get_or_none(layer_idx, "o")

    # ── Metadata ────────────────────────────────────────────────────

    @property
    def metadata(self) -> DeltaStoreMetadata:
        return self._metadata

    @property
    def adapter(self) -> "ModelAdapter":
        return self._adapter

    @property
    def layer_filter(self) -> Optional[list[int]]:
        return self._metadata.layer_filter

    @property
    def full_deltas_available(self) -> bool:
        return self._metadata.layer_filter is None

    # ── Aggregate summaries ─────────────────────────────────────────

    def aggregate_spectral_summary(self) -> dict:
        """Identical to DeltaStore.aggregate_spectral_summary()."""
        if not self._spectral:
            return {
                "mean_eff_rank": 0.0,
                "std_eff_rank": 0.0,
                "mean_top1_share": 0.0,
                "attn_mean_rank": 0.0,
                "mlp_mean_rank": 0.0,
                "n_sublayers": 0,
            }

        import statistics

        eff_ranks = [s.eff_rank for s in self._spectral.values()]
        top1 = [s.top1_share for s in self._spectral.values()]
        attn_roles = {"q", "k", "v", "o"}
        attn_ranks = [s.eff_rank for (l, r), s in self._spectral.items()
                      if r in attn_roles]
        mlp_ranks = [s.eff_rank for (l, r), s in self._spectral.items()
                     if r not in attn_roles]

        def _mean(xs): return float(statistics.mean(xs)) if xs else 0.0
        def _std(xs): return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0

        return {
            "mean_eff_rank": _mean(eff_ranks),
            "std_eff_rank": _std(eff_ranks),
            "mean_top1_share": _mean(top1),
            "attn_mean_rank": _mean(attn_ranks),
            "mlp_mean_rank": _mean(mlp_ranks),
            "n_sublayers": len(eff_ranks),
        }

    def total_bytes(self) -> int:
        """Total bytes of tensor data in the mmap file."""
        return sum(
            info["rows"] * info["cols"] * self._element_size
            for info in self._index.values()
        )
