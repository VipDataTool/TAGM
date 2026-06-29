"""Persistent on-disk cache for TAGM artifacts.

Holds capture-config presets, probe embeddings (content-addressed), SVD
precomputes, and optionally delta tensors. Stdlib only: JSON for small
artifacts, numpy .npz for large ones, plain files for presets.

Layout:

    ~/.tagm/cache/                 # default, configurable
    ├── presets/                   # saved CaptureConfig presets
    │   └── *.json
    ├── probes/                    # probe embeddings
    │   └── <probe_hash>.npz
    ├── svd/                       # SVD precomputes for SFD-style measurements
    │   └── <pair_hash>_<role>_k<k>.pt
    ├── sessions/                  # exported session records
    │   └── *.json.gz
    └── meta.json                  # cache metadata

The Cache is a thin directory-managing utility. Content-addressing,
serialization, and validation are the caller's responsibility (the probe
subsystem owns its own store built on top of this). The Cache itself
just knows where things live on disk and how to enumerate them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CacheLayout:
    """Paths for the on-disk cache. Created lazily on first access."""
    root: Path

    @property
    def presets(self) -> Path: return self.root / "presets"
    @property
    def probes(self) -> Path: return self.root / "probes"
    @property
    def svd(self) -> Path: return self.root / "svd"
    @property
    def sessions(self) -> Path: return self.root / "sessions"
    @property
    def meta_file(self) -> Path: return self.root / "meta.json"


class Cache:
    """On-disk cache manager. All TAGM artifacts live under one root."""

    def __init__(self, root: Optional[Path] = None):
        if root is None:
            env_root = os.environ.get("TAGM_CACHE_DIR")
            root = Path(env_root) if env_root else Path.home() / ".tagm" / "cache"
        self.layout = CacheLayout(Path(root))
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        for p in (self.layout.root, self.layout.presets, self.layout.probes,
                  self.layout.svd, self.layout.sessions):
            p.mkdir(parents=True, exist_ok=True)
        if not self.layout.meta_file.exists():
            self._write_meta({
                "schema_version": 1,
                "created_by": "tagm",
            })

    def _read_meta(self) -> dict:
        try:
            with open(self.layout.meta_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_meta(self, meta: dict) -> None:
        with open(self.layout.meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    # ── Preset management ───────────────────────────────────────────
    def list_presets(self) -> list[str]:
        """Names of all saved capture config presets (without .json extension)."""
        return sorted(
            p.stem for p in self.layout.presets.glob("*.json")
        )

    def save_preset(self, name: str, config_json: str) -> Path:
        """Save a CaptureConfig as a named preset. Returns the path written.

        `config_json` is the output of CaptureConfig.to_json() — not
        reparsed, so the caller retains authority over the serialization.
        """
        safe = safe_filename(name)
        path = self.layout.presets / f"{safe}.json"
        with open(path, "w") as f:
            f.write(config_json)
        return path

    def load_preset(self, name: str) -> str:
        """Load a saved preset as its JSON string. Raises FileNotFoundError
        if absent; caller parses with CaptureConfig.from_json."""
        safe = safe_filename(name)
        path = self.layout.presets / f"{safe}.json"
        with open(path) as f:
            return f.read()

    def delete_preset(self, name: str) -> bool:
        """Delete a preset. Returns True if a file was removed."""
        safe = safe_filename(name)
        path = self.layout.presets / f"{safe}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # ── Probe artifact paths ────────────────────────────────────────
    def probe_path(self, probe_hash: str) -> Path:
        """Path for a probe artifact with a given content hash."""
        return self.layout.probes / f"{probe_hash}.npz"

    def list_probes(self) -> list[str]:
        """Content hashes of all stored probe artifacts."""
        return sorted(p.stem for p in self.layout.probes.glob("*.npz"))

    # ── SVD cache paths ─────────────────────────────────────────────
    def svd_path(self, pair_hash: str, role: str, k: int) -> Path:
        """Path for an SVD precompute keyed by model pair + role + rank."""
        return self.layout.svd / f"{pair_hash}_{role}_k{k}.pt"

    # ── Session paths ───────────────────────────────────────────────
    def session_path(self, session_id: str) -> Path:
        """Path for an exported session record."""
        safe = _safe_filename(session_id)
        return self.layout.sessions / f"{safe}.json.gz"

    def list_sessions(self) -> list[str]:
        return sorted(
            p.name.removesuffix(".json.gz")
            for p in self.layout.sessions.glob("*.json.gz")
        )

    # ── Diagnostics ─────────────────────────────────────────────────
    def disk_usage(self) -> dict:
        """Byte totals per subdirectory. For the cache-management UI."""
        out = {}
        for name, path in (
            ("presets", self.layout.presets),
            ("probes", self.layout.probes),
            ("svd", self.layout.svd),
            ("sessions", self.layout.sessions),
        ):
            total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            out[name] = total
        out["total"] = sum(out.values())
        return out


def safe_filename(name: str) -> str:
    """Sanitize an arbitrary string into a safe filename.

    Replaces path separators and non-printable characters with underscores.
    Trims length to 200 characters. Empty input becomes 'unnamed'.
    """
    if not name:
        return "unnamed"
    safe = "".join(
        c if c.isalnum() or c in "._- " else "_"
        for c in name.strip()
    )
    safe = safe.strip().replace(" ", "_")
    return safe[:200] or "unnamed"


# ═══════════════════════════════════════════════════════════════════
# HEP utilities — cache cleanup and system resource reporting
# ═══════════════════════════════════════════════════════════════════

def get_hf_cache_dir() -> Path:
    """Locate the HuggingFace hub cache directory."""
    import os
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_size() -> int:
    """Total bytes used by the HuggingFace model cache."""
    cache_dir = get_hf_cache_dir()
    if not cache_dir.exists():
        return 0
    return sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())


def clear_hf_cache() -> dict:
    """Remove all cached model files from the HuggingFace hub cache.

    Removes models--* directories (model weight files). Preserves
    non-model files (tokens, settings, version).

    Returns dict with bytes_freed, n_removed, errors.
    """
    import shutil
    cache_dir = get_hf_cache_dir()
    result = {"bytes_freed": 0, "n_removed": 0, "errors": []}
    if not cache_dir.exists():
        return result

    for entry in cache_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            try:
                size = sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())
                shutil.rmtree(entry)
                result["bytes_freed"] += size
                result["n_removed"] += 1
            except Exception as e:
                result["errors"].append(f"{entry.name}: {e}")

    return result


def evict_hf_model(model_id: str) -> dict:
    """Remove a specific model from the HuggingFace hub cache.

    HF caches models in directories named models--{org}--{name},
    e.g. Qwen/Qwen2.5-3B → models--Qwen--Qwen2.5-3B.

    Returns dict with bytes_freed, removed (bool), error.
    """
    import shutil
    cache_dir = get_hf_cache_dir()
    result = {"bytes_freed": 0, "removed": False, "model_id": model_id, "error": None}
    if not cache_dir.exists():
        return result

    # HF cache directory name: org/model → models--org--model
    dir_name = "models--" + model_id.replace("/", "--")
    target = cache_dir / dir_name

    if target.exists() and target.is_dir():
        try:
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target)
            result["bytes_freed"] = size
            result["removed"] = True
        except Exception as e:
            result["error"] = str(e)

    return result


def clear_mmap_deltas() -> dict:
    """Remove all mmap delta files."""
    delta_dir = Path.home() / ".tagm" / "cache" / "deltas"
    result = {"bytes_freed": 0, "n_removed": 0}
    if not delta_dir.exists():
        return result
    for f in delta_dir.glob("*.tagm"):
        try:
            result["bytes_freed"] += f.stat().st_size
            f.unlink()
            result["n_removed"] += 1
        except Exception:
            pass
    return result


def system_resources() -> dict:
    """Return current disk and memory usage. No external dependencies."""
    import shutil

    # Disk
    disk = shutil.disk_usage("/")

    # RAM (Linux /proc/meminfo)
    ram_total = 0
    ram_available = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    ram_available = int(line.split()[1]) * 1024
    except Exception:
        pass

    return {
        "disk_total": disk.total,
        "disk_used": disk.used,
        "disk_free": disk.free,
        "ram_total": ram_total,
        "ram_available": ram_available,
        "hf_cache_bytes": hf_cache_size(),
    }


# Backwards-compatible private alias (pre-existing internal callers).
_safe_filename = safe_filename
