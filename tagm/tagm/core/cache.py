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
        safe = _safe_filename(name)
        path = self.layout.presets / f"{safe}.json"
        with open(path, "w") as f:
            f.write(config_json)
        return path

    def load_preset(self, name: str) -> str:
        """Load a saved preset as its JSON string. Raises FileNotFoundError
        if absent; caller parses with CaptureConfig.from_json."""
        safe = _safe_filename(name)
        path = self.layout.presets / f"{safe}.json"
        with open(path) as f:
            return f.read()

    def delete_preset(self, name: str) -> bool:
        """Delete a preset. Returns True if a file was removed."""
        safe = _safe_filename(name)
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


def _safe_filename(name: str) -> str:
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
