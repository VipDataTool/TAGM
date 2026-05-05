"""Session: stores flat per-prompt result dicts.

TASM's session is a list of flat dicts — each dict is the output of
result_to_dict(). This module provides the same contract with disk
persistence and basic query methods.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tagm")


class Session:
    """Accumulates flat prompt-result dicts for one experimental session.

    Results are stored as-is from result_to_dict() — same flat shape
    the frontend reads. No nesting, no translation.
    """

    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / "current"
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{self.timestamp}"
        self.model_name: str = ""
        self.results: list[dict] = []

        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def n_results(self) -> int:
        return len(self.results)

    @property
    def categories(self) -> dict:
        cats: dict[str, int] = {}
        for r in self.results:
            c = r.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        return cats

    def set_model(self, name: str):
        self.model_name = name
        meta = {
            "model": name,
            "started": self.timestamp,
            "session_id": self.session_id,
        }
        meta_path = self.session_dir / "session.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except OSError:
            pass

    def add_result(self, result_dict: dict) -> int:
        """Add a flat result dict to the session. Returns the index."""
        idx = len(self.results)
        result_dict["_index"] = idx
        self.results.append(result_dict)
        return idx

    def remove_indices(self, indices: list[int]):
        """Remove results at the given indices and reindex."""
        indices_set = set(indices)
        self.results = [r for i, r in enumerate(self.results)
                        if i not in indices_set]
        for i, r in enumerate(self.results):
            r["_index"] = i

    def clear(self):
        """Clear all results."""
        self.results.clear()

    def get_cache_size(self) -> int:
        """Approximate bytes of session data on disk."""
        total = 0
        if self.session_dir.exists():
            for f in self.session_dir.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    # ── Disk persistence ────────────────────────────────────────────

    def save_to_disk(self):
        """Persist current results to disk."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / "results.json"
        try:
            with open(path, "w") as f:
                json.dump(self.results, f)
        except OSError as e:
            logger.warning(f"[SESSION] Failed to save: {e}")

    @classmethod
    def restore(cls, base_dir: str = "datasets") -> Optional["Session"]:
        """Restore session from disk. Returns None if nothing to restore."""
        base = Path(base_dir)
        results_path = base / "current" / "results.json"
        if not results_path.exists():
            return None
        try:
            with open(results_path) as f:
                results = json.load(f)
            if not isinstance(results, list) or not results:
                return None
        except (json.JSONDecodeError, OSError):
            return None

        obj = object.__new__(cls)
        obj.base_dir = base
        obj.session_dir = base / "current"
        obj.results = results
        obj.model_name = ""
        obj.session_id = ""
        obj.timestamp = ""

        meta_path = obj.session_dir / "session.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                obj.model_name = meta.get("model", "")
                obj.session_id = meta.get("session_id", "")
                obj.timestamp = meta.get("started", "")
            except (json.JSONDecodeError, OSError):
                pass

        logger.info(f"[SESSION] Restored {len(results)} results from disk")
        return obj

    @staticmethod
    def has_session_on_disk(base_dir: str = "datasets") -> Optional[dict]:
        """Check if restorable data exists. Returns info dict or None."""
        results_path = Path(base_dir) / "current" / "results.json"
        if not results_path.exists():
            return None
        try:
            size = results_path.stat().st_size
            if size < 3:
                return None
        except OSError:
            return None

        info = {"path": str(results_path.parent), "has_results": True,
                "results_size_bytes": size}

        meta_path = results_path.parent / "session.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                info["model"] = meta.get("model", "")
                info["n_results"] = 0
                # Quick count
                with open(results_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        info["n_results"] = len(data)
            except (json.JSONDecodeError, OSError):
                pass

        return info
