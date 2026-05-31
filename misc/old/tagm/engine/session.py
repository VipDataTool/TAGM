"""Session: DB-backed per-prompt result store.

Drop-in replacement for the original JSON-file Session. All public
attributes and methods keep the same signatures. The key difference:
``add_result()`` is now a single INSERT instead of rewriting a
monolithic JSON file, and ``results`` is a lazy list-like accessor
that loads rows on demand.

Migration: on first import the module checks for legacy JSON files
(``datasets/current/results.json``) and migrates them into SQLite
automatically. See ``tagm.core.db.migrate_json_to_db``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from misc.old.tagm.core.db import Database, ResultsList, get_db, migrate_json_to_db

logger = logging.getLogger("tagm")


class Session:
    """Accumulates per-prompt result dicts for one experimental session.

    Results are stored directly in SQLite via the ``ResultsList`` proxy.
    The ``results`` attribute behaves like a regular list — indexing,
    iteration, ``len()`` all work — but each access loads from the
    database on demand.
    """

    def __init__(self, base_dir: str = "datasets", db: Optional[Database] = None):
        self._db = db or get_db()
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / "current"
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{self.timestamp}"
        self.model_name: str = ""

        # Create session directory (still used by modules for output)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Register this session in the DB
        self._db.create_session(self.session_id, "", self.timestamp)

        # results is a list-like proxy backed by the DB
        self.results: ResultsList = ResultsList(self._db, self.session_id)

    @property
    def n_results(self) -> int:
        return len(self.results)

    @property
    def categories(self) -> dict:
        return self._db.session_categories(self.session_id)

    def set_model(self, name: str):
        self.model_name = name
        self._db.update_session_model(self.session_id, name)

    def add_result(self, result_dict: dict) -> int:
        """Add a result dict. Returns the index."""
        idx = self.results.append(result_dict)
        return idx

    def remove_indices(self, indices: list[int]):
        """Remove results at given indices and reindex."""
        self._db.remove_results(self.session_id, indices)
        self.results.invalidate()

    def clear(self):
        """Clear all results for this session."""
        self.results.clear()

    def get_cache_size(self) -> int:
        """Approximate bytes of session data."""
        return self._db.session_size_bytes(self.session_id)

    # ── Disk persistence (now mostly no-ops) ────────────────────

    def save_to_disk(self):
        """No-op — results are already persisted on every add_result.

        Kept for API compatibility; callers that do
        ``session.save_to_disk()`` after batch operations won't break.
        """
        pass

    @classmethod
    def restore(cls, base_dir: str = "datasets",
                db: Optional[Database] = None) -> Optional["Session"]:
        """Restore the most recent session from the database.

        Falls back to migrating legacy JSON files if the DB has no
        sessions yet.
        """
        _db = db or get_db()

        # Try legacy JSON migration first if DB is empty
        sessions = _db.list_sessions()
        if not sessions:
            project_root = Path(base_dir).resolve().parent
            summary = migrate_json_to_db(_db, project_root)
            if summary["results"] > 0:
                sessions = _db.list_sessions()

        if not sessions:
            return None

        # Pick the most recent session with results
        target = None
        for s in sessions:
            if s["n_results"] > 0:
                target = s
                break
        if target is None:
            return None

        # Build a Session object wired to the existing session_id
        obj = object.__new__(cls)
        obj._db = _db
        obj.base_dir = Path(base_dir)
        obj.session_dir = obj.base_dir / "current"
        obj.session_id = target["session_id"]
        obj.model_name = target.get("model", "")
        obj.timestamp = target.get("started", "")
        obj.results = ResultsList(_db, obj.session_id)

        logger.info(
            f"[SESSION] Restored {target['n_results']} results "
            f"from session {obj.session_id}")
        return obj

    @staticmethod
    def has_session_on_disk(base_dir: str = "datasets") -> Optional[dict]:
        """Check if restorable data exists. Returns info dict or None."""
        try:
            db = get_db()
        except Exception:
            return None

        sessions = db.list_sessions()
        if not sessions:
            # Check for legacy JSON files that could be migrated
            results_path = Path(base_dir) / "current" / "results.json"
            if results_path.exists():
                try:
                    size = results_path.stat().st_size
                    if size >= 3:
                        return {
                            "path": str(results_path.parent),
                            "has_results": True,
                            "results_size_bytes": size,
                            "legacy_json": True,
                        }
                except OSError:
                    pass
            return None

        # Find most recent session with results
        for s in sessions:
            if s["n_results"] > 0:
                return {
                    "session_id": s["session_id"],
                    "has_results": True,
                    "n_results": s["n_results"],
                    "model": s.get("model", ""),
                }

        return None

    # ── Dashboard query (direct SQL, no decompression) ──────────

    def get_dashboard_rows(self) -> list[dict]:
        """Return slim scalar-only rows for the dashboard view.

        Uses indexed columns — never touches the compressed blobs.
        """
        return self._db.get_dashboard_rows(self.session_id)

    def get_results_page(self, offset: int = 0,
                         limit: int = 50) -> list[dict]:
        """Load a page of full result dicts."""
        return self._db.get_results_page(self.session_id, offset, limit)
