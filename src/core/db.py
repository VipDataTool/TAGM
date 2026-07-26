"""SQLite persistence layer for TAGM.

Replaces the scattered JSON-file storage (models.json, results.json,
session.json, ui_config.json) with a single SQLite database that lives
at ``~/.tagm/tagm.db`` (overridable via ``TAGM_DB_PATH``).

Design principles
─────────────────
* **Zero infrastructure** — sqlite3 is stdlib; no server, no driver.
* **WAL mode** — concurrent readers never block writers.
* **Compressed results** — each result dict is zlib-compressed JSON in a
  BLOB column (``data_blob``).  Key scalar metrics are *also* stored in
  indexed REAL/INTEGER columns so the dashboard can query without
  decompressing.
* **List-compatible accessor** — ``ResultsList`` quacks like a Python
  list so call-sites that do ``session.results[i]`` or
  ``for r in session.results`` keep working.
* **Auto-migration** — on first run, existing ``models.json``,
  ``datasets/current/results.json``, and ``ui_config.json`` are imported
  and the originals renamed to ``*.migrated``.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import os
import sqlite3
import threading
import time
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

logger = logging.getLogger("src")

# ── Default path ────────────────────────────────────────────────

def _default_db_path() -> Path:
    env = os.environ.get("TAGM_DB_PATH")
    if env:
        return Path(env)
    return Path.home() / ".tagm" / "tagm.db"


# ── Schema ──────────────────────────────────────────────────────

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id        TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    base      TEXT NOT NULL,
    instruct  TEXT NOT NULL,
    ram_gb    INTEGER DEFAULT 0,
    notes     TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    model       TEXT DEFAULT '',
    started     TEXT DEFAULT '',
    status      TEXT DEFAULT 'active',
    created_at  REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    idx             INTEGER NOT NULL,
    prompt          TEXT DEFAULT '',
    category        TEXT DEFAULT '',
    seq_len         INTEGER DEFAULT 0,
    stress_score    REAL DEFAULT 0,
    net_correction  REAL DEFAULT 0,
    entropy         REAL DEFAULT 0,
    top2_share      REAL DEFAULT 0,
    middle_share    REAL DEFAULT 0,
    interior_cv     REAL DEFAULT 0,
    kl_divergence   REAL,
    n_negative_tokens INTEGER DEFAULT 0,
    has_negative_tokens INTEGER DEFAULT 0,
    ltp_mean_m      REAL,
    ltp_mean_v      REAL,
    ltp_max_prc     REAL,
    ltp_n_directional INTEGER,
    sfd_density_mean REAL,
    rd_mean_tau     REAL,
    rd_mean_overlap REAL,
    delta_scale     REAL,
    full_capture_enabled INTEGER DEFAULT 0,
    family_index    INTEGER,
    rung_index      INTEGER,
    data_blob       BLOB NOT NULL,
    created_at      REAL DEFAULT (strftime('%s','now')),
    UNIQUE(session_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_results_session
    ON results(session_id, idx);
CREATE INDEX IF NOT EXISTS idx_results_category
    ON results(session_id, category);
CREATE INDEX IF NOT EXISTS idx_results_stress
    ON results(session_id, stress_score);

CREATE TABLE IF NOT EXISTS config (
    namespace TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT,
    PRIMARY KEY(namespace, key)
);

CREATE TABLE IF NOT EXISTS prompts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt   TEXT NOT NULL,
    category TEXT DEFAULT ''
);
"""


# ── Compression helpers ─────────────────────────────────────────

def _compress(result_dict: dict) -> bytes:
    """Serialize a result dict to zlib-compressed JSON bytes."""
    raw = json.dumps(result_dict, default=_json_fallback, separators=(",", ":"))
    return zlib.compress(raw.encode("utf-8"), level=6)


def _decompress(blob: bytes) -> dict:
    """Decompress a data_blob back into a result dict."""
    raw = zlib.decompress(blob).decode("utf-8")
    return json.loads(raw)


def _json_fallback(obj):
    """Handle numpy types during JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return str(obj)


# ── Scalar extraction (result dict → indexed columns) ──────────

def _extract_scalars(d: dict) -> dict:
    """Pull queryable scalar fields out of a result dict.

    These go into dedicated columns so the dashboard can SELECT them
    without decompressing every blob.
    """
    def _safe_float(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return None

    ltp = d.get("ltp") or {}
    sfd = d.get("sfd") or {}
    rd = d.get("rank_displacement") or {}

    return {
        "prompt": d.get("prompt", ""),
        "category": d.get("category", ""),
        "seq_len": int(d.get("seq_len", 0)),
        "stress_score": _safe_float(d.get("stress_score")),
        "net_correction": _safe_float(d.get("net_correction")),
        "entropy": _safe_float(d.get("entropy")),
        "top2_share": _safe_float(d.get("top2_share")),
        "middle_share": _safe_float(d.get("middle_share")),
        "interior_cv": _safe_float(d.get("interior_cv")),
        "kl_divergence": _safe_float(d.get("kl_divergence")),
        "n_negative_tokens": int(d.get("n_negative_tokens", 0)),
        "has_negative_tokens": 1 if d.get("has_negative_tokens") else 0,
        "ltp_mean_m": _safe_float(ltp.get("mean_M")),
        "ltp_mean_v": _safe_float(ltp.get("mean_V")),
        "ltp_max_prc": _safe_float(ltp.get("max_prc")),
        "ltp_n_directional": ltp.get("n_directional"),
        "sfd_density_mean": _safe_float(sfd.get("density_mean")),
        "rd_mean_tau": _safe_float(rd.get("mean_tau")),
        "rd_mean_overlap": _safe_float(rd.get("mean_overlap")),
        "delta_scale": _safe_float(d.get("delta_scale")),
        "full_capture_enabled": 1 if d.get("full_capture_enabled") else 0,
        # Ladder identity — present only on deconstructed prompts, else None.
        "family_index": d.get("family_index"),
        "rung_index": d.get("rung_index"),
    }


# ── ResultsList — list-like lazy accessor ───────────────────────

class ResultsList:
    """Thin wrapper that makes DB-backed results look like a plain list.

    Supports ``len()``, indexing, slicing, iteration, ``append`` (as
    ``add``), and index assignment — everything the existing codebase
    does with ``session.results``.

    Results are loaded from the database on demand and cached in an LRU
    dict to avoid redundant decompression during tight loops (e.g.
    batch-plot renderers that iterate twice).
    """

    def __init__(self, db: "Database", session_id: str):
        self._db = db
        self._session_id = session_id
        self._len: Optional[int] = None  # cached count
        self._cache: dict[int, dict] = {}
        self._cache_max = 256

    def invalidate(self):
        """Clear cached count and row cache."""
        self._len = None
        self._cache.clear()

    # ── Sequence protocol ───────────────────────────────────────

    def _count_rows(self) -> int:
        """Authoritative row count, straight from the database."""
        row = self._db.query_one(
            "SELECT COUNT(*) FROM results WHERE session_id = ?",
            (self._session_id,),
        )
        return row[0] if row else 0

    def __len__(self) -> int:
        if self._len is None:
            self._len = self._count_rows()
        return self._len

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self._load(i) for i in indices]
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"result index {key} out of range")
            return self._load(key)
        raise TypeError(f"indices must be integers or slices, not {type(key).__name__}")

    def __setitem__(self, idx: int, result_dict: dict):
        """Replace the result at position ``idx`` (used by rerun)."""
        if idx < 0:
            idx += len(self)
        scalars = _extract_scalars(result_dict)
        blob = _compress(result_dict)
        self._db.execute(
            """UPDATE results
               SET prompt = ?, category = ?, seq_len = ?,
                   stress_score = ?, net_correction = ?,
                   entropy = ?, top2_share = ?, middle_share = ?,
                   interior_cv = ?, kl_divergence = ?,
                   n_negative_tokens = ?, has_negative_tokens = ?,
                   ltp_mean_m = ?, ltp_mean_v = ?, ltp_max_prc = ?,
                   ltp_n_directional = ?, sfd_density_mean = ?,
                   rd_mean_tau = ?, rd_mean_overlap = ?,
                   delta_scale = ?,
                   full_capture_enabled = ?,
                   family_index = ?, rung_index = ?,
                   data_blob = ?
               WHERE session_id = ? AND idx = ?""",
            (
                scalars["prompt"], scalars["category"], scalars["seq_len"],
                scalars["stress_score"], scalars["net_correction"],
                scalars["entropy"], scalars["top2_share"], scalars["middle_share"],
                scalars["interior_cv"], scalars["kl_divergence"],
                scalars["n_negative_tokens"], scalars["has_negative_tokens"],
                scalars["ltp_mean_m"], scalars["ltp_mean_v"], scalars["ltp_max_prc"],
                scalars["ltp_n_directional"], scalars["sfd_density_mean"],
                scalars["rd_mean_tau"], scalars["rd_mean_overlap"],
                scalars["delta_scale"],
                scalars["full_capture_enabled"],
                scalars["family_index"], scalars["rung_index"],
                blob, self._session_id, idx,
            ),
        )
        self._db.commit()
        self._cache.pop(idx, None)

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all results in index order, streaming from DB.

        Fetches in short-lived pages rather than holding one cursor open
        across yields — a long-lived cursor on the shared connection is
        unsafe against concurrent writers.
        """
        offset = 0
        page = 200
        while True:
            rows = self._db.query_all(
                "SELECT idx, data_blob FROM results "
                "WHERE session_id = ? ORDER BY idx LIMIT ? OFFSET ?",
                (self._session_id, page, offset),
            )
            if not rows:
                return
            offset += len(rows)
            yield from self._iter_rows(rows)

    def _iter_rows(self, rows):
        for row in rows:
            idx, blob = row
            d = _decompress(blob)
            d["_index"] = idx
            # Same bound as _load — iteration previously cached every
            # decompressed result with no eviction, pinning whole sessions
            # in RAM and defeating the paging design.
            if len(self._cache) >= self._cache_max:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[idx] = d
            yield d

    def __contains__(self, item) -> bool:
        # Rarely used; fall back to iteration
        return any(r is item or r == item for r in self)

    # ── Mutations ───────────────────────────────────────────────

    def append(self, result_dict: dict) -> int:
        """Insert a new result and return its index.

        The index read, the INSERT, and the commit all happen inside ONE
        transaction.  Previously the lock was taken for execute(), released,
        then retaken for commit(), so another thread's commit in between could
        flush this half-written row; and `idx = len(self)` read a cached length
        that went stale as soon as anything else wrote to the session.
        """
        with self._db.transaction():
            return self._append_locked(result_dict)

    def _append_locked(self, result_dict: dict) -> int:
        # Recomputed from the DB under the transaction lock, not from _len.
        idx = self._count_rows()
        result_dict["_index"] = idx
        scalars = _extract_scalars(result_dict)
        blob = _compress(result_dict)
        self._db.execute(
            """INSERT INTO results
               (session_id, idx, prompt, category, seq_len,
                stress_score, net_correction,
                entropy, top2_share, middle_share, interior_cv,
                kl_divergence, n_negative_tokens, has_negative_tokens,
                ltp_mean_m, ltp_mean_v, ltp_max_prc, ltp_n_directional,
                sfd_density_mean, rd_mean_tau, rd_mean_overlap,
                delta_scale, full_capture_enabled,
                family_index, rung_index,
                data_blob)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self._session_id, idx,
                scalars["prompt"], scalars["category"], scalars["seq_len"],
                scalars["stress_score"], scalars["net_correction"],
                scalars["entropy"], scalars["top2_share"], scalars["middle_share"],
                scalars["interior_cv"], scalars["kl_divergence"],
                scalars["n_negative_tokens"], scalars["has_negative_tokens"],
                scalars["ltp_mean_m"], scalars["ltp_mean_v"], scalars["ltp_max_prc"],
                scalars["ltp_n_directional"], scalars["sfd_density_mean"],
                scalars["rd_mean_tau"], scalars["rd_mean_overlap"],
                scalars["delta_scale"],
                scalars["full_capture_enabled"],
                scalars["family_index"], scalars["rung_index"],
                blob,
            ),
        )
        # No commit here — the enclosing transaction() commits on exit.
        # Invalidate rather than increment: if the enclosing commit fails and
        # rolls back, an incremented _len would leave every later __len__ and
        # __getitem__ off by one.  None forces a recount from the DB.
        self._len = None
        return idx

    def clear(self):
        """Delete all results for this session."""
        self._db.execute(
            "DELETE FROM results WHERE session_id = ?",
            (self._session_id,),
        )
        self._db.commit()
        self.invalidate()

    # ── Internal ────────────────────────────────────────────────

    def _load(self, idx: int) -> dict:
        if idx in self._cache:
            return self._cache[idx]
        row = self._db.query_one(
            "SELECT data_blob FROM results WHERE session_id = ? AND idx = ?",
            (self._session_id, idx),
        )
        if row is None:
            raise IndexError(f"No result at index {idx} in session {self._session_id}")
        d = _decompress(row[0])
        d["_index"] = idx
        # Simple bounded cache
        if len(self._cache) >= self._cache_max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[idx] = d
        return d


# ── Database ────────────────────────────────────────────────────

class Database:
    """Singleton-ish SQLite connection with schema bootstrapping.

    Usage::

        db = get_db()            # module-level singleton
        db.add_model(...)
        db.insert_result(...)
    """

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path) if path else _default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        # One connection shared across the FastAPI threadpool, the load
        # worker, the job thread, and module threads
        # (check_same_thread=False). Serialize it: without this,
        # multi-statement writes from two threads interleave and a
        # commit() from one thread flushes another's half-finished
        # transaction. RLock so transaction() can nest the per-statement
        # acquisitions.
        self._lock = threading.RLock()
        # Depth of open transaction() blocks. >0 suppresses commit().
        self._txn_depth = 0
        self._connect()
        self._bootstrap()

    @contextmanager
    def transaction(self):
        """Hold the connection for a multi-statement write, then commit.

        Use for any logical transaction spanning more than one execute()
        (e.g. delete-then-reinsert patterns) so no other thread can
        interleave statements or commit mid-way.

        While a transaction is open, ``commit()`` is a NO-OP.  Several helpers
        (create_session, insert_result, set_config, ...) commit internally, so
        without this suppression a "transaction" wrapping them committed every
        statement as it ran and rollback could only ever undo the last one —
        which silently defeated the atomicity the caller asked for.  Nesting
        is depth-counted: only the outermost block commits.
        """
        with self._lock:
            self._txn_depth += 1
            try:
                yield self
            except Exception:
                # Unwind to zero so a failed inner block cannot leave commits
                # suppressed for the rest of the connection's life.
                self._txn_depth = 0
                self._conn.rollback()
                raise
            else:
                self._txn_depth -= 1
                if self._txn_depth == 0:
                    self._conn.commit()

    # ── Connection management ───────────────────────────────────

    def _connect(self):
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=30,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")

    def _bootstrap(self):
        """Create tables if missing, run migrations."""
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate_columns()
        # Stamp version
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            ("version", str(_SCHEMA_VERSION)),
        )
        self._conn.commit()
        logger.info(f"[db] Database ready at {self.path}")

    def _migrate_columns(self):
        """Add any columns missing from the results table.

        ``CREATE TABLE IF NOT EXISTS`` won't alter an existing table,
        so when the schema gains new indexed columns we need to add
        them via ``ALTER TABLE``.
        """
        # Desired columns and their DDL types/defaults
        expected = {
            "delta_scale": "REAL",
            "full_capture_enabled": "INTEGER DEFAULT 0",
            "family_index": "INTEGER",
            "rung_index": "INTEGER",
        }
        # Get existing column names
        cursor = self._conn.execute("PRAGMA table_info(results)")
        existing = {row[1] for row in cursor.fetchall()}
        for col, typedef in expected.items():
            if col not in existing:
                self._conn.execute(
                    f"ALTER TABLE results ADD COLUMN {col} {typedef}"
                )
                logger.info(f"[db] Added column results.{col}")
        self._conn.commit()

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        """Run a statement and return its cursor.

        NOTE for SELECT callers: the lock is released when this returns, so
        consuming the cursor afterwards reads rows *outside* the lock, and an
        interleaved write on this shared connection can shift the result set
        underneath you.  Use query_all()/query_one(), which fetch inside the
        lock.  This method remains for INSERT/UPDATE/DELETE.
        """
        with self._lock:
            return self._conn.execute(sql, params)

    def query_all(self, sql: str, params=()) -> list:
        """SELECT, fully fetched while the connection lock is still held."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params=()):
        """Single-row SELECT, fetched while the connection lock is held."""
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def executemany(self, sql: str, seq) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executemany(sql, seq)

    def commit(self):
        """Commit, unless a transaction() block is open.

        Suppressing the commit inside a transaction is what makes the helpers
        that commit internally (create_session, insert_result, ...) safe to
        compose into one atomic unit.
        """
        with self._lock:
            if self._txn_depth == 0:
                self._conn.commit()

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ── Models ──────────────────────────────────────────────────

    def list_models(self) -> list[dict]:
        rows = self.query_all(
            "SELECT id, name, base, instruct, ram_gb, notes FROM models ORDER BY id"
        )
        return [
            {"id": r[0], "name": r[1], "base": r[2],
             "instruct": r[3], "ram_gb": r[4], "notes": r[5]}
            for r in rows
        ]

    def upsert_model(self, id: str, name: str, base: str, instruct: str,
                     ram_gb: int = 0, notes: str = "") -> None:
        self.execute(
            """INSERT INTO models (id, name, base, instruct, ram_gb, notes)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name, base=excluded.base,
                   instruct=excluded.instruct, ram_gb=excluded.ram_gb,
                   notes=excluded.notes""",
            (id, name, base, instruct, ram_gb, notes),
        )
        self.commit()

    def get_model(self, id: str) -> Optional[dict]:
        row = self.query_one(
            "SELECT id, name, base, instruct, ram_gb, notes FROM models WHERE id = ?",
            (id,),
        )
        if row is None:
            return None
        return {"id": row[0], "name": row[1], "base": row[2],
                "instruct": row[3], "ram_gb": row[4], "notes": row[5]}

    # ── Sessions ────────────────────────────────────────────────

    def create_session(self, session_id: str, model: str = "",
                       started: str = "") -> None:
        self.execute(
            """INSERT OR IGNORE INTO sessions (session_id, model, started)
               VALUES (?, ?, ?)""",
            (session_id, model, started),
        )
        self.commit()

    def update_session_model(self, session_id: str, model: str) -> None:
        self.execute(
            "UPDATE sessions SET model = ? WHERE session_id = ?",
            (model, session_id),
        )
        self.commit()

    def get_session_meta(self, session_id: str) -> Optional[dict]:
        row = self.query_one(
            "SELECT session_id, model, started, status FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        if row is None:
            return None
        return {"session_id": row[0], "model": row[1],
                "started": row[2], "status": row[3]}

    def list_sessions(self) -> list[dict]:
        rows = self.query_all(
            """SELECT s.session_id, s.model, s.started, s.status,
                      COUNT(r.id) as n_results
               FROM sessions s
               LEFT JOIN results r ON r.session_id = s.session_id
               GROUP BY s.session_id
               ORDER BY s.created_at DESC"""
        )
        return [
            {"session_id": r[0], "model": r[1], "started": r[2],
             "status": r[3], "n_results": r[4]}
            for r in rows
        ]

    def session_result_count(self, session_id: str) -> int:
        row = self.query_one(
            "SELECT COUNT(*) FROM results WHERE session_id = ?",
            (session_id,),
        )
        return row[0] if row else 0

    def session_categories(self, session_id: str) -> dict[str, int]:
        rows = self.query_all(
            "SELECT category, COUNT(*) FROM results WHERE session_id = ? GROUP BY category",
            (session_id,),
        )
        return {r[0] or "unknown": r[1] for r in rows}

    # ── Results ─────────────────────────────────────────────────

    def insert_result(self, session_id: str, idx: int,
                      result_dict: dict) -> int:
        """Insert a result, returning the row id."""
        scalars = _extract_scalars(result_dict)
        blob = _compress(result_dict)
        cursor = self.execute(
            """INSERT INTO results
               (session_id, idx, prompt, category, seq_len,
                stress_score, net_correction,
                entropy, top2_share, middle_share, interior_cv,
                kl_divergence, n_negative_tokens, has_negative_tokens,
                ltp_mean_m, ltp_mean_v, ltp_max_prc, ltp_n_directional,
                sfd_density_mean, rd_mean_tau, rd_mean_overlap,
                delta_scale, full_capture_enabled,
                family_index, rung_index,
                data_blob)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id, idx,
                scalars["prompt"], scalars["category"], scalars["seq_len"],
                scalars["stress_score"], scalars["net_correction"],
                scalars["entropy"], scalars["top2_share"], scalars["middle_share"],
                scalars["interior_cv"], scalars["kl_divergence"],
                scalars["n_negative_tokens"], scalars["has_negative_tokens"],
                scalars["ltp_mean_m"], scalars["ltp_mean_v"], scalars["ltp_max_prc"],
                scalars["ltp_n_directional"], scalars["sfd_density_mean"],
                scalars["rd_mean_tau"], scalars["rd_mean_overlap"],
                scalars["delta_scale"],
                scalars["full_capture_enabled"],
                scalars["family_index"], scalars["rung_index"],
                blob,
            ),
        )
        self.commit()
        return cursor.lastrowid

    def get_result(self, session_id: str, idx: int) -> Optional[dict]:
        row = self.query_one(
            "SELECT data_blob FROM results WHERE session_id = ? AND idx = ?",
            (session_id, idx),
        )
        if row is None:
            return None
        d = _decompress(row[0])
        d["_index"] = idx
        return d

    def get_results_page(self, session_id: str, offset: int = 0,
                         limit: int = 50) -> list[dict]:
        """Load a page of full result dicts."""
        rows = self.query_all(
            "SELECT idx, data_blob FROM results "
            "WHERE session_id = ? ORDER BY idx LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        )
        out = []
        for idx, blob in rows:
            d = _decompress(blob)
            d["_index"] = idx
            out.append(d)
        return out

    def get_dashboard_rows(self, session_id: str) -> list[dict]:
        """Slim scalar-only rows for the dashboard — no decompression."""
        rows = self.query_all(
            """SELECT idx, prompt, category, seq_len,
                      stress_score, net_correction, entropy,
                      top2_share, middle_share, interior_cv,
                      kl_divergence, n_negative_tokens, has_negative_tokens,
                      ltp_mean_m, ltp_mean_v, ltp_max_prc, ltp_n_directional,
                      sfd_density_mean, rd_mean_tau, rd_mean_overlap,
                      delta_scale, full_capture_enabled,
                      family_index, rung_index
               FROM results
               WHERE session_id = ?
               ORDER BY idx""",
            (session_id,),
        )
        out = []
        for r in rows:
            s = {
                "_index": r[0], "prompt": r[1], "category": r[2],
                "seq_len": r[3], "stress_score": r[4],
                "net_correction": r[5], "entropy": r[6],
                "top2_share": r[7], "middle_share": r[8],
                "interior_cv": r[9], "kl_divergence": r[10],
                "n_negative_tokens": r[11],
                "has_negative_tokens": bool(r[12]),
                "delta_scale": r[20],
                "full_capture_enabled": bool(r[21]),
                "family_index": r[22],
                "rung_index": r[23],
            }
            # Nested structures the dashboard expects
            if r[13] is not None:
                s["ltp"] = {
                    "mean_M": r[13], "mean_V": r[14],
                    "max_prc": r[15], "n_directional": r[16],
                }
            if r[17] is not None:
                s["sfd"] = {"density_mean": r[17]}
            if r[18] is not None:
                s["rank_displacement"] = {
                    "mean_tau": r[18], "mean_overlap": r[19],
                }
            out.append(s)
        return out

    def next_family_base(self, session_id: str) -> int:
        """Lowest family_index value safe to assign to a new ladder.

        Returns ``MAX(family_index) + 1`` for the session (0 if none).
        Monotonic against surviving rows, so it never reuses a family id
        that an existing ladder still holds — even after remove_results
        reindexes idx.
        """
        row = self.query_one(
            "SELECT MAX(family_index) FROM results WHERE session_id = ?",
            (session_id,),
        )
        return int(row[0]) + 1 if row and row[0] is not None else 0

    def remove_results(self, session_id: str, indices: list[int]) -> None:
        """Remove specific result indices and reindex remaining rows."""
        if not indices:
            return
        placeholders = ",".join("?" * len(indices))
        with self.transaction():
            self.execute(
                f"DELETE FROM results WHERE session_id = ? AND idx IN ({placeholders})",
                [session_id] + list(indices),
            )
            # Reindex remaining rows
            rows = self.query_all(
                "SELECT id, idx FROM results WHERE session_id = ? ORDER BY idx",
                (session_id,),
            )
            for new_idx, (row_id, _old_idx) in enumerate(rows):
                self.execute(
                    "UPDATE results SET idx = ? WHERE id = ?",
                    (new_idx, row_id),
                )

    def delete_session_results(self, session_id: str) -> None:
        self.execute(
            "DELETE FROM results WHERE session_id = ?", (session_id,),
        )
        self.commit()

    # ── Config key-value store ──────────────────────────────────

    def get_config(self, namespace: str) -> dict:
        rows = self.query_all(
            "SELECT key, value FROM config WHERE namespace = ?",
            (namespace,),
        )
        out = {}
        for k, v in rows:
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                out[k] = v
        return out

    def set_config(self, namespace: str, data: dict) -> None:
        """Replace all keys in a namespace (atomically)."""
        with self.transaction():
            self.execute("DELETE FROM config WHERE namespace = ?", (namespace,))
            for k, v in data.items():
                self.execute(
                    "INSERT INTO config (namespace, key, value) VALUES (?, ?, ?)",
                    (namespace, k, json.dumps(v)),
                )

    def set_config_key(self, namespace: str, key: str, value) -> None:
        self.execute(
            """INSERT INTO config (namespace, key, value) VALUES (?, ?, ?)
               ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value""",
            (namespace, key, json.dumps(value)),
        )
        self.commit()

    # ── Prompts library ─────────────────────────────────────────

    def list_prompts(self) -> list[dict]:
        rows = self.query_all(
            "SELECT prompt, category FROM prompts ORDER BY id"
        )
        return [{"prompt": r[0], "category": r[1]} for r in rows]

    def add_prompt(self, prompt: str, category: str = "") -> None:
        self.execute(
            "INSERT INTO prompts (prompt, category) VALUES (?, ?)",
            (prompt, category),
        )
        self.commit()

    # ── DB size (replaces get_cache_size) ───────────────────────

    def db_size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def session_size_bytes(self, session_id: str) -> int:
        """Approximate bytes for one session's results."""
        row = self.query_one(
            "SELECT SUM(LENGTH(data_blob)) FROM results WHERE session_id = ?",
            (session_id,),
        )
        return row[0] or 0


# ── Migration ───────────────────────────────────────────────────

def migrate_json_to_db(db: Database, project_root: Path) -> dict:
    """Import existing JSON files into the database.

    Renames originals to ``*.migrated`` after success. Idempotent:
    skips files that have already been migrated.

    Returns a summary dict of what was imported.
    """
    summary = {"models": 0, "results": 0, "config": False, "prompts": 0}

    # ── models.json ─────────────────────────────────────────────
    models_file = project_root / "models.json"
    if models_file.exists():
        try:
            with open(models_file) as f:
                models = json.load(f)
            for m in models:
                db.upsert_model(
                    id=m.get("id", ""),
                    name=m.get("name", ""),
                    base=m.get("base", ""),
                    instruct=m.get("instruct", ""),
                    ram_gb=m.get("ram_gb", 0),
                    notes=m.get("notes", ""),
                )
                summary["models"] += 1
            # Rename like the other legacy files (the docs always claimed
            # this happened). Without the rename, every startup re-imported
            # the file and silently reverted any model edits made via
            # POST /api/models back to the file's values.
            models_file.rename(models_file.with_suffix(".json.migrated"))
            logger.info(f"[migrate] Imported {summary['models']} models "
                        f"(models.json -> models.json.migrated)")
        except Exception as e:
            logger.warning(f"[migrate] models.json failed: {e}")

    # ── datasets/current/results.json ───────────────────────────
    results_file = project_root / "datasets" / "current" / "results.json"
    session_meta_file = project_root / "datasets" / "current" / "session.json"
    if results_file.exists():
        try:
            # Read session metadata
            session_id = f"migrated_{int(time.time())}"
            model_name = ""
            started = ""
            if session_meta_file.exists():
                with open(session_meta_file) as f:
                    meta = json.load(f)
                session_id = meta.get("session_id", session_id)
                model_name = meta.get("model", "")
                started = meta.get("started", "")

            # Read results
            with open(results_file) as f:
                results = json.load(f)

            if isinstance(results, list) and results:
                # All-or-nothing.  Previously each result was inserted in its
                # own transaction and the source file was renamed only after
                # the loop, so one failing insert left the session
                # half-imported AND left results.json in place — the next
                # startup retried the same session_id and died on
                # UNIQUE(session_id, idx) at i=0, permanently wedged, with the
                # failure swallowed by the except below.
                with db.transaction():
                    db.create_session(session_id, model_name, started)
                    for i, rd in enumerate(results):
                        rd["_index"] = i
                        db.insert_result(session_id, i, rd)
                        summary["results"] += 1

                results_file.rename(results_file.with_suffix(".json.migrated"))
                if session_meta_file.exists():
                    session_meta_file.rename(
                        session_meta_file.with_suffix(".json.migrated"))
                logger.info(
                    f"[migrate] Imported {summary['results']} results "
                    f"into session {session_id}")
        except Exception as e:
            # The transaction rolled back, so the DB is clean and results.json
            # is untouched; the next startup retries from scratch.
            summary["results"] = 0
            logger.warning(
                f"[migrate] results.json failed, rolled back "
                f"(will retry on next startup): {e}", exc_info=True)

    # ── ui_config.json ──────────────────────────────────────────
    config_file = project_root / "ui_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config_data = json.load(f)
            db.set_config("ui", config_data)
            config_file.rename(config_file.with_suffix(".json.migrated"))
            summary["config"] = True
            logger.info("[migrate] Imported ui_config.json")
        except Exception as e:
            logger.warning(f"[migrate] ui_config.json failed: {e}")

    # ── prompts.csv ─────────────────────────────────────────────
    import csv
    prompts_file = project_root / "prompts.csv"
    if prompts_file.exists() and db.query_one(
            "SELECT COUNT(*) FROM prompts")[0] == 0:
        try:
            with open(prompts_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    db.add_prompt(
                        row.get("prompt", ""),
                        row.get("category", ""),
                    )
                    summary["prompts"] += 1
            logger.info(f"[migrate] Imported {summary['prompts']} prompts")
            # Don't rename prompts.csv — it's also a data file users edit
        except Exception as e:
            logger.warning(f"[migrate] prompts.csv failed: {e}")

    return summary


# ── Module-level singleton ──────────────────────────────────────

_db_instance: Optional[Database] = None


def get_db(path: Optional[Union[str, Path]] = None) -> Database:
    """Return the module-level Database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(path)
    return _db_instance
