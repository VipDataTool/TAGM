"""Session export and import.

Exports the session's result dicts as gzipped JSON. Works with both
the legacy list-based session and the new DB-backed ResultsList.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Union

import numpy as np


def export_session(session, path: Union[str, Path]) -> Path:
    """Write a session to disk as gzipped JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Materialize results from DB-backed ResultsList
    results = list(session.results)
    data = {
        "session_id": getattr(session, "session_id", ""),
        "model": getattr(session, "model_name", ""),
        "n_results": len(results),
        "results": results,
    }
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=_json_default)
    return p


def load_session(path: Union[str, Path]) -> dict:
    """Load a session from gzipped JSON. Returns the data dict."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return str(obj)
