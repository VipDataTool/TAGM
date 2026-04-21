"""Session export and import.

JSON format, gzipped on write (.json.gz). Stdlib only. Versioned
schema with explicit migration path: if SCHEMA_VERSION changes,
`load_session` dispatches on the stored version and applies migrations.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Union

from tagm.service.session import SCHEMA_VERSION, Session, SessionRecord


def export_session(session: Union[Session, SessionRecord],
                    path: Union[str, Path]) -> Path:
    """Write a session to disk as gzipped JSON.

    Returns the path written. Creates parent directories as needed.
    """
    rec = session.record if isinstance(session, Session) else session
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = rec.to_dict()
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=_json_default)
    return p


def load_session(path: Union[str, Path]) -> SessionRecord:
    """Load a session from disk. Handles both .json and .json.gz."""
    p = Path(path)
    if p.suffix == ".gz" or str(p).endswith(".json.gz"):
        opener = lambda: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda: open(p, "r", encoding="utf-8")
    with opener() as f:
        data = json.load(f)

    stored_version = int(data.get("schema_version", 1))
    if stored_version > SCHEMA_VERSION:
        raise ValueError(
            f"Session at {p} was written with schema_version={stored_version}, "
            f"but this TAGM only supports up to {SCHEMA_VERSION}. "
            f"Upgrade TAGM or export with an older version."
        )
    # Future: migration dispatch here when SCHEMA_VERSION > 1 and
    # stored_version < SCHEMA_VERSION.

    return SessionRecord.from_dict(data)


def _json_default(obj):
    """Fallback for json.dump to handle numpy scalars/arrays."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        import math
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
