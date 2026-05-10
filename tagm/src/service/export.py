"""Session export and import.

Exports the session's result dicts as gzipped JSON. Works with both
the legacy list-based session and the new DB-backed ResultsList.

Two export modes:
  - export_session():       Single gzipped JSON (original, simple, large)
  - export_session_split(): Zip archive with lean JSON + embedding CSVs
                            (~60% smaller, browser-friendly JSON)
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Union

import numpy as np

# Embedding fields that get split into companion CSVs.
_EMBEDDING_FIELDS = (
    "per_token_domain_emb",
    "per_token_escalation_emb",
    "per_token_final_emb",
)

# CSV filenames inside the zip, keyed by field name.
_EMB_CSV_NAMES = {
    "per_token_domain_emb":       "embeddings_domain.csv.gz",
    "per_token_escalation_emb":   "embeddings_escalation.csv.gz",
    "per_token_final_emb":        "embeddings_final.csv.gz",
}


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


def export_session_split(session, path: Union[str, Path],
                         float_precision: int = 12,
                         module_results: dict = None) -> Path:
    """Write a session as a zip: lean JSON + embedding CSVs + module reports.

    The zip contains:
      session.json.gz              Scalar/structural data (no embeddings)
      embeddings_domain.csv.gz     Per-token hidden states at domain layer
      embeddings_escalation.csv.gz Per-token hidden states at escalation layer
      embeddings_final.csv.gz      Per-token hidden states at final layer
      modules/_{name}.json.gz       Module analytical reports (one per module)

    float_precision controls significant digits in the embedding CSV (default 12).
    module_results is {name: results_dict} from ModuleRunner.collect_results().

    Returns the path to the zip file.
    """
    p = Path(path)
    if p.suffix != ".zip":
        p = p.with_suffix(".zip")
    p.parent.mkdir(parents=True, exist_ok=True)

    results = list(session.results)

    # Detect embedding dimensionality from first result that has embeddings
    n_dims = 0
    for r in results:
        for field in _EMBEDDING_FIELDS:
            emb = r.get(field)
            if emb and len(emb) > 0 and len(emb[0]) > 0:
                n_dims = len(emb[0])
                break
        if n_dims > 0:
            break

    # Build lean results (strip embedding fields)
    lean_results = []
    for r in results:
        lean = {k: v for k, v in r.items() if k not in _EMBEDDING_FIELDS}
        lean_results.append(lean)

    # Module report filenames
    module_results = module_results or {}
    module_files = {name: f"modules/{name}.json.gz" for name in module_results}

    session_data = {
        "session_id": getattr(session, "session_id", ""),
        "model": getattr(session, "model_name", ""),
        "n_results": len(lean_results),
        "results": lean_results,
        "_embedding_files": list(_EMB_CSV_NAMES.values()),
        "_embedding_dims": n_dims,
        "_module_files": module_files,
    }

    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        # Write the lean JSON (gzip inside zip for max compression)
        json_buf = io.BytesIO()
        with gzip.open(json_buf, "wt", encoding="utf-8") as gf:
            json.dump(session_data, gf, indent=1, default=_json_default)
        zf.writestr("session.json.gz", json_buf.getvalue())

        # Write each embedding layer as a gzipped CSV
        if n_dims > 0:
            header = ["prompt_index", "token_index"] + [
                f"dim_{i}" for i in range(n_dims)
            ]

            for field, csv_name in _EMB_CSV_NAMES.items():
                csv_buf = io.BytesIO()
                fmt = f"%.{float_precision}g"
                with gzip.open(csv_buf, "wt", encoding="utf-8",
                               newline="") as gf:
                    writer = csv.writer(gf)
                    writer.writerow(header)
                    for pi, r in enumerate(results):
                        emb = r.get(field)
                        if not emb:
                            continue
                        for ti, vec in enumerate(emb):
                            row = [pi, ti] + [
                                fmt % v for v in vec
                            ]
                            writer.writerow(row)
                zf.writestr(csv_name, csv_buf.getvalue())

        # Write module reports
        for name, mod_data in module_results.items():
            mod_buf = io.BytesIO()
            with gzip.open(mod_buf, "wt", encoding="utf-8") as gf:
                json.dump(mod_data, gf, indent=1, default=_json_default)
            zf.writestr(f"modules/{name}.json.gz", mod_buf.getvalue())

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
