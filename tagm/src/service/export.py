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

    Streams results from the database one at a time. Peak memory is
    proportional to one result, not the full session — safe for large
    batches on memory-constrained hardware.

    float_precision controls significant digits in the embedding CSV (default 12).
    module_results is {name: results_dict} from ModuleRunner.collect_results().

    Returns the path to the zip file.
    """
    p = Path(path)
    if p.suffix != ".zip":
        p = p.with_suffix(".zip")
    p.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(session.results)
    page_size = 20  # process 20 results at a time

    # First pass (one page): detect embedding dimensionality
    n_dims = 0
    first_page = session.get_results_page(0, 1)
    if first_page:
        for field in _EMBEDDING_FIELDS:
            emb = first_page[0].get(field)
            if emb and len(emb) > 0 and len(emb[0]) > 0:
                n_dims = len(emb[0])
                break

    # Module report filenames
    module_results = module_results or {}
    module_files = {name: f"modules/{name}.json.gz" for name in module_results}

    fmt = f"%.{float_precision}g"

    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:

        # ── Pass 1: build lean JSON + embedding CSVs simultaneously ──
        # Lean results (no embeddings) are small enough to hold in RAM.
        # Embeddings are written to CSV buffers as we go.

        lean_results = []

        # Prepare CSV buffers for each embedding layer
        emb_buffers = {}
        emb_gzips = {}
        emb_writers = {}
        if n_dims > 0:
            header = ["prompt_index", "token_index"] + [
                f"dim_{i}" for i in range(n_dims)
            ]
            for field, csv_name in _EMB_CSV_NAMES.items():
                buf = io.BytesIO()
                gz = gzip.open(buf, "wt", encoding="utf-8", newline="")
                writer = csv.writer(gz)
                writer.writerow(header)
                emb_buffers[field] = buf
                emb_gzips[field] = gz
                emb_writers[field] = writer

        # Stream results in pages
        for offset in range(0, n_total, page_size):
            page = session.get_results_page(offset, page_size)
            for local_i, r in enumerate(page):
                pi = offset + local_i

                # Build lean result (strip embeddings)
                lean = {k: v for k, v in r.items()
                        if k not in _EMBEDDING_FIELDS}
                lean_results.append(lean)

                # Write embeddings to CSV writers
                if n_dims > 0:
                    for field in _EMBEDDING_FIELDS:
                        emb = r.get(field)
                        if not emb:
                            continue
                        writer = emb_writers.get(field)
                        if writer is None:
                            continue
                        for ti, vec in enumerate(emb):
                            row = [pi, ti] + [fmt % v for v in vec]
                            writer.writerow(row)

                # Let the page's results be GC'd after processing
            del page

        # Close gzip streams and write to zip
        if n_dims > 0:
            for field, csv_name in _EMB_CSV_NAMES.items():
                emb_gzips[field].close()
                zf.writestr(csv_name, emb_buffers[field].getvalue())
                emb_buffers[field].close()

        # Write the lean JSON
        session_data = {
            "session_id": getattr(session, "session_id", ""),
            "model": getattr(session, "model_name", ""),
            "n_results": len(lean_results),
            "results": lean_results,
            "_embedding_files": list(_EMB_CSV_NAMES.values()),
            "_embedding_dims": n_dims,
            "_module_files": module_files,
        }

        json_buf = io.BytesIO()
        with gzip.open(json_buf, "wt", encoding="utf-8") as gf:
            json.dump(session_data, gf, indent=1, default=_json_default)
        zf.writestr("session.json.gz", json_buf.getvalue())

        # Write module reports
        for name, mod_data in module_results.items():
            mod_buf = io.BytesIO()
            with gzip.open(mod_buf, "wt", encoding="utf-8") as gf:
                json.dump(mod_data, gf, indent=1, default=_json_default)
            zf.writestr(f"modules/{name}.json.gz", mod_buf.getvalue())

    return p

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
