"""Session export and import.

Exports the session's flat result dicts as gzipped JSON.

When `opts={"moduleResults": True}` and a `module_runner` is provided,
each completed module's `run()` output is embedded under a top-level
`modules` key, mapping module name → result dict. Backward compatible:
consumers that don't know about `modules` see the original
{session_id, model, n_results, results} shape unchanged.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ─── Filename schema ─────────────────────────────────────────────────
#
# Exports get a descriptive dynamic filename that captures the minimum
# useful provenance: tool tag, model, prompt count, optional content
# marker, timestamp. Keeps same-day exports unambiguous and makes a
# directory of exports legible at a glance.
#
#   tagm_{model_slug}_{N}p[+modules]_{timestamp}.json.gz
#
# Examples:
#   tagm_llama-3.2-1b_52p_20260504_141502.json.gz
#   tagm_qwen2.5-1.5b_120p+modules_20260504_141502.json.gz
#   tagm_52p_20260504_141502.json.gz                  (no model loaded)

# Common HF id suffixes that don't carry useful identity for our purposes.
# Stripped case-insensitively from the end of the slug after the bare
# model name has been extracted. Order matters when suffixes nest, so
# longest-first.
_MODEL_NAME_NOISE_SUFFIXES = (
    "-instruct", "-chat", "-it", "-base",
)


def _slug_model_name(name: str) -> str:
    """Reduce an HF model id to a short, filename-safe slug.

    `meta-llama/Llama-3.2-1B-Instruct` → `llama-3.2-1b`
    `Qwen/Qwen2.5-1.5B`                 → `qwen2.5-1.5b`
    `""`                                 → `""` (caller decides what to do)
    """
    if not name:
        return ""
    # Take the segment after the last slash (drops org prefix).
    short = name.rsplit("/", 1)[-1].lower()
    # Strip noise suffixes iteratively so '-it-instruct' becomes ''.
    changed = True
    while changed:
        changed = False
        for suf in _MODEL_NAME_NOISE_SUFFIXES:
            if short.endswith(suf):
                short = short[: -len(suf)]
                changed = True
                break
    # Replace anything that isn't a-z0-9.- with a hyphen, then collapse
    # repeats and trim. We keep '.' because version dots ('3.2', '1.5b')
    # are part of how humans recognize these models at a glance.
    short = re.sub(r"[^a-z0-9.\-]+", "-", short)
    short = re.sub(r"-+", "-", short).strip("-")
    return short


def build_export_filename(
    session,
    opts: Optional[dict] = None,
    *,
    timestamp: Optional[str] = None,
) -> str:
    """Build the output filename for a session export.

    See module docstring for the schema. `timestamp` is an injectable
    override for tests; production callers leave it unset and get a
    fresh wall-clock stamp at call time.
    """
    opts = opts or {}
    parts = ["tagm"]

    model_slug = _slug_model_name(getattr(session, "model_name", "") or "")
    if model_slug:
        parts.append(model_slug)

    n_results = len(getattr(session, "results", []) or [])
    count_part = f"{n_results}p"
    # Markers attach to the prompt count to keep them visually associated
    # with content scope rather than freestanding ('52p+modules' reads as
    # "52 prompts plus module results").
    if opts.get("moduleResults"):
        count_part += "+modules"
    parts.append(count_part)

    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)

    return "_".join(parts) + ".json.gz"


def export_session(
    session,
    path: Union[str, Path],
    *,
    module_runner: Optional[Any] = None,
    opts: Optional[dict] = None,
) -> Path:
    """Write a session to disk as gzipped JSON.

    Args:
        session: the live Session object.
        path: output path (parent dirs created if missing).
        module_runner: optional ModuleRunner; required if opts wants modules.
        opts: dict of export toggles forwarded from the UI. Recognized keys:
            - moduleResults (bool): if true, embed completed-module outputs
              under a top-level `modules` key.
            Other format toggles (pdf, json, charts) are accepted for
            forward compatibility but currently no-op here.

    Returns:
        The path written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    opts = opts or {}

    data: dict[str, Any] = {
        "session_id": getattr(session, "session_id", ""),
        "model": getattr(session, "model_name", ""),
        "n_results": len(session.results),
        "results": session.results,
    }

    if opts.get("moduleResults") and module_runner is not None:
        modules_payload = _collect_module_results(module_runner)
        # Always emit the key when requested, even if empty — distinguishes
        # "user asked, nothing was completed" from "user didn't ask".
        data["modules"] = modules_payload

    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=_json_default)
    return p


def _collect_module_results(module_runner) -> dict:
    """Return {name: results_dict} for every completed module.

    Reads from the runner's in-memory state (the canonical copy used by
    the /api/modules/{name}/results endpoint). Modules that are idle,
    running, errored, or have no results are skipped silently — best
    effort: whatever has run successfully at export time, ships.
    """
    out: dict[str, Any] = {}
    state_map = getattr(module_runner, "_state", {}) or {}
    for name, st in state_map.items():
        try:
            if getattr(st, "status", None) != "completed":
                continue
            results = getattr(st, "results", None)
            if results is None:
                continue
            out[name] = results
        except Exception as e:
            # Best-effort: a single misbehaving module shouldn't tank the
            # whole export. Log and skip.
            logger.warning(f"[EXPORT] skipping module {name}: {e}")
            continue
    return out


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
