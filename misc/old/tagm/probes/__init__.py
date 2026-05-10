"""Probe subsystem.

Reads probe CSVs (input templates and Probe Generator's autoprobe outputs),
runs them through the instruct model to produce per-depth embeddings, and
caches results to ``<project_root>/probe_cache/``. The active probe set
is recorded in ``<project_root>/probe_config.json``.

Everything lives in :mod:`tagm.probes.io`. The active-set state — the
binding between probe CSV, model id, and embedding depths — is exposed
as :class:`ActiveProbeSet`; consumer modules should call
:func:`get_active_probe_set` to obtain it and route every cache lookup
through that object's :meth:`cache_path` and :meth:`validate_against`.
"""
from misc.old.tagm.probes.io import (
    ActiveProbeSet,
    embed_and_cache_probes,
    embed_and_activate_probe_set,
    load_probes,
    detect_level_cols,
    parse_meta,
    load_probe_cache,
    probe_cache_path,
    get_active_probe,
    get_active_probe_set,
    set_active_probe,
    PROBE_CACHE_DIR,
    PROBE_CONFIG,
    META_TAG,
    FIXED_COLS,
    NORM_EPS,
)

__all__ = [
    "ActiveProbeSet",
    "embed_and_cache_probes",
    "embed_and_activate_probe_set",
    "load_probes",
    "detect_level_cols",
    "parse_meta",
    "load_probe_cache",
    "probe_cache_path",
    "get_active_probe",
    "get_active_probe_set",
    "set_active_probe",
    "PROBE_CACHE_DIR",
    "PROBE_CONFIG",
    "META_TAG",
    "FIXED_COLS",
    "NORM_EPS",
]
