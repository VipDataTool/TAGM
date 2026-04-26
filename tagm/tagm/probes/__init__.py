"""Probe subsystem.

Reads probe CSVs (input templates and Probe Generator's autoprobe outputs),
runs them through the instruct model to produce per-depth embeddings, and
caches results to ``<project_root>/probe_cache/``. The active probe set
is recorded in ``<project_root>/probe_config.json``.

Everything lives in :mod:`tagm.probes.io`. See that module for the public
surface (``embed_and_cache_probes``, ``load_probes``, ``detect_level_cols``,
``parse_meta``, ``load_probe_cache``, ``probe_cache_path``,
``get_active_probe``, plus the ``PROBE_CACHE_DIR`` / ``PROBE_CONFIG`` /
``META_TAG`` / ``FIXED_COLS`` / ``NORM_EPS`` constants).
"""
from tagm.probes.io import (
    embed_and_cache_probes,
    embed_and_activate_probe_set,
    load_probes,
    detect_level_cols,
    parse_meta,
    load_probe_cache,
    probe_cache_path,
    get_active_probe,
    set_active_probe,
    PROBE_CACHE_DIR,
    PROBE_CONFIG,
    META_TAG,
    FIXED_COLS,
    NORM_EPS,
)

__all__ = [
    "embed_and_cache_probes",
    "embed_and_activate_probe_set",
    "load_probes",
    "detect_level_cols",
    "parse_meta",
    "load_probe_cache",
    "probe_cache_path",
    "get_active_probe",
    "set_active_probe",
    "PROBE_CACHE_DIR",
    "PROBE_CONFIG",
    "META_TAG",
    "FIXED_COLS",
    "NORM_EPS",
]
