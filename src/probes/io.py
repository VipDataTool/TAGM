"""Probe-IO helpers: CSV loading + per-depth embedding cache.

The probe-set apply flow:

1. The user uploads a probe CSV (or the Probe Generator module produces
   one — its "autoprobes" output is a flat ``subject, subclass, text``
   CSV with one term per row).
2. ``embed_and_cache_probes`` reads the CSV, runs each probe text
   through the instruct model with a forward hook on
   ``pre_attn_norm`` at the requested depth, and writes a JSON cache
   to ``<project_root>/probe_cache/<stem>__<safe_model>__L<frac>.json``.
3. The active probe set is recorded in
   ``<project_root>/probe_config.json`` as a v2 record holding the
   ``(probe_file, model_id, depths, n_probes)`` tuple. This record IS
   the resolver: every consumer module asks for it and reads back an
   ``ActiveProbeSet`` object that knows exactly which cache file to
   load. No filename heuristics, no directory scans.

Cross-module callers (``engine/modules/{domain_surface,
probe_generator}.py`` and ``app.py``) all import from this module.
These helpers are an explicit public API of the probe subsystem;
they do not carry leading underscores.

Public surface
--------------
Constants:
    META_TAG, FIXED_COLS, NORM_EPS, PROBE_CACHE_DIR, PROBE_CONFIG

CSV parsing:
    parse_meta(csv_path) -> dict
    detect_level_cols(csv_path) -> (level_cols, level_names)
    load_probes(csv_path) -> list[dict]

Embedding cache (path-addressed JSON):
    probe_cache_path(project_root, probe_file, model_id, layer_frac, projected) -> str
    load_probe_cache(cache_path) -> dict | None
    embed_and_cache_probes(model, tokenizer, adapter, project_root, probe_file,
                           model_id, layer_frac, progress, delta_matrix) -> str | None

Active-set state (single source of truth):
    ActiveProbeSet                                              # dataclass
    get_active_probe_set(project_root) -> ActiveProbeSet | None
    get_active_probe(project_root) -> str | None                # legacy: filename only
    set_active_probe(project_root, probe_file, model_id, depths, n_probes) -> None
"""
from __future__ import annotations

import csv
import re
import datetime as _dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import numpy as np

logger = logging.getLogger("src")


# ─── Constants ──────────────────────────────────────────────────

# Probe CSV structure
FIXED_COLS = {"subject", "anchor_id"}  # Non-escalation columns in probe CSVs
META_TAG = "_meta"                      # Reserved first-column value for metadata rows

# Numerical stability threshold for norm checks
NORM_EPS = 1e-12

# Legacy defaults — used only if CSV has no header or auto-detection fails.
_DEFAULT_LEVEL_COLS = ["nouns", "phrase", "question", "instruction", "meta_instruction"]
_DEFAULT_LEVEL_NAMES = ["nouns", "phrase", "question", "instruct", "meta"]

# Embedding cache + active-set pointer locations (relative to project root)
PROBE_CACHE_DIR = "probe_cache"
PROBE_CONFIG = "probe_config.json"


# ─── Probe Metadata ──────────────────────────────────────────

def parse_meta(csv_path):
    """Extract metadata from _meta rows in a probe CSV.

    Meta rows have '_meta' as their first column value. The remaining
    columns are key-value pairs read positionally from the header.

    Template convention for layer depths:
      - Position 1 (anchor_id column): subject/angular probe depth (layer_low)
      - Position 2 (first subclass column): escalation/radial probe depth (layer_high)
      - Position 3+: version, description, etc.

    After parsing by column name, normalizes into canonical keys
    'layer_low' and 'layer_high' if not already present.

    Returns a dict of metadata values. Returns empty dict if no meta rows.
    """
    meta = {}
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return meta
            for row in reader:
                if not row or row[0].strip() != META_TAG:
                    continue
                for i, val in enumerate(row[1:], 1):
                    if i < len(header):
                        key = header[i].strip()
                        val = val.strip()
                        if val:
                            # Try numeric conversion
                            try:
                                meta[key] = float(val)
                            except ValueError:
                                meta[key] = val
    except Exception:
        pass

    # ── Normalize positional depth values into canonical keys ──
    # Template convention: anchor_id column holds layer_low,
    # first subclass column holds layer_high.
    if "layer_low" not in meta and "anchor_id" in meta:
        v = meta["anchor_id"]
        if isinstance(v, (int, float)) and 0.0 <= v <= 1.0:
            meta["layer_low"] = v

    if "layer_high" not in meta:
        # First numeric value in [0,1] from a non-anchor_id column
        # that isn't already claimed as layer_low
        for key, val in meta.items():
            if key in ("anchor_id", "layer_low"):
                continue
            if isinstance(val, (int, float)) and 0.0 <= val <= 1.0:
                meta["layer_high"] = val
                break

    return meta


# ─── Probe Loading ────────────────────────────────────────────

def _is_flat_probe_csv(csv_path):
    """Check whether a probe CSV uses the flat (subject, subclass, text) format."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = {h.strip().lower() for h in (reader.fieldnames or [])}
    return "subject" in headers and "subclass" in headers and "text" in headers


def _flat_subclass_order(csv_path):
    """Return the unique subclass values from a flat probe CSV, in first-seen order."""
    order = []
    seen = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row.get("subclass", "").strip()
            if sub and sub != META_TAG and sub not in seen:
                order.append(sub)
                seen.add(sub)
    return order


def detect_level_cols(csv_path):
    """Read the CSV header and return escalation columns (everything after subject/anchor_id).

    Supports both formats:
      - Flat:  subject, subclass, text  →  level_cols are unique subclass values from data
      - Wide:  subject, anchor_id, col1, col2, ...  →  level_cols are header columns

    Returns (level_cols, level_names) where level_names are display-friendly versions.
    Returns ([], []) if the CSV is not a valid probe file (no 'subject' column).
    Skips _meta rows.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

    # A valid probe CSV must have a 'subject' column
    header_lower = {h.strip().lower() for h in headers}
    if "subject" not in header_lower:
        return [], []

    # ── Flat format (subject, subclass, text) ──
    if _is_flat_probe_csv(csv_path):
        level_cols = _flat_subclass_order(csv_path)
        if not level_cols:
            return _DEFAULT_LEVEL_COLS[:], _DEFAULT_LEVEL_NAMES[:]
        level_names = [col.replace("_", " ").strip() for col in level_cols]
        return level_cols, level_names

    # ── Wide format (subject, anchor_id, col1, col2, ...) ──
    level_cols = [h for h in headers if h.strip().lower() not in {c.lower() for c in FIXED_COLS}]

    if not level_cols:
        return _DEFAULT_LEVEL_COLS[:], _DEFAULT_LEVEL_NAMES[:]

    # Generate display names: replace underscores with spaces
    level_names = [col.replace("_", " ").strip() for col in level_cols]

    return level_cols, level_names


def load_probes(csv_path):
    """Load probes from CSV. Auto-detects format.

    Supports both formats:
      - Flat:  subject, subclass, text  →  one probe per row
      - Wide:  subject, anchor_id, col1, col2, ...  →  one probe per non-empty cell

    Returns list of dicts with subject, anchor_id, level, text.
    Returns empty list if the CSV is not a valid probe file.
    Skips _meta rows.
    """
    # ── Flat format ──
    if _is_flat_probe_csv(csv_path):
        subclass_order = _flat_subclass_order(csv_path)
        sub_to_level = {s: i for i, s in enumerate(subclass_order)}
        probes = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row.get("subject", "").strip()
                if not subj or subj == META_TAG:
                    continue
                sub = row.get("subclass", "").strip()
                text = row.get("text", "").strip()
                if sub and text:
                    probes.append({
                        "subject": subj,
                        "anchor_id": row.get("anchor_id", "").strip(),
                        "level": sub_to_level.get(sub, 0),
                        "text": text,
                    })
        return probes

    # ── Wide format (legacy) ──
    level_cols, _ = detect_level_cols(csv_path)
    if not level_cols:
        return []

    probes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "subject" not in row:
                continue
            if row["subject"].strip() == META_TAG:
                continue
            for level, col in enumerate(level_cols):
                text = row.get(col, "").strip()
                if text:
                    probes.append({
                        "subject": row["subject"].strip(),
                        "anchor_id": row.get("anchor_id", "").strip(),
                        "level": level,
                        "text": text,
                    })
    return probes


# ─── Probe Embedding Cache ────────────────────────────────────

def probe_csv_fingerprint(project_root, probe_file) -> str:
    """Short content hash of a probe CSV.

    The cache key used to be the FILENAME STEM only, so editing a probe CSV in
    place silently reused the embeddings of the previous contents.  Hashing the
    bytes makes an edit produce a different cache file.
    """
    path = os.path.join(project_root, probe_file)
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        # Unreadable: fall back to a marker that never matches a real hash, so
        # we recompute rather than reuse something unrelated.
        return "nofile"


def probe_cache_path(project_root, probe_file, model_id, layer_frac=0.50,
                     projected=False):
    """Build the cache file path for pre-computed probe embeddings.

    The key covers everything that changes the embeddings: CSV contents, model,
    capture depth, projection, and the two tokenization/pooling flags.
    """
    # Imported lazily, matching the rest of this module: src.engine.config
    # pulls in engine state that must not be imported at probe-IO import time.
    from src.engine import config as engine_config

    cache_dir = os.path.join(project_root, PROBE_CACHE_DIR)
    safe_model = model_id.replace("/", "__").replace("\\", "__")
    stem = os.path.splitext(os.path.basename(probe_file))[0]
    layer_tag = str(int(layer_frac * 100))
    proj_tag = "_proj" if projected else ""
    content_tag = probe_csv_fingerprint(project_root, probe_file)
    tok_tag = "{}{}".format(
        int(bool(engine_config.get("add_special_tokens"))),
        int(bool(engine_config.get("include_first_token"))),
    )
    return os.path.join(
        cache_dir,
        f"{stem}__{safe_model}__L{layer_tag}{proj_tag}__c{content_tag}__t{tok_tag}.json")


def load_probe_cache(cache_path):
    """Load cached probe embeddings. Returns dict or None.

    Validates that embeddings don't contain NaN (can happen if cache
    was generated with a dtype that caused overflow, e.g. fp16 on CPU).
    """
    from src.engine import config as engine_config

    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        # Pooling-schema check: caches written before pool_version 2 used
        # default special-token handling and unconditional first-token
        # dropping — inconsistent with prompt analysis. Invalidate so the
        # apply flow regenerates them under the current scheme.
        if data.get("pool_version") != 2:
            logger.warning(f"[DOMAIN] Probe cache uses the pre-v2 pooling "
                           f"scheme — invalidating: {cache_path}")
            os.remove(cache_path)
            return None
        # The two tokenization flags were WRITTEN into the cache but never read
        # back, so flipping either one silently reused embeddings produced
        # under the other scheme.  They are part of the filename now; this is
        # the belt-and-braces check for pre-existing caches.
        want_special = bool(engine_config.get("add_special_tokens"))
        want_first = bool(engine_config.get("include_first_token"))
        if (data.get("add_special_tokens") != want_special
                or data.get("include_first_token") != want_first):
            logger.warning(
                f"[DOMAIN] Probe cache tokenization mismatch "
                f"(cached add_special={data.get('add_special_tokens')}, "
                f"include_first={data.get('include_first_token')}; "
                f"config wants {want_special}/{want_first}) — invalidating: "
                f"{cache_path}")
            os.remove(cache_path)
            return None
        # Validate: check all embeddings for NaN
        embeddings = data.get("embeddings", [])
        for emb in embeddings:
            if any(v is None or (isinstance(v, float) and (v != v)) for v in emb):
                logger.warning(f"[DOMAIN] Probe cache contains NaN — invalidating stale cache: {cache_path}")
                os.remove(cache_path)
                return None
        return data
    except Exception as e:
        logger.warning(f"[DOMAIN] Failed to load probe cache {cache_path}: {e}")
        return None


def embed_and_cache_probes(model, tokenizer, adapter, project_root, probe_file,
                           model_id, layer_frac=0.50, progress=None,
                           delta_matrix=None):
    """Pre-embed all probes from a CSV and save to cache.

    Registers a forward hook via the adapter's hook resolution (model-family
    agnostic), runs each probe text through the model, and stores the mean
    hidden-state embedding.

    Args:
        model: HuggingFace causal LM (instruct model).
        tokenizer: HuggingFace tokenizer.
        adapter: ModelAdapter instance for hook resolution.
        project_root: Project root directory.
        probe_file: Probe CSV filename (relative to project_root).
        model_id: Model identifier string for cache key.
        layer_frac: Capture depth as fraction of model depth (0.0–1.0).
        progress: Optional status callback.
        delta_matrix: Optional o_proj delta tensor for correction-space
            projection. If provided, embeddings are projected through
            h @ delta.T before normalization.

    Returns:
        Path to the saved cache file, or None on failure.
    """
    import torch
    from src.core.locks import MODEL_LOCK
    from src.engine import config as engine_config

    csv_path = os.path.join(project_root, probe_file)
    if not os.path.exists(csv_path):
        logger.warning(f"[DOMAIN] Probe file not found: {csv_path}")
        return None

    probes = load_probes(csv_path)
    if not probes:
        logger.warning(f"[DOMAIN] No probes loaded from {csv_path}")
        return None

    n_layers = adapter.n_layers(model)
    target_layer = max(0, min(n_layers - 1, int(layer_frac * n_layers)))

    # Tokenization + pooling follow the SAME invariants as prompt analysis
    # (see INVARIANTS.md): the configured add_special_tokens flag, and the
    # include_first_token setting for pooling. Previously this path
    # tokenized with specials unconditionally and always dropped position 0
    # — skipping BOS on Llama but discarding the first CONTENT token on
    # Qwen, where many probes are 1–3 token vocabulary terms.
    add_special = engine_config.get("add_special_tokens")
    start_pos = 0 if engine_config.get("include_first_token") else 1

    # Hook via adapter — model-family agnostic
    captured = {}

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    embeddings = []
    # The hook stays installed for the whole loop, so hold the model lock
    # for the duration: a concurrent analysis forward would land in
    # ``captured`` (and our forwards would land in the analyzer's capture).
    with MODEL_LOCK:
        target_module = adapter.resolve_hook_target(
            model, "pre_attn_norm", target_layer)
        handle = target_module.register_forward_hook(hook_fn)
        try:
            for i, probe in enumerate(probes):
                # Clear between probes.  `captured` was created once outside
                # the loop, so if a forward ever failed to fire the hook this
                # probe silently inherited the PREVIOUS probe's embedding.
                captured.clear()
                inputs = tokenizer(probe["text"], return_tensors="pt",
                                   add_special_tokens=add_special)
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    model(**inputs)
                if "h" not in captured:
                    raise RuntimeError(
                        f"probe {i} ({probe.get('text', '')[:40]!r}): forward "
                        f"pass did not fire the capture hook")
                h = captured["h"][0]
                # Promote BEFORE the reduction: averaging in bf16 (8 mantissa
                # bits) and casting afterwards loses precision the .float()
                # cannot recover.
                if h.shape[0] > start_pos:
                    emb = h[start_pos:].float().mean(dim=0)
                else:
                    # Degenerate: the sequence is shorter than start_pos, so
                    # there is nothing to pool.  h[0] is the very token
                    # start_pos exists to skip, so flag it rather than pretend.
                    logger.warning(
                        "[DOMAIN] probe %d is shorter than start_pos=%d; "
                        "falling back to position 0 (the token start_pos "
                        "skips)", i, start_pos)
                    emb = h[0].float()
                if delta_matrix is not None:
                    emb = torch.matmul(emb.cpu(), delta_matrix.float().cpu().T)
                emb = emb.cpu().numpy()
                norm = float(np.linalg.norm(emb))
                if norm > NORM_EPS:
                    emb = emb / norm
                embeddings.append(emb.tolist())
                if progress and (i + 1) % 10 == 0:
                    progress("probes", f"Embedding probes: {i+1}/{len(probes)}")
        finally:
            handle.remove()

    # Save cache
    cache_path = probe_cache_path(project_root, probe_file, model_id, layer_frac,
                                  projected=delta_matrix is not None)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache_data = {
        "model_id": model_id,
        "layer": target_layer,
        "layer_frac": layer_frac,
        "n_layers": n_layers,
        "probe_file": probe_file,
        "n_probes": len(probes),
        # Pooling/tokenization schema. v2 = configured add_special_tokens +
        # include_first_token-aware pooling. Caches without this stamp were
        # produced under the old (inconsistent) scheme and are invalidated
        # on load.
        "pool_version": 2,
        "add_special_tokens": bool(add_special),
        "include_first_token": bool(engine_config.get("include_first_token")),
        "embeddings": embeddings,
    }

    with open(cache_path, "w") as f:
        json.dump(cache_data, f, separators=(',', ':'))

    logger.info(f"[DOMAIN] Cached {len(embeddings)} probe embeddings → {cache_path}")
    return cache_path


# ─── Active-set state ──────────────────────────────────────────
#
# The active probe set is the single binding "this probe lattice belongs to
# this model at these depths." It is persisted in probe_config.json (v2)
# and is the authoritative resolver for every module that needs to load
# probe embeddings: modules ask for ActiveProbeSet, then call .cache_path()
# and .validate_against(pipeline). Filename heuristics and directory scans
# are forbidden — caches are only located via the recorded model_id.
#
# Schema (v2):
#   {
#     "version": 2,
#     "active": {
#       "probe_file":  "<csv filename>",
#       "model_id":    "<HF model id used to embed>",
#       "depths":      [<float>, ...],
#       "n_probes":    <int>,
#       "applied_at":  "<ISO 8601 UTC>"
#     }
#   }
#
# Schema (v1, legacy, read-only):
#   {"active": ["<csv filename>"]}
#
# A v1 record loads as an ActiveProbeSet with model_id=None and depths=(),
# which forces validate_against() to fail with "re-Apply"; this lets users
# upgrading from earlier TAGM see a clear remediation instead of silent
# wrong-cache use.


@dataclass(frozen=True)
class ActiveProbeSet:
    """The currently-applied (probe_file, model_id, depths, projected)
    binding.

    All consumers of probe embeddings must go through this object. It owns
    cache-path resolution (so no module reinvents the filename pattern)
    and validation against the loaded pipeline.

    The ``projected`` flag records whether embeddings were projected
    through the layer's o_proj delta at apply time (controlled by the
    engine's ``probe_projection_space`` setting). It's part of the cache
    identity: a projected and an unprojected cache are different artifacts
    even for the same (probe_file, model_id, depth) triple.
    """
    probe_file: str
    model_id: Optional[str]                # None for legacy v1 records
    depths: tuple = ()                     # tuple of layer fracs (e.g. (0.5, 0.75))
    n_probes: int = 0
    projected: bool = False
    applied_at: str = ""

    def is_legacy(self) -> bool:
        """True if this record came from a v1 probe_config.json that
        didn't store the model_id. Such records cannot resolve a cache
        unambiguously and require a re-Apply."""
        return not self.model_id

    def subject_layer_frac(self) -> float:
        """Lowest recorded depth (the "subject" / L_low embedding).
        Falls back to 0.50 if no depths are recorded."""
        return float(min(self.depths)) if self.depths else 0.50

    def escalation_layer_frac(self) -> float:
        """Highest recorded depth (the "escalation" / L_high embedding).
        Falls back to 0.75 if no depths are recorded."""
        return float(max(self.depths)) if self.depths else 0.75

    def cache_path(self, project_root: str, layer_frac: float,
                   projected: Optional[bool] = None) -> str:
        """Exact cache file path for this (probe, model, depth, projected)
        tuple. Uses the recorded ``projected`` flag from the active record
        unless the caller passes an explicit override. Raises if the active
        set is legacy (no model_id known)."""
        if self.is_legacy():
            raise RuntimeError(
                "Active probe set has no recorded model_id "
                "(legacy probe_config.json). Apply the probe set "
                "in the Configuration → Probe Set panel.")
        eff_proj = self.projected if projected is None else bool(projected)
        return probe_cache_path(project_root, self.probe_file,
                                 self.model_id, layer_frac, eff_proj)

    def validate_against(self, pipeline) -> tuple:
        """Confirm this active set is compatible with the loaded pipeline.

        Returns (ok: bool, message: str). Modules should call this BEFORE
        any cache lookup and surface the message verbatim — it tells the
        user exactly what to do."""
        if pipeline is None or not getattr(pipeline, "loaded", False):
            return False, "No model loaded."
        if self.is_legacy():
            return False, (
                f"Active probe set {self.probe_file!r} was applied under "
                f"an older TAGM version that did not record the model. "
                f"Apply it in Configuration → Probe Set so this run "
                f"can resolve the correct cache.")
        cur = getattr(pipeline, "instruct_model_id", None)
        if cur and cur != self.model_id:
            return False, (
                f"Active probe set {self.probe_file!r} was applied for "
                f"{self.model_id!r}, but {cur!r} is currently loaded. "
                f"Apply the probe set in Configuration → Probe Set "
                f"to embed it against the current model.")
        return True, "OK"

    def to_status_dict(self) -> dict:
        """Public-facing dict for the /api/probe_set/status endpoint."""
        return {
            "probe_file": self.probe_file,
            "model_id": self.model_id,
            "depths": list(self.depths),
            "n_probes": self.n_probes,
            "projected": self.projected,
            "applied_at": self.applied_at,
            "legacy": self.is_legacy(),
        }


def get_active_probe_set(project_root) -> Optional[ActiveProbeSet]:
    """Read the active probe set from probe_config.json.

    Returns an ActiveProbeSet for v2 records (full binding) or v1 records
    (filename-only, marked legacy via model_id=None). Returns None if no
    record exists or the file is malformed.
    """
    config_path = os.path.join(project_root, PROBE_CONFIG)
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"[PROBES] Failed to read {PROBE_CONFIG}: {e}")
        return None

    active = data.get("active")
    if not active:
        return None

    # v2 — active is a dict with the full binding.
    if isinstance(active, dict):
        probe_file = active.get("probe_file") or active.get("filename")
        if not probe_file:
            return None
        depths_raw = active.get("depths") or []
        try:
            depths = tuple(float(d) for d in depths_raw)
        except (TypeError, ValueError):
            depths = ()
        return ActiveProbeSet(
            probe_file=str(probe_file),
            model_id=active.get("model_id") or None,
            depths=depths,
            n_probes=int(active.get("n_probes") or 0),
            projected=bool(active.get("projected", False)),
            applied_at=str(active.get("applied_at") or ""),
        )

    # v1 — active is a list of filenames; only the first is honored. Treat
    # as legacy: no model_id recorded → forces re-Apply.
    if isinstance(active, list) and active:
        return ActiveProbeSet(probe_file=str(active[0]), model_id=None)

    return None


def get_active_probe(project_root) -> Optional[str]:
    """Return just the active probe CSV filename, or None.

    Compatibility shim for callers that only need the filename (e.g. the
    /api/data/export endpoint) and do not consume embeddings. New code
    should prefer ``get_active_probe_set`` to obtain the full binding.
    """
    aps = get_active_probe_set(project_root)
    return aps.probe_file if aps else None


def set_active_probe(project_root, probe_file, model_id=None,
                     depths=None, n_probes=0, projected=False):
    """Write the active probe set as a v2 record.

    All consumers (``embed_and_activate_probe_set`` is the only intended
    caller in production) must supply at minimum ``probe_file`` and
    ``model_id``; depths, n_probes, and projected round out the binding for
    module-side cache lookup. ``model_id=None`` is permitted only to
    satisfy unusual test scenarios — it produces a record that
    ``validate_against`` will reject.
    """
    config_path = os.path.join(project_root, PROBE_CONFIG)
    record = {
        "probe_file": probe_file,
        "model_id": model_id,
        "depths": [float(d) for d in (depths or [])],
        "n_probes": int(n_probes),
        "projected": bool(projected),
        "applied_at": _dt.datetime.now(_dt.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(config_path, "w") as f:
        json.dump({"version": 2, "active": record}, f, indent=2)


# ─── High-level: embed + activate ──────────────────────────────

def embed_and_activate_probe_set(pipeline, project_root, filename, progress=None):
    """Embed a probe CSV at both depths and activate it.

    Single entry point for "make this probe file the active one." Used by
    /api/probe_set/apply (Configuration tab) and /api/modules/probe_generator/
    embed_active (Probe Generator card). Caller supplies a loaded Pipeline.

    Reads layer depths from CSV meta (layer_low/layer_high keys) if present,
    otherwise from engine config (domain_embedding_layer_frac / domain_
    escalation_layer_frac), with hard fallbacks of 0.50 / 0.75. Honors the
    probe_projection_space engine-config flag for o_delta projection.

    Returns a dict on success:
        {applied, filename, n_probes, n_subjects, n_levels, levels, depths}

    Returns {"applied": False, "error": "..."} on any failure. Never raises.
    """
    if pipeline is None or not getattr(pipeline, "loaded", False):
        return {"applied": False, "error": "No model loaded"}

    csv_path = os.path.join(project_root, filename)
    if not os.path.exists(csv_path):
        return {"applied": False, "error": f"Probe file not found: {filename}"}

    model = pipeline.instruct_model
    tokenizer = pipeline.tokenizer
    adapter = pipeline.adapter
    model_id = pipeline.instruct_model_id

    if progress:
        progress("Embedding probe set...")

    # Resolve depths: CSV meta overrides engine config overrides hard defaults
    meta = parse_meta(csv_path)
    try:
        from src.engine import config as engine_config
        use_proj = engine_config.get("probe_projection_space")
    except Exception:
        use_proj = False
        engine_config = None

    if "layer_low" in meta and "layer_high" in meta:
        subj_frac = max(0.0, min(1.0, float(meta["layer_low"])))
        esc_frac = max(0.0, min(1.0, float(meta["layer_high"])))
    elif engine_config is not None:
        try:
            subj_frac = max(0.0, min(1.0, float(engine_config.get(
                "domain_embedding_layer_frac") or 0.50)))
            esc_frac = max(0.0, min(1.0, float(engine_config.get(
                "domain_escalation_layer_frac") or 0.75)))
        except Exception:
            subj_frac, esc_frac = 0.50, 0.75
    else:
        subj_frac, esc_frac = 0.50, 0.75

    depths = sorted(set([subj_frac, esc_frac]))
    n_layers = adapter.n_layers(model)
    embedded = 0

    for frac in depths:
        if progress:
            progress(f"Embedding at L{int(frac * 100)}...")

        delta = None
        if use_proj:
            target_layer = max(0, min(n_layers - 1, int(frac * n_layers)))
            delta = pipeline.delta_store.o_delta_or_none(target_layer)

        try:
            embed_and_cache_probes(
                model, tokenizer, adapter,
                project_root, filename, model_id,
                layer_frac=frac,
                progress=(lambda stage, msg, _f=frac: progress(
                    f"L{int(_f * 100)}: {msg}") if progress else None),
                delta_matrix=delta if use_proj else None)
            embedded += 1
        except Exception as e:
            logger.warning(f"[PROBES] Embed failed at L{int(frac * 100)}: {e}")

    if embedded == 0:
        return {"applied": False, "error": "Failed to embed probes at any depth"}

    probes = load_probes(csv_path)
    level_cols, level_names = detect_level_cols(csv_path)
    subjects = sorted(set(p["subject"] for p in probes))

    try:
        set_active_probe(project_root, filename,
                         model_id=model_id, depths=depths,
                         n_probes=len(probes),
                         projected=bool(use_proj))
    except Exception as e:
        return {"applied": False,
                "error": f"Embedded but failed to activate: {e}"}

    if progress:
        progress(f"Complete: {filename} embedded at {len(depths)} "
                 f"depth(s) and activated")

    return {
        "applied": True,
        "filename": filename,
        "n_probes": len(probes),
        "n_subjects": len(subjects),
        "n_levels": len(level_cols),
        "levels": level_names,
        "depths": [int(f * 100) for f in depths],
        "model_id": model_id,
    }


# ═══════════════════════════════════════════════════════════════
# Template store — one directory, one resolver
# ═══════════════════════════════════════════════════════════════
# Historically templates were referenced against three inconsistent
# locations (repo root, root/templates, src/templates). Everything
# below funnels through resolve_data_file / templates_dir so a bare
# name means one thing everywhere.

_TEMPLATE_DIR_CANDIDATES = ("src/templates", "templates")


def templates_dir(project_root):
    """The canonical template directory (first existing candidate;
    src/templates is created if none exist — that is where the
    shipped templates live)."""
    for cand in _TEMPLATE_DIR_CANDIDATES:
        d = os.path.join(project_root, cand)
        if os.path.isdir(d):
            return d
    d = os.path.join(project_root, _TEMPLATE_DIR_CANDIDATES[0])
    os.makedirs(d, exist_ok=True)
    return d


def resolve_data_file(project_root, name):
    """Resolve a template/stopword reference to an absolute path.

    Order: absolute as-is; then relative to project root; then the
    bare name inside each template directory candidate. Returns the
    first existing path, else the root join (so callers' not-found
    errors report a sensible location).
    """
    if not name:
        return name
    if os.path.isabs(name):
        return name
    root_join = os.path.join(project_root, name)
    if os.path.exists(root_join):
        return root_join
    base = os.path.basename(name)
    for cand in _TEMPLATE_DIR_CANDIDATES:
        p = os.path.join(project_root, cand, base)
        if os.path.exists(p):
            return p
    return root_join


def _safe_template_name(name, suffix=".csv"):
    base = os.path.basename(str(name or "")).strip()
    base = re.sub(r"[^A-Za-z0-9._ \-]", "_", base)
    if not base or base.startswith("."):
        raise ValueError("Invalid template name.")
    if not base.endswith(suffix):
        raise ValueError(f"Template name must end in {suffix}")
    return base


def list_templates(project_root):
    """List CSV templates in the store with a light structural summary."""
    d = templates_dir(project_root)
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(d, fn)
        n_classes = 0
        n_cols = 0
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                lower = {h.strip().lower() for h in headers}
                if "subject" not in lower:
                    continue
                cols = [h for h in headers
                        if h.strip().lower() not in ("subject", "anchor_id")]
                n_cols = len(cols)
                seen = set()
                for row in reader:
                    v = (row.get("subject") or "").strip()
                    if v and v != "_meta":
                        seen.add(v)
                n_classes = len(seen)
        except Exception:
            continue
        rel = os.path.relpath(path, project_root)
        entry = {"name": fn, "path": rel,
                 "n_classes": n_classes, "n_columns": n_cols,
                 "has_axes": os.path.exists(path + ".axes.json")}
        out.append(entry)
    return out


def save_template(project_root, name, csv_text, axes=None):
    """Write a template (and optional axes sidecar) into the store.

    Returns the project-root-relative path the module machinery can
    resolve.
    """
    base = _safe_template_name(name)
    d = templates_dir(project_root)
    path = os.path.join(d, base)
    with open(path, "w", newline="") as f:
        f.write(csv_text)
    sidecar = path + ".axes.json"
    if axes is not None:
        with open(sidecar, "w") as f:
            json.dump(axes, f, indent=1)
    elif os.path.exists(sidecar):
        os.remove(sidecar)   # stale sidecar must not describe a new file
    return os.path.relpath(path, project_root)


def load_template_raw(project_root, name):
    """Return (csv_text, axes_or_None, resolved_path) for a template."""
    path = resolve_data_file(project_root, name)
    if not os.path.exists(path):
        raise FileNotFoundError(name)
    with open(path) as f:
        csv_text = f.read()
    axes = None
    sidecar = path + ".axes.json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar) as f:
                axes = json.load(f)
        except Exception:
            axes = None
    return csv_text, axes, path


def parse_template_echo(project_root, name):
    """What the machinery will see: run the PRODUCTION parsers over a
    template and return their interpretation. This is the maker's
    no-erroneous-parsing guarantee — the same code paths the generator
    executes at run time, not a validator that imitates them.
    """
    path = resolve_data_file(project_root, name)
    from src.engine.modules.probe_generator import _parse_template
    classes, subclass_cols, cell_seeds = _parse_template(path)
    level_cols, level_names = detect_level_cols(path)
    meta = parse_meta(path)
    return {
        "classes": classes,
        "subclass_cols": subclass_cols,
        "level_names": level_names,
        "cells": [[cls, col, cell_seeds.get((cls, col), "")]
                  for cls in classes for col in subclass_cols],
        "n_cells": len(classes) * len(subclass_cols),
        "layer_low": meta.get("layer_low"),
        "layer_high": meta.get("layer_high"),
    }
