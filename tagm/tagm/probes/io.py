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
   ``<project_root>/probe_config.json`` (``{"active": [filename]}``).

Cross-module callers (``engine/modules/{domain_surface,
correction_heatmap, correction_manifold, correction_backscatter,
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

Active-set pointer:
    get_active_probe(project_root) -> str | None
"""
from __future__ import annotations

import csv
import json
import logging
import os
from glob import glob

import numpy as np

logger = logging.getLogger("tagm")


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

def probe_cache_path(project_root, probe_file, model_id, layer_frac=0.50,
                     projected=False):
    """Build the cache file path for pre-computed probe embeddings."""
    cache_dir = os.path.join(project_root, PROBE_CACHE_DIR)
    safe_model = model_id.replace("/", "__").replace("\\", "__")
    stem = os.path.splitext(probe_file)[0]
    layer_tag = str(int(layer_frac * 100))
    proj_tag = "_proj" if projected else ""
    return os.path.join(cache_dir, f"{stem}__{safe_model}__L{layer_tag}{proj_tag}.json")


def load_probe_cache(cache_path):
    """Load cached probe embeddings. Returns dict or None.

    Validates that embeddings don't contain NaN (can happen if cache
    was generated with a dtype that caused overflow, e.g. fp16 on CPU).
    """
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
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

    # Hook via adapter — model-family agnostic
    captured = {}

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    target_module = adapter.resolve_hook_target(model, "pre_attn_norm", target_layer)
    handle = target_module.register_forward_hook(hook_fn)

    embeddings = []
    try:
        for i, probe in enumerate(probes):
            inputs = tokenizer(probe["text"], return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            h = captured["h"][0]
            emb = h[1:].mean(dim=0).float() if h.shape[0] > 1 else h[0].float()
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
        "embeddings": embeddings,
    }

    with open(cache_path, "w") as f:
        json.dump(cache_data, f, separators=(',', ':'))

    logger.info(f"[DOMAIN] Cached {len(embeddings)} probe embeddings → {cache_path}")
    return cache_path


# ─── Active-set pointer ────────────────────────────────────────

def get_active_probe(project_root):
    """Read the active probe file from probe_config.json."""
    config_path = os.path.join(project_root, PROBE_CONFIG)
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                data = json.load(f)
            active = data.get("active", [])
            if active:
                return active[0]
        except Exception:
            pass
    return None
