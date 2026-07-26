"""Probe-set lattice diagnostics.

Extracted verbatim from the ``/api/probe_diagnostic`` endpoint body in
``src/app.py``; the endpoint is now a thin wrapper around
:func:`compute_diagnostic`. Reads from disk (active probe set or a named
file) and is independent of any module's last-run output. Returns
lattice properties: cell coverage, sample terms per cell, cross-class /
cross-level collisions. Embedding-tier metrics are added when a probe
cache exists for the loaded model.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Optional

from src.api._util import resolve_under


def compute_diagnostic(project_root, filename: Optional[str],
                       pipeline) -> dict:
    """Compute lattice properties of a probe set on disk.

    ``filename``: CSV name relative to ``project_root``. If falsy, the
        active probe set from probe_config.json is used.
    ``pipeline``: the loaded Pipeline, or None. Only used for the
        optional embedding tier.
    """
    from src.probes.io import (
        get_active_probe, get_active_probe_set, load_probes,
        detect_level_cols, parse_meta, probe_cache_path, load_probe_cache)

    project_root = str(project_root)

    filename = filename or get_active_probe(project_root)
    if not filename:
        return {"ok": False, "error": "No active probe set."}

    # Request-supplied ?file= — resolve and confirm it stays under the
    # project root. Relative subpaths ("probe_cache/x.csv") are legitimate
    # here, so a flat sanitizer would break the endpoint; an unchecked
    # os.path.join let "../../etc/passwd" be parsed and echoed back.
    resolved = resolve_under(project_root, filename)
    if resolved is None:
        return {"ok": False, "error": f"Invalid probe file path: {filename}"}
    csv_path = str(resolved)
    if not os.path.exists(csv_path):
        return {"ok": False, "error": f"Probe file not found: {filename}"}

    probes = load_probes(csv_path)
    if not probes:
        return {"ok": False, "error": "No probes loaded from CSV."}

    level_cols, level_names = detect_level_cols(csv_path)
    meta = parse_meta(csv_path)

    subjects = sorted(set(p["subject"] for p in probes))
    n_levels = len(level_names) if level_names else max(
        (p["level"] for p in probes), default=-1) + 1

    # ── Cell coverage: (subject, level) → list of probe texts ──
    cells = defaultdict(list)
    for p in probes:
        cells[(p["subject"], p["level"])].append(p["text"])

    cell_grid = []
    for s in subjects:
        row = []
        for l in range(n_levels):
            terms = cells.get((s, l), [])
            row.append({"count": len(terms), "sample": terms[:8]})
        cell_grid.append(row)

    counts_flat = [c["count"] for row in cell_grid for c in row]
    n_populated = sum(1 for c in counts_flat if c > 0)
    n_empty = sum(1 for c in counts_flat if c == 0)

    # ── Cross-class collisions: term appears in multiple subjects ──
    term_subjects = defaultdict(set)
    term_levels_per_subject = defaultdict(lambda: defaultdict(set))
    for p in probes:
        term_subjects[p["text"]].add(p["subject"])
        term_levels_per_subject[p["subject"]][p["text"]].add(p["level"])

    # ── In-cell duplicates: same term twice in the same cell ──
    # Invisible to both collision checks below, but inflates counts and
    # drags intra-cell spread toward zero (a duplicate's pairwise
    # similarity with itself is 1.0).
    _seen = Counter()
    for p in probes:
        _seen[(p["subject"], p["level"], p["text"])] += 1
    in_cell_duplicates = []
    for (s_, l_, t_), n_ in _seen.items():
        if n_ > 1:
            in_cell_duplicates.append({
                "term": t_, "subject": s_, "level": l_,
                "level_name": (level_names[l_] if l_ < len(level_names)
                               else str(l_)),
                "count": n_,
            })
    in_cell_duplicates.sort(key=lambda r: (-r["count"], r["subject"], r["term"]))

    cross_class = []
    for term, subjs in term_subjects.items():
        if len(subjs) > 1:
            cross_class.append({
                "term": term,
                "subjects": sorted(subjs),
            })
    cross_class.sort(key=lambda r: (-len(r["subjects"]), r["term"]))

    # ── Cross-level collisions: term appears in multiple levels of same subject ──
    cross_level = []
    for s, term_lvls in term_levels_per_subject.items():
        for term, lvls in term_lvls.items():
            if len(lvls) > 1:
                cross_level.append({
                    "term": term,
                    "subject": s,
                    "levels": sorted(lvls),
                    "level_names": [level_names[l] for l in sorted(lvls)
                                    if l < len(level_names)],
                })
    cross_level.sort(key=lambda r: (-len(r["levels"]), r["subject"], r["term"]))

    # ── Embedding tier (best-effort): load cache for active model if present ──
    embedding_tier = None
    if pipeline is not None and pipeline.loaded:
        model_id = pipeline.instruct_model_id

        # When diagnosing the ACTIVE set, the probe_config record knows
        # the exact depths it was embedded at — use the resolver rather
        # than guessing, so config drift after apply can't point us at
        # a missing or stale cache. The guess chain (CSV meta → engine
        # config → 0.50) remains only for arbitrary ?file= sets that
        # were never applied.
        active = get_active_probe_set(project_root)
        if active is not None and active.probe_file == filename:
            frac = active.subject_layer_frac()
            cache_path = active.cache_path(project_root, frac)
        else:
            if "layer_low" in meta:
                try:
                    frac = max(0.0, min(1.0, float(meta["layer_low"])))
                except Exception:
                    frac = 0.50
            else:
                try:
                    from src.engine import config as engine_config
                    frac = max(0.0, min(1.0, float(engine_config.get(
                        "domain_embedding_layer_frac") or 0.50)))
                except Exception:
                    frac = 0.50
            cache_path = probe_cache_path(project_root, filename, model_id,
                                          frac, projected=False)
        cache = load_probe_cache(cache_path)
        if cache and cache.get("embeddings"):
            import numpy as np
            embs = np.array(cache["embeddings"], dtype=np.float32)
            # Index alignment: cache embeddings parallel the load_probes() order
            if len(embs) == len(probes):
                # Group embeddings by cell
                cell_embs = defaultdict(list)
                for i, p in enumerate(probes):
                    cell_embs[(p["subject"], p["level"])].append(embs[i])

                # Intra-cell cosine spread: 1 - mean pairwise cosine similarity
                # within each cell
                intra_grid = []
                centroids = {}
                for s in subjects:
                    row = []
                    for l in range(n_levels):
                        vecs = cell_embs.get((s, l), [])
                        if len(vecs) >= 2:
                            M = np.stack(vecs)
                            sims = M @ M.T
                            n = sims.shape[0]
                            mask = ~np.eye(n, dtype=bool)
                            mean_sim = float(sims[mask].mean())
                            spread = 1.0 - mean_sim
                            cent = M.mean(axis=0)
                            cent_norm = np.linalg.norm(cent)
                            if cent_norm > 1e-12:
                                centroids[(s, l)] = cent / cent_norm
                            row.append(round(spread, 4))
                        elif len(vecs) == 1:
                            centroids[(s, l)] = vecs[0]
                            row.append(None)
                        else:
                            row.append(None)
                    intra_grid.append(row)

                # Inter-cell separation: mean cosine distance between centroids
                cell_keys = list(centroids.keys())
                if len(cell_keys) >= 2:
                    C = np.stack([centroids[k] for k in cell_keys])
                    cs = C @ C.T
                    n = cs.shape[0]
                    mask = ~np.eye(n, dtype=bool)
                    inter_mean = 1.0 - float(cs[mask].mean())
                    inter_min = 1.0 - float(cs[mask].max())  # tightest pair
                    # Name the offending pair — the number alone isn't
                    # actionable.
                    _masked = np.where(mask, cs, -np.inf)
                    _i, _j = np.unravel_index(int(np.argmax(_masked)),
                                              _masked.shape)

                    def _cell_label(k):
                        s_, l_ = k
                        ln = (level_names[l_] if l_ < len(level_names)
                              else str(l_))
                        return f"{s_} × {ln}"
                    tightest_pair = [_cell_label(cell_keys[_i]),
                                     _cell_label(cell_keys[_j])]
                else:
                    inter_mean = None
                    inter_min = None
                    tightest_pair = None

                embedding_tier = {
                    "model_id": model_id,
                    "layer_frac": frac,
                    "intra_cell_spread": intra_grid,
                    "inter_cell_mean_distance": (
                        round(inter_mean, 4) if inter_mean is not None else None),
                    "inter_cell_min_distance": (
                        round(inter_min, 4) if inter_min is not None else None),
                    "tightest_pair": tightest_pair,
                    "n_cells_with_centroid": len(cell_keys),
                }

    return {
        "ok": True,
        "filename": filename,
        "n_probes": len(probes),
        "n_subjects": len(subjects),
        "n_levels": n_levels,
        "subjects": subjects,
        "level_names": level_names,
        "cell_grid": cell_grid,
        "summary": {
            "populated_cells": n_populated,
            "empty_cells": n_empty,
            "min_count": min(counts_flat) if counts_flat else 0,
            "max_count": max(counts_flat) if counts_flat else 0,
            "mean_count": (round(sum(counts_flat) / len(counts_flat), 1)
                           if counts_flat else 0),
        },
        "cross_class_collisions": cross_class,
        "cross_level_collisions": cross_level,
        "in_cell_duplicates": in_cell_duplicates,
        "embedding_tier": embedding_tier,
    }
