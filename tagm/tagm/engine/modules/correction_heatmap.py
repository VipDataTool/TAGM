"""
Correction Heatmap Module for TASM.

Measures how each prompt's tokens interact with the correction field
by projecting them through probe refinement deltas.

Process:
  1. Load probe embeddings at L50 (subject) and L75 (escalation)
  2. Compute probe deltas: L75 - L50 per probe (the refinement direction)
  3. For each analyzed prompt, take per-token final hidden states
  4. Dot product each token's final state against each probe delta
  5. Aggregate per cell (subject × subclass) into a heatmap

The heatmap shows how strongly each prompt's tokens excite each
region of the probe landscape when viewed through the refinement lens.

Pure post-processor: requires session results with per_token_final_emb
and probe caches at both depths.
"""

import os
import json
import logging
import numpy as np
from collections import defaultdict

from .base import TASMModule, ModuleParameter
from tagm.probes.io import (
    detect_level_cols,
    parse_meta,
    load_probe_cache,
    probe_cache_path,
    load_probes,
    get_active_probe_set,
)

logger = logging.getLogger("tasm")


class CorrectionHeatmapModule(TASMModule):
    name = "correction_heatmap"
    display_name = "Correction Heatmap"
    description = (
        "Projects prompt tokens through probe refinement deltas "
        "(L75 - L50) to produce an aggregate heatmap of correction "
        "field interaction across subject × subclass cells. "
        "Reveals training-data coverage structure, not per-prompt categories."
    )
    version = "0.1.0"

    min_results = 1
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    def __init__(self):
        super().__init__()
        self._project_root = None
        self._pipeline = None

    def set_project_root(self, root):
        self._project_root = root

    def set_pipeline(self, pipeline):
        """Receive the pipeline reference so validate() can confirm the
        active probe set was applied for the currently loaded model."""
        self._pipeline = pipeline

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="projection_method",
                display_name="Projection Method",
                description=(
                    "How to measure interaction intensity. "
                    "abs = linear magnitude. squared = energy. "
                    "signed = raw directional (positive = aligned, negative = opposed)."
                ),
                type="select",
                default="abs",
                options=["abs", "squared", "signed"],
            ),
        ]

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        active = get_active_probe_set(self._project_root)
        if active is None:
            return False, (
                "No probe set active. Apply a probe set in the "
                "Configuration tab first."
            )
        ok, msg = active.validate_against(self._pipeline)
        if not ok:
            return False, msg

        has_final = any(r.get("per_token_final_emb") for r in session_results)
        if not has_final:
            return False, (
                "No final-layer token embeddings found in session results. "
                "Re-analyze prompts with the current build to capture them."
            )
        return True, "OK"

    def run(self, session_results, params, progress=None):
        active = get_active_probe_set(self._project_root)
        if active is None:
            raise RuntimeError("No probe set active. Apply one in "
                               "Configuration → Probe Set.")
        ok, msg = active.validate_against(self._pipeline)
        if not ok:
            raise RuntimeError(msg)
        probe_file = active.probe_file
        proj_method = params.get("projection_method", "abs")

        if progress:
            progress("Loading probe structure...")

        # ── Load probe CSV for subject × subclass structure ──
        csv_path = os.path.join(self._project_root, probe_file)
        level_cols, level_names = detect_level_cols(csv_path)
        if not level_cols:
            raise RuntimeError(f"No level columns found in {probe_file}")

        raw_probes = load_probes(csv_path)
        if not raw_probes:
            raise RuntimeError(f"No probes loaded from {probe_file}")

        # Build subject list (ordered by first appearance)
        subjects = []
        for p in raw_probes:
            if p["subject"] not in subjects:
                subjects.append(p["subject"])
        n_subj = len(subjects)
        n_levels = len(level_cols)
        subj_idx = {s: i for i, s in enumerate(subjects)}

        # ── Layer depths come from the active probe set ──
        # The depths recorded in probe_config.json are what was actually
        # embedded at apply time; that's the authoritative source. CSV
        # meta and engine config are used at apply time, not at run time.
        subj_frac = active.subject_layer_frac()
        esc_frac = active.escalation_layer_frac()
        logger.info(f"[HEATMAP] Active depths: "
                    f"L{int(subj_frac*100)}, L{int(esc_frac*100)} "
                    f"(projected={active.projected})")

        if progress:
            progress(f"Loading probe embeddings at L{int(subj_frac*100)} "
                     f"and L{int(esc_frac*100)}...")

        # ── Load both depths via the active-set resolver ──
        # No directory scan; cache_path() composes the exact filename from
        # (probe_file, model_id, depth, projected). If the file isn't
        # there, the active record is stale relative to disk and the user
        # is told to re-Apply.
        def _load_at(frac):
            cache_path = active.cache_path(self._project_root, frac)
            data = load_probe_cache(cache_path)
            if not data or not data.get("embeddings"):
                raise RuntimeError(
                    f"Probe cache not found at {cache_path} "
                    f"(L{int(frac*100)}). Re-Apply the probe set in "
                    f"Configuration → Probe Set to regenerate it.")
            embs = data["embeddings"]
            if len(embs) != len(raw_probes):
                raise RuntimeError(
                    f"Probe count mismatch in {os.path.basename(cache_path)}: "
                    f"cache has {len(embs)}, CSV has {len(raw_probes)}. "
                    f"Re-Apply the probe set to refresh.")
            logger.info(f"[HEATMAP] Using cache: {os.path.basename(cache_path)}")
            return embs

        embs_L50 = _load_at(subj_frac)
        embs_L75 = _load_at(esc_frac)

        # ── Compute probe deltas ──
        if progress:
            progress("Computing probe refinement deltas (L75 - L50)...")

        mat_L50 = np.array(embs_L50, dtype=np.float32)
        mat_L75 = np.array(embs_L75, dtype=np.float32)
        deltas = mat_L75 - mat_L50  # [n_probes, hidden_dim]

        # L2-normalize deltas so dot products are comparable
        delta_norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        delta_norms[delta_norms < 1e-12] = 1.0
        deltas_n = deltas / delta_norms

        # Map each probe to its cell
        probe_cells = []  # [(subj_idx, level_idx)] per probe
        for p in raw_probes:
            si = subj_idx.get(p["subject"], 0)
            li = p["level"]
            probe_cells.append((si, li))

        # ── Project prompt tokens through probe deltas ──
        if progress:
            progress("Projecting prompt tokens through refinement deltas...")

        n_prompts = len(session_results)
        per_prompt_heatmaps = []
        aggregate_heatmap = np.zeros((n_subj, n_levels))
        aggregate_count = 0

        # Per-cell token tracking: {(si,li): {token: {cat: [values]}}}
        cell_token_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Probe texts per cell
        cell_probes = defaultdict(list)
        for p in raw_probes:
            si = subj_idx.get(p["subject"], 0)
            li = p["level"]
            cell_probes[(si, li)].append(p.get("text", p.get("anchor_id", "")))

        for pi, r in enumerate(session_results):
            final_emb = r.get("per_token_final_emb")
            if final_emb is None or len(final_emb) == 0:
                per_prompt_heatmaps.append(None)
                continue

            if progress and (pi + 1) % 10 == 0:
                progress(f"Processing prompt {pi+1}/{n_prompts}...")

            tok_mat = np.array(final_emb, dtype=np.float32)  # [n_tokens, hidden_dim]
            tokens = r.get("tokens", [])
            cat = (r.get("category", "") or "unknown")[:1]  # b/m/h/j shorthand

            # Dot product: each token against each probe delta
            # [n_tokens, hidden_dim] @ [hidden_dim, n_probes] = [n_tokens, n_probes]
            projections = tok_mat @ deltas_n.T

            # Aggregate per cell: mean of absolute projections across
            # all tokens and all probes in that cell
            cell_grid = np.zeros((n_subj, n_levels))
            cell_counts = np.zeros((n_subj, n_levels))

            # Per-token, per-cell activation for context-variance tracking
            n_tok = min(len(tokens), tok_mat.shape[0])
            cell_tok_vals = defaultdict(lambda: np.zeros(n_tok))
            cell_tok_counts = defaultdict(int)

            for probe_idx, (si, li) in enumerate(probe_cells):
                p = projections[:n_tok, probe_idx]
                if proj_method == "squared":
                    cell_grid[si, li] += (p ** 2).mean()
                    cell_tok_vals[(si, li)] += p[:n_tok] ** 2
                elif proj_method == "signed":
                    cell_grid[si, li] += p.mean()
                    cell_tok_vals[(si, li)] += p[:n_tok]
                else:
                    cell_grid[si, li] += np.abs(p).mean()
                    cell_tok_vals[(si, li)] += np.abs(p[:n_tok])
                cell_counts[si, li] += 1
                cell_tok_counts[(si, li)] += 1

            # Normalize by number of probes per cell
            mask = cell_counts > 0
            cell_grid[mask] /= cell_counts[mask]

            # Normalize per-token values and record
            for (si, li), vals in cell_tok_vals.items():
                cnt = cell_tok_counts[(si, li)]
                if cnt > 0:
                    vals = vals / cnt
                for ti in range(min(n_tok, len(tokens))):
                    tok = tokens[ti].strip()
                    if tok:
                        cell_token_data[(si, li)][tok][cat].append(float(vals[ti]))

            per_prompt_heatmaps.append(cell_grid.tolist())
            aggregate_heatmap += cell_grid
            aggregate_count += 1

        if aggregate_count > 0:
            aggregate_heatmap /= aggregate_count

        # ── Per-cell variance across prompts ──
        if progress:
            progress("Computing per-cell variance...")

        cell_values = defaultdict(list)
        for hm in per_prompt_heatmaps:
            if hm is None:
                continue
            arr = np.array(hm)
            for si in range(n_subj):
                for li in range(n_levels):
                    cell_values[(si, li)].append(arr[si, li])

        variance_grid = np.zeros((n_subj, n_levels))
        for (si, li), vals in cell_values.items():
            if len(vals) > 1:
                variance_grid[si, li] = float(np.var(vals))

        # ── Build output ──
        if progress:
            progress("Building output...")

        subj_short = [s.replace("_", " ").title()[:14] for s in subjects]

        # ── Compute per-cell token specificity ──
        if progress:
            progress("Computing cell-specific token rankings...")

        TOP_N = 10
        cat_names = {"b": "benign", "m": "mild", "h": "harmful", "j": "jailbreak",
                     "a": "adversarial", "d": "dual-use", "u": "unknown"}
        all_cats_seen = set()

        # Pass 1: compute each token's mean activation in each cell
        # cell_means[tok][(si,li)] = mean activation
        token_cell_means = defaultdict(dict)
        token_cell_cats = defaultdict(dict)  # per-cat breakdowns
        for (si, li), tok_dict in cell_token_data.items():
            for tok, cat_dict in tok_dict.items():
                all_vals = []
                per_cat = {}
                for cat_short, vals in cat_dict.items():
                    all_cats_seen.add(cat_short)
                    m = float(np.mean(vals))
                    per_cat[cat_short] = {"mean": round(m, 6), "n": len(vals)}
                    all_vals.extend(vals)
                token_cell_means[tok][(si, li)] = float(np.mean(all_vals))
                token_cell_cats[tok][(si, li)] = per_cat

        # Pass 2: compute each token's global mean and std across all cells
        token_global = {}
        for tok, cell_dict in token_cell_means.items():
            means = list(cell_dict.values())
            token_global[tok] = {
                "global_mean": float(np.mean(means)),
                "global_std": float(np.std(means)) if len(means) > 1 else 0.0,
                "n_cells": len(means),
            }

        # Pass 3: for each cell, rank tokens by cell-specificity z-score
        # z = |cell_mean - global_mean| / global_std
        # Tokens that activate THIS cell differently from their average
        # across all cells score highest. "the" scores low because it
        # activates every cell roughly the same.
        cell_details = {}
        for (si, li), tok_dict in cell_token_data.items():
            cell_key = f"{si}_{li}"
            token_rows = []
            for tok in tok_dict:
                cell_mean = token_cell_means[tok].get((si, li), 0)
                g = token_global[tok]
                g_mean = g["global_mean"]
                g_std = g["global_std"]
                n_cells = g["n_cells"]

                # Cell-specificity: how far this cell's activation deviates
                # from the token's average activation across all cells
                deviation = cell_mean - g_mean
                z_score = abs(deviation) / g_std if g_std > 1e-10 else 0.0

                # Total observations of this token in this cell
                all_vals = []
                for vals in tok_dict[tok].values():
                    all_vals.extend(vals)

                token_rows.append({
                    "token": tok,
                    "cell_mean": round(cell_mean, 6),
                    "global_mean": round(g_mean, 6),
                    "deviation": round(deviation, 6),
                    "z": round(z_score, 3),
                    "n": len(all_vals),
                    "n_cells": n_cells,
                    "cats": token_cell_cats[tok].get((si, li), {}),
                })

            # Sort by z-score — most cell-specific tokens first
            token_rows.sort(key=lambda x: x["z"], reverse=True)
            top = token_rows[:TOP_N]

            cell_details[cell_key] = {
                "tokens": top,
                "probes": cell_probes.get((si, li), []),
                "n_unique_tokens": len(token_rows),
            }

        output = {
            "version": self.version,
            "probe_file": probe_file,
            "subjects": subjects,
            "subj_short": subj_short,
            "levels": level_names,
            "n_prompts": n_prompts,
            "n_probes": len(raw_probes),

            # Aggregate heatmap (session mean)
            "aggregate": aggregate_heatmap.tolist(),

            # Per-cell variance (turbulence)
            "variance": variance_grid.tolist(),

            # Per-cell token context variance
            "cell_details": cell_details,

            # Categories observed
            "categories": {k: cat_names.get(k, k) for k in sorted(all_cats_seen)},

            # Summary stats per subject
            "per_subject": {
                subj: {
                    "mean_activation": float(aggregate_heatmap[si].mean()),
                    "max_activation": float(aggregate_heatmap[si].max()),
                    "mean_variance": float(variance_grid[si].mean()),
                }
                for si, subj in enumerate(subjects)
            },
        }

        if progress:
            progress(f"Complete: {n_subj} subjects × {n_levels} levels, "
                     f"{aggregate_count} prompts mapped")

        return output
