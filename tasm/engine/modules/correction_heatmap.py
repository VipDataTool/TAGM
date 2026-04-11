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
from .domain_surface import (_detect_level_cols, _parse_meta,
                              _load_probe_cache, _probe_cache_path,
                              _load_probes)

logger = logging.getLogger("tasm")

PROBE_CONFIG = "probe_config.json"


def _get_active_probe(project_root):
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

    def set_project_root(self, root):
        self._project_root = root

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

        probe_file = _get_active_probe(self._project_root)
        if not probe_file:
            return False, (
                "No probe set active. Apply a probe set in the "
                "Configuration tab first."
            )

        has_final = any(r.get("per_token_final_emb") for r in session_results)
        if not has_final:
            return False, (
                "No final-layer token embeddings found in session results. "
                "Re-analyze prompts with the current build to capture them."
            )
        return True, "OK"

    def run(self, session_results, params, progress=None):
        probe_file = _get_active_probe(self._project_root)
        proj_method = params.get("projection_method", "abs")

        if progress:
            progress("Loading probe structure...")

        # ── Load probe CSV for subject × subclass structure ──
        csv_path = os.path.join(self._project_root, probe_file)
        level_cols, level_names = _detect_level_cols(csv_path)
        if not level_cols:
            raise RuntimeError(f"No level columns found in {probe_file}")

        raw_probes = _load_probes(csv_path)
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

        # ── Resolve layer depths (template meta overrides global config) ──
        meta = _parse_meta(csv_path)

        try:
            from engine import engine_config
            use_proj = engine_config.get("probe_projection_space")
        except Exception:
            use_proj = False

        if "layer_low" in meta and "layer_high" in meta:
            subj_frac = max(0, min(1, float(meta["layer_low"])))
            esc_frac = max(0, min(1, float(meta["layer_high"])))
            logger.info(f"[HEATMAP] Using template depths: "
                        f"L{int(subj_frac*100)}, L{int(esc_frac*100)}")
        else:
            try:
                subj_frac = max(0, min(1, engine_config.get("domain_embedding_layer_frac") or 0.50))
                esc_frac = max(0, min(1, engine_config.get("domain_escalation_layer_frac") or 0.75))
            except Exception:
                subj_frac = 0.50
                esc_frac = 0.75

        if progress:
            progress(f"Loading probe embeddings at L{int(subj_frac*100)} and L{int(esc_frac*100)}...")

        projected = use_proj

        # Find probe caches
        cache_dir = os.path.join(self._project_root, "probe_cache")
        stem = os.path.splitext(probe_file)[0]

        def _find_cache(frac):
            """Find a matching probe cache file."""
            # Try exact path first
            if hasattr(session_results[0], 'get'):
                # Scan cache dir for matching files
                if os.path.isdir(cache_dir):
                    tag = f"__L{int(frac * 100)}"
                    for fn in sorted(os.listdir(cache_dir)):
                        if fn.startswith(stem) and tag in fn and fn.endswith(".json"):
                            data = _load_probe_cache(os.path.join(cache_dir, fn))
                            if data and data.get("embeddings"):
                                logger.info(f"[HEATMAP] Using cache: {fn}")
                                return data["embeddings"]
            return None

        embs_L50 = _find_cache(subj_frac)
        embs_L75 = _find_cache(esc_frac)

        if embs_L50 is None:
            raise RuntimeError(
                f"Probe cache at L{int(subj_frac*100)} not found for {probe_file}. "
                "Regenerate caches.")
        if embs_L75 is None:
            raise RuntimeError(
                f"Probe cache at L{int(esc_frac*100)} not found for {probe_file}. "
                "Regenerate caches.")

        if len(embs_L50) != len(raw_probes) or len(embs_L75) != len(raw_probes):
            raise RuntimeError(
                f"Probe count mismatch: CSV={len(raw_probes)}, "
                f"L50={len(embs_L50)}, L75={len(embs_L75)}. "
                "Regenerate caches.")

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

            # Per-token, per-cell activation for discriminative tracking
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

        # ── Compute per-cell discriminative tokens ──
        if progress:
            progress("Computing per-cell discriminative tokens...")

        TOP_N = 10
        cat_names = {"b": "benign", "m": "mild", "h": "harmful", "j": "jailbreak",
                     "a": "adversarial", "d": "dual-use", "u": "unknown"}
        all_cats_seen = set()
        cell_details = {}

        for (si, li), tok_dict in cell_token_data.items():
            cell_key = f"{si}_{li}"
            token_rows = []
            for tok, cat_dict in tok_dict.items():
                all_vals = []
                per_cat = {}
                for cat_short, vals in cat_dict.items():
                    all_cats_seen.add(cat_short)
                    m = float(np.mean(vals))
                    per_cat[cat_short] = {"mean": round(m, 6), "n": len(vals)}
                    all_vals.extend(vals)
                overall_mean = float(np.mean(all_vals)) if all_vals else 0
                # Discriminative score: variance across category means
                cat_means = [v["mean"] for v in per_cat.values()]
                disc_score = float(np.var(cat_means)) if len(cat_means) > 1 else 0
                token_rows.append({
                    "token": tok,
                    "mean": round(overall_mean, 6),
                    "n": len(all_vals),
                    "disc": round(disc_score, 8),
                    "cats": per_cat,
                })

            # Sort by discriminative score, take top N
            token_rows.sort(key=lambda x: x["disc"], reverse=True)
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

            # Per-cell discriminative token details
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
