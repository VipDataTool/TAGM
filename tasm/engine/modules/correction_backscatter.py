"""
Correction Backscatter Module for TASM (v0.3.0).

Measures correction-field intensity coupling between prompt tokens and
probe vocabulary by projecting BOTH through weight delta lenses and
comparing the resulting energies (norms), not directions.

v0.3.0 adds full delta decomposition: computes separate heatmaps for
each projection type (Q, K, V, O, QK, QKV) in a single run.  This
shows how different components of the correction field respond to
each probe cell — the attention routing topology (QK) vs value
computation (V) vs output mixing (O).

No SVD truncation.  Full spectral tail preserved on all projections.

Requires:
  - Model loaded (for ΔW access at signal layers)
  - Active probe set with cached embeddings
  - Session results with per_token_final_emb
"""

import os
import json
import logging
import numpy as np
from collections import defaultdict

from .base import TASMModule, ModuleParameter
from .domain_surface import (_detect_level_cols, _parse_meta,
                              _load_probe_cache, _load_probes)
from .correction_heatmap import _get_active_probe

logger = logging.getLogger("tasm")

# All attn projection types and their composite groupings
PROJ_TYPES = {
    "q":   ["q_proj.weight"],
    "k":   ["k_proj.weight"],
    "v":   ["v_proj.weight"],
    "o":   ["o_proj.weight"],
    "qk":  ["q_proj.weight", "k_proj.weight"],
    "qkv": ["q_proj.weight", "k_proj.weight", "v_proj.weight"],
}

PROJ_LABELS = {
    "q": "Q (query routing)",
    "k": "K (key routing)",
    "v": "V (value content)",
    "o": "O (output mixing)",
    "qk": "QK (attention topology)",
    "qkv": "QKV (full attention)",
}

# Display order
PROJ_ORDER = ["v", "q", "k", "o", "qk", "qkv"]


class CorrectionBackscatterModule(TASMModule):
    name = "correction_backscatter"
    display_name = "Correction Backscatter"
    description = (
        "Projects prompt tokens and probe vocabulary through the full "
        "ΔW correction lens and compares intensity (norm), not direction. "
        "Computes separate heatmaps for Q, K, V, O, QK, and QKV projections "
        "to show how different components of the correction field respond. "
        "Reveals training structure without truncating the spectral tail."
    )
    version = "0.3.0"

    min_results = 1
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    def __init__(self):
        super().__init__()
        self._project_root = None
        self._model_manager = None

    def set_project_root(self, root):
        self._project_root = root

    def set_model_manager(self, mm):
        self._model_manager = mm

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="aggregation",
                display_name="Aggregation",
                description=(
                    "How to combine per-token backscatter values within a prompt. "
                    "mean = average across tokens (default). "
                    "max = peak backscatter. "
                    "sum = total backscatter energy."
                ),
                type="select",
                default="mean",
                options=["mean", "max", "sum"],
            ),
            ModuleParameter(
                name="primary_projection",
                display_name="Primary Projection",
                description=(
                    "Which projection to use for the main aggregate heatmap "
                    "and cell-detail token rankings. All projections are always "
                    "computed and shown in the decomposition panel."
                ),
                type="select",
                default="qkv",
                options=["v", "q", "k", "o", "qk", "qkv"],
            ),
        ]

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        if self._model_manager is None or self._model_manager.state is None:
            return False, "Model not loaded. Backscatter requires ΔW access."

        state = self._model_manager.state
        if not state.loaded or not state.signal_layers:
            return False, "Model not fully loaded or no signal layers."

        if not _get_active_probe(self._project_root):
            return False, "No probe set active."

        if not any(r.get("per_token_final_emb") for r in session_results):
            return False, "No final-layer token embeddings in session."

        return True, "OK"

    def _compute_energies_for_suffix(self, suffix, signal_layers, state,
                                      probe_mat, tok_mats, n_toks):
        """Compute probe and per-prompt token energies for one delta suffix.

        Returns (probe_energies, list_of_token_energy_arrays) or None if
        no deltas are available for this suffix.
        """
        n_probes = probe_mat.shape[0]
        probe_energies = np.zeros(n_probes)
        token_energies_list = [np.zeros(nt) for nt in n_toks]
        contributions = 0

        for layer_idx in signal_layers:
            dname = f"model.layers.{layer_idx}.self_attn.{suffix}"
            dw = state.deltas.get(dname)
            fnorm = state.delta_frob_norms.get(dname, 0)
            if dw is None or fnorm <= 0:
                continue

            dw_np = dw.float().cpu().numpy()
            if probe_mat.shape[1] != dw_np.shape[1]:
                continue

            # Probe energies
            proj_p = probe_mat @ dw_np.T
            probe_energies += np.linalg.norm(proj_p, axis=1) / fnorm

            # Token energies per prompt
            for pi, (tm, nt) in enumerate(zip(tok_mats, n_toks)):
                if tm is None or nt == 0:
                    continue
                if tm.shape[1] != dw_np.shape[1]:
                    continue
                proj_t = tm[:nt] @ dw_np.T
                token_energies_list[pi] += np.linalg.norm(proj_t, axis=1) / fnorm

            contributions += 1

        if contributions == 0:
            return None

        probe_energies /= contributions
        for te in token_energies_list:
            te /= contributions

        return probe_energies, token_energies_list

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[BACKSCATTER] {msg}")

        state = self._model_manager.state
        probe_file = _get_active_probe(self._project_root)
        aggregation = params.get("aggregation", "mean")
        primary = params.get("primary_projection", "qkv")

        # ── Load probe structure ──
        prog("Loading probe structure...")
        csv_path = os.path.join(self._project_root, probe_file)
        level_cols, level_names = _detect_level_cols(csv_path)
        if not level_cols:
            raise RuntimeError(f"No level columns in {probe_file}")

        raw_probes = _load_probes(csv_path)
        if not raw_probes:
            raise RuntimeError(f"No probes from {probe_file}")

        subjects = []
        for p in raw_probes:
            if p["subject"] not in subjects:
                subjects.append(p["subject"])
        n_subj = len(subjects)
        n_levels = len(level_cols)
        subj_idx = {s: i for i, s in enumerate(subjects)}

        # ── Load probe embeddings ──
        prog("Loading probe embeddings...")
        meta = _parse_meta(csv_path)
        if "layer_low" in meta:
            subj_frac = max(0, min(1, float(meta["layer_low"])))
        else:
            try:
                from engine import engine_config
                subj_frac = max(0, min(1,
                    engine_config.get("domain_embedding_layer_frac") or 0.50))
            except Exception:
                subj_frac = 0.50

        cache_dir = os.path.join(self._project_root, "probe_cache")
        stem = os.path.splitext(probe_file)[0]

        probe_embs_raw = None
        if os.path.isdir(cache_dir):
            tag = f"__L{int(subj_frac * 100)}"
            candidates = sorted(os.listdir(cache_dir))
            non_proj = [fn for fn in candidates
                        if fn.startswith(stem) and tag in fn
                        and fn.endswith(".json") and "_proj.json" not in fn]
            proj = [fn for fn in candidates
                    if fn.startswith(stem) and tag in fn
                    and fn.endswith(".json") and "_proj.json" in fn]
            for fn in non_proj + proj:
                data = _load_probe_cache(os.path.join(cache_dir, fn))
                if data and data.get("embeddings"):
                    embs = data["embeddings"]
                    if len(embs) == len(raw_probes):
                        logger.info(f"[BACKSCATTER] Using cache: {fn}")
                        probe_embs_raw = embs
                        break

        if probe_embs_raw is None:
            raise RuntimeError(
                f"No probe cache for {probe_file} with {len(raw_probes)} probes.")

        probe_mat = np.array(probe_embs_raw, dtype=np.float32)
        n_probes = probe_mat.shape[0]
        probe_dim = probe_mat.shape[1]

        # Map probes to cells
        probe_cells = []
        cell_probes = defaultdict(list)
        for p in raw_probes:
            si = subj_idx.get(p["subject"], 0)
            li = p["level"]
            probe_cells.append((si, li))
            cell_probes[(si, li)].append(
                p.get("text", p.get("anchor_id", "")))

        signal_layers = state.signal_layers

        # ── Pre-load token matrices ──
        prog("Loading token embeddings...")
        tok_mats = []
        n_toks = []
        tokens_list = []
        cats_list = []
        cat_names = {"b": "benign", "m": "mild", "h": "harmful",
                     "j": "jailbreak", "a": "adversarial",
                     "d": "dual-use", "u": "unknown"}
        all_cats_seen = set()

        for r in session_results:
            fe = r.get("per_token_final_emb")
            if fe and len(fe) > 0:
                tm = np.array(fe, dtype=np.float32)
                tokens = r.get("tokens", [])
                nt = min(len(tokens), tm.shape[0])
                tok_mats.append(tm)
                n_toks.append(nt)
                tokens_list.append(tokens)
                cat = (r.get("category", "") or "unknown")[:1]
                cats_list.append(cat)
                all_cats_seen.add(cat)
            else:
                tok_mats.append(None)
                n_toks.append(0)
                tokens_list.append([])
                cats_list.append("u")

        n_prompts = len(session_results)

        # ── Compute per-suffix energies ──
        # For each individual suffix, compute probe + token energies
        suffix_list = ["q_proj.weight", "k_proj.weight",
                       "v_proj.weight", "o_proj.weight"]
        suffix_short = {"q_proj.weight": "q", "k_proj.weight": "k",
                        "v_proj.weight": "v", "o_proj.weight": "o"}
        suffix_energies = {}  # short_name → (probe_energies, [token_energies])

        for sfx in suffix_list:
            short = suffix_short[sfx]
            prog(f"Computing {short.upper()} projection energies...")
            result = self._compute_energies_for_suffix(
                sfx, signal_layers, state, probe_mat, tok_mats, n_toks)
            if result is not None:
                suffix_energies[short] = result

        # Build composite projections (QK, QKV) by averaging
        def _composite(keys):
            """Average probe and token energies across multiple suffixes."""
            available = [k for k in keys if k in suffix_energies]
            if not available:
                return None
            probe_sum = np.zeros(n_probes)
            tok_sums = [np.zeros(nt) for nt in n_toks]
            for k in available:
                pe, te_list = suffix_energies[k]
                probe_sum += pe
                for pi, te in enumerate(te_list):
                    tok_sums[pi] += te
            n = len(available)
            return probe_sum / n, [ts / n for ts in tok_sums]

        suffix_energies["qk"] = _composite(["q", "k"])
        suffix_energies["qkv"] = _composite(["q", "k", "v"])

        # ── Build heatmaps for each projection type ──
        def _build_heatmap(proj_key):
            """Build aggregate heatmap and probe energy grid for one projection."""
            if proj_key not in suffix_energies or suffix_energies[proj_key] is None:
                return None

            probe_e, tok_e_list = suffix_energies[proj_key]

            # Probe energy per cell
            cell_pe = np.zeros((n_subj, n_levels))
            cell_pc = np.zeros((n_subj, n_levels))
            for pi2, (si, li) in enumerate(probe_cells):
                cell_pe[si, li] += probe_e[pi2]
                cell_pc[si, li] += 1
            m = cell_pc > 0
            cell_pe[m] /= cell_pc[m]

            # Aggregate heatmap
            agg = np.zeros((n_subj, n_levels))
            count = 0
            for pi2 in range(n_prompts):
                te = tok_e_list[pi2]
                nt = n_toks[pi2]
                if nt == 0:
                    continue
                grid = np.zeros((n_subj, n_levels))
                for si in range(n_subj):
                    for li in range(n_levels):
                        pe_val = cell_pe[si, li]
                        if pe_val <= 0:
                            continue
                        bs = te[:nt] * pe_val
                        if aggregation == "max":
                            grid[si, li] = float(np.max(bs))
                        elif aggregation == "sum":
                            grid[si, li] = float(np.sum(bs))
                        else:
                            grid[si, li] = float(np.mean(bs))
                agg += grid
                count += 1

            if count > 0:
                agg /= count

            return {
                "aggregate": agg.tolist(),
                "probe_energy": cell_pe.tolist(),
            }

        decomposition = {}
        for pk in PROJ_ORDER:
            prog(f"Building {pk.upper()} heatmap...")
            hm = _build_heatmap(pk)
            if hm is not None:
                decomposition[pk] = hm

        # ── Primary projection: full detail with token tracking ──
        prog(f"Computing cell-detail tokens for primary ({primary.upper()})...")

        primary_data = decomposition.get(primary)
        if primary_data is None:
            # Fall back to first available
            for pk in PROJ_ORDER:
                if pk in decomposition:
                    primary = pk
                    primary_data = decomposition[pk]
                    break

        aggregate_heatmap = np.array(primary_data["aggregate"]) if primary_data else np.zeros((n_subj, n_levels))
        cell_probe_energy = np.array(primary_data["probe_energy"]) if primary_data else np.zeros((n_subj, n_levels))

        # Token-level detail for primary projection
        cell_token_data = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))

        if primary in suffix_energies and suffix_energies[primary] is not None:
            _, tok_e_list = suffix_energies[primary]
            for pi2 in range(n_prompts):
                te = tok_e_list[pi2]
                nt = n_toks[pi2]
                tokens = tokens_list[pi2]
                cat = cats_list[pi2]
                if nt == 0:
                    continue
                for si in range(n_subj):
                    for li in range(n_levels):
                        pe_val = cell_probe_energy[si, li]
                        if pe_val <= 0:
                            continue
                        for ti in range(min(nt, len(tokens))):
                            tok = tokens[ti].strip()
                            if tok:
                                cell_token_data[(si, li)][tok][cat].append(
                                    float(te[ti] * pe_val))

        # ── Per-cell variance (primary only) ──
        variance_grid = np.zeros((n_subj, n_levels))
        if primary in suffix_energies and suffix_energies[primary] is not None:
            _, tok_e_list = suffix_energies[primary]
            cell_values = defaultdict(list)
            for pi2 in range(n_prompts):
                te = tok_e_list[pi2]
                nt = n_toks[pi2]
                if nt == 0:
                    continue
                for si in range(n_subj):
                    for li in range(n_levels):
                        pe_val = cell_probe_energy[si, li]
                        if pe_val <= 0:
                            continue
                        bs = te[:nt] * pe_val
                        if aggregation == "max":
                            val = float(np.max(bs))
                        elif aggregation == "sum":
                            val = float(np.sum(bs))
                        else:
                            val = float(np.mean(bs))
                        cell_values[(si, li)].append(val)
            for (si, li), vals in cell_values.items():
                if len(vals) > 1:
                    variance_grid[si, li] = float(np.var(vals))

        # ── Cell-specific token rankings ──
        TOP_N = 10
        token_cell_means = defaultdict(dict)
        token_cell_cats = defaultdict(dict)
        for (si, li), tok_dict in cell_token_data.items():
            for tok, cat_dict in tok_dict.items():
                all_vals = []
                per_cat = {}
                for cat_short, vals in cat_dict.items():
                    m = float(np.mean(vals))
                    per_cat[cat_short] = {"mean": round(m, 6), "n": len(vals)}
                    all_vals.extend(vals)
                token_cell_means[tok][(si, li)] = float(np.mean(all_vals))
                token_cell_cats[tok][(si, li)] = per_cat

        token_global = {}
        for tok, cell_dict in token_cell_means.items():
            means = list(cell_dict.values())
            token_global[tok] = {
                "global_mean": float(np.mean(means)),
                "global_std": float(np.std(means)) if len(means) > 1 else 0.0,
                "n_cells": len(means),
            }

        cell_details = {}
        for (si, li), tok_dict in cell_token_data.items():
            cell_key = f"{si}_{li}"
            token_rows = []
            for tok in tok_dict:
                cell_mean = token_cell_means[tok].get((si, li), 0)
                g = token_global[tok]
                deviation = cell_mean - g["global_mean"]
                z_score = (abs(deviation) / g["global_std"]
                           if g["global_std"] > 1e-10 else 0.0)
                all_vals = []
                for vals in tok_dict[tok].values():
                    all_vals.extend(vals)
                token_rows.append({
                    "token": tok,
                    "cell_mean": round(cell_mean, 6),
                    "global_mean": round(g["global_mean"], 6),
                    "deviation": round(deviation, 6),
                    "z": round(z_score, 3),
                    "n": len(all_vals),
                    "n_cells": g["n_cells"],
                    "cats": token_cell_cats[tok].get((si, li), {}),
                })
            token_rows.sort(key=lambda x: x["z"], reverse=True)
            cell_details[cell_key] = {
                "tokens": token_rows[:TOP_N],
                "probes": cell_probes.get((si, li), []),
                "probe_energy": round(float(cell_probe_energy[si, li]), 6),
                "n_unique_tokens": len(token_rows),
            }

        # ── Build output ──
        subj_short = [s.replace("_", " ").title()[:14] for s in subjects]

        output = {
            "version": self.version,
            "module": "correction_backscatter",
            "probe_file": probe_file,
            "subjects": subjects,
            "subj_short": subj_short,
            "levels": level_names,
            "n_prompts": n_prompts,
            "n_prompts_projected": sum(1 for nt in n_toks if nt > 0),
            "n_probes": len(raw_probes),

            # Primary projection heatmap
            "primary_projection": primary,
            "aggregate": aggregate_heatmap.tolist(),
            "probe_energy_grid": cell_probe_energy.tolist(),
            "variance": variance_grid.tolist(),
            "cell_details": cell_details,

            # Full decomposition: all projection types
            "decomposition": decomposition,
            "projection_labels": PROJ_LABELS,
            "projection_order": [pk for pk in PROJ_ORDER if pk in decomposition],

            # Categories
            "categories": {k: cat_names.get(k, k)
                           for k in sorted(all_cats_seen)},

            # Per-subject summary (primary)
            "per_subject": {
                subj: {
                    "mean_backscatter": float(aggregate_heatmap[si].mean()),
                    "max_backscatter": float(aggregate_heatmap[si].max()),
                    "mean_variance": float(variance_grid[si].mean()),
                    "mean_probe_energy": float(cell_probe_energy[si].mean()),
                }
                for si, subj in enumerate(subjects)
            },

            # Config
            "config": {
                "aggregation": aggregation,
                "primary_projection": primary,
                "n_signal_layers": len(signal_layers),
                "signal_layers": signal_layers,
                "probe_dim": probe_dim,
                "model_dim": state.hidden_size,
                "projections_computed": [pk for pk in PROJ_ORDER
                                         if pk in decomposition],
            },
        }

        prog(f"Complete: {n_subj}x{n_levels} grid, "
             f"{len(decomposition)} projections computed, "
             f"primary={primary}")

        return output
