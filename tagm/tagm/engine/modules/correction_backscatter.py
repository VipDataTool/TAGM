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
from tagm.probes.io import (
    detect_level_cols,
    parse_meta,
    load_probe_cache,
    load_probes,
    get_active_probe_set,
)

logger = logging.getLogger("tasm")


class _PipelineState:
    """Adapter providing TASM-compatible state interface over a TAGM Pipeline.

    Backscatter accesses deltas by full state_dict key like
    "model.layers.5.self_attn.v_proj.weight". This class translates
    those lookups into Pipeline.delta_store.get(layer_idx, role) calls.
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._adapter = pipeline.adapter
        self._model = pipeline.instruct_model
        self._delta_store = pipeline.delta_store

        # Build delta lookup and Frobenius norms
        self._deltas = {}
        self._frob_norms = {}
        n = self._adapter.n_layers(self._model)
        # Adapter role names are "q","k","v","o" but backscatter constructs keys
        # like "model.layers.5.self_attn.v_proj.weight" — adapter generates the same
        role_map = {"q": "q_proj.weight", "k": "k_proj.weight",
                    "v": "v_proj.weight", "o": "o_proj.weight"}
        for layer_idx in range(n):
            for adapter_role in ("q", "k", "v", "o"):
                dw = self._delta_store.get_or_none(layer_idx, adapter_role)
                if dw is not None:
                    key = self._adapter.projection_weight_key(adapter_role, layer_idx)
                    self._deltas[key] = dw
                    self._frob_norms[key] = float(dw.float().norm().item())

    @property
    def loaded(self):
        return self._pipeline.loaded

    @property
    def hidden_size(self):
        return self._adapter.hidden_size(self._model)

    @property
    def signal_layers(self):
        from tagm.engine import config as engine_config
        n = self._adapter.n_layers(self._model)
        return list(range(n))

    @property
    def deltas(self):
        return self._deltas

    @property
    def delta_frob_norms(self):
        return self._frob_norms


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
        self._pipeline = None

    def set_project_root(self, root):
        self._project_root = root

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline

    def set_model_manager(self, mm):
        pass  # use set_pipeline instead

    def _make_state(self):
        """Build a state-like object from the pipeline for delta access."""
        return _PipelineState(self._pipeline)

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

        if self._pipeline is None or not self._pipeline.loaded:
            return False, "Model not loaded. Backscatter requires ΔW access."

        state = self._make_state()
        if not state.signal_layers:
            return False, "No signal layers configured."

        active = get_active_probe_set(self._project_root)
        if active is None:
            return False, ("No probe set active. Apply one in "
                           "Configuration → Probe Set.")
        ok, msg = active.validate_against(self._pipeline)
        if not ok:
            return False, msg

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

        state = self._make_state()
        active = get_active_probe_set(self._project_root)
        if active is None:
            raise RuntimeError("No probe set active. Apply one in "
                               "Configuration → Probe Set.")
        ok, msg = active.validate_against(self._pipeline)
        if not ok:
            raise RuntimeError(msg)
        probe_file = active.probe_file
        aggregation = params.get("aggregation", "mean")
        primary = params.get("primary_projection", "qkv")

        # ── Load probe structure ──
        prog("Loading probe structure...")
        csv_path = os.path.join(self._project_root, probe_file)
        level_cols, level_names = detect_level_cols(csv_path)
        if not level_cols:
            raise RuntimeError(f"No level columns in {probe_file}")

        raw_probes = load_probes(csv_path)
        if not raw_probes:
            raise RuntimeError(f"No probes from {probe_file}")

        subjects = []
        for p in raw_probes:
            if p["subject"] not in subjects:
                subjects.append(p["subject"])
        n_subj = len(subjects)
        n_levels = len(level_cols)
        subj_idx = {s: i for i, s in enumerate(subjects)}

        # ── Load probe embeddings via the active-set resolver ──
        # subject_layer_frac() is whatever depth was actually embedded at
        # apply time — recorded in probe_config.json. The cache path is
        # exact; no directory scan, no alphabetical first-match. If the
        # cache is missing or stale the resolver tells the user to
        # re-Apply.
        subj_frac = active.subject_layer_frac()
        prog(f"Loading probe embeddings at L{int(subj_frac * 100)}...")
        cache_path = active.cache_path(self._project_root, subj_frac)
        data = load_probe_cache(cache_path)
        if not data or not data.get("embeddings"):
            raise RuntimeError(
                f"Probe cache not found at {cache_path}. The active probe "
                f"set claims it should exist; re-Apply the probe set in "
                f"Configuration → Probe Set to regenerate it.")
        probe_embs_raw = data["embeddings"]
        if len(probe_embs_raw) != len(raw_probes):
            raise RuntimeError(
                f"Probe count mismatch in {cache_path}: cache has "
                f"{len(probe_embs_raw)}, CSV has {len(raw_probes)}. "
                f"Re-Apply the probe set to refresh.")
        logger.info(f"[BACKSCATTER] Using cache: {os.path.basename(cache_path)}")

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
        prompts_list = []
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
                prompts_list.append(r.get("prompt", "") or "")
            else:
                tok_mats.append(None)
                n_toks.append(0)
                tokens_list.append([])
                cats_list.append("u")
                prompts_list.append(r.get("prompt", "") or "")

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
            """Build aggregate heatmap, per-category heatmaps, and per-prompt
            heatmaps for one projection.

            Returns:
                aggregate:      (n_subj, n_levels) mean across all prompts
                probe_energy:   (n_subj, n_levels) intrinsic detector sensitivity
                per_category:   dict of {category_code: {aggregate, n_prompts}}
                per_prompt:     list of {prompt_idx, category, grid} (one per prompt)
            """
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

            # Aggregate heatmap (all prompts) + per-category + per-prompt
            agg = np.zeros((n_subj, n_levels))
            count = 0
            cat_accum = defaultdict(lambda: {"grid": np.zeros((n_subj, n_levels)),
                                              "count": 0})
            per_prompt = []

            # Per-probe-per-prompt backscatter:
            #   shape (n_projected_prompts, n_probes), aggregation applied
            #   across tokens in that prompt using the same rule as cells.
            # We store a sparse dict keyed by prompt_idx to avoid emitting
            # a fat 2D array for prompts that had no projected tokens.
            probe_backscatter_per_prompt = {}

            for pi2 in range(n_prompts):
                te = tok_e_list[pi2]
                nt = n_toks[pi2]
                if nt == 0:
                    # Skip empty prompts but keep their slot for indexing
                    per_prompt.append(None)
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

                cat = cats_list[pi2]
                cat_accum[cat]["grid"] += grid
                cat_accum[cat]["count"] += 1

                per_prompt.append({
                    "prompt_idx": pi2,
                    "category": cat,
                    "grid": grid.tolist(),
                })

                # ── Per-probe backscatter for this prompt ──
                # For each probe, backscatter = aggregate(token_energy * probe_energy)
                # across the tokens in this prompt.
                probe_row = np.zeros(n_probes)
                token_slice = te[:nt]  # shape (nt,)
                for probe_i in range(n_probes):
                    pe_probe = probe_e[probe_i]
                    if pe_probe <= 0:
                        continue
                    bs = token_slice * pe_probe  # shape (nt,)
                    if aggregation == "max":
                        probe_row[probe_i] = float(np.max(bs))
                    elif aggregation == "sum":
                        probe_row[probe_i] = float(np.sum(bs))
                    else:
                        probe_row[probe_i] = float(np.mean(bs))
                probe_backscatter_per_prompt[pi2] = [
                    round(float(v), 8) for v in probe_row
                ]

            if count > 0:
                agg /= count

            per_category = {}
            for cat, info in cat_accum.items():
                if info["count"] > 0:
                    info["grid"] /= info["count"]
                per_category[cat] = {
                    "aggregate": info["grid"].tolist(),
                    "n_prompts": info["count"],
                }

            return {
                "aggregate": agg.tolist(),
                "probe_energy": cell_pe.tolist(),
                "per_category": per_category,
                "per_prompt": per_prompt,
                "probe_backscatter_per_prompt": probe_backscatter_per_prompt,
            }

        decomposition = {}
        for pk in PROJ_ORDER:
            prog(f"Building {pk.upper()} heatmap...")
            hm = _build_heatmap(pk)
            if hm is not None:
                decomposition[pk] = hm

        # ── Primary projection: full detail with probe-level tracking ──
        prog(f"Computing cell-detail probes for primary ({primary.upper()})...")

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

        # ── Per-probe energy detail ──
        # For each projection, extract individual probe energies
        probe_energy_vectors = {}
        for pk in PROJ_ORDER:
            if pk in suffix_energies and suffix_energies[pk] is not None:
                pe, _ = suffix_energies[pk]
                probe_energy_vectors[pk] = pe

        # Global probe stats (across all probes, primary projection)
        primary_pe = probe_energy_vectors.get(primary, np.zeros(n_probes))
        global_probe_mean = float(np.mean(primary_pe)) if n_probes > 0 else 0.0
        global_probe_std = float(np.std(primary_pe)) if n_probes > 1 else 0.0

        cell_details = {}
        for si in range(n_subj):
            for li in range(n_levels):
                cell_key = f"{si}_{li}"
                # Find probe indices belonging to this cell
                cell_probe_idx = [pi for pi, (s, l) in enumerate(probe_cells)
                                  if s == si and l == li]
                if not cell_probe_idx:
                    cell_details[cell_key] = {
                        "probes": [],
                        "probe_energy": round(float(cell_probe_energy[si, li]), 6),
                        "n_probes": 0,
                    }
                    continue

                probe_rows = []
                for pi in cell_probe_idx:
                    text = raw_probes[pi].get("text", "") if pi < len(raw_probes) else ""
                    energy = float(primary_pe[pi])
                    deviation = energy - global_probe_mean
                    z = abs(deviation) / global_probe_std if global_probe_std > 1e-12 else 0.0

                    # Per-projection energy for this probe
                    per_proj = {}
                    for pk in PROJ_ORDER:
                        if pk in probe_energy_vectors:
                            per_proj[pk] = round(float(probe_energy_vectors[pk][pi]), 6)

                    probe_rows.append({
                        "probe_idx": pi,
                        "text": text,
                        "energy": round(energy, 6),
                        "global_mean": round(global_probe_mean, 6),
                        "deviation": round(deviation, 6),
                        "z": round(z, 3),
                        "projections": per_proj,
                    })

                probe_rows.sort(key=lambda x: x["energy"], reverse=True)

                cell_details[cell_key] = {
                    "probes": probe_rows,
                    "probe_energy": round(float(cell_probe_energy[si, li]), 6),
                    "n_probes": len(probe_rows),
                    "cell_mean": round(float(np.mean([primary_pe[pi] for pi in cell_probe_idx])), 6),
                    "cell_std": round(float(np.std([primary_pe[pi] for pi in cell_probe_idx])), 6),
                    "cell_min": round(float(np.min([primary_pe[pi] for pi in cell_probe_idx])), 6),
                    "cell_max": round(float(np.max([primary_pe[pi] for pi in cell_probe_idx])), 6),
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

            # Prompt metadata for slideshow / filtering
            "prompts": [
                {
                    "idx": pi,
                    "category": cats_list[pi],
                    "text": (prompts_list[pi] or "")[:200],
                    "projected": n_toks[pi] > 0,
                }
                for pi in range(n_prompts)
            ],

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
                "global_probe_mean": round(global_probe_mean, 6),
                "global_probe_std": round(global_probe_std, 6),
                # Active-set provenance: record exactly which cache was
                # consumed and which model the resolver bound to. Useful
                # in module-log JSON for after-the-fact debugging when a
                # user looks at last week's run and asks "wait, which
                # probe cache did this come from?"
                "probe_cache_used": os.path.basename(cache_path),
                "active_probe_model_id": active.model_id,
            },
        }

        prog(f"Complete: {n_subj}x{n_levels} grid, "
             f"{len(decomposition)} projections computed, "
             f"primary={primary}")

        return output
