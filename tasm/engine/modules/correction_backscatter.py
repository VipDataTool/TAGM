"""
Correction Backscatter Module for TASM.

Measures correction-field intensity coupling between prompt tokens and
probe vocabulary by projecting BOTH through the full ΔW_V weight delta
lens and comparing the resulting energies (norms), not directions.

This is the "wave" view: it asks "does the correction lens respond with
similar intensity to both inputs?" rather than "does it push them in the
same direction?"  Two inputs that both strongly engage the correction
field produce a hot cell regardless of whether they're being pushed in
the same direction — because the measurement is about how much the
medium vibrates, not which way.

The probe lattice defines the coordinate system.  Each cell has a
pre-computed correction energy from its vocabulary passing through the
lens.  When a new prompt arrives, each token gets its own correction
energy from the same lens.  The product of token energy and probe energy
at each cell produces the heatmap.  Structure comes from the template.

No SVD truncation.  No directional comparison in high-dimensional space.
The full lens is used on both sides and the tail is preserved.

Requires:
  - Model loaded (for ΔW_V access at signal layers)
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


class CorrectionBackscatterModule(TASMModule):
    name = "correction_backscatter"
    display_name = "Correction Backscatter"
    description = (
        "Projects prompt tokens and probe vocabulary through the full "
        "ΔW_V correction lens and compares intensity (norm), not direction. "
        "Measures how strongly the correction field responds to both — "
        "hot cells mean both the prompt and the probe vocabulary strongly "
        "engage the same correction lens. Reveals training structure "
        "without truncating the spectral tail."
    )
    version = "0.2.0"

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
        """Provide access to the ModelManager for ΔW access."""
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
                name="delta_projections",
                display_name="Delta Projections",
                description=(
                    "Which weight delta projections to use as the lens. "
                    "v = V-projection only (ASM primary lens). "
                    "qkv = average across Q, K, V (broader, matches stress)."
                ),
                type="select",
                default="v",
                options=["v", "qkv"],
            ),
        ]

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        if self._model_manager is None or self._model_manager.state is None:
            return False, (
                "Model not loaded. The backscatter module requires "
                "ΔW access from the loaded model."
            )

        state = self._model_manager.state
        if not state.loaded or not state.signal_layers:
            return False, (
                "Model not fully loaded or no signal layers computed. "
                "Load a model first."
            )

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
                "Re-analyze prompts to capture them."
            )

        return True, "OK"

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[BACKSCATTER] {msg}")

        state = self._model_manager.state
        probe_file = _get_active_probe(self._project_root)
        aggregation = params.get("aggregation", "mean")
        delta_mode = params.get("delta_projections", "v")

        # ── Load probe structure ──
        prog("Loading probe structure...")
        csv_path = os.path.join(self._project_root, probe_file)
        level_cols, level_names = _detect_level_cols(csv_path)
        if not level_cols:
            raise RuntimeError(f"No level columns found in {probe_file}")

        raw_probes = _load_probes(csv_path)
        if not raw_probes:
            raise RuntimeError(f"No probes loaded from {probe_file}")

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

        # Find probe cache — prefer non-projected (raw representational)
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
                f"No probe cache found for {probe_file} matching "
                f"{len(raw_probes)} probes. Apply the probe set first.")

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

        # ── Determine which deltas to use ──
        signal_layers = state.signal_layers
        if delta_mode == "qkv":
            proj_suffixes = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
        else:
            proj_suffixes = ["v_proj.weight"]

        # ── Compute probe correction energies ──
        # For each probe, project through ΔW at each signal layer, take norm.
        # Average across layers and projections → one scalar per probe.
        prog(f"Computing probe correction energies across "
             f"{len(signal_layers)} signal layers...")

        probe_energies = np.zeros(n_probes)
        probe_contributions = 0

        for layer_idx in signal_layers:
            for suffix in proj_suffixes:
                dname = f"model.layers.{layer_idx}.self_attn.{suffix}"
                dw = state.deltas.get(dname)
                fnorm = state.delta_frob_norms.get(dname, 0)

                if dw is None or fnorm <= 0:
                    continue

                dw_np = dw.float().cpu().numpy()

                if probe_mat.shape[1] != dw_np.shape[1]:
                    continue

                # ‖probe @ ΔW.T‖ / ‖ΔW‖_F → scalar energy per probe
                projected = probe_mat @ dw_np.T
                norms = np.linalg.norm(projected, axis=1) / fnorm
                probe_energies += norms
                probe_contributions += 1

        if probe_contributions > 0:
            probe_energies /= probe_contributions

        # Aggregate probe energies per cell (mean of probes in each cell)
        cell_probe_energy = np.zeros((n_subj, n_levels))
        cell_probe_counts = np.zeros((n_subj, n_levels))
        for pi, (si, li) in enumerate(probe_cells):
            cell_probe_energy[si, li] += probe_energies[pi]
            cell_probe_counts[si, li] += 1

        mask = cell_probe_counts > 0
        cell_probe_energy[mask] /= cell_probe_counts[mask]

        prog(f"Probe energies computed: min={probe_energies.min():.6f}, "
             f"max={probe_energies.max():.6f}, "
             f"mean={probe_energies.mean():.6f}")

        # ── Process prompts ──
        prog("Computing per-prompt backscatter...")

        n_prompts = len(session_results)
        per_prompt_heatmaps = []
        aggregate_heatmap = np.zeros((n_subj, n_levels))
        aggregate_count = 0

        # Per-cell token tracking
        cell_token_data = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        cat_names = {"b": "benign", "m": "mild", "h": "harmful",
                     "j": "jailbreak", "a": "adversarial",
                     "d": "dual-use", "u": "unknown"}
        all_cats_seen = set()

        for pi, r in enumerate(session_results):
            final_emb = r.get("per_token_final_emb")
            if final_emb is None or len(final_emb) == 0:
                per_prompt_heatmaps.append(None)
                continue

            if progress and (pi + 1) % 10 == 0:
                prog(f"Processing prompt {pi+1}/{n_prompts}...")

            tok_mat = np.array(final_emb, dtype=np.float32)
            tokens = r.get("tokens", [])
            cat = (r.get("category", "") or "unknown")[:1]
            all_cats_seen.add(cat)
            n_tok = min(len(tokens), tok_mat.shape[0])

            # Compute per-token correction energy
            # ‖token @ ΔW.T‖ / ‖ΔW‖_F, averaged across signal layers
            token_energies = np.zeros(n_tok)
            tok_contributions = 0

            for layer_idx in signal_layers:
                for suffix in proj_suffixes:
                    dname = f"model.layers.{layer_idx}.self_attn.{suffix}"
                    dw = state.deltas.get(dname)
                    fnorm = state.delta_frob_norms.get(dname, 0)

                    if dw is None or fnorm <= 0:
                        continue

                    dw_np = dw.float().cpu().numpy()
                    if tok_mat.shape[1] != dw_np.shape[1]:
                        continue

                    projected = tok_mat[:n_tok] @ dw_np.T
                    norms = np.linalg.norm(projected, axis=1) / fnorm
                    token_energies += norms
                    tok_contributions += 1

            if tok_contributions > 0:
                token_energies /= tok_contributions

            # Backscatter: token_energy × probe_energy per cell
            cell_grid = np.zeros((n_subj, n_levels))

            for si in range(n_subj):
                for li in range(n_levels):
                    pe = cell_probe_energy[si, li]
                    if pe <= 0:
                        continue

                    # Per-token backscatter for this cell
                    tok_backscatter = token_energies * pe

                    if aggregation == "max":
                        cell_grid[si, li] = float(np.max(tok_backscatter))
                    elif aggregation == "sum":
                        cell_grid[si, li] = float(np.sum(tok_backscatter))
                    else:
                        cell_grid[si, li] = float(np.mean(tok_backscatter))

                    # Track per-token values for cell detail
                    for ti in range(min(n_tok, len(tokens))):
                        tok = tokens[ti].strip() if ti < len(tokens) else ""
                        if tok:
                            cell_token_data[(si, li)][tok][cat].append(
                                float(tok_backscatter[ti]))

            per_prompt_heatmaps.append(cell_grid.tolist())
            aggregate_heatmap += cell_grid
            aggregate_count += 1

        if aggregate_count > 0:
            aggregate_heatmap /= aggregate_count

        # ── Per-cell variance across prompts ──
        prog("Computing per-cell variance...")
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

        # ── Cell-specific token rankings ──
        prog("Computing cell-specific token rankings...")

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
            "n_prompts_projected": aggregate_count,
            "n_probes": len(raw_probes),

            # Energy-product heatmap
            "aggregate": aggregate_heatmap.tolist(),

            # Per-cell variance
            "variance": variance_grid.tolist(),

            # Per-cell token specificity
            "cell_details": cell_details,

            # Categories observed
            "categories": {k: cat_names.get(k, k)
                           for k in sorted(all_cats_seen)},

            # Per-subject summary
            "per_subject": {
                subj: {
                    "mean_backscatter": float(aggregate_heatmap[si].mean()),
                    "max_backscatter": float(aggregate_heatmap[si].max()),
                    "mean_variance": float(variance_grid[si].mean()),
                    "mean_probe_energy": float(cell_probe_energy[si].mean()),
                }
                for si, subj in enumerate(subjects)
            },

            # Probe energy map — the detector array's intrinsic sensitivity
            "probe_energy_grid": cell_probe_energy.tolist(),

            # Configuration
            "config": {
                "aggregation": aggregation,
                "delta_projections": delta_mode,
                "n_signal_layers": len(signal_layers),
                "signal_layers": signal_layers,
                "probe_dim": probe_dim,
                "model_dim": state.hidden_size,
            },
        }

        prog(f"Complete: {n_subj}x{n_levels} grid, "
             f"{aggregate_count} prompts, "
             f"aggregate range [{aggregate_heatmap.min():.6f}, "
             f"{aggregate_heatmap.max():.6f}]")

        return output
