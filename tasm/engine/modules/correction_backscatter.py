"""
Correction Backscatter Module for TASM.

Measures correction-field coupling between prompt tokens and probe
vocabulary by projecting BOTH through the weight delta lens (ΔW_V).

Unlike the Correction Heatmap (which compares prompt token representations
against probe inter-layer rotation directions), the Backscatter Heatmap
projects both sides through the same ΔW_V lens at signal layers and
measures similarity in correction-field space.  This reveals where
alignment training applies similar correction strategies to prompt
content and probe vocabulary — a measurement of the training's internal
structure rather than the prompt's representational properties.

Analogy: the prompt tokens are a signal source, ΔW_V is the medium,
and the probes are a detector array.  What lights up tells you about
the medium's structure — which domain regions the correction field
couples to for this input.

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
        "Projects prompt tokens and probe vocabulary through the same "
        "ΔW_V correction lens at signal layers. Measures coupling in "
        "correction-field space — which probe regions the correction "
        "field treats the same way as the input. Reveals training "
        "structure rather than representational similarity."
    )
    version = "0.1.0"

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
                name="projection_method",
                display_name="Projection Method",
                description=(
                    "How to aggregate the correction-space dot products. "
                    "abs = magnitude (default, best for initial exploration). "
                    "squared = energy (amplifies strong coupling). "
                    "signed = directional (positive = co-corrected, "
                    "negative = counter-corrected)."
                ),
                type="select",
                default="abs",
                options=["abs", "squared", "signed"],
            ),
            ModuleParameter(
                name="delta_projections",
                display_name="Delta Projections",
                description=(
                    "Which weight delta projections to use as the lens. "
                    "v = V-projection only (ASM primary lens). "
                    "qkv = average across Q, K, V (broader, matches stress). "
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

        # Require loaded model for ΔW access
        if self._model_manager is None or self._model_manager.state is None:
            return False, (
                "Model not loaded. The backscatter module requires "
                "ΔW_V access from the loaded model."
            )

        state = self._model_manager.state
        if not state.loaded or not state.signal_layers:
            return False, (
                "Model not fully loaded or no signal layers computed. "
                "Load a model first."
            )

        # Check for active probe set
        probe_file = _get_active_probe(self._project_root)
        if not probe_file:
            return False, (
                "No probe set active. Apply a probe set in the "
                "Configuration tab first."
            )

        # Check for per_token_final_emb in session
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
        proj_method = params.get("projection_method", "abs")
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

        # Build subject/level maps
        subjects = []
        for p in raw_probes:
            if p["subject"] not in subjects:
                subjects.append(p["subject"])
        n_subj = len(subjects)
        n_levels = len(level_cols)
        subj_idx = {s: i for i, s in enumerate(subjects)}

        # ── Load probe embeddings ──
        prog("Loading probe embeddings...")

        # Resolve layer depths from template meta or global config
        meta = _parse_meta(csv_path)
        try:
            from engine import engine_config
            use_proj = engine_config.get("probe_projection_space")
        except Exception:
            use_proj = False

        if "layer_low" in meta and "layer_high" in meta:
            subj_frac = max(0, min(1, float(meta["layer_low"])))
        else:
            try:
                subj_frac = max(0, min(1,
                    engine_config.get("domain_embedding_layer_frac") or 0.50))
            except Exception:
                subj_frac = 0.50

        cache_dir = os.path.join(self._project_root, "probe_cache")
        stem = os.path.splitext(probe_file)[0]

        # Session embedding dimension for cache validation
        session_dim = None
        for r in session_results:
            fe = r.get("per_token_final_emb")
            if fe and len(fe) > 0:
                session_dim = len(fe[0])
                break

        def _find_cache(frac):
            """Find probe cache — prefer non-projected (raw representational)."""
            if os.path.isdir(cache_dir):
                tag = f"__L{int(frac * 100)}"
                # Prefer non-projected caches for backscatter (we project
                # through ΔW ourselves), but fall back to projected if needed
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
                        if session_dim is not None and len(embs) > 0:
                            cache_dim = len(embs[0])
                            if cache_dim != session_dim:
                                # Dimension mismatch OK here — probes and
                                # prompt tokens go through ΔW separately.
                                # We only need probe count to match CSV.
                                pass
                        if len(embs) == len(raw_probes):
                            is_proj = "_proj.json" in fn
                            logger.info(f"[BACKSCATTER] Using cache: {fn} "
                                        f"(projected={is_proj})")
                            return embs, is_proj
            return None, False

        probe_embs_raw, probes_pre_projected = _find_cache(subj_frac)

        if probe_embs_raw is None:
            raise RuntimeError(
                f"No probe cache found for {probe_file} matching "
                f"{len(raw_probes)} probes. Apply the probe set first.")

        probe_mat = np.array(probe_embs_raw, dtype=np.float32)
        probe_dim = probe_mat.shape[1]
        n_probes = probe_mat.shape[0]
        prog(f"Loaded {n_probes} probe embeddings ({probe_dim}-dim, "
             f"pre-projected={probes_pre_projected})")

        # Map probes to cells
        probe_cells = []
        cell_probes = defaultdict(list)
        for p in raw_probes:
            si = subj_idx.get(p["subject"], 0)
            li = p["level"]
            probe_cells.append((si, li))
            cell_probes[(si, li)].append(
                p.get("text", p.get("anchor_id", "")))

        # ── Collect ΔW projections at signal layers ──
        signal_layers = state.signal_layers
        prog(f"Collecting ΔW at {len(signal_layers)} signal layers "
             f"(mode={delta_mode})...")

        # Determine which delta keys to use
        if delta_mode == "qkv":
            proj_suffixes = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
        else:
            proj_suffixes = ["v_proj.weight"]

        # Pre-project probe embeddings through each layer's ΔW
        # and prompt tokens through same ΔW, then dot product.
        # Accumulate across signal layers.

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

            # Accumulate backscatter across signal layers and delta projections
            cell_grid = np.zeros((n_subj, n_levels))
            cell_counts = np.zeros((n_subj, n_levels))
            n_contributions = 0

            for layer_idx in signal_layers:
                for suffix in proj_suffixes:
                    dname = f"model.layers.{layer_idx}.self_attn.{suffix}"
                    dw = state.deltas.get(dname)
                    fnorm = state.delta_frob_norms.get(dname, 0)

                    if dw is None or fnorm <= 0:
                        continue

                    dw_np = dw.float().cpu().numpy()

                    # Project prompt tokens through ΔW: [n_tok, hidden] @ [hidden, proj_dim] → [n_tok, proj_dim]
                    if tok_mat.shape[1] != dw_np.shape[1]:
                        continue  # dimension mismatch (shouldn't happen with loaded model)
                    tok_corrected = (tok_mat[:n_tok] @ dw_np.T) / fnorm

                    # Project probes through same ΔW
                    if probe_mat.shape[1] == dw_np.shape[1]:
                        # Probes have same dim as model — direct projection
                        probe_corrected = (probe_mat @ dw_np.T) / fnorm
                    elif probes_pre_projected:
                        # Probes were already projected through o_proj —
                        # different subspace, skip ΔW projection for probes
                        # and use them as-is in their projected space.
                        # This is an approximation.
                        probe_corrected = probe_mat
                    else:
                        # Dimension mismatch and not pre-projected — skip
                        logger.warning(
                            f"[BACKSCATTER] Probe dim {probe_mat.shape[1]} != "
                            f"model dim {dw_np.shape[1]}, skipping layer {layer_idx}")
                        continue

                    # Dot product in correction space:
                    # [n_tok, proj_dim] @ [proj_dim, n_probes] → [n_tok, n_probes]
                    backscatter = tok_corrected @ probe_corrected.T

                    # Aggregate per cell
                    for probe_idx, (si, li) in enumerate(probe_cells):
                        col = backscatter[:n_tok, probe_idx]
                        if proj_method == "squared":
                            cell_grid[si, li] += (col ** 2).mean()
                        elif proj_method == "signed":
                            cell_grid[si, li] += col.mean()
                        else:
                            cell_grid[si, li] += np.abs(col).mean()
                        cell_counts[si, li] += 1

                    # Per-token cell values for token tracking
                    for probe_idx, (si, li) in enumerate(probe_cells):
                        col = backscatter[:n_tok, probe_idx]
                        if proj_method == "squared":
                            vals = col ** 2
                        elif proj_method == "signed":
                            vals = col
                        else:
                            vals = np.abs(col)
                        for ti in range(min(n_tok, len(tokens))):
                            tok = tokens[ti].strip()
                            if tok:
                                cell_token_data[(si, li)][tok][cat].append(
                                    float(vals[ti]))

                    n_contributions += 1

            # Normalize by number of contributions (layers × projections × probes)
            mask = cell_counts > 0
            cell_grid[mask] /= cell_counts[mask]

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
                g_mean = g["global_mean"]
                g_std = g["global_std"]
                deviation = cell_mean - g_mean
                z_score = abs(deviation) / g_std if g_std > 1e-10 else 0.0
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
                    "n_cells": g["n_cells"],
                    "cats": token_cell_cats[tok].get((si, li), {}),
                })
            token_rows.sort(key=lambda x: x["z"], reverse=True)
            cell_details[cell_key] = {
                "tokens": token_rows[:TOP_N],
                "probes": cell_probes.get((si, li), []),
                "n_unique_tokens": len(token_rows),
            }

        # ── Build output (same format as correction_heatmap for viz compat) ──
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

            # Correction-space heatmap
            "aggregate": aggregate_heatmap.tolist(),

            # Per-cell variance
            "variance": variance_grid.tolist(),

            # Per-cell token specificity
            "cell_details": cell_details,

            # Categories observed
            "categories": {k: cat_names.get(k, k) for k in sorted(all_cats_seen)},

            # Per-subject summary
            "per_subject": {
                subj: {
                    "mean_activation": float(aggregate_heatmap[si].mean()),
                    "max_activation": float(aggregate_heatmap[si].max()),
                    "mean_variance": float(variance_grid[si].mean()),
                }
                for si, subj in enumerate(subjects)
            },

            # Backscatter-specific metadata
            "config": {
                "projection_method": proj_method,
                "delta_projections": delta_mode,
                "n_signal_layers": len(signal_layers),
                "signal_layers": signal_layers,
                "probe_dim": probe_dim,
                "session_dim": session_dim,
                "probes_pre_projected": probes_pre_projected,
            },
        }

        prog(f"Complete: {n_subj} subjects × {n_levels} levels, "
             f"{aggregate_count} prompts projected through "
             f"{len(signal_layers)} signal layers "
             f"({delta_mode} mode, {proj_method} aggregation)")

        return output
