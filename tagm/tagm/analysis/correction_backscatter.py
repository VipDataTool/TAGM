"""Correction Backscatter: delta-decomposed energy coupling.

Ported from TASM's engine/modules/correction_backscatter.py. Projects
probe embeddings and per-token embeddings through weight deltas (Q/K/V/O)
to measure correction field coupling. Output shape matches TASM renderer.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")

PROJ_TYPES = {
    "q": ["q"], "k": ["k"], "v": ["v"], "o": ["o"],
    "qk": ["q", "k"], "qkv": ["q", "k", "v"],
}
PROJ_LABELS = {
    "q": "Q (query routing)", "k": "K (key routing)",
    "v": "V (value content)", "o": "O (output mixing)",
    "qk": "QK (attention topology)", "qkv": "QKV (full attention)",
}
PROJ_ORDER = ["v", "q", "k", "o", "qk", "qkv"]


def _get_final_emb(r):
    pte = ((r.get("measurements") or {}).get("per_token_embedding") or {})
    return (pte.get("objects") or {}).get("per_token_embeddings", {}).get("final")


@register_analysis
class CorrectionBackscatter(AnalysisModule):
    name = "correction_backscatter"
    display_name = "Correction Backscatter"
    description = (
        "Delta-decomposed energy coupling between prompt tokens and probe "
        "vocabulary through Q/K/V/O weight deltas."
    )
    version = "1.0.0"
    min_results = 1

    depends_on_measurements = ("per_token_embedding",)

    parameters = [
        ModuleParameter(name="aggregation", display_name="Aggregation",
                        description="How to combine per-prompt results.",
                        kind="select", default="mean",
                        options=("mean", "max", "median")),
        ModuleParameter(name="primary_projection", display_name="Primary Projection",
                        description="Which projection type to use as primary.",
                        kind="select", default="qkv",
                        options=("v", "q", "k", "o", "qk", "qkv")),
    ]

    def __init__(self):
        self._pipeline = None
        self._probe_store = None

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline

    def set_probe_store(self, probe_store):
        self._probe_store = probe_store

    def check_dependencies(self, session: dict) -> list[str]:
        errors = super().check_dependencies(session)
        if self._pipeline is None:
            errors.append("Backscatter requires a loaded model (for delta weights).")
        if self._probe_store is None:
            errors.append("Backscatter requires a probe store.")
        return errors

    def _compute_energies(self, role, layers, delta_store, probe_mat, tok_mats, n_toks):
        """Compute probe and token energies for one delta role."""
        n_probes = probe_mat.shape[0]
        probe_energies = np.zeros(n_probes)
        token_energies_list = [np.zeros(nt) for nt in n_toks]
        contributions = 0

        for layer_idx in layers:
            try:
                dw = delta_store.get(layer_idx, role)
            except (KeyError, RuntimeError):
                continue
            fnorm = float(torch.linalg.norm(dw).item())
            if fnorm <= 0:
                continue
            dw_np = dw.float().cpu().numpy()
            if probe_mat.shape[1] != dw_np.shape[1]:
                continue

            proj_p = probe_mat @ dw_np.T
            probe_energies += np.linalg.norm(proj_p, axis=1) / fnorm

            for pi, (tm, nt) in enumerate(zip(tok_mats, n_toks)):
                if tm is None or nt == 0 or tm.shape[1] != dw_np.shape[1]:
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

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []
        aggregation = params.get("aggregation", "mean")
        primary = params.get("primary_projection", "qkv")

        probe_set = self._get_active_probe_set()
        if probe_set is None:
            return {"error": "No active probe set."}

        delta_store = self._pipeline.delta_store
        layers = delta_store.layers()
        if not layers:
            return {"error": "No delta weights available."}

        subjects = list(dict.fromkeys(p.row for p in probe_set.probes))
        subclasses = list(dict.fromkeys(p.column for p in probe_set.probes))
        n_subj = len(subjects)
        n_levels = len(subclasses)
        subj_idx = {s: i for i, s in enumerate(subjects)}
        level_idx = {l: i for i, l in enumerate(subclasses)}

        # Get probe embedding matrix at subject depth
        subj_mat, _ = probe_set.embeddings_matrix(probe_set.depth_labels[0])
        probe_mat = subj_mat.astype(np.float32)
        n_probes = probe_mat.shape[0]

        probe_cells = []
        for p in probe_set.probes:
            probe_cells.append((subj_idx.get(p.row, 0), level_idx.get(p.column, 0)))

        # Load token matrices
        tok_mats = []
        n_toks = []
        cats_list = []
        prompts_list = []
        cat_names = {"b": "benign", "m": "mild", "h": "harmful",
                     "j": "jailbreak", "a": "adversarial", "d": "dual-use", "u": "unknown"}
        all_cats_seen = set()

        for r in prompts:
            fe = _get_final_emb(r)
            if fe and len(fe) > 0:
                tm = np.array(fe, dtype=np.float32)
                tokens = r.get("tokens", [])
                nt = min(len(tokens), tm.shape[0])
                tok_mats.append(tm)
                n_toks.append(nt)
                cat = ((r.get("category") or "unknown")[:1]).lower()
                cats_list.append(cat)
                all_cats_seen.add(cat)
                prompts_list.append((r.get("prompt") or "")[:200])
            else:
                tok_mats.append(None)
                n_toks.append(0)
                cats_list.append("u")
                prompts_list.append((r.get("prompt") or "")[:200])

        n_prompts = len(prompts)

        # Compute per-role energies
        role_energies = {}
        for role in ["q", "k", "v", "o"]:
            result = self._compute_energies(role, layers, delta_store,
                                             probe_mat, tok_mats, n_toks)
            if result is not None:
                role_energies[role] = result

        # Composites
        def _composite(keys):
            available = [k for k in keys if k in role_energies]
            if not available:
                return None
            probe_sum = np.zeros(n_probes)
            tok_sums = [np.zeros(nt) for nt in n_toks]
            for k in available:
                pe, te_list = role_energies[k]
                probe_sum += pe
                for pi, te in enumerate(te_list):
                    tok_sums[pi] += te
            n = len(available)
            return probe_sum / n, [ts / n for ts in tok_sums]

        role_energies["qk"] = _composite(["q", "k"])
        role_energies["qkv"] = _composite(["q", "k", "v"])

        # Build heatmaps per projection type
        def _build_heatmap(proj_key):
            if proj_key not in role_energies or role_energies[proj_key] is None:
                return None
            pe, te_list = role_energies[proj_key]
            cell_pe = np.zeros((n_subj, n_levels))
            cell_pe_cnt = np.zeros((n_subj, n_levels))
            for pi_probe, (si, li) in enumerate(probe_cells):
                cell_pe[si, li] += pe[pi_probe]
                cell_pe_cnt[si, li] += 1
            mask = cell_pe_cnt > 0
            cell_pe[mask] /= cell_pe_cnt[mask]

            per_prompt_grids = []
            for pi in range(n_prompts):
                if n_toks[pi] == 0:
                    per_prompt_grids.append(None)
                    continue
                te = te_list[pi]
                grid = np.zeros((n_subj, n_levels))
                for ci, (si, li) in enumerate(probe_cells):
                    if ci < len(pe):
                        grid[si, li] += float(np.mean(te[:n_toks[pi]])) * pe[ci]
                per_prompt_grids.append(grid)

            valid = [g for g in per_prompt_grids if g is not None]
            if not valid:
                return None
            stacked = np.stack(valid)
            agg = np.mean(stacked, axis=0)
            var = np.var(stacked, axis=0) if len(valid) > 1 else np.zeros_like(agg)
            return {"aggregate": agg.tolist(), "variance": var.tolist(),
                    "probe_energy": cell_pe.tolist()}

        decomposition = {}
        for pk in PROJ_ORDER:
            result = _build_heatmap(pk)
            if result is not None:
                decomposition[pk] = result

        # Primary heatmap
        primary_data = decomposition.get(primary, {})
        aggregate = np.array(primary_data.get("aggregate", np.zeros((n_subj, n_levels)).tolist()))
        variance = np.array(primary_data.get("variance", np.zeros((n_subj, n_levels)).tolist()))
        cell_pe = np.array(primary_data.get("probe_energy", np.zeros((n_subj, n_levels)).tolist()))

        subj_short = [s.replace("_", " ").title()[:14] for s in subjects]
        level_names = [l.replace("_", " ") for l in subclasses]

        return {
            "version": self.version,
            "module": "correction_backscatter",
            "probe_file": probe_set.template_name,
            "subjects": subjects,
            "subj_short": subj_short,
            "levels": level_names,
            "n_prompts": n_prompts,
            "n_prompts_projected": sum(1 for nt in n_toks if nt > 0),
            "n_probes": n_probes,
            "primary_projection": primary,
            "aggregate": aggregate.tolist() if isinstance(aggregate, np.ndarray) else aggregate,
            "probe_energy_grid": cell_pe.tolist() if isinstance(cell_pe, np.ndarray) else cell_pe,
            "variance": variance.tolist() if isinstance(variance, np.ndarray) else variance,
            "cell_details": {},
            "decomposition": decomposition,
            "projection_labels": PROJ_LABELS,
            "projection_order": [pk for pk in PROJ_ORDER if pk in decomposition],
            "categories": {k: cat_names.get(k, k) for k in sorted(all_cats_seen)},
            "prompts": [
                {"idx": pi, "category": cats_list[pi],
                 "text": prompts_list[pi], "projected": n_toks[pi] > 0}
                for pi in range(n_prompts)
            ],
            "per_subject": {
                subj: {
                    "mean_backscatter": float(aggregate[si].mean()) if isinstance(aggregate, np.ndarray) else 0,
                    "max_backscatter": float(aggregate[si].max()) if isinstance(aggregate, np.ndarray) else 0,
                    "mean_variance": float(variance[si].mean()) if isinstance(variance, np.ndarray) else 0,
                    "mean_probe_energy": float(cell_pe[si].mean()) if isinstance(cell_pe, np.ndarray) else 0,
                }
                for si, subj in enumerate(subjects)
            },
            "config": {
                "aggregation": aggregation,
                "primary_projection": primary,
                "n_signal_layers": len(layers),
                "signal_layers": layers,
                "projections_computed": [pk for pk in PROJ_ORDER if pk in decomposition],
            },
        }

    def _get_active_probe_set(self):
        if self._probe_store is None:
            return None
        sets = self._probe_store.list()
        if not sets:
            return None
        return self._probe_store.get_by_id(sets[-1]["set_id"])
