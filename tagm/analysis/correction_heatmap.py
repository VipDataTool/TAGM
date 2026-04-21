"""Correction Heatmap: per-cell aggregate of correction field interaction.

Ported from TASM's engine/modules/correction_heatmap.py. Projects prompt
tokens through probe refinement deltas to produce aggregate heatmaps.
Data reading adapted to TAGM's ProbeStore + native measurement schema.
Output shape matches TASM so renderCorrectionHeatmapResults works unchanged.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")

TOP_N = 15


def _get_final_emb(r):
    pte = ((r.get("measurements") or {}).get("per_token_embedding") or {})
    return (pte.get("objects") or {}).get("per_token_embeddings", {}).get("final")


@register_analysis
class CorrectionHeatmap(AnalysisModule):
    name = "correction_heatmap"
    display_name = "Correction Heatmap"
    description = (
        "Projects prompt tokens through probe refinement deltas "
        "(escalation - subject depth) to produce an aggregate heatmap "
        "of correction field interaction across subject × subclass cells."
    )
    version = "1.0.0"
    min_results = 1

    depends_on_measurements = ("per_token_embedding",)

    parameters = [
        ModuleParameter(name="aggregation", display_name="Aggregation",
                        description="How to combine per-prompt heatmaps.",
                        kind="select", default="mean",
                        options=("mean", "max", "median")),
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
        if self._probe_store is None:
            errors.append("Correction Heatmap requires a probe store.")
        return errors

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []
        aggregation = params.get("aggregation", "mean")

        # Find active probe set
        probe_set = self._get_active_probe_set()
        if probe_set is None:
            return {"error": "No active probe set. Generate probes first."}

        subjects = list(dict.fromkeys(p.row for p in probe_set.probes))
        subclasses = list(dict.fromkeys(p.column for p in probe_set.probes))
        n_subj = len(subjects)
        n_levels = len(subclasses)
        subj_idx = {s: i for i, s in enumerate(subjects)}
        level_idx = {l: i for i, l in enumerate(subclasses)}

        # Get probe embeddings at subject and escalation depths
        depth_labels = probe_set.depth_labels
        if len(depth_labels) < 2:
            return {"error": "Probe set needs at least 2 depths (subject + escalation)."}

        subj_depth = depth_labels[0]
        esc_depth = depth_labels[1]

        subj_mat, subj_labels = probe_set.embeddings_matrix(subj_depth)
        esc_mat, esc_labels = probe_set.embeddings_matrix(esc_depth)

        if subj_mat.shape[0] == 0 or esc_mat.shape[0] == 0:
            return {"error": "Probe embeddings missing at required depths."}

        # Compute probe deltas: escalation - subject
        delta_mat = esc_mat - subj_mat  # (n_probes, hidden_size)

        # Map probes to grid cells
        probe_cell_map = []
        for p in probe_set.probes:
            si = subj_idx.get(p.row, 0)
            li = level_idx.get(p.column, 0)
            probe_cell_map.append((si, li))

        # Process prompts: project tokens through probe deltas
        per_prompt_grids = []
        all_cats_seen = set()
        cat_names = {"b": "benign", "m": "mild", "h": "harmful",
                     "j": "jailbreak", "a": "adversarial",
                     "d": "dual-use", "u": "unknown"}
        n_prompts_projected = 0

        for r in prompts:
            fe = _get_final_emb(r)
            if fe is None or len(fe) == 0:
                per_prompt_grids.append(None)
                continue

            emb_arr = np.array(fe, dtype=np.float32)
            tokens = r.get("tokens", [])
            n_tok = min(len(tokens), emb_arr.shape[0])
            cat = ((r.get("category") or "unknown")[:1]).lower()
            all_cats_seen.add(cat)

            # Dot product: each token against each probe delta
            # (n_tok, hidden) @ (n_probes, hidden).T = (n_tok, n_probes)
            if emb_arr.shape[1] != delta_mat.shape[1]:
                per_prompt_grids.append(None)
                continue

            scores = emb_arr[:n_tok] @ delta_mat.T  # (n_tok, n_probes)

            # Aggregate into grid cells
            grid = np.zeros((n_subj, n_levels))
            grid_count = np.zeros((n_subj, n_levels))
            for pi, (si, li) in enumerate(probe_cell_map):
                cell_scores = scores[:, pi]  # all tokens for this probe
                grid[si, li] += float(np.mean(np.abs(cell_scores)))
                grid_count[si, li] += 1

            mask = grid_count > 0
            grid[mask] /= grid_count[mask]

            per_prompt_grids.append(grid)
            n_prompts_projected += 1

        # Aggregate across prompts
        valid_grids = [g for g in per_prompt_grids if g is not None]
        if not valid_grids:
            return {"error": "No prompts with token embeddings."}

        stacked = np.stack(valid_grids)
        if aggregation == "max":
            aggregate = np.max(stacked, axis=0)
        elif aggregation == "median":
            aggregate = np.median(stacked, axis=0)
        else:
            aggregate = np.mean(stacked, axis=0)

        variance = np.var(stacked, axis=0) if len(valid_grids) > 1 else np.zeros_like(aggregate)

        subj_short = [s.replace("_", " ").title()[:14] for s in subjects]
        level_names = [l.replace("_", " ") for l in subclasses]

        return {
            "version": self.version,
            "probe_file": probe_set.template_name,
            "subjects": subjects,
            "subj_short": subj_short,
            "levels": level_names,
            "n_prompts": len(prompts),
            "n_probes": len(probe_set.probes),
            "aggregate": aggregate.tolist(),
            "variance": variance.tolist(),
            "cell_details": {},
            "categories": {k: cat_names.get(k, k) for k in sorted(all_cats_seen)},
            "per_subject": {
                subj: {
                    "mean_activation": float(aggregate[si].mean()),
                    "max_activation": float(aggregate[si].max()),
                    "mean_variance": float(variance[si].mean()),
                }
                for si, subj in enumerate(subjects)
            },
        }

    def _get_active_probe_set(self):
        if self._probe_store is None:
            return None
        sets = self._probe_store.list()
        if not sets:
            return None
        # Use most recent probe set
        latest = sets[-1]
        return self._probe_store.get_by_id(latest["set_id"])
