"""Correction Heatmap analysis (ported from TASM).

This module is a direct port of TASM's
`engine/modules/correction_heatmap.py` into TAGM's analysis plumbing. The
algorithm and output wire format match TASM exactly so the existing
frontend renderer (`renderCorrectionHeatmapResults` in static/js/main.js)
produces the same heatmap it did under TASM.

Process:
  1. Resolve the active probe set from the probe store.
  2. Collect probe embeddings at two depths ("subject" ≈ L50,
     "escalation" ≈ L75) and compute L2-normalized refinement deltas.
  3. For each analyzed prompt, take per-token final-layer hidden states
     (from TAGM's per_token_embedding measurement with include_in_export
     on) and dot-product each token against each probe delta.
  4. Aggregate per (subject, level) cell into a heatmap; track per-cell
     token-level activations for the cell-specificity table.
  5. Compute per-cell variance across prompts (turbulence).
  6. Rank per-cell contributing tokens by z-score of cell-mean vs the
     token's global mean across all cells.

Inputs TAGM must provide:
  • per_token_embedding measurement on every prompt with
    include_in_export=True (this is the default after the rewire; the
    measurement writes objects.per_token_embeddings["final"]).
  • An active probe set in state.probe_store, applied via
    POST /api/probe_set/apply.

Output shape (matches TASM's wire format verbatim):
  {
    "version": str,
    "probe_file": str,
    "subjects": [str, ...],
    "subj_short": [str, ...],
    "levels": [str, ...],
    "n_prompts": int,
    "n_probes": int,
    "aggregate": [[float, ...], ...],      # n_subjects × n_levels
    "variance": [[float, ...], ...],        # n_subjects × n_levels
    "cell_details": {cell_key: {tokens, probes, n_unique_tokens}},
    "categories": {short_code: long_name},
    "per_subject": {subject: {mean_activation, max_activation,
                              mean_variance}},
  }
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


_CAT_NAMES = {
    "b": "benign", "m": "mild", "h": "harmful", "j": "jailbreak",
    "a": "adversarial", "d": "dual-use", "u": "unknown",
}


@register_analysis
class CorrectionHeatmap(AnalysisModule):
    """Subjects × levels correction heatmap produced from probe deltas."""

    name = "correction_heatmap"
    display_name = "Correction Heatmap"
    description = (
        "Projects prompt tokens through probe refinement deltas "
        "(escalation − subject) to produce an aggregate heatmap of "
        "correction-field interaction across subject × subclass cells. "
        "Reveals training-data coverage structure, not per-prompt "
        "categories. Requires an active probe set and per-token "
        "final-layer embeddings on every analyzed prompt."
    )
    version = "1.0.0"

    # Per-prompt requirement: per_token_embedding must populate
    # objects.per_token_embeddings["final"] on every prompt.
    depends_on_measurements = ("per_token_embedding",)

    parameters = [
        ModuleParameter(
            name="projection_method",
            display_name="Projection Method",
            description=(
                "How to measure interaction intensity. abs = linear "
                "magnitude; squared = energy; signed = raw directional "
                "(positive = aligned with refinement, negative = opposed)."
            ),
            kind="select",
            default="abs",
            options=("abs", "squared", "signed"),
        ),
        ModuleParameter(
            name="subject_depth",
            display_name="Subject depth label",
            description=(
                "ProbeSet depth used as the subject-layer embedding. "
                "Normally 'subject' (the L50 depth the probe generator "
                "assigns by default)."
            ),
            kind="string", default="subject",
        ),
        ModuleParameter(
            name="escalation_depth",
            display_name="Escalation depth label",
            description=(
                "ProbeSet depth used as the escalation-layer embedding. "
                "Normally 'escalation' (L75 by default)."
            ),
            kind="string", default="escalation",
        ),
    ]

    # ──────────────────────────────────────────────────────────────
    # Main entrypoint
    # ──────────────────────────────────────────────────────────────
    def run(self, session, params, probes=None, context=None):
        proj_method = params.get("projection_method", "abs")
        subject_depth = params.get("subject_depth", "subject")
        escalation_depth = params.get("escalation_depth", "escalation")

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={
                "projection_method": proj_method,
                "subject_depth": subject_depth,
                "escalation_depth": escalation_depth,
            },
        )

        # ── Resolve active probe set from context ──
        ps_info = self._resolve_active_probe_set(context)
        if "error" in ps_info:
            result.warnings.append(ps_info["error"])
            # Still emit the UI-expected wire format with an error field
            # so the renderer shows a helpful card instead of nothing.
            result.objects.update(_empty_output(error=ps_info["error"]))
            return result

        probe_set = ps_info["probe_set"]
        template_name = probe_set.template_name or "probe_set"

        # ── Pull subject/escalation embeddings ──
        subj_mat, subj_labels = probe_set.embeddings_matrix(subject_depth)
        esc_mat, esc_labels = probe_set.embeddings_matrix(escalation_depth)

        if subj_mat.shape[0] == 0 or esc_mat.shape[0] == 0:
            msg = (f"Active probe set has no embeddings at depths "
                   f"({subject_depth!r}, {escalation_depth!r}). "
                   f"Re-apply the probe set with those depths configured.")
            result.warnings.append(msg)
            result.objects.update(_empty_output(error=msg))
            return result

        if subj_labels != esc_labels:
            # The generator stores probes in a consistent order, so this
            # shouldn't happen — but flag defensively.
            result.warnings.append(
                "Subject and escalation embedding orders diverge; using "
                "subject order as canonical and re-aligning.")
            # Align by label
            esc_by_label = {
                lbl: esc_mat[i] for i, lbl in enumerate(esc_labels)
            }
            aligned = []
            for lbl in subj_labels:
                if lbl in esc_by_label:
                    aligned.append(esc_by_label[lbl])
                else:
                    aligned.append(np.zeros(esc_mat.shape[1],
                                             dtype=esc_mat.dtype))
            esc_mat = np.stack(aligned) if aligned else esc_mat

        deltas = (esc_mat - subj_mat).astype(np.float32)
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        deltas_n = deltas / norms                               # (n_probes, hidden)

        # Build subjects list preserving first-appearance order across probes
        subjects: list[str] = []
        probe_cells: list[tuple[int, int]] = []
        cell_probe_texts: dict[tuple[int, int], list[str]] = defaultdict(list)

        # Derive levels from the union of probe.column values encountered
        # (in first-appearance order), or from the active_template if
        # given — prefer the template since its ordering is canonical.
        tpl_info = context.get("active_probe_template") if context else None
        if tpl_info and tpl_info.get("levels"):
            levels = list(tpl_info["levels"])
        else:
            levels = []
            for p in probe_set.probes:
                if p.column not in levels:
                    levels.append(p.column)

        for p, label in zip(probe_set.probes, subj_labels):
            if p.row not in subjects:
                subjects.append(p.row)
            si = subjects.index(p.row)
            try:
                li = levels.index(p.column)
            except ValueError:
                levels.append(p.column)
                li = len(levels) - 1
            probe_cells.append((si, li))
            cell_probe_texts[(si, li)].append(p.label)

        n_subj = len(subjects)
        n_levels = len(levels)
        n_probes = len(probe_set.probes)
        subj_short = [s.replace("_", " ").title()[:14] for s in subjects]

        # ── Project prompt tokens through deltas ──
        prompts = session.get("prompts") or []

        per_prompt_heatmaps: list[Optional[np.ndarray]] = []
        aggregate_heatmap = np.zeros((n_subj, n_levels), dtype=np.float64)
        aggregate_count = 0

        # Per-cell, per-token activation tracking for cell-specificity
        # z-scores: cell_token_data[(si,li)][token][cat_short] = [values]
        cell_token_data: dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))

        missing_emb = 0
        for pi, p in enumerate(prompts):
            pte = ((p.get("measurements") or {}).get("per_token_embedding")
                   or {})
            pte_objs = pte.get("objects") or {}
            pte_emb_dict = pte_objs.get("per_token_embeddings") or {}
            final_emb = pte_emb_dict.get("final")

            if not final_emb:
                per_prompt_heatmaps.append(None)
                missing_emb += 1
                continue

            tokens = p.get("tokens") or []
            cat_short = ((p.get("category") or "unknown")[:1] or "u").lower()

            try:
                tok_mat = np.array(final_emb, dtype=np.float32)
            except (TypeError, ValueError):
                per_prompt_heatmaps.append(None)
                missing_emb += 1
                continue
            if tok_mat.ndim != 2 or tok_mat.shape[0] == 0:
                per_prompt_heatmaps.append(None)
                missing_emb += 1
                continue

            # Validate hidden-dim match. If it doesn't match, that means
            # the session was analyzed under a different model than the
            # probe set was embedded for — skip with a warning.
            if tok_mat.shape[1] != deltas_n.shape[1]:
                if pi == 0:
                    result.warnings.append(
                        f"Token/embedding hidden-dim mismatch: "
                        f"prompts have dim={tok_mat.shape[1]}, probe "
                        f"deltas have dim={deltas_n.shape[1]}. Re-apply "
                        f"probe set under the currently loaded model.")
                per_prompt_heatmaps.append(None)
                missing_emb += 1
                continue

            # Projections: (n_tokens, n_probes)
            projections = tok_mat @ deltas_n.T

            cell_grid = np.zeros((n_subj, n_levels), dtype=np.float64)
            cell_counts = np.zeros((n_subj, n_levels), dtype=np.float64)

            n_tok = min(len(tokens), tok_mat.shape[0])
            cell_tok_vals: dict[tuple[int, int], np.ndarray] = defaultdict(
                lambda: np.zeros(n_tok, dtype=np.float64))
            cell_tok_counts: dict[tuple[int, int], int] = defaultdict(int)

            for probe_idx, (si, li) in enumerate(probe_cells):
                col = projections[:n_tok, probe_idx]
                if proj_method == "squared":
                    v = col ** 2
                elif proj_method == "signed":
                    v = col
                else:
                    v = np.abs(col)
                cell_grid[si, li] += float(v.mean()) if v.size else 0.0
                cell_tok_vals[(si, li)] = cell_tok_vals[(si, li)] + v
                cell_counts[si, li] += 1
                cell_tok_counts[(si, li)] += 1

            mask = cell_counts > 0
            cell_grid[mask] /= cell_counts[mask]

            for (si, li), vals in cell_tok_vals.items():
                cnt = cell_tok_counts[(si, li)]
                if cnt > 0:
                    vals = vals / cnt
                for ti in range(n_tok):
                    tok = (tokens[ti] or "").strip()
                    if tok:
                        cell_token_data[(si, li)][tok][cat_short].append(
                            float(vals[ti]))

            per_prompt_heatmaps.append(cell_grid)
            aggregate_heatmap += cell_grid
            aggregate_count += 1

        if aggregate_count > 0:
            aggregate_heatmap /= aggregate_count

        if missing_emb:
            result.warnings.append(
                f"{missing_emb} prompt(s) had no per-token final "
                f"embeddings; they were skipped. Make sure the "
                f"per_token_embedding measurement ran with "
                f"include_in_export=True on those prompts.")

        # ── Per-cell variance across prompts ──
        cell_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        for hm in per_prompt_heatmaps:
            if hm is None:
                continue
            for si in range(n_subj):
                for li in range(n_levels):
                    cell_values[(si, li)].append(float(hm[si, li]))

        variance_grid = np.zeros((n_subj, n_levels), dtype=np.float64)
        for (si, li), vals in cell_values.items():
            if len(vals) > 1:
                variance_grid[si, li] = float(np.var(vals))

        # ── Cell-specificity token ranking ──
        TOP_N = 10
        all_cats_seen: set[str] = set()

        token_cell_means: dict[str, dict[tuple[int, int], float]] = \
            defaultdict(dict)
        token_cell_cats: dict[str, dict[tuple[int, int], dict]] = \
            defaultdict(dict)
        for (si, li), tok_dict in cell_token_data.items():
            for tok, cat_dict in tok_dict.items():
                all_vals = []
                per_cat: dict[str, dict] = {}
                for cat_short, vals in cat_dict.items():
                    all_cats_seen.add(cat_short)
                    m = float(np.mean(vals))
                    per_cat[cat_short] = {
                        "mean": round(m, 6), "n": len(vals)}
                    all_vals.extend(vals)
                token_cell_means[tok][(si, li)] = float(np.mean(all_vals))
                token_cell_cats[tok][(si, li)] = per_cat

        token_global: dict[str, dict] = {}
        for tok, cell_dict in token_cell_means.items():
            means = list(cell_dict.values())
            token_global[tok] = {
                "global_mean": float(np.mean(means)),
                "global_std": (float(np.std(means))
                               if len(means) > 1 else 0.0),
                "n_cells": len(means),
            }

        cell_details: dict[str, dict] = {}
        for (si, li), tok_dict in cell_token_data.items():
            cell_key = f"{si}_{li}"
            rows = []
            for tok in tok_dict:
                cm = token_cell_means[tok].get((si, li), 0.0)
                g = token_global[tok]
                g_mean, g_std = g["global_mean"], g["global_std"]
                dev = cm - g_mean
                z = abs(dev) / g_std if g_std > 1e-10 else 0.0
                all_vals = []
                for vals in tok_dict[tok].values():
                    all_vals.extend(vals)
                rows.append({
                    "token": tok,
                    "cell_mean": round(cm, 6),
                    "global_mean": round(g_mean, 6),
                    "deviation": round(dev, 6),
                    "z": round(z, 3),
                    "n": len(all_vals),
                    "n_cells": g["n_cells"],
                    "cats": token_cell_cats[tok].get((si, li), {}),
                })
            rows.sort(key=lambda x: x["z"], reverse=True)
            cell_details[cell_key] = {
                "tokens": rows[:TOP_N],
                "probes": list(cell_probe_texts.get((si, li), [])),
                "n_unique_tokens": len(rows),
            }

        # ── Build output in TASM wire format ──
        per_subject_stats: dict[str, dict] = {}
        agg = aggregate_heatmap
        for si, subj in enumerate(subjects):
            row = agg[si]
            per_subject_stats[subj] = {
                "mean_activation": (float(row.mean())
                                     if row.size else 0.0),
                "max_activation": (float(row.max())
                                    if row.size else 0.0),
                "mean_variance": (float(variance_grid[si].mean())
                                   if variance_grid[si].size else 0.0),
            }

        tasm_shape: dict[str, Any] = {
            "version": self.version,
            "probe_file": template_name + ".csv",
            "subjects": subjects,
            "subj_short": subj_short,
            "levels": levels,
            "n_prompts": aggregate_count,
            "n_probes": int(n_probes),
            "aggregate": aggregate_heatmap.tolist(),
            "variance": variance_grid.tolist(),
            "cell_details": cell_details,
            "categories": {
                k: _CAT_NAMES.get(k, k) for k in sorted(all_cats_seen)
            },
            "per_subject": per_subject_stats,
        }

        # Pack into AnalysisResult. Store the whole TASM-shape payload
        # under .objects (TAGM's container for non-scalar data); the
        # /api/modules/{name}/results projection hoists it to top-level
        # for the UI via modules_runner's shape-flattening step.
        result.objects.update(tasm_shape)
        result.scalars["n_prompts"] = aggregate_count
        result.scalars["n_probes"] = int(n_probes)
        result.scalars["n_subjects"] = n_subj
        result.scalars["n_levels"] = n_levels

        return result

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def check_dependencies(self, session):
        # Skip the strict per-measurement dependency check:
        # per_token_embedding populates per_token_embeddings["final"]
        # only when include_in_export=True (the new default). Validate
        # loosely instead: need >=1 prompt with the final embedding
        # available. Per-prompt skipping handles any gaps.
        prompts = session.get("prompts") or []
        if not prompts:
            return [f"Analysis '{self.name}' needs at least one prompt."]
        any_final = False
        for p in prompts:
            pte = (p.get("measurements") or {}).get("per_token_embedding") or {}
            objs = pte.get("objects") or {}
            if (objs.get("per_token_embeddings") or {}).get("final"):
                any_final = True
                break
        if not any_final:
            return [
                f"Analysis '{self.name}' requires per-token final "
                f"embeddings. Run per_token_embedding on the prompts "
                f"with include_in_export=True (the default) and retry."
            ]
        return []

    def _resolve_active_probe_set(self, context) -> dict:
        """Find the active probe set via context. Returns either
        {'probe_set': ProbeSet} or {'error': str}."""
        if not context:
            return {"error": (
                "No runtime context available to locate the probe store. "
                "correction_heatmap needs an applied probe set "
                "(Configuration → Probe Set → Apply).")}

        tpl = context.get("active_probe_template")
        store = context.get("probe_store")
        if store is None:
            return {"error": "Probe store not available in runtime context."}
        if not tpl:
            return {"error": (
                "No active probe set. Apply one via the Configuration "
                "tab before running correction_heatmap.")}

        set_id = tpl.get("set_id")
        if not set_id:
            return {"error": "Active probe template has no set_id."}

        # Look up the probe set from the store.
        npz_path = store.root / f"{set_id}.npz"
        if not npz_path.exists():
            return {"error": (
                f"Active probe set {set_id!r} is not present on disk. "
                f"Re-apply the probe set.")}

        from tagm.probes.artifact import ProbeSet
        try:
            probe_set = ProbeSet.load(npz_path)
        except Exception as e:
            return {"error": f"Could not load probe set {set_id!r}: {e}"}

        return {"probe_set": probe_set}


def _empty_output(error: str) -> dict:
    """Emit UI-expected fields with empty arrays so the renderer displays
    an error message rather than a blank card."""
    return {
        "version": CorrectionHeatmap.version,
        "probe_file": "",
        "subjects": [],
        "subj_short": [],
        "levels": [],
        "n_prompts": 0,
        "n_probes": 0,
        "aggregate": [],
        "variance": [],
        "cell_details": {},
        "categories": {},
        "per_subject": {},
        "error": error,
    }
