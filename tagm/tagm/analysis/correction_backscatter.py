"""CorrectionBackscatter (ported from TASM).

For each (layer, role) weight delta and each probe and token, computes
`||probe @ dW^T|| / ||dW||_F` (probe energy) and
`||token @ dW^T|| / ||dW||_F` (token energy), summed and averaged across
layers. Backscatter per cell per prompt is token_energy × probe_energy
aggregated (mean/max/sum) across the prompt's tokens, then summarized
per (subject, level) cell.

Emits the JSON shape the UI's `renderCorrectionBackscatterResults`
reads: `subjects`, `subj_short`, `levels`, `aggregate`, `variance`,
`probe_energy_grid`, `decomposition` (per-role), `projection_labels`,
`projection_order`, `per_subject`, `cell_details`, `config`,
`categories`, `prompts`.

Runtime inputs (via the analysis context dict):
  - context["pipeline"].delta_store        — ΔW per (layer, role)
  - context["probe_store"]                 — active ProbeSet lookup
  - context["active_probe_template"]       — set_id + levels ordering

Per-prompt inputs (from session_dict):
  - prompt.measurements.per_token_embedding.objects.per_token_embeddings["final"]
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


# TASM-matching projection role registry and display ordering
_PROJ_LABELS = {
    "q": "Q (query routing)",
    "k": "K (key routing)",
    "v": "V (value content)",
    "o": "O (output mixing)",
    "qk": "QK (attention topology)",
    "qkv": "QKV (full attention)",
}
_PROJ_ORDER = ["v", "q", "k", "o", "qk", "qkv"]
_BASE_ROLES = ["q", "k", "v", "o"]
_COMPOSITES = {"qk": ["q", "k"], "qkv": ["q", "k", "v"]}

_CAT_NAMES = {
    "b": "benign", "m": "mild", "h": "harmful", "j": "jailbreak",
    "a": "adversarial", "d": "dual-use", "u": "unknown",
}


@register_analysis
class CorrectionBackscatter(AnalysisModule):
    name = "correction_backscatter"
    display_name = "Correction Backscatter"
    description = (
        "Projects prompt tokens and probe vocabulary through the ΔW "
        "correction lens at signal layers and measures intensity "
        "(norm), not direction. Produces separate subjects×levels "
        "heatmaps for Q, K, V, O, QK, and QKV projections to reveal "
        "how each component of the correction field responds to each "
        "probe cell."
    )
    version = "1.0.0"

    depends_on_measurements = ("per_token_embedding",)

    parameters = [
        ModuleParameter(
            name="aggregation",
            display_name="Aggregation",
            description=(
                "How to combine per-token backscatter within a prompt. "
                "mean = average across tokens; max = peak; "
                "sum = total energy."
            ),
            kind="select", default="mean",
            options=("mean", "max", "sum"),
        ),
        ModuleParameter(
            name="primary_projection",
            display_name="Primary projection",
            description=(
                "Which projection drives the main aggregate heatmap "
                "and cell-detail probe rankings. All projections are "
                "always computed and shown in the decomposition panel."
            ),
            kind="select", default="qkv",
            options=("v", "q", "k", "o", "qk", "qkv"),
        ),
    ]

    def check_dependencies(self, session):
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
                f"embeddings. Run per_token_embedding with "
                f"include_in_export=True (the default) and retry."
            ]
        return []

    def run(self, session, params, probes=None, context=None):
        aggregation = params.get("aggregation", "mean")
        primary = params.get("primary_projection", "qkv")

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"aggregation": aggregation,
                        "primary_projection": primary},
        )

        # ── Context resources ──
        ctx = context or {}
        pipeline = ctx.get("pipeline")
        delta_store = getattr(pipeline, "delta_store", None) if pipeline else None
        probe_store = ctx.get("probe_store")
        tpl_info = ctx.get("active_probe_template") or {}

        if delta_store is None:
            err = ("Correction backscatter requires the loaded model's "
                   "ΔW store. Load a model pair first.")
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                primary=primary,
                                                aggregation=aggregation))
            return result

        if not probe_store or not tpl_info.get("set_id"):
            err = ("No active probe set. Apply one via the "
                   "Configuration tab before running backscatter.")
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                primary=primary,
                                                aggregation=aggregation))
            return result

        # ── Load ProbeSet ──
        npz_path = probe_store.root / f"{tpl_info['set_id']}.npz"
        if not npz_path.exists():
            err = f"Probe set {tpl_info['set_id']} missing on disk."
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                primary=primary,
                                                aggregation=aggregation))
            return result

        from tagm.probes.artifact import ProbeSet
        try:
            probe_set = ProbeSet.load(npz_path)
        except Exception as e:
            err = f"Could not load probe set: {e}"
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                primary=primary,
                                                aggregation=aggregation))
            return result

        probe_mat, probe_labels_list = probe_set.embeddings_matrix("subject")
        if probe_mat.shape[0] == 0:
            err = "Probe set has no subject-depth embeddings."
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                primary=primary,
                                                aggregation=aggregation))
            return result

        # ── Build subjects / levels from the probe set ──
        subjects: list[str] = []
        probe_cells: list[tuple[int, int]] = []
        cell_probe_texts: dict[tuple[int, int], list[str]] = defaultdict(list)

        if tpl_info.get("levels"):
            levels = list(tpl_info["levels"])
        else:
            levels = []
            for p in probe_set.probes:
                if p.column not in levels:
                    levels.append(p.column)

        for p in probe_set.probes:
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
        n_probes = probe_mat.shape[0]
        subj_short = [s.replace("_", " ").title()[:14] for s in subjects]

        # ── Collect per-prompt token matrices ──
        prompts = session.get("prompts") or []
        n_prompts = len(prompts)
        tok_mats: list[Optional[np.ndarray]] = []
        n_toks: list[int] = []
        cats_list: list[str] = []
        prompts_text: list[str] = []
        all_cats_seen: set[str] = set()

        for p in prompts:
            pte = (p.get("measurements") or {}).get(
                "per_token_embedding") or {}
            pte_emb = ((pte.get("objects") or {}).get(
                "per_token_embeddings") or {}).get("final")
            cat = ((p.get("category") or "unknown")[:1] or "u").lower()
            all_cats_seen.add(cat)
            prompts_text.append((p.get("prompt") or ""))
            cats_list.append(cat)
            if not pte_emb:
                tok_mats.append(None)
                n_toks.append(0)
                continue
            try:
                tm = np.asarray(pte_emb, dtype=np.float32)
            except (TypeError, ValueError):
                tok_mats.append(None)
                n_toks.append(0)
                continue
            if tm.ndim != 2 or tm.shape[0] == 0:
                tok_mats.append(None)
                n_toks.append(0)
                continue
            tokens = p.get("tokens") or []
            tok_mats.append(tm)
            n_toks.append(min(len(tokens), tm.shape[0]))

        # ── Resolve signal layers from delta_store ──
        try:
            delta_layers = sorted(set(delta_store.layers()))
        except Exception:
            delta_layers = []
        if not delta_layers:
            err = "Delta store is empty — no model deltas to project through."
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                primary=primary,
                                                aggregation=aggregation))
            return result

        # ── Compute probe + token energies per base role (q/k/v/o) ──
        # Returns {role: (probe_energies[n_probes], tok_energies[pi][:n_toks[pi]])}
        suffix_energies: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
        for role in _BASE_ROLES:
            energies = _project_for_role(
                role=role,
                delta_store=delta_store,
                layers=delta_layers,
                probe_mat=probe_mat,
                tok_mats=tok_mats,
                n_toks=n_toks,
            )
            if energies is not None:
                suffix_energies[role] = energies

        # Composite projections: average base-role energies
        for comp_key, bases in _COMPOSITES.items():
            avail = [r for r in bases if r in suffix_energies]
            if not avail:
                continue
            pe_sum = np.zeros(n_probes)
            te_sum = [np.zeros(nt) if nt > 0 else np.zeros(0)
                       for nt in n_toks]
            for r in avail:
                pe_r, te_r = suffix_energies[r]
                pe_sum = pe_sum + pe_r
                for pi, te in enumerate(te_r):
                    te_sum[pi] = te_sum[pi] + te
            n = len(avail)
            suffix_energies[comp_key] = (pe_sum / n, [te / n for te in te_sum])

        if not suffix_energies:
            err = ("No ΔW roles were projectable. Make sure the model "
                   "pair has a delta store with q/k/v/o for at least "
                   "one signal layer.")
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                primary=primary,
                                                aggregation=aggregation))
            return result

        # ── Build per-projection heatmap grids ──
        def _build_heatmap(proj_key: str) -> Optional[dict]:
            if proj_key not in suffix_energies:
                return None
            probe_e, tok_e_list = suffix_energies[proj_key]

            cell_pe = np.zeros((n_subj, n_levels))
            cell_pc = np.zeros((n_subj, n_levels))
            for pi, (si, li) in enumerate(probe_cells):
                cell_pe[si, li] += probe_e[pi]
                cell_pc[si, li] += 1
            mask = cell_pc > 0
            cell_pe[mask] /= cell_pc[mask]

            agg = np.zeros((n_subj, n_levels))
            agg_count = 0
            cat_accum: dict[str, dict] = defaultdict(
                lambda: {"grid": np.zeros((n_subj, n_levels)), "count": 0})
            per_prompt: list[Any] = []
            probe_bs_per_prompt: dict[int, list[float]] = {}

            for pi in range(n_prompts):
                te = tok_e_list[pi] if pi < len(tok_e_list) else None
                nt = n_toks[pi]
                if te is None or nt == 0:
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
                agg = agg + grid
                agg_count += 1
                cat = cats_list[pi]
                cat_accum[cat]["grid"] = cat_accum[cat]["grid"] + grid
                cat_accum[cat]["count"] += 1
                per_prompt.append({
                    "prompt_idx": pi,
                    "category": cat,
                    "grid": grid.tolist(),
                })

                # Per-probe backscatter for this prompt
                probe_row = np.zeros(n_probes)
                tok_slice = te[:nt]
                for pi2 in range(n_probes):
                    pe_probe = probe_e[pi2]
                    if pe_probe <= 0:
                        continue
                    bs = tok_slice * pe_probe
                    if aggregation == "max":
                        probe_row[pi2] = float(np.max(bs))
                    elif aggregation == "sum":
                        probe_row[pi2] = float(np.sum(bs))
                    else:
                        probe_row[pi2] = float(np.mean(bs))
                probe_bs_per_prompt[pi] = [round(float(v), 8)
                                            for v in probe_row]

            if agg_count > 0:
                agg = agg / agg_count

            per_cat_out: dict[str, dict] = {}
            for cat, info in cat_accum.items():
                if info["count"] > 0:
                    info["grid"] = info["grid"] / info["count"]
                per_cat_out[cat] = {
                    "aggregate": info["grid"].tolist(),
                    "n_prompts": info["count"],
                }

            return {
                "aggregate": agg.tolist(),
                "probe_energy": cell_pe.tolist(),
                "per_category": per_cat_out,
                "per_prompt": per_prompt,
                "probe_backscatter_per_prompt": probe_bs_per_prompt,
            }

        decomposition: dict[str, dict] = {}
        for pk in _PROJ_ORDER:
            hm = _build_heatmap(pk)
            if hm is not None:
                decomposition[pk] = hm

        # ── Primary projection: resolve or fall back ──
        primary_data = decomposition.get(primary)
        if primary_data is None and decomposition:
            for pk in _PROJ_ORDER:
                if pk in decomposition:
                    primary = pk
                    primary_data = decomposition[pk]
                    break

        if primary_data is None:
            aggregate_heatmap = np.zeros((n_subj, n_levels))
            cell_probe_energy = np.zeros((n_subj, n_levels))
        else:
            aggregate_heatmap = np.asarray(primary_data["aggregate"])
            cell_probe_energy = np.asarray(primary_data["probe_energy"])

        # ── Per-cell variance for primary ──
        variance_grid = np.zeros((n_subj, n_levels))
        if primary in suffix_energies:
            _, tok_e_list = suffix_energies[primary]
            cell_values: dict[tuple[int, int], list[float]] = defaultdict(list)
            for pi in range(n_prompts):
                te = tok_e_list[pi] if pi < len(tok_e_list) else None
                nt = n_toks[pi]
                if te is None or nt == 0:
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

        # ── Per-probe global energy for primary ──
        probe_energy_vectors: dict[str, np.ndarray] = {}
        for pk in _PROJ_ORDER:
            if pk in suffix_energies:
                probe_energy_vectors[pk] = suffix_energies[pk][0]

        primary_pe = probe_energy_vectors.get(primary, np.zeros(n_probes))
        global_probe_mean = (float(np.mean(primary_pe))
                              if n_probes > 0 else 0.0)
        global_probe_std = (float(np.std(primary_pe))
                             if n_probes > 1 else 0.0)

        # ── Cell details: z-ranked probes per cell ──
        cell_details: dict[str, dict] = {}
        for si in range(n_subj):
            for li in range(n_levels):
                cell_key = f"{si}_{li}"
                cell_probe_idx = [i for i, (s, l) in enumerate(probe_cells)
                                   if s == si and l == li]
                if not cell_probe_idx:
                    cell_details[cell_key] = {
                        "probes": [],
                        "probe_energy":
                            round(float(cell_probe_energy[si, li]), 6),
                        "n_probes": 0,
                    }
                    continue
                probe_rows = []
                for pi in cell_probe_idx:
                    energy = float(primary_pe[pi]) if pi < len(primary_pe) \
                        else 0.0
                    dev = energy - global_probe_mean
                    z = (abs(dev) / global_probe_std
                         if global_probe_std > 1e-12 else 0.0)
                    per_proj = {}
                    for pk in _PROJ_ORDER:
                        vec = probe_energy_vectors.get(pk)
                        if vec is not None and pi < len(vec):
                            per_proj[pk] = round(float(vec[pi]), 6)
                    probe_rows.append({
                        "probe_idx": pi,
                        "text": (probe_labels_list[pi]
                                  if pi < len(probe_labels_list) else ""),
                        "energy": round(energy, 6),
                        "global_mean": round(global_probe_mean, 6),
                        "deviation": round(dev, 6),
                        "z": round(z, 3),
                        "projections": per_proj,
                    })
                probe_rows.sort(key=lambda x: x["energy"], reverse=True)

                cell_energies = np.asarray(
                    [primary_pe[pi] for pi in cell_probe_idx if pi < len(primary_pe)])
                cell_details[cell_key] = {
                    "probes": probe_rows,
                    "probe_energy": round(float(cell_probe_energy[si, li]), 6),
                    "n_probes": len(probe_rows),
                    "cell_mean": (round(float(cell_energies.mean()), 6)
                                   if cell_energies.size else 0.0),
                    "cell_std": (round(float(cell_energies.std()), 6)
                                  if cell_energies.size else 0.0),
                    "cell_min": (round(float(cell_energies.min()), 6)
                                  if cell_energies.size else 0.0),
                    "cell_max": (round(float(cell_energies.max()), 6)
                                  if cell_energies.size else 0.0),
                }

        per_subject: dict[str, dict] = {}
        for si, subj in enumerate(subjects):
            per_subject[subj] = {
                "mean_backscatter": float(aggregate_heatmap[si].mean())
                    if n_levels else 0.0,
                "max_backscatter": float(aggregate_heatmap[si].max())
                    if n_levels else 0.0,
                "mean_variance": float(variance_grid[si].mean())
                    if n_levels else 0.0,
                "mean_probe_energy": float(cell_probe_energy[si].mean())
                    if n_levels else 0.0,
            }

        config = {
            "aggregation": aggregation,
            "primary_projection": primary,
            "n_signal_layers": len(delta_layers),
            "signal_layers": list(delta_layers),
            "probe_dim": int(probe_mat.shape[1]),
            "model_dim": int(probe_mat.shape[1]),
            "projections_computed": [pk for pk in _PROJ_ORDER
                                      if pk in decomposition],
            "global_probe_mean": round(global_probe_mean, 6),
            "global_probe_std": round(global_probe_std, 6),
        }

        output = {
            "version": self.version,
            "module": "correction_backscatter",
            "probe_file": (probe_set.template_name or "probe_set") + ".csv",
            "subjects": subjects,
            "subj_short": subj_short,
            "levels": levels,
            "n_prompts": n_prompts,
            "n_prompts_projected": sum(1 for nt in n_toks if nt > 0),
            "n_probes": n_probes,
            "primary_projection": primary,
            "aggregate": aggregate_heatmap.tolist(),
            "probe_energy_grid": cell_probe_energy.tolist(),
            "variance": variance_grid.tolist(),
            "cell_details": cell_details,
            "decomposition": decomposition,
            "projection_labels": _PROJ_LABELS,
            "projection_order": [pk for pk in _PROJ_ORDER
                                  if pk in decomposition],
            "categories": {k: _CAT_NAMES.get(k, k)
                            for k in sorted(all_cats_seen)},
            "prompts": [
                {"idx": pi,
                 "category": cats_list[pi],
                 "text": (prompts_text[pi] or "")[:200],
                 "projected": n_toks[pi] > 0}
                for pi in range(n_prompts)
            ],
            "per_subject": per_subject,
            "config": config,
        }

        result.objects.update(output)
        result.scalars["n_prompts"] = n_prompts
        result.scalars["n_prompts_projected"] = sum(1 for nt in n_toks if nt > 0)
        result.scalars["n_probes"] = n_probes
        result.scalars["n_subjects"] = n_subj
        result.scalars["n_levels"] = n_levels
        return result


def _project_for_role(role, delta_store, layers, probe_mat,
                       tok_mats, n_toks):
    """Sum norm-projections across layers for one role. Returns
    (probe_energies[n_probes], [token_energies[nt] per prompt]) or
    None if no layer had a projectable delta for this role."""
    n_probes = probe_mat.shape[0]
    probe_e = np.zeros(n_probes)
    tok_e: list[np.ndarray] = [np.zeros(nt) if nt > 0 else np.zeros(0)
                                for nt in n_toks]
    contributions = 0

    import torch
    probe_t = torch.from_numpy(probe_mat.astype(np.float32))

    for layer_idx in layers:
        dw = delta_store.get_or_none(layer_idx, role)
        if dw is None:
            continue
        try:
            fnorm = float(delta_store.frob_norm(layer_idx, role))
        except Exception:
            fnorm = 0.0
        if fnorm <= 0:
            continue
        if dw.shape[1] != probe_t.shape[1]:
            continue

        # Probe energies
        proj_p = torch.matmul(probe_t, dw.float().T)
        probe_e = probe_e + (proj_p.norm(dim=-1).cpu().numpy() / fnorm)

        # Token energies
        for pi, (tm, nt) in enumerate(zip(tok_mats, n_toks)):
            if tm is None or nt == 0 or tm.shape[1] != dw.shape[1]:
                continue
            tok_t = torch.from_numpy(tm[:nt].astype(np.float32))
            proj_t = torch.matmul(tok_t, dw.float().T)
            tok_e[pi] = tok_e[pi] + (proj_t.norm(dim=-1).cpu().numpy()
                                      / fnorm)

        contributions += 1

    if contributions == 0:
        return None
    probe_e /= contributions
    tok_e = [te / contributions for te in tok_e]
    return probe_e, tok_e


def _empty_output(error: str, primary: str, aggregation: str) -> dict:
    return {
        "error": error,
        "version": CorrectionBackscatter.version,
        "module": "correction_backscatter",
        "probe_file": "",
        "subjects": [], "subj_short": [], "levels": [],
        "n_prompts": 0, "n_prompts_projected": 0, "n_probes": 0,
        "primary_projection": primary,
        "aggregate": [], "probe_energy_grid": [], "variance": [],
        "cell_details": {}, "decomposition": {},
        "projection_labels": _PROJ_LABELS, "projection_order": [],
        "categories": {}, "prompts": [], "per_subject": {},
        "config": {
            "aggregation": aggregation,
            "primary_projection": primary,
            "n_signal_layers": 0, "signal_layers": [],
            "probe_dim": 0, "model_dim": 0,
            "projections_computed": [],
            "global_probe_mean": 0.0, "global_probe_std": 0.0,
        },
    }
