"""MI Instrumentation (ported from TASM).

Direct port of TASM's `engine/modules/mi_instrumentation.py`.
Emits the JSON shape `renderMIInstrumentationResults` in
static/js/main.js reads:

  - summary: {refusal_separates, stress_separates, refusal_auroc,
              stress_auroc, random_mean, best_layer_auroc, best_layer}
  - refusal_direction: {auroc, stress_auroc, refusal_norm,
                         mean_cosine_safe/risk, std_cosine_safe/risk,
                         separation, per_category}
  - per_layer_auroc: {layers: [{layer, auroc, mean_safe, mean_risk, ratio}]}
  - patching_priority: {top_points, layer_marginal, matrix, layers}
  - random_projection: {random_mean_auroc, random_std_auroc, refusal_direction, ...}
  - auroc, delta_over_random, empirical_p, n_prompts, n_safe, n_risk, hidden_dim

Inputs from TAGM measurements (per prompt):
  per_token_embedding.objects.per_token_embeddings["final"]   — final embs
  stress_score.scalars.stress_mean                             — scalar stress
  stress_score.per_token.stress                                — per-token stress
  last_position_attribution.per_layer_per_token
      .signed_attribution_to_last_by_layer                     — {layer: [per-tok]}
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


def _safe_auroc(scores, labels):
    if len(scores) < 4:
        return 0.5
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_s = scores[pos]
    neg_s = scores[neg]
    u = 0.0
    for p in pos_s:
        u += float((neg_s < p).sum()) + 0.5 * float((neg_s == p).sum())
    return float(u / (n_pos * n_neg))


@register_analysis
class MIInstrumentation(AnalysisModule):
    name = "mi_instrumentation"
    display_name = "MI Instrumentation"
    description = (
        "Produces MI-relevant measurements from session data: "
        "refusal-direction extraction, activation-patching priority "
        "map, per-layer AUROC curve, and random-projection control. "
        "Requires per-token final embeddings and per-layer signed "
        "attribution."
    )
    version = "1.0.0"

    depends_on_measurements = ("per_token_embedding", "stress_score",
                                "last_position_attribution")

    parameters = [
        ModuleParameter(
            name="safe_categories",
            display_name="Safe categories",
            description="Comma-separated list of categories treated as safe (class 0).",
            kind="string", default="benign,mild",
        ),
        ModuleParameter(
            name="n_random_trials",
            display_name="Random projection trials",
            description="Number of random directions for the projection control.",
            kind="int", default=50, min_value=1, max_value=500,
        ),
        ModuleParameter(
            name="random_seed",
            display_name="Random seed",
            description="Seed for reproducible random projections.",
            kind="int", default=42, min_value=0, max_value=2**31-1,
            advanced=True,
        ),
        ModuleParameter(
            name="max_patching_tokens",
            display_name="Max patching-map tokens",
            description="Max token positions in the patching priority map.",
            kind="int", default=30, min_value=4, max_value=200,
        ),
    ]

    def run(self, session, params, probes=None, context=None):
        safe_cats = set(c.strip().lower() for c in
                         str(params.get("safe_categories", "benign,mild"))
                            .split(",") if c.strip())
        n_random = int(params.get("n_random_trials", 50))
        seed = int(params.get("random_seed", 42))
        max_patch = int(params.get("max_patching_tokens", 30))
        rng = np.random.RandomState(seed)

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={
                "safe_categories": sorted(safe_cats),
                "n_random_trials": n_random,
                "random_seed": seed,
                "max_patching_tokens": max_patch,
            },
        )

        risk_cats = {"harmful", "jailbreak", "adversarial",
                     "dual-use"} - safe_cats
        all_cats = safe_cats | risk_cats

        prompts = session.get("prompts") or []

        embeddings_list: list[np.ndarray] = []
        labels_list: list[int] = []
        categories: list[str] = []
        stress_scores: list[float] = []
        layer_attrs: dict[int, list[float]] = defaultdict(list)
        token_attrs: list[tuple[int, list[float], list[str]]] = []
        signal_layers: Optional[list[int]] = None
        n_skipped = 0

        for p in prompts:
            cat = (p.get("category") or "").lower().strip()
            if cat not in all_cats:
                n_skipped += 1
                continue

            meas = p.get("measurements") or {}
            pte_objs = (meas.get("per_token_embedding") or {}).get("objects") or {}
            emb = (pte_objs.get("per_token_embeddings") or {}).get("final")
            if not emb:
                n_skipped += 1
                continue

            try:
                emb_arr = np.asarray(emb, dtype=np.float32)
            except (TypeError, ValueError):
                n_skipped += 1
                continue
            if emb_arr.ndim != 2 or emb_arr.shape[0] == 0:
                n_skipped += 1
                continue

            pooled = emb_arr.mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            if norm > 1e-10:
                pooled = pooled / norm

            embeddings_list.append(pooled)
            labels_list.append(0 if cat in safe_cats else 1)
            categories.append(cat)

            stress_mean = ((meas.get("stress_score") or {}).get("scalars")
                           or {}).get("stress_mean")
            if stress_mean is None:
                stress_mean = ((meas.get("stress_score") or {}).get("scalars")
                                or {}).get("stress")
            stress_scores.append(float(stress_mean or 0.0))

            # Per-layer attribution
            lpa = (meas.get("last_position_attribution") or {})
            plt = (lpa.get("per_layer_per_token") or {}).get(
                "signed_attribution_to_last_by_layer") or {}
            if plt:
                layer_keys = sorted(int(k) for k in plt.keys())
                if signal_layers is None:
                    signal_layers = list(layer_keys)
                for li in layer_keys:
                    vals = plt.get(str(li)) or plt.get(li) or []
                    if not vals:
                        continue
                    mean_abs = float(np.mean(np.abs(vals)))
                    layer_attrs[li].append(mean_abs)

            # Token-level data for patching map
            tokens = p.get("tokens") or []
            stress_per_tok = ((meas.get("stress_score") or {}).get(
                "per_token") or {}).get("stress")
            if plt and stress_per_tok is not None:
                try:
                    stress_arr = np.asarray(stress_per_tok, dtype=np.float32)
                except (TypeError, ValueError):
                    stress_arr = None
                if stress_arr is not None:
                    for k, attr_vals in plt.items():
                        try:
                            li = int(k)
                        except ValueError:
                            continue
                        if not attr_vals:
                            continue
                        attr_arr = np.abs(np.asarray(attr_vals,
                                                       dtype=np.float32))
                        n_tok = min(len(attr_arr), len(stress_arr))
                        if n_tok == 0:
                            continue
                        combined = (attr_arr[:n_tok] *
                                    stress_arr[:n_tok])
                        token_attrs.append(
                            (li, combined.tolist(),
                             list(tokens[:n_tok])))

        if not embeddings_list:
            err = (f"No prompts with per-token final embeddings in any "
                   f"labeled category. {n_skipped} prompts skipped.")
            result.warnings.append(err)
            result.objects["error"] = err
            result.objects["n_prompts"] = 0
            return result

        embeddings = np.stack(embeddings_list)
        labels = np.asarray(labels_list, dtype=np.int32)
        n_prompts = len(labels)
        n_safe = int((labels == 0).sum())
        n_risk = int((labels == 1).sum())
        hidden_dim = int(embeddings.shape[1])

        if n_prompts < 6 or n_safe < 2 or n_risk < 2:
            err = (f"Need ≥2 safe and ≥2 risk prompts with embeddings. "
                   f"Found {n_safe} safe, {n_risk} risk.")
            result.warnings.append(err)
            result.objects["error"] = err
            result.objects["n_prompts"] = n_prompts
            result.objects["n_safe"] = n_safe
            result.objects["n_risk"] = n_risk
            result.objects["hidden_dim"] = hidden_dim
            return result

        output: dict = {"n_prompts": n_prompts,
                        "n_safe": n_safe, "n_risk": n_risk,
                        "hidden_dim": hidden_dim}

        # ── 1. Refusal direction ──
        safe_embs = embeddings[labels == 0]
        risk_embs = embeddings[labels == 1]
        mean_safe = safe_embs.mean(axis=0)
        mean_risk = risk_embs.mean(axis=0)
        refusal_dir = mean_risk - mean_safe
        refusal_norm = float(np.linalg.norm(refusal_dir))
        if refusal_norm > 1e-10:
            refusal_dir = refusal_dir / refusal_norm

        cosines = embeddings @ refusal_dir
        refusal_auroc = _safe_auroc(cosines, labels)

        stress_arr = np.asarray(stress_scores, dtype=np.float32)
        stress_auroc = _safe_auroc(stress_arr, labels)

        cat_cosines: dict[str, list[float]] = defaultdict(list)
        for i, cat in enumerate(categories):
            cat_cosines[cat].append(float(cosines[i]))

        output["refusal_direction"] = {
            "auroc": round(refusal_auroc, 4),
            "stress_auroc": round(stress_auroc, 4),
            "refusal_norm": round(refusal_norm, 6),
            "mean_cosine_safe": round(float(cosines[labels == 0].mean()), 4),
            "mean_cosine_risk": round(float(cosines[labels == 1].mean()), 4),
            "std_cosine_safe": round(float(cosines[labels == 0].std()), 4),
            "std_cosine_risk": round(float(cosines[labels == 1].std()), 4),
            "separation": round(float(
                cosines[labels == 1].mean() - cosines[labels == 0].mean()), 4),
            "per_category": {
                cat: {
                    "mean": round(float(np.mean(vs)), 4),
                    "std": round(float(np.std(vs)), 4),
                    "n": len(vs),
                }
                for cat, vs in sorted(cat_cosines.items())
            },
        }

        # ── 2. Per-layer AUROC ──
        layer_auroc_results = []
        if signal_layers and layer_attrs:
            valid_layers = [li for li in signal_layers
                             if li in layer_attrs
                             and len(layer_attrs[li]) == n_prompts]
            for li in valid_layers:
                vals = np.asarray(layer_attrs[li], dtype=np.float32)
                auc = _safe_auroc(vals, labels)
                safe_mean = float(vals[labels == 0].mean())
                risk_mean = float(vals[labels == 1].mean())
                layer_auroc_results.append({
                    "layer": int(li),
                    "auroc": round(auc, 4),
                    "mean_safe": round(safe_mean, 6),
                    "mean_risk": round(risk_mean, 6),
                    "ratio": round(
                        risk_mean / max(safe_mean, 1e-10), 3),
                })
        output["per_layer_auroc"] = {
            "layers": layer_auroc_results,
            "n_layers": len(layer_auroc_results),
        }

        # ── 3. Activation patching priority map ──
        if token_attrs and signal_layers:
            n_layers = len(signal_layers)
            layer_idx_map = {li: idx for idx, li in enumerate(signal_layers)}
            pos_data: dict = defaultdict(lambda: defaultdict(list))
            for li, combined, toks in token_attrs:
                if li not in layer_idx_map:
                    continue
                for pos, val in enumerate(combined[:max_patch]):
                    pos_data[layer_idx_map[li]][pos].append(val)
            n_pos = min(max_patch,
                         max((max(p.keys()) + 1
                              for p in pos_data.values()), default=0))
            patching_matrix = np.zeros((n_layers, n_pos))
            patching_counts = np.zeros((n_layers, n_pos))
            for li_idx, positions in pos_data.items():
                for pos, vals in positions.items():
                    if pos < n_pos:
                        patching_matrix[li_idx, pos] = float(np.mean(vals))
                        patching_counts[li_idx, pos] = len(vals)

            top_k = 20
            flat = patching_matrix.flatten()
            top_points = []
            if flat.size:
                top_indices = np.argsort(flat)[-top_k:][::-1]
                for idx in top_indices:
                    li_idx = idx // max(1, n_pos)
                    pos = idx % max(1, n_pos)
                    if li_idx < len(signal_layers):
                        top_points.append({
                            "layer": int(signal_layers[li_idx]),
                            "position": int(pos),
                            "intensity": round(
                                float(patching_matrix[li_idx, pos]), 6),
                            "n_observations": int(
                                patching_counts[li_idx, pos]),
                        })

            layer_marginal = []
            for li_idx in range(n_layers):
                row = patching_matrix[li_idx]
                mask = patching_counts[li_idx] > 0
                if mask.any():
                    layer_marginal.append({
                        "layer": int(signal_layers[li_idx]),
                        "mean_intensity": round(
                            float(row[mask].mean()), 6),
                        "max_intensity": round(float(row.max()), 6),
                    })

            output["patching_priority"] = {
                "top_points": top_points,
                "layer_marginal": layer_marginal,
                "matrix_shape": [int(n_layers), int(n_pos)],
                "matrix": patching_matrix.round(6).tolist(),
                "layers": [int(x) for x in signal_layers],
            }
        else:
            output["patching_priority"] = {
                "top_points": [],
                "layer_marginal": [],
                "matrix_shape": [0, 0],
                "matrix": [],
                "layers": [],
            }

        # ── 4. Random projection control ──
        random_aurocs = []
        for t in range(n_random):
            rd = rng.randn(hidden_dim).astype(np.float32)
            n = float(np.linalg.norm(rd))
            if n > 1e-10:
                rd = rd / n
            proj = embeddings @ rd
            random_aurocs.append(_safe_auroc(proj, labels))
        random_arr = np.asarray(random_aurocs)
        rmean = float(random_arr.mean())
        rstd = float(random_arr.std())
        rmax = float(random_arr.max())
        rp95 = float(np.percentile(random_arr, 95))
        refusal_delta = refusal_auroc - rmean
        stress_delta = stress_auroc - rmean
        refusal_p = float((random_arr >= refusal_auroc).mean())
        stress_p = float((random_arr >= stress_auroc).mean())

        output["random_projection"] = {
            "n_trials": n_random,
            "random_mean_auroc": round(rmean, 4),
            "random_std_auroc": round(rstd, 4),
            "random_max_auroc": round(rmax, 4),
            "random_p95_auroc": round(rp95, 4),
            "refusal_direction": {
                "auroc": round(refusal_auroc, 4),
                "delta_over_random": round(refusal_delta, 4),
                "empirical_p": round(refusal_p, 4),
            },
            "tagm_stress": {
                "auroc": round(stress_auroc, 4),
                "delta_over_random": round(stress_delta, 4),
                "empirical_p": round(stress_p, 4),
            },
            "histogram": [round(float(x), 4) for x in random_arr],
        }

        # Top-level AUROC / p convenience fields the UI reads directly
        output["auroc"] = round(refusal_auroc, 4)
        output["delta_over_random"] = round(refusal_delta, 4)
        output["empirical_p"] = round(refusal_p, 4)

        # ── Summary ──
        best_layer_auroc = max(
            (x["auroc"] for x in layer_auroc_results), default=0.5)
        best_layer = (
            max(layer_auroc_results, key=lambda x: x["auroc"])["layer"]
            if layer_auroc_results else None)
        output["summary"] = {
            "refusal_auroc": round(refusal_auroc, 4),
            "stress_auroc": round(stress_auroc, 4),
            "random_mean": round(rmean, 4),
            "best_layer_auroc": round(best_layer_auroc, 4),
            "best_layer": best_layer,
            "refusal_separates": bool(refusal_auroc > rp95),
            "stress_separates": bool(stress_auroc > rp95),
            "refusal_beats_stress": bool(refusal_auroc > stress_auroc),
        }

        result.objects.update(output)
        result.scalars["n_prompts"] = n_prompts
        result.scalars["n_safe"] = n_safe
        result.scalars["n_risk"] = n_risk
        result.scalars["hidden_dim"] = hidden_dim
        result.scalars["refusal_auroc"] = round(refusal_auroc, 4)
        result.scalars["stress_auroc"] = round(stress_auroc, 4)
        return result
