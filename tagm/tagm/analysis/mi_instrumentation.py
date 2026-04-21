"""MI Instrumentation: refusal direction, patching map, per-layer AUROC.

Ported from TASM's engine/modules/mi_instrumentation.py. Full computation
preserved; data reading adapted to TAGM's native session schema.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.analysis.statistics import extract_scalar
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


def _safe_auroc(scores, labels):
    if len(scores) < 4:
        return 0.5
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_scores = scores[pos]
    neg_scores = scores[neg]
    u = 0.0
    for ps in pos_scores:
        u += (neg_scores < ps).sum() + 0.5 * (neg_scores == ps).sum()
    return float(u / (n_pos * n_neg))


def _get_final_emb(r):
    """Extract per-token final embeddings from TAGM measurement."""
    pte = ((r.get("measurements") or {}).get("per_token_embedding") or {})
    embs = (pte.get("objects") or {}).get("per_token_embeddings") or {}
    return embs.get("final")


def _get_per_layer_attr(r):
    """Extract per-layer signed attribution dict from TAGM measurement."""
    lpa = ((r.get("measurements") or {}).get("last_position_attribution") or {})
    return (lpa.get("per_layer_per_token") or {}).get(
        "signed_attribution_to_last_by_layer") or {}


def _get_per_token_stress(r):
    """Extract per-token stress array from TAGM measurement."""
    ss = ((r.get("measurements") or {}).get("stress_score") or {})
    return (ss.get("per_token") or {}).get("stress")


@register_analysis
class MIInstrumentation(AnalysisModule):
    name = "mi_instrumentation"
    display_name = "MI Instrumentation"
    description = (
        "Refusal direction extraction, activation patching priority map, "
        "per-layer AUROC curve, and random projection control."
    )
    version = "1.0.0"
    min_results = 10

    depends_on_measurements = ("stress_score", "last_position_attribution")

    parameters = [
        ModuleParameter(name="safe_categories", display_name="Safe Categories",
                        description="Comma-separated safe categories.",
                        kind="string", default="benign,mild"),
        ModuleParameter(name="n_random_trials", display_name="Random Projection Trials",
                        description="Number of random directions for control.",
                        kind="int", default=50, min_value=5, max_value=200),
        ModuleParameter(name="random_seed", display_name="Random Seed",
                        description="Seed for reproducibility.",
                        kind="int", default=42, min_value=0, max_value=99999),
        ModuleParameter(name="max_patching_tokens", display_name="Max Patching Tokens",
                        description="Max token positions in the patching map.",
                        kind="int", default=30, min_value=5, max_value=100),
    ]

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []

        safe_cats = set(
            c.strip().lower()
            for c in params.get("safe_categories", "benign,mild").split(",")
            if c.strip()
        )
        n_random = int(params.get("n_random_trials", 50))
        seed = int(params.get("random_seed", 42))
        max_patch_tokens = int(params.get("max_patching_tokens", 30))
        rng = np.random.RandomState(seed)

        risk_cats = {"harmful", "jailbreak", "adversarial", "dual-use"} - safe_cats
        all_cats = safe_cats | risk_cats

        embeddings = []
        labels = []
        categories = []
        stress_scores = []
        layer_attrs = {}
        token_attrs = []
        prompt_tokens = []

        # Discover signal layers from first prompt with per-layer data
        signal_layers = None
        for r in prompts:
            pla = _get_per_layer_attr(r)
            if pla:
                signal_layers = sorted(int(k) for k in pla.keys())
                break

        n_skipped = 0
        for r in prompts:
            cat = (r.get("category") or "").lower().strip()
            if cat not in all_cats:
                n_skipped += 1
                continue

            emb = _get_final_emb(r)
            if emb is None or len(emb) == 0:
                n_skipped += 1
                continue

            emb_arr = np.array(emb, dtype=np.float32)
            prompt_emb = emb_arr.mean(axis=0)
            norm = np.linalg.norm(prompt_emb)
            if norm > 1e-10:
                prompt_emb /= norm

            embeddings.append(prompt_emb)
            labels.append(0 if cat in safe_cats else 1)
            categories.append(cat)
            stress_val = extract_scalar(r, "stress_score", "stress_mean")
            stress_scores.append(float(stress_val or 0))

            pla = _get_per_layer_attr(r)
            if signal_layers and pla:
                for li in signal_layers:
                    key = str(li)
                    if key in pla:
                        attr_vals = pla[key]
                        mean_attr = float(np.mean(np.abs(attr_vals)))
                        if li not in layer_attrs:
                            layer_attrs[li] = []
                        layer_attrs[li].append(mean_attr)

            tokens = r.get("tokens", [])
            prompt_tokens.append(tokens)
            stress = _get_per_token_stress(r)
            if pla and stress is not None:
                stress_arr = np.array(stress, dtype=np.float32)
                for key_str, attr_vals in pla.items():
                    li = int(key_str)
                    attr_arr = np.abs(np.array(attr_vals, dtype=np.float32))
                    n_tok = min(len(attr_arr), len(stress_arr))
                    combined = attr_arr[:n_tok] * stress_arr[:n_tok]
                    token_attrs.append((li, combined.tolist(), tokens[:n_tok]))

        if not embeddings:
            return {"error": "No prompts with embeddings and valid categories."}

        embeddings = np.array(embeddings, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        n_prompts = len(labels)
        n_safe = int((labels == 0).sum())
        n_risk = int((labels == 1).sum())
        hidden_dim = embeddings.shape[1]

        if n_prompts < 6 or n_safe < 2 or n_risk < 2:
            return {"error": f"Need ≥2 safe and ≥2 risk with embeddings. "
                             f"Found {n_safe} safe, {n_risk} risk."}

        output = {"n_prompts": n_prompts, "n_safe": n_safe,
                  "n_risk": n_risk, "hidden_dim": hidden_dim}

        # 1. Refusal direction
        safe_embs = embeddings[labels == 0]
        risk_embs = embeddings[labels == 1]
        refusal_dir = risk_embs.mean(axis=0) - safe_embs.mean(axis=0)
        refusal_norm = np.linalg.norm(refusal_dir)
        if refusal_norm > 1e-10:
            refusal_dir /= refusal_norm

        cosines = embeddings @ refusal_dir
        refusal_auroc = _safe_auroc(cosines, labels)
        stress_arr = np.array(stress_scores, dtype=np.float32)
        stress_auroc = _safe_auroc(stress_arr, labels)

        cat_cosines = defaultdict(list)
        for i, cat in enumerate(categories):
            cat_cosines[cat].append(float(cosines[i]))

        output["refusal_direction"] = {
            "auroc": round(refusal_auroc, 4),
            "stress_auroc": round(stress_auroc, 4),
            "refusal_norm": round(float(refusal_norm), 6),
            "mean_cosine_safe": round(float(cosines[labels == 0].mean()), 4),
            "mean_cosine_risk": round(float(cosines[labels == 1].mean()), 4),
            "std_cosine_safe": round(float(cosines[labels == 0].std()), 4),
            "std_cosine_risk": round(float(cosines[labels == 1].std()), 4),
            "separation": round(float(cosines[labels == 1].mean() - cosines[labels == 0].mean()), 4),
            "per_category": {
                cat: {"mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals)), 4), "n": len(vals)}
                for cat, vals in sorted(cat_cosines.items())
            },
        }

        # 2. Per-layer AUROC
        layer_auroc_results = []
        if signal_layers and layer_attrs:
            valid_layers = [li for li in signal_layers
                           if li in layer_attrs and len(layer_attrs[li]) == n_prompts]
            for li in valid_layers:
                vals = np.array(layer_attrs[li], dtype=np.float32)
                auc = _safe_auroc(vals, labels)
                layer_auroc_results.append({
                    "layer": li, "auroc": round(auc, 4),
                    "mean_safe": round(float(vals[labels == 0].mean()), 6),
                    "mean_risk": round(float(vals[labels == 1].mean()), 6),
                    "ratio": round(float(vals[labels == 1].mean() /
                                         max(vals[labels == 0].mean(), 1e-10)), 3),
                })

        output["per_layer_auroc"] = {"layers": layer_auroc_results,
                                      "n_layers": len(layer_auroc_results)}

        # 3. Patching priority map
        if token_attrs and signal_layers:
            n_layers = len(signal_layers)
            layer_idx_map = {li: idx for idx, li in enumerate(signal_layers)}
            pos_data = defaultdict(lambda: defaultdict(list))
            for (li, combined, toks) in token_attrs:
                if li not in layer_idx_map:
                    continue
                for pos, val in enumerate(combined[:max_patch_tokens]):
                    pos_data[layer_idx_map[li]][pos].append(val)

            n_pos = min(max_patch_tokens,
                        max((max(positions.keys()) + 1
                             for positions in pos_data.values()), default=0))
            patching_matrix = np.zeros((n_layers, n_pos))
            patching_counts = np.zeros((n_layers, n_pos))
            for li_idx, positions in pos_data.items():
                for pos, vals in positions.items():
                    if pos < n_pos:
                        patching_matrix[li_idx, pos] = np.mean(vals)
                        patching_counts[li_idx, pos] = len(vals)

            top_k = 20
            flat = patching_matrix.flatten()
            top_indices = np.argsort(flat)[-top_k:][::-1]
            top_points = []
            for idx in top_indices:
                li_idx = idx // n_pos
                pos = idx % n_pos
                if li_idx < len(signal_layers):
                    top_points.append({
                        "layer": int(signal_layers[li_idx]),
                        "position": int(pos),
                        "intensity": round(float(patching_matrix[li_idx, pos]), 6),
                        "n_observations": int(patching_counts[li_idx, pos]),
                    })

            layer_marginal = []
            for li_idx in range(n_layers):
                row = patching_matrix[li_idx]
                mask = patching_counts[li_idx] > 0
                if mask.any():
                    layer_marginal.append({
                        "layer": int(signal_layers[li_idx]),
                        "mean_intensity": round(float(row[mask].mean()), 6),
                        "max_intensity": round(float(row.max()), 6),
                    })

            output["patching_priority"] = {
                "top_points": top_points, "layer_marginal": layer_marginal,
                "matrix_shape": [int(n_layers), int(n_pos)],
                "matrix": patching_matrix.round(6).tolist(),
                "layers": [int(x) for x in signal_layers],
            }
        else:
            output["patching_priority"] = {
                "top_points": [], "layer_marginal": [],
                "matrix_shape": [0, 0], "matrix": [], "layers": [],
            }

        # 4. Random projection control
        random_aurocs = []
        for trial in range(n_random):
            rand_dir = rng.randn(hidden_dim).astype(np.float32)
            rand_dir /= np.linalg.norm(rand_dir) + 1e-10
            projections = embeddings @ rand_dir
            auc = _safe_auroc(projections, labels)
            random_aurocs.append(auc)

        random_aurocs = np.array(random_aurocs)
        random_mean = float(random_aurocs.mean())
        random_std = float(random_aurocs.std())
        random_max = float(random_aurocs.max())
        random_p95 = float(np.percentile(random_aurocs, 95))
        refusal_delta = refusal_auroc - random_mean
        stress_delta = stress_auroc - random_mean

        output["random_projection"] = {
            "n_trials": n_random,
            "random_mean_auroc": round(random_mean, 4),
            "random_std_auroc": round(random_std, 4),
            "random_max_auroc": round(random_max, 4),
            "random_p95_auroc": round(random_p95, 4),
            "refusal_direction": {
                "auroc": round(refusal_auroc, 4),
                "delta_over_random": round(refusal_delta, 4),
                "empirical_p": round(float((random_aurocs >= refusal_auroc).mean()), 4),
            },
            "tagm_stress": {
                "auroc": round(stress_auroc, 4),
                "delta_over_random": round(stress_delta, 4),
                "empirical_p": round(float((random_aurocs >= stress_auroc).mean()), 4),
            },
            "histogram": [round(float(x), 4) for x in random_aurocs],
        }

        # Summary
        output["summary"] = {
            "refusal_auroc": round(refusal_auroc, 4),
            "stress_auroc": round(stress_auroc, 4),
            "random_mean": round(random_mean, 4),
            "best_layer_auroc": round(
                max((x["auroc"] for x in layer_auroc_results), default=0.5), 4),
            "best_layer": (max(layer_auroc_results, key=lambda x: x["auroc"])["layer"]
                           if layer_auroc_results else None),
            "refusal_separates": refusal_auroc > random_p95,
            "stress_separates": stress_auroc > random_p95,
            "refusal_beats_stress": refusal_auroc > stress_auroc,
        }

        return output
