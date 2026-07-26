"""
MI Instrumentation Module for TASM.

Produces mechanistic-interpretability-relevant measurements from
session data. Unlike MI Readiness (which evaluates data quality),
this module generates the actual MI outputs a researcher would cite.

Features:
  1. Refusal Direction Extraction — empirical refusal direction from
     mean hidden states, per-prompt cosine alignment, AUROC comparison
  2. Activation Patching Priority Map — (layer × position) correction
     intensity matrix identifying highest-value intervention points
  3. Per-Layer AUROC — discrimination power at each model depth,
     showing where the correction signal concentrates
  4. Random Projection Control — validates that weight-delta projection
     outperforms random directions of matched dimensionality

All features operate on existing session data (per_token_final_emb,
per_layer_signed_attr, per_token_stress). No additional inference required.
"""

import logging
import numpy as np
from collections import defaultdict

from .base import TASMModule, ModuleParameter, classify_category
from src.engine.metrics import auroc as _metric_auroc, fit_direction_holdout

logger = logging.getLogger("tasm")


def _safe_auroc(scores, labels):
    """Compute AUROC from score/label arrays. Returns 0.5 if degenerate.

    WAS WRONG (as a design): one of five divergent AUROC copies. Body now
    delegates to the single verified implementation in src.engine.metrics.
    Behaviour is unchanged (same n < 4 guard, same tie handling), so no
    number produced by this function moves.
    """
    return _metric_auroc(scores, labels)


class MIInstrumentationModule(TASMModule):
    name = "mi_instrumentation"
    display_name = "MI Instrumentation"
    description = (
        "Produces MI-relevant measurements from session data: "
        "refusal direction extraction, activation patching priority map, "
        "per-layer AUROC curve, and random projection control. "
        "Requires full_capture session data (per_token_final_emb)."
    )
    version = "1.0.0"

    min_results = 10
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="safe_categories",
            display_name="Safe Categories",
            description="Comma-separated list of categories treated as safe (class 0).",
            type="str",
            default="benign,mild",
        ),
        ModuleParameter(
            name="n_random_trials",
            display_name="Random Projection Trials",
            description="Number of random directions for the projection control.",
            type="int",
            default=50,
        ),
        ModuleParameter(
            name="random_seed",
            display_name="Random Seed",
            description="Seed for reproducible random projections.",
            type="int",
            default=42,
        ),
        ModuleParameter(
            name="max_patching_tokens",
            display_name="Max Tokens (Patching Map)",
            description="Maximum token positions to include in the patching priority map.",
            type="int",
            default=30,
        ),
    ]

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[MI-INST] {msg}")

        safe_cats = set(
            c.strip().lower()
            for c in params.get("safe_categories", "benign,mild").split(",")
            if c.strip()
        )
        n_random = int(params.get("n_random_trials", 50))
        seed = int(params.get("random_seed", 42))
        max_patch_tokens = int(params.get("max_patching_tokens", 30))
        rng = np.random.RandomState(seed)

        # ── Gather embeddings and labels ──
        prog("Extracting embeddings and labels...")

        embeddings = []       # mean-pooled final-layer embeddings per prompt
        labels = []           # 0=safe, 1=risk
        categories = []
        stress_scores = []
        layer_attrs = {}      # {layer_idx: [per-prompt mean attribution]}
        token_attrs = []      # per-prompt: list of (layer, position, value) for patching map
        prompt_tokens = []    # token lists per prompt

        # TAXONOMY (was wrong): this derived risk_cats from a private literal
        # set, one of six divergent vocabularies in this package. Prompts are
        # now classified by the canonical taxonomy in src.engine.metrics, with
        # the module's safe_categories parameter honoured as an explicit
        # override on top of it. Practical effect: "risk"/"dangerous" now
        # count as risk, "neutral"/"baseline"/"safe" now count as safe, and
        # "unknown" is EXCLUDED. n_safe / n_risk will change.
        def _cls(cat):
            return classify_category(cat, safe_override=safe_cats)

        # Collect signal layers from first result that has them
        signal_layers = None
        for r in session_results:
            sl = r.get("signal_layer_indices")
            if sl and len(sl) > 0:
                signal_layers = sorted(int(x) for x in sl)
                break

        n_skipped = 0
        n_unclassified = 0
        for ri, r in enumerate(session_results):
            cat = (r.get("category") or "").lower().strip()
            cls = _cls(cat)
            if cls is None:
                n_skipped += 1
                n_unclassified += 1
                continue

            emb = r.get("per_token_final_emb")
            if emb is None or len(emb) == 0:
                n_skipped += 1
                continue

            # Mean-pool token embeddings
            emb_arr = np.array(emb, dtype=np.float32)
            prompt_emb = emb_arr.mean(axis=0)
            norm = np.linalg.norm(prompt_emb)
            if norm > 1e-10:
                prompt_emb /= norm

            embeddings.append(prompt_emb)
            labels.append(0 if cls == "safe" else 1)
            categories.append(cat)
            stress_scores.append(float(r.get("stress_score", 0) or 0))

            # Per-layer signed attribution
            pla = r.get("per_layer_signed_attr", {})
            if signal_layers:
                for li in signal_layers:
                    key = str(li)
                    if key in pla:
                        attr_vals = pla[key]
                        mean_attr = float(np.mean(np.abs(attr_vals)))
                        if li not in layer_attrs:
                            layer_attrs[li] = []
                        layer_attrs[li].append(mean_attr)

            # Token-level data for patching map
            tokens = r.get("tokens", [])
            prompt_tokens.append(tokens)
            stress = r.get("per_token_stress")
            if pla and stress is not None:
                stress_arr = np.array(stress, dtype=np.float32) if not isinstance(stress, np.ndarray) else stress
                for key_str, attr_vals in pla.items():
                    li = int(key_str)
                    attr_arr = np.abs(np.array(attr_vals, dtype=np.float32))
                    n_tok = min(len(attr_arr), len(stress_arr))
                    combined = attr_arr[:n_tok] * stress_arr[:n_tok]
                    token_attrs.append((li, combined.tolist(), tokens[:n_tok]))

        embeddings = np.array(embeddings, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        n_prompts = len(labels)
        n_safe = (labels == 0).sum()
        n_risk = (labels == 1).sum()
        hidden_dim = embeddings.shape[1] if n_prompts > 0 else 0

        prog(f"Found {n_prompts} prompts ({n_safe} safe, {n_risk} risk), "
             f"dim={hidden_dim}, skipped {n_skipped}")

        if n_prompts < 6 or n_safe < 2 or n_risk < 2:
            return {"error": f"Need at least 2 safe and 2 risk prompts with "
                             f"embeddings. Found {n_safe} safe, {n_risk} risk."}

        output = {"n_prompts": n_prompts, "n_safe": int(n_safe),
                  "n_risk": int(n_risk), "hidden_dim": hidden_dim,
                  # Prompts excluded because their category is neither harm
                  # nor safe under the canonical taxonomy (e.g. "unknown").
                  "n_excluded_unclassified": int(n_unclassified)}

        # ════════════════════════════════════════════════════════
        # 1. REFUSAL DIRECTION EXTRACTION
        # ════════════════════════════════════════════════════════
        prog("Extracting refusal direction...")

        safe_embs = embeddings[labels == 0]
        risk_embs = embeddings[labels == 1]

        mean_safe = safe_embs.mean(axis=0)
        mean_risk = risk_embs.mean(axis=0)

        refusal_dir = mean_risk - mean_safe
        refusal_norm = np.linalg.norm(refusal_dir)
        if refusal_norm > 1e-10:
            refusal_dir /= refusal_norm

        # Per-prompt cosine similarity with refusal direction
        cosines = embeddings @ refusal_dir

        # ── CIRCULAR EVALUATION (was wrong) ──────────────────────
        # refusal_dir = mean_risk - mean_safe was fitted on ALL embeddings and
        # then scored on those SAME embeddings, so `refusal_auroc` measured
        # the fit, not the effect. The random-projection control below does
        # not correct for it: a random direction was never fitted to the
        # labels, so `delta_over_random` measured the value of fitting.
        # cv_auroc refits the direction inside every fold and scores it
        # out-of-fold, with a label-permutation null run through the same
        # fit-and-score pipeline. On pure noise train_auroc can sit near 0.83
        # while cv_auroc sits near 0.20 with p ~ 0.98.
        # NUMBERS CHANGE: the headline "refusal AUROC" is now cv_auroc and
        # will typically be LOWER than every previously reported value; the
        # old number is preserved verbatim as refusal_train_auroc.
        refusal_train_auroc = _safe_auroc(cosines, labels)
        cv = fit_direction_holdout(risk_embs, safe_embs)
        refusal_auroc = cv["cv_auroc"]

        # Compare with stress score AUROC. NOTE: stress_score is not fitted to
        # the labels at all, so this one is honest as an in-sample number.
        stress_arr = np.array(stress_scores, dtype=np.float32)
        stress_auroc = _safe_auroc(stress_arr, labels)

        # Per-category mean cosine
        cat_cosines = defaultdict(list)
        for i, cat in enumerate(categories):
            cat_cosines[cat].append(float(cosines[i]))

        output["refusal_direction"] = {
            # Headline: out-of-fold, direction refitted per fold.
            "auroc": round(refusal_auroc, 4),
            "cv_auroc": round(refusal_auroc, 4),
            # IN-SAMPLE: the direction was fitted on the rows it scores.
            # Kept only for comparison with cv_auroc; a large gap means the
            # "direction" is mostly fitting noise.
            "train_auroc": round(refusal_train_auroc, 4),
            "cv_p_value": (round(cv["p_value"], 4)
                           if cv["p_value"] is not None else None),
            "cv_underpowered": bool(cv["underpowered"]),
            "cv_n_risk": int(cv["n_pos"]),
            "cv_n_safe": int(cv["n_neg"]),
            "stress_auroc": round(stress_auroc, 4),
            "refusal_norm": round(float(refusal_norm), 6),
            "mean_cosine_safe": round(float(cosines[labels == 0].mean()), 4),
            "mean_cosine_risk": round(float(cosines[labels == 1].mean()), 4),
            "std_cosine_safe": round(float(cosines[labels == 0].std()), 4),
            "std_cosine_risk": round(float(cosines[labels == 1].std()), 4),
            "separation": round(float(
                cosines[labels == 1].mean() - cosines[labels == 0].mean()
            ), 4),
            "per_category": {
                cat: {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "n": len(vals),
                }
                for cat, vals in sorted(cat_cosines.items())
            },
        }

        prog(f"Refusal direction: cv AUROC={refusal_auroc:.3f} "
             f"(p={cv['p_value']}), in-sample train AUROC="
             f"{refusal_train_auroc:.3f}, "
             f"stress AUROC={stress_auroc:.3f}, "
             f"separation={output['refusal_direction']['separation']:.4f}")

        # ════════════════════════════════════════════════════════
        # 2. PER-LAYER AUROC CURVE
        # ════════════════════════════════════════════════════════
        prog("Computing per-layer AUROC...")

        layer_auroc_results = []
        if signal_layers and layer_attrs:
            # Ensure all layers have the same number of entries
            valid_layers = [
                li for li in signal_layers
                if li in layer_attrs and len(layer_attrs[li]) == n_prompts
            ]

            for li in valid_layers:
                vals = np.array(layer_attrs[li], dtype=np.float32)
                auc = _safe_auroc(vals, labels)
                layer_auroc_results.append({
                    "layer": li,
                    "auroc": round(auc, 4),
                    "mean_safe": round(float(vals[labels == 0].mean()), 6),
                    "mean_risk": round(float(vals[labels == 1].mean()), 6),
                    "ratio": round(
                        float(vals[labels == 1].mean() /
                              max(vals[labels == 0].mean(), 1e-10)),
                        3),
                })

            if layer_auroc_results:
                best = max(layer_auroc_results, key=lambda x: x["auroc"])
                prog(f"Per-layer AUROC: {len(layer_auroc_results)} layers, "
                     f"best={best['auroc']:.3f} at layer {best['layer']}")

        output["per_layer_auroc"] = {
            "layers": layer_auroc_results,
            "n_layers": len(layer_auroc_results),
        }

        # ════════════════════════════════════════════════════════
        # 3. ACTIVATION PATCHING PRIORITY MAP
        # ════════════════════════════════════════════════════════
        prog("Building activation patching priority map...")

        # Aggregate (layer × position) correction intensity
        if token_attrs and signal_layers:
            n_layers = len(signal_layers)
            layer_idx_map = {li: idx for idx, li in enumerate(signal_layers)}

            # Collect per-position mean intensity across all prompts
            pos_data = defaultdict(lambda: defaultdict(list))
            for (li, combined, toks) in token_attrs:
                if li not in layer_idx_map:
                    continue
                for pos, val in enumerate(combined[:max_patch_tokens]):
                    pos_data[layer_idx_map[li]][pos].append(val)

            # Build aggregate matrix
            n_pos = min(max_patch_tokens,
                        max((max(positions.keys()) + 1
                             for positions in pos_data.values()),
                            default=0))

            patching_matrix = np.zeros((n_layers, n_pos))
            patching_counts = np.zeros((n_layers, n_pos))
            for li_idx, positions in pos_data.items():
                for pos, vals in positions.items():
                    if pos < n_pos:
                        patching_matrix[li_idx, pos] = np.mean(vals)
                        patching_counts[li_idx, pos] = len(vals)

            # Find top-k intervention points
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

            # Layer-marginal (mean across positions)
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
                "top_points": top_points,
                "layer_marginal": layer_marginal,
                "matrix_shape": [int(n_layers), int(n_pos)],
                "matrix": patching_matrix.round(6).tolist(),
                "layers": [int(x) for x in signal_layers],
            }
            prog(f"Patching map: {n_layers} layers × {n_pos} positions, "
                 f"top intensity={top_points[0]['intensity']:.6f} "
                 f"at L{top_points[0]['layer']}:P{top_points[0]['position']}"
                 if top_points else "empty")
        else:
            output["patching_priority"] = {
                "top_points": [],
                "layer_marginal": [],
                "matrix_shape": [0, 0],
                "matrix": [],
                "layers": [],
            }
            prog("Patching map: no per-layer attribution data available")

        # ════════════════════════════════════════════════════════
        # 4. RANDOM PROJECTION CONTROL
        # ════════════════════════════════════════════════════════
        prog(f"Running random projection control ({n_random} trials)...")

        random_aurocs = []
        for trial in range(n_random):
            # Generate random unit vector
            rand_dir = rng.randn(hidden_dim).astype(np.float32)
            rand_dir /= np.linalg.norm(rand_dir) + 1e-10

            # Project prompt embeddings
            projections = embeddings @ rand_dir
            auc = _safe_auroc(projections, labels)
            random_aurocs.append(auc)

        random_aurocs = np.array(random_aurocs)
        random_mean = float(random_aurocs.mean())
        random_std = float(random_aurocs.std())
        random_max = float(random_aurocs.max())
        random_p95 = float(np.percentile(random_aurocs, 95))

        # Delta over random for each real measurement.
        # DEMOTED (was the primary control): a random direction was never
        # fitted to the labels, so comparing a FITTED direction's in-sample
        # AUROC against it measured the value of fitting, not the existence of
        # a refusal direction. It is kept as a secondary control and is now
        # computed against the out-of-fold cv_auroc, which is a fair contest.
        # The correct primary control is cv_p_value above (permutation null
        # through the identical fit-and-score pipeline).
        refusal_delta = refusal_auroc - random_mean
        stress_delta = stress_auroc - random_mean

        # Empirical p-value: fraction of random trials that beat the real AUROC
        refusal_p = float((random_aurocs >= refusal_auroc).mean())
        stress_p = float((random_aurocs >= stress_auroc).mean())

        output["random_projection"] = {
            "control_role": "secondary",
            "note": ("Random directions are unfitted, so this cannot validate "
                     "a fitted direction on its own. The primary control is "
                     "refusal_direction.cv_p_value."),
            "n_trials": n_random,
            "random_mean_auroc": round(random_mean, 4),
            "random_std_auroc": round(random_std, 4),
            "random_max_auroc": round(random_max, 4),
            "random_p95_auroc": round(random_p95, 4),
            "refusal_direction": {
                "auroc": round(refusal_auroc, 4),   # cv_auroc, out-of-fold
                "train_auroc": round(refusal_train_auroc, 4),  # in-sample
                "delta_over_random": round(refusal_delta, 4),
                "empirical_p": round(refusal_p, 4),
            },
            "tagm_stress": {
                "auroc": round(stress_auroc, 4),
                "delta_over_random": round(stress_delta, 4),
                "empirical_p": round(stress_p, 4),
            },
            "histogram": [round(float(x), 4) for x in random_aurocs],
        }

        prog(f"Random projection: mean={random_mean:.3f}±{random_std:.3f}, "
             f"refusal Δ={refusal_delta:+.3f} (p={refusal_p:.3f}), "
             f"stress Δ={stress_delta:+.3f} (p={stress_p:.3f})")

        # ════════════════════════════════════════════════════════
        # SUMMARY
        # ════════════════════════════════════════════════════════
        prog("Generating summary...")

        output["summary"] = {
            # Headline is the honest out-of-fold number.
            "refusal_auroc": round(refusal_auroc, 4),
            "refusal_train_auroc_in_sample": round(refusal_train_auroc, 4),
            "refusal_cv_p_value": (round(cv["p_value"], 4)
                                   if cv["p_value"] is not None else None),
            "stress_auroc": round(stress_auroc, 4),
            "random_mean": round(random_mean, 4),
            "best_layer_auroc": round(
                max((x["auroc"] for x in layer_auroc_results), default=0.5),
                4),
            "best_layer": (max(layer_auroc_results,
                               key=lambda x: x["auroc"])["layer"]
                           if layer_auroc_results else None),
            "refusal_separates": refusal_auroc > random_p95,
            "stress_separates": stress_auroc > random_p95,
            "refusal_beats_stress": refusal_auroc > stress_auroc,
        }

        prog("MI Instrumentation complete.")
        return output
