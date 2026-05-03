"""
Domain Surface Module for TASM.

Maps per-token correction signals onto a subject-matter domain surface
defined by configurable probes. Embeds probes and session prompts into
a shared PCA space, merges per-token RD/ASM/SFD, computes 2D
nearest-probe proximity.

Pure post-processor: uses pre-computed domain_embedding fields from
the analyzer (prompt embeddings) and cached probe embeddings generated
at model load time.  No model access required at run time.

Original concept: Ostrander (2026).
See README.md in the domain_surface_module distribution for full docs.
"""

import os
import csv
import json
import math
import logging
import numpy as np
from glob import glob
from collections import defaultdict

from .base import TASMModule, ModuleParameter
from tagm.probes.io import (
    NORM_EPS,
    PROBE_CACHE_DIR,
    parse_meta,
    detect_level_cols,
    load_probes,
    probe_cache_path,
    load_probe_cache,
    embed_and_cache_probes,
    get_active_probe_set,
)

logger = logging.getLogger("tasm")

# Display truncation limits for frontend output
PROBE_TEXT_DISPLAY_LEN = 50
PROMPT_TEXT_DISPLAY_LEN = 80



# ─── PCA + Proximity ─────────────────────────────────────────

def _cofit_pca(prompt_embs, probe_embs, n_components=2):
    """PCA on concatenated prompt + probe embeddings."""
    from sklearn.decomposition import PCA

    all_embs = np.vstack([prompt_embs, probe_embs])
    pca = PCA(n_components=n_components)
    all_coords = pca.fit_transform(all_embs)

    n_p = len(prompt_embs)
    prompt_coords = all_coords[:n_p]
    probe_coords = all_coords[n_p:]

    variance = [round(float(v) * 100, 1)
                for v in pca.explained_variance_ratio_[:n_components]]
    return prompt_coords, probe_coords, variance


# ─── Token-level η² (category-dependence) ────────────────────
# Computes per-token eta-squared on the SFD per_token_density channel
# directly from session results. Used by the token-selection step to
# rank content words above function words: tokens whose density signal
# varies WITH category get higher weight than tokens whose signal
# varies independently of category.
#
# This was previously the Token Variance module's `eta_sq_density`
# output; inlined here so Domain Surface is self-contained. Only the
# density channel is needed — stress and signed_attr are irrelevant
# for token selection — so this is much cheaper than the full module.

def _compute_token_eta_density(session_results, min_appearances=3,
                                min_seq_len=3, exclude_first=True):
    """Return {token_str: eta_sq_density} computed from session_results.

    Returns an empty dict if the session lacks the required fields
    (sfd.per_token_density). Caller can detect the empty case and fall
    back to frequency-only selection.
    """
    # Per-token list of (density, category)
    per_token = defaultdict(list)
    for d in session_results:
        tokens = d.get("tokens")
        sfd = d.get("sfd") or {}
        ptd = sfd.get("per_token_density")
        if not tokens or not ptd:
            continue
        seq_len = d.get("seq_len", len(tokens))
        if seq_len < min_seq_len:
            continue
        category = d.get("category", "")
        if not category:
            continue
        for i, tok in enumerate(tokens):
            if i >= len(ptd):
                continue
            if exclude_first and i == 0:
                continue
            per_token[tok.strip()].append((float(ptd[i]), category))

    weights = {}
    for tok, entries in per_token.items():
        if len(entries) < min_appearances:
            continue
        # Group density values by category
        groups = defaultdict(list)
        for v, c in entries:
            groups[c].append(v)
        # Need ≥2 categories represented to define between-group variance
        non_empty = [g for g in groups.values() if len(g) >= 1]
        if len(non_empty) < 2:
            continue
        all_vals = np.concatenate([np.array(g) for g in non_empty])
        grand_mean = float(np.mean(all_vals))
        ss_total = float(np.sum((all_vals - grand_mean) ** 2))
        if ss_total < 1e-15:
            continue
        ss_between = sum(
            len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in non_empty
        )
        weights[tok] = float(ss_between / ss_total)
    return weights


def _nearest_probe(dx, dy, anchor_pts):
    """Find nearest probe by 2D Euclidean distance."""
    best_dist = float("inf")
    best_idx = 0
    for i, a in enumerate(anchor_pts):
        d = math.sqrt((dx - a["x"]) ** 2 + (dy - a["y"]) ** 2)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist


# ─── Observation Builder ─────────────────────────────────────

def _build_observations(session_results, prompt_coords, anchor_pts,
                        subjects, top_n=20, min_appearances=2, progress=None,
                        probe_embs=None, probes=None,
                        esc_probe_embs=None,
                        tv_eta_weights=None,
                        probe_neighbors=5,
                        knn_sharpness=10.0,
                        eta_floor=0.01):
    """Build per-token observations with all metrics and proximity.

    Split-depth probe matching: each token gets TWO independent probe lookups.
      - Subject (angular wedge): cosine similarity of token's domain-layer
        embedding against probe embeddings cached at domain_embedding_layer_frac.
      - Escalation (ring): cosine similarity of token's escalation-layer
        embedding against probe embeddings cached at domain_escalation_layer_frac.

    If tv_eta_weights is provided (dict of token → eta_sq_density from Token
    Variance module), content words are weighted higher than function words
    in the top-N selection. Without it, falls back to pure frequency ranking.

    Falls back to prompt-level PCA nearest-probe matching when per-token
    embeddings are not available.
    """
    token_freq = defaultdict(int)
    raw_obs = []

    # Pre-build subject index for probe-level matching
    subj_set = sorted(set(p["subject"] for p in probes)) if probes else sorted(subjects)
    subj_to_idx = {s: i for i, s in enumerate(subj_set)}

    for pi, sr in enumerate(session_results):
        dx, dy = float(prompt_coords[pi, 0]), float(prompt_coords[pi, 1])
        cat = (sr.get("category", "") or "?")[0]
        toks = sr.get("tokens", [])
        rd = sr.get("rank_displacement", {})
        per_pos = rd.get("per_position", []) if rd else []
        stress = sr.get("per_token_stress", [])
        sfd = sr.get("sfd", {})
        sfd_d = sfd.get("per_token_density", []) if sfd else []

        # Per-token embeddings at both depths (if available)
        ptde = sr.get("per_token_domain_emb")       # subject layer
        ptee = sr.get("per_token_escalation_emb")    # escalation layer
        ptde_offset = sr.get("per_token_domain_offset", 0)

        for pos in range(len(per_pos)):
            if pos >= len(toks):
                continue
            tok = toks[pos].strip()
            if not tok:
                continue
            token_freq[tok] += 1

            pd = per_pos[pos]
            adj = pos - ptde_offset
            raw_obs.append({
                "tok": tok, "cat": cat,
                "dx": dx, "dy": dy,
                "disp": float(pd.get("total_disp", 0)),
                "repl": float(pd.get("replacement_ratio", 0)),
                "asm": float(stress[pos]) if pos < len(stress) else 0,
                "sfd_d": float(sfd_d[pos]) if pos < len(sfd_d) else 0,
                "pi": pi, "pos": pos,
                "_emb": ptde[adj] if ptde and pos >= ptde_offset and adj < len(ptde) else None,
                "_esc_emb": ptee[adj] if ptee and pos >= ptde_offset and adj < len(ptee) else None,
            })

    # Top tokens by frequency, filter by min appearances, compute CV, order by CV
    qualified = {t: n for t, n in token_freq.items() if n >= min_appearances}

    # If token variance eta² is available, weight selection so category-
    # dependent tokens (high eta_sq = content words) rank above category-
    # independent tokens (low eta_sq = function words).
    # score = freq × (eta² + floor)
    if tv_eta_weights:
        top = sorted(qualified.keys(),
                     key=lambda t: -(qualified[t] * (tv_eta_weights.get(t, eta_floor) + eta_floor)))[:top_n]
    else:
        top = sorted(qualified.keys(), key=lambda t: -qualified[t])[:top_n]
    token_cv = {}
    for tok in top:
        disps = [o["disp"] for o in raw_obs if o["tok"] == tok]
        if len(disps) >= 2:
            m = np.mean(disps)
            token_cv[tok] = round(float(np.std(disps) / max(abs(m), NORM_EPS)), 3)
        else:
            token_cv[tok] = 0
    ordered = sorted(top, key=lambda t: token_cv.get(t, 0))

    # Filter and compute proximity
    top_set = set(ordered)
    subj_idx = {s: i for i, s in enumerate(subjects)}
    obs_export = []

    # Precompute probe embedding matrices
    subj_probe_mat = np.array(probe_embs) if probe_embs is not None else None
    esc_probe_mat = np.array(esc_probe_embs) if esc_probe_embs is not None else subj_probe_mat

    # Subject angles for continuous positioning
    n_subj = len(subjects)
    _subj_angles = np.linspace(0, 2 * np.pi, n_subj, endpoint=False) - np.pi / 2

    token_knn = probe_neighbors

    # Per-probe engagement accumulator. Records, for each probe index appearing
    # in any token's top-k, the distances (1 - cos_sim) and category labels of
    # engaging token observations. Used to build the "Probes by CV" table:
    # CV is computed over engagement distances.
    probe_engagement = defaultdict(lambda: {"distances": [], "cats": []})

    # Bipartite engagement accumulator. Same data, keyed by (prompt_idx,
    # probe_idx) instead of just probe_idx — preserves which prompt
    # engaged which probe so we can build the ladder graph payload.
    bipartite = defaultdict(lambda: {"distances": []})

    for o in raw_obs:
        if o["tok"] not in top_set:
            continue

        subj_emb = o.get("_emb")
        esc_emb = o.get("_esc_emb")

        # ── Subject assignment (kNN from subject-layer embedding) ──
        near_angle = 0.0
        if subj_emb is not None and subj_probe_mat is not None and probes is not None:
            tok_vec = np.array(subj_emb, dtype=np.float32)
            tok_norm = np.linalg.norm(tok_vec)
            if tok_norm > NORM_EPS:
                tok_vec = tok_vec / tok_norm
            sims = subj_probe_mat @ tok_vec

            # Top-k nearest probes
            k = min(token_knn, len(sims))
            top_k = np.argsort(sims)[-k:][::-1]
            top_sims = sims[top_k]
            best_dist = float(1.0 - top_sims[0])

            # Record engagement: every probe in top-k counts as engaged by
            # this observation. Distance recorded is per-probe (not the
            # observation's best_dist), so each probe gets its own
            # distribution.
            for idx, sim in zip(top_k, top_sims):
                if idx < len(probes):
                    dist = float(1.0 - sim)
                    probe_engagement[int(idx)]["distances"].append(dist)
                    probe_engagement[int(idx)]["cats"].append(o["cat"])
                    # Bipartite: track per-(prompt, probe) too.
                    bipartite[(o["pi"], int(idx))]["distances"].append(dist)

            # Similarity-weighted position
            weights = np.exp(top_sims * knn_sharpness)
            weights /= weights.sum()

            sin_sum = 0
            cos_sum = 0
            subj_w = defaultdict(float)
            for idx, w in zip(top_k, weights):
                if idx < len(probes):
                    si = subj_idx.get(probes[idx]["subject"], 0)
                    sin_sum += w * np.sin(_subj_angles[si])
                    cos_sum += w * np.cos(_subj_angles[si])
                    subj_w[si] += w

            near_angle = float(np.arctan2(sin_sum, cos_sum))
            near_subj = max(subj_w, key=subj_w.get) if subj_w else 0
        else:
            # Fallback: prompt-level PCA
            aidx, best_dist = _nearest_probe(o["dx"], o["dy"], anchor_pts)
            near_subj = subj_idx.get(anchor_pts[aidx]["subject"], 0)
            near_angle = float(_subj_angles[near_subj])

        # ── Escalation assignment (kNN from escalation-layer embedding) ──
        if esc_emb is not None and esc_probe_mat is not None and probes is not None:
            esc_vec = np.array(esc_emb, dtype=np.float32)
            esc_norm = np.linalg.norm(esc_vec)
            if esc_norm > NORM_EPS:
                esc_vec = esc_vec / esc_norm
            esc_sims = esc_probe_mat @ esc_vec

            k = min(token_knn, len(esc_sims))
            esc_top_k = np.argsort(esc_sims)[-k:][::-1]
            esc_weights = np.exp(esc_sims[esc_top_k] * knn_sharpness)
            esc_weights /= esc_weights.sum()
            level = float(sum(probes[idx]["level"] * w
                              for idx, w in zip(esc_top_k, esc_weights)
                              if idx < len(probes)))
        elif subj_emb is not None and subj_probe_mat is not None and probes is not None:
            # Use subject-layer kNN for level too
            level = float(sum(probes[idx]["level"] * w
                              for idx, w in zip(top_k, weights)
                              if idx < len(probes)))
        else:
            # Fallback: prompt-level PCA
            aidx, _ = _nearest_probe(o["dx"], o["dy"], anchor_pts)
            level = anchor_pts[aidx]["level"]

        obs_export.append([
            o["tok"], o["cat"],
            round(o["dy"], 4), round(o["disp"], 3), round(o["repl"], 2),
            round(o["dx"], 4),
            round(o["asm"], 2), round(o["sfd_d"], 3),
            o["pi"], o["pos"],
            round(best_dist, 4), level, near_subj,
            round(near_angle, 4),  # index 14: kNN-weighted continuous angle
        ])

    if progress:
        progress(f"Built {len(obs_export)} observations for {len(ordered)} tokens")

    # Per-probe aggregates: one row per probe in the active set, including
    # those never engaged (count=0, dimmed in the UI). Distance distribution
    # is over (token observation, probe) pairs where the probe appeared in
    # the token's top-k similarity ranking.
    probe_rows = []
    if probes is not None:
        for pi, probe in enumerate(probes):
            stats = probe_engagement.get(pi)
            if stats and stats["distances"]:
                dists = np.array(stats["distances"], dtype=np.float64)
                cats = stats["cats"]
                mean_d = float(dists.mean())
                std_d = float(dists.std())
                cv = (std_d / mean_d) if mean_d > NORM_EPS else 0.0
                cat_mix = {
                    "b": cats.count("b"),
                    "m": cats.count("m"),
                    "h": cats.count("h"),
                    "j": cats.count("j"),
                }
            else:
                mean_d = 0.0
                cv = 0.0
                cat_mix = {"b": 0, "m": 0, "h": 0, "j": 0}
                dists = np.array([])

            probe_rows.append({
                "subject": probe["subject"],
                "level": int(probe.get("level", 0)),
                "text": probe["text"][:PROBE_TEXT_DISPLAY_LEN],
                "n": int(len(dists)),
                "mean_dist": round(mean_d, 4),
                "cv": round(cv, 4),
                "cat_mix": cat_mix,
            })

    return obs_export, ordered, token_cv, probe_rows, bipartite


def _stratification(obs, subjects):
    """Compute category counts by nearest level and subject."""
    by_level = defaultdict(lambda: defaultdict(int))
    by_subject = defaultdict(lambda: defaultdict(int))

    for o in obs:
        cat = o[1]
        by_level[int(round(o[11]))][cat] += 1
        if o[12] < len(subjects):
            by_subject[subjects[o[12]]][cat] += 1

    return {
        "by_level": {str(k): dict(v) for k, v in sorted(by_level.items())},
        "by_subject": {k: dict(v) for k, v in sorted(by_subject.items())},
    }


def _build_ladder(bipartite, probes, prompts_meta, subjects, level_names,
                   top_k_storage=100, contributors_max=5):
    """Build the ladder-graph payload from the bipartite engagement matrix.

    bipartite:    dict[(prompt_idx, probe_idx)] -> {"distances": [float, ...]}
                  populated in _build_observations from token-knn engagements.
    probes:       list of probe dicts (active set, in lattice order).
    prompts_meta: list of session_results entries, in their projected order.
    subjects:     list of subject names, ordered.
    level_names:  list of level names ordered by level index.

    Returns a dict suitable for direct serialization into the module
    output's "ladder" field. See spec for shape.

    Server pre-computes:
      - The two metric values per (prompt, probe): engagement_count + similarity
      - Top-k_storage ranked left-items per right-item, for all four
        (prompt aggregated yes/no × probe aggregated yes/no) combinations
      - Contributor lists for aggregated views (top contributors_max)
      - Default item orderings on both axes

    The popout slices to user-chosen k from the stored ranked list.
    """
    n_probes = len(probes)
    n_prompts = len(prompts_meta)
    subj_idx = {s: i for i, s in enumerate(subjects)}

    # ── Step 1: Build dense (n_prompts, n_probes) score matrices ──
    # engagement_count[pi][bi] = number of token-knn engagements
    # similarity[pi][bi]      = 1 - mean_distance (higher = closer in space)
    # We keep them as nested dicts to stay sparse — most cells will be 0.
    by_prompt_count = defaultdict(dict)        # pi -> {bi: count}
    by_prompt_sim   = defaultdict(dict)        # pi -> {bi: similarity}
    for (pi, bi), rec in bipartite.items():
        dists = rec["distances"]
        if not dists:
            continue
        by_prompt_count[pi][bi] = len(dists)
        by_prompt_sim[pi][bi]   = 1.0 - (sum(dists) / len(dists))

    # ── Step 2: Build axis item lists for both granularities ──
    left_individual_items = []
    for bi, p in enumerate(probes):
        left_individual_items.append({
            "id":          bi,
            "label":       (p.get("text") or "")[:PROBE_TEXT_DISPLAY_LEN],
            "subject":     p.get("subject", ""),
            "subject_idx": subj_idx.get(p.get("subject", ""), 0),
            "level":       int(p.get("level", 0)),
        })

    # Aggregated left = subject × level cells
    cell_to_probes = defaultdict(list)         # (si, li) -> [bi, ...]
    for bi, p in enumerate(probes):
        si = subj_idx.get(p.get("subject", ""), 0)
        li = int(p.get("level", 0))
        cell_to_probes[(si, li)].append(bi)

    left_aggregated_items = []
    for (si, li), bis in sorted(cell_to_probes.items()):
        subj_name = subjects[si] if si < len(subjects) else f"s{si}"
        lvl_name  = level_names[li] if li < len(level_names) else f"L{li}"
        left_aggregated_items.append({
            "id":          f"{si}_{li}",
            "label":       f"{subj_name} / {lvl_name}",
            "subject_idx": si,
            "level":       li,
            "n_probes":    len(bis),
        })

    # Right: individual prompts and aggregated categories
    right_individual_items = []
    cat_to_prompts = defaultdict(list)
    for pi, sr in enumerate(prompts_meta):
        cat = (sr.get("category", "") or "?")[0]
        cat_to_prompts[cat].append(pi)
        text = (sr.get("prompt") or "")
        right_individual_items.append({
            "id":       pi,
            "label":    text[:PROMPT_TEXT_DISPLAY_LEN],
            "category": cat,
            "n_tokens": len(sr.get("tokens", []) or []),
        })

    cat_names = {"b": "benign", "m": "mild", "h": "harmful",
                 "j": "jailbreak", "a": "adversarial",
                 "d": "dual-use", "u": "unknown", "?": "unknown"}
    right_aggregated_items = []
    for cat in sorted(cat_to_prompts.keys()):
        right_aggregated_items.append({
            "id":        cat,
            "label":     cat_names.get(cat, cat),
            "category":  cat,
            "n_prompts": len(cat_to_prompts[cat]),
        })

    # ── Step 3: Helper to rank+slice a flat dict[left_id -> score] ──
    def _rank_slice(scores, top_k):
        """scores: dict[left_id -> float]. Returns sorted [(id, score), ...]."""
        if not scores:
            return []
        items = sorted(scores.items(), key=lambda x: -x[1])
        return items[:top_k]

    # Tooltip metadata helpers — populate both metrics regardless of which
    # is the displayed score, so swapping metrics in the popout doesn't
    # require re-fetching.
    def _tooltip_meta_individual(pi, bi):
        return {
            "engagement_count": by_prompt_count.get(pi, {}).get(bi, 0),
            "mean_proximity":   round(by_prompt_sim.get(pi, {}).get(bi, 0.0), 4),
        }

    # ── Step 4: PI_PI — prompts (individual) → probes (individual) ──
    PI_PI = []
    for pi in range(n_prompts):
        scores = by_prompt_count.get(pi, {})
        ranked = []
        for rank, (bi, score) in enumerate(_rank_slice(scores, top_k_storage), 1):
            ranked.append({
                "left_id":      bi,
                "score":        float(score),
                "rank":         rank,
                "tooltip_meta": _tooltip_meta_individual(pi, bi),
            })
        PI_PI.append({"right_id": pi, "ranked": ranked})

    # ── Step 5: PI_PA — prompts (individual) → cells (aggregated) ──
    # Cell score = sum of probe scores in that cell, for that prompt.
    PI_PA = []
    for pi in range(n_prompts):
        prompt_scores = by_prompt_count.get(pi, {})
        if not prompt_scores:
            PI_PA.append({"right_id": pi, "ranked": []})
            continue
        cell_scores = {}
        cell_contributors = defaultdict(list)  # cell_id -> [(bi, score)]
        for (si, li), bis in cell_to_probes.items():
            cell_id = f"{si}_{li}"
            total = 0.0
            for bi in bis:
                s = prompt_scores.get(bi, 0)
                if s > 0:
                    total += s
                    cell_contributors[cell_id].append((bi, s))
            if total > 0:
                cell_scores[cell_id] = total
        ranked = []
        for rank, (cell_id, score) in enumerate(
                _rank_slice(cell_scores, top_k_storage), 1):
            contribs = sorted(cell_contributors[cell_id], key=lambda x: -x[1])
            contribs = contribs[:contributors_max]
            ranked.append({
                "left_id": cell_id,
                "score":   float(score),
                "rank":    rank,
                "contributors": [
                    {
                        "probe_idx": bi,
                        "label":     left_individual_items[bi]["label"],
                        "score":     int(s) if isinstance(s, int) else float(s),
                    }
                    for bi, s in contribs
                ],
            })
        PI_PA.append({"right_id": pi, "ranked": ranked})

    # ── Step 6: PA_PI — categories (aggregated) → probes (individual) ──
    # Category score per probe = mean of its prompts' scores for that probe.
    PA_PI = []
    for cat in sorted(cat_to_prompts.keys()):
        member_pids = cat_to_prompts[cat]
        if not member_pids:
            PA_PI.append({"right_id": cat, "ranked": []})
            continue
        all_bids = set()
        for pi in member_pids:
            all_bids.update(by_prompt_count.get(pi, {}).keys())
        probe_scores = {}
        probe_contributors = defaultdict(list)  # bi -> [(pi, score)]
        for bi in all_bids:
            scores_per_pi = []
            for pi in member_pids:
                s = by_prompt_count.get(pi, {}).get(bi, 0)
                scores_per_pi.append(s)
                if s > 0:
                    probe_contributors[bi].append((pi, s))
            if scores_per_pi:
                probe_scores[bi] = sum(scores_per_pi) / len(scores_per_pi)
        ranked = []
        for rank, (bi, score) in enumerate(
                _rank_slice(probe_scores, top_k_storage), 1):
            contribs = sorted(probe_contributors[bi], key=lambda x: -x[1])
            contribs = contribs[:contributors_max]
            ranked.append({
                "left_id": bi,
                "score":   round(float(score), 4),
                "rank":    rank,
                "contributors": [
                    {
                        "prompt_idx": pi,
                        "label":      right_individual_items[pi]["label"],
                        "score":      int(s) if isinstance(s, int) else float(s),
                    }
                    for pi, s in contribs
                ],
            })
        PA_PI.append({"right_id": cat, "ranked": ranked})

    # ── Step 7: PA_PA — categories (aggregated) → cells (aggregated) ──
    # Cell score per category = mean across the category's prompts of
    # (sum of probe scores in that cell for that prompt).
    PA_PA = []
    for cat in sorted(cat_to_prompts.keys()):
        member_pids = cat_to_prompts[cat]
        if not member_pids:
            PA_PA.append({"right_id": cat, "ranked": []})
            continue
        cell_score_per_pi = defaultdict(list)  # cell_id -> [score per pi]
        cell_contributors = defaultdict(list)  # cell_id -> [(pi, score)]
        for (si, li), bis in cell_to_probes.items():
            cell_id = f"{si}_{li}"
            for pi in member_pids:
                ps = by_prompt_count.get(pi, {})
                total = sum(ps.get(bi, 0) for bi in bis)
                cell_score_per_pi[cell_id].append(total)
                if total > 0:
                    cell_contributors[cell_id].append((pi, total))
        cell_scores = {}
        for cell_id, vals in cell_score_per_pi.items():
            if vals:
                m = sum(vals) / len(vals)
                if m > 0:
                    cell_scores[cell_id] = m
        ranked = []
        for rank, (cell_id, score) in enumerate(
                _rank_slice(cell_scores, top_k_storage), 1):
            contribs = sorted(cell_contributors[cell_id], key=lambda x: -x[1])
            contribs = contribs[:contributors_max]
            ranked.append({
                "left_id": cell_id,
                "score":   round(float(score), 4),
                "rank":    rank,
                "contributors": [
                    {
                        "prompt_idx": pi,
                        "label":      right_individual_items[pi]["label"],
                        "score":      int(s) if isinstance(s, int) else float(s),
                    }
                    for pi, s in contribs
                ],
            })
        PA_PA.append({"right_id": cat, "ranked": ranked})

    n_active = sum(1 for bi in range(n_probes)
                    if any(bi in by_prompt_count.get(pi, {})
                           for pi in range(n_prompts)))

    return {
        "metric":           "engagement_count",
        "metrics_available": ["engagement_count", "mean_proximity"],
        "n_prompts_projected": n_prompts,
        "n_probes_active":     n_active,
        "categories":          sorted(cat_to_prompts.keys()),
        "category_names":      {c: cat_names.get(c, c)
                                for c in cat_to_prompts.keys()},
        "left_individual": {
            "axis":  "probes",
            "items": left_individual_items,
        },
        "left_aggregated": {
            "axis":  "cells",
            "items": left_aggregated_items,
        },
        "right_individual": {
            "axis":  "prompts",
            "items": right_individual_items,
        },
        "right_aggregated": {
            "axis":  "categories",
            "items": right_aggregated_items,
        },
        "engagements": {
            "PI_PI": PI_PI,
            "PI_PA": PI_PA,
            "PA_PI": PA_PI,
            "PA_PA": PA_PA,
        },
        "top_k_storage":      top_k_storage,
        "contributors_max":   contributors_max,
    }


# ─── Module Class ────────────────────────────────────────────

class DomainSurfaceModule(TASMModule):
    """Subject-matter domain surface analysis.

    Pure post-processor: uses pre-computed domain_embedding fields
    from the analyzer and cached probe embeddings from model load.
    """

    name = "domain_surface"
    display_name = "Domain Surface Geometry"
    description = (
        "Maps per-token correction signals onto a subject-matter domain "
        "surface defined by configurable probes. Reveals how alignment "
        "training treats the same token across different topics and "
        "discourse frames."
    )
    version = "0.2.0"

    min_results = 10
    requires_sfd = True
    requires_ltp = False
    requires_rd = True

    def __init__(self):
        super().__init__()
        self._project_root = None
        self._pipeline = None

    def set_project_root(self, root):
        """Set project root."""
        self._project_root = root

    def set_pipeline(self, pipeline):
        """Receive the pipeline reference so validate() can confirm the
        active probe set was applied for the currently loaded model."""
        self._pipeline = pipeline

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="top_tokens",
                display_name="Top Tokens",
                description="Number of most-frequent tokens to include",
                type="int",
                default=100,
                min_val=5,
            ),
            ModuleParameter(
                name="min_appearances",
                display_name="Min Appearances",
                description="Minimum times a token must appear across prompts to be included",
                type="int",
                default=2,
                min_val=1,
                max_val=20,
            ),
            ModuleParameter(
                name="probe_neighbors",
                display_name="Probe Neighbors (k)",
                description=(
                    "Number of nearest probes to consider when assigning "
                    "each token to a cell in the lattice. Higher values "
                    "produce smoother assignments but dilute sharp boundaries. "
                    "Lower values give crisper cell assignments but are more "
                    "sensitive to probe placement."
                ),
                type="int",
                default=5,
                min_val=1,
                max_val=20,
            ),
            ModuleParameter(
                name="knn_sharpness",
                display_name="kNN Sharpness",
                description=(
                    "Temperature multiplier for similarity-weighted probe "
                    "matching. Controls how aggressively the nearest probe "
                    "dominates the weighted assignment. Higher values make "
                    "the best-matching probe almost fully determine cell "
                    "assignment (hard kNN). Lower values spread weight more "
                    "evenly across the k neighbors (soft kNN)."
                ),
                type="float",
                default=10.0,
                min_val=1.0,
                max_val=50.0,
            ),
            ModuleParameter(
                name="eta_floor",
                display_name="Eta² Floor",
                description=(
                    "Minimum eta-squared weight applied during token "
                    "selection. Prevents tokens with negligible category-"
                    "dependence from being completely suppressed. Higher "
                    "values reduce the influence of η² weighting, making "
                    "selection closer to pure frequency."
                ),
                type="float",
                default=0.01,
                min_val=0.001,
                max_val=0.5,
            ),
            ModuleParameter(
                name="tv_min_appearances",
                display_name="η² Min Token Appearances",
                description=(
                    "Minimum interior appearances for a token to be assigned "
                    "a category-dependence (η²) weight. Tokens that appear "
                    "fewer times keep a default weight (no boost). Higher "
                    "values mean only well-attested tokens get boosted; "
                    "lower values let rarer tokens influence selection."
                ),
                type="int",
                default=3,
                min_val=2,
                max_val=20,
            ),
        ]

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        # Check that domain embeddings are present
        n_with_emb = sum(1 for r in session_results
                         if r.get("domain_embedding") is not None)
        if n_with_emb == 0:
            return False, (
                "No domain embeddings found in session results. "
                "Re-run analysis to capture domain embeddings."
            )

        active = get_active_probe_set(self._project_root)
        if active is None:
            return False, (
                "No probe set active. Apply a probe set in the "
                "Configuration tab first."
            )
        ok, msg = active.validate_against(self._pipeline)
        if not ok:
            return False, msg
        path = os.path.join(self._project_root, active.probe_file)
        if not os.path.exists(path):
            return False, f"Active probe file not found: {active.probe_file}"

        return True, "OK"

    def run(self, session_results, params, progress=None):
        """Execute domain surface analysis using pre-computed embeddings."""
        active = get_active_probe_set(self._project_root)
        if active is None:
            raise RuntimeError("No probe set active. Apply one in "
                               "Configuration → Probe Set.")
        ok, msg = active.validate_against(self._pipeline)
        if not ok:
            raise RuntimeError(msg)
        probe_file = active.probe_file
        top_tokens = params.get("top_tokens", 100)
        min_appearances = params.get("min_appearances", 2)
        pca_components = params.get("pca_components", 2)
        probe_neighbors = int(params.get("probe_neighbors", 5))
        knn_sharpness = float(params.get("knn_sharpness", 10.0))
        eta_floor = float(params.get("eta_floor", 0.01))
        tv_min_appearances = int(params.get("tv_min_appearances", 3))

        # Resolve probe path
        if self._project_root:
            probe_path = os.path.join(self._project_root, probe_file)
        else:
            probe_path = probe_file

        # Load probes
        if progress:
            progress(f"Loading probes from {probe_file}")
        probes = load_probes(probe_path)
        _, level_names = detect_level_cols(probe_path)
        subjects = sorted(set(p["subject"] for p in probes))
        logger.info(f"[DOMAIN] Loaded {len(probes)} probes across "
                     f"{len(subjects)} subjects, {len(level_names)} levels: {level_names}")

        # Load prompt embeddings from session results
        if progress:
            progress("Loading pre-computed domain embeddings...")
        prompt_embs = []
        valid_indices = []
        for i, sr in enumerate(session_results):
            emb = sr.get("domain_embedding")
            if emb is not None:
                prompt_embs.append(emb)
                valid_indices.append(i)

        if len(prompt_embs) < self.min_results:
            raise RuntimeError(
                f"Only {len(prompt_embs)} results have domain embeddings "
                f"(need {self.min_results}). Re-run analysis to capture them."
            )

        prompt_embs = np.array(prompt_embs)
        logger.info(f"[DOMAIN] {len(prompt_embs)} prompt embeddings loaded")

        # Build index map: position in prompt_embs → original session index
        # so observations reference correct prompts
        session_subset = [session_results[i] for i in valid_indices]

        # ── Cached probe embeddings via the active-set resolver ──
        if progress:
            progress("Loading cached probe embeddings...")
        probe_embs = self._load_probe_embeddings(active)
        if probe_embs is None:
            raise RuntimeError(
                f"Probe cache missing for {active.probe_file!r} at "
                f"L{int(active.subject_layer_frac()*100)}. Re-Apply the "
                f"probe set in Configuration → Probe Set to regenerate it."
            )

        if len(probe_embs) != len(probes):
            raise RuntimeError(
                f"Probe embedding count ({len(probe_embs)}) does not match "
                f"probe count ({len(probes)}). Cache may be stale — "
                f"Re-Apply the probe set to regenerate."
            )

        probe_embs = np.array(probe_embs)

        # Load escalation-layer probe embeddings (for split-depth ring assignment)
        subj_frac = active.subject_layer_frac()
        esc_frac = active.escalation_layer_frac()

        esc_probe_embs = None
        if esc_frac != subj_frac:
            if progress:
                progress("Loading escalation-layer probe embeddings...")
            esc_raw = self._load_probe_embeddings(active, layer_frac=esc_frac)
            if esc_raw is not None and len(esc_raw) == len(probes):
                esc_probe_embs = np.array(esc_raw)
                logger.info(f"[DOMAIN] Split-depth: escalation probes from "
                            f"L{int(esc_frac*100)}, subject probes from L{int(subj_frac*100)}")
            else:
                logger.warning(f"[DOMAIN] Escalation probe cache not found at L{int(esc_frac*100)}, "
                               f"using single-depth matching")

        # Co-fit PCA
        if progress:
            progress("Fitting PCA...")
        prompt_coords, probe_coords, variance = _cofit_pca(
            prompt_embs, probe_embs, n_components=pca_components)
        logger.info(f"[DOMAIN] PCA variance: {variance}")

        # Build anchor points
        anchor_pts = []
        for i, p in enumerate(probes):
            anchor_pts.append({
                "subject": p["subject"],
                "anchor_id": p.get("anchor_id", ""),
                "level": p["level"],
                "text": p["text"][:PROBE_TEXT_DISPLAY_LEN],
                "x": float(probe_coords[i, 0]),
                "y": float(probe_coords[i, 1]),
            })

        # Build observations
        if progress:
            progress("Building per-token observations...")

        # Compute per-token η² (category-dependence on density) directly
        # from the session. This was previously read from a separate
        # Token Variance module's JSON output; now it's inlined so
        # Domain Surface is self-sufficient. The signal is the same:
        # tokens whose density correlates with category rank above
        # tokens whose density looks the same in every category.
        if progress:
            progress("Computing token category-dependence weights...")
        tv_weights = _compute_token_eta_density(
            session_subset,
            min_appearances=tv_min_appearances,
            min_seq_len=3,
            exclude_first=True,
        )
        if not tv_weights:
            tv_weights = None
            logger.info("[DOMAIN] No per-token density signal in session — "
                        "using frequency-only token selection.")
        else:
            logger.info(f"[DOMAIN] Computed token category-dependence "
                        f"weights for {len(tv_weights)} tokens")

        obs, ordered_tokens, token_cv, probe_rows, bipartite = _build_observations(
            session_subset, prompt_coords, anchor_pts,
            subjects, top_tokens, min_appearances, progress,
            probe_embs=probe_embs, probes=probes,
            esc_probe_embs=esc_probe_embs,
            tv_eta_weights=tv_weights,
            probe_neighbors=probe_neighbors,
            knn_sharpness=knn_sharpness,
            eta_floor=eta_floor)

        # Stratification
        strat = _stratification(obs, subjects)

        # Log stratification summary
        if progress:
            for level_idx in sorted(strat["by_level"].keys()):
                counts = strat["by_level"][level_idx]
                total = sum(counts.values())
                li = int(round(float(level_idx)))
                if li < len(level_names):
                    progress(f"  {level_names[li]}: {total} obs "
                             f"(b={counts.get('b',0)} m={counts.get('m',0)} "
                             f"h={counts.get('h',0)} j={counts.get('j',0)})")

        # Compact anchors for output
        subj_idx = {s: i for i, s in enumerate(subjects)}
        anchors_compact = [{
            "s": subj_idx[a["subject"]],
            "l": a["level"],
            "t": a["text"],
            "x": round(a["x"], 4),
            "y": round(a["y"], 4),
        } for a in anchor_pts]

        # Prompt texts (truncated)
        prompts = [r["prompt"][:PROMPT_TEXT_DISPLAY_LEN] for r in session_subset]

        # Build ladder-graph payload (bipartite prompt × probe engagement)
        if progress:
            progress("Building ladder graph payload...")
        ladder = _build_ladder(bipartite, probes, session_subset,
                                subjects, level_names)

        # Build output
        output = {
            "pca": variance,
            "pca_components": pca_components,
            "layer": "middle",
            "n_prompts_used": len(prompt_embs),
            "n_prompts_total": len(session_results),
            "min_appearances": min_appearances,
            "subjects": subjects,
            "tokens": ordered_tokens,
            "token_cv": token_cv,
            "anchors": anchors_compact,
            "observations": obs,
            "prompts": prompts,
            "fields": [
                "tok", "cat", "dy", "disp", "repl", "dx",
                "asm", "sfd_d", "pi", "pos",
                "near_dist", "near_level", "near_subj_idx",
                "near_angle",
            ],
            "probes": probe_rows,
            "probe_fields": [
                "subject", "level", "text", "n",
                "mean_dist", "cv", "cat_mix",
            ],
            "stratification": strat,
            "probe_file": probe_file,
            "level_names": level_names,
            "ladder": ladder,
        }

        if progress:
            progress(f"Complete: {len(obs)} observations, "
                     f"{len(anchors_compact)} anchors, "
                     f"{len(subjects)} subjects")

        return output

    def _load_probe_embeddings(self, active, layer_frac=None):
        """Load cached probe embeddings via the active-set resolver.

        Resolves the cache path exactly from (probe_file, model_id,
        layer_frac, projected) — no directory scanning. The active-set
        encodes the binding established at apply time, so this is the
        only sound way to find the right cache.

        Args:
            active: ActiveProbeSet from get_active_probe_set().
            layer_frac: Layer depth to load. Defaults to the active
                set's subject depth.

        Returns the embeddings list, or None if the cache file is
        missing on disk.
        """
        if active is None:
            return None
        if layer_frac is None:
            layer_frac = active.subject_layer_frac()

        cache_path = active.cache_path(self._project_root, layer_frac)
        cache = load_probe_cache(cache_path)
        if cache is None:
            logger.warning(
                f"[DOMAIN] Probe cache missing at {cache_path}. "
                f"Re-Apply the probe set to regenerate it.")
            return None

        embs = cache.get("embeddings", [])
        if not embs:
            return None

        logger.info(f"[DOMAIN] Using probe cache: {os.path.basename(cache_path)} "
                    f"(model={cache.get('model_id', '?')}, "
                    f"layer={cache.get('layer', '?')}, "
                    f"frac={cache.get('layer_frac', '?')})")
        return embs
