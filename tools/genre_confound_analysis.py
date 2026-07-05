#!/usr/bin/env python3
"""
Genre confound analysis — computes effective rank, angular concentration,
and transfer proximity from benchmark harness JSONL output.

Usage:
  python tools/genre_confound_analysis.py benchmark_out/genre_confound/results.jsonl

Reads the per-token density traces, computes per-cell geometry, and
evaluates kill conditions GC-1 through GC-5.
"""

import json
import sys
import numpy as np
from collections import defaultdict


def load_results(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "error" not in rec and rec.get("per_token_density"):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def effective_rank(vectors):
    """Effective rank of a set of vectors via SVD spectral entropy."""
    if len(vectors) < 2:
        return float("nan")
    X = np.array(vectors)
    X = X - X.mean(axis=0)  # center
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def angular_concentration(vectors):
    """Mean pairwise cosine similarity and angular std."""
    if len(vectors) < 2:
        return float("nan"), float("nan")
    X = np.array(vectors)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_normed = X / norms
    cosines = X_normed @ X_normed.T
    # Extract upper triangle (exclude diagonal)
    idx = np.triu_indices(len(vectors), k=1)
    pairwise = cosines[idx]
    mean_cos = float(np.mean(pairwise))
    # Angular std: std of arccos(cosine)
    angles = np.arccos(np.clip(pairwise, -1, 1))
    ang_std = float(np.std(angles))
    return mean_cos, ang_std


def cohens_d(group_a, group_b):
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    pooled_var = (np.var(group_a) + np.var(group_b)) / 2
    if pooled_var < 1e-20:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / np.sqrt(pooled_var))


def get_vector(rec):
    """Use per-token density trace as the prompt's representation vector."""
    return rec["per_token_density"]


def pad_vectors(vectors):
    """Pad to equal length for matrix operations."""
    if not vectors:
        return []
    max_len = max(len(v) for v in vectors)
    return [v + [0.0] * (max_len - len(v)) for v in vectors]


def main():
    if len(sys.argv) < 2:
        print("Usage: python genre_confound_analysis.py <results.jsonl>")
        sys.exit(1)

    records = load_results(sys.argv[1])
    print(f"Loaded {len(records)} records.\n")

    # Split by experiment type
    core = [r for r in records if r.get("category") != "principle"
            and r.get("harmfulness") != "transfer"
            and r.get("category") != "transfer"]
    transfers = [r for r in records if r.get("harmfulness") == "transfer"
                 or r.get("category") == "transfer"]
    principles = [r for r in records if r.get("category") == "principle"]

    registers = ["news", "technical", "conversational", "literary", "academic"]
    harm_levels = ["benign", "harmful"]

    # =====================================================
    # §5.1 — Primary analysis: what does concentration track?
    # =====================================================
    print("=" * 60)
    print("§5.1 — Effective rank by cell")
    print("=" * 60)

    cells = defaultdict(list)
    for r in core:
        reg = r.get("register", "")
        harm = r.get("harmfulness", r.get("category", ""))
        if reg in registers and harm in harm_levels:
            cells[(harm, reg)].append(get_vector(r))

    print(f"\n{'Register':<16} {'Harm erank':>11} {'Benign erank':>13} {'Diff':>8}")
    print("-" * 52)

    gc1_failures = 0
    for reg in registers:
        h_vecs = pad_vectors(cells.get(("harmful", reg), []))
        b_vecs = pad_vectors(cells.get(("benign", reg), []))
        h_er = effective_rank(h_vecs) if len(h_vecs) >= 2 else float("nan")
        b_er = effective_rank(b_vecs) if len(b_vecs) >= 2 else float("nan")
        diff = h_er - b_er if not (np.isnan(h_er) or np.isnan(b_er)) else float("nan")
        print(f"{reg:<16} {h_er:>11.2f} {b_er:>13.2f} {diff:>8.2f}")
        # GC-1: harm should have LOWER rank than benign in every register
        if not np.isnan(diff) and diff >= 0:
            gc1_failures += 1

    # Cross-genre effective rank
    all_harm_vecs = pad_vectors([get_vector(r) for r in core
                                  if r.get("harmfulness") == "harmful"])
    all_benign_vecs = pad_vectors([get_vector(r) for r in core
                                    if r.get("harmfulness") == "benign"])

    cross_harm_er = effective_rank(all_harm_vecs)
    cross_benign_er = effective_rank(all_benign_vecs)
    print(f"\n{'ALL (cross-genre)':<16} {cross_harm_er:>11.2f} {cross_benign_er:>13.2f} "
          f"{cross_harm_er - cross_benign_er:>8.2f}")

    # =====================================================
    # §5.2 — Angular concentration
    # =====================================================
    print(f"\n{'=' * 60}")
    print("§5.2 — Angular concentration by cell")
    print("=" * 60)

    print(f"\n{'Register':<16} {'Harm cos':>9} {'Harm σ':>8} {'Ben cos':>9} {'Ben σ':>8}")
    print("-" * 54)

    for reg in registers:
        h_vecs = pad_vectors(cells.get(("harmful", reg), []))
        b_vecs = pad_vectors(cells.get(("benign", reg), []))
        h_cos, h_sig = angular_concentration(h_vecs)
        b_cos, b_sig = angular_concentration(b_vecs)
        print(f"{reg:<16} {h_cos:>9.4f} {h_sig:>8.4f} {b_cos:>9.4f} {b_sig:>8.4f}")

    h_cos_all, h_sig_all = angular_concentration(pad_vectors(
        [get_vector(r) for r in core if r.get("harmfulness") == "harmful"]))
    b_cos_all, b_sig_all = angular_concentration(pad_vectors(
        [get_vector(r) for r in core if r.get("harmfulness") == "benign"]))
    print(f"\n{'ALL':<16} {h_cos_all:>9.4f} {h_sig_all:>8.4f} {b_cos_all:>9.4f} {b_sig_all:>8.4f}")

    # =====================================================
    # §5.3 — Transfer condition
    # =====================================================
    print(f"\n{'=' * 60}")
    print("§5.3 — Transfer proximity")
    print("=" * 60)

    if transfers and all_harm_vecs and all_benign_vecs:
        # Compute centroids
        harm_centroid = np.mean(pad_vectors(
            [get_vector(r) for r in core if r.get("harmfulness") == "harmful"
             and r.get("register") == "news"]), axis=0)
        benign_centroids = {}
        for reg in ["literary", "academic"]:
            vecs = [get_vector(r) for r in core
                    if r.get("harmfulness") == "benign" and r.get("register") == reg]
            if vecs:
                benign_centroids[reg] = np.mean(pad_vectors(vecs), axis=0)

        print(f"\n{'Transfer prompt':<12} {'Register':>10} {'Cos→harm':>10} {'Cos→genre':>11}")
        print("-" * 46)

        for r in transfers:
            tv = np.array(get_vector(r))
            target_reg = r.get("register", "")

            # Pad to match centroid length
            max_len = max(len(tv), len(harm_centroid))
            tv_pad = np.pad(tv, (0, max(0, max_len - len(tv))))
            hc_pad = np.pad(harm_centroid, (0, max(0, max_len - len(harm_centroid))))

            cos_harm = float(np.dot(tv_pad, hc_pad) /
                           (np.linalg.norm(tv_pad) * np.linalg.norm(hc_pad) + 1e-10))

            cos_genre = float("nan")
            if target_reg in benign_centroids:
                gc = benign_centroids[target_reg]
                gc_pad = np.pad(gc, (0, max(0, max_len - len(gc))))
                cos_genre = float(np.dot(tv_pad, gc_pad) /
                                (np.linalg.norm(tv_pad) * np.linalg.norm(gc_pad) + 1e-10))

            src = r.get("transfer_source_id", "?")
            print(f"{src:<12} {target_reg:>10} {cos_harm:>10.4f} {cos_genre:>11.4f}")
    else:
        print("No transfer data or insufficient core data.")

    # =====================================================
    # §5.4 — Principle-level axis
    # =====================================================
    print(f"\n{'=' * 60}")
    print("§5.4 — Principle-level probes")
    print("=" * 60)

    if principles:
        pres = [get_vector(r) for r in principles
                if r.get("harmfulness") == "coherence-preserving"]
        dis = [get_vector(r) for r in principles
               if r.get("harmfulness") == "coherence-disrupting"]

        if pres and dis:
            all_p = pad_vectors(pres + dis)
            pres_pad = pad_vectors(pres)
            dis_pad = pad_vectors(dis)

            er_pres = effective_rank(pres_pad)
            er_dis = effective_rank(dis_pad)
            er_both = effective_rank(all_p)

            print(f"\nPreserving erank:  {er_pres:.2f}  (n={len(pres)})")
            print(f"Disrupting erank:  {er_dis:.2f}  (n={len(dis)})")
            print(f"Combined erank:    {er_both:.2f}")

            # PCA: does PC1 separate the classes?
            X = np.array(all_p)
            X = X - X.mean(axis=0)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            pc1 = X @ Vt[0]
            pres_pc1 = pc1[:len(pres)]
            dis_pc1 = pc1[len(pres):]
            sep_d = cohens_d(dis_pc1, pres_pc1)
            print(f"PC1 separation:    Cohen's d = {sep_d:.2f}")
            print(f"  preserving PC1:  mean={np.mean(pres_pc1):.4f}")
            print(f"  disrupting PC1:  mean={np.mean(dis_pc1):.4f}")
        else:
            print("Insufficient principle-level data.")
    else:
        print("No principle-level probes found.")

    # =====================================================
    # §5.5 — Scalar summaries (density, stress by cell)
    # =====================================================
    print(f"\n{'=' * 60}")
    print("Scalar summaries by cell")
    print("=" * 60)

    print(f"\n{'Register':<16} {'Harm den':>9} {'Ben den':>9} {'Harm str':>9} {'Ben str':>9}")
    print("-" * 56)

    for reg in registers:
        h_den = [np.mean(r["per_token_density"]) for r in core
                 if r.get("register") == reg and r.get("harmfulness") == "harmful"]
        b_den = [np.mean(r["per_token_density"]) for r in core
                 if r.get("register") == reg and r.get("harmfulness") == "benign"]
        h_str = [r["stress_score"] for r in core
                 if r.get("register") == reg and r.get("harmfulness") == "harmful"]
        b_str = [r["stress_score"] for r in core
                 if r.get("register") == reg and r.get("harmfulness") == "benign"]

        print(f"{reg:<16} "
              f"{np.mean(h_den):>9.4f} {np.mean(b_den):>9.4f} "
              f"{np.mean(h_str):>9.4f} {np.mean(b_str):>9.4f}")

    # =====================================================
    # Kill conditions
    # =====================================================
    print(f"\n{'=' * 60}")
    print("Kill condition evaluation")
    print("=" * 60)

    print(f"\nGC-1 (harm concentrated within every register):")
    print(f"  Registers where harmful rank >= benign rank: {gc1_failures}/5")
    if gc1_failures >= 3:
        print(f"  ** KILL: {gc1_failures} >= 3 — H-structure is dead as stated **")
    else:
        print(f"  SURVIVES ({gc1_failures} < 3)")

    gc2 = abs(cross_harm_er - np.mean([
        effective_rank(pad_vectors(cells.get(("harmful", r), [])))
        for r in registers
        if len(cells.get(("harmful", r), [])) >= 2
    ])) if cross_harm_er == cross_harm_er else float("nan")
    print(f"\nGC-2 (cross-genre ≈ within-genre harmful rank):")
    print(f"  Cross-genre harmful erank: {cross_harm_er:.2f}")
    print(f"  (compare to within-genre values above)")

    print(f"\nGC-4 (principle-level bipolar axis):")
    if principles:
        print(f"  (see PC1 separation above)")
    else:
        print(f"  NO DATA")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
