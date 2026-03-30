"""
Spectrum Token Merge v2.

Reads subject matter probes from CSV, embeds them alongside session
prompts in a co-fit PCA, merges per-token RD/ASM/SFD, computes 2D
nearest-anchor proximity. Exports everything for the token column chart.

Run:
    python spectrum_merge_v2.py results.json alignment_probes.csv [--layer 0.5]

Args:
    results.json         Session results from TASM analyzer
    alignment_probes.csv Probe definitions (subject, nouns, phrase, question, etc.)
    --layer FRAC         Layer fraction (0.0-1.0). Default 0.5 = middle layer.
"""

import sys, os, json, csv, math, argparse
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

LEVEL_COLS = ["nouns", "phrase", "question", "instruction", "meta_instruction"]
LEVEL_NAMES = ["L0:nouns", "L1:phrase", "L2:question", "L3:instruct", "L4:meta"]


def load_probes(csv_path):
    """Load probes from CSV. Returns list of {subject, anchor_id, level, text}."""
    probes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for level, col in enumerate(LEVEL_COLS):
                text = row.get(col, "").strip()
                if text:
                    probes.append({
                        "subject": row["subject"],
                        "anchor_id": row["anchor_id"],
                        "level": level,
                        "text": text,
                    })
    return probes


def get_embedding(model, tokenizer, text, captured):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    h = captured["h"][0]
    return h[1:].mean(dim=0).numpy() if h.shape[0] > 1 else h[0].numpy()


def main():
    parser = argparse.ArgumentParser(description="Spectrum Token Merge v2")
    parser.add_argument("results", help="Path to results.json")
    parser.add_argument("probes", help="Path to alignment_probes.csv")
    parser.add_argument("--layer", type=float, default=0.5,
                        help="Layer fraction 0.0-1.0 (default: 0.5 = middle)")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID")
    parser.add_argument("--top-tokens", type=int, default=20,
                        help="Number of top tokens to include")
    args = parser.parse_args()

    # Load inputs
    print(f"Loading session: {args.results}")
    with open(args.results) as f:
        session = json.load(f)
    print(f"  {len(session)} prompts")

    print(f"Loading probes: {args.probes}")
    probes = load_probes(args.probes)
    subjects = sorted(set(p["subject"] for p in probes))
    print(f"  {len(probes)} probes across {len(subjects)} subjects")
    for s in subjects:
        n = sum(1 for p in probes if p["subject"] == s)
        print(f"    {s:25s} {n} entries")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    n_layers = model.config.num_hidden_layers
    target_layer = max(0, min(n_layers - 1, int(args.layer * n_layers)))
    print(f"  {n_layers} layers, target = layer {target_layer} "
          f"({args.layer:.0%} depth)")

    # Hook
    captured = {}
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    handle = model.model.layers[target_layer].input_layernorm.register_forward_hook(hook_fn)

    # Embed session prompts
    print("\nEmbedding session prompts...")
    prompt_embs = []
    for i, r in enumerate(session):
        emb = get_embedding(model, tokenizer, r["prompt"], captured)
        prompt_embs.append(emb)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(session)}")

    # Embed probes
    print(f"Embedding {len(probes)} probes...")
    probe_embs = []
    for i, p in enumerate(probes):
        emb = get_embedding(model, tokenizer, p["text"], captured)
        probe_embs.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(probes)}")

    handle.remove()

    # Normalize
    all_embs = np.array(prompt_embs + probe_embs)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1
    all_n = all_embs / norms

    n_p = len(session)
    n_a = len(probes)

    # Co-fit PCA
    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(all_n)
    prompt_coords = all_coords[:n_p]
    probe_coords = all_coords[n_p:]

    print(f"\nPCA: {pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"{pca.explained_variance_ratio_[1]*100:.1f}%")

    # Per-token observations
    print("Building per-token observations...")
    observations = []
    token_freq = defaultdict(int)

    for pi, (sr, coord) in enumerate(zip(session, prompt_coords)):
        dx = float(coord[0])
        dy = float(coord[1])
        cat = sr.get("category", "?")[0]
        toks = sr.get("tokens", [])
        rd = sr.get("rank_displacement", {})
        per_pos = rd.get("per_position", [])
        stress = sr.get("per_token_stress", [])
        sfd = sr.get("sfd", {})
        sfd_e = sfd.get("per_token_energy", [])
        sfd_d = sfd.get("per_token_density", [])

        for pos in range(len(per_pos)):
            if pos >= len(toks):
                continue
            tok = toks[pos].strip()
            if not tok:
                continue
            token_freq[tok] += 1

            pd = per_pos[pos]
            disp = float(pd.get("total_disp", 0))
            repl = float(pd.get("replacement_ratio", 0))
            asm = float(stress[pos]) if pos < len(stress) else 0
            se = float(sfd_e[pos]) if pos < len(sfd_e) else 0
            sd = float(sfd_d[pos]) if pos < len(sfd_d) else 0

            observations.append({
                "tok": tok, "cat": cat,
                "dx": dx, "dy": dy,
                "disp": disp, "repl": repl,
                "asm": asm, "sfd_e": se, "sfd_d": sd,
                "pi": pi, "pos": pos,
            })

    # Top tokens by freq, compute CV, order by CV
    top = sorted(token_freq.keys(), key=lambda t: -token_freq[t])[:args.top_tokens]
    token_cv = {}
    for tok in top:
        disps = [o["disp"] for o in observations if o["tok"] == tok]
        if len(disps) >= 2:
            m = np.mean(disps)
            token_cv[tok] = round(float(np.std(disps) / max(m, 1e-12)), 3)
        else:
            token_cv[tok] = 0
    ordered = sorted(top, key=lambda t: token_cv.get(t, 0))

    # Filter to top tokens
    top_set = set(ordered)
    filtered = [o for o in observations if o["tok"] in top_set]

    # Build anchor lookup for 2D proximity
    anchor_pts = []
    for i, p in enumerate(probes):
        anchor_pts.append({
            "subject": p["subject"],
            "anchor_id": p["anchor_id"],
            "level": p["level"],
            "text": p["text"][:50],
            "x": float(probe_coords[i, 0]),
            "y": float(probe_coords[i, 1]),
        })

    # Compute 2D nearest anchor for each observation
    print("Computing 2D nearest-anchor proximity...")
    subj_idx = {s: i for i, s in enumerate(subjects)}

    obs_export = []
    for o in filtered:
        best_dist = 999
        best_anc = None
        for a in anchor_pts:
            d = math.sqrt((o["dx"] - a["x"])**2 + (o["dy"] - a["y"])**2)
            if d < best_dist:
                best_dist = d
                best_anc = a

        obs_export.append([
            o["tok"], o["cat"],
            round(o["dy"], 4),
            round(o["disp"], 3),
            round(o["repl"], 2),
            round(o["dx"], 4),
            round(o["asm"], 2),
            round(o["sfd_e"], 3),
            round(o["sfd_d"], 3),
            o["pi"], o["pos"],
            round(best_dist, 4),
            best_anc["level"],
            subj_idx[best_anc["subject"]],
        ])

    # Report
    print(f"\n{'='*60}")
    print(f"STRATIFICATION REPORT")
    print(f"{'='*60}")

    from collections import Counter

    print(f"\nNearest LEVEL by category:")
    level_cat = Counter()
    for o in obs_export:
        level_cat[(o[12], o[1])] += 1
    for l in range(5):
        counts = {c: level_cat.get((l, c), 0) for c in 'bmhj'}
        total = sum(counts.values())
        if total == 0:
            continue
        print(f"  {LEVEL_NAMES[l]:15s}  b={counts['b']:3d} m={counts['m']:3d} "
              f"h={counts['h']:3d} j={counts['j']:3d}  total={total}")

    print(f"\nNearest SUBJECT by category:")
    subj_cat = Counter()
    for o in obs_export:
        subj_cat[(subjects[o[13]], o[1])] += 1
    for s in subjects:
        counts = {c: subj_cat.get((s, c), 0) for c in 'bmhj'}
        total = sum(counts.values())
        if total == 0:
            continue
        print(f"  {s:25s}  b={counts['b']:3d} m={counts['m']:3d} "
              f"h={counts['h']:3d} j={counts['j']:3d}  total={total}")

    # Level means
    print(f"\nAnchor level positions:")
    for l in range(5):
        pts = [a for a in anchor_pts if a["level"] == l]
        if not pts:
            continue
        mx = np.mean([p["x"] for p in pts])
        my = np.mean([p["y"] for p in pts])
        print(f"  {LEVEL_NAMES[l]:15s}  mean=({mx:+.3f}, {my:+.3f})  n={len(pts)}")

    # Export
    prompts = [r["prompt"][:80] for r in session]

    export = {
        "pca": [round(float(x) * 100, 1) for x in pca.explained_variance_ratio_[:2]],
        "layer": target_layer,
        "layer_frac": args.layer,
        "n_layers": n_layers,
        "tokens": ordered,
        "token_cv": token_cv,
        "subjects": subjects,
        "anchors": anchor_pts,
        "obs": obs_export,
        "prompts": prompts,
        "fields": [
            "tok", "cat", "dy", "disp", "repl", "dx",
            "asm", "sfd_e", "sfd_d", "pi", "pos",
            "near_dist", "near_level", "near_subj_idx",
        ],
    }

    out_path = "spectrum_token_view.json"
    with open(out_path, "w") as f:
        json.dump(export, f, separators=(',', ':'),
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    sz = os.path.getsize(out_path)
    print(f"\nExported {len(obs_export)} obs, {len(anchor_pts)} anchors, "
          f"{len(ordered)} tokens to {out_path} ({sz//1024}KB)")


if __name__ == "__main__":
    main()