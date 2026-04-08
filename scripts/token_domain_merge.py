"""
Per-token 3-axis merge.

Takes:
  1. anchor_session_map.json (prompt-level domain coordinates + anchors)
  2. results.json (per-position RD data)

Produces: token_domain_3axis.json
  Each observation = one token at one prompt's domain position, with RD metrics.

Run:
    python token_domain_merge.py anchor_session_map.json /path/to/results.json
"""

import sys
import json
import numpy as np
from collections import defaultdict

def main():
    if len(sys.argv) < 3:
        print("Usage: python token_domain_merge.py anchor_session_map.json results.json")
        sys.exit(1)

    print("Loading anchor map...")
    with open(sys.argv[1]) as f:
        amap = json.load(f)

    print("Loading session results...")
    with open(sys.argv[2]) as f:
        session = json.load(f)

    assert len(amap["prompts"]) == len(session), \
        f"Mismatch: {len(amap['prompts'])} vs {len(session)}"

    # Build per-token observations
    observations = []
    token_freq = defaultdict(int)

    for pi, (pm, sr) in enumerate(zip(amap["prompts"], session)):
        dx = pm["x"]
        dy = pm["y"]
        cat = pm["cat"]
        prompt = pm["p"]
        tokens = sr.get("tokens", [])
        rd = sr.get("rank_displacement", {})
        per_pos = rd.get("per_position", [])

        for pos, pos_data in enumerate(per_pos):
            if pos >= len(tokens):
                continue

            tok = tokens[pos].strip()
            if not tok:
                continue

            token_freq[tok] += 1

            nm = int(pos_data.get("n_matched", 0))
            np_ = int(pos_data.get("n_promoted", 0))
            nd = int(pos_data.get("n_demoted", 0))
            total = nm + np_ + nd
            jacc = nm / total if total > 0 else 1.0

            observations.append({
                "pi": pi,
                "t": tok,
                "pos": pos,
                "cat": cat,
                "dx": dx,
                "dy": dy,
                "disp": round(float(pos_data.get("total_disp", 0)), 4),
                "repl": round(float(pos_data.get("replacement_ratio", 0)), 3),
                "jacc": round(float(jacc), 3),
                "nm": nm,
                "near_dom": pm["near_dom"],
                "near_sim": pm["near_sim"],
            })

    print(f"Total observations: {len(observations)}")
    print(f"Unique tokens: {len(token_freq)}")

    # Top tokens by frequency
    top_tokens = sorted(token_freq.keys(), key=lambda t: -token_freq[t])[:25]
    print(f"Top 25 tokens: {top_tokens}")

    # Per-token summary
    print(f"\n{'='*70}")
    print(f"TOKEN SUMMARY ACROSS DOMAIN SURFACE (top 25)")
    print(f"{'='*70}\n")
    print(f"  {'Token':12s} {'n':>4s} {'disp':>7s} {'std':>7s} {'cv':>6s} "
          f"{'repl':>6s} {'jacc':>6s}  cat distribution")
    print(f"  {'-'*75}")

    token_summary = {}
    for tok in top_tokens:
        t_obs = [o for o in observations if o["t"] == tok]
        if len(t_obs) < 2:
            continue
        disps = [o["disp"] for o in t_obs]
        repls = [o["repl"] for o in t_obs]
        jaccs = [o["jacc"] for o in t_obs]
        cat_dist = defaultdict(int)
        for o in t_obs:
            cat_dist[o["cat"]] += 1

        cv = float(np.std(disps) / max(np.mean(disps), 1e-12))
        cat_str = " ".join(f"{c[0]}:{cat_dist[c]}" for c in
                           ["benign","mild","harmful","jailbreak"] if cat_dist[c] > 0)

        token_summary[tok] = {
            "n": len(t_obs),
            "disp_mean": round(float(np.mean(disps)), 4),
            "disp_std": round(float(np.std(disps)), 4),
            "disp_cv": round(cv, 3),
            "repl_mean": round(float(np.mean(repls)), 3),
            "jacc_mean": round(float(np.mean(jaccs)), 3),
            "cats": dict(cat_dist),
        }

        print(f"  {tok:12s} {len(t_obs):4d} {np.mean(disps):7.4f} {np.std(disps):7.4f} "
              f"{cv:6.3f} {np.mean(repls):6.3f} {np.mean(jaccs):6.3f}  {cat_str}")

    # Category comparison for top tokens
    print(f"\n{'='*70}")
    print(f"BENIGN vs ADVERSARIAL DISPLACEMENT (per token)")
    print(f"{'='*70}\n")
    print(f"  {'Token':12s} {'benign':>8s} {'mild':>8s} {'harmful':>8s} {'jbreak':>8s} {'gap(h-b)':>8s}")
    print(f"  {'-'*58}")

    for tok in top_tokens:
        t_obs = [o for o in observations if o["t"] == tok]
        if len(t_obs) < 4:
            continue
        by_cat = defaultdict(list)
        for o in t_obs:
            by_cat[o["cat"]].append(o["disp"])

        vals = {}
        for c in ["benign","mild","harmful","jailbreak"]:
            vals[c] = np.mean(by_cat[c]) if by_cat[c] else None

        b = vals["benign"]
        h = vals["harmful"]
        gap = (h - b) if (b is not None and h is not None) else None

        print(f"  {tok:12s}", end="")
        for c in ["benign","mild","harmful","jailbreak"]:
            v = vals[c]
            print(f" {v:8.4f}" if v is not None else f" {'--':>8s}", end="")
        print(f" {gap:+8.4f}" if gap is not None else f" {'--':>8s}")

    # Filter observations to top tokens for export
    top_set = set(top_tokens)
    export_obs = [o for o in observations if o["t"] in top_set]

    export = {
        "pca": amap["pca"],
        "top_tokens": top_tokens,
        "token_summary": token_summary,
        "anchors": amap["anchors"],
        "observations": export_obs,
        "n_total": len(observations),
        "n_export": len(export_obs),
    }

    with open("token_domain_3axis.json", "w") as f:
        json.dump(export, f, indent=1,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"\nExported {len(export_obs)} observations ({len(top_tokens)} tokens) "
          f"to token_domain_3axis.json")


if __name__ == "__main__":
    main()