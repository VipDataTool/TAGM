"""
3-Axis Test: Domain surface × Token × RD displacement

Standalone script. Loads both base and instruct Qwen 0.5B, runs 30 prompts,
captures hidden state embeddings (domain axis) and computes per-token
rank displacement between instruct and base predictions (token × RD axes).

Outputs domain_3axis.json for plotting.

Run:
    python domain_3axis_test.py
"""

import torch
import numpy as np
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

INSTRUCT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_ID = "Qwen/Qwen2.5-0.5B"
K = 8  # top-k candidates

PROMPTS = [
    "What are the early symptoms of type 2 diabetes?",
    "How do I implement a binary search tree in Python?",
    "What is the proper technique for making a French roux?",
    "Explain the difference between a felony and a misdemeanor.",
    "How does compound interest work on a savings account?",
    "What causes antibiotic resistance in bacteria?",
    "How do I set up a REST API with authentication in Node.js?",
    "How do I get a crispy skin when roasting a whole chicken?",
    "What rights does a tenant have when facing eviction?",
    "What factors matter when choosing between a Roth and traditional IRA?",
    "How does chemotherapy target cancer cells?",
    "What is the difference between a mutex and a semaphore?",
    "What is the difference between braising and stewing meat?",
    "What are the key elements required to prove negligence?",
    "How do options contracts work in stock trading?",
    "Explain the mechanism of action of SSRIs in treating depression.",
    "How does garbage collection work in Java?",
    "How should I temper chocolate for making truffles?",
    "How does the process of filing a patent differ from a trademark?",
    "What is the difference between a balance sheet and an income statement?",
    "What is the difference between an MRI and a CT scan?",
    "How do I optimize a SQL query that runs slowly on large tables?",
    "What is the best way to knead bread dough by hand?",
    "What is the statute of limitations for personal injury claims?",
    "How do bond yields relate to interest rates?",
    "What are the warning signs of a stroke?",
    "How does recursion work and when should I avoid it?",
    "What temperature should I cook a medium-rare steak to?",
    "What is the difference between civil and criminal court proceedings?",
    "How do index funds compare to actively managed mutual funds?",
]


def compute_rd_at_position(instruct_topk, base_topk):
    """Compute rank displacement between instruct and base top-k at one position."""
    inst = {t: p for t, p in instruct_topk}
    base = {t: p for t, p in base_topk}

    matched = set(inst) & set(base)
    promoted = set(inst) - set(base)
    demoted = set(base) - set(inst)

    matched_disp = sum(abs(inst[t] - base[t]) for t in matched)
    promoted_mass = sum(inst[t] for t in promoted)
    demoted_mass = sum(base[t] for t in demoted)
    total_disp = matched_disp + promoted_mass + demoted_mass
    replacement_ratio = (promoted_mass + demoted_mass) / total_disp if total_disp > 1e-12 else 0
    jaccard = len(matched) / len(set(inst) | set(base)) if (set(inst) | set(base)) else 1

    return {
        "total_disp": round(float(total_disp), 4),
        "replacement_ratio": round(float(replacement_ratio), 3),
        "jaccard": round(float(jaccard), 3),
        "n_matched": len(matched),
        "promoted": sorted(promoted, key=lambda t: -inst[t]),
        "demoted": sorted(demoted, key=lambda t: -base[t]),
    }


def main():
    print(f"Loading instruct: {INSTRUCT_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_ID)
    instruct_model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_ID, torch_dtype=torch.float32, device_map="cpu")
    instruct_model.eval()

    print(f"Loading base: {BASE_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.float32, device_map="cpu")
    base_model.eval()

    n_layers = instruct_model.config.num_hidden_layers
    mid_layer = n_layers // 2
    hidden_size = instruct_model.config.hidden_size
    print(f"Model: {n_layers}L, {hidden_size}d, mid={mid_layer}")

    # Hook for hidden state capture (instruct model only)
    captured = {}
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    handle = instruct_model.model.layers[mid_layer].input_layernorm.register_forward_hook(hook_fn)

    # ── Run all prompts ──
    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_strs = [tokenizer.decode(t).strip() for t in inputs["input_ids"][0]]
        seq_len = len(tokens)

        # Instruct forward
        with torch.no_grad():
            inst_out = instruct_model(**inputs)
        h = captured["h"][0]
        domain_emb = h[1:].mean(dim=0).numpy()  # skip pos 0

        # Instruct top-k per position
        inst_logits = inst_out.logits[0]
        inst_topk_all = []
        for pos in range(seq_len):
            probs = torch.softmax(inst_logits[pos], dim=-1)
            topk = torch.topk(probs, K)
            alts = [(tokenizer.decode(idx).strip(), round(float(p), 4))
                    for idx, p in zip(topk.indices.tolist(), topk.values.tolist())]
            inst_topk_all.append(alts)

        # Base forward
        with torch.no_grad():
            base_out = base_model(**inputs)
        base_logits = base_out.logits[0]
        base_topk_all = []
        for pos in range(seq_len):
            probs = torch.softmax(base_logits[pos], dim=-1)
            topk = torch.topk(probs, K)
            alts = [(tokenizer.decode(idx).strip(), round(float(p), 4))
                    for idx, p in zip(topk.indices.tolist(), topk.values.tolist())]
            base_topk_all.append(alts)

        # Compute RD per interior token position (skip 0 and last)
        token_rds = []
        for pos in range(1, seq_len - 1):
            rd = compute_rd_at_position(inst_topk_all[pos], base_topk_all[pos])
            rd["token"] = token_strs[pos]
            rd["pos"] = pos
            rd["inst_cands"] = [t for t, _ in inst_topk_all[pos]]
            rd["base_cands"] = [t for t, _ in base_topk_all[pos]]
            token_rds.append(rd)

        prompt_result = {
            "prompt": prompt,
            "tokens": token_strs,
            "domain_emb": domain_emb,
            "token_rds": token_rds,
        }
        all_results.append(prompt_result)
        n_interior = len(token_rds)
        mean_disp = np.mean([r["total_disp"] for r in token_rds]) if token_rds else 0
        print(f"  [{pi:2d}] {prompt[:65]:65s} toks={seq_len} rd={mean_disp:.3f}")

    handle.remove()

    # ── PCA on domain embeddings ──
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    embs = np.array([r["domain_emb"] for r in all_results])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1
    embs_n = embs / norms

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embs_n)
    print(f"\nPCA variance: {pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"{pca.explained_variance_ratio_[1]*100:.1f}%")

    # Cluster
    best_k, best_sil = 2, -1
    for k in range(2, 10):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embs_n)
        sil = silhouette_score(embs_n, labels)
        if sil > best_sil:
            best_k, best_sil = k, sil
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(embs_n)
    print(f"Best k={best_k} (silhouette={best_sil:.3f})")

    # ── Find most common tokens across all prompts ──
    token_freq = defaultdict(int)
    for r in all_results:
        for rd in r["token_rds"]:
            token_freq[rd["token"]] += 1

    common_tokens = sorted(token_freq.keys(), key=lambda t: -token_freq[t])
    top_tokens = [t for t in common_tokens if len(t.strip()) > 0][:20]
    print(f"\nTop 20 tokens: {top_tokens}")

    # ── Build 3-axis export ──
    # Each observation: one token at one prompt position
    # Axes: domain_x, domain_y (from PCA), token identity, RD metrics
    observations = []
    for pi, r in enumerate(all_results):
        for rd in r["token_rds"]:
            observations.append({
                "pi": pi,
                "p": r["prompt"][:80],
                "cl": int(labels[pi]),
                "dx": round(float(coords[pi, 0]), 4),
                "dy": round(float(coords[pi, 1]), 4),
                "t": rd["token"],
                "pos": rd["pos"],
                "disp": rd["total_disp"],
                "repl": rd["replacement_ratio"],
                "jacc": rd["jaccard"],
                "nm": rd["n_matched"],
                "prom": rd["promoted"][:3],
                "dem": rd["demoted"][:3],
            })

    # Summary per token: how does its RD vary across the domain surface?
    token_summary = {}
    for t in top_tokens:
        t_obs = [o for o in observations if o["t"] == t]
        if len(t_obs) < 2:
            continue
        disps = [o["disp"] for o in t_obs]
        repls = [o["repl"] for o in t_obs]
        jaccs = [o["jacc"] for o in t_obs]
        token_summary[t] = {
            "n": len(t_obs),
            "disp_mean": round(float(np.mean(disps)), 4),
            "disp_std": round(float(np.std(disps)), 4),
            "disp_cv": round(float(np.std(disps) / max(np.mean(disps), 1e-12)), 3),
            "repl_mean": round(float(np.mean(repls)), 3),
            "jacc_mean": round(float(np.mean(jaccs)), 3),
            "positions": [(o["dx"], o["dy"], o["disp"], o["repl"]) for o in t_obs],
        }

    print(f"\n{'='*60}")
    print(f"TOKEN RD VARIANCE ACROSS DOMAIN SURFACE")
    print(f"{'='*60}\n")
    print(f"  {'Token':12s} {'n':>3s} {'disp':>7s} {'std':>7s} {'cv':>6s} {'repl':>6s} {'jacc':>6s}")
    print(f"  {'-'*52}")
    for t in top_tokens:
        if t not in token_summary:
            continue
        s = token_summary[t]
        print(f"  {t:12s} {s['n']:3d} {s['disp_mean']:7.4f} {s['disp_std']:7.4f} "
              f"{s['disp_cv']:6.3f} {s['repl_mean']:6.3f} {s['jacc_mean']:6.3f}")

    # ── Export ──
    export = {
        "pca": [round(float(x) * 100, 1) for x in pca.explained_variance_ratio_[:2]],
        "k": int(best_k),
        "sil": round(float(best_sil), 3),
        "top_tokens": top_tokens,
        "token_summary": token_summary,
        "prompts": [{
            "p": r["prompt"],
            "cl": int(labels[i]),
            "dx": round(float(coords[i, 0]), 4),
            "dy": round(float(coords[i, 1]), 4),
        } for i, r in enumerate(all_results)],
        "observations": observations,
    }

    with open("domain_3axis.json", "w") as f:
        json.dump(export, f, indent=1, default=lambda x: round(float(x), 4) if hasattr(x, 'item') else str(x))
    print(f"\nExported {len(observations)} observations to domain_3axis.json")
    print(f"({len(top_tokens)} tracked tokens × {len(PROMPTS)} prompts)")


if __name__ == "__main__":
    main()