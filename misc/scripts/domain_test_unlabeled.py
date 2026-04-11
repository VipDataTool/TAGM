"""
Domain structure test: unsupervised discovery from hidden states.

No labels. Prompts go in, embeddings come out, structure emerges or it doesn't.

Run from the tasm/ directory:
    python domain_test_unlabeled.py

Outputs:
  - Pairwise cosine similarity matrix (are there natural groupings?)
  - Optimal k via silhouette score
  - Cluster membership (what landed together?)
  - PCA coordinates for plotting
  - Distinctive vocabulary per cluster (from base counterfactuals)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# 30 prompts. No labels. Mixed subject matter.
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


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    mid_layer = n_layers // 2
    hidden_size = model.config.hidden_size
    print(f"Model: {n_layers} layers, {hidden_size}d, middle layer = {mid_layer}")

    # Hook middle layer
    captured = {}
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    handle = model.model.layers[mid_layer].input_layernorm.register_forward_hook(hook_fn)

    # Also capture base counterfactual-style info: top-k predictions at each position
    # (We use the instruct model's own predictions here as a vocabulary fingerprint)
    embeddings = []
    prompt_topk = []  # per-prompt vocabulary fingerprint

    for i, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)

        # Hidden state embedding
        h = captured["h"][0]  # [seq_len, hidden_size]
        interior_h = h[1:].mean(dim=0).numpy()  # skip pos 0
        embeddings.append(interior_h)

        # Vocabulary fingerprint: top-5 tokens at positions 2-5
        logits = out.logits[0]
        seq_len = logits.shape[0]
        top_tokens = []
        for pos in range(min(2, seq_len), min(6, seq_len)):
            probs = torch.softmax(logits[pos], dim=-1)
            topk = torch.topk(probs, 5)
            for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
                tok = tokenizer.decode(idx).strip()
                if tok:
                    top_tokens.append((tok, prob))
        prompt_topk.append(top_tokens)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"  [{i:2d}] {prompt[:70]:70s} seq={len(tokens)}")

    handle.remove()

    # ── Analysis ──
    vecs = np.array(embeddings)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1
    vecs_n = vecs / norms

    # Pairwise cosine similarity
    cos_sim = vecs_n @ vecs_n.T

    # Find natural k
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs_n)
    print(f"\nPCA variance: {pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"{pca.explained_variance_ratio_[1]*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"UNSUPERVISED CLUSTERING")
    print(f"{'='*60}\n")

    best_k, best_sil = 2, -1
    for k in range(2, 10):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(vecs_n)
        sil = silhouette_score(vecs_n, labels)
        print(f"  k={k}  silhouette={sil:.3f}")
        if sil > best_sil:
            best_k, best_sil = k, sil

    print(f"\n  Best k={best_k} (silhouette={best_sil:.3f})")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(vecs_n)

    # ── Report clusters ──
    print(f"\n{'='*60}")
    print(f"DISCOVERED CLUSTERS (k={best_k})")
    print(f"{'='*60}")

    from collections import Counter

    for c in range(best_k):
        members = [i for i in range(len(PROMPTS)) if labels[i] == c]
        print(f"\n  ── Cluster {c} ({len(members)} prompts) ──")

        # Prompts
        for i in members:
            print(f"    \"{PROMPTS[i][:75]}\"")

        # Distinctive vocabulary: pool top-k tokens from this cluster vs others
        cluster_vocab = Counter()
        other_vocab = Counter()
        for i in range(len(PROMPTS)):
            for tok, prob in prompt_topk[i]:
                if labels[i] == c:
                    cluster_vocab[tok] += prob
                else:
                    other_vocab[tok] += prob

        # Normalize by cluster size
        n_in = len(members)
        n_out = len(PROMPTS) - n_in
        distinctive = {}
        for tok in cluster_vocab:
            in_rate = cluster_vocab[tok] / n_in
            out_rate = other_vocab.get(tok, 0) / max(n_out, 1)
            if in_rate > 0.05:  # minimum threshold
                distinctive[tok] = in_rate - out_rate

        top_distinctive = sorted(distinctive.items(), key=lambda x: -x[1])[:8]
        if top_distinctive:
            print(f"    Vocabulary signature: {[t for t,_ in top_distinctive]}")

    # ── Within vs between cluster similarity ──
    print(f"\n{'='*60}")
    print(f"CLUSTER COHESION")
    print(f"{'='*60}\n")

    within, between = [], []
    for i in range(len(PROMPTS)):
        for j in range(i+1, len(PROMPTS)):
            if labels[i] == labels[j]:
                within.append(cos_sim[i,j])
            else:
                between.append(cos_sim[i,j])

    print(f"  Within-cluster:  {np.mean(within):.4f} (std {np.std(within):.4f})")
    print(f"  Between-cluster: {np.mean(between):.4f} (std {np.std(between):.4f})")
    print(f"  Gap:             {np.mean(within)-np.mean(between):+.4f}")

    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(within, between, alternative='greater')
    d = (np.mean(within) - np.mean(between)) / np.sqrt(
        (np.std(within)**2 + np.std(between)**2) / 2)
    print(f"  Cohen's d = {d:.3f}, p = {p:.6f}")

    # ── PCA coordinates ──
    print(f"\n{'='*60}")
    print(f"PCA COORDINATES")
    print(f"{'='*60}\n")
    print(f"  {'Cl':>2s}  {'PC1':>7s} {'PC2':>7s}  Prompt")
    for i in range(len(PROMPTS)):
        print(f"  {labels[i]:2d}  {coords[i,0]:+7.3f} {coords[i,1]:+7.3f}  {PROMPTS[i][:60]}")

    # ── Export for plotting ──
    import json
    export = []
    for i in range(len(PROMPTS)):
        top_vocab = sorted(prompt_topk[i], key=lambda x: -x[1])[:5]
        export.append({
            "p": PROMPTS[i],
            "cl": int(labels[i]),
            "x": round(float(coords[i,0]), 4),
            "y": round(float(coords[i,1]), 4),
            "v": [t for t,_ in top_vocab],
        })
    with open("domain_unlabeled.json", "w") as f:
        json.dump({"k": int(best_k), "sil": round(float(best_sil), 3),
                    "pca": [round(float(x)*100, 1) for x in pca.explained_variance_ratio_[:2]],
                    "gap": round(float(np.mean(within)-np.mean(between)), 4),
                    "d": round(float(d), 3),
                    "pts": export}, f, indent=1)
    print(f"\n  Exported to domain_unlabeled.json")


if __name__ == "__main__":
    main()