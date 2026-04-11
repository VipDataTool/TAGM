"""
Domain separation test: do middle-layer hidden states encode subject domain?

Run from the tasm/ directory:
    python domain_test.py

Loads Qwen 2.5-0.5B-Instruct, runs 20 prompts (5 domains x 4 prompts),
captures mean hidden state at the middle layer, computes pairwise cosine
similarity within vs between domains.

If within-domain similarity >> between-domain, the domain axis exists
in hidden states and can be used for the RD scatter vertical axis.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = {
    "medical": [
        "What are the early symptoms of type 2 diabetes?",
        "How does chemotherapy target cancer cells differently from healthy tissue?",
        "Explain the mechanism of action of SSRIs in treating depression.",
        "What is the difference between an MRI and a CT scan for diagnosing injuries?",
    ],
    "legal": [
        "What are the key elements required to prove negligence in a civil lawsuit?",
        "How does the process of filing a patent differ from registering a trademark?",
        "Explain the difference between a felony and a misdemeanor in criminal law.",
        "What rights does a tenant have when facing eviction without proper notice?",
    ],
    "coding": [
        "How do I implement a binary search tree in Python?",
        "What is the difference between a mutex and a semaphore in concurrent programming?",
        "Explain how garbage collection works in Java compared to manual memory management.",
        "How do I set up a REST API with authentication using Node.js and Express?",
    ],
    "cooking": [
        "What is the proper technique for making a French roux for bechamel sauce?",
        "How do I get a crispy skin when roasting a whole chicken?",
        "What is the difference between braising and stewing meat?",
        "How should I temper chocolate for making truffles at home?",
    ],
    "finance": [
        "How does compound interest differ from simple interest on a savings account?",
        "What factors should I consider when deciding between a Roth and traditional IRA?",
        "Explain how options contracts work in stock trading.",
        "What is the difference between a balance sheet and an income statement?",
    ],
}

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

    # Hook to capture hidden state at middle layer input
    captured = {}

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    layer = model.model.layers[mid_layer]
    handle = layer.input_layernorm.register_forward_hook(hook_fn)

    # Run all prompts and collect mean hidden states
    domain_embeddings = {}  # domain -> list of mean hidden vectors
    all_labels = []
    all_vecs = []

    for domain, prompts in PROMPTS.items():
        domain_embeddings[domain] = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                model(**inputs)

            h = captured["h"][0]  # [seq_len, hidden_size]
            # Mean pool across all positions (simple domain fingerprint)
            mean_h = h.mean(dim=0).numpy()
            # Also try: skip position 0, use positions 1 to n-1
            interior_h = h[1:].mean(dim=0).numpy()

            domain_embeddings[domain].append(interior_h)
            all_labels.append(domain)
            all_vecs.append(interior_h)

            seq_len = h.shape[0]
            print(f"  [{domain:8s}] {prompt[:60]:60s} seq={seq_len}")

    handle.remove()

    # Compute pairwise cosine similarity matrix
    vecs = np.array(all_vecs)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1
    vecs_n = vecs / norms
    cos_sim = vecs_n @ vecs_n.T

    domains = list(PROMPTS.keys())
    n_per = len(list(PROMPTS.values())[0])

    print(f"\n{'='*60}")
    print(f"COSINE SIMILARITY: WITHIN vs BETWEEN DOMAINS")
    print(f"{'='*60}\n")

    # Within-domain vs between-domain
    within_sims = []
    between_sims = []
    domain_within = {}
    domain_between = {}

    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            sim = cos_sim[i, j]
            if all_labels[i] == all_labels[j]:
                within_sims.append(sim)
                d = all_labels[i]
                domain_within.setdefault(d, []).append(sim)
            else:
                between_sims.append(sim)
                for d in [all_labels[i], all_labels[j]]:
                    domain_between.setdefault(d, []).append(sim)

    print(f"  WITHIN-domain mean:  {np.mean(within_sims):.4f}  (std {np.std(within_sims):.4f})")
    print(f"  BETWEEN-domain mean: {np.mean(between_sims):.4f}  (std {np.std(between_sims):.4f})")
    gap = np.mean(within_sims) - np.mean(between_sims)
    print(f"  GAP:                 {gap:+.4f}")
    print()

    # Per-domain breakdown
    print(f"  {'Domain':12s} {'Within':>8s} {'Between':>8s} {'Gap':>8s}")
    print(f"  {'-'*40}")
    for d in domains:
        w = np.mean(domain_within.get(d, [0]))
        b = np.mean(domain_between.get(d, [0]))
        print(f"  {d:12s} {w:8.4f} {b:8.4f} {w-b:+8.4f}")

    # Full domain x domain similarity matrix
    print(f"\n  DOMAIN x DOMAIN MEAN COSINE SIMILARITY")
    print(f"  {'':12s}", end="")
    for d in domains:
        print(f" {d:>8s}", end="")
    print()
    for da in domains:
        print(f"  {da:12s}", end="")
        for db in domains:
            idx_a = [i for i, l in enumerate(all_labels) if l == da]
            idx_b = [i for i, l in enumerate(all_labels) if l == db]
            sims = []
            for i in idx_a:
                for j in idx_b:
                    if i != j:
                        sims.append(cos_sim[i, j])
            print(f" {np.mean(sims):8.4f}", end="")
        print()

    # PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs_n)
    print(f"\n  PCA variance: {pca.explained_variance_ratio_[0]*100:.1f}%, {pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"\n  PCA COORDINATES (for plotting)")
    for i, (label, (x, y)) in enumerate(zip(all_labels, coords)):
        prompt_short = list(PROMPTS[label])[i % n_per][:50]
        print(f"  {label:8s} ({x:+6.3f}, {y:+6.3f})  {prompt_short}")

    # Effect size
    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(within_sims, between_sims, alternative='greater')
    cohens_d = (np.mean(within_sims) - np.mean(between_sims)) / np.sqrt(
        (np.std(within_sims)**2 + np.std(between_sims)**2) / 2)
    print(f"\n  Mann-Whitney U={U:.0f}, p={p:.6f}")
    print(f"  Cohen's d = {cohens_d:.3f}")

    if gap > 0.02 and p < 0.05:
        print(f"\n  RESULT: Domain axis EXISTS in hidden states (gap={gap:.4f}, d={cohens_d:.3f})")
    elif gap > 0.01:
        print(f"\n  RESULT: Weak domain signal (gap={gap:.4f}, d={cohens_d:.3f}) — may need larger model")
    else:
        print(f"\n  RESULT: No meaningful domain separation (gap={gap:.4f}) — hidden states at this scale don't encode it")


if __name__ == "__main__":
    main()
