"""
Domain label projection test.

Takes the PCA space from the 30-prompt domain test and projects
single-word domain labels into it. If "medical" lands near the
medical prompts, the model's vocabulary IS the taxonomy.

Run after domain_3axis_test.py (uses same model):
    python domain_labels_test.py
"""

import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Same prompts as before — we need to rebuild the PCA space
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

# Domain label words to project into the same space
LABELS = [
    # Broad domains
    "medical", "healthcare", "clinical", "diagnosis",
    "legal", "law", "court", "liability",
    "coding", "programming", "software", "algorithm",
    "cooking", "culinary", "recipe", "kitchen",
    "finance", "banking", "investment", "trading",
    # Safety-relevant
    "weapons", "explosives", "hacking", "malware",
    "drugs", "poison", "surveillance", "fraud",
    # Meta
    "science", "education", "technology", "engineering",
    "politics", "religion", "philosophy", "history",
    # Potentially ambiguous
    "security", "chemistry", "biology", "research",
    "defense", "intelligence", "network", "encryption",
]


def get_embedding(model, tokenizer, text, mid_layer, captured):
    """Get mean-pooled hidden state at middle layer for a text."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    h = captured["h"][0]
    # For single words, might only be 1-2 tokens. Use all.
    return h.mean(dim=0).numpy()


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    n_layers = model.config.num_hidden_layers
    mid_layer = n_layers // 2

    captured = {}
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    handle = model.model.layers[mid_layer].input_layernorm.register_forward_hook(hook_fn)

    # Step 1: Get embeddings for all prompts (rebuild PCA space)
    print("\n── Embedding prompts ──")
    prompt_embs = []
    for i, p in enumerate(PROMPTS):
        emb = get_embedding(model, tokenizer, p, mid_layer, captured)
        prompt_embs.append(emb)

    prompt_embs = np.array(prompt_embs)
    norms = np.linalg.norm(prompt_embs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1
    prompt_embs_n = prompt_embs / norms

    # Fit PCA on prompts
    pca = PCA(n_components=2)
    prompt_coords = pca.fit_transform(prompt_embs_n)
    print(f"PCA variance: {pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"{pca.explained_variance_ratio_[1]*100:.1f}%")

    # Step 2: Get embeddings for label words and PROJECT into same PCA
    print("\n── Embedding label words ──")
    label_embs = []
    label_names = []
    for word in LABELS:
        emb = get_embedding(model, tokenizer, word, mid_layer, captured)
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        label_embs.append(emb)
        label_names.append(word)

    label_embs = np.array(label_embs)
    label_coords = pca.transform(label_embs)

    handle.remove()

    # Step 3: For each label, find nearest prompts
    print(f"\n{'='*70}")
    print(f"LABEL WORD PROJECTIONS INTO PROMPT PCA SPACE")
    print(f"{'='*70}\n")

    # Cosine similarity between labels and prompts
    cos_sim = label_embs @ prompt_embs_n.T

    print(f"  {'Label':<16s} {'PC1':>7s} {'PC2':>7s}  Nearest prompts")
    print(f"  {'-'*66}")

    label_results = []
    for i, word in enumerate(label_names):
        x, y = label_coords[i]
        # Find top 3 nearest prompts by cosine similarity
        top_idx = np.argsort(cos_sim[i])[-3:][::-1]
        nearest = [(int(j), round(float(cos_sim[i, j]), 3), PROMPTS[j][:50]) for j in top_idx]

        print(f"  {word:<16s} {x:+7.3f} {y:+7.3f}  {nearest[0][2]}")
        for _, sim, ptext in nearest[1:]:
            print(f"  {'':16s} {'':7s} {'':7s}  {ptext}  ({sim:.3f})")

        label_results.append({
            "word": word,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "nearest": [{"pi": int(j), "sim": round(float(cos_sim[i,j]), 3),
                          "prompt": PROMPTS[j][:70]} for j in top_idx],
        })

    # Step 4: Check — do safety-relevant labels land near harmful territory?
    print(f"\n{'='*70}")
    print(f"DOMAIN COVERAGE ANALYSIS")
    print(f"{'='*70}\n")

    # Group labels by category
    categories = {
        "medical": LABELS[:4],
        "legal": LABELS[4:8],
        "coding": LABELS[8:12],
        "cooking": LABELS[12:16],
        "finance": LABELS[16:20],
        "safety": LABELS[20:28],
        "academic": LABELS[28:36],
        "ambiguous": LABELS[36:],
    }

    print(f"  {'Category':<12s} {'mean PC1':>8s} {'mean PC2':>8s}  Label words")
    print(f"  {'-'*60}")
    cat_coords = {}
    for cat, words in categories.items():
        idx = [label_names.index(w) for w in words]
        cx = np.mean([label_coords[i, 0] for i in idx])
        cy = np.mean([label_coords[i, 1] for i in idx])
        cat_coords[cat] = (cx, cy)
        print(f"  {cat:<12s} {cx:+8.3f} {cy:+8.3f}  {', '.join(words)}")

    # Distances between category centroids
    print(f"\n  INTER-CATEGORY DISTANCES (Euclidean in PCA space)")
    cat_names = list(categories.keys())
    print(f"  {'':12s}", end="")
    for c in cat_names:
        print(f" {c:>10s}", end="")
    print()
    for ca in cat_names:
        print(f"  {ca:<12s}", end="")
        for cb in cat_names:
            d = np.sqrt((cat_coords[ca][0]-cat_coords[cb][0])**2 +
                        (cat_coords[ca][1]-cat_coords[cb][1])**2)
            print(f" {d:10.3f}", end="")
        print()

    # Export
    export = {
        "pca": [round(float(x)*100, 1) for x in pca.explained_variance_ratio_[:2]],
        "prompts": [{"p": PROMPTS[i][:70], "x": round(float(prompt_coords[i,0]), 4),
                      "y": round(float(prompt_coords[i,1]), 4)} for i in range(len(PROMPTS))],
        "labels": label_results,
        "categories": {cat: {"words": words,
                              "cx": round(float(cat_coords[cat][0]), 4),
                              "cy": round(float(cat_coords[cat][1]), 4)}
                        for cat, words in categories.items()},
    }
    with open("domain_labels.json", "w") as f:
        json.dump(export, f, indent=1, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\n  Exported to domain_labels.json")


if __name__ == "__main__":
    main()