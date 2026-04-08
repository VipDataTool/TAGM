"""
Noun sequence anchor test.

Do short sequences of domain-specific nouns create distinct anchor
points on the domain surface? Test 1-noun, 2-noun, and 3-noun
sequences to find the minimum context for separation.

Run:
    python noun_anchor_test.py
"""

import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Domain noun sequences: 1, 2, and 3 nouns
ANCHORS = {
    "medical": {
        1: ["scalpel", "diagnosis", "tumor", "anesthesia"],
        2: ["scalpel diagnosis", "tumor biopsy", "anesthesia surgery", "insulin diabetes"],
        3: ["scalpel surgery anesthesia", "tumor biopsy pathology", "insulin glucose diabetes", "stethoscope diagnosis patient"],
    },
    "legal": {
        1: ["verdict", "plaintiff", "statute", "jurisdiction"],
        2: ["verdict plaintiff", "statute jurisdiction", "deposition testimony", "indictment arraignment"],
        3: ["verdict plaintiff defendant", "statute jurisdiction precedent", "deposition testimony subpoena", "indictment arraignment bail"],
    },
    "coding": {
        1: ["compiler", "recursion", "mutex", "repository"],
        2: ["compiler debugger", "recursion stack", "mutex thread", "repository commit"],
        3: ["compiler debugger breakpoint", "recursion stack overflow", "mutex thread deadlock", "repository commit branch"],
    },
    "cooking": {
        1: ["spatula", "roux", "julienne", "blanching"],
        2: ["spatula skillet", "roux bechamel", "julienne mandoline", "blanching colander"],
        3: ["spatula skillet sautee", "roux bechamel butter", "julienne mandoline knife", "blanching colander broth"],
    },
    "finance": {
        1: ["dividend", "amortization", "portfolio", "arbitrage"],
        2: ["dividend yield", "amortization principal", "portfolio allocation", "arbitrage spread"],
        3: ["dividend yield equity", "amortization principal mortgage", "portfolio allocation diversification", "arbitrage spread derivatives"],
    },
    "cybersecurity": {
        1: ["firewall", "exploit", "phishing", "ransomware"],
        2: ["firewall intrusion", "exploit vulnerability", "phishing credential", "ransomware encryption"],
        3: ["firewall intrusion detection", "exploit vulnerability payload", "phishing credential harvesting", "ransomware encryption decryption"],
    },
    "chemistry": {
        1: ["catalyst", "reagent", "isotope", "polymer"],
        2: ["catalyst reaction", "reagent solvent", "isotope decay", "polymer monomer"],
        3: ["catalyst reaction kinetics", "reagent solvent titration", "isotope decay radioactive", "polymer monomer synthesis"],
    },
}

DOMAINS = list(ANCHORS.keys())


def get_embedding(model, tokenizer, text, captured):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    h = captured["h"][0]
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

    # Test each n-gram length
    for n_nouns in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"  {n_nouns}-NOUN SEQUENCES")
        print(f"{'='*60}")

        all_embs = []
        all_labels = []
        all_texts = []

        for domain in DOMAINS:
            for text in ANCHORS[domain][n_nouns]:
                emb = get_embedding(model, tokenizer, text, captured)
                norm = np.linalg.norm(emb)
                if norm > 1e-12:
                    emb = emb / norm
                all_embs.append(emb)
                all_labels.append(domain)
                all_texts.append(text)

        X = np.array(all_embs)

        # Within vs between domain similarity
        within, between = [], []
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                sim = float(X[i] @ X[j])
                if all_labels[i] == all_labels[j]:
                    within.append(sim)
                else:
                    between.append(sim)

        gap = np.mean(within) - np.mean(between)
        d = gap / np.sqrt((np.std(within)**2 + np.std(between)**2) / 2)

        print(f"\n  Within:  {np.mean(within):.4f} (std {np.std(within):.4f})")
        print(f"  Between: {np.mean(between):.4f} (std {np.std(between):.4f})")
        print(f"  Gap:     {gap:+.4f}")
        print(f"  Cohen's d: {d:.3f}")

        # PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        print(f"  PCA: {pca.explained_variance_ratio_[0]*100:.1f}%, "
              f"{pca.explained_variance_ratio_[1]*100:.1f}%")

        # Domain x domain similarity
        print(f"\n  {'':14s}", end="")
        for dom in DOMAINS:
            print(f" {dom[:6]:>7s}", end="")
        print()
        for da in DOMAINS:
            print(f"  {da:<14s}", end="")
            for db in DOMAINS:
                idx_a = [i for i, l in enumerate(all_labels) if l == da]
                idx_b = [i for i, l in enumerate(all_labels) if l == db]
                sims = [float(X[i] @ X[j]) for i in idx_a for j in idx_b if i != j]
                print(f" {np.mean(sims):7.3f}", end="")
            print()

        # Print coordinates
        print(f"\n  {'Domain':<14s} {'Text':<35s} {'PC1':>7s} {'PC2':>7s}")
        print(f"  {'-'*65}")
        for i in range(len(all_texts)):
            print(f"  {all_labels[i]:<14s} {all_texts[i]:<35s} {coords[i,0]:+7.3f} {coords[i,1]:+7.3f}")

    handle.remove()

    # Final comparison summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: SEPARATION BY SEQUENCE LENGTH")
    print(f"{'='*60}")
    print(f"\n  Run complete. Compare Cohen's d across 1/2/3 noun sequences")
    print(f"  to determine minimum anchor length for domain separation.")


if __name__ == "__main__":
    main()