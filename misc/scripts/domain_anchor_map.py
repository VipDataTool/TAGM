"""
Anchor-referenced session map.

Projects the 100-prompt TASM session AND the noun triplet anchors
into the same PCA space. Anchors are fixed landmarks. Session prompts
carry their RD/stress/displacement data from results.json.

Run from the tasm/ directory with results.json from a session:
    python anchor_session_map.py /path/to/results.json

Outputs anchor_session_map.json for plotting.
"""

import sys
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

ANCHORS = {
    "medical": [
        "scalpel surgery anesthesia",
        "tumor biopsy pathology",
        "insulin glucose diabetes",
        "stethoscope diagnosis patient",
    ],
    "legal": [
        "verdict plaintiff defendant",
        "statute jurisdiction precedent",
        "deposition testimony subpoena",
        "indictment arraignment bail",
    ],
    "coding": [
        "compiler debugger breakpoint",
        "recursion stack overflow",
        "mutex thread deadlock",
        "repository commit branch",
    ],
    "cooking": [
        "spatula skillet sautee",
        "roux bechamel butter",
        "julienne mandoline knife",
        "blanching colander broth",
    ],
    "finance": [
        "dividend yield equity",
        "amortization principal mortgage",
        "portfolio allocation diversification",
        "arbitrage spread derivatives",
    ],
    "cybersecurity": [
        "firewall intrusion detection",
        "exploit vulnerability payload",
        "phishing credential harvesting",
        "ransomware encryption decryption",
    ],
    "chemistry": [
        "catalyst reaction kinetics",
        "reagent solvent titration",
        "isotope decay radioactive",
        "polymer monomer synthesis",
    ],
}


def get_embedding(model, tokenizer, text, captured):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    h = captured["h"][0]
    return h[1:].mean(dim=0).numpy() if h.shape[0] > 1 else h[0].numpy()


def main():
    if len(sys.argv) < 2:
        print("Usage: python anchor_session_map.py /path/to/results.json")
        sys.exit(1)

    results_path = sys.argv[1]
    print(f"Loading session: {results_path}")
    with open(results_path) as f:
        session = json.load(f)
    print(f"  {len(session)} prompts")

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    n_layers = model.config.num_hidden_layers
    mid_layer = n_layers // 2
    print(f"  {n_layers} layers, mid={mid_layer}")

    captured = {}
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    handle = model.model.layers[mid_layer].input_layernorm.register_forward_hook(hook_fn)

    # ── Embed all session prompts ──
    print("\nEmbedding session prompts...")
    prompt_embs = []
    for i, r in enumerate(session):
        emb = get_embedding(model, tokenizer, r["prompt"], captured)
        prompt_embs.append(emb)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(session)}")

    # ── Embed all anchors ──
    print("Embedding anchors...")
    anchor_embs = []
    anchor_meta = []
    for domain, texts in ANCHORS.items():
        for text in texts:
            emb = get_embedding(model, tokenizer, text, captured)
            anchor_embs.append(emb)
            anchor_meta.append({"domain": domain, "text": text})

    handle.remove()

    # ── Normalize everything ──
    all_embs = np.array(prompt_embs + anchor_embs)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1
    all_embs_n = all_embs / norms

    n_prompts = len(session)
    n_anchors = len(anchor_meta)

    # ── PCA on prompts only, project anchors into same space ──
    prompt_embs_n = all_embs_n[:n_prompts]
    anchor_embs_n = all_embs_n[n_prompts:]

    pca = PCA(n_components=2)
    prompt_coords = pca.fit_transform(prompt_embs_n)
    anchor_coords = pca.transform(anchor_embs_n)

    print(f"PCA: {pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"{pca.explained_variance_ratio_[1]*100:.1f}%")

    # ── Find nearest anchor for each prompt ──
    cos_sim = prompt_embs_n @ anchor_embs_n.T  # [n_prompts, n_anchors]
    nearest_anchor_idx = np.argmax(cos_sim, axis=1)
    nearest_anchor_sim = np.max(cos_sim, axis=1)

    # ── Build export ──
    prompt_export = []
    for i, r in enumerate(session):
        rd = r.get("rank_displacement", {})
        nearest = anchor_meta[nearest_anchor_idx[i]]
        prompt_export.append({
            "p": r["prompt"][:80],
            "cat": r.get("category", ""),
            "x": round(float(prompt_coords[i, 0]), 4),
            "y": round(float(prompt_coords[i, 1]), 4),
            "stress": round(float(r.get("stress_score", 0)), 3),
            "disp": round(float(rd.get("mean_disp_per_token", 0)), 4),
            "repl": round(float(rd.get("mean_replacement", 0)), 3),
            "kl": round(float(r.get("kl_divergence", 0) or 0), 3),
            "near_dom": nearest["domain"],
            "near_anc": nearest["text"],
            "near_sim": round(float(nearest_anchor_sim[i]), 3),
        })

    anchor_export = []
    for i, meta in enumerate(anchor_meta):
        anchor_export.append({
            "domain": meta["domain"],
            "text": meta["text"],
            "x": round(float(anchor_coords[i, 0]), 4),
            "y": round(float(anchor_coords[i, 1]), 4),
        })

    # ── Summary: category x domain crosstab ──
    print(f"\n{'='*60}")
    print("CATEGORY × NEAREST DOMAIN CROSSTAB")
    print(f"{'='*60}\n")

    domains = sorted(ANCHORS.keys())
    categories = sorted(set(r.get("category", "") for r in session))

    print(f"  {'':12s}", end="")
    for d in domains:
        print(f" {d[:8]:>8s}", end="")
    print()

    for cat in categories:
        print(f"  {cat:12s}", end="")
        for dom in domains:
            n = sum(1 for p in prompt_export
                    if p["cat"] == cat and p["near_dom"] == dom)
            print(f" {n:8d}", end="")
        print()

    # ── Per-domain stress/displacement summary ──
    print(f"\n{'='*60}")
    print("METRICS BY NEAREST DOMAIN")
    print(f"{'='*60}\n")

    print(f"  {'Domain':<14s} {'n':>4s} {'stress':>8s} {'disp':>8s} {'repl':>8s} {'kl':>8s}")
    print(f"  {'-'*54}")
    for dom in domains:
        pts = [p for p in prompt_export if p["near_dom"] == dom]
        if not pts:
            continue
        n = len(pts)
        print(f"  {dom:<14s} {n:4d} "
              f"{np.mean([p['stress'] for p in pts]):8.3f} "
              f"{np.mean([p['disp'] for p in pts]):8.4f} "
              f"{np.mean([p['repl'] for p in pts]):8.3f} "
              f"{np.mean([p['kl'] for p in pts]):8.3f}")

    # ── Per-category per-domain stress ──
    print(f"\n{'='*60}")
    print("STRESS BY CATEGORY × DOMAIN")
    print(f"{'='*60}\n")
    print(f"  {'':12s}", end="")
    for d in domains:
        print(f" {d[:8]:>8s}", end="")
    print()
    for cat in categories:
        print(f"  {cat:12s}", end="")
        for dom in domains:
            pts = [p for p in prompt_export
                   if p["cat"] == cat and p["near_dom"] == dom]
            if pts:
                print(f" {np.mean([p['stress'] for p in pts]):8.3f}", end="")
            else:
                print(f" {'--':>8s}", end="")
        print()

    # ── Export ──
    export = {
        "pca": [round(float(x) * 100, 1) for x in pca.explained_variance_ratio_[:2]],
        "n_prompts": n_prompts,
        "n_anchors": n_anchors,
        "prompts": prompt_export,
        "anchors": anchor_export,
    }

    with open("anchor_session_map.json", "w") as f:
        json.dump(export, f, indent=1,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nExported to anchor_session_map.json")


if __name__ == "__main__":
    main()