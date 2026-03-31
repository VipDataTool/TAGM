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

logger = logging.getLogger("tasm")

LEVEL_COLS = ["nouns", "phrase", "question", "instruction", "meta_instruction"]
LEVEL_NAMES = ["nouns", "phrase", "question", "instruct", "meta"]


# ─── Probe Loading ────────────────────────────────────────────

def _discover_probe_files(root_dir):
    """Find all *_probes.csv files in the project root."""
    pattern = os.path.join(root_dir, "*_probes.csv")
    files = sorted(glob(pattern))
    return [os.path.basename(f) for f in files]


def _load_probes(csv_path):
    """Load probes from CSV. Returns list of dicts with subject, anchor_id, level, text."""
    probes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for level, col in enumerate(LEVEL_COLS):
                text = row.get(col, "").strip()
                if text:
                    probes.append({
                        "subject": row["subject"].strip(),
                        "anchor_id": row.get("anchor_id", "").strip(),
                        "level": level,
                        "text": text,
                    })
    return probes


# ─── Probe Embedding Cache ────────────────────────────────────

PROBE_CACHE_DIR = "probe_cache"


def _probe_cache_path(project_root, probe_file, model_id, layer_frac=0.50):
    """Build the cache file path for pre-computed probe embeddings."""
    cache_dir = os.path.join(project_root, PROBE_CACHE_DIR)
    safe_model = model_id.replace("/", "__").replace("\\", "__")
    stem = os.path.splitext(probe_file)[0]
    layer_tag = str(int(layer_frac * 100))
    return os.path.join(cache_dir, f"{stem}__{safe_model}__L{layer_tag}.json")


def _load_probe_cache(cache_path):
    """Load cached probe embeddings. Returns dict or None."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[DOMAIN] Failed to load probe cache {cache_path}: {e}")
        return None


def embed_and_cache_probes(model, tokenizer, project_root, probe_file,
                           model_id, layer_frac=0.50, progress=None):
    """Pre-embed all probes from a CSV and save to cache.

    Called from app.py after model load.  Registers a forward hook on the
    target layer's input_layernorm, runs each probe text through the model,
    and stores the mean hidden-state embedding.

    Args:
        model: HuggingFace causal LM (instruct model).
        tokenizer: HuggingFace tokenizer.
        project_root: TASM project root directory.
        probe_file: Probe CSV filename (relative to project_root).
        model_id: Model identifier string for cache key.
        layer_frac: Capture depth as fraction of model depth (0.0–1.0).
        progress: Optional status callback.

    Returns:
        Path to the saved cache file, or None on failure.
    """
    import torch

    csv_path = os.path.join(project_root, probe_file)
    if not os.path.exists(csv_path):
        logger.warning(f"[DOMAIN] Probe file not found: {csv_path}")
        return None

    probes = _load_probes(csv_path)
    if not probes:
        logger.warning(f"[DOMAIN] No probes loaded from {csv_path}")
        return None

    n_layers = model.config.num_hidden_layers
    target_layer = max(0, min(n_layers - 1, int(layer_frac * n_layers)))

    # Set up hook
    captured = {}

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    handle = model.model.layers[target_layer].input_layernorm.register_forward_hook(hook_fn)

    embeddings = []
    try:
        for i, probe in enumerate(probes):
            inputs = tokenizer(probe["text"], return_tensors="pt")
            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            h = captured["h"][0]
            emb = h[1:].mean(dim=0).float().cpu().numpy() if h.shape[0] > 1 else h[0].float().cpu().numpy()
            norm = float(np.linalg.norm(emb))
            if norm > 1e-12:
                emb = emb / norm
            embeddings.append(emb.tolist())
            if progress and (i + 1) % 50 == 0:
                progress("probes", f"Embedding probes: {i+1}/{len(probes)}")
    finally:
        handle.remove()

    # Save cache
    cache_path = _probe_cache_path(project_root, probe_file, model_id, layer_frac)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache_data = {
        "model_id": model_id,
        "layer": target_layer,
        "layer_frac": layer_frac,
        "n_layers": n_layers,
        "probe_file": probe_file,
        "n_probes": len(probes),
        "embeddings": embeddings,
    }

    with open(cache_path, "w") as f:
        json.dump(cache_data, f, separators=(',', ':'))

    logger.info(f"[DOMAIN] Cached {len(embeddings)} probe embeddings → {cache_path}")
    return cache_path


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
                        subjects, top_n=20, min_appearances=2, progress=None):
    """Build per-token observations with all metrics and proximity."""
    token_freq = defaultdict(int)
    raw_obs = []

    for pi, sr in enumerate(session_results):
        dx, dy = float(prompt_coords[pi, 0]), float(prompt_coords[pi, 1])
        cat = sr.get("category", "?")[0]
        toks = sr.get("tokens", [])
        rd = sr.get("rank_displacement", {})
        per_pos = rd.get("per_position", []) if rd else []
        stress = sr.get("per_token_stress", [])
        sfd = sr.get("sfd", {})
        sfd_e = sfd.get("per_token_energy", []) if sfd else []
        sfd_d = sfd.get("per_token_density", []) if sfd else []

        for pos in range(len(per_pos)):
            if pos >= len(toks):
                continue
            tok = toks[pos].strip()
            if not tok:
                continue
            token_freq[tok] += 1

            pd = per_pos[pos]
            raw_obs.append({
                "tok": tok, "cat": cat,
                "dx": dx, "dy": dy,
                "disp": float(pd.get("total_disp", 0)),
                "repl": float(pd.get("replacement_ratio", 0)),
                "asm": float(stress[pos]) if pos < len(stress) else 0,
                "sfd_e": float(sfd_e[pos]) if pos < len(sfd_e) else 0,
                "sfd_d": float(sfd_d[pos]) if pos < len(sfd_d) else 0,
                "pi": pi, "pos": pos,
            })

    # Top tokens by frequency, filter by min appearances, compute CV, order by CV
    qualified = {t: n for t, n in token_freq.items() if n >= min_appearances}
    top = sorted(qualified.keys(), key=lambda t: -qualified[t])[:top_n]
    token_cv = {}
    for tok in top:
        disps = [o["disp"] for o in raw_obs if o["tok"] == tok]
        if len(disps) >= 2:
            m = np.mean(disps)
            token_cv[tok] = round(float(np.std(disps) / max(abs(m), 1e-12)), 3)
        else:
            token_cv[tok] = 0
    ordered = sorted(top, key=lambda t: token_cv.get(t, 0))

    # Filter and compute proximity
    top_set = set(ordered)
    subj_idx = {s: i for i, s in enumerate(subjects)}
    obs_export = []

    for o in raw_obs:
        if o["tok"] not in top_set:
            continue

        aidx, adist = _nearest_probe(o["dx"], o["dy"], anchor_pts)
        a = anchor_pts[aidx]

        obs_export.append([
            o["tok"], o["cat"],
            round(o["dy"], 4), round(o["disp"], 3), round(o["repl"], 2),
            round(o["dx"], 4),
            round(o["asm"], 2), round(o["sfd_e"], 3), round(o["sfd_d"], 3),
            o["pi"], o["pos"],
            round(adist, 4), a["level"], subj_idx.get(a["subject"], 0),
        ])

    if progress:
        progress(f"Built {len(obs_export)} observations for {len(ordered)} tokens")

    return obs_export, ordered, token_cv


def _stratification(obs, subjects):
    """Compute category counts by nearest level and subject."""
    by_level = defaultdict(lambda: defaultdict(int))
    by_subject = defaultdict(lambda: defaultdict(int))

    for o in obs:
        cat = o[1]
        by_level[o[12]][cat] += 1
        if o[13] < len(subjects):
            by_subject[subjects[o[13]]][cat] += 1

    return {
        "by_level": {str(k): dict(v) for k, v in sorted(by_level.items())},
        "by_subject": {k: dict(v) for k, v in sorted(by_subject.items())},
    }


# ─── Module Class ────────────────────────────────────────────

class DomainSurfaceModule(TASMModule):
    """Subject-matter domain surface analysis.

    Pure post-processor: uses pre-computed domain_embedding fields
    from the analyzer and cached probe embeddings from model load.
    """

    name = "domain_surface"
    display_name = "Domain Surface"
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
        self._probe_files = []
        self._project_root = None

    def set_project_root(self, root):
        """Set project root for probe file discovery."""
        self._project_root = root
        self._probe_files = _discover_probe_files(root)

    @property
    def parameters(self):
        options = self._probe_files if self._probe_files else ["alignment_probes.csv"]
        return [
            ModuleParameter(
                name="probe_file",
                display_name="Probe File",
                description="Subject-matter probe definitions (CSV)",
                type="select",
                default=options[0] if options else "alignment_probes.csv",
                options=options,
            ),
            ModuleParameter(
                name="top_tokens",
                display_name="Top Tokens",
                description="Number of most-frequent tokens to include",
                type="int",
                default=30,
                min_val=5,
                max_val=100,
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

        # Check probe file
        probe_file = params.get("probe_file", "alignment_probes.csv")
        if self._project_root:
            path = os.path.join(self._project_root, probe_file)
            if not os.path.exists(path):
                return False, f"Probe file not found: {path}"

        return True, "OK"

    def run(self, session_results, params, progress=None):
        """Execute domain surface analysis using pre-computed embeddings.

        Requires:
            - domain_embedding fields in session_results (from analyzer)
            - Cached probe embeddings (from model load)
        """
        probe_file = params.get("probe_file", "alignment_probes.csv")
        top_tokens = params.get("top_tokens", 30)
        min_appearances = params.get("min_appearances", 2)
        pca_components = params.get("pca_components", 2)

        # Resolve probe path
        if self._project_root:
            probe_path = os.path.join(self._project_root, probe_file)
        else:
            probe_path = probe_file

        # Load probes
        if progress:
            progress(f"Loading probes from {probe_file}")
        probes = _load_probes(probe_path)
        subjects = sorted(set(p["subject"] for p in probes))
        logger.info(f"[DOMAIN] Loaded {len(probes)} probes across "
                     f"{len(subjects)} subjects")

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

        # Load cached probe embeddings
        if progress:
            progress("Loading cached probe embeddings...")
        probe_embs = self._load_probe_embeddings(probe_file, session_results)
        if probe_embs is None:
            raise RuntimeError(
                "Probe embeddings not found. They are generated automatically "
                "at model load time. Try reloading the model, or check that "
                f"{probe_file} exists in the project root."
            )

        if len(probe_embs) != len(probes):
            raise RuntimeError(
                f"Probe embedding count ({len(probe_embs)}) does not match "
                f"probe count ({len(probes)}). Cache may be stale — "
                f"reload the model to regenerate."
            )

        probe_embs = np.array(probe_embs)

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
                "text": p["text"][:50],
                "x": float(probe_coords[i, 0]),
                "y": float(probe_coords[i, 1]),
            })

        # Build observations
        if progress:
            progress("Building per-token observations...")
        obs, ordered_tokens, token_cv = _build_observations(
            session_subset, prompt_coords, anchor_pts,
            subjects, top_tokens, min_appearances, progress)

        # Stratification
        strat = _stratification(obs, subjects)

        # Log stratification summary
        if progress:
            for level_idx in sorted(strat["by_level"].keys()):
                counts = strat["by_level"][level_idx]
                total = sum(counts.values())
                li = int(level_idx)
                if li < len(LEVEL_NAMES):
                    progress(f"  {LEVEL_NAMES[li]}: {total} obs "
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
        prompts = [r["prompt"][:80] for r in session_subset]

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
                "asm", "sfd_e", "sfd_d", "pi", "pos",
                "near_dist", "near_level", "near_subj_idx",
            ],
            "stratification": strat,
            "probe_file": probe_file,
            "level_names": LEVEL_NAMES,
        }

        if progress:
            progress(f"Complete: {len(obs)} observations, "
                     f"{len(anchors_compact)} anchors, "
                     f"{len(subjects)} subjects")

        return output

    def _load_probe_embeddings(self, probe_file, session_results):
        """Load cached probe embeddings matching the current model and layer config.

        Scans the probe cache directory for matching files. Prefers caches
        that match the current domain_embedding_layer_frac from engine config.
        """
        if not self._project_root:
            return None

        cache_dir = os.path.join(self._project_root, PROBE_CACHE_DIR)
        if not os.path.isdir(cache_dir):
            return None

        stem = os.path.splitext(probe_file)[0]
        candidates = sorted(glob(os.path.join(cache_dir, f"{stem}__*.json")))
        if not candidates:
            return None

        # Try to match the current layer config
        try:
            from engine import engine_config
            layer_frac = engine_config.get("domain_embedding_layer_frac") or 0.50
        except Exception:
            layer_frac = 0.50
        layer_tag = f"__L{int(layer_frac * 100)}.json"

        # Prefer cache matching current layer, fall back to any available
        matched = [c for c in candidates if layer_tag in c]
        cache_path = matched[-1] if matched else candidates[-1]

        cache = _load_probe_cache(cache_path)
        if cache is None:
            return None

        logger.info(f"[DOMAIN] Using probe cache: {os.path.basename(cache_path)} "
                     f"(model={cache.get('model_id', '?')}, "
                     f"layer={cache.get('layer', '?')}, "
                     f"frac={cache.get('layer_frac', '?')})")

        return cache.get("embeddings")
