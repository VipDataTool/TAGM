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

FIXED_COLS = {"subject", "anchor_id"}  # Non-escalation columns in probe CSVs

# Legacy defaults — used only if CSV has no header or auto-detection fails.
_DEFAULT_LEVEL_COLS = ["nouns", "phrase", "question", "instruction", "meta_instruction"]
_DEFAULT_LEVEL_NAMES = ["nouns", "phrase", "question", "instruct", "meta"]


# ─── Probe Loading ────────────────────────────────────────────

def _discover_probe_files(root_dir):
    """Find all *_probes.csv files in the project root."""
    pattern = os.path.join(root_dir, "*_probes.csv")
    files = sorted(glob(pattern))
    return [os.path.basename(f) for f in files]


def _detect_level_cols(csv_path):
    """Read the CSV header and return escalation columns (everything after subject/anchor_id).

    Returns (level_cols, level_names) where level_names are display-friendly versions.
    Returns ([], []) if the CSV is not a valid probe file (no 'subject' column).
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

    # A valid probe CSV must have a 'subject' column
    header_lower = {h.strip().lower() for h in headers}
    if "subject" not in header_lower:
        return [], []

    level_cols = [h for h in headers if h.strip().lower() not in {c.lower() for c in FIXED_COLS}]

    if not level_cols:
        return _DEFAULT_LEVEL_COLS[:], _DEFAULT_LEVEL_NAMES[:]

    # Generate display names: replace underscores with spaces
    level_names = [col.replace("_", " ").strip() for col in level_cols]

    return level_cols, level_names


def _load_probes(csv_path):
    """Load probes from CSV. Auto-detects escalation columns from header.

    Returns list of dicts with subject, anchor_id, level, text.
    Returns empty list if the CSV is not a valid probe file.
    """
    level_cols, _ = _detect_level_cols(csv_path)
    if not level_cols:
        return []

    probes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "subject" not in row:
                continue
            for level, col in enumerate(level_cols):
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


def _probe_cache_path(project_root, probe_file, model_id, layer_frac=0.50,
                      projected=False):
    """Build the cache file path for pre-computed probe embeddings."""
    cache_dir = os.path.join(project_root, PROBE_CACHE_DIR)
    safe_model = model_id.replace("/", "__").replace("\\", "__")
    stem = os.path.splitext(probe_file)[0]
    layer_tag = str(int(layer_frac * 100))
    proj_tag = "_proj" if projected else ""
    return os.path.join(cache_dir, f"{stem}__{safe_model}__L{layer_tag}{proj_tag}.json")


def _load_probe_cache(cache_path):
    """Load cached probe embeddings. Returns dict or None.
    
    Validates that embeddings don't contain NaN (can happen if cache
    was generated with a dtype that caused overflow, e.g. fp16 on CPU).
    """
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        # Validate: check for NaN in embeddings
        embeddings = data.get("embeddings", [])
        for emb in embeddings[:3]:  # spot-check first few
            if any(v is None or (isinstance(v, float) and (v != v)) for v in emb):
                logger.warning(f"[DOMAIN] Probe cache contains NaN — invalidating stale cache: {cache_path}")
                os.remove(cache_path)
                return None
        return data
    except Exception as e:
        logger.warning(f"[DOMAIN] Failed to load probe cache {cache_path}: {e}")
        return None


def embed_and_cache_probes(model, tokenizer, project_root, probe_file,
                           model_id, layer_frac=0.50, progress=None,
                           delta_matrix=None):
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
        delta_matrix: Optional o_proj delta tensor for correction-space
            projection. If provided, embeddings are projected through
            h @ delta.T before normalization.

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
            emb = h[1:].mean(dim=0).float() if h.shape[0] > 1 else h[0].float()
            # Project through correction field if delta provided
            if delta_matrix is not None:
                emb = torch.matmul(emb.cpu(), delta_matrix.float().cpu().T)
            emb = emb.cpu().numpy()
            norm = float(np.linalg.norm(emb))
            if norm > 1e-12:
                emb = emb / norm
            embeddings.append(emb.tolist())
            if progress and (i + 1) % 50 == 0:
                progress("probes", f"Embedding probes: {i+1}/{len(probes)}")
    finally:
        handle.remove()

    # Save cache
    cache_path = _probe_cache_path(project_root, probe_file, model_id, layer_frac,
                                    projected=delta_matrix is not None)
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
                        subjects, top_n=20, min_appearances=2, progress=None,
                        probe_embs=None, probes=None,
                        esc_probe_embs=None,
                        tv_eta_weights=None):
    """Build per-token observations with all metrics and proximity.

    Split-depth probe matching: each token gets TWO independent probe lookups.
      - Subject (angular wedge): cosine similarity of token's domain-layer
        embedding against probe embeddings cached at domain_embedding_layer_frac.
      - Escalation (ring): cosine similarity of token's escalation-layer
        embedding against probe embeddings cached at domain_escalation_layer_frac.

    If tv_eta_weights is provided (dict of token → eta_sq_density from Token
    Variance module), content words are weighted higher than function words
    in the top-N selection. Without it, falls back to pure frequency ranking.

    Falls back to prompt-level PCA nearest-probe matching when per-token
    embeddings are not available.
    """
    token_freq = defaultdict(int)
    raw_obs = []

    # Pre-build subject index for probe-level matching
    subj_set = sorted(set(p["subject"] for p in probes)) if probes else sorted(subjects)
    subj_to_idx = {s: i for i, s in enumerate(subj_set)}

    for pi, sr in enumerate(session_results):
        dx, dy = float(prompt_coords[pi, 0]), float(prompt_coords[pi, 1])
        cat = (sr.get("category", "") or "?")[0]
        toks = sr.get("tokens", [])
        rd = sr.get("rank_displacement", {})
        per_pos = rd.get("per_position", []) if rd else []
        stress = sr.get("per_token_stress", [])
        sfd = sr.get("sfd", {})
        sfd_e = sfd.get("per_token_energy", []) if sfd else []
        sfd_d = sfd.get("per_token_density", []) if sfd else []

        # Per-token embeddings at both depths (if available)
        ptde = sr.get("per_token_domain_emb")       # subject layer
        ptee = sr.get("per_token_escalation_emb")    # escalation layer
        ptde_offset = sr.get("per_token_domain_offset", 1)

        for pos in range(len(per_pos)):
            if pos >= len(toks):
                continue
            tok = toks[pos].strip()
            if not tok:
                continue
            token_freq[tok] += 1

            pd = per_pos[pos]
            adj = pos - ptde_offset
            raw_obs.append({
                "tok": tok, "cat": cat,
                "dx": dx, "dy": dy,
                "disp": float(pd.get("total_disp", 0)),
                "repl": float(pd.get("replacement_ratio", 0)),
                "asm": float(stress[pos]) if pos < len(stress) else 0,
                "sfd_e": float(sfd_e[pos]) if pos < len(sfd_e) else 0,
                "sfd_d": float(sfd_d[pos]) if pos < len(sfd_d) else 0,
                "pi": pi, "pos": pos,
                "_emb": ptde[adj] if ptde and pos >= ptde_offset and adj < len(ptde) else None,
                "_esc_emb": ptee[adj] if ptee and pos >= ptde_offset and adj < len(ptee) else None,
            })

    # Top tokens by frequency, filter by min appearances, compute CV, order by CV
    qualified = {t: n for t, n in token_freq.items() if n >= min_appearances}

    # If token variance eta² is available, weight selection so category-
    # dependent tokens (high eta_sq = content words) rank above category-
    # independent tokens (low eta_sq = function words).
    # score = freq × (eta² + floor)
    ETA_FLOOR = 0.01
    if tv_eta_weights:
        top = sorted(qualified.keys(),
                     key=lambda t: -(qualified[t] * (tv_eta_weights.get(t, ETA_FLOOR) + ETA_FLOOR)))[:top_n]
    else:
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

    # Precompute probe embedding matrices
    subj_probe_mat = np.array(probe_embs) if probe_embs is not None else None
    esc_probe_mat = np.array(esc_probe_embs) if esc_probe_embs is not None else subj_probe_mat

    # Subject angles for continuous positioning
    n_subj = len(subjects)
    _subj_angles = np.linspace(0, 2 * np.pi, n_subj, endpoint=False) - np.pi / 2

    token_knn = 5  # k nearest probes per token

    for o in raw_obs:
        if o["tok"] not in top_set:
            continue

        subj_emb = o.get("_emb")
        esc_emb = o.get("_esc_emb")

        # ── Subject assignment (kNN from subject-layer embedding) ──
        near_angle = 0.0
        if subj_emb is not None and subj_probe_mat is not None and probes is not None:
            tok_vec = np.array(subj_emb, dtype=np.float32)
            tok_norm = np.linalg.norm(tok_vec)
            if tok_norm > 1e-12:
                tok_vec = tok_vec / tok_norm
            sims = subj_probe_mat @ tok_vec

            # Top-k nearest probes
            k = min(token_knn, len(sims))
            top_k = np.argsort(sims)[-k:][::-1]
            top_sims = sims[top_k]
            best_dist = float(1.0 - top_sims[0])

            # Similarity-weighted position
            weights = np.exp(top_sims * 10)
            weights /= weights.sum()

            sin_sum = 0
            cos_sum = 0
            subj_w = defaultdict(float)
            for idx, w in zip(top_k, weights):
                if idx < len(probes):
                    si = subj_idx.get(probes[idx]["subject"], 0)
                    sin_sum += w * np.sin(_subj_angles[si])
                    cos_sum += w * np.cos(_subj_angles[si])
                    subj_w[si] += w

            near_angle = float(np.arctan2(sin_sum, cos_sum))
            near_subj = max(subj_w, key=subj_w.get) if subj_w else 0
        else:
            # Fallback: prompt-level PCA
            aidx, best_dist = _nearest_probe(o["dx"], o["dy"], anchor_pts)
            near_subj = subj_idx.get(anchor_pts[aidx]["subject"], 0)
            near_angle = float(_subj_angles[near_subj])

        # ── Escalation assignment (kNN from escalation-layer embedding) ──
        if esc_emb is not None and esc_probe_mat is not None and probes is not None:
            esc_vec = np.array(esc_emb, dtype=np.float32)
            esc_norm = np.linalg.norm(esc_vec)
            if esc_norm > 1e-12:
                esc_vec = esc_vec / esc_norm
            esc_sims = esc_probe_mat @ esc_vec

            k = min(token_knn, len(esc_sims))
            esc_top_k = np.argsort(esc_sims)[-k:][::-1]
            esc_weights = np.exp(esc_sims[esc_top_k] * 10)
            esc_weights /= esc_weights.sum()
            level = float(sum(probes[idx]["level"] * w
                              for idx, w in zip(esc_top_k, esc_weights)
                              if idx < len(probes)))
        elif subj_emb is not None and subj_probe_mat is not None and probes is not None:
            # Use subject-layer kNN for level too
            level = float(sum(probes[idx]["level"] * w
                              for idx, w in zip(top_k, weights)
                              if idx < len(probes)))
        else:
            # Fallback: prompt-level PCA
            aidx, _ = _nearest_probe(o["dx"], o["dy"], anchor_pts)
            level = anchor_pts[aidx]["level"]

        obs_export.append([
            o["tok"], o["cat"],
            round(o["dy"], 4), round(o["disp"], 3), round(o["repl"], 2),
            round(o["dx"], 4),
            round(o["asm"], 2), round(o["sfd_e"], 3), round(o["sfd_d"], 3),
            o["pi"], o["pos"],
            round(best_dist, 4), level, near_subj,
            round(near_angle, 4),  # index 14: kNN-weighted continuous angle
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
        _, level_names = _detect_level_cols(probe_path)
        subjects = sorted(set(p["subject"] for p in probes))
        logger.info(f"[DOMAIN] Loaded {len(probes)} probes across "
                     f"{len(subjects)} subjects, {len(level_names)} levels: {level_names}")

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

        # Load escalation-layer probe embeddings (for split-depth ring assignment)
        try:
            from engine import engine_config as _ec
            esc_frac = _ec.get("domain_escalation_layer_frac") or 0.75
            subj_frac = _ec.get("domain_embedding_layer_frac") or 0.50
        except Exception:
            esc_frac = 0.75
            subj_frac = 0.50

        esc_probe_embs = None
        if esc_frac != subj_frac:
            if progress:
                progress("Loading escalation-layer probe embeddings...")
            esc_raw = self._load_probe_embeddings(
                probe_file, session_results, layer_frac=esc_frac)
            if esc_raw is not None and len(esc_raw) == len(probes):
                esc_probe_embs = np.array(esc_raw)
                logger.info(f"[DOMAIN] Split-depth: escalation probes from "
                            f"L{int(esc_frac*100)}, subject probes from L{int(subj_frac*100)}")
            else:
                logger.warning(f"[DOMAIN] Escalation probe cache not found at L{int(esc_frac*100)}, "
                               f"using single-depth matching")

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

        # Load token variance data if available (to weight content words
        # over function words in top-N selection). Uses eta-squared
        # (category-dependence) rather than raw CV — tokens whose correction
        # signals vary WITH category (content words) rank above tokens whose
        # signals vary independently of category (function words).
        tv_weights = None
        tv_paths = [
            "datasets/current/module_token_variance.json",
            "module_token_variance.json",
        ]
        for tv_path in tv_paths:
            if os.path.exists(tv_path):
                try:
                    with open(tv_path) as f:
                        tv_data = json.load(f)
                    all_tv = tv_data.get("all_tokens", [])
                    if all_tv:
                        tv_weights = {t["token"]: t.get("eta_sq_density", 0)
                                      for t in all_tv}
                        logger.info(f"[DOMAIN] Loaded token variance eta² for "
                                    f"{len(tv_weights)} tokens")
                    break
                except Exception as e:
                    logger.warning(f"[DOMAIN] Failed to load token variance: {e}")

        if tv_weights is None:
            logger.info("[DOMAIN] Token variance data not found — using frequency-only "
                        "token selection. Run Token Variance first for better subject accuracy.")

        obs, ordered_tokens, token_cv = _build_observations(
            session_subset, prompt_coords, anchor_pts,
            subjects, top_tokens, min_appearances, progress,
            probe_embs=probe_embs, probes=probes,
            esc_probe_embs=esc_probe_embs,
            tv_eta_weights=tv_weights)

        # Stratification
        strat = _stratification(obs, subjects)

        # Log stratification summary
        if progress:
            for level_idx in sorted(strat["by_level"].keys()):
                counts = strat["by_level"][level_idx]
                total = sum(counts.values())
                li = int(level_idx)
                if li < len(level_names):
                    progress(f"  {level_names[li]}: {total} obs "
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
                "near_angle",
            ],
            "stratification": strat,
            "probe_file": probe_file,
            "level_names": level_names,
        }

        if progress:
            progress(f"Complete: {len(obs)} observations, "
                     f"{len(anchors_compact)} anchors, "
                     f"{len(subjects)} subjects")

        return output

    def _load_probe_embeddings(self, probe_file, session_results,
                               layer_frac=None):
        """Load cached probe embeddings matching the current model and layer config.

        Scans the probe cache directory for matching files. Prefers caches
        that match the specified layer_frac (defaults to domain_embedding_layer_frac).
        Validates embedding dimensions against session data to prevent
        crosstalk when switching between models of different sizes.
        Tries all candidates until one passes validation.

        Args:
            layer_frac: Override which layer depth to match. If None, uses
                domain_embedding_layer_frac from engine config.
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

        # Determine target layer frac
        if layer_frac is None:
            try:
                from engine import engine_config
                layer_frac = engine_config.get("domain_embedding_layer_frac") or 0.50
            except Exception:
                layer_frac = 0.50

        # Check projection mode
        try:
            from engine import engine_config as _ec
            use_proj = _ec.get("probe_projection_space")
        except Exception:
            use_proj = False
        proj_tag = "_proj" if use_proj else ""
        layer_tag = f"__L{int(layer_frac * 100)}{proj_tag}.json"

        # Order: layer-matched candidates first, then others
        matched = [c for c in candidates if layer_tag in c]
        unmatched = [c for c in candidates if layer_tag not in c]
        ordered = matched + unmatched

        # Determine session embedding dimension for validation
        session_dim = None
        for r in session_results:
            de = r.get("domain_embedding")
            if de and len(de) > 0:
                session_dim = len(de)
                break

        # Try each candidate until one passes dimension validation
        for cache_path in ordered:
            cache = _load_probe_cache(cache_path)
            if cache is None:
                continue

            probe_embs = cache.get("embeddings", [])
            if probe_embs and session_dim is not None:
                probe_dim = len(probe_embs[0])
                if probe_dim != session_dim:
                    logger.warning(f"[DOMAIN] Probe cache dimension mismatch: "
                                   f"cache={probe_dim}, session={session_dim}. "
                                   f"Skipping {os.path.basename(cache_path)}.")
                    continue

            logger.info(f"[DOMAIN] Using probe cache: {os.path.basename(cache_path)} "
                         f"(model={cache.get('model_id', '?')}, "
                         f"layer={cache.get('layer', '?')}, "
                         f"frac={cache.get('layer_frac', '?')})")
            return probe_embs

        logger.warning(f"[DOMAIN] No valid probe cache found for dimension {session_dim}. "
                       f"Re-run with the current model to regenerate.")
        return None
