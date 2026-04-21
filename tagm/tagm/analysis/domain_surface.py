"""Domain Surface: per-token correction signals on a probe-defined surface.

Ported from TASM's engine/modules/domain_surface.py. Maps per-token
embeddings onto a probe-defined PCA surface, computes nearest-probe
assignments, and builds per-token observations with metrics.

Data reading adapted to TAGM's ProbeStore + measurement schema.
Output shape matches TASM so renderDomainSurfaceResults works unchanged.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.analysis.statistics import extract_scalar
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")

NORM_EPS = 1e-12
PROBE_TEXT_DISPLAY_LEN = 30
PROMPT_TEXT_DISPLAY_LEN = 100


def _get_final_emb(r):
    pte = ((r.get("measurements") or {}).get("per_token_embedding") or {})
    embs = (pte.get("objects") or {}).get("per_token_embeddings") or {}
    return embs.get("final")


def _get_depth_emb(r, depth_label):
    pte = ((r.get("measurements") or {}).get("per_token_embedding") or {})
    embs = (pte.get("objects") or {}).get("per_token_embeddings") or {}
    return embs.get(depth_label)


def _get_per_token_stress(r):
    ss = ((r.get("measurements") or {}).get("stress_score") or {})
    return (ss.get("per_token") or {}).get("stress")


def _get_per_token_density(r):
    sfd = ((r.get("measurements") or {}).get("spectral_field_density") or {})
    return (sfd.get("per_token") or {}).get("density")


def _get_rd_per_position(r):
    rd = ((r.get("measurements") or {}).get("rank_displacement") or {})
    return (rd.get("objects") or {}).get("per_position")


def _cofit_pca(prompt_embs, probe_embs, n_components=2):
    """Co-fit PCA on combined prompt + probe embeddings."""
    combined = np.vstack([prompt_embs, probe_embs])
    mu = combined.mean(axis=0)
    centered = combined - mu

    C = np.cov(centered.T)
    if C.ndim == 0:
        C = np.array([[C]])
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:n_components]]
    eigvals = eigvals[idx]

    total = eigvals.sum()
    variance = [round(float(eigvals[i] / total * 100), 1)
                for i in range(min(n_components, len(eigvals)))]

    projected = centered @ eigvecs
    n_prompts = prompt_embs.shape[0]
    return projected[:n_prompts], projected[n_prompts:], variance


def _nearest_probe(token_emb, probe_mat, k=5, sharpness=10.0):
    """Find nearest probe index using cosine similarity with soft-kNN."""
    if token_emb is None or probe_mat is None:
        return 0, 0.0
    token_norm = np.linalg.norm(token_emb)
    if token_norm < NORM_EPS:
        return 0, 0.0
    sims = probe_mat @ (token_emb / token_norm)
    top_k = min(k, len(sims))
    top_idx = np.argsort(sims)[-top_k:][::-1]
    top_sims = sims[top_idx]

    # Soft-weighted selection
    weights = np.exp(sharpness * (top_sims - top_sims.max()))
    weights /= weights.sum() + NORM_EPS

    best_idx = top_idx[np.argmax(weights)]
    best_dist = float(sims[best_idx])
    return int(best_idx), best_dist


@register_analysis
class DomainSurface(AnalysisModule):
    name = "domain_surface"
    display_name = "Domain Surface"
    description = (
        "Maps per-token correction signals onto a probe-defined domain "
        "surface. Reveals how alignment training treats tokens across "
        "different topics and discourse frames."
    )
    version = "1.0.0"
    min_results = 10

    depends_on_measurements = ("per_token_embedding",)

    parameters = [
        ModuleParameter(name="top_tokens", display_name="Top Tokens",
                        description="Number of most-frequent tokens to include.",
                        kind="int", default=30, min_value=5, max_value=100),
        ModuleParameter(name="min_appearances", display_name="Min Appearances",
                        description="Minimum token appearances across prompts.",
                        kind="int", default=2, min_value=1, max_value=20),
        ModuleParameter(name="probe_neighbors", display_name="Probe Neighbors (k)",
                        description="Nearest probes for cell assignment.",
                        kind="int", default=5, min_value=1, max_value=20),
        ModuleParameter(name="knn_sharpness", display_name="kNN Sharpness",
                        description="Temperature for similarity-weighted matching.",
                        kind="float", default=10.0, min_value=1.0, max_value=50.0),
    ]

    def __init__(self):
        self._pipeline = None
        self._probe_store = None

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline

    def set_probe_store(self, probe_store):
        self._probe_store = probe_store

    def check_dependencies(self, session: dict) -> list[str]:
        errors = super().check_dependencies(session)
        if self._probe_store is None:
            errors.append("Domain Surface requires a probe store.")
        return errors

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []
        top_tokens_n = int(params.get("top_tokens", 30))
        min_appearances = int(params.get("min_appearances", 2))
        probe_neighbors = int(params.get("probe_neighbors", 5))
        knn_sharpness = float(params.get("knn_sharpness", 10.0))

        probe_set = self._get_active_probe_set()
        if probe_set is None:
            return {"error": "No active probe set."}

        subjects = list(dict.fromkeys(p.row for p in probe_set.probes))
        subclasses = list(dict.fromkeys(p.column for p in probe_set.probes))
        subj_idx = {s: i for i, s in enumerate(subjects)}
        level_idx = {l: i for i, l in enumerate(subclasses)}
        level_names = [l.replace("_", " ") for l in subclasses]

        # Get probe embeddings at subject depth
        depth_labels = probe_set.depth_labels
        subj_depth = depth_labels[0] if depth_labels else "subject"
        subj_mat, subj_labels = probe_set.embeddings_matrix(subj_depth)
        if subj_mat.shape[0] == 0:
            return {"error": "No probe embeddings at subject depth."}

        # Normalize probe matrix for cosine similarity
        norms = np.linalg.norm(subj_mat, axis=1, keepdims=True)
        norms[norms < NORM_EPS] = 1.0
        probe_mat_normed = subj_mat / norms

        # Build per-prompt mean embeddings for PCA
        prompt_embs = []
        valid_indices = []
        for pi, r in enumerate(prompts):
            fe = _get_final_emb(r)
            if fe is not None and len(fe) > 0:
                arr = np.array(fe, dtype=np.float32)
                prompt_embs.append(arr.mean(axis=0))
                valid_indices.append(pi)

        if len(prompt_embs) < self.min_results:
            return {"error": f"Only {len(prompt_embs)} prompts with embeddings "
                             f"(need {self.min_results})."}

        prompt_embs_arr = np.array(prompt_embs, dtype=np.float32)

        # Co-fit PCA
        prompt_coords, probe_coords, variance = _cofit_pca(
            prompt_embs_arr, subj_mat.astype(np.float32))

        # Build anchor points
        anchors_compact = []
        for i, p in enumerate(probe_set.probes):
            if i < probe_coords.shape[0]:
                anchors_compact.append({
                    "s": subj_idx.get(p.row, 0),
                    "l": level_idx.get(p.column, 0),
                    "t": p.label[:PROBE_TEXT_DISPLAY_LEN],
                    "x": round(float(probe_coords[i, 0]), 4),
                    "y": round(float(probe_coords[i, 1]), 4),
                })

        # Build per-token observations
        token_freq = Counter()
        raw_obs = []
        session_subset = [prompts[i] for i in valid_indices]

        for pi_local, sr in enumerate(session_subset):
            dx = float(prompt_coords[pi_local, 0])
            dy = float(prompt_coords[pi_local, 1])
            cat = ((sr.get("category") or "?")[:1]).lower()
            toks = sr.get("tokens") or []
            stress = _get_per_token_stress(sr) or []
            sfd_d = _get_per_token_density(sr) or []
            rd_pos = _get_rd_per_position(sr) or []
            fe = _get_final_emb(sr)

            for pos in range(len(toks)):
                tok = toks[pos].strip()
                if not tok:
                    continue
                token_freq[tok] += 1

                disp = 0.0
                repl = 0.0
                if pos < len(rd_pos) and rd_pos[pos]:
                    pd = rd_pos[pos]
                    disp = float(pd.get("total_disp", 0) if isinstance(pd, dict) else 0)
                    repl = float(pd.get("replacement_ratio", 0) if isinstance(pd, dict) else 0)

                asm = float(stress[pos]) if pos < len(stress) else 0
                density = float(sfd_d[pos]) if pos < len(sfd_d) else 0

                # Token embedding for probe matching
                tok_emb = None
                if fe is not None and pos < len(fe):
                    tok_emb = np.array(fe[pos], dtype=np.float32)

                raw_obs.append({
                    "tok": tok, "cat": cat,
                    "dx": dx, "dy": dy,
                    "disp": disp, "repl": repl,
                    "asm": asm, "sfd_d": density,
                    "pi": pi_local, "pos": pos,
                    "_emb": tok_emb,
                })

        # Select top tokens
        qualified = {t: n for t, n in token_freq.items() if n >= min_appearances}
        top_tokens = sorted(qualified.keys(), key=lambda t: -qualified[t])[:top_tokens_n]

        # Token CV
        token_cv = {}
        for tok in top_tokens:
            disps = [o["disp"] for o in raw_obs if o["tok"] == tok]
            if len(disps) >= 2:
                m = np.mean(disps)
                token_cv[tok] = round(float(np.std(disps) / max(abs(m), NORM_EPS)), 3)
            else:
                token_cv[tok] = 0
        ordered_tokens = sorted(top_tokens, key=lambda t: token_cv.get(t, 0))

        # Filter observations and compute probe proximity
        top_set = set(ordered_tokens)
        obs_export = []

        for o in raw_obs:
            if o["tok"] not in top_set:
                continue

            near_idx, near_dist = 0, 0.0
            near_level = 0
            near_subj_idx = 0
            near_angle = 0.0

            if o["_emb"] is not None and probe_mat_normed.shape[0] > 0:
                near_idx, near_dist = _nearest_probe(
                    o["_emb"], probe_mat_normed,
                    k=probe_neighbors, sharpness=knn_sharpness)
                if near_idx < len(probe_set.probes):
                    p = probe_set.probes[near_idx]
                    near_level = level_idx.get(p.column, 0)
                    near_subj_idx = subj_idx.get(p.row, 0)
                    if probe_coords.shape[0] > near_idx:
                        pc = probe_coords[near_idx]
                        near_angle = float(np.arctan2(pc[1], pc[0]))

            obs_export.append([
                o["tok"], o["cat"], o["dy"], o["disp"], o["repl"], o["dx"],
                round(o["asm"], 4), round(o["sfd_d"], 4),
                o["pi"], o["pos"],
                round(near_dist, 4), near_level, near_subj_idx,
                round(near_angle, 4),
            ])

        # Stratification
        strat = _stratification(obs_export, subjects)

        # Prompt texts
        prompt_texts = [(sr.get("prompt") or "")[:PROMPT_TEXT_DISPLAY_LEN]
                        for sr in session_subset]

        pca_components = [round(float(probe_coords[i, :].tolist()[0]), 4)
                          for i in range(min(2, probe_coords.shape[0]))]

        return {
            "pca": variance,
            "pca_components": [],
            "layer": "middle",
            "n_prompts_used": len(prompt_embs),
            "n_prompts_total": len(prompts),
            "min_appearances": min_appearances,
            "subjects": subjects,
            "tokens": ordered_tokens,
            "token_cv": token_cv,
            "anchors": anchors_compact,
            "observations": obs_export,
            "prompts": prompt_texts,
            "fields": [
                "tok", "cat", "dy", "disp", "repl", "dx",
                "asm", "sfd_d", "pi", "pos",
                "near_dist", "near_level", "near_subj_idx",
                "near_angle",
            ],
            "stratification": strat,
            "probe_file": probe_set.template_name,
            "level_names": level_names,
        }

    def _get_active_probe_set(self):
        if self._probe_store is None:
            return None
        sets = self._probe_store.list()
        if not sets:
            return None
        return self._probe_store.get_by_id(sets[-1]["set_id"])


def _stratification(obs, subjects):
    """Compute stratification statistics from observation tuples."""
    by_level = defaultdict(Counter)
    by_subject = defaultdict(Counter)

    for o in obs:
        cat = o[1]
        by_level[int(round(o[11]))][cat] += 1
        if o[12] < len(subjects):
            by_subject[subjects[o[12]]][cat] += 1

    return {
        "by_level": {str(k): dict(v) for k, v in sorted(by_level.items())},
        "by_subject": {k: dict(v) for k, v in sorted(by_subject.items())},
    }
