"""Domain Surface analysis (ported from TASM).

Co-embeds prompts and probe set via PCA on subject-depth embeddings,
produces per-token observations positioned in the (PC1, PC2) plane with
their displacement, attribution, density, and SFD metrics, and computes
stratification counts by nearest probe's level and subject.

Emits the JSON shape `renderDomainSurfaceResults` in static/js/main.js
reads:

  {
    "pca":              [pc1_pct, pc2_pct],
    "pca_components":   int,
    "layer":            "middle",
    "n_prompts_used":   int,
    "n_prompts_total":  int,
    "subjects":         [str, ...],
    "tokens":           [str, ...],
    "token_cv":         {token: cv, ...},
    "anchors":          [{s, l, t, x, y}, ...],
    "observations":     [[tok, cat, dy, disp, repl, dx, asm, sfd_d,
                          pi, pos, near_dist, near_level,
                          near_subj_idx, near_angle], ...],
    "prompts":          [str, ...],
    "fields":           [...],
    "stratification":   {by_level: {...}, by_subject: {...}},
    "probe_file":       str,
    "level_names":      [str, ...],
  }
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")

NORM_EPS = 1e-10
PROBE_TEXT_DISPLAY = 50
PROMPT_TEXT_DISPLAY = 80


def _cofit_pca(prompt_embs, probe_embs, n_components=2):
    """PCA on stacked prompt + probe embeddings via numpy."""
    all_e = np.vstack([prompt_embs, probe_embs])
    mu = all_e.mean(axis=0)
    Xc = all_e - mu
    cov = np.cov(Xc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    coords = Xc @ eigvecs[:, :n_components]
    total = float(eigvals.sum())
    if total > NORM_EPS:
        variance = [round(float(eigvals[i] / total) * 100, 1)
                    for i in range(n_components)]
    else:
        variance = [0.0] * n_components
    n_p = len(prompt_embs)
    return coords[:n_p], coords[n_p:], variance


def _nearest_probe(dx, dy, anchor_pts):
    best_dist = float("inf")
    best_idx = 0
    for i, a in enumerate(anchor_pts):
        d = math.sqrt((dx - a["x"]) ** 2 + (dy - a["y"]) ** 2)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist


@register_analysis
class DomainSurface(AnalysisModule):
    name = "domain_surface"
    display_name = "Domain Surface"
    description = (
        "Projects prompts and probes into a shared 2D PCA space "
        "(co-fit on their subject-depth embeddings) and builds per-"
        "token observations with displacement, attribution, SFD, and "
        "nearest-probe (subject, level) assignment. Shows how prompts "
        "distribute over the probe lattice and where each token lands."
    )
    version = "1.0.0"

    depends_on_measurements = ("per_token_embedding",)

    parameters = [
        ModuleParameter(
            name="top_tokens",
            display_name="Top N tokens",
            description=(
                "Number of most-frequent tokens to include in the "
                "observations and token table."
            ),
            kind="int", default=20, min_value=5, max_value=200,
        ),
        ModuleParameter(
            name="min_appearances",
            display_name="Min appearances per token",
            description="Skip tokens seen fewer times than this.",
            kind="int", default=2, min_value=1, max_value=50,
        ),
    ]

    def check_dependencies(self, session):
        prompts = session.get("prompts") or []
        if not prompts:
            return [f"Analysis '{self.name}' needs at least one prompt."]
        for p in prompts:
            pte = (p.get("measurements") or {}).get(
                "per_token_embedding") or {}
            if ((pte.get("objects") or {}).get(
                    "per_token_embeddings") or {}).get("subject"):
                return []
        return [
            f"Analysis '{self.name}' requires per-token subject-layer "
            f"embeddings. Run per_token_embedding with "
            f"include_in_export=True (the default) and retry."
        ]

    def run(self, session, params, probes=None, context=None):
        top_n = int(params.get("top_tokens", 20))
        min_app = int(params.get("min_appearances", 2))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"top_tokens": top_n, "min_appearances": min_app},
        )

        # ── Resolve active probe set ──
        ctx = context or {}
        probe_store = ctx.get("probe_store")
        tpl_info = ctx.get("active_probe_template") or {}
        if not probe_store or not tpl_info.get("set_id"):
            err = ("No active probe set. Apply one via the Configuration "
                   "tab before running domain_surface.")
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err))
            return result

        npz_path = probe_store.root / f"{tpl_info['set_id']}.npz"
        if not npz_path.exists():
            err = f"Probe set {tpl_info['set_id']} missing on disk."
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err))
            return result

        from tagm.probes.artifact import ProbeSet
        try:
            probe_set = ProbeSet.load(npz_path)
        except Exception as e:
            err = f"Could not load probe set: {e}"
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err))
            return result

        probe_mat, probe_labels = probe_set.embeddings_matrix("subject")
        if probe_mat.shape[0] == 0:
            err = "Probe set has no subject-depth embeddings."
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err))
            return result

        # Build subjects / levels
        subjects: list[str] = []
        level_names: list[str]
        if tpl_info.get("levels"):
            level_names = list(tpl_info["levels"])
        else:
            level_names = []
            for p in probe_set.probes:
                if p.column not in level_names:
                    level_names.append(p.column)

        probe_info: list[dict] = []
        for i, p in enumerate(probe_set.probes):
            if p.row not in subjects:
                subjects.append(p.row)
            try:
                level_idx = level_names.index(p.column)
            except ValueError:
                level_names.append(p.column)
                level_idx = len(level_names) - 1
            probe_info.append({
                "subject": p.row,
                "level": level_idx,
                "text": p.label,
            })

        subj_idx = {s: i for i, s in enumerate(subjects)}
        n_subj = len(subjects)
        subj_angles = (np.linspace(0, 2 * np.pi, n_subj, endpoint=False)
                        - np.pi / 2) if n_subj > 0 else np.array([])

        # ── Gather prompt mean-pooled subject-depth embeddings ──
        session_prompts = session.get("prompts") or []
        prompt_embs: list[np.ndarray] = []
        prompt_indices: list[int] = []
        for pi, p in enumerate(session_prompts):
            pte = (p.get("measurements") or {}).get(
                "per_token_embedding") or {}
            sub = ((pte.get("objects") or {}).get(
                "per_token_embeddings") or {}).get("subject")
            if not sub:
                continue
            try:
                arr = np.asarray(sub, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            pooled = arr.mean(axis=0)
            n = float(np.linalg.norm(pooled))
            if n > NORM_EPS:
                pooled = pooled / n
            if pooled.shape[0] != probe_mat.shape[1]:
                continue
            prompt_embs.append(pooled)
            prompt_indices.append(pi)

        if len(prompt_embs) < 2:
            err = (f"Only {len(prompt_embs)} prompts have usable "
                   f"subject-depth embeddings; need at least 2.")
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                 subjects=subjects,
                                                 level_names=level_names))
            return result

        prompt_mat = np.stack(prompt_embs)

        # ── Co-fit PCA ──
        prompt_coords, probe_coords, variance = _cofit_pca(
            prompt_mat, probe_mat.astype(np.float32), n_components=2)

        anchor_pts = []
        for i, pinfo in enumerate(probe_info):
            anchor_pts.append({
                "subject": pinfo["subject"],
                "level": pinfo["level"],
                "text": pinfo["text"][:PROBE_TEXT_DISPLAY],
                "x": float(probe_coords[i, 0]),
                "y": float(probe_coords[i, 1]),
            })

        # ── Build per-token raw observations ──
        token_freq: dict[str, int] = defaultdict(int)
        raw_obs: list[dict] = []
        session_subset = [session_prompts[i] for i in prompt_indices]

        for coord_i, orig_pi in enumerate(prompt_indices):
            p = session_prompts[orig_pi]
            dx = float(prompt_coords[coord_i, 0])
            dy = float(prompt_coords[coord_i, 1])
            cat_full = (p.get("category") or "")
            cat = (cat_full[:1] or "?").lower()
            tokens = p.get("tokens") or []
            meas = p.get("measurements") or {}

            # Per-token stress (asm)
            stress_per_tok = ((meas.get("stress_score") or {}).get(
                "per_token") or {}).get("stress") or []
            # Per-token density (sfd_d)
            density_per_tok = ((meas.get("spectral_field_density") or {}).get(
                "per_token") or {}).get("density") or []
            # Per-token displacement (disp) from rank_displacement.objects.
            # instruct_disp_profiles: list[pos] of list[float]; sum = total_disp.
            rd_objs = ((meas.get("rank_displacement") or {}).get(
                "objects") or {})
            idp = rd_objs.get("instruct_disp_profiles") or []
            bdp = rd_objs.get("base_disp_profiles") or []

            # Per-token subject/escalation embeddings (if export on)
            pte_objs = (meas.get("per_token_embedding") or {}).get(
                "objects") or {}
            ptde = (pte_objs.get("per_token_embeddings") or {}).get(
                "subject") or []
            ptee = (pte_objs.get("per_token_embeddings") or {}).get(
                "escalation") or []

            for pos in range(len(tokens)):
                tok = (tokens[pos] or "").strip()
                if not tok:
                    continue
                token_freq[tok] += 1

                i_prof = idp[pos] if pos < len(idp) else []
                b_prof = bdp[pos] if pos < len(bdp) else []
                total_disp = (float(sum(v for v in i_prof if v is not None))
                              if i_prof else 0.0)
                # "Replacement ratio" approximated as instruct/(instruct+base)
                i_mag = total_disp
                b_mag = (float(sum(v for v in b_prof if v is not None))
                         if b_prof else 0.0)
                repl_ratio = (i_mag / (i_mag + b_mag)
                              if (i_mag + b_mag) > NORM_EPS else 0.0)

                asm = (float(stress_per_tok[pos])
                       if pos < len(stress_per_tok)
                       and stress_per_tok[pos] is not None else 0.0)
                sfd_d = (float(density_per_tok[pos])
                         if pos < len(density_per_tok)
                         and density_per_tok[pos] is not None else 0.0)

                emb_sub = ptde[pos] if pos < len(ptde) else None
                emb_esc = ptee[pos] if pos < len(ptee) else None

                raw_obs.append({
                    "tok": tok, "cat": cat,
                    "dx": dx, "dy": dy,
                    "disp": total_disp,
                    "repl": repl_ratio,
                    "asm": asm,
                    "sfd_d": sfd_d,
                    "pi": coord_i, "pos": pos,
                    "_emb": emb_sub,
                    "_esc_emb": emb_esc,
                })

        # ── Top tokens by frequency ──
        qualified = {t: n for t, n in token_freq.items() if n >= min_app}
        top = sorted(qualified.keys(),
                      key=lambda t: -qualified[t])[:top_n]
        token_cv: dict[str, float] = {}
        for tok in top:
            disps = [o["disp"] for o in raw_obs if o["tok"] == tok]
            if len(disps) >= 2:
                m = float(np.mean(disps))
                token_cv[tok] = round(
                    float(np.std(disps) / max(abs(m), NORM_EPS)), 3)
            else:
                token_cv[tok] = 0.0
        ordered_tokens = sorted(top, key=lambda t: token_cv.get(t, 0.0))

        # ── Build observations with nearest-probe assignment ──
        top_set = set(ordered_tokens)
        obs_export: list[list] = []
        # Precompute probe matrix normalized for cosine
        probe_sub_mat = probe_mat.astype(np.float32)
        probe_sub_norms = np.linalg.norm(probe_sub_mat, axis=1, keepdims=True)
        probe_sub_norms = np.where(probe_sub_norms < NORM_EPS, 1.0,
                                    probe_sub_norms)
        probe_sub_n = probe_sub_mat / probe_sub_norms

        for o in raw_obs:
            if o["tok"] not in top_set:
                continue

            # Subject assignment
            subj_emb = o.get("_emb")
            near_angle = 0.0
            near_subj = 0
            best_dist = 0.0
            if subj_emb is not None and n_subj > 0:
                try:
                    tok_vec = np.asarray(subj_emb, dtype=np.float32)
                    tn = float(np.linalg.norm(tok_vec))
                    if tn > NORM_EPS:
                        tok_vec = tok_vec / tn
                    if tok_vec.shape[0] == probe_sub_n.shape[1]:
                        sims = probe_sub_n @ tok_vec
                        k = min(5, len(sims))
                        top_k = np.argsort(sims)[-k:][::-1]
                        top_sims = sims[top_k]
                        best_dist = float(1.0 - top_sims[0])
                        weights = np.exp(top_sims * 10.0)
                        ws = weights.sum()
                        if ws > 0:
                            weights = weights / ws
                        sin_sum = cos_sum = 0.0
                        subj_w: dict[int, float] = defaultdict(float)
                        for idx, w in zip(top_k, weights):
                            if idx < len(probe_info):
                                si = subj_idx.get(
                                    probe_info[idx]["subject"], 0)
                                sin_sum += float(w) * float(
                                    np.sin(subj_angles[si]))
                                cos_sum += float(w) * float(
                                    np.cos(subj_angles[si]))
                                subj_w[si] += float(w)
                        near_angle = float(math.atan2(sin_sum, cos_sum))
                        near_subj = (max(subj_w, key=subj_w.get)
                                      if subj_w else 0)
                    else:
                        aidx, best_dist = _nearest_probe(
                            o["dx"], o["dy"], anchor_pts)
                        near_subj = subj_idx.get(
                            anchor_pts[aidx]["subject"], 0)
                        near_angle = float(subj_angles[near_subj])
                except Exception:
                    aidx, best_dist = _nearest_probe(
                        o["dx"], o["dy"], anchor_pts)
                    near_subj = subj_idx.get(
                        anchor_pts[aidx]["subject"], 0)
                    near_angle = float(subj_angles[near_subj])
            else:
                if anchor_pts:
                    aidx, best_dist = _nearest_probe(
                        o["dx"], o["dy"], anchor_pts)
                    near_subj = subj_idx.get(
                        anchor_pts[aidx]["subject"], 0)
                    if n_subj > 0:
                        near_angle = float(subj_angles[near_subj])

            # Level assignment: nearest-anchor level (PCA fallback or exact)
            if anchor_pts:
                aidx, _ = _nearest_probe(o["dx"], o["dy"], anchor_pts)
                level = anchor_pts[aidx]["level"]
            else:
                level = 0

            obs_export.append([
                o["tok"], o["cat"],
                round(o["dy"], 4), round(o["disp"], 3),
                round(o["repl"], 2), round(o["dx"], 4),
                round(o["asm"], 2), round(o["sfd_d"], 3),
                o["pi"], o["pos"],
                round(best_dist, 4), level, near_subj,
                round(near_angle, 4),
            ])

        # ── Stratification: counts by nearest level / subject ──
        by_level: dict[int, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        by_subject: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        for row in obs_export:
            cat = row[1]
            by_level[int(round(row[11]))][cat] += 1
            if row[12] < n_subj:
                by_subject[subjects[row[12]]][cat] += 1

        stratification = {
            "by_level": {str(k): dict(v)
                          for k, v in sorted(by_level.items())},
            "by_subject": {k: dict(v)
                            for k, v in sorted(by_subject.items())},
        }

        anchors_compact = [{
            "s": subj_idx.get(a["subject"], 0),
            "l": a["level"],
            "t": a["text"],
            "x": round(a["x"], 4),
            "y": round(a["y"], 4),
        } for a in anchor_pts]

        prompts_text = [(p.get("prompt") or "")[:PROMPT_TEXT_DISPLAY]
                         for p in session_subset]

        output = {
            "version": self.version,
            "pca": variance,
            "pca_components": 2,
            "layer": "middle",
            "n_prompts_used": len(prompt_embs),
            "n_prompts_total": len(session_prompts),
            "min_appearances": min_app,
            "subjects": subjects,
            "tokens": ordered_tokens,
            "token_cv": token_cv,
            "anchors": anchors_compact,
            "observations": obs_export,
            "prompts": prompts_text,
            "fields": [
                "tok", "cat", "dy", "disp", "repl", "dx",
                "asm", "sfd_d", "pi", "pos",
                "near_dist", "near_level", "near_subj_idx",
                "near_angle",
            ],
            "stratification": stratification,
            "probe_file": (probe_set.template_name or "probe_set") + ".csv",
            "level_names": level_names,
        }

        result.objects.update(output)
        result.scalars["n_prompts_used"] = len(prompt_embs)
        result.scalars["n_prompts_total"] = len(session_prompts)
        result.scalars["n_tokens"] = len(ordered_tokens)
        result.scalars["n_observations"] = len(obs_export)
        return result


def _empty_output(error: str, subjects=None, level_names=None) -> dict:
    return {
        "error": error,
        "pca": [0.0, 0.0], "pca_components": 2, "layer": "middle",
        "n_prompts_used": 0, "n_prompts_total": 0,
        "subjects": subjects or [],
        "tokens": [], "token_cv": {},
        "anchors": [], "observations": [], "prompts": [],
        "fields": [
            "tok", "cat", "dy", "disp", "repl", "dx",
            "asm", "sfd_d", "pi", "pos",
            "near_dist", "near_level", "near_subj_idx", "near_angle",
        ],
        "stratification": {"by_level": {}, "by_subject": {}},
        "probe_file": "",
        "level_names": level_names or [],
    }
