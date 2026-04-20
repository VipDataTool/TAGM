"""CorrectionManifold analysis (ported from TASM).

Projects each prompt's final-layer activations through the probe
refinement lattice (escalation − subject deltas, same as
correction_heatmap). Each prompt becomes an n_classes × n_subclasses
"fingerprint". K-means clusters fingerprints; PCA reduces to 2D for
visualization. Clusters are compared against human category labels for
validation.

Emits the UI wire format `renderCorrectionManifoldResults` reads:
classes, subclasses, pca_explained, silhouette_score, cluster_mapping,
classification, prompts, clusters, n_cells, n_clusters, n_prompts.

Inputs identical to correction_heatmap:
  - ProbeSet with "subject" and "escalation" depth embeddings
  - per-prompt per_token_embedding["final"]
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


# ── Numpy k-means + silhouette + PCA (port of TASM helpers) ────

def _kmeans(X, k, max_iter=100, n_init=10, seed=42):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    best_labels, best_centroids, best_inertia = None, None, np.inf
    for _ in range(n_init):
        centroids = np.empty((k, d))
        idx = rng.randint(n)
        centroids[0] = X[idx]
        for ci in range(1, k):
            dists = np.min(
                [np.sum((X - centroids[j]) ** 2, axis=1)
                 for j in range(ci)], axis=0)
            s = dists.sum()
            if s <= 0:
                centroids[ci] = X[rng.randint(n)]
                continue
            probs = dists / s
            idx = rng.choice(n, p=probs)
            centroids[ci] = X[idx]
        for _ in range(max_iter):
            dists = np.stack(
                [np.sum((X - centroids[j]) ** 2, axis=1) for j in range(k)],
                axis=1)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.empty_like(centroids)
            for j in range(k):
                m = X[labels == j]
                new_centroids[j] = (m.mean(axis=0) if len(m)
                                     else X[rng.randint(n)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        inertia = sum(np.sum((X[labels == j] - centroids[j]) ** 2)
                      for j in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
    return best_labels, best_centroids, best_inertia


def _silhouette_score(X, labels):
    n = len(X)
    k = len(set(labels))
    if k < 2 or k >= n:
        return 0.0
    scores = np.zeros(n)
    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        if own_mask.sum() <= 1:
            scores[i] = 0.0
            continue
        a = np.mean(np.sqrt(np.sum((X[own_mask] - X[i]) ** 2, axis=1)))
        b = np.inf
        for j in range(k):
            if j == own:
                continue
            mj = labels == j
            if mj.sum() == 0:
                continue
            d = np.mean(np.sqrt(np.sum((X[mj] - X[i]) ** 2, axis=1)))
            if d < b:
                b = d
        scores[i] = ((b - a) / max(a, b)) if max(a, b) > 0 else 0.0
    return float(np.mean(scores))


def _pca_2d(X):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    coords = Xc @ eigvecs[:, :2]
    total = eigvals.sum()
    explained = eigvals[:2] / total if total > 0 else np.zeros(2)
    return coords, explained, eigvecs


def _cluster_label_accuracy(labels_pred, labels_true):
    """Greedy best-match confusion → accuracy + matches."""
    unique_clusters = sorted(set(labels_pred))
    unique_cats = sorted(set(labels_true))
    conf = defaultdict(Counter)
    for pred, true in zip(labels_pred, labels_true):
        conf[pred][true] += 1
    pairs = [(conf[cl][cat], cl, cat)
             for cl in unique_clusters for cat in unique_cats]
    pairs.sort(reverse=True)
    matches = []
    assigned_clusters: set = set()
    assigned_cats: set = set()
    for count, cl, cat in pairs:
        if cl in assigned_clusters or cat in assigned_cats:
            continue
        matches.append((cl, cat, count))
        assigned_clusters.add(cl)
        assigned_cats.add(cat)
    correct = sum(c for _, _, c in matches)
    total = len(labels_pred)
    return (correct / total if total else 0.0), matches


# ── Analysis module ───────────────────────────────────────────

@register_analysis
class CorrectionManifold(AnalysisModule):
    name = "correction_manifold"
    display_name = "Correction Manifold"
    description = (
        "Projects prompts through the probe delta lattice to produce "
        "correction field fingerprints. K-means discovers natural "
        "clusters; PCA reduces to 2D. Compares discovered clusters "
        "against human category labels."
    )
    version = "1.0.0"

    depends_on_measurements = ("per_token_embedding",)

    parameters = [
        ModuleParameter(
            name="n_clusters",
            display_name="Clusters (k)",
            description=(
                "Number of clusters for k-means. 0 = auto-select via "
                "silhouette score (tries k=2..8)."
            ),
            kind="int", default=0, min_value=0, max_value=20,
        ),
    ]

    def check_dependencies(self, session):
        prompts = session.get("prompts") or []
        if len(prompts) < 4:
            return [
                f"Analysis '{self.name}' needs at least 4 prompts "
                f"with per-token final embeddings; have {len(prompts)}."
            ]
        any_final = False
        for p in prompts:
            pte = (p.get("measurements") or {}).get("per_token_embedding") or {}
            if ((pte.get("objects") or {}).get("per_token_embeddings") or {}).get("final"):
                any_final = True
                break
        if not any_final:
            return [
                f"Analysis '{self.name}' requires per-token final "
                f"embeddings. Run per_token_embedding with "
                f"include_in_export=True and retry."
            ]
        return []

    def run(self, session, params, probes=None, context=None):
        n_clusters_param = int(params.get("n_clusters", 0))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"n_clusters": n_clusters_param},
        )

        ctx = context or {}
        probe_store = ctx.get("probe_store")
        tpl_info = ctx.get("active_probe_template") or {}
        if not probe_store or not tpl_info.get("set_id"):
            err = ("No active probe set. Apply one via the Configuration "
                   "tab before running correction_manifold.")
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

        subj_mat, subj_labels = probe_set.embeddings_matrix("subject")
        esc_mat, esc_labels = probe_set.embeddings_matrix("escalation")
        if subj_mat.shape[0] == 0 or esc_mat.shape[0] == 0:
            err = ("Active probe set is missing subject or escalation "
                   "embeddings; re-apply.")
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err))
            return result

        # Compute L2-normalized deltas (same as correction_heatmap)
        deltas = (esc_mat - subj_mat).astype(np.float32)
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        deltas_n = deltas / norms

        # ── Lattice structure ──
        classes: list[str] = []
        probe_cells: list[tuple[int, int]] = []

        if tpl_info.get("levels"):
            subclasses = list(tpl_info["levels"])
        else:
            subclasses = []
            for p in probe_set.probes:
                if p.column not in subclasses:
                    subclasses.append(p.column)

        for p in probe_set.probes:
            if p.row not in classes:
                classes.append(p.row)
            si = classes.index(p.row)
            try:
                li = subclasses.index(p.column)
            except ValueError:
                subclasses.append(p.column)
                li = len(subclasses) - 1
            probe_cells.append((si, li))

        n_classes = len(classes)
        n_subclasses = len(subclasses)
        n_cells = n_classes * n_subclasses

        # ── Build per-prompt fingerprints ──
        prompts = session.get("prompts") or []
        n_prompts_total = len(prompts)
        fingerprints = np.zeros((n_prompts_total, n_cells))
        valid_mask = np.zeros(n_prompts_total, dtype=bool)

        for pi, p in enumerate(prompts):
            pte = (p.get("measurements") or {}).get("per_token_embedding") or {}
            emb = ((pte.get("objects") or {}).get("per_token_embeddings")
                    or {}).get("final")
            if not emb:
                continue
            try:
                tok_mat = np.asarray(emb, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if tok_mat.ndim != 2 or tok_mat.shape[0] == 0:
                continue
            if tok_mat.shape[1] != deltas_n.shape[1]:
                continue

            projections = tok_mat @ deltas_n.T  # (n_tokens, n_probes)
            cell_grid = np.zeros((n_classes, n_subclasses))
            cell_counts = np.zeros((n_classes, n_subclasses))
            for probe_idx, (si, li) in enumerate(probe_cells):
                cell_grid[si, li] += np.abs(projections[:, probe_idx]).mean()
                cell_counts[si, li] += 1
            mask = cell_counts > 0
            cell_grid[mask] /= cell_counts[mask]
            fingerprints[pi] = cell_grid.flatten()
            valid_mask[pi] = True

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 4:
            err = (f"Only {len(valid_indices)} prompts have per-token "
                   f"final embeddings; need at least 4.")
            result.warnings.append(err)
            result.objects.update(_empty_output(error=err,
                                                 classes=classes,
                                                 subclasses=subclasses,
                                                 n_cells=n_cells))
            return result

        X = fingerprints[valid_indices]
        valid_prompts = [prompts[i] for i in valid_indices]

        # ── PCA ──
        coords, explained, eigvecs = _pca_2d(X)
        cmax = float(np.abs(coords).max()) if coords.size else 0.0
        if cmax > 0:
            coords = coords / cmax * 0.85

        # ── K-means ──
        if n_clusters_param == 0:
            best_k, best_sil = 2, -1.0
            for k in range(2, min(9, len(X))):
                lbl_k, _, _ = _kmeans(X, k)
                sil = _silhouette_score(X, lbl_k)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            n_clusters = best_k
        else:
            n_clusters = min(n_clusters_param, max(2, len(X) - 1))

        labels, centroids, _inertia = _kmeans(X, n_clusters)
        sil_score = _silhouette_score(X, labels)

        # Project centroids to PCA plane
        X_mean = X.mean(axis=0)
        centroid_coords = (centroids - X_mean) @ eigvecs[:, :2]
        if cmax > 0:
            centroid_coords = centroid_coords / cmax * 0.85

        # ── Categories / cluster comparison ──
        categories: list[str] = []
        cat_full: list[str] = []
        for p in valid_prompts:
            cat = (p.get("category") or "unknown")
            cat_full.append(cat)
            if cat == "dual-use":
                categories.append("d")
            elif cat:
                categories.append(cat[0])
            else:
                categories.append("?")

        accuracy, cluster_mapping = _cluster_label_accuracy(
            labels.tolist(), categories)

        safe_cats = {"b", "m"}
        binary_true = ["safe" if c in safe_cats else "risk"
                        for c in categories]
        cluster_binary: dict[int, str] = {}
        for cl in range(n_clusters):
            cl_binary = [binary_true[i] for i, l in enumerate(labels)
                          if l == cl]
            cluster_binary[cl] = (Counter(cl_binary).most_common(1)[0][0]
                                   if cl_binary else "safe")
        binary_pred = [cluster_binary[int(l)] for l in labels]
        correct = sum(1 for p, t in zip(binary_pred, binary_true) if p == t)
        binary_accuracy = correct / len(binary_true) if binary_true else 0.0

        cat_centroids: dict[str, dict] = {}
        for cat in sorted(set(categories)):
            if cat == "?":
                continue
            idxs = [i for i, c in enumerate(categories) if c == cat]
            if idxs:
                cc = coords[idxs]
                cat_centroids[cat] = {
                    "x": round(float(cc[:, 0].mean()), 4),
                    "y": round(float(cc[:, 1].mean()), 4),
                    "n": len(idxs),
                }

        prompts_viz = []
        for i in range(len(valid_indices)):
            p = valid_prompts[i]
            prompts_viz.append({
                "prompt": (p.get("prompt") or "")[:80],
                "category": categories[i],
                "category_full": cat_full[i],
                "cluster": int(labels[i]),
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "fingerprint": [round(float(v), 4) for v in X[i]],
            })

        clusters_viz = []
        for j in range(n_clusters):
            cl_cats = [categories[i] for i in range(len(labels))
                        if labels[i] == j]
            cat_dist = dict(Counter(cl_cats))
            clusters_viz.append({
                "id": j,
                "cx": round(float(centroid_coords[j, 0]), 4),
                "cy": round(float(centroid_coords[j, 1]), 4),
                "size": int((labels == j).sum()),
                "category_distribution": cat_dist,
                "majority_category": (Counter(cl_cats).most_common(1)[0][0]
                                        if cl_cats else "?"),
            })

        output = {
            "version": self.version,
            "n_prompts": int(len(valid_indices)),
            "n_cells": n_cells,
            "n_clusters": int(n_clusters),
            "classes": classes,
            "subclasses": subclasses,
            "class_short": [s.replace("_", " ").title()[:14] for s in classes],
            "pca_explained": [round(float(e), 4) for e in explained],
            "silhouette_score": round(sil_score, 4),
            "cluster_mapping": [
                {"cluster": int(cl), "category": cat, "overlap": int(cnt)}
                for cl, cat, cnt in cluster_mapping
            ],
            "classification": {
                "overall_accuracy": round(accuracy, 4),
                "binary": {
                    "best_accuracy": round(binary_accuracy, 4),
                    "n_safe": binary_true.count("safe"),
                    "n_risk": binary_true.count("risk"),
                },
            },
            "prompts": prompts_viz,
            "clusters": clusters_viz,
            "category_centroids": cat_centroids,
        }

        result.objects.update(output)
        result.scalars["n_prompts"] = int(len(valid_indices))
        result.scalars["n_clusters"] = int(n_clusters)
        result.scalars["silhouette_score"] = round(sil_score, 4)
        return result


def _empty_output(error: str, classes=None, subclasses=None, n_cells=0) -> dict:
    return {
        "error": error,
        "version": CorrectionManifold.version,
        "n_prompts": 0, "n_cells": n_cells, "n_clusters": 0,
        "classes": classes or [],
        "subclasses": subclasses or [],
        "class_short": [],
        "pca_explained": [0.0, 0.0],
        "silhouette_score": 0.0,
        "cluster_mapping": [],
        "classification": {
            "overall_accuracy": 0.0,
            "binary": {"best_accuracy": 0.0, "n_safe": 0, "n_risk": 0},
        },
        "prompts": [], "clusters": [], "category_centroids": {},
    }
