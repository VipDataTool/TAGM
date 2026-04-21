"""Correction Manifold: fingerprint-based prompt clustering.

Ported from TASM's engine/modules/correction_manifold.py. Projects each
prompt through probe delta lattice to produce a fingerprint, then PCA + 
k-means for spatial clustering. Output shape matches TASM renderer.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


def _get_final_emb(r):
    pte = ((r.get("measurements") or {}).get("per_token_embedding") or {})
    return (pte.get("objects") or {}).get("per_token_embeddings", {}).get("final")


def _kmeans(X, k, max_iter=100, n_init=10, seed=42):
    """K-means clustering. Returns (labels, centroids, inertia)."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    best_labels = None
    best_centroids = None
    best_inertia = np.inf

    for _ in range(n_init):
        centroids = np.empty((k, d))
        centroids[0] = X[rng.randint(n)]
        for ci in range(1, k):
            dists = np.min([np.sum((X - centroids[j])**2, axis=1)
                            for j in range(ci)], axis=0)
            probs = dists / (dists.sum() + 1e-15)
            centroids[ci] = X[rng.choice(n, p=probs)]

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
            new_labels = np.argmin(dists, axis=0)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    centroids[ci] = X[mask].mean(axis=0)

        inertia = sum(np.sum((X[labels == ci] - centroids[ci])**2)
                       for ci in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids, best_inertia


def _silhouette(X, labels):
    """Mean silhouette score."""
    n = len(labels)
    if n < 2:
        return 0.0
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            scores.append(0.0)
            continue
        a = np.mean(np.sqrt(np.sum((X[same] - X[i])**2, axis=1)))
        b_vals = []
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if other.sum() > 0:
                b_vals.append(np.mean(np.sqrt(np.sum((X[other] - X[i])**2, axis=1))))
        b = min(b_vals) if b_vals else a
        scores.append((b - a) / max(a, b, 1e-10))
    return float(np.mean(scores))


@register_analysis
class CorrectionManifold(AnalysisModule):
    name = "correction_manifold"
    display_name = "Correction Manifold"
    description = (
        "Projects prompt tokens through probe delta lattice to produce "
        "per-prompt fingerprints, then PCA + k-means clustering."
    )
    version = "1.0.0"
    min_results = 5

    depends_on_measurements = ("per_token_embedding",)

    parameters = [
        ModuleParameter(name="n_clusters", display_name="Clusters",
                        description="Number of k-means clusters.",
                        kind="int", default=4, min_value=2, max_value=20),
        ModuleParameter(name="random_seed", display_name="Random Seed",
                        description="Seed for k-means.",
                        kind="int", default=42, min_value=0, max_value=99999),
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
            errors.append("Correction Manifold requires a probe store.")
        return errors

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []
        n_clusters = int(params.get("n_clusters", 4))
        seed = int(params.get("random_seed", 42))

        probe_set = self._get_active_probe_set()
        if probe_set is None:
            return {"error": "No active probe set."}

        classes = list(dict.fromkeys(p.row for p in probe_set.probes))
        subclasses = list(dict.fromkeys(p.column for p in probe_set.probes))
        n_cells = len(classes) * len(subclasses)
        subj_idx = {s: i for i, s in enumerate(classes)}
        level_idx = {l: i for i, l in enumerate(subclasses)}

        depth_labels = probe_set.depth_labels
        if len(depth_labels) < 2:
            return {"error": "Need at least 2 probe depths."}

        subj_mat, _ = probe_set.embeddings_matrix(depth_labels[0])
        esc_mat, _ = probe_set.embeddings_matrix(depth_labels[1])
        delta_mat = esc_mat - subj_mat

        # Build per-prompt fingerprints
        fingerprints = []
        valid_indices = []
        categories = []

        for pi, r in enumerate(prompts):
            fe = _get_final_emb(r)
            if fe is None or len(fe) == 0:
                continue
            emb = np.array(fe, dtype=np.float32)
            tokens = r.get("tokens", [])
            n_tok = min(len(tokens), emb.shape[0])
            if n_tok == 0 or emb.shape[1] != delta_mat.shape[1]:
                continue

            scores = emb[:n_tok] @ delta_mat.T
            grid = np.zeros(n_cells)
            for ci, probe in enumerate(probe_set.probes):
                si = subj_idx.get(probe.row, 0)
                li = level_idx.get(probe.column, 0)
                cell_idx = si * len(subclasses) + li
                if cell_idx < n_cells:
                    grid[cell_idx] += float(np.mean(np.abs(scores[:, ci])))

            fingerprints.append(grid)
            valid_indices.append(pi)
            categories.append((r.get("category") or "unknown").lower().strip())

        if len(fingerprints) < 3:
            return {"error": f"Need at least 3 prompts with embeddings. Have {len(fingerprints)}."}

        X = np.array(fingerprints, dtype=np.float64)

        # Z-score normalize
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-10] = 1.0
        Z = (X - mu) / sd

        # PCA to 2D
        C = np.cov(Z.T)
        if C.ndim == 0:
            C = np.array([[C]])
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        total = eigvals.sum()
        explained = eigvals[:2] / total if total > 1e-10 else np.zeros(2)
        proj_2d = Z @ eigvecs[:, :2]

        # K-means
        k = min(n_clusters, len(fingerprints))
        labels, centroids, _ = _kmeans(Z, k, seed=seed)
        sil_score = _silhouette(Z, labels)

        # Cluster-to-category mapping
        cluster_mapping = []
        for cl in range(k):
            cl_cats = [categories[i] for i in range(len(categories)) if labels[i] == cl]
            if cl_cats:
                most_common = Counter(cl_cats).most_common(1)[0]
                cluster_mapping.append((cl, most_common[0], most_common[1]))

        # Classification accuracy
        cat_to_cluster = {}
        for cl, cat, cnt in cluster_mapping:
            if cat not in cat_to_cluster:
                cat_to_cluster[cat] = cl
        correct = sum(1 for i, cat in enumerate(categories)
                      if cat_to_cluster.get(cat) == labels[i])
        accuracy = correct / len(categories) if categories else 0

        # Binary accuracy
        safe_cats = {"benign", "mild"}
        binary_true = ["safe" if c in safe_cats else "risk" for c in categories]
        safe_cluster = None
        for cl, cat, cnt in cluster_mapping:
            if cat in safe_cats:
                safe_cluster = cl
                break
        binary_pred = ["safe" if labels[i] == safe_cluster else "risk"
                       for i in range(len(labels))]
        binary_accuracy = sum(1 for a, b in zip(binary_true, binary_pred) if a == b) / max(len(binary_true), 1)

        # Viz data
        prompts_viz = []
        for i in range(len(valid_indices)):
            prompts_viz.append({
                "idx": int(valid_indices[i]),
                "x": round(float(proj_2d[i, 0]), 4),
                "y": round(float(proj_2d[i, 1]), 4),
                "cluster": int(labels[i]),
                "category": categories[i],
                "prompt": (prompts[valid_indices[i]].get("prompt") or "")[:80],
            })

        centroids_2d = centroids @ eigvecs[:, :2]
        clusters_viz = [
            {"cluster": cl, "x": round(float(centroids_2d[cl, 0]), 4),
             "y": round(float(centroids_2d[cl, 1]), 4),
             "n_prompts": int((labels == cl).sum())}
            for cl in range(k)
        ]

        cat_centroids = {}
        for cat in set(categories):
            mask = np.array([c == cat for c in categories])
            if mask.sum() > 0:
                cat_centroids[cat] = {
                    "x": round(float(proj_2d[mask, 0].mean()), 4),
                    "y": round(float(proj_2d[mask, 1].mean()), 4),
                    "n": int(mask.sum()),
                }

        level_names = [l.replace("_", " ") for l in subclasses]

        return {
            "version": self.version,
            "n_prompts": len(valid_indices),
            "n_cells": n_cells,
            "n_clusters": k,
            "classes": classes,
            "subclasses": level_names,
            "class_short": [s.replace("_", " ").title()[:14] for s in classes],
            "pca_explained": [round(float(e), 4) for e in explained],
            "silhouette_score": round(sil_score, 4),
            "cluster_mapping": [
                {"cluster": cl, "category": cat, "overlap": cnt}
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

    def _get_active_probe_set(self):
        if self._probe_store is None:
            return None
        sets = self._probe_store.list()
        if not sets:
            return None
        return self._probe_store.get_by_id(sets[-1]["set_id"])
