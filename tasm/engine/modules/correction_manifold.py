"""
Correction Manifold Module for TASM.

Projects each prompt's final-layer activations through the probe delta
lattice (same projection as the Correction Heatmap) to produce a
high-dimensional fingerprint — one energy value per class × subclass
cell.  Then:

  1. PCA reduces the fingerprint vectors to 2D for visualization
  2. K-means discovers natural clusters from the correction field geometry
  3. Clusters are compared against human category labels for validation

The manifold and heatmap are two views of the same projection:
the heatmap is the tabular view (energy per cell), the manifold is
the spatial view (prompts positioned by their fingerprint similarity).

Prompts that interact with the correction field the same way end up
near each other.  Structure emerges from the data, not from an
imposed layout.
"""

import os
import csv
import json
import logging
import numpy as np
from collections import defaultdict, Counter

from .base import TASMModule, ModuleParameter
from .domain_surface import (_detect_level_cols, _parse_meta,
                              _load_probe_cache, _probe_cache_path,
                              _load_probes)
from .correction_heatmap import _get_active_probe

logger = logging.getLogger("tasm")


# ─── Numpy-only K-Means ──────────────────────────────────────

def _kmeans(X, k, max_iter=100, n_init=10, seed=42):
    """K-means clustering. Returns (labels, centroids, inertia)."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    best_labels = None
    best_centroids = None
    best_inertia = np.inf

    for _ in range(n_init):
        # K-means++ initialization
        centroids = np.empty((k, d))
        idx = rng.randint(n)
        centroids[0] = X[idx]
        for ci in range(1, k):
            dists = np.min([np.sum((X - centroids[j]) ** 2, axis=1)
                           for j in range(ci)], axis=0)
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centroids[ci] = X[idx]

        # Iterate
        for _ in range(max_iter):
            dists = np.stack([np.sum((X - centroids[j]) ** 2, axis=1)
                             for j in range(k)], axis=1)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.empty_like(centroids)
            for j in range(k):
                members = X[labels == j]
                if len(members) == 0:
                    new_centroids[j] = X[rng.randint(n)]
                else:
                    new_centroids[j] = members.mean(axis=0)
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
    """Mean silhouette score (numpy-only)."""
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
        # Mean intra-cluster distance
        a = np.mean(np.sqrt(np.sum((X[own_mask] - X[i]) ** 2, axis=1)))
        # Mean nearest-cluster distance
        b = np.inf
        for j in range(k):
            if j == own:
                continue
            mask_j = labels == j
            if mask_j.sum() == 0:
                continue
            d = np.mean(np.sqrt(np.sum((X[mask_j] - X[i]) ** 2, axis=1)))
            if d < b:
                b = d
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(np.mean(scores))


# ─── PCA (numpy-only) ────────────────────────────────────────

def _pca_2d(X):
    """Project X to 2D via PCA. Returns (coords_2d, explained_variance_ratio)."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    coords = X_centered @ eigenvectors[:, :2]
    total_var = eigenvalues.sum()
    explained = eigenvalues[:2] / total_var if total_var > 0 else np.zeros(2)
    return coords, explained


# ─── Cluster-Label Matching ───────────────────────────────────

def _cluster_label_accuracy(labels_pred, labels_true):
    """Best-case accuracy matching clusters to categories via Hungarian-style greedy."""
    unique_clusters = sorted(set(labels_pred))
    unique_cats = sorted(set(labels_true))

    # Build confusion matrix
    conf = defaultdict(Counter)
    for pred, true in zip(labels_pred, labels_true):
        conf[pred][true] += 1

    # Greedy assignment: largest overlap first
    assigned_cats = set()
    assigned_clusters = set()
    matches = []

    pairs = []
    for cl in unique_clusters:
        for cat in unique_cats:
            pairs.append((conf[cl][cat], cl, cat))
    pairs.sort(reverse=True)

    for count, cl, cat in pairs:
        if cl in assigned_clusters or cat in assigned_cats:
            continue
        matches.append((cl, cat, count))
        assigned_clusters.add(cl)
        assigned_cats.add(cat)

    correct = sum(c for _, _, c in matches)
    total = len(labels_pred)
    return correct / total if total > 0 else 0.0, matches


# ─── Module Class ─────────────────────────────────────────────

class CorrectionManifoldModule(TASMModule):
    """Correction manifold — the witness plate.

    Projects prompts through the probe delta lattice to produce
    per-prompt fingerprints, clusters them, and reduces to 2D.
    """

    name = "correction_manifold"
    display_name = "Correction Manifold"
    description = (
        "Projects prompts through the probe delta lattice to produce "
        "correction field fingerprints. K-means discovers natural "
        "clusters; PCA reduces to 2D. Compares discovered clusters "
        "against human category labels."
    )
    version = "0.3.0"

    min_results = 10
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    def __init__(self):
        super().__init__()
        self._project_root = None

    def set_project_root(self, root):
        self._project_root = root

    def set_session_dir(self, path):
        self._session_dir = path

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="n_clusters",
                display_name="Clusters (k)",
                description=(
                    "Number of clusters for k-means. Set to 0 for auto "
                    "selection via silhouette score (tries k=2..8)."
                ),
                type="int",
                default=0,
                min_val=0,
                max_val=20,
            ),
        ]

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        probe_file = _get_active_probe(self._project_root)
        if not probe_file:
            return False, (
                "No probe set active. Apply a probe set in the "
                "Configuration tab first."
            )

        has_final = any(r.get("per_token_final_emb") for r in session_results)
        if not has_final:
            return False, (
                "No final-layer token embeddings found in session results. "
                "Re-analyze prompts with the current build to capture them."
            )
        return True, "OK"

    def run(self, session_results, params, progress=None):
        probe_file = _get_active_probe(self._project_root)
        n_clusters = int(params.get("n_clusters", 0))

        if progress:
            progress("Loading probe structure...")

        # ── Load probe CSV ──
        csv_path = os.path.join(self._project_root, probe_file)
        level_cols, level_names = _detect_level_cols(csv_path)
        if not level_cols:
            raise RuntimeError(f"No subclass columns found in {probe_file}")

        raw_probes = _load_probes(csv_path)
        if not raw_probes:
            raise RuntimeError(f"No probes loaded from {probe_file}")

        classes = []
        for p in raw_probes:
            if p["subject"] not in classes:
                classes.append(p["subject"])
        n_classes = len(classes)
        n_subclasses = len(level_cols)
        n_cells = n_classes * n_subclasses
        class_idx = {s: i for i, s in enumerate(classes)}

        # ── Resolve layer depths (template meta overrides global config) ──
        meta = _parse_meta(csv_path)

        if "layer_low" in meta and "layer_high" in meta:
            subj_frac = max(0, min(1, float(meta["layer_low"])))
            esc_frac = max(0, min(1, float(meta["layer_high"])))
            logger.info(f"[MANIFOLD] Using template depths: "
                        f"L{int(subj_frac*100)}, L{int(esc_frac*100)}")
        else:
            try:
                from engine import engine_config
                subj_frac = max(0, min(1, engine_config.get("domain_embedding_layer_frac") or 0.50))
                esc_frac = max(0, min(1, engine_config.get("domain_escalation_layer_frac") or 0.75))
            except Exception:
                subj_frac = 0.50
                esc_frac = 0.75

        if progress:
            progress(f"Loading probe embeddings at L{int(subj_frac*100)} and L{int(esc_frac*100)}...")

        cache_dir = os.path.join(self._project_root, "probe_cache")
        stem = os.path.splitext(probe_file)[0]

        # Determine session embedding dimension for cache validation
        session_dim = None
        for r in session_results:
            fe = r.get("per_token_final_emb")
            if fe and len(fe) > 0:
                session_dim = len(fe[0])
                break

        def _find_cache(frac):
            if os.path.isdir(cache_dir):
                tag = f"__L{int(frac * 100)}"
                for fn in sorted(os.listdir(cache_dir)):
                    if fn.startswith(stem) and tag in fn and fn.endswith(".json"):
                        data = _load_probe_cache(os.path.join(cache_dir, fn))
                        if data and data.get("embeddings"):
                            embs = data["embeddings"]
                            # Dimension validation: skip caches from a different model
                            if session_dim is not None and len(embs) > 0:
                                cache_dim = len(embs[0])
                                if cache_dim != session_dim:
                                    cache_model = data.get("model_id", "unknown")
                                    logger.warning(
                                        f"[MANIFOLD] Probe cache dimension mismatch: "
                                        f"cache={cache_dim} (model={cache_model}), "
                                        f"session={session_dim}. Skipping {fn}.")
                                    continue
                            logger.info(f"[MANIFOLD] Using cache: {fn}")
                            return embs
            return None

        embs_L50 = _find_cache(subj_frac)
        embs_L75 = _find_cache(esc_frac)

        if embs_L50 is None:
            raise RuntimeError(
                f"No probe cache at L{int(subj_frac*100)} matches the current model "
                f"(hidden_dim={session_dim}). Apply the probe set with the current "
                f"model loaded to regenerate caches.")
        if embs_L75 is None:
            raise RuntimeError(
                f"No probe cache at L{int(esc_frac*100)} matches the current model "
                f"(hidden_dim={session_dim}). Apply the probe set with the current "
                f"model loaded to regenerate caches.")

        if len(embs_L50) != len(raw_probes) or len(embs_L75) != len(raw_probes):
            raise RuntimeError("Probe count mismatch. Regenerate caches.")

        # ── Compute probe deltas ──
        if progress:
            progress("Computing probe refinement deltas...")

        mat_L50 = np.array(embs_L50, dtype=np.float32)
        mat_L75 = np.array(embs_L75, dtype=np.float32)
        deltas = mat_L75 - mat_L50

        delta_norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        delta_norms[delta_norms < 1e-12] = 1.0
        deltas_n = deltas / delta_norms

        # Map probes to cells
        probe_cells = []
        for p in raw_probes:
            si = class_idx.get(p["subject"], 0)
            li = p["level"]
            probe_cells.append((si, li))

        # ── Project prompts through probe deltas → fingerprints ──
        if progress:
            progress("Computing per-prompt correction fingerprints...")

        n_prompts = len(session_results)
        fingerprints = np.zeros((n_prompts, n_cells))
        valid_mask = np.ones(n_prompts, dtype=bool)

        for pi, r in enumerate(session_results):
            final_emb = r.get("per_token_final_emb")
            if final_emb is None or len(final_emb) == 0:
                valid_mask[pi] = False
                continue

            if progress and (pi + 1) % 10 == 0:
                progress(f"Projecting prompt {pi+1}/{n_prompts}...")

            tok_mat = np.array(final_emb, dtype=np.float32)
            projections = tok_mat @ deltas_n.T  # [n_tokens, n_probes]

            # Aggregate per cell
            cell_grid = np.zeros((n_classes, n_subclasses))
            cell_counts = np.zeros((n_classes, n_subclasses))

            for probe_idx, (si, li) in enumerate(probe_cells):
                cell_grid[si, li] += np.abs(projections[:, probe_idx]).mean()
                cell_counts[si, li] += 1

            mask = cell_counts > 0
            cell_grid[mask] /= cell_counts[mask]

            fingerprints[pi] = cell_grid.flatten()

        # Filter out prompts without embeddings
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 4:
            raise RuntimeError(
                f"Only {len(valid_indices)} prompts have final embeddings. "
                "Need at least 4."
            )

        X = fingerprints[valid_indices]
        valid_results = [session_results[i] for i in valid_indices]

        # ── PCA to 2D ──
        if progress:
            progress("PCA reduction to 2D...")

        coords, explained = _pca_2d(X)

        # Normalize coords to [-1, 1] for visualization
        cmax = np.abs(coords).max()
        if cmax > 0:
            coords = coords / cmax * 0.85

        # ── K-means clustering ──
        if progress:
            progress("Clustering...")

        if n_clusters == 0:
            # Auto-select k via silhouette score
            best_k = 2
            best_sil = -1
            for k in range(2, min(9, len(X))):
                labels_k, _, _ = _kmeans(X, k)
                sil = _silhouette_score(X, labels_k)
                if progress:
                    progress(f"k={k}: silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            n_clusters = best_k
            logger.info(f"[MANIFOLD] Auto-selected k={best_k} "
                        f"(silhouette={best_sil:.3f})")

        labels, centroids, inertia = _kmeans(X, n_clusters)
        sil_score = _silhouette_score(X, labels)

        # Project centroids to 2D
        X_mean = fingerprints[valid_indices].mean(axis=0)
        cov = np.cov(fingerprints[valid_indices] - X_mean, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx_sort]
        centroid_coords = (centroids - X_mean) @ eigenvectors[:, :2]
        if cmax > 0:
            centroid_coords = centroid_coords / cmax * 0.85

        # ── Compare clusters to human labels ──
        categories = []
        cat_full = []
        for r in valid_results:
            cat = r.get("category", "unknown")
            cat_full.append(cat)
            if cat == "dual-use":
                categories.append("d")
            elif cat:
                categories.append(cat[0])
            else:
                categories.append("?")

        accuracy, cluster_mapping = _cluster_label_accuracy(
            labels.tolist(), categories)

        # Binary accuracy (safe vs risk)
        safe_cats = {"b", "m"}
        binary_true = ["safe" if c in safe_cats else "risk" for c in categories]
        binary_pred_raw = []
        # Map each cluster to majority binary label
        cluster_binary = {}
        for cl in range(n_clusters):
            cl_mask = labels == cl
            cl_binary = [binary_true[i] for i in range(len(labels)) if cl_mask[i]]
            if cl_binary:
                cluster_binary[cl] = Counter(cl_binary).most_common(1)[0][0]
            else:
                cluster_binary[cl] = "safe"
        binary_pred = [cluster_binary[int(l)] for l in labels]
        binary_correct = sum(1 for p, t in zip(binary_pred, binary_true) if p == t)
        binary_accuracy = binary_correct / len(binary_true) if binary_true else 0

        # ── Category centroids ──
        cat_centroids = {}
        for cat in sorted(set(categories)):
            if cat == "?":
                continue
            cat_mask = [i for i, c in enumerate(categories) if c == cat]
            if cat_mask:
                cat_coords = coords[cat_mask]
                cat_centroids[cat] = {
                    "x": round(float(cat_coords[:, 0].mean()), 4),
                    "y": round(float(cat_coords[:, 1].mean()), 4),
                    "n": len(cat_mask),
                }

        # ── Build visualization data ──
        if progress:
            progress("Building visualization data...")

        prompts_viz = []
        for i in range(len(valid_indices)):
            r = valid_results[i]
            prompts_viz.append({
                "prompt": r.get("prompt", "")[:80],
                "category": categories[i],
                "category_full": cat_full[i],
                "cluster": int(labels[i]),
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "fingerprint": [round(float(v), 4) for v in X[i]],
            })

        clusters_viz = []
        for j in range(n_clusters):
            cl_cats = [categories[i] for i in range(len(labels)) if labels[i] == j]
            cat_dist = dict(Counter(cl_cats))
            clusters_viz.append({
                "id": j,
                "cx": round(float(centroid_coords[j, 0]), 4),
                "cy": round(float(centroid_coords[j, 1]), 4),
                "size": int((labels == j).sum()),
                "category_distribution": cat_dist,
                "majority_category": Counter(cl_cats).most_common(1)[0][0] if cl_cats else "?",
            })

        # ── Build output ──
        output = {
            "version": self.version,
            "n_prompts": len(valid_indices),
            "n_cells": n_cells,
            "n_clusters": n_clusters,

            # Lattice structure
            "classes": classes,
            "subclasses": level_names,
            "class_short": [s.replace("_", " ").title()[:14] for s in classes],

            # PCA info
            "pca_explained": [round(float(e), 4) for e in explained],

            # Clustering
            "silhouette_score": round(sil_score, 4),
            "cluster_mapping": [
                {"cluster": cl, "category": cat, "overlap": cnt}
                for cl, cat, cnt in cluster_mapping
            ],

            # Classification
            "classification": {
                "overall_accuracy": round(accuracy, 4),
                "binary": {
                    "best_accuracy": round(binary_accuracy, 4),
                    "n_safe": binary_true.count("safe"),
                    "n_risk": binary_true.count("risk"),
                },
            },

            # Viz data
            "prompts": prompts_viz,
            "clusters": clusters_viz,
            "category_centroids": cat_centroids,
        }

        if progress:
            progress(f"Complete: {len(valid_indices)} prompts → "
                     f"{n_clusters} clusters, "
                     f"accuracy={accuracy:.1%}, "
                     f"silhouette={sil_score:.3f}")

        return output
