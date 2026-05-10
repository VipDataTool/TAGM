"""CorrectionManifold: cluster prompts by correction-signature similarity.

Builds a per-prompt signature vector from the scalar outputs of each
measurement, z-scores it across the session, computes a full distance
matrix (cosine or Euclidean), then runs a simple agglomerative clustering
using single-linkage on the distance matrix.

Translated from the core algorithm of TASM's
`engine/modules/correction_manifold.py`. Visualization helpers (PCA
projection for the 2D manifold plot, ESC-style trajectories) are skipped
here — the service layer computes them on demand from the signatures
output. This is flagged in NOTES.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from misc.old.tagm.analysis.base import AnalysisModule, AnalysisResult
from misc.old.tagm.analysis.registry import register_analysis
from misc.old.tagm.analysis.statistics import extract_scalar
from misc.old.tagm.measurement.parameters import ModuleParameter


# Default signature fields: (measurement, scalar, display)
_SIGNATURE_FIELDS = [
    ("stress_score", "stress_mean", "Stress"),
    ("last_position_attribution", "net_correction_to_last", "NetCorrection"),
    ("last_position_attribution", "entropy", "Entropy"),
    ("last_position_attribution", "top2_share", "Top2Share"),
    ("lateral_tension_profile", "mean_M", "LTP_M"),
    ("lateral_tension_profile", "mean_V", "LTP_V"),
    ("lateral_tension_profile", "max_prc", "MaxPRC"),
    ("spectral_field_density", "density_mean", "SFD_Mean"),
    ("spectral_field_density", "density_var", "SFD_Var"),
    ("rank_displacement", "mean_replacement", "RD_Replacement"),
]


@register_analysis
class CorrectionManifold(AnalysisModule):
    name = "correction_manifold"
    display_name = "Correction Manifold"
    description = (
        "Cluster prompts by similarity of their correction signatures, "
        "yielding a grouping of prompts whose correction patterns resemble "
        "each other across the selected measurement fields."
    )
    version = "0.1.0"

    depends_on_measurements = ("stress_score",)

    parameters = [
        ModuleParameter(
            name="n_clusters",
            display_name="Number of clusters",
            description="Target cluster count for agglomerative clustering.",
            kind="int", default=4, min_value=2, max_value=20,
        ),
        ModuleParameter(
            name="distance",
            display_name="Distance metric",
            description="Distance between z-scored signature vectors.",
            kind="select", default="cosine", options=("cosine", "euclidean"),
        ),
        ModuleParameter(
            name="pca_2d",
            display_name="Include 2D PCA projection",
            description="Compute and include a 2D PCA projection for plotting.",
            kind="bool", default=True,
        ),
    ]

    def run(self, session, params, probes=None):
        n_clusters = int(params.get("n_clusters", 4))
        distance = params.get("distance", "cosine")
        want_pca = bool(params.get("pca_2d", True))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"n_clusters": n_clusters, "distance": distance,
                        "pca_2d": want_pca,
                        "signature_fields": [
                            {"measurement": m, "field": f, "display": d}
                            for m, f, d in _SIGNATURE_FIELDS
                        ]},
        )

        prompts = session.get("prompts") or []
        if len(prompts) < 2:
            result.warnings.append("Need at least 2 prompts for manifold analysis")
            return result

        # Build signature matrix
        sig_rows: list[list[float]] = []
        field_labels = [d for _, _, d in _SIGNATURE_FIELDS]
        for p in prompts:
            row = []
            for m, f, _ in _SIGNATURE_FIELDS:
                v = extract_scalar(p, m, f)
                row.append(v if v is not None else np.nan)
            sig_rows.append(row)

        sigs = np.array(sig_rows, dtype=float)  # (n_prompts, n_fields)

        # Drop columns that are all-NaN
        valid_cols = [i for i in range(sigs.shape[1])
                      if not np.all(np.isnan(sigs[:, i]))]
        if not valid_cols:
            result.warnings.append("No signature fields produced usable data")
            return result
        sigs = sigs[:, valid_cols]
        field_labels = [field_labels[i] for i in valid_cols]

        # Impute per-column means for remaining NaNs
        col_means = np.nanmean(sigs, axis=0)
        inds = np.where(np.isnan(sigs))
        sigs[inds] = np.take(col_means, inds[1])

        # Z-score
        mu = sigs.mean(axis=0)
        sigma = sigs.std(axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        z = (sigs - mu) / sigma

        # Distance matrix
        dmat = _pairwise_distance(z, distance)

        # Agglomerative clustering (single linkage)
        labels = _single_linkage_clusters(dmat, n_clusters=n_clusters)

        # Cluster membership + per-cluster profile (mean z-scores)
        per_cluster: dict[int, dict] = {}
        for cid in set(labels):
            members = [i for i, l in enumerate(labels) if l == cid]
            if not members:
                continue
            cluster_sigs = z[members]
            per_cluster[int(cid)] = {
                "n_members": len(members),
                "member_prompt_indices": members,
                "mean_z": cluster_sigs.mean(axis=0).tolist(),
                "std_z": cluster_sigs.std(axis=0).tolist(),
            }

        result.objects["signature_fields"] = field_labels
        result.objects["signatures_z"] = z.tolist()
        result.objects["cluster_labels"] = [int(l) for l in labels]
        result.objects["per_cluster"] = per_cluster
        result.objects["distance_matrix"] = dmat.tolist()
        result.scalars["n_prompts"] = len(prompts)
        result.scalars["n_clusters_found"] = len(per_cluster)
        result.scalars["n_signature_fields"] = len(field_labels)

        if want_pca:
            pca = _pca_2d(z)
            if pca is not None:
                result.objects["pca_2d"] = pca.tolist()

        return result


# ── Helpers ─────────────────────────────────────────────────────────

def _pairwise_distance(X: np.ndarray, metric: str) -> np.ndarray:
    n = X.shape[0]
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        Xn = X / norms
        sim = Xn @ Xn.T
        return (1.0 - sim).astype(float)
    # euclidean
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _single_linkage_clusters(dmat: np.ndarray, n_clusters: int) -> list[int]:
    """Simple single-linkage agglomerative clustering.

    Not optimized — O(n^3) — but fine for session sizes (hundreds of prompts).
    Starts with each point as its own cluster and merges the nearest pair
    until only `n_clusters` remain.
    """
    n = dmat.shape[0]
    if n <= n_clusters:
        return list(range(n))

    # Each cluster is represented by its member list
    clusters: list[list[int]] = [[i] for i in range(n)]

    def _min_inter_distance(a, b):
        return min(dmat[i, j] for i in a for j in b)

    while len(clusters) > n_clusters:
        best = (0, 1, float("inf"))
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = _min_inter_distance(clusters[i], clusters[j])
                if d < best[2]:
                    best = (i, j, d)
        i, j, _ = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    labels = [0] * n
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Project X into 2D via SVD. Returns (n, 2) or None on failure."""
    if X.shape[0] < 2 or X.shape[1] < 1:
        return None
    centered = X - X.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(2, Vt.shape[0])
        proj = centered @ Vt[:k].T
        if k < 2:
            proj = np.hstack([proj, np.zeros((proj.shape[0], 1))])
        return proj
    except np.linalg.LinAlgError:
        return None
