"""Probe artifacts: ProbeSet (collection) and ProbeEmbedding (single probe).

A ProbeSet is the content-addressed unit stored in the ProbeStore. It holds:
  - A list of ProbeEmbeddings (one per template cell).
  - Metadata about the template, the model pair it was generated for,
    the capture config used, and the parameters (pooling, projection, etc).

Stored as numpy .npz files under ~/.tagm/cache/probes/<hash>.npz
(embedding matrices) plus a sidecar .json (metadata and labels). This
keeps the data format stdlib-only and lets the metadata be inspected
without loading the numerics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ProbeEmbedding:
    """A single probe: label + one or more depth-keyed embedding vectors.

    A probe can be embedded at multiple depths (e.g. "subject" and
    "escalation" layers); each depth's embedding is stored under a
    string key. All embeddings are L2-normalized at storage time.
    """
    label: str                              # human-readable label (e.g. "firearms")
    row: str                                # template row (e.g. class)
    column: str                             # template column (e.g. subclass)
    cell: str                               # cell identifier (row|column)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    # Map depth_label -> L2-normalized vector (hidden_size,)


@dataclass
class ProbeSet:
    """A content-addressed collection of probe embeddings.

    The set's hash (its identity in the ProbeStore) is derived from
    (template_id, capture_signature, model_pair_id, parameters). Any
    change to any of those produces a different hash and therefore a
    different stored set — this is how we avoid staleness.
    """
    set_id: str                             # the content hash (also the cache filename)
    template_id: str
    template_name: str                      # human-readable template name
    capture_signature: str                  # hash of the CaptureConfig used
    model_pair_id: str                      # "<instruct>|<base>"
    adapter_family: str
    parameters: dict[str, Any] = field(default_factory=dict)
    # ("depth_layers": {...}, "pool": "mean"|"attn", "projection_space": bool, ...)

    probes: list[ProbeEmbedding] = field(default_factory=list)
    depth_labels: tuple[str, ...] = ()      # e.g. ("subject", "escalation", "final")
    hidden_size: int = 0

    # ── Lookup helpers ──────────────────────────────────────────────
    def by_label(self, label: str) -> Optional[ProbeEmbedding]:
        for p in self.probes:
            if p.label == label:
                return p
        return None

    def by_cell(self, row: str, column: str) -> Optional[ProbeEmbedding]:
        for p in self.probes:
            if p.row == row and p.column == column:
                return p
        return None

    def embeddings_matrix(self, depth_label: str) -> tuple[np.ndarray, list[str]]:
        """Return (N, hidden_size) matrix of embeddings at a given depth,
        plus the list of labels in the same order."""
        labels: list[str] = []
        rows: list[np.ndarray] = []
        for p in self.probes:
            emb = p.embeddings.get(depth_label)
            if emb is None:
                continue
            labels.append(p.label)
            rows.append(emb)
        if not rows:
            return np.zeros((0, self.hidden_size)), []
        return np.stack(rows), labels

    # ── Subset filter ───────────────────────────────────────────────
    def filtered(self, class_filter: Optional[tuple] = None,
                 subclass_filter: Optional[tuple] = None) -> "ProbeSet":
        """Return a new ProbeSet containing only probes that match the filter.

        Filters are applied by row (class) and column (subclass). Metadata
        is preserved but `probes` is subset.
        """
        probes = self.probes
        if class_filter:
            probes = [p for p in probes if p.row in class_filter]
        if subclass_filter:
            probes = [p for p in probes if p.column in subclass_filter]
        return ProbeSet(
            set_id=self.set_id,
            template_id=self.template_id,
            template_name=self.template_name,
            capture_signature=self.capture_signature,
            model_pair_id=self.model_pair_id,
            adapter_family=self.adapter_family,
            parameters=dict(self.parameters),
            probes=probes,
            depth_labels=self.depth_labels,
            hidden_size=self.hidden_size,
        )

    # ── Serialization ───────────────────────────────────────────────
    def save(self, npz_path: Path) -> None:
        """Write embedding matrices to <path>.npz and metadata to <path>.json."""
        npz_path = Path(npz_path)
        arrays: dict[str, np.ndarray] = {}
        for depth in self.depth_labels:
            mat, labels = self.embeddings_matrix(depth)
            arrays[f"emb_{depth}"] = mat
        np.savez_compressed(npz_path, **arrays)

        meta = {
            "set_id": self.set_id,
            "template_id": self.template_id,
            "template_name": self.template_name,
            "capture_signature": self.capture_signature,
            "model_pair_id": self.model_pair_id,
            "adapter_family": self.adapter_family,
            "parameters": self.parameters,
            "depth_labels": list(self.depth_labels),
            "hidden_size": int(self.hidden_size),
            "probes": [
                {"label": p.label, "row": p.row, "column": p.column, "cell": p.cell}
                for p in self.probes
            ],
        }
        with open(npz_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, npz_path: Path) -> "ProbeSet":
        npz_path = Path(npz_path)
        meta_path = npz_path.with_suffix(".json")
        with open(meta_path) as f:
            meta = json.load(f)
        depth_labels = tuple(meta["depth_labels"])

        arrs = np.load(npz_path)
        per_depth: dict[str, np.ndarray] = {
            d: arrs[f"emb_{d}"] for d in depth_labels if f"emb_{d}" in arrs.files
        }

        probes = []
        for i, pm in enumerate(meta["probes"]):
            embeddings: dict[str, np.ndarray] = {}
            for d, mat in per_depth.items():
                if i < len(mat):
                    embeddings[d] = mat[i]
            probes.append(ProbeEmbedding(
                label=pm["label"], row=pm["row"],
                column=pm["column"], cell=pm["cell"],
                embeddings=embeddings,
            ))

        return cls(
            set_id=meta["set_id"],
            template_id=meta["template_id"],
            template_name=meta["template_name"],
            capture_signature=meta["capture_signature"],
            model_pair_id=meta["model_pair_id"],
            adapter_family=meta["adapter_family"],
            parameters=meta.get("parameters") or {},
            probes=probes,
            depth_labels=depth_labels,
            hidden_size=int(meta.get("hidden_size", 0)),
        )
