"""ProbeStore: content-addressed persistent storage for ProbeSets.

Sits on top of the Cache subsystem's probe/ directory. A ProbeSet is
identified by a content hash derived from (template_id, capture_signature,
model_pair_id, parameters). Callers either `put` a freshly generated
set (which computes the hash and writes to disk) or `get` an existing
one by constructing the identity tuple.

Listing iterates over all .json sidecar files; loading fetches the
corresponding .npz.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from tagm.probes.artifact import ProbeSet


def compute_probe_set_id(
    template_id: str,
    capture_signature: str,
    model_pair_id: str,
    parameters: dict[str, Any],
) -> str:
    """Content hash for a probe set's identity. Short prefix of SHA-256."""
    canonical = json.dumps({
        "template_id": template_id,
        "capture_signature": capture_signature,
        "model_pair_id": model_pair_id,
        "parameters": parameters,
    }, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


class ProbeStore:
    """Content-addressed ProbeSet storage.

    Each set is one .npz (embedding matrices) plus one .json sidecar
    (metadata + labels). Sets are looked up by (template, capture, pair,
    params) tuples or by explicit set_id.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Write ───────────────────────────────────────────────────────
    def put(self, probe_set: ProbeSet) -> Path:
        """Persist a ProbeSet. Returns the path written.

        If the set already exists on disk (same set_id), it's overwritten.
        """
        path = self._npz_path(probe_set.set_id)
        probe_set.save(path)
        return path

    # ── Read ────────────────────────────────────────────────────────
    def get(
        self,
        template_id: str,
        capture_signature: str,
        model_pair_id: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> Optional[ProbeSet]:
        """Look up a probe set by its identity tuple.

        Returns None if not found. Use has() if you only need existence.
        """
        set_id = compute_probe_set_id(
            template_id=template_id,
            capture_signature=capture_signature,
            model_pair_id=model_pair_id,
            parameters=parameters or {},
        )
        return self.get_by_id(set_id)

    def get_by_id(self, set_id: str) -> Optional[ProbeSet]:
        path = self._npz_path(set_id)
        if not path.exists():
            return None
        return ProbeSet.load(path)

    def has(self, template_id: str, capture_signature: str,
            model_pair_id: str,
            parameters: Optional[dict[str, Any]] = None) -> bool:
        set_id = compute_probe_set_id(
            template_id=template_id,
            capture_signature=capture_signature,
            model_pair_id=model_pair_id,
            parameters=parameters or {},
        )
        return self._npz_path(set_id).exists()

    # ── List ────────────────────────────────────────────────────────
    def list(self) -> list[dict]:
        """Return light-weight descriptors for every stored probe set.

        Reads only the .json sidecars; embeddings are not loaded.
        """
        out: list[dict] = []
        for sidecar in self.root.glob("*.json"):
            try:
                with open(sidecar) as f:
                    meta = json.load(f)
                meta["set_id"] = sidecar.stem
                meta["n_probes"] = len(meta.get("probes") or [])
                meta.pop("probes", None)
                out.append(meta)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(out, key=lambda m: m.get("template_name", ""))

    # ── Delete ──────────────────────────────────────────────────────
    def delete(self, set_id: str) -> bool:
        """Delete a probe set by id. Returns True if a file was removed."""
        npz = self._npz_path(set_id)
        meta = npz.with_suffix(".json")
        removed = False
        for p in (npz, meta):
            if p.exists():
                p.unlink()
                removed = True
        return removed

    # ── Paths ───────────────────────────────────────────────────────
    def _npz_path(self, set_id: str) -> Path:
        return self.root / f"{set_id}.npz"
