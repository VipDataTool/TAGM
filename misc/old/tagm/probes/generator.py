"""Probe embedding generator.

Generates probe embeddings for template cells by running each cell's
tokens through the instruct model and capturing hidden states at
requested depths. Uses the engine's ActivationCapture for hooks.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from misc.old.tagm.core.types import ProgressCallback, noop_progress
from misc.old.tagm.probes.artifact import ProbeEmbedding, ProbeSet
from misc.old.tagm.probes.store import compute_probe_set_id
from misc.old.tagm.probes.template import ProbeTemplate, load_stopwords

if TYPE_CHECKING:
    from misc.old.tagm.core.pipeline import Pipeline

logger = logging.getLogger("tagm")


@dataclass
class GenerationParams:
    """Generator parameters."""
    depth_layers: dict[str, int]
    include_final_norm: bool = True
    filter_stopwords: bool = True
    project_through_o_delta: bool = False
    min_tokens_per_cell: int = 0

    def to_dict(self) -> dict:
        return {
            "depth_layers": {k: int(v) for k, v in self.depth_layers.items()},
            "include_final_norm": bool(self.include_final_norm),
            "filter_stopwords": bool(self.filter_stopwords),
            "project_through_o_delta": bool(self.project_through_o_delta),
            "min_tokens_per_cell": int(self.min_tokens_per_cell),
        }


class EmbeddingGenerator:
    """Generates probe embeddings from a loaded model pair."""

    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    @property
    def adapter(self):
        return self.pipeline.adapter

    @property
    def instruct_model(self):
        return self.pipeline.instruct_model

    def generate(
        self,
        template: ProbeTemplate,
        params: GenerationParams,
        progress: Optional[ProgressCallback] = None,
    ) -> ProbeSet:
        """Generate a ProbeSet for the given template and parameters."""
        from misc.old.tagm.engine.hooks import ActivationCapture

        log = progress or noop_progress

        depth_labels: list[str] = list(params.depth_layers.keys())
        layers_by_depth: dict[str, int] = dict(params.depth_layers)
        if params.include_final_norm:
            depth_labels.append("final")

        # Build a capture signature for identity
        sig_parts = sorted(f"{d}:{l}" for d, l in params.depth_layers.items())
        if params.include_final_norm:
            sig_parts.append("final_norm")
        capture_signature = hashlib.sha256(
            "|".join(sig_parts).encode()).hexdigest()[:16]

        stopwords = load_stopwords() if params.filter_stopwords else set()
        probes: list[ProbeEmbedding] = []
        hidden_size = self.adapter.hidden_size(self.instruct_model)
        cells_total = len(template.cells)

        capture = ActivationCapture()

        for cell_idx, cell in enumerate(template.cells):
            log("generator",
                f"Cell {cell_idx+1}/{cells_total}: {cell.row}/{cell.column}")

            tokens_to_embed = [
                t for t in cell.tokens
                if t.strip() and t.strip().lower() not in stopwords
            ]
            if len(tokens_to_embed) < params.min_tokens_per_cell:
                continue

            per_depth_accum: dict[str, list[np.ndarray]] = {d: [] for d in depth_labels}

            for token_text in tokens_to_embed:
                embs = self._embed_token(
                    token_text, capture, depth_labels, layers_by_depth,
                    project_through_o_delta=params.project_through_o_delta,
                )
                if embs is None:
                    continue
                for d, vec in embs.items():
                    per_depth_accum[d].append(vec)

            if not any(per_depth_accum[d] for d in depth_labels):
                continue

            cell_embs: dict[str, np.ndarray] = {}
            for d, vecs in per_depth_accum.items():
                if not vecs:
                    continue
                mean = np.mean(np.stack(vecs), axis=0)
                norm = np.linalg.norm(mean)
                if norm > 1e-12:
                    mean = mean / norm
                cell_embs[d] = mean.astype(np.float32)

            probes.append(ProbeEmbedding(
                label=f"{cell.row}/{cell.column}",
                row=cell.row, column=cell.column,
                cell=f"{cell.row}|{cell.column}",
                embeddings=cell_embs,
            ))

        capture.remove()

        model_pair_id = f"{self.pipeline.instruct_model_id}|{self.pipeline.base_model_id}"
        set_id = compute_probe_set_id(
            template_id=template.template_id,
            capture_signature=capture_signature,
            model_pair_id=model_pair_id,
            parameters=params.to_dict(),
        )

        return ProbeSet(
            set_id=set_id,
            template_id=template.template_id,
            template_name=template.name,
            capture_signature=capture_signature,
            model_pair_id=model_pair_id,
            adapter_family=self.adapter.family_id,
            parameters=params.to_dict(),
            probes=probes,
            depth_labels=tuple(depth_labels),
            hidden_size=hidden_size,
        )

    def _embed_token(
        self,
        token_text: str,
        capture,
        depth_labels: list[str],
        layers_by_depth: dict[str, int],
        project_through_o_delta: bool,
    ) -> Optional[dict[str, np.ndarray]]:
        """Run one forward pass and extract per-depth embeddings."""
        # Hook the required layers
        hook_layers = list(set(layers_by_depth.values()))
        capture.install(
            self.instruct_model, self.adapter,
            signal_layers=hook_layers,
        )

        try:
            tokens, inputs, output = capture.forward(
                self.instruct_model, self.pipeline.tokenizer, token_text)
        except Exception as e:
            logger.warning(f"[generator] Forward failed for {token_text!r}: {e}")
            return None

        seq_len = len(tokens)
        if seq_len == 0:
            return None

        out: dict[str, np.ndarray] = {}
        for depth in depth_labels:
            if depth == "final":
                t = capture.activations.get("final_norm_h")
            else:
                layer_idx = layers_by_depth.get(depth)
                t = capture.activations.get(f"layer_{layer_idx}_h")
            if t is None:
                continue

            arr = t[0, :seq_len].float().cpu().numpy()
            mean = arr.mean(axis=0)

            if project_through_o_delta and depth != "final":
                layer_idx = layers_by_depth.get(depth)
                dw_o = self.pipeline.delta_store.o_delta_or_none(layer_idx) \
                    if layer_idx is not None else None
                if dw_o is not None:
                    mean_t = torch.from_numpy(mean.astype(np.float32))
                    mean = torch.matmul(mean_t, dw_o.float().cpu().T).numpy()

            norm = np.linalg.norm(mean)
            if norm > 1e-12:
                mean = mean / norm
            out[depth] = mean.astype(np.float32)

        return out
