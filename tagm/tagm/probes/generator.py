"""Embedding Generator: generates ProbeSets from templates using a loaded Pipeline.

This is a core operation of TAGM, not a measurement module. It uses the
same Pipeline as prompt analysis, but instead of feeding activations to
measurement modules it extracts per-token embeddings from user-submitted
template tokens and writes them to the ProbeStore.

For each (row, column, token) entry in the template:
  1. Tokenize the token string. If it produces multiple input tokens,
     mean-pool the captured hidden states across them (per v1 spec default).
  2. Run the model with a CaptureConfig that records hidden states at the
     requested depths (subject layer, escalation layer, final norm).
  3. L2-normalize the per-token embedding vector.
  4. For cells with multiple tokens, mean-pool across tokens after
     per-token normalization.

The output is a ProbeSet with one ProbeEmbedding per non-empty cell.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from tagm.core.capture.config import CaptureConfig, CapturePoint
from tagm.core.types import ProgressCallback, noop_progress
from tagm.probes.artifact import ProbeEmbedding, ProbeSet
from tagm.probes.store import compute_probe_set_id
from tagm.probes.template import ProbeTemplate, load_stopwords

if TYPE_CHECKING:
    from tagm.core.pipeline import Pipeline

logger = logging.getLogger("tagm")


@dataclass
class GenerationParams:
    """Generator parameters. Recorded on the ProbeSet for reproducibility."""
    depth_layers: dict[str, int]            # e.g. {"subject": 12, "escalation": 18}
    include_final_norm: bool = True          # also capture final_norm as "final" depth
    filter_stopwords: bool = True
    project_through_o_delta: bool = False    # project embeddings via o_proj delta
    min_tokens_per_cell: int = 0              # cells with fewer survivors are dropped

    def to_dict(self) -> dict:
        return {
            "depth_layers": {k: int(v) for k, v in self.depth_layers.items()},
            "include_final_norm": bool(self.include_final_norm),
            "filter_stopwords": bool(self.filter_stopwords),
            "project_through_o_delta": bool(self.project_through_o_delta),
            "min_tokens_per_cell": int(self.min_tokens_per_cell),
        }


class EmbeddingGenerator:
    """Runs a template through a Pipeline to produce a ProbeSet."""

    def __init__(self, pipeline: "Pipeline"):
        if not pipeline.loaded:
            raise RuntimeError(
                "EmbeddingGenerator requires a loaded Pipeline. "
                "Call Pipeline.load() first.")
        self.pipeline = pipeline
        self.adapter = pipeline.adapter
        self.tokenizer = pipeline.tokenizer
        self.instruct_model = pipeline.instruct_model
        self.device = pipeline.device

    # ── Main entry point ────────────────────────────────────────────
    def generate(
        self,
        template: ProbeTemplate,
        params: GenerationParams,
        progress: Optional[ProgressCallback] = None,
    ) -> ProbeSet:
        """Generate a ProbeSet for the given template and parameters.

        One forward pass per template token. Embeddings are captured at
        every requested depth in the same pass (single-pass discipline).
        """
        log = progress or noop_progress

        # Build the CaptureConfig covering all requested depths
        capture_points: list[CapturePoint] = []
        depth_labels: list[str] = []
        layers_by_depth: dict[str, int] = {}

        for depth_label, layer_idx in params.depth_layers.items():
            capture_points.append(CapturePoint(
                layer=layer_idx,
                hook_point="residual_post_block",
                capture=frozenset({"hidden"}),
            ))
            depth_labels.append(depth_label)
            layers_by_depth[depth_label] = layer_idx

        if params.include_final_norm:
            capture_points.append(CapturePoint(
                layer=None, hook_point="final_norm",
                capture=frozenset({"hidden"}),
            ))
            depth_labels.append("final")

        cfg = CaptureConfig(
            name="probe-generation",
            description="Embedding Generator capture config",
            points=tuple(capture_points),
        )

        # Stopword filter
        stopwords = load_stopwords() if params.filter_stopwords else set()

        # Process each cell
        probes: list[ProbeEmbedding] = []
        hidden_size = self.adapter.hidden_size(self.instruct_model)
        cells_total = len(template.cells)

        for cell_idx, cell in enumerate(template.cells):
            log("generator",
                f"Cell {cell_idx+1}/{cells_total}: {cell.row}/{cell.column} "
                f"({len(cell.tokens)} tokens)")

            tokens_to_embed = [
                t for t in cell.tokens
                if t.strip() and t.strip().lower() not in stopwords
            ]
            if len(tokens_to_embed) < params.min_tokens_per_cell:
                continue

            per_depth_accum: dict[str, list[np.ndarray]] = {d: [] for d in depth_labels}

            for token_text in tokens_to_embed:
                embs = self._embed_token(
                    token_text, cfg, depth_labels, layers_by_depth,
                    project_through_o_delta=params.project_through_o_delta,
                )
                if embs is None:
                    continue
                for d, vec in embs.items():
                    per_depth_accum[d].append(vec)

            if not any(per_depth_accum[d] for d in depth_labels):
                continue

            # Mean-pool per depth, then re-normalize
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

        # Compute set identity
        model_pair_id = f"{self.pipeline.instruct_model_id}|{self.pipeline.base_model_id}"
        set_id = compute_probe_set_id(
            template_id=template.template_id,
            capture_signature=cfg.signature(),
            model_pair_id=model_pair_id,
            parameters=params.to_dict(),
        )

        return ProbeSet(
            set_id=set_id,
            template_id=template.template_id,
            template_name=template.name,
            capture_signature=cfg.signature(),
            model_pair_id=model_pair_id,
            adapter_family=self.adapter.family_id,
            parameters=params.to_dict(),
            probes=probes,
            depth_labels=tuple(depth_labels),
            hidden_size=hidden_size,
        )

    # ── Single-token embedding ──────────────────────────────────────
    def _embed_token(
        self,
        token_text: str,
        cfg: CaptureConfig,
        depth_labels: list[str],
        layers_by_depth: dict[str, int],
        project_through_o_delta: bool,
    ) -> Optional[dict[str, np.ndarray]]:
        """Run one forward pass on `token_text` and extract per-depth embeddings.

        Returns a dict mapping depth label -> L2-normalized embedding vector,
        or None if the token produced no input tokens (rare; whitespace-only
        strings are already filtered upstream).
        """
        try:
            result = self.pipeline.run(token_text, cfg, use_base=False)
        except Exception as e:
            logger.warning(f"[generator] Forward pass failed for {token_text!r}: {e}")
            return None

        store = result.activations
        seq_len = result.seq_len
        if seq_len == 0:
            return None

        out: dict[str, np.ndarray] = {}
        for depth in depth_labels:
            if depth == "final":
                t = store.get_or_none(None, "final_norm", "hidden")
            else:
                layer_idx = layers_by_depth.get(depth)
                t = store.get_or_none(layer_idx, "residual_post_block", "hidden")
            if t is None:
                continue
            # Mean-pool across input tokens — variable-length subword split
            # handling (see measurement spec §8 open question)
            arr = t[0, :seq_len].float().cpu().numpy()
            mean = arr.mean(axis=0)

            # Optional projection through o_proj delta
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
