"""CorrectionHeatmap: aggregate correction measures per probe cell.

For each (row, column) cell of the active probe template, aggregates
the correction measures (stress, attribution, density) of every token
across the session whose best-matching probe landed in that cell.

Produces a (n_rows × n_columns) grid of cell-level mean/std values per
channel, plus drill-down lists of contributing tokens. Translated from
TASM's `engine/modules/correction_heatmap.py`. TAGM's per-token alignment
contract eliminates the off-by-one TASM bug where per_token_final_emb
was indexed from position 1.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter


_CHANNELS = {
    "stress":  ("stress_score", "stress"),
    "attr":    ("last_position_attribution", "signed_attribution_to_last"),
    "density": ("spectral_field_density", "density"),
}


@register_analysis
class CorrectionHeatmap(AnalysisModule):
    name = "correction_heatmap"
    display_name = "Correction Heatmap"
    description = (
        "Per-cell aggregation of correction measures over all tokens whose "
        "best-matching probe was assigned to that cell. Grid output "
        "suitable for rendering a template-shaped heatmap."
    )
    version = "0.1.0"

    depends_on_measurements = ("probe_projection", "stress_score")

    parameters = [
        ModuleParameter(
            name="channel",
            display_name="Channel",
            description="Which correction channel to aggregate.",
            kind="select", default="stress",
            options=("stress", "attr", "density"),
        ),
        ModuleParameter(
            name="min_tokens_per_cell",
            display_name="Minimum tokens per cell",
            description="Cells with fewer contributing tokens are reported "
                        "as NaN (under-powered).",
            kind="int", default=5, min_value=1, max_value=100,
        ),
    ]

    def run(self, session, params, probes=None):
        channel = params.get("channel", "stress")
        min_n = int(params.get("min_tokens_per_cell", 5))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"channel": channel, "min_tokens_per_cell": min_n},
        )

        if channel not in _CHANNELS:
            result.warnings.append(f"Unknown channel '{channel}'")
            return result

        meas_name, field_name = _CHANNELS[channel]
        prompts = session.get("prompts") or []

        # Cell-keyed accumulators
        cell_values: dict[str, list[float]] = defaultdict(list)
        cell_tokens: dict[str, list[dict]] = defaultdict(list)
        probe_labels_seen: set[str] = set()

        for prompt_idx, p in enumerate(prompts):
            ms = p.get("measurements") or {}
            proj = ms.get("probe_projection") or {}
            labels = (proj.get("objects") or {}).get("probe_labels") or []
            per_token = proj.get("per_token") or {}
            best_idx = per_token.get("best_class_idx") or []
            best_score = per_token.get("best_score") or []

            if not labels or not best_idx:
                continue

            ch_meas = ms.get(meas_name) or {}
            values = (ch_meas.get("per_token") or {}).get(field_name) or []
            tokens = p.get("tokens") or []

            seq_len = min(len(best_idx), len(values), len(tokens))
            for i in range(seq_len):
                idx = best_idx[i]
                val = values[i]
                if idx is None or val is None:
                    continue
                try:
                    idx_i = int(idx)
                    val_f = float(val)
                except (TypeError, ValueError):
                    continue
                if np.isnan(val_f) or np.isinf(val_f):
                    continue
                if not (0 <= idx_i < len(labels)):
                    continue
                label = labels[idx_i]
                probe_labels_seen.add(label)
                cell_values[label].append(val_f)
                score = best_score[i] if i < len(best_score) else None
                cell_tokens[label].append({
                    "prompt_idx": prompt_idx,
                    "position": i,
                    "token": tokens[i],
                    "value": val_f,
                    "match_score": float(score) if score is not None else None,
                })

        # Compute cell statistics
        cell_stats: dict[str, dict] = {}
        for label in sorted(probe_labels_seen):
            vals = cell_values[label]
            if len(vals) < min_n:
                cell_stats[label] = {
                    "n": len(vals),
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "underpowered": True,
                }
                continue
            arr = np.array(vals, dtype=float)
            cell_stats[label] = {
                "n": len(vals),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "underpowered": False,
            }

        # Top contributing tokens per cell (sorted by value magnitude)
        top_tokens_per_cell = {}
        for label, recs in cell_tokens.items():
            top_tokens_per_cell[label] = sorted(
                recs, key=lambda r: abs(r["value"]), reverse=True,
            )[:20]

        result.objects["cell_stats"] = cell_stats
        result.objects["top_tokens_per_cell"] = top_tokens_per_cell
        result.scalars["n_cells"] = len(cell_stats)
        result.scalars["n_tokens_attributed"] = sum(s["n"] for s in cell_stats.values())
        result.scalars["n_prompts"] = len(prompts)
        return result
