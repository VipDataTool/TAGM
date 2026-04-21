"""Correction Field Topology: displacement field validation and statistics.

Ported from TASM's engine/modules/displacement_field.py. Validates
session data for 3D displacement field visualization and computes
aggregate topology statistics. Data reading adapted to TAGM's native
session schema; output shape matches TASM.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


def _get_measurement_flat(prompt: dict, measurement_name: str) -> dict:
    """Extract a measurement's data merged into a flat dict.

    Merges scalars + per_token + per_layer + objects so downstream code
    can access fields the same way TASM's flat results did.
    """
    ms = (prompt.get("measurements") or {}).get(measurement_name)
    if not ms:
        return {}
    flat = {}
    flat.update(ms.get("scalars") or {})
    flat.update(ms.get("per_token") or {})
    flat.update(ms.get("per_layer") or {})
    flat.update(ms.get("objects") or {})
    return flat


def _compute_topology_stats(prompts):
    """Compute aggregate topology statistics from TAGM session prompts."""
    stats = {
        "n_total": len(prompts),
        "n_with_displacement": 0,
        "n_with_ltp_fallback": 0,
        "n_with_base_candidates": 0,
        "n_skipped": 0,
        "by_category": {},
        "token_stats": {
            "total_tokens": 0,
            "mean_tokens_per_prompt": 0,
            "max_tokens": 0,
        },
        "displacement_stats": None,
        "asymmetry_stats": None,
    }

    disp_magnitudes_by_cat = defaultdict(list)
    asymmetry_by_cat = defaultdict(list)
    all_token_counts = []

    for r in prompts:
        cat = (r.get("category") or "unknown").lower().strip()
        ltp = _get_measurement_flat(r, "lateral_tension_profile")
        rd = _get_measurement_flat(r, "rank_displacement")
        tokens = r.get("tokens") or []
        n_tok = len(tokens)

        has_disp = bool(rd.get("instruct_disp_profiles"))
        has_ltp = bool(ltp.get("profiles"))

        if not has_disp and not has_ltp:
            stats["n_skipped"] += 1
            continue

        if has_disp:
            stats["n_with_displacement"] += 1
        else:
            stats["n_with_ltp_fallback"] += 1

        # Check for base counterfactual tokens (in LTP objects)
        has_base_cf = bool(ltp.get("counterfactual_tokens"))
        if has_base_cf:
            stats["n_with_base_candidates"] += 1

        all_token_counts.append(n_tok)

        if cat not in stats["by_category"]:
            stats["by_category"][cat] = {"count": 0, "mean_tokens": 0, "token_counts": []}
        stats["by_category"][cat]["count"] += 1
        stats["by_category"][cat]["token_counts"].append(n_tok)

        if has_disp:
            i_profs = rd.get("instruct_disp_profiles") or []
            b_profs = rd.get("base_disp_profiles") or []
            for pos in range(min(n_tok, len(i_profs))):
                ip = i_profs[pos] if pos < len(i_profs) else []
                bp = b_profs[pos] if pos < len(b_profs) else []
                i_mag = sum(v for v in ip if v is not None) if ip else 0
                b_mag = sum(v for v in bp if v is not None) if bp else 0
                disp_magnitudes_by_cat[cat].append(i_mag + b_mag)
                total = i_mag + b_mag
                if total > 1e-10:
                    asymmetry_by_cat[cat].append((i_mag - b_mag) / total)

    if all_token_counts:
        stats["token_stats"]["total_tokens"] = sum(all_token_counts)
        stats["token_stats"]["mean_tokens_per_prompt"] = round(
            float(np.mean(all_token_counts)), 1)
        stats["token_stats"]["max_tokens"] = max(all_token_counts)

    for cat, info in stats["by_category"].items():
        counts = info.pop("token_counts")
        info["mean_tokens"] = round(float(np.mean(counts)), 1) if counts else 0

    if disp_magnitudes_by_cat:
        all_mags = []
        per_cat = {}
        for cat, mags in disp_magnitudes_by_cat.items():
            all_mags.extend(mags)
            arr = np.array(mags)
            per_cat[cat] = {
                "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std()), 6),
                "median": round(float(np.median(arr)), 6),
                "max": round(float(arr.max()), 6),
            }
        all_arr = np.array(all_mags)
        stats["displacement_stats"] = {
            "overall": {
                "mean": round(float(all_arr.mean()), 6),
                "std": round(float(all_arr.std()), 6),
                "median": round(float(np.median(all_arr)), 6),
            },
            "by_category": per_cat,
        }

    if asymmetry_by_cat:
        per_cat_asym = {}
        all_asym = []
        for cat, vals in asymmetry_by_cat.items():
            all_asym.extend(vals)
            arr = np.array(vals)
            per_cat_asym[cat] = {
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "instruct_dominant_frac": round(float(np.mean(arr > 0)), 4),
            }
        all_arr = np.array(all_asym)
        stats["asymmetry_stats"] = {
            "overall": {
                "mean": round(float(all_arr.mean()), 4),
                "instruct_dominant_frac": round(float(np.mean(all_arr > 0)), 4),
            },
            "by_category": per_cat_asym,
        }

    return stats


@register_analysis
class CorrectionFieldTopology(AnalysisModule):
    name = "correction_field_topology"
    display_name = "Correction Field Topology"
    description = (
        "3D displacement field visualization of the alignment correction. "
        "Dual-bank terrain maps per-token probability displacement between "
        "base and instruct models. Surface height = displacement magnitude."
    )
    version = "1.0.0"
    min_results = 1

    depends_on_measurements = ()

    parameters = [
        ModuleParameter(
            name="category",
            display_name="Category Filter",
            description="Filter prompts by category, or show all.",
            kind="select",
            default="all",
            options=("all", "benign", "mild", "harmful", "jailbreak",
                     "adversarial", "dual-use"),
        ),
        ModuleParameter(
            name="record_limit",
            display_name="Record Limit",
            description="Maximum prompts to load into the visualization.",
            kind="int",
            default=100,
            min_value=1,
            max_value=2000,
        ),
        ModuleParameter(
            name="token_limit",
            display_name="Token Limit",
            description="Maximum tokens rendered per prompt terrain.",
            kind="int",
            default=20,
            min_value=4,
            max_value=100,
        ),
        ModuleParameter(
            name="char_limit",
            display_name="Prompt Label Length",
            description="Character limit for prompt labels in the viewer dropdown.",
            kind="int",
            default=50,
            min_value=20,
            max_value=200,
        ),
        ModuleParameter(
            name="auto_rotate",
            display_name="Auto-Rotate",
            description="Spin terrain around vertical axis on launch.",
            kind="bool",
            default=False,
        ),
        ModuleParameter(
            name="rotate_speed",
            display_name="Rotate Speed (RPM)",
            description="Auto-rotation speed in revolutions per minute.",
            kind="float",
            default=0.3,
            min_value=0.1,
            max_value=2.0,
        ),
    ]

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []
        stats = _compute_topology_stats(prompts)

        n_usable = stats["n_with_displacement"] + stats["n_with_ltp_fallback"]
        if n_usable == 0:
            return {
                "error": "No prompts with displacement or LTP profile data.",
                "stats": stats,
            }

        stats["launch_params"] = {
            "category": params.get("category", "all"),
            "record_limit": params.get("record_limit", 100),
            "token_limit": params.get("token_limit", 20),
            "char_limit": params.get("char_limit", 50),
            "auto_rotate": params.get("auto_rotate", False),
            "rotate_speed": params.get("rotate_speed", 0.3),
        }

        return stats
