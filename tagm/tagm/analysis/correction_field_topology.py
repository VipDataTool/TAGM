"""Correction Field Topology — validation + summary for the 3D terrain viewer.

The actual 3D renderer is Three.js code in static/index.html
(`_terrainRendererCore` et al.). It reads per-prompt session records
directly. This module's job is narrower: count what's available, aggregate
displacement magnitude and bank-asymmetry statistics for the Modules-tab
summary pane, and echo the launch parameters so the Launch button can
pick them up.

Reads from session prompt records at the keys the frontend itself uses:
  - `r.rank_displacement.instruct_disp_profiles`
  - `r.rank_displacement.base_disp_profiles`
  - `r.ltp.profiles` (fallback if displacement profiles absent)
  - `r.base_counterfactual_tokens` (at root; presence implies dual-bank data)
  - `r.tokens`, `r.category`

Returns a plain dict whose top-level keys are what `renderCFTResults()`
reads: n_total, n_with_displacement, n_with_ltp_fallback, n_with_base_candidates,
n_skipped, token_stats, by_category, displacement_stats, asymmetry_stats,
launch_params.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


@register_analysis
class CorrectionFieldTopology(AnalysisModule):
    name = "correction_field_topology"
    display_name = "Correction Field Topology"
    description = (
        "3D displacement field visualization of the alignment correction. "
        "Dual-bank terrain maps per-token probability displacement between "
        "base and instruct models. Surface height = displacement magnitude. "
        "Instruct bank (warm) shows promoted candidates; base bank (cool) "
        "shows demoted candidates. Asymmetry reveals where RLHF reshaped "
        "the output distribution."
    )
    version = "1.0.0"

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
            description="Maximum prompts to load into the visualization. "
                        "Higher = slower but more complete.",
            kind="int", default=100, min_value=1, max_value=2000,
        ),
        ModuleParameter(
            name="token_limit",
            display_name="Token Limit",
            description="Maximum tokens rendered per prompt terrain. "
                        "Higher = more detail but heavier.",
            kind="int", default=20, min_value=4, max_value=100,
        ),
        ModuleParameter(
            name="char_limit",
            display_name="Prompt Label Length",
            description="Character limit for prompt labels in the viewer dropdown.",
            kind="int", default=50, min_value=20, max_value=200,
        ),
        ModuleParameter(
            name="auto_rotate",
            display_name="Auto-Rotate",
            description="Spin terrain around vertical axis on launch.",
            kind="bool", default=False,
        ),
        ModuleParameter(
            name="rotate_speed",
            display_name="Rotate Speed (RPM)",
            description="Auto-rotation speed in revolutions per minute.",
            kind="float", default=0.3, min_value=0.1, max_value=2.0,
        ),
    ]

    # ── Dependency check ────────────────────────────────────────────
    #
    # Overrides the base class's `check_dependencies`, which reads an old
    # nested `measurements` sub-dict that no longer exists on the record.
    # We validate against the actual flat record shape: at least one
    # prompt must carry either rank-displacement profiles or LTP profiles.

    def check_dependencies(self, session: dict) -> list[str]:
        prompts = session.get("prompts") or []
        if not prompts:
            return ["No prompts in session. Analyze some prompts first."]
        for r in prompts:
            if _has_terrain_data(r):
                return []
        return [
            "No prompts have displacement or LTP profile data. "
            "Re-run analysis with Rank Displacement and/or Lateral "
            "Tension Profile enabled."
        ]

    # ── Run ─────────────────────────────────────────────────────────

    def run(self, session: dict, params: dict,
             probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []

        stats: dict[str, Any] = {
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

        disp_magnitudes_by_cat: dict[str, list[float]] = defaultdict(list)
        asymmetry_by_cat: dict[str, list[float]] = defaultdict(list)
        all_token_counts: list[int] = []

        for r in prompts:
            cat = (r.get("category") or "unknown").lower().strip()
            rd = r.get("rank_displacement") or {}
            ltp = r.get("ltp") or {}
            tokens = r.get("tokens") or []
            n_tok = len(tokens)

            has_disp = bool(rd.get("instruct_disp_profiles"))
            has_ltp = bool(ltp.get("profiles"))
            has_base_cf = bool(r.get("base_counterfactual_tokens"))

            if not has_disp and not has_ltp:
                stats["n_skipped"] += 1
                continue

            if has_disp:
                stats["n_with_displacement"] += 1
            else:
                stats["n_with_ltp_fallback"] += 1

            if has_base_cf:
                stats["n_with_base_candidates"] += 1

            all_token_counts.append(n_tok)

            bc = stats["by_category"].setdefault(
                cat, {"count": 0, "mean_tokens": 0, "_token_counts": []})
            bc["count"] += 1
            bc["_token_counts"].append(n_tok)

            if has_disp:
                i_profs = rd.get("instruct_disp_profiles") or []
                b_profs = rd.get("base_disp_profiles") or []
                n_profs = min(n_tok, len(i_profs))
                for pos in range(n_profs):
                    ip = i_profs[pos] if pos < len(i_profs) else []
                    bp = b_profs[pos] if pos < len(b_profs) else []
                    i_mag = _sum_nonnull(ip)
                    b_mag = _sum_nonnull(bp)
                    disp_magnitudes_by_cat[cat].append(i_mag + b_mag)
                    total = i_mag + b_mag
                    if total > 1e-10:
                        asymmetry_by_cat[cat].append((i_mag - b_mag) / total)

        # Token aggregates
        if all_token_counts:
            stats["token_stats"]["total_tokens"] = int(sum(all_token_counts))
            stats["token_stats"]["mean_tokens_per_prompt"] = round(
                float(np.mean(all_token_counts)), 1)
            stats["token_stats"]["max_tokens"] = int(max(all_token_counts))

        # Flatten per-category accumulator
        for cat_name, info in stats["by_category"].items():
            counts = info.pop("_token_counts")
            info["mean_tokens"] = round(
                float(np.mean(counts)), 1) if counts else 0

        # Displacement magnitude stats
        if disp_magnitudes_by_cat:
            all_mags: list[float] = []
            per_cat: dict[str, dict] = {}
            for cat_name, mags in disp_magnitudes_by_cat.items():
                all_mags.extend(mags)
                arr = np.array(mags, dtype=float)
                per_cat[cat_name] = {
                    "mean": round(float(arr.mean()), 6),
                    "std": round(float(arr.std()), 6),
                    "median": round(float(np.median(arr)), 6),
                    "max": round(float(arr.max()), 6),
                }
            a = np.array(all_mags, dtype=float)
            stats["displacement_stats"] = {
                "overall": {
                    "mean": round(float(a.mean()), 6),
                    "std": round(float(a.std()), 6),
                    "median": round(float(np.median(a)), 6),
                },
                "by_category": per_cat,
            }

        # Bank asymmetry stats (instruct vs base dominance)
        if asymmetry_by_cat:
            per_cat_a: dict[str, dict] = {}
            all_a: list[float] = []
            for cat_name, vals in asymmetry_by_cat.items():
                all_a.extend(vals)
                arr = np.array(vals, dtype=float)
                per_cat_a[cat_name] = {
                    "mean": round(float(arr.mean()), 4),
                    "std": round(float(arr.std()), 4),
                    "instruct_dominant_frac": round(
                        float(np.mean(arr > 0)), 4),
                }
            a = np.array(all_a, dtype=float)
            stats["asymmetry_stats"] = {
                "overall": {
                    "mean": round(float(a.mean()), 4),
                    "instruct_dominant_frac": round(
                        float(np.mean(a > 0)), 4),
                },
                "by_category": per_cat_a,
            }

        n_usable = stats["n_with_displacement"] + stats["n_with_ltp_fallback"]
        if n_usable == 0:
            return {
                "error": ("No prompts with displacement or LTP profile data."),
                "stats": stats,
            }

        # Echo launch parameters so the UI's Launch button can pick them up.
        stats["launch_params"] = {
            "category": params.get("category", "all"),
            "record_limit": int(params.get("record_limit", 100)),
            "token_limit": int(params.get("token_limit", 20)),
            "char_limit": int(params.get("char_limit", 50)),
            "auto_rotate": bool(params.get("auto_rotate", False)),
            "rotate_speed": float(params.get("rotate_speed", 0.3)),
        }

        logger.info(
            "[CFT] complete: %d usable (%d displacement, %d LTP fallback), "
            "%d skipped",
            n_usable, stats["n_with_displacement"],
            stats["n_with_ltp_fallback"], stats["n_skipped"])

        return stats


# ── helpers (module-local) ──────────────────────────────────────────

def _sum_nonnull(xs) -> float:
    if not xs:
        return 0.0
    total = 0.0
    for v in xs:
        if v is None:
            continue
        try:
            total += float(v)
        except (TypeError, ValueError):
            continue
    return total


def _has_terrain_data(r: dict) -> bool:
    rd = r.get("rank_displacement") or {}
    if rd.get("instruct_disp_profiles"):
        return True
    ltp = r.get("ltp") or {}
    if ltp.get("profiles"):
        return True
    return False
