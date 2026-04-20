"""Correction Field Topology analysis (ported from TASM's displacement_field).

Direct port of TASM's `engine/modules/displacement_field.py`, adapted to
read TAGM's per-prompt measurement shape and emit TASM's wire format so
the existing frontend renderer (`renderCFTResults` in static/js/main.js)
produces the same Topology Summary, Category Breakdown, Displacement
Magnitude, and Bank Asymmetry cards it did under TASM.

The underlying quantities are preserved: `displacement_stats` pulls from
rank_displacement's instruct_disp_profiles / base_disp_profiles, and
`asymmetry_stats` derives from the same pair. `n_with_ltp_fallback` is
the count of prompts that have no rank_displacement but do have
lateral_tension_profile.profiles, matching TASM's semantics exactly.

Input mapping (TASM flat → TAGM nested):
  r.ltp.profiles       → p.measurements.lateral_tension_profile.objects.profiles
  r.rank_displacement
    .instruct_disp_profiles
                        → p.measurements.rank_displacement.objects.instruct_disp_profiles
    .base_disp_profiles → p.measurements.rank_displacement.objects.base_disp_profiles
  r.base_counterfactual_tokens
                        → p.measurements.lateral_tension_profile
                            .objects.counterfactual_tokens

Output shape matches TASM verbatim; see renderCFTResults for fields.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


def _extract_tasm_shape(p: dict) -> dict:
    """Project a TAGM PromptRecord dict into the flat shape TASM's
    topology-stats function reads. Returns a dict with keys 'ltp',
    'rank_displacement', 'base_counterfactual_tokens', 'tokens',
    'category' matching the fields TASM emitted on each session result.
    """
    meas = p.get("measurements") or {}
    ltp = meas.get("lateral_tension_profile") or {}
    ltp_objs = ltp.get("objects") or {}
    rd = meas.get("rank_displacement") or {}
    rd_objs = rd.get("objects") or {}

    return {
        "tokens": p.get("tokens") or [],
        "category": p.get("category") or "",
        "ltp": {
            "profiles": ltp_objs.get("profiles") or [],
            "base_profiles": ltp_objs.get("base_profiles") or [],
        },
        "rank_displacement": {
            "instruct_disp_profiles":
                rd_objs.get("instruct_disp_profiles") or [],
            "base_disp_profiles":
                rd_objs.get("base_disp_profiles") or [],
        },
        "base_counterfactual_tokens":
            ltp_objs.get("counterfactual_tokens") or [],
    }


def _compute_topology_stats(results: list[dict]) -> dict:
    """Compute aggregate topology statistics.

    Faithful port of TASM's `_compute_topology_stats`. Takes a list of
    TASM-shape per-prompt dicts (use _extract_tasm_shape to project
    TAGM PromptRecords into that shape) and returns the stats dict the
    TASM UI expects.
    """
    stats: dict[str, Any] = {
        "n_total": len(results),
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

    for r in results:
        cat = (r.get("category") or "unknown").lower().strip()
        ltp = r.get("ltp") or {}
        rd = r.get("rank_displacement") or {}
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

        if cat not in stats["by_category"]:
            stats["by_category"][cat] = {
                "count": 0, "mean_tokens": 0, "token_counts": [],
            }
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
                    asymmetry_by_cat[cat].append(
                        (i_mag - b_mag) / total)

    if all_token_counts:
        stats["token_stats"]["total_tokens"] = sum(all_token_counts)
        stats["token_stats"]["mean_tokens_per_prompt"] = round(
            float(np.mean(all_token_counts)), 1)
        stats["token_stats"]["max_tokens"] = max(all_token_counts)

    for cat, info in stats["by_category"].items():
        counts = info.pop("token_counts")
        info["mean_tokens"] = (round(float(np.mean(counts)), 1)
                               if counts else 0)

    if disp_magnitudes_by_cat:
        all_mags: list[float] = []
        per_cat: dict[str, dict] = {}
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
        per_cat_asym: dict[str, dict] = {}
        all_asym: list[float] = []
        for cat, vals in asymmetry_by_cat.items():
            all_asym.extend(vals)
            arr = np.array(vals)
            per_cat_asym[cat] = {
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "instruct_dominant_frac":
                    round(float(np.mean(arr > 0)), 4),
            }
        all_arr = np.array(all_asym)
        stats["asymmetry_stats"] = {
            "overall": {
                "mean": round(float(all_arr.mean()), 4),
                "instruct_dominant_frac":
                    round(float(np.mean(all_arr > 0)), 4),
            },
            "by_category": per_cat_asym,
        }

    return stats


@register_analysis
class CorrectionFieldTopology(AnalysisModule):
    """3D displacement-field topology statistics.

    Ported from TASM's CorrectionFieldTopologyModule. Validates session
    data for the Three.js terrain visualization and computes the
    aggregate stats the UI Topology Summary card displays.
    """

    name = "correction_field_topology"
    display_name = "Correction Field Topology"
    description = (
        "3D displacement-field visualization of the alignment "
        "correction. Dual-bank terrain maps per-token probability "
        "displacement between base and instruct models. Surface height "
        "= displacement magnitude. Instruct bank (warm) shows promoted "
        "candidates; base bank (cool) shows demoted candidates. "
        "Asymmetry reveals where RLHF reshaped the output distribution."
    )
    version = "1.0.0"

    # Soft dependency: LTP profiles OR rank_displacement profiles must
    # be present on at least one prompt. check_dependencies enforces the
    # disjunction.
    depends_on_measurements = ()

    parameters = [
        ModuleParameter(
            name="category",
            display_name="Category Filter",
            description="Filter prompts by category, or show all.",
            kind="select", default="all",
            options=("all", "benign", "mild", "harmful", "jailbreak",
                     "adversarial", "dual-use"),
        ),
        ModuleParameter(
            name="record_limit",
            display_name="Record Limit",
            description=(
                "Maximum prompts to load into the visualization. "
                "Higher = slower but more complete."
            ),
            kind="int", default=100, min_value=1, max_value=2000,
        ),
        ModuleParameter(
            name="token_limit",
            display_name="Token Limit",
            description=(
                "Maximum tokens rendered per prompt terrain. "
                "Higher = more detail but heavier."
            ),
            kind="int", default=20, min_value=4, max_value=100,
        ),
        ModuleParameter(
            name="char_limit",
            display_name="Prompt Label Length",
            description=(
                "Character limit for prompt labels in the viewer "
                "dropdown."
            ),
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
            description=(
                "Auto-rotation speed in revolutions per minute."
            ),
            kind="float", default=0.3, min_value=0.1, max_value=2.0,
        ),
    ]

    def check_dependencies(self, session):
        prompts = session.get("prompts") or []
        if not prompts:
            return [f"Analysis '{self.name}' needs at least one prompt."]
        for p in prompts:
            proj = _extract_tasm_shape(p)
            if (proj["rank_displacement"].get("instruct_disp_profiles")
                    or proj["ltp"].get("profiles")):
                return []
        return [
            f"Analysis '{self.name}' requires displacement or LTP "
            f"profile data on at least one prompt. Re-run analysis "
            f"with rank_displacement and/or lateral_tension_profile "
            f"enabled (both are in the default selection)."
        ]

    def run(self, session, params, probes=None, context=None):
        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={
                "category": params.get("category", "all"),
                "record_limit": params.get("record_limit", 100),
                "token_limit": params.get("token_limit", 20),
                "char_limit": params.get("char_limit", 50),
                "auto_rotate": params.get("auto_rotate", False),
                "rotate_speed": params.get("rotate_speed", 0.3),
            },
        )

        prompts = session.get("prompts") or []
        projected = [_extract_tasm_shape(p) for p in prompts]

        logger.info("[CFT] Analyzing displacement field topology "
                    "(%d prompts)...", len(projected))
        stats = _compute_topology_stats(projected)

        n_usable = (stats["n_with_displacement"]
                    + stats["n_with_ltp_fallback"])
        logger.info(
            "[CFT] Found %d prompts with terrain data "
            "(%d displacement, %d LTP fallback)",
            n_usable, stats["n_with_displacement"],
            stats["n_with_ltp_fallback"])

        if n_usable == 0:
            err = ("No prompts with displacement or LTP profile data. "
                   "Re-run analysis with rank_displacement and/or "
                   "lateral_tension_profile enabled.")
            result.warnings.append(err)
            result.objects["error"] = err
            result.objects.update(stats)
            return result

        stats["launch_params"] = {
            "category": params.get("category", "all"),
            "record_limit": params.get("record_limit", 100),
            "token_limit": params.get("token_limit", 20),
            "char_limit": params.get("char_limit", 50),
            "auto_rotate": params.get("auto_rotate", False),
            "rotate_speed": params.get("rotate_speed", 0.3),
        }

        # Emit TASM wire format under objects; modules_runner's flatten
        # step exposes them at top-level for the UI reader.
        result.objects.update(stats)
        result.scalars["n_total"] = stats["n_total"]
        result.scalars["n_with_displacement"] = stats["n_with_displacement"]
        result.scalars["n_with_ltp_fallback"] = stats["n_with_ltp_fallback"]
        result.scalars["n_with_base_candidates"] = stats[
            "n_with_base_candidates"]
        result.scalars["n_skipped"] = stats["n_skipped"]
        return result
