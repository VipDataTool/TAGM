"""
Correction Field Topology Module for TASM.

Validates session data for 3D displacement field visualization and
computes aggregate topology statistics. The visualization itself
renders client-side via Three.js; this module provides the data
validation, summary statistics, and parameter surface for the UI.

The displacement field shows per-token probability displacement
between base and instruct models, decomposed into dual banks:
  - Instruct bank: candidates promoted by alignment training
  - Base bank: candidates demoted by alignment training

Surface height encodes displacement magnitude. Asymmetry between
banks reveals where alignment training reshaped the output distribution.
"""

import logging
import numpy as np
from collections import defaultdict

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")


def _compute_topology_stats(results):
    """Compute aggregate topology statistics from session data.

    Analyzes displacement/LTP profiles to produce summary metrics
    about the correction field's structure across the session.
    """
    stats = {
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

    disp_magnitudes_by_cat = defaultdict(list)
    asymmetry_by_cat = defaultdict(list)
    all_token_counts = []

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

        # Category breakdown
        if cat not in stats["by_category"]:
            stats["by_category"][cat] = {"count": 0, "mean_tokens": 0, "token_counts": []}
        stats["by_category"][cat]["count"] += 1
        stats["by_category"][cat]["token_counts"].append(n_tok)

        # Displacement magnitude statistics
        if has_disp:
            i_profs = rd.get("instruct_disp_profiles") or []
            b_profs = rd.get("base_disp_profiles") or []
            for pos in range(min(n_tok, len(i_profs))):
                ip = i_profs[pos] if pos < len(i_profs) else []
                bp = b_profs[pos] if pos < len(b_profs) else []
                i_mag = sum(v for v in ip if v is not None) if ip else 0
                b_mag = sum(v for v in bp if v is not None) if bp else 0
                disp_magnitudes_by_cat[cat].append(i_mag + b_mag)
                # Asymmetry: positive = instruct-dominant, negative = base-dominant
                total = i_mag + b_mag
                if total > 1e-10:
                    asymmetry_by_cat[cat].append((i_mag - b_mag) / total)

    # Aggregate token stats
    if all_token_counts:
        stats["token_stats"]["total_tokens"] = sum(all_token_counts)
        stats["token_stats"]["mean_tokens_per_prompt"] = round(
            float(np.mean(all_token_counts)), 1)
        stats["token_stats"]["max_tokens"] = max(all_token_counts)

    # Per-category mean tokens
    for cat, info in stats["by_category"].items():
        counts = info.pop("token_counts")
        info["mean_tokens"] = round(float(np.mean(counts)), 1) if counts else 0

    # Displacement magnitude stats
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

    # Asymmetry stats (instruct vs base dominance)
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


class CorrectionFieldTopologyModule(TASMModule):
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

    min_results = 1
    requires_sfd = False
    requires_ltp = False  # can use LTP or RD profiles
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="category",
            display_name="Category Filter",
            description="Filter prompts by category, or show all.",
            type="select",
            default="all",
            options=["all", "benign", "mild", "harmful", "jailbreak",
                     "adversarial", "dual-use"],
        ),
        ModuleParameter(
            name="record_limit",
            display_name="Record Limit",
            description="Maximum prompts to load into the visualization. Higher = slower but more complete.",
            type="int",
            default=100,
            min_val=1,
            max_val=2000,
        ),
        ModuleParameter(
            name="token_limit",
            display_name="Token Limit",
            description="Maximum tokens rendered per prompt terrain. Higher = more detail but heavier.",
            type="int",
            default=20,
            min_val=4,
            max_val=100,
        ),
        ModuleParameter(
            name="char_limit",
            display_name="Prompt Label Length",
            description="Character limit for prompt labels in the viewer dropdown.",
            type="int",
            default=50,
            min_val=20,
            max_val=200,
        ),
        ModuleParameter(
            name="auto_rotate",
            display_name="Auto-Rotate",
            description="Spin terrain around vertical axis on launch.",
            type="bool",
            default=False,
        ),
        ModuleParameter(
            name="rotate_speed",
            display_name="Rotate Speed (RPM)",
            description="Auto-rotation speed in revolutions per minute.",
            type="float",
            default=0.3,
            min_val=0.1,
            max_val=2.0,
        ),
    ]

    def validate(self, session_results, params):
        """Check that at least one prompt has displacement or LTP profile data."""
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        has_data = False
        for r in session_results:
            ltp = r.get("ltp") or {}
            rd = r.get("rank_displacement") or {}
            if (rd.get("instruct_disp_profiles") or
                    (ltp.get("profiles") and len(ltp["profiles"]) > 0)):
                has_data = True
                break

        if not has_data:
            return False, (
                "No displacement or LTP profile data found. "
                "Re-run analysis with LTP and/or Rank Displacement enabled."
            )
        return True, "OK"

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[CFT] {msg}")

        prog("Analyzing displacement field topology...")
        stats = _compute_topology_stats(session_results)

        n_usable = stats["n_with_displacement"] + stats["n_with_ltp_fallback"]
        prog(f"Found {n_usable} prompts with terrain data "
             f"({stats['n_with_displacement']} displacement, "
             f"{stats['n_with_ltp_fallback']} LTP fallback)")

        if n_usable == 0:
            return {
                "error": "No prompts with displacement or LTP profile data.",
                "stats": stats,
            }

        # Include launch parameters in results so the UI can read them
        stats["launch_params"] = {
            "category": params.get("category", "all"),
            "record_limit": params.get("record_limit", 100),
            "token_limit": params.get("token_limit", 20),
            "char_limit": params.get("char_limit", 50),
            "auto_rotate": params.get("auto_rotate", False),
            "rotate_speed": params.get("rotate_speed", 0.3),
        }

        prog("Topology analysis complete. Ready to launch visualization.")
        return stats
