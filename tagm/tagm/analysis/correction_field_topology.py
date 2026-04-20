"""Correction Field Topology analysis.

Reference module for the TAGM analysis layer. Ported from TASM's
`engine/modules/displacement_field.py` (class
CorrectionFieldTopologyModule). The algorithm in
`_compute_topology_stats` is TASM's, verbatim. Everything else is
TAGM interface scaffolding.

The module validates session data for 3D displacement-field
visualization and computes aggregate topology statistics. The
visualization itself renders client-side via Three.js from the
session record; this module provides the data validation, summary
statistics, and parameter surface for the UI.

Surface height encodes displacement magnitude. Asymmetry between
instruct-bank and base-bank reveals where alignment training
reshaped the output distribution.

Per the TAGM analysis contract (see TAGM_analysis_layer_interface.md):
  - `run()` returns a ModuleOutput; the framework wraps it in the
    mailbox.
  - `_prompt_view()` is the module-local translation from TAGM's
    measurement-keyed prompt record to the TASM-shape dict the
    algorithm reads. This is the only place in the codebase that
    knows this module's field-name mapping.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from tagm.analysis.base import AnalysisModule, ModuleOutput
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")


# ── TASM algorithm (verbatim) ──────────────────────────────────────
# Source: tasm/engine/modules/displacement_field.py
# Behavior unchanged. Consumes the TASM-shape records produced by
# CorrectionFieldTopology._prompt_view below.

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
            stats["by_category"][cat] = {"count": 0, "mean_tokens": 0,
                                          "token_counts": []}
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

    # Asymmetry stats
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


# ── TAGM analysis module ───────────────────────────────────────────

@register_analysis
class CorrectionFieldTopology(AnalysisModule):
    """Reference analysis: aggregate topology stats for the 3D viz."""

    # ── Identity ───────────────────────────────────────────────────
    name = "correction_field_topology"
    display_name = "Correction Field Topology"
    description = (
        "3D displacement-field visualization of the alignment correction. "
        "Dual-bank terrain maps per-token probability displacement between "
        "base and instruct models. Surface height = displacement magnitude. "
        "Instruct bank (warm) shows promoted candidates; base bank (cool) "
        "shows demoted candidates. Asymmetry reveals where RLHF reshaped "
        "the output distribution."
    )
    version = "1.0.0"

    # Disjunctive dependency: each prompt must have at least one of
    # these measurements. The algorithm degrades gracefully between
    # them (RD preferred, LTP fallback). See AnalysisModule.check_dependencies.
    depends_on_measurements = ("rank_displacement", "lateral_tension_profile")

    requires_probe_set = False
    requires_delta_store = False
    requires_pipeline = False

    min_prompts = 1

    parameters = [
        ModuleParameter(
            name="category",
            display_name="Category Filter",
            description="Filter prompts by category, or show all.",
            kind="select",
            default="all",
            options=("all", "benign", "mild", "harmful",
                     "jailbreak", "adversarial", "dual-use"),
        ),
        ModuleParameter(
            name="record_limit",
            display_name="Record Limit",
            description=("Maximum prompts to load into the visualization. "
                         "Higher = slower but more complete."),
            kind="int",
            default=100,
            min_value=1,
            max_value=2000,
        ),
        ModuleParameter(
            name="token_limit",
            display_name="Token Limit",
            description=("Maximum tokens rendered per prompt terrain. "
                         "Higher = more detail but heavier."),
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

    # ── Execution ──────────────────────────────────────────────────
    def run(self, session, params, *, progress,
            probes=None, delta_store=None, pipeline=None) -> ModuleOutput:
        progress("Analyzing displacement field topology...")

        # Translate each TAGM prompt into the TASM-shape dict the
        # algorithm reads. All field-name knowledge lives in
        # _prompt_view below — nowhere else in the module, nowhere
        # else in TAGM.
        records = [self._prompt_view(p)
                   for p in (session.get("prompts") or [])]

        stats = _compute_topology_stats(records)

        n_usable = (stats["n_with_displacement"]
                    + stats["n_with_ltp_fallback"])
        progress(
            f"Found {n_usable} prompts with terrain data "
            f"({stats['n_with_displacement']} displacement, "
            f"{stats['n_with_ltp_fallback']} LTP fallback)")

        # Echo launch params into the output so the UI's "Launch
        # Visualization" button can read them in one shot.
        launch_params = {
            "category":     params["category"],
            "record_limit": params["record_limit"],
            "token_limit":  params["token_limit"],
            "char_limit":   params["char_limit"],
            "auto_rotate":  params["auto_rotate"],
            "rotate_speed": params["rotate_speed"],
        }

        return ModuleOutput(
            scalars={
                "n_total":                stats["n_total"],
                "n_with_displacement":    stats["n_with_displacement"],
                "n_with_ltp_fallback":    stats["n_with_ltp_fallback"],
                "n_with_base_candidates": stats["n_with_base_candidates"],
                "n_skipped":              stats["n_skipped"],
                "total_tokens":           stats["token_stats"]["total_tokens"],
                "mean_tokens_per_prompt": stats["token_stats"]["mean_tokens_per_prompt"],
                "max_tokens":             stats["token_stats"]["max_tokens"],
            },
            objects={
                "by_category":        stats["by_category"],
                "displacement_stats": stats["displacement_stats"],
                "asymmetry_stats":    stats["asymmetry_stats"],
                "launch_params":      launch_params,
            },
            per_prompt={},   # aggregate-only; per-prompt terrain data
                              # lives on the prompts themselves, not here.
        )

    # ── TASM→TAGM translation (module-local, nowhere else) ────────
    @staticmethod
    def _prompt_view(p: dict) -> dict:
        """Present one TAGM prompt as the TASM-shape dict the algorithm
        reads.

        This is the only place in the module (and the only place in
        TAGM) that maps TAGM measurement output names to the TASM
        field names `_compute_topology_stats` expects. If CFT ever
        needed another field, this is where it would be added.

        TAGM sources:
          tokens, category        <- PromptRecord fields
          ltp.profiles            <- measurements.lateral_tension_profile.objects.profiles
          rank_displacement.*     <- measurements.rank_displacement.objects.*
          base_counterfactual_tokens
                                  <- measurements.lateral_tension_profile.objects.counterfactual_tokens
        """
        meas = p.get("measurements") or {}

        # LTP reconstruction
        ltp_m = meas.get("lateral_tension_profile") or {}
        ltp_objects = (ltp_m.get("objects") or {})
        ltp_view = {
            "profiles": ltp_objects.get("profiles") or [],
        }

        # Rank displacement reconstruction
        rd_m = meas.get("rank_displacement") or {}
        rd_objects = (rd_m.get("objects") or {})
        rd_view = {
            "instruct_disp_profiles": rd_objects.get("instruct_disp_profiles") or [],
            "base_disp_profiles":     rd_objects.get("base_disp_profiles") or [],
        }

        # Base-model counterfactual tokens. TAGM stores these on LTP's
        # objects.counterfactual_tokens (the LTP measurement is the one
        # that consults base-side logits for its bank). If LTP wasn't
        # run, base counterfactuals are absent; algorithm handles None
        # correctly.
        base_cf = ltp_objects.get("counterfactual_tokens") or []

        return {
            "tokens":   p.get("tokens") or [],
            "category": p.get("category") or "",
            "ltp":      ltp_view,
            "rank_displacement":         rd_view,
            "base_counterfactual_tokens": base_cf,
        }
