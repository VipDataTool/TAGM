"""
Comparative Analysis Module for TASM.

Computes cross-prompt aggregate statistics, category separability,
and batch-level comparative visualizations. This is the primary
analysis module for evaluating alignment signal across a session.

Produces:
  - Bootstrapped per-category metric estimates with 95% CIs
  - Cohen's d effect sizes for safe/risk separation
  - Optimal classification thresholds
  - Available batch visualization plot keys
"""

import json
import logging
from pathlib import Path

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")


class ComparativeAnalysisModule(TASMModule):
    name = "comparative_analysis"
    display_name = "Comparative Analysis"
    description = (
        "Cross-prompt aggregate statistics and category separability. "
        "Computes bootstrapped metric estimates, effect sizes, and "
        "optimal classification thresholds across the session. "
        "Generates batch-level comparative visualizations."
    )
    version = "1.0.0"

    min_results = 2
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="force_recompute",
            display_name="Force Recompute",
            description="Recompute aggregate statistics even if cached results exist.",
            type="bool",
            default=False,
        ),
    ]

    def __init__(self):
        super().__init__()
        self._session_dir = None

    def set_session_dir(self, path):
        self._session_dir = Path(path) if path else None

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[COMPARATIVE] {msg}")

        force = params.get("force_recompute", False)
        n = len(session_results)

        # ── Check for cached aggregate stats ──
        agg = None
        agg_path = None
        if self._session_dir:
            agg_path = self._session_dir / "aggregate_statistics.json"
            if not force and agg_path.exists():
                try:
                    with open(agg_path) as f:
                        cached = json.load(f)
                    cached_n = cached.get("n_total", 0)
                    if cached_n == n:
                        agg = cached
                        prog(f"Using cached aggregate ({cached_n} prompts)")
                    else:
                        prog(f"Cache stale ({cached_n} != {n}), recomputing")
                except Exception as e:
                    prog(f"Cache read failed ({e}), recomputing")

        # ── Recompute if needed ──
        if agg is None:
            prog(f"Computing aggregate statistics for {n} prompts...")
            from misc.old.tasm.engine.analyzer import PromptResult
            from misc.old.tasm.engine.statistics import aggregate_batch

            pr_list = [PromptResult.from_dict(r, mode="scalar") for r in session_results]
            agg = aggregate_batch(pr_list)

            # Persist to session directory
            if agg_path:
                try:
                    with open(agg_path, "w") as f:
                        json.dump(agg, f, indent=1, default=str)
                    prog("Cached aggregate statistics to disk")
                except Exception as e:
                    logger.warning(f"[COMPARATIVE] Failed to cache: {e}")

        # ── Determine available batch plots ──
        available_plots = [
            "batch_summary", "separability",
            "key_scatters", "discriminative_sublayers", "proof1_summary",
            "exp_trajectory_overlay", "exp_difference_from_benign",
            "exp_metric_scatters", "exp_behavioral_comparison",
            "exp_ltp_category_comparison", "exp_ltp_m_vs_stress",
            "exp_ltp_profile_shapes", "exp_sfd_category_comparison",
            "exp_sfd_vs_asm", "exp_rank_displacement",
        ]

        # ── Category breakdown ──
        cats = agg.get("categories", {})
        cat_names = sorted(cats.keys())

        prog(f"Complete: {n} prompts, {len(cat_names)} categories, "
             f"{len(available_plots)} visualizations available")

        return {
            "aggregate": agg,
            "plot_keys": available_plots,
            "n_prompts": n,
            "categories": cat_names,
            "category_details": cats,
        }
