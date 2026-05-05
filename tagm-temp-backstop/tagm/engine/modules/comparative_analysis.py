"""
Comparative Analysis Module for TASM.

Computes cross-prompt aggregate statistics, category separability,
and batch-level comparative visualizations. This is the primary
analysis module for evaluating alignment signal across a session.

Produces:
  - Bootstrapped per-category metric estimates with 95% CIs
  - Cohen's d effect sizes for safe/risk separation
  - Optimal classification thresholds
  - Available batch visualization plots, with display titles and
    descriptions, ready for the frontend to render
"""

import json
import logging
from pathlib import Path

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")


# ─── Batch plot catalog ──────────────────────────────────────────
# Single source of truth for the batch-level plots this module
# advertises. Each entry is the contract between the renderer in
# tagm/engine/comparative.py (which produces the plot under that key)
# and the frontend (which displays the title/desc as a popout label).
# Order in this list = display order in the UI's popout list.
#
# To add a new batch plot:
#   1. Implement the renderer in tagm/engine/comparative.py
#   2. Add an entry below with its key, human title, and one-line desc
#   3. Make sure the renderer wires up at the corresponding plot key
# No frontend changes needed — titles and descriptions are returned
# from run() and rendered as-is.
BATCH_PLOTS = [
    {"key": "separability",
     "title": "Effect Sizes (Forest)",
     "desc":  "Cohen's d with CIs for proven metrics"},
    {"key": "batch_summary",
     "title": "Category Distributions",
     "desc":  "Strip plots with mean+CI for 4 key metrics"},
    {"key": "key_scatters",
     "title": "Separability Scatters",
     "desc":  "2-panel scatter with confidence ellipses"},
    {"key": "discriminative_sublayers",
     "title": "Discriminative Sublayers",
     "desc":  "Sublayers ranked by adversarial-benign delta"},
    {"key": "proof1_summary",
     "title": "Proof 1 Verification",
     "desc":  "Mathematical exactness of the decomposition"},
    {"key": "exp_trajectory_overlay",
     "title": "Amplitude Trajectories",
     "desc":  "All prompts overlaid across sublayer depth — "
              "category separation visible at middle layers"},
    {"key": "exp_difference_from_benign",
     "title": "Difference from Benign",
     "desc":  "Per-category trajectory minus benign mean"},
    {"key": "exp_metric_scatters",
     "title": "Full Scatter Grid",
     "desc":  "All pairwise metric scatters including weak metrics"},
    {"key": "exp_behavioral_comparison",
     "title": "Behavioral Divergence",
     "desc":  "Instruct vs base probabilities and KL"},
    {"key": "exp_ltp_category_comparison",
     "title": "LTP by Category",
     "desc":  "LTP M,C,V box plots across categories"},
    {"key": "exp_ltp_m_vs_stress",
     "title": "LTP M vs Stress",
     "desc":  "Scatter showing Stress-M anticorrelation"},
    {"key": "exp_ltp_profile_shapes",
     "title": "Profile Shapes",
     "desc":  "Shape distribution across categories"},
    {"key": "exp_sfd_category_comparison",
     "title": "SFD by Category",
     "desc":  "SFD density box plot across categories"},
    {"key": "exp_sfd_vs_asm",
     "title": "SFD vs ASM",
     "desc":  "Scatter: QK density vs ASM middle share"},
    {"key": "exp_rank_displacement",
     "title": "Rank Displacement",
     "desc":  "Kendall tau by category — base vs instruct reordering"},
]


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
            from types import SimpleNamespace
            from tagm.engine.statistics import aggregate_batch

            pr_list = [SimpleNamespace(**r) for r in session_results]
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
        # The frontend needs both the keys (to build URLs) and human
        # titles/descs (to render labels). Send both, in display order,
        # straight from BATCH_PLOTS — the single source of truth.
        plots = list(BATCH_PLOTS)

        # ── Category breakdown ──
        cats = agg.get("categories", {})
        cat_names = sorted(cats.keys())

        prog(f"Complete: {n} prompts, {len(cat_names)} categories, "
             f"{len(plots)} visualizations available")

        return {
            "aggregate": agg,
            "plots": plots,
            "n_prompts": n,
            "categories": cat_names,
            "category_details": cats,
        }
