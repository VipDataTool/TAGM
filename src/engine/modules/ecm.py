"""ECM Module — analytical module for Entropic Cascade Mitigation data.

This module analyzes ECM replay data that was collected during the
inference pipeline when the ECM checkbox was enabled. It does NOT run
its own inference — it reads from session results, just like
Comparative Analysis and Token Pair Coupling.

Data source: the ``ecm`` block attached to each result by
``ecm_analysis.attach_ecm_analysis()`` when the compute_ecm checkbox
is checked. Each result's ECM block contains per-channel cascade
detector replays (stress, kl, density) with per-token signal traces,
intervention counts, and peak signal magnitudes.

What this module produces:
  - Per-category and overall aggregate ECM statistics
  - Cross-channel signal correlation analysis
  - Intervention-rate distributions and thresholds
  - Category separability measured by ECM signals
  - Per-record detail summaries with channel breakdowns

The module's parameters control aggregation thresholds and what to
include in the output, NOT detector hyperparameters — those are set in
the Configuration panel and take effect when the ECM checkbox collects
data during analysis.

Architecture note: The ECM checkbox in the control panel is the data
collection feature (like LTP and SFD). This module is the secondary
analytical layer that reads the collected data. The detector
hyperparameters (n_scales, deadband, agreement) in the Configuration
tab control what the checkbox collects; this module only reads and
summarizes what was collected.

Original ECM concept and v2 formulation: Ostrander (2026).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")


# ── Helpers ──────────────────────────────────────────────────────

def _f(v):
    """Safe float conversion; returns None for non-finite or missing."""
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _mean(vals):
    """Mean of a list of numbers, skipping None/NaN."""
    clean = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    return round(sum(clean) / len(clean), 6) if clean else None


def _median(vals):
    clean = sorted(v for v in vals if isinstance(v, (int, float)) and math.isfinite(v))
    if not clean:
        return None
    n = len(clean)
    if n % 2 == 1:
        return clean[n // 2]
    return (clean[n // 2 - 1] + clean[n // 2]) / 2.0


def _std(vals):
    clean = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(clean) < 2:
        return None
    m = sum(clean) / len(clean)
    var = sum((x - m) ** 2 for x in clean) / (len(clean) - 1)
    return round(math.sqrt(var), 6)


def _pct(count, total):
    return round(count / total, 4) if total > 0 else 0.0


# ── Per-result extraction ────────────────────────────────────────

def _extract_ecm_record(result: dict, idx: int) -> Optional[dict]:
    """Extract ECM summary from a single session result.

    Returns None if the result has no ECM data.
    """
    ecm = result.get("ecm")
    if not ecm or not isinstance(ecm, dict):
        return None
    if ecm.get("mode") != "replay":
        return None

    channels = ecm.get("channels", {})
    if not channels:
        return None

    detector = ecm.get("detector", {})
    category = result.get("category", "uncategorized")
    prompt = result.get("prompt", "")

    # Per-channel summaries
    channel_summaries = {}
    total_interventions = 0
    total_tokens = 0
    max_signal_overall = 0.0
    any_fired = False

    for ch_name, ch_data in channels.items():
        n_int = ch_data.get("n_interventions", 0)
        n_tok = ch_data.get("n_tokens", 0)
        max_sig = ch_data.get("max_signal", 0.0)
        mean_sig = ch_data.get("mean_signal", 0.0)
        int_rate = ch_data.get("intervention_rate", 0.0)
        first_idx = ch_data.get("first_signal_idx")

        channel_summaries[ch_name] = {
            "n_interventions": n_int,
            "n_tokens": n_tok,
            "intervention_rate": int_rate,
            "max_signal": max_sig,
            "mean_signal": mean_sig,
            "first_signal_idx": first_idx,
        }
        total_interventions += n_int
        if n_tok > total_tokens:
            total_tokens = n_tok
        if max_sig > max_signal_overall:
            max_signal_overall = max_sig
        if n_int > 0:
            any_fired = True

    # Also pull companion metrics from the result for correlation
    stress = _f(result.get("stress_score"))
    kl = _f(result.get("kl_divergence"))
    entropy = _f(result.get("entropy"))
    sfd = result.get("sfd") or {}
    density_mean = _f(sfd.get("density_mean")) if isinstance(sfd, dict) else None

    return {
        "index": idx,
        "prompt": prompt,
        "category": category,
        "channels": channel_summaries,
        "n_channels": len(channel_summaries),
        "total_interventions": total_interventions,
        "n_tokens": total_tokens,
        "max_signal": round(max_signal_overall, 6),
        "any_fired": any_fired,
        "detector": detector,
        # Companion metrics for correlation analysis
        "stress_score": stress,
        "kl_divergence": kl,
        "entropy": entropy,
        "density_mean": density_mean,
    }


# ── Category aggregation ─────────────────────────────────────────

def _aggregate_category(records: list[dict]) -> dict:
    """Compute aggregate ECM statistics for a group of records."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    n_fired = sum(1 for r in records if r["any_fired"])
    all_max_signals = [r["max_signal"] for r in records]
    all_interventions = [r["total_interventions"] for r in records]

    # Per-channel aggregation
    all_channels = set()
    for r in records:
        all_channels.update(r["channels"].keys())

    channel_agg = {}
    for ch_name in sorted(all_channels):
        ch_rates = []
        ch_max = []
        ch_mean = []
        ch_n_int = []
        ch_fired = 0
        for r in records:
            ch = r["channels"].get(ch_name)
            if ch is None:
                continue
            ch_rates.append(ch["intervention_rate"])
            ch_max.append(ch["max_signal"])
            ch_mean.append(ch["mean_signal"])
            ch_n_int.append(ch["n_interventions"])
            if ch["n_interventions"] > 0:
                ch_fired += 1

        channel_agg[ch_name] = {
            "n_with_data": len(ch_rates),
            "n_fired": ch_fired,
            "fired_frac": _pct(ch_fired, len(ch_rates)),
            "mean_intervention_rate": _mean(ch_rates),
            "median_intervention_rate": _median(ch_rates),
            "mean_max_signal": _mean(ch_max),
            "max_max_signal": max(ch_max) if ch_max else 0.0,
            "mean_mean_signal": _mean(ch_mean),
            "mean_n_interventions": _mean(ch_n_int),
        }

    # Correlation between ECM signal and companion metrics
    correlations = {}
    for metric_name in ("stress_score", "kl_divergence", "entropy", "density_mean"):
        pairs = [(r["max_signal"], r[metric_name])
                 for r in records
                 if r[metric_name] is not None and r["max_signal"] is not None]
        if len(pairs) >= 3:
            xs, ys = zip(*pairs)
            correlations[metric_name] = {
                "n": len(pairs),
                "pearson_r": _pearson(xs, ys),
            }

    return {
        "n": n,
        "n_fired": n_fired,
        "fired_frac": _pct(n_fired, n),
        "mean_max_signal": _mean(all_max_signals),
        "median_max_signal": _median(all_max_signals),
        "std_max_signal": _std(all_max_signals),
        "max_max_signal": round(max(all_max_signals), 6) if all_max_signals else 0.0,
        "mean_total_interventions": _mean(all_interventions),
        "channels": channel_agg,
        "correlations": correlations,
    }


def _pearson(xs, ys):
    """Simple Pearson r for two same-length sequences."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return round(cov / (sx * sy), 4)


# ── Module ───────────────────────────────────────────────────────

class EcmModule(TASMModule):
    """Analytical module for ECM replay data collected during inference.

    Reads the ``ecm`` block from session results (attached by the
    compute_ecm checkbox) and produces aggregate statistics, category
    comparisons, and per-channel breakdowns. Does not run inference.
    """

    name = "ecm"
    display_name = "ECM — Entropic Cascade Mitigation"
    description = (
        "Analyzes ECM cascade-detector replay data collected during "
        "inference (via the ECM checkbox). Produces per-category "
        "intervention statistics, cross-channel signal correlations, "
        "and category separability by ECM measures. Requires session "
        "results analyzed with ECM enabled."
    )
    version = "2.1.0"

    # Operates on collected session data — standard analytical module
    min_results = 1
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="signal_threshold",
            display_name="Signal threshold",
            description=(
                "Minimum max_signal for a record to count as 'fired'. "
                "Records below this are treated as ECM-quiet."
            ),
            type="float", default=0.0, min_val=0.0, max_val=5.0,
        ),
        ModuleParameter(
            name="include_per_record",
            display_name="Include per-record details",
            description=(
                "Include per-record channel summaries in the output. "
                "Disable for very large sessions to keep the results "
                "payload small."
            ),
            type="bool", default=True,
        ),
        ModuleParameter(
            name="strip_token_limit",
            display_name="Strip token limit",
            description=(
                "Maximum tokens rendered per intervention strip. "
                "0 = render the full trace. Truncation is visual only; "
                "statistics and the JSONL always use the full trace."
            ),
            type="int", default=0, min_val=0, max_val=512,
        ),
        ModuleParameter(
            name="include_traces",
            display_name="Include per-token traces",
            description=(
                "Copy per-token signal arrays into the module results "
                "for the in-app visualizer. Disable for very large "
                "sessions."
            ),
            type="bool", default=True,
        ),
    ]

    def __init__(self):
        super().__init__()
        self._session_dir = None

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir

    # ── Validation ──────────────────────────────────────────────

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        # Check that at least some results have ECM data
        n_ecm = sum(1 for r in session_results
                     if r.get("ecm") and isinstance(r["ecm"], dict)
                     and r["ecm"].get("mode") == "replay")
        if n_ecm == 0:
            return False, (
                "No ECM data found in session results. Re-run analysis "
                "with the ECM checkbox enabled to collect cascade "
                "detector replay data."
            )
        return True, "OK"

    # ── Run ─────────────────────────────────────────────────────

    def run(self, session_results, params, progress=None):
        import time as _time

        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[ECM] {msg}")

        t0 = _time.time()
        threshold = float(params.get("signal_threshold", 0.0))
        include_records = bool(params.get("include_per_record", True))
        include_traces = bool(params.get("include_traces", True))
        strip_token_limit = int(params.get("strip_token_limit", 0) or 0)

        # ── Extract ECM records from session ──
        prog("Extracting ECM data from session results...")
        records = []
        for i, result in enumerate(session_results):
            rec = _extract_ecm_record(result, i)
            if rec is not None:
                # Apply threshold override
                if threshold > 0:
                    rec["any_fired"] = rec["max_signal"] >= threshold
                records.append(rec)

        n_total = len(session_results)
        n_ecm = len(records)
        prog(f"Found ECM data in {n_ecm}/{n_total} results")

        if not records:
            return {
                "ok": False,
                "error": "No ECM replay data found in session results.",
                "n_total": n_total,
                "n_ecm": 0,
            }

        # ── Category breakdown ──
        prog("Computing category aggregates...")
        by_cat: dict[str, list] = {}
        for rec in records:
            by_cat.setdefault(rec["category"], []).append(rec)

        categories = {}
        for cat_name in sorted(by_cat.keys()):
            categories[cat_name] = _aggregate_category(by_cat[cat_name])

        overall = _aggregate_category(records)

        # ── Detector config (from first record) ──
        detector_config = records[0].get("detector", {}) if records else {}

        # ── Per-record details (optional) ──
        record_details = None
        if include_records:
            record_details = []
            for rec in records:
                detail = {
                    "index": rec["index"],
                    "prompt": rec["prompt"],
                    "category": rec["category"],
                    "n_tokens": rec["n_tokens"],
                    "max_signal": rec["max_signal"],
                    "any_fired": rec["any_fired"],
                    "total_interventions": rec["total_interventions"],
                    "channels": rec["channels"],
                    "stress_score": rec["stress_score"],
                    "kl_divergence": rec["kl_divergence"],
                    "density_mean": rec["density_mean"],
                }
                # Optionally include the full per-token traces
                if include_traces:
                    src_result = session_results[rec["index"]]
                    ecm_block = src_result.get("ecm", {})
                    ch_data = ecm_block.get("channels", {})
                    traces = {}
                    for ch_name, ch in ch_data.items():
                        sig = ch.get("per_token_signal")
                        if sig:
                            traces[ch_name] = [
                                round(v, 4) if isinstance(v, float) else v
                                for v in sig
                            ]
                    if traces:
                        detail["traces"] = traces
                        # Index-aligned token strings for the strip renderer
                        toks = src_result.get("tokens")
                        if toks:
                            detail["tokens"] = toks
                record_details.append(detail)

        elapsed = round(_time.time() - t0, 2)
        prog(f"Complete: {n_ecm} records, {len(categories)} categories, "
             f"{elapsed}s")

        # ── Persist JSONL ──
        jsonl_path = None
        if self._session_dir and record_details:
            jsonl_path = str(Path(self._session_dir) / "ecm_analysis.jsonl")
            try:
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for detail in record_details:
                        f.write(json.dumps(detail, default=str) + "\n")
                prog(f"Saved {len(record_details)} records to {jsonl_path}")
            except Exception as e:
                logger.warning(f"[ECM] Failed to write JSONL: {e}")
                jsonl_path = None

        return {
            "ok": True,
            "n_total": n_total,
            "n_ecm": n_ecm,
            "n_without_ecm": n_total - n_ecm,
            "signal_threshold": threshold,
            "strip_token_limit": strip_token_limit,
            "detector": detector_config,
            "summary": {
                "overall": overall,
                "categories": categories,
            },
            "records": record_details,
            "n_errors": 0,
            "n_pairs": n_ecm,
            "elapsed_s": elapsed,
            "jsonl_path": jsonl_path,
        }
