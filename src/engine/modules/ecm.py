"""ECM Module — analytical module for Entropic Cascade Mitigation data.

This module REPLAYS the cascade detector over the per-token traces in
session results, live at Run time. It does NOT run inference — it reads
collected traces, like Comparative Analysis and Token Pair Coupling.

v3 architecture change (footgun removal): detector hyperparameters
(scales, deadband, agreement, warmup) are MODULE PARAMETERS, applied
when you click Run. Earlier versions read the frozen ``result["ecm"]``
block computed at collection time with Configuration-panel values, so
changing detector settings silently did nothing until a full re-analysis.
Now the workflow is the intended one: run prompts once (traces are
collected), open this module, adjust detector parameters, click Run,
get a fresh analytical set. Iterating on detector settings never
requires re-running inference.

Trace sources (collected during analysis; the module reports coverage
and the reason for any missing channel — nothing is dropped silently):

    stress   — result["per_token_stress"]         (always collected)
    kl       — result["per_token_kl"]             (KL divergence checkbox)
    density  — result["sfd"]["per_token_density"] (SFD collection; negated:
               collapse presents as a rise to the one-sided detector)
    entropy  — ecm_harvest.ecm_diagnostics entropy trace (harvest records
               generated with Cascade Mitigation ON; calibration channel)

"Replay audit" = would the detector fire on these traces, where, how
hard (σ-excess). "Live actuation" = what the runtime controller actually
did during harvest generation. Both are reported; they are different
questions.

Original ECM concept and v2 formulation: Ostrander (2026).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from .base import TASMModule, ModuleParameter
from src.engine.ecm_analysis import replay_trace

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


# ── Trace discovery ──────────────────────────────────────────────
# Each entry: channel name → (extractor, negate, human reason-if-missing)

def _get_stress(result):
    return result.get("per_token_stress")


def _get_kl(result):
    return result.get("per_token_kl")


def _get_density(result):
    sfd = result.get("sfd") or {}
    return sfd.get("per_token_density") if isinstance(sfd, dict) else None


def _get_entropy(result):
    live = (result.get("ecm_harvest") or {}).get("ecm_diagnostics")
    if isinstance(live, dict):
        ent = ((live.get("channels") or {}).get("entropy") or {})
        return ent.get("per_token_value") or live.get("per_token_entropy")
    return None


_CHANNEL_SOURCES = [
    ("stress", _get_stress, False,
     "no per_token_stress trace on this result"),
    ("kl", _get_kl, False,
     "KL divergence checkbox was off when this result was analyzed"),
    ("density", _get_density, True,
     "SFD collection was off when this result was analyzed"),
    ("entropy", _get_entropy, False,
     "no live entropy trace (harvest with Cascade Mitigation ON to record one)"),
]


# ── Per-result extraction: replay at Run time ────────────────────

def _extract_ecm_record(result: dict, idx: int, det: dict) -> Optional[dict]:
    """Replay all available traces for one result through fresh
    CascadeDetectors using the module's detector parameters.

    Returns None only if the result has no replayable traces at all;
    the per-channel missing reasons are recorded either way via the
    returned/accumulated coverage entry on the record.
    """
    category = result.get("category", "uncategorized")
    prompt = result.get("prompt", "")

    channel_summaries = {}
    traces = {}
    missing = {}
    total_interventions = 0
    total_tokens = 0
    max_signal_overall = 0.0
    any_fired = False

    for ch_name, getter, negate, reason in _CHANNEL_SOURCES:
        raw = None
        try:
            raw = getter(result)
        except Exception:
            raw = None
        if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
            missing[ch_name] = reason
            continue
        try:
            block = replay_trace(
                list(raw), det["n_scales"], det["deadband"],
                det["agreement"], negate=negate, warmup=det["warmup"])
        except Exception as e:
            missing[ch_name] = f"replay failed: {e}"
            continue

        sig = block.get("per_token_signal") or []
        traces[ch_name] = [round(float(v), 4) for v in sig]
        n_int = block.get("n_interventions", 0)
        n_tok = block.get("n_tokens", len(sig))
        max_sig = block.get("max_signal", 0.0)

        channel_summaries[ch_name] = {
            "n_interventions": n_int,
            "n_tokens": n_tok,
            "intervention_rate": block.get("intervention_rate", 0.0),
            "max_signal": max_sig,
            "mean_signal": block.get("mean_signal", 0.0),
            "first_signal_idx": block.get("first_signal_idx"),
        }
        total_interventions += n_int
        if n_tok > total_tokens:
            total_tokens = n_tok
        if max_sig > max_signal_overall:
            max_signal_overall = max_sig
        if n_int > 0:
            any_fired = True

    if not channel_summaries:
        return None

    # Live actuation summary (harvest records carry the generation-time
    # diagnostics under ecm_harvest.ecm_diagnostics, mode="live")
    live = None
    harvest = result.get("ecm_harvest")
    if isinstance(harvest, dict):
        hd = harvest.get("ecm_diagnostics")
        if isinstance(hd, dict):
            live = {
                "version": hd.get("version", "v2"),
                "n_interventions": hd.get("n_interventions", 0),
                "intervention_rate": _f(hd.get("intervention_rate")) or 0.0,
                "max_signal": _f(hd.get("max_cascade_signal")) or 0.0,
                "n_tokens": hd.get("n_tokens", 0),
                "n_loop_releases": hd.get("n_loop_releases", 0),
            }

    # Companion metrics from the result for correlation
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
        "live": live,
        "missing_channels": missing,
        "_traces": traces,
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


# ── Module ───────────────────────────────────────────────────────

class EcmModule(TASMModule):
    """Analytical module replaying the cascade detector over session
    traces with Run-time detector parameters. Does not run inference.
    """

    name = "ecm"
    display_name = "ECM — Entropic Cascade Mitigation"
    description = (
        "Replays the cascade detector over the per-token traces in this "
        "session (stress; KL and density when collected; harvest entropy "
        "when available) using the detector settings below — applied "
        "when you click Run, so you can iterate on detector parameters "
        "without re-running analysis. Reports where the detector would "
        "fire and how hard (replay audit), what the live controller did "
        "during harvest generation (live actuation), per-category "
        "separability, and cross-channel correlations."
    )
    version = "3.0.0"

    # Operates on collected session data — standard analytical module
    min_results = 1
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        # ── Detector — the actual cascade-detector math, applied at Run ──
        ModuleParameter(
            name="n_scales",
            display_name="EWMA time scales",
            description=(
                "How many moving averages watch the trace at once, at "
                "doubling time horizons: scale 1 reacts in ~2 tokens, "
                "scale 2 in ~4, scale 3 in ~8, and so on. More scales "
                "extend sensitivity to slower drifts; fewer scales make "
                "the detector a pure spike-catcher. Default 5 covers "
                "~2–32 token horizons."
            ),
            type="int", default=5, min_val=2, max_val=8,
            group="Detector (applied at Run — no re-analysis needed)",
        ),
        ModuleParameter(
            name="deadband",
            display_name="Deadband (σ)",
            description=(
                "Ignore-below threshold, in units of the trace's own "
                "typical volatility (σ). A moving average's per-token "
                "slope must exceed this to count at all. Raise it (e.g. "
                "1.0–1.5) to silence jitter and keep only strong "
                "excursions; lower it (e.g. 0.5) to expose mild "
                "cascades. Directly trades false positives against "
                "sensitivity."
            ),
            type="float", default=0.75, min_val=0.0, max_val=3.0,
            group="Detector (applied at Run — no re-analysis needed)",
        ),
        ModuleParameter(
            name="agreement",
            display_name="Scale agreement",
            description=(
                "How many time scales must clear the deadband "
                "simultaneously before the detector fires. 1 = any "
                "single scale triggers (sensitive, spike-prone). 2+ "
                "= corroboration required across horizons, rejecting "
                "single-scale flukes. The reported σ-excess is the "
                "WEAKEST corroborating scale's — conservative by "
                "construction. Cannot exceed the number of scales."
            ),
            type="int", default=2, min_val=1, max_val=8,
            group="Detector (applied at Run — no re-analysis needed)",
        ),
        ModuleParameter(
            name="warmup",
            display_name="Warmup tokens",
            description=(
                "Tokens at the start of each trace during which the "
                "detector only calibrates (learning the trace's scale "
                "and volatility) and cannot fire. Short prompts need "
                "low warmup to produce any signal at all; response "
                "traces (64 tokens) tolerate the default comfortably. "
                "Live generation always uses 8; replaying with a "
                "different value here is how you isolate warmup "
                "effects."
            ),
            type="int", default=4, min_val=0, max_val=32,
            group="Detector (applied at Run — no re-analysis needed)",
        ),
        # ── Aggregation ──
        ModuleParameter(
            name="signal_threshold",
            display_name="Fired threshold (σ)",
            description=(
                "A record counts as 'fired' only if its peak σ-excess "
                "meets this value. 0 = any nonzero signal counts. "
                "Raise to focus fired-rate statistics on strong "
                "detections only; per-token signals and traces are "
                "unaffected."
            ),
            type="float", default=0.0, min_val=0.0, max_val=5.0,
            group="Aggregation",
        ),
        # ── Output ──
        ModuleParameter(
            name="include_per_record",
            display_name="Per-record details",
            description=(
                "Include every record's channel breakdown and prompt in "
                "the results (and the JSONL export). Disable only for "
                "very large sessions where the payload gets heavy."
            ),
            type="bool", default=True,
            group="Output",
        ),
        ModuleParameter(
            name="include_traces",
            display_name="Per-token signal strips",
            description=(
                "Copy per-token σ-excess arrays into the results so the "
                "in-app strips can render where each channel fired, "
                "token by token. Disable for very large sessions."
            ),
            type="bool", default=True,
            group="Output",
        ),
        ModuleParameter(
            name="strip_token_limit",
            display_name="Strip token limit",
            description=(
                "Maximum tokens rendered per signal strip. 0 = full "
                "trace. Truncation is visual only — statistics and the "
                "JSONL always use the full trace."
            ),
            type="int", default=0, min_val=0, max_val=512,
            group="Output", advanced=True,
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

        n_replayable = 0
        for r in session_results:
            for _, getter, _, _ in _CHANNEL_SOURCES:
                try:
                    t = getter(r)
                except Exception:
                    t = None
                if t is not None and (not hasattr(t, "__len__") or len(t) > 0):
                    n_replayable += 1
                    break
        if n_replayable == 0:
            return False, (
                "No replayable per-token traces found in this session. "
                "Stress traces are collected by every analysis run; if "
                "this is an imported or stripped session, re-run the "
                "prompts. Enable the KL divergence checkbox and SFD "
                "collection for the kl and density channels."
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
        n_scales = int(params.get("n_scales", 5))
        agreement = max(1, min(int(params.get("agreement", 2)), n_scales))
        det = {
            "n_scales": n_scales,
            "deadband": float(params.get("deadband", 0.75)),
            "agreement": agreement,
            "warmup": int(params.get("warmup", 4)),
        }
        threshold = float(params.get("signal_threshold", 0.0))
        include_records = bool(params.get("include_per_record", True))
        include_traces = bool(params.get("include_traces", True))
        strip_token_limit = int(params.get("strip_token_limit", 0) or 0)

        # ── Replay every result's traces with the Run-time detector ──
        prog(f"Replaying traces (scales={det['n_scales']}, "
             f"deadband={det['deadband']}σ, agreement={det['agreement']}, "
             f"warmup={det['warmup']})...")
        records = []
        skipped = []
        coverage: dict[str, dict] = {
            ch: {"available": 0, "missing": 0, "reasons": {}}
            for ch, _, _, _ in _CHANNEL_SOURCES
        }
        n_total = len(session_results)
        for i, result in enumerate(session_results):
            if progress and n_total > 20 and i % 10 == 0:
                prog(f"Replaying {i + 1}/{n_total}...")
            rec = _extract_ecm_record(result, i, det)
            if rec is None:
                skipped.append({
                    "index": i,
                    "prompt": (result.get("prompt") or "")[:80],
                    "reason": "no replayable traces on this result",
                })
                for ch, _, _, reason in _CHANNEL_SOURCES:
                    coverage[ch]["missing"] += 1
                    coverage[ch]["reasons"][reason] = \
                        coverage[ch]["reasons"].get(reason, 0) + 1
                continue
            if threshold > 0:
                rec["any_fired"] = rec["max_signal"] >= threshold
            for ch, _, _, _ in _CHANNEL_SOURCES:
                if ch in rec["channels"]:
                    coverage[ch]["available"] += 1
                else:
                    coverage[ch]["missing"] += 1
                    reason = rec["missing_channels"].get(ch, "unavailable")
                    coverage[ch]["reasons"][reason] = \
                        coverage[ch]["reasons"].get(reason, 0) + 1
            records.append(rec)

        n_ecm = len(records)
        prog(f"Replayed {n_ecm}/{n_total} results")

        if not records:
            return {
                "ok": False,
                "error": "No replayable per-token traces in session results.",
                "n_total": n_total,
                "n_ecm": 0,
                "skipped": skipped,
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

        # ── Live actuation aggregates (harvest records only) ──
        # The replay audit above asks "would the detector fire on the
        # analytical traces"; this reports what the live controller
        # actually did during generation.
        def _agg_live(blocks: list[dict]) -> dict:
            n = len(blocks)
            fired = sum(1 for b in blocks if b["n_interventions"] > 0)
            return {
                "n": n,
                "n_fired": fired,
                "fired_frac": round(fired / n, 4) if n else 0.0,
                "mean_interventions": _mean(
                    [b["n_interventions"] for b in blocks]),
                "mean_intervention_rate": _mean(
                    [b["intervention_rate"] for b in blocks]),
                "max_signal": round(max(
                    (b["max_signal"] for b in blocks), default=0.0), 6),
                "n_loop_releases": sum(
                    b.get("n_loop_releases", 0) for b in blocks),
            }

        live_summary = None
        live_records = [r for r in records if r.get("live")]
        if live_records:
            by_cat_live: dict[str, list] = {}
            for r in live_records:
                by_cat_live.setdefault(r["category"], []).append(r["live"])
            live_summary = {
                "overall": _agg_live([r["live"] for r in live_records]),
                "categories": {c: _agg_live(v)
                               for c, v in sorted(by_cat_live.items())},
            }

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
                    "missing_channels": rec["missing_channels"],
                    "live": rec["live"],
                    "stress_score": rec["stress_score"],
                    "kl_divergence": rec["kl_divergence"],
                    "density_mean": rec["density_mean"],
                }
                if include_traces and rec["_traces"]:
                    detail["traces"] = rec["_traces"]
                    src_result = session_results[rec["index"]]
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
            # Detector settings actually used for THIS replay (the
            # module's Run-time parameters, not collection-time config).
            "detector": det,
            "coverage": coverage,
            "skipped": skipped,
            # "summary" is the REPLAY AUDIT: would the detector fire on
            # the analytical traces of each record's text (mode=replay).
            # "live_summary" is the ACTUATION RECORD: what the controller
            # actually did during harvest generation (mode=live).
            "summary_mode": "replay_audit",
            "summary": {
                "overall": overall,
                "categories": categories,
            },
            "live_summary": live_summary,
            "records": record_details,
            "n_errors": 0,
            "n_pairs": n_ecm,
            "elapsed_s": elapsed,
            "jsonl_path": jsonl_path,
        }
