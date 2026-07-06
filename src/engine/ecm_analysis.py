"""ECM analysis-mode replay — the extraction-layer counterpart to the
runtime processor.

The ECM's native habitat is generation (ecm_v4.ECMProcessorV4 cooling
the sampler token by token). Analysis mode is a forward pass over given
text, so there is nothing to actuate — but the detector question is
still well-posed: *would* the ECM have fired on this trajectory, where,
and how hard? This module answers it by replaying each available
per-token trace from an analysis result through its own
CascadeDetector, exactly the offline bridge the benchmark harness used,
now attached to session results the way LTP and SFD are.

Channels are whatever the analysis actually extracted:

    stress   — result["per_token_stress"]        (always present)
    kl       — result["per_token_kl"]            (KL checkbox)
    density  — result["sfd"]["per_token_density"] (SFD checkbox; negated
               at the source: collapse presents as a rise to the
               one-sided detector, matching DensitySignal's convention)

Detector hyperparameters come from engine config (ecm_n_scales,
ecm_deadband, ecm_agreement) — the same values the Configuration
panel's ECM controls edit, so the checkbox needs no widgets of its own.

The attached block is honest about what it is:

    result["ecm"] = {
        "mode": "replay",          # detection only; no actuation occurred
        "detector": {...},
        "channels": {
            "<name>": {
                "n_interventions": int,   # steps where signal > 0
                "n_tokens": int,
                "intervention_rate": float,
                "max_signal": float,      # σ-excess units
                "mean_signal": float,     # mean over firing steps
                "first_signal_idx": int | None,
                "per_token_signal": [float, ...],   # index-aligned
            }, ...
        },
    }

Per-token signal arrays stay index-aligned with the token sequence
(gaps recorded as 0.0 without advancing the detector — v4's gap rule,
so a missing observation can never masquerade as a slope), which makes
them drop-in scalar measures for the topology renderer's channel
registry.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger("tasm")


def replay_trace(trace, n_scales: int, deadband: float, agreement: int,
                 negate: bool = False, warmup: int = None) -> dict:
    """Feed one per-token scalar trace through a fresh CascadeDetector.

    Returns the summary block described in the module docstring.
    None/NaN entries are recorded as 0.0 signal without advancing the
    detector (gap rule).
    """
    from src.engine.ecm_v4 import CascadeDetector, _WARMUP_TOKENS

    if warmup is None:
        warmup = _WARMUP_TOKENS
    detector = CascadeDetector(n_scales=n_scales, deadband=deadband,
                               agreement=agreement, warmup=warmup)
    signals = []
    n_observed = 0
    firing = []
    first_idx = None

    for i, value in enumerate(trace or []):
        bad = value is None or (isinstance(value, float) and math.isnan(value))
        if bad:
            signals.append(0.0)
            continue
        step = detector.update(-float(value) if negate else float(value))
        n_observed += 1
        sig = float(step.signal)
        signals.append(round(sig, 6))
        if sig > 0:
            firing.append(sig)
            if first_idx is None:
                first_idx = i

    return {
        "n_interventions": len(firing),
        "n_tokens": n_observed,
        "intervention_rate": (round(len(firing) / n_observed, 4)
                              if n_observed else 0.0),
        "max_signal": round(max(firing), 6) if firing else 0.0,
        "mean_signal": (round(sum(firing) / len(firing), 6)
                        if firing else 0.0),
        "first_signal_idx": first_idx,
        "per_token_signal": signals,
    }


def attach_ecm_analysis(result_dict: dict) -> dict:
    """Attach the ECM replay block to a serialized analysis result.

    Operates on the plain dict produced by result_to_dict — the one
    seam every analysis path (single, batch, deconstruct rungs, rerun)
    flows through — reading whichever per-token traces the enabled
    extraction flags produced. Missing channels are skipped silently:
    the block reports what could be read, not what was wished for.
    """
    from src.engine import config as engine_config

    n_scales = int(engine_config.get("ecm_n_scales"))
    deadband = float(engine_config.get("ecm_deadband"))
    agreement = int(engine_config.get("ecm_agreement"))
    warmup = int(engine_config.get("ecm_replay_warmup"))

    channels = {}

    stress = result_dict.get("per_token_stress")
    if stress:
        channels["stress"] = replay_trace(
            stress, n_scales, deadband, agreement, warmup=warmup)

    kl = result_dict.get("per_token_kl")
    if kl:
        channels["kl"] = replay_trace(kl, n_scales, deadband, agreement,
                                      warmup=warmup)

    sfd = result_dict.get("sfd") or {}
    density = sfd.get("per_token_density") if isinstance(sfd, dict) else None
    if density:
        channels["density"] = replay_trace(
            density, n_scales, deadband, agreement, negate=True,
            warmup=warmup)

    result_dict["ecm"] = {
        "mode": "replay",
        "detector": {
            "n_scales": n_scales,
            "deadband": deadband,
            "agreement": agreement,
            "warmup": warmup,
        },
        "channels": channels,
    }
    return result_dict

