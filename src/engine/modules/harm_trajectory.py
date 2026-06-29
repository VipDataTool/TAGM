"""
Harm Trajectory Accumulator Module for TAGM.

Computes a cumulative harm-trajectory signal across each prompt's
token sequence using SFD spectral connectivity and LTP correction
strain. The signal accumulates as tokens with specific, harm-adjacent
spectral fingerprints appear in sequence.

The module does not classify prompts as harmful or benign. It produces
a continuous trajectory signal that rises when the token stream enters
regions of concentrated, strained spectral connectivity. What to do
with that signal is a policy decision, not a detection decision.

Requires: SFD and LTP data in session results.

Produces:
  - Per-prompt trajectory curves (signal over token position)
  - Harm fingerprint extracted from labeled session data
  - Separability statistics across prompt categories
  - Per-token contribution breakdown

v0.1 — scalar accumulator on existing session data.
       Uses SFD density and LTP tension magnitude only.
       No spectral direction vectors (see v0.2 roadmap below).

v0.2 roadmap:
  - Persist spectral direction vectors from sfd.py
    (retain c = V_k @ acts[t] in compute_sfd return value)
  - Build directional harm fingerprint (spectral subspace centroids)
  - Direction-aware accumulation
"""

import logging
import numpy as np
from collections import defaultdict

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")


# ── Category sets ───────────────────────────────────────────────
# Lowercase category labels used to partition session results
# into harm and benign groups for fingerprinting.

HARM_CATEGORIES = {"harmful", "jailbreak", "adversarial", "risk", "dangerous"}
BENIGN_CATEGORIES = {"benign", "safe", "neutral"}


class HarmTrajectoryModule(TASMModule):
    name = "harm_trajectory"
    display_name = "Harm Trajectory"
    description = (
        "Cumulative harm-trajectory signal from SFD spectral "
        "connectivity and LTP correction strain. Tracks how the "
        "token stream's spectral fingerprint drifts toward "
        "concentrated, harm-adjacent routing patterns."
    )
    version = "0.1.0"

    min_results = 4
    requires_sfd = True
    requires_ltp = True
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="decay",
            display_name="Decay Factor",
            description=(
                "Exponential decay applied to the running signal. "
                "Higher values retain more history. 1.0 = no decay."
            ),
            type="float",
            default=0.92,
            min_val=0.5,
            max_val=1.0,
        ),
        ModuleParameter(
            name="specificity_ceiling",
            display_name="Specificity Ceiling",
            description=(
                "SFD density value at which specificity score reaches "
                "zero. Tokens with density above this are considered "
                "too broadly connected (versatile) to carry signal. "
                "Tokens below contribute proportionally more."
            ),
            type="float",
            default=0.50,
            min_val=0.2,
            max_val=0.8,
        ),
        ModuleParameter(
            name="strain_cap",
            display_name="Strain Cap",
            description=(
                "Maximum multiplier from LTP tension magnitude. "
                "Caps the influence of extremely strained tokens "
                "to prevent outlier domination."
            ),
            type="float",
            default=3.0,
            min_val=1.0,
            max_val=10.0,
        ),
        ModuleParameter(
            name="fingerprint_mode",
            display_name="Fingerprint Mode",
            description=(
                "How to derive the harm fingerprint. 'session' uses "
                "category labels from this session's results. "
                "'mild_as_benign' treats mild/unknown as benign."
            ),
            type="select",
            default="session",
            options=["session", "mild_as_benign"],
        ),
    ]

    # ── Validation ──────────────────────────────────────────────

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        # Check that we have at least two categories
        cats = {r.get("category", "unknown").lower() for r in session_results}
        has_harm = bool(cats & HARM_CATEGORIES)
        has_benign = bool(cats & BENIGN_CATEGORIES)
        if not has_harm or not has_benign:
            available = ", ".join(sorted(cats))
            return False, (
                f"Need at least one harm and one benign category. "
                f"Found: {available}. Label prompts with categories "
                f"like 'benign', 'harmful', 'jailbreak'."
            )
        return True, "OK"

    # ── Main run ────────────────────────────────────────────────

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[HARM_TRAJ] {msg}")

        decay = params.get("decay", 0.92)
        spec_ceiling = params.get("specificity_ceiling", 0.50)
        strain_cap = params.get("strain_cap", 3.0)
        fp_mode = params.get("fingerprint_mode", "session")

        n = len(session_results)
        prog(f"Processing {n} prompts (decay={decay}, "
             f"ceiling={spec_ceiling}, cap={strain_cap})")

        # ── Phase 1: Extract per-token features ─────────────────

        prompt_features = []
        for i, r in enumerate(session_results):
            sfd = r.get("sfd", {})
            ltp = r.get("ltp", {})
            tokens = r.get("tokens", [])
            category = r.get("category", "unknown")

            densities = sfd.get("per_token_density", [])
            tensions = ltp.get("tension_magnitudes", [])
            kls = r.get("per_token_kl", [])
            stresses = r.get("per_token_stress", [])

            seq_len = len(tokens)
            features = []
            for t in range(seq_len):
                features.append({
                    "token": tokens[t] if t < len(tokens) else "",
                    "sfd_density": (densities[t]
                                    if t < len(densities) else 0.0),
                    "ltp_tension": (tensions[t]
                                   if t < len(tensions) else 0.0),
                    "kl": kls[t] if t < len(kls) else 0.0,
                    "stress": stresses[t] if t < len(stresses) else 0.0,
                })

            prompt_features.append({
                "index": i,
                "prompt": r.get("prompt", ""),
                "category": category,
                "seq_len": seq_len,
                "features": features,
                "sfd_mean": sfd.get("density_mean", 0),
                "sfd_var": sfd.get("density_var", 0),
                "ltp_mean_M": ltp.get("mean_M", 0),
                "ltp_mean_V": ltp.get("mean_V", 0),
            })

        # ── Phase 2: Build harm fingerprint ─────────────────────

        prog("Building harm fingerprint from session labels...")

        harm_densities = []
        harm_tensions = []
        benign_densities = []
        benign_tensions = []

        for pf in prompt_features:
            cat = pf["category"].lower()
            is_harm = cat in HARM_CATEGORIES
            is_benign = cat in BENIGN_CATEGORIES

            if fp_mode == "mild_as_benign" and not is_harm:
                is_benign = True

            for feat in pf["features"]:
                d = feat["sfd_density"]
                m = feat["ltp_tension"]
                if is_harm:
                    harm_densities.append(d)
                    harm_tensions.append(m)
                elif is_benign:
                    benign_densities.append(d)
                    benign_tensions.append(m)

        fingerprint = _build_fingerprint(
            harm_densities, harm_tensions,
            benign_densities, benign_tensions,
        )

        prog(f"Fingerprint: {fingerprint['n_harm_tokens']} harm, "
             f"{fingerprint['n_benign_tokens']} benign tokens")

        # ── Phase 3: Run accumulator ────────────────────────────

        prog("Computing trajectory signals...")

        b_sfd_mean = fingerprint["benign_sfd_mean"]
        b_sfd_std = fingerprint["benign_sfd_std"] or 1e-10
        b_ltp_mean = fingerprint["benign_ltp_mean"]
        b_ltp_std = fingerprint["benign_ltp_std"] or 1e-10

        trajectories = []

        for pi, pf in enumerate(prompt_features):
            running = 0.0
            curve = []
            contributions = []

            for feat in pf["features"]:
                d = feat["sfd_density"]
                m = feat["ltp_tension"]

                # Specificity: low SFD density = specific token =
                # more signal. High density = versatile = less.
                # Linear ramp from 1.0 at density=0 to 0.0 at
                # density=spec_ceiling.
                specificity = max(0.0, 1.0 - d / spec_ceiling)

                # Strain: how much LTP tension relative to benign
                # baseline? Capped to prevent outlier domination.
                strain_z = (m - b_ltp_mean) / (b_ltp_std + 1e-10)
                strain = min(max(0.0, strain_z), strain_cap)

                # SFD deviation: how much more specific (lower
                # density) is this token than the benign average?
                # Positive when token is more specific than typical.
                sfd_dev = (b_sfd_mean - d) / (b_sfd_std + 1e-10)
                sfd_signal = max(0.0, sfd_dev)

                # Combined contribution: specific + strained +
                # deviating from benign = high contribution.
                contribution = specificity * (1.0 + strain) * sfd_signal

                # Accumulate with exponential decay
                running = decay * running + contribution

                curve.append(round(running, 6))
                contributions.append({
                    "token": feat["token"],
                    "specificity": round(specificity, 4),
                    "strain": round(strain, 4),
                    "sfd_signal": round(sfd_signal, 4),
                    "contribution": round(contribution, 4),
                    "cumulative": round(running, 6),
                })

            peak = max(curve) if curve else 0.0
            final = curve[-1] if curve else 0.0

            trajectories.append({
                "index": pi,
                "prompt": pf["prompt"],
                "category": pf["category"],
                "seq_len": pf["seq_len"],
                "trajectory": curve,
                "peak_signal": round(peak, 6),
                "final_signal": round(final, 6),
                "contributions": contributions,
            })

            if (pi + 1) % 50 == 0:
                prog(f"Processed {pi + 1}/{n} prompts...")

        # ── Phase 4: Separability statistics ────────────────────

        prog("Computing separability statistics...")

        category_peaks = defaultdict(list)
        category_finals = defaultdict(list)
        for pt in trajectories:
            cat = pt["category"]
            category_peaks[cat].append(pt["peak_signal"])
            category_finals[cat].append(pt["final_signal"])

        harm_peaks = []
        benign_peaks = []
        for cat, peaks in category_peaks.items():
            if cat.lower() in HARM_CATEGORIES:
                harm_peaks.extend(peaks)
            elif cat.lower() in BENIGN_CATEGORIES:
                benign_peaks.extend(peaks)

        separability = _compute_separability(harm_peaks, benign_peaks)

        # Per-category summary
        category_summary = {}
        for cat in sorted(category_peaks.keys()):
            peaks = category_peaks[cat]
            finals = category_finals[cat]
            category_summary[cat] = {
                "n": len(peaks),
                "peak_mean": round(float(np.mean(peaks)), 6),
                "peak_std": round(float(np.std(peaks)), 6),
                "peak_max": round(float(max(peaks)), 6),
                "peak_min": round(float(min(peaks)), 6),
                "final_mean": round(float(np.mean(finals)), 6),
                "final_std": round(float(np.std(finals)), 6),
            }

        # ── Phase 5: Token-level leaderboard ────────────────────
        # Which individual tokens contributed most across the
        # entire session? This surfaces the "beacon" tokens.

        token_total_contribution = defaultdict(float)
        token_count = defaultdict(int)
        for pt in trajectories:
            for c in pt["contributions"]:
                tok = c["token"].strip()
                if tok:
                    token_total_contribution[tok] += c["contribution"]
                    token_count[tok] += 1

        token_leaderboard = sorted(
            [
                {
                    "token": tok,
                    "total_contribution": round(total, 4),
                    "occurrences": token_count[tok],
                    "mean_contribution": round(
                        total / token_count[tok], 4
                    ),
                }
                for tok, total in token_total_contribution.items()
                if total > 0
            ],
            key=lambda x: x["mean_contribution"],
            reverse=True,
        )[:50]

        d_str = separability.get("cohens_d", "N/A")
        prog(f"Complete. {n} prompts, Cohen's d = {d_str}")

        return {
            "fingerprint": fingerprint,
            "trajectories": trajectories,
            "separability": separability,
            "category_summary": category_summary,
            "token_leaderboard": token_leaderboard,
            "params": {
                "decay": decay,
                "specificity_ceiling": spec_ceiling,
                "strain_cap": strain_cap,
                "fingerprint_mode": fp_mode,
            },
            "n_prompts": n,
        }


# ── Helpers ─────────────────────────────────────────────────────

def _build_fingerprint(harm_d, harm_t, benign_d, benign_t):
    """Build statistical fingerprint from harm/benign token pools."""
    return {
        "harm_sfd_mean": float(np.mean(harm_d)) if harm_d else 0,
        "harm_sfd_std": float(np.std(harm_d)) if harm_d else 1,
        "harm_ltp_mean": float(np.mean(harm_t)) if harm_t else 0,
        "harm_ltp_std": float(np.std(harm_t)) if harm_t else 1,
        "benign_sfd_mean": float(np.mean(benign_d)) if benign_d else 0,
        "benign_sfd_std": float(np.std(benign_d)) if benign_d else 1,
        "benign_ltp_mean": float(np.mean(benign_t)) if benign_t else 0,
        "benign_ltp_std": float(np.std(benign_t)) if benign_t else 1,
        "n_harm_tokens": len(harm_d),
        "n_benign_tokens": len(benign_d),
    }


def _compute_separability(harm_peaks, benign_peaks):
    """Cohen's d and basic separation statistics."""
    if not harm_peaks or not benign_peaks:
        return {"error": "insufficient data for separability"}

    h_mean = float(np.mean(harm_peaks))
    b_mean = float(np.mean(benign_peaks))
    h_std = float(np.std(harm_peaks)) or 1e-10
    b_std = float(np.std(benign_peaks)) or 1e-10
    pooled = np.sqrt((h_std ** 2 + b_std ** 2) / 2)
    d = (h_mean - b_mean) / pooled if pooled > 0 else 0

    # Simple threshold: midpoint between means
    threshold = (h_mean + b_mean) / 2

    # Classification accuracy at midpoint threshold
    correct = (
        sum(1 for p in harm_peaks if p >= threshold) +
        sum(1 for p in benign_peaks if p < threshold)
    )
    total = len(harm_peaks) + len(benign_peaks)
    accuracy = correct / total if total > 0 else 0

    return {
        "harm_peak_mean": round(h_mean, 6),
        "harm_peak_std": round(h_std, 6),
        "benign_peak_mean": round(b_mean, 6),
        "benign_peak_std": round(b_std, 6),
        "cohens_d": round(float(d), 4),
        "midpoint_threshold": round(float(threshold), 6),
        "midpoint_accuracy": round(float(accuracy), 4),
        "n_harm": len(harm_peaks),
        "n_benign": len(benign_peaks),
    }
