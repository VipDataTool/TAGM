"""Entropic Cascade Mitigation — adaptive sampling via multi-scale entropy tracking.

Monitors the entropy trajectory of the output distribution across multiple
time scales during autoregressive generation. When entropy rises across any
scale (the cascade signature — uncertainty propagating forward rather than
resolving), the processor tightens the output distribution by reducing
effective temperature, biasing toward higher-confidence tokens consistent
with the model's strongest structural priors.

The mechanism:
  1. Compute entropy of the logit distribution at each token step.
  2. Track entropy via a bank of EWMAs at dyadic scales (λ = 0.5^k).
  3. Detect rising-entropy trajectories (positive slope at any scale).
  4. Reduce effective temperature proportionally to the steepest rise.

Design principles:
  - No auxiliary model, no trained discriminator, no new weights.
  - The model's own distributional uncertainty is the only signal.
  - Multi-scale tracking prevents adversarial evasion via fixed-window
    padding (a spike is visible at short scales even when long scales
    are diluted).
  - Adaptive conservatism: tightens only when instability is detected,
    leaves generation unmodified when all scales are stable.
  - Non-agentic: reshapes the probability landscape, does not pursue
    a goal. The model flows through the landscape as before; ECM
    adjusts the terrain.

Integration:
  Designed as a HuggingFace LogitsProcessor. Pass to model.generate()
  via the logits_processor argument. When ECM is active, set the
  generate() temperature to 1.0 and let ECMProcessor handle all
  temperature scaling — this works because processors fire before
  warpers in the HuggingFace pipeline, and temperature=1.0 causes
  the TemperatureLogitsWarper to be skipped entirely (it's the
  multiplicative identity).

Usage:
    from src.engine.ecm import ECMProcessor

    processor = ECMProcessor(
        temperature=0.7,
        n_scales=5,
        gain=2.0,
        floor=0.1,
    )
    outputs = model.generate(
        **inputs,
        temperature=1.0,           # delegate to ECM
        logits_processor=[processor],
    )
    diagnostics = processor.diagnostics_to_dict()

Original concept: Ostrander (2026).
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger("src")


# ── Diagnostics ─────────────────────────────────────────────────

@dataclass
class ECMDiagnostics:
    """Per-generation diagnostic data for downstream analysis.

    Stored in the chat response and optionally persisted to the session
    for A/B comparison with unsteered generations.
    """
    per_token_entropy: list[float] = field(default_factory=list)
    per_token_cascade_signal: list[float] = field(default_factory=list)
    per_token_temperature: list[float] = field(default_factory=list)
    ema_trajectories: dict[int, list[float]] = field(default_factory=dict)
    n_interventions: int = 0     # tokens where cascade_signal > 0
    max_cascade_signal: float = 0.0


# ── Processor ───────────────────────────────────────────────────

class ECMProcessor:
    """LogitsProcessor implementing multi-scale entropy cascade detection.

    Tracks entropy of the logit distribution across K exponentially-weighted
    moving averages at dyadic scales. When any scale shows rising entropy
    (positive EWMA slope), the effective temperature is reduced proportionally
    to the steepest rise — biasing toward higher-confidence tokens.

    The processor is stateful across tokens within a single generation
    but resets between generations via reset().

    Parameters
    ----------
    temperature : float
        Base (ceiling) temperature. ECM tightens below this, never above.
        Should match the user's intended generation temperature.
    n_scales : int
        Number of EWMA scales. Dyadic: effective windows ≈ 2, 4, 8, ...
        Default 5 covers 2-token through 32-token structure.
    gain : float
        Cascade signal → temperature reduction multiplier. Higher values
        produce more aggressive tightening. Start at 2.0 and titrate
        based on intervention frequency in the diagnostics.
    floor : float
        Minimum temperature. ARBITRARY — no first-principles derivation.
        Prevents degenerate collapse (temperature → 0 → deterministic
        argmax → repetitive loops). Tune by watching per_token_temperature
        diagnostics: raise if outputs degenerate, lower if ECM never bites.
    """

    def __init__(
        self,
        temperature: float = 0.7,
        n_scales: int = 5,
        gain: float = 2.0,
        floor: float = 0.1,   # ARBITRARY — no derivation. Tune empirically.
    ):
        self.base_temperature = temperature
        self.n_scales = n_scales
        self.gain = gain
        self.floor = floor

        # Dyadic decay rates: λ = 0.5, 0.25, 0.125, 0.0625, 0.03125
        # Effective windows ≈ 1/λ = 2, 4, 8, 16, 32 tokens
        self.lambdas = [0.5 ** (k + 1) for k in range(n_scales)]

        self.reset()

    def reset(self):
        """Reset all state for a new generation."""
        self._emas = [0.0] * self.n_scales
        self._emas_prev = [0.0] * self.n_scales
        self._initialized = False
        self._step = 0
        self._diagnostics = ECMDiagnostics(
            ema_trajectories={k: [] for k in range(self.n_scales)}
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Called by model.generate() at each token step.

        Args:
            input_ids: [batch, seq_len] token IDs generated so far.
            scores: [batch, vocab_size] raw logits for next token.

        Returns:
            Temperature-scaled logits. Shape unchanged.
        """
        # ── 1. Compute entropy of the current distribution ──────────
        probs = torch.softmax(scores.float(), dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-10))
        # Mean entropy across batch (typically batch=1 for chat)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        # ── 2. Update EWMA bank ────────────────────────────────────
        if not self._initialized:
            # Seed all scales with the first observed entropy so the
            # initial slopes are zero (no false trigger on token 0).
            for k in range(self.n_scales):
                self._emas[k] = entropy
                self._emas_prev[k] = entropy
            self._initialized = True
        else:
            for k in range(self.n_scales):
                self._emas_prev[k] = self._emas[k]
                lam = self.lambdas[k]
                self._emas[k] = lam * entropy + (1.0 - lam) * self._emas[k]

        # ── 3. Compute slopes → cascade signal ────────────────────
        # Slope at each scale: positive = entropy rising at this resolution.
        # Cascade signal = worst-case (steepest rise) across all scales,
        # clamped to non-negative. If all scales are flat or falling,
        # cascade_signal is 0 and ECM does nothing.
        slopes = [
            self._emas[k] - self._emas_prev[k]
            for k in range(self.n_scales)
        ]
        cascade_signal = max(0.0, max(slopes))

        # ── 4. Compute effective temperature ───────────────────────
        # temp_effective = base * (1 - gain * signal), clamped to [floor, base]
        # When signal=0: temp_effective = base (no intervention)
        # When signal>0: temperature drops proportionally
        temp_effective = self.base_temperature * (1.0 - self.gain * cascade_signal)
        temp_effective = max(self.floor, min(self.base_temperature, temp_effective))

        # ── 5. Apply temperature scaling to logits ─────────────────
        scores_out = scores / temp_effective

        # ── 6. Record diagnostics ──────────────────────────────────
        diag = self._diagnostics
        diag.per_token_entropy.append(entropy)
        diag.per_token_cascade_signal.append(cascade_signal)
        diag.per_token_temperature.append(temp_effective)
        for k in range(self.n_scales):
            diag.ema_trajectories[k].append(self._emas[k])
        if cascade_signal > 0:
            diag.n_interventions += 1
        diag.max_cascade_signal = max(diag.max_cascade_signal, cascade_signal)

        self._step += 1
        return scores_out

    # ── Diagnostics access ──────────────────────────────────────

    def get_diagnostics(self) -> ECMDiagnostics:
        """Return the raw diagnostics dataclass from the most recent generation."""
        return self._diagnostics

    def diagnostics_to_dict(self) -> dict:
        """Serialize diagnostics for JSON transport / session storage."""
        d = self._diagnostics
        return {
            "per_token_entropy": d.per_token_entropy,
            "per_token_cascade_signal": d.per_token_cascade_signal,
            "per_token_temperature": d.per_token_temperature,
            "ema_trajectories": {str(k): v for k, v in d.ema_trajectories.items()},
            "n_interventions": d.n_interventions,
            "max_cascade_signal": round(d.max_cascade_signal, 6),
            "n_tokens": len(d.per_token_entropy),
            "scales": {
                str(k): {"lambda": self.lambdas[k],
                          "effective_window": round(1.0 / self.lambdas[k])}
                for k in range(self.n_scales)
            },
            "config": {
                "base_temperature": self.base_temperature,
                "n_scales": self.n_scales,
                "gain": self.gain,
                "floor": self.floor,
            },
        }
