"""Entropic Cascade Mitigation — adaptive sampling via multi-scale entropy tracking.

Monitors the entropy trajectory of the output distribution across multiple
time scales during autoregressive generation. When entropy rises *coherently
across scales* (the cascade signature — uncertainty propagating forward
rather than resolving), the processor tightens the output distribution by
reducing effective temperature, biasing toward higher-confidence tokens
consistent with the model's strongest structural priors.

The mechanism (v2):
  1. Compute entropy of the logit distribution at each token step
     (via log_softmax — numerically stable, no clamped logs).
  2. Track entropy via a bank of EWMAs at dyadic scales (λ = 0.5^k).
  3. Normalize each scale's slope by a running estimate of the entropy
     standard deviation (EW variance at a mid scale), so the signal is
     in σ-units and model/vocab independent.
  4. Require *agreement*: the cascade signal is nonzero only when at
     least `agreement` scales show a normalized rise above `deadband`.
     A spike visible at one scale only (ordinary word-boundary jitter)
     is free. The signal magnitude is the excess of the agreement-th
     largest slope — the weakest required corroborator — which keeps
     the response conservative.
  5. Reduce effective temperature proportionally to that excess.
  6. Loop guard: if the recent token tail is periodic (an n-gram loop),
     never cool — cooling into a loop entrenches it. Temperature is
     released to base and the event is recorded in diagnostics.

Why v1's formulation was replaced:
  v1 used max(slopes) in raw nats. Per-token entropy in natural text is
  intrinsically spiky (huge after "the", near-zero after "Barack"), so
  the fastest scale registered a positive slope on roughly half of all
  healthy tokens, and nats-scale slopes with gain=2.0 pinned the
  temperature at the floor. v1 detected spikes; v2 detects cascades.

Design principles:
  - No auxiliary model, no trained discriminator, no new weights.
  - The model's own distributional uncertainty is the only signal.
  - Multi-scale tracking prevents adversarial evasion via fixed-window
    padding (a spike is visible at short scales even when long scales
    are diluted); the agreement rule prevents the converse failure of
    triggering on every short-scale fluctuation.
  - Adaptive conservatism: tightens only when instability is corroborated
    across scales, leaves generation unmodified otherwise.
  - Non-agentic: reshapes the probability landscape, does not pursue
    a goal. The model flows through the landscape as before; ECM
    adjusts the terrain.

Interaction with top-p:
  Sampling warpers (top_p) run *after* this processor, so cooling the
  distribution also shrinks the nucleus. The effective intervention is
  therefore somewhat stronger than the temperature trace alone suggests.
  This compounding is intentional but worth remembering when tuning.

Integration:
  Designed as a HuggingFace LogitsProcessor. Pass to model.generate()
  via the logits_processor argument. When ECM is active, set the
  generate() temperature to 1.0 and let ECMProcessor handle all
  temperature scaling — this works because custom processors fire before
  warpers in the HuggingFace pipeline, and temperature=1.0 causes
  the TemperatureLogitsWarper to be skipped entirely (it's the
  multiplicative identity).

  Batch note: entropy is averaged across the batch and a single shared
  temperature is applied to every row. Correct for chat (batch=1).
  Do NOT reuse in a batched path without making the state per-row —
  one unstable sequence would throttle its batchmates.

Usage:
    from src.engine.ecm import ECMProcessor

    processor = ECMProcessor(
        temperature=0.7,
        n_scales=5,
        gain=0.5,        # per σ of excess signal (v2 units)
        floor=0.1,
        deadband=0.75,   # σ-units; jitter below this is free
        agreement=2,     # scales that must corroborate
    )
    outputs = model.generate(
        **inputs,
        temperature=1.0,           # delegate to ECM
        logits_processor=[processor],
    )
    diagnostics = processor.diagnostics_to_dict()

Original concept: Ostrander (2026). v2 signal formulation follows
review recommendations (multi-scale agreement, σ-normalization,
deadband, loop guard).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch

logger = logging.getLogger("src")

# ── Fixed internals (not exposed to the UI to limit knob sprawl) ──
_VAR_LAMBDA = 0.125      # EW-variance scale for σ estimation (~8-token window)
_STD_FLOOR = 1e-2        # nats; prevents z-score blowup when σ ≈ 0
_WARMUP_TOKENS = 8       # no interventions before the σ estimate has data
_LOOP_MAX_PERIOD = 8     # detect n-gram loops with period 1..8
_LOOP_MIN_REPEATS = 3    # consecutive periods required to call it a loop


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
    per_token_entropy_std: list[float] = field(default_factory=list)
    per_token_loop: list[bool] = field(default_factory=list)
    ema_trajectories: dict[int, list[float]] = field(default_factory=dict)
    n_interventions: int = 0     # tokens where cascade_signal > 0
    n_loop_releases: int = 0     # tokens where the loop guard forced base temp
    max_cascade_signal: float = 0.0


# ── Processor ───────────────────────────────────────────────────

class ECMProcessor:
    """LogitsProcessor implementing multi-scale entropy cascade detection.

    Tracks entropy of the logit distribution across K exponentially-weighted
    moving averages at dyadic scales. Slopes are normalized to σ-units by a
    running EW estimate of entropy standard deviation. When at least
    `agreement` scales show a normalized rise above `deadband`, the effective
    temperature is reduced proportionally to the excess of the agreement-th
    largest slope — biasing toward higher-confidence tokens.

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
        Temperature reduction per σ of excess cascade signal. v2 signal
        is z-scored, so this is model- and vocab-independent. Start at
        0.5 and titrate using intervention_rate in the diagnostics.
        (v1's gain operated on raw nats — old values do not transfer.)
    floor : float
        Minimum temperature. ARBITRARY — no first-principles derivation.
        Prevents degenerate collapse (temperature → 0 → deterministic
        argmax → repetitive loops). Tune by watching per_token_temperature
        diagnostics: raise if outputs degenerate, lower if ECM never bites.
    deadband : float
        σ-units of slope below which a scale does not count as rising.
        Ordinary word-boundary entropy jitter lives below ~0.75σ; raise
        this if intervention_rate on benign prompts exceeds ~10-15%.
    agreement : int
        Number of scales that must simultaneously exceed the deadband
        for the cascade signal to fire. 1 reproduces v1's any-scale
        behavior (spike detection); 2-3 requires cross-scale coherence
        (cascade detection). Clamped to [1, n_scales].
    """

    def __init__(
        self,
        temperature: float = 0.7,
        n_scales: int = 5,
        gain: float = 0.5,
        floor: float = 0.1,   # ARBITRARY — no derivation. Tune empirically.
        deadband: float = 0.75,
        agreement: int = 2,
    ):
        self.base_temperature = temperature
        self.n_scales = n_scales
        self.gain = gain
        self.floor = floor
        self.deadband = max(0.0, float(deadband))
        self.agreement = max(1, min(int(agreement), n_scales))

        # Dyadic decay rates: λ = 0.5, 0.25, 0.125, 0.0625, 0.03125
        # Effective windows ≈ 1/λ = 2, 4, 8, 16, 32 tokens
        self.lambdas = [0.5 ** (k + 1) for k in range(n_scales)]

        self.reset()

    def reset(self):
        """Reset all state for a new generation."""
        self._emas = [0.0] * self.n_scales
        self._emas_prev = [0.0] * self.n_scales
        self._var_mean = 0.0     # EW mean for the variance tracker
        self._var = 0.0          # EW variance of entropy
        self._initialized = False
        self._step = 0
        self._diagnostics = ECMDiagnostics(
            ema_trajectories={k: [] for k in range(self.n_scales)}
        )

    # ── Loop detection ──────────────────────────────────────────

    @staticmethod
    def _tail_is_periodic(ids: list[int]) -> bool:
        """True if the tail of ids is an n-gram loop.

        A loop with period p requires the last p tokens to equal the
        previous p tokens, _LOOP_MIN_REPEATS times consecutively.
        p=1 catches "the the the"; larger p catches phrase cycles.
        """
        n = len(ids)
        for p in range(1, _LOOP_MAX_PERIOD + 1):
            need = p * _LOOP_MIN_REPEATS
            if n < need:
                break
            block = ids[n - p:]
            if all(
                ids[n - (r + 1) * p: n - r * p] == block
                for r in range(1, _LOOP_MIN_REPEATS)
            ):
                return True
        return False

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
        # log_softmax is numerically stable; no clamped logs needed.
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        probs = log_probs.exp()
        # Mean entropy across batch (typically batch=1 for chat —
        # see the batch note in the module docstring before reusing
        # this in any batched path).
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        # ── 2. Update EWMA bank + EW variance tracker ───────────────
        if not self._initialized:
            # Seed all scales with the first observed entropy so the
            # initial slopes are zero (no false trigger on token 0).
            for k in range(self.n_scales):
                self._emas[k] = entropy
                self._emas_prev[k] = entropy
            self._var_mean = entropy
            self._var = 0.0
            self._initialized = True
        else:
            for k in range(self.n_scales):
                self._emas_prev[k] = self._emas[k]
                lam = self.lambdas[k]
                self._emas[k] = lam * entropy + (1.0 - lam) * self._emas[k]

        # σ baseline is read BEFORE the update and only updated from
        # tokens the detector considers normal (see step 3b below) —
        # otherwise a sustained entropy climb bleeds into the variance
        # estimate and deflates the very z-scores meant to detect it.
        entropy_std = max(_STD_FLOOR, math.sqrt(max(0.0, self._var)))

        # ── 3. Normalized slopes → agreement-gated cascade signal ──
        # Slope at each scale in σ-units: positive = entropy rising at
        # this resolution, measured against the sequence's own volatility.
        z_slopes = [
            (self._emas[k] - self._emas_prev[k]) / entropy_std
            for k in range(self.n_scales)
        ]
        # The cascade signal fires only when at least `agreement` scales
        # exceed the deadband. Its magnitude is the excess of the
        # agreement-th largest slope (the weakest required corroborator),
        # which is conservative by construction. If fewer scales agree,
        # the signal is 0 and ECM does nothing.
        ranked = sorted(z_slopes, reverse=True)
        kth = ranked[self.agreement - 1]
        cascade_signal = max(0.0, kth - self.deadband)

        # Warmup: the σ estimate is meaningless for the first few tokens
        # (variance seeded at 0 → z-scores blow up). Record diagnostics
        # but do not intervene.
        in_warmup = self._step < _WARMUP_TOKENS
        if in_warmup:
            cascade_signal = 0.0

        # ── 3b. Anomaly-gated baseline update ──────────────────────
        # EW variance (West-style recursion) at a fixed mid scale.
        # During warmup: always update (learn the sequence's natural
        # scale). After warmup: update only when the signal is quiet,
        # with the innovation winsorized to ±3σ. While the signal fires,
        # the baseline is frozen — the EMAs still adapt, so when the
        # cascade resolves the signal releases and learning resumes.
        if self._initialized and (in_warmup or cascade_signal == 0.0):
            delta = entropy - self._var_mean
            if not in_warmup:
                delta = max(-3.0 * entropy_std, min(3.0 * entropy_std, delta))
            self._var_mean += _VAR_LAMBDA * delta
            self._var = (1.0 - _VAR_LAMBDA) * (self._var + _VAR_LAMBDA * delta * delta)

        # ── 4. Loop guard ───────────────────────────────────────────
        # Cooling into an n-gram loop entrenches it (a loop is low,
        # stable entropy — it looks like "success" to the detector).
        # If the recent tail is periodic, force base temperature.
        is_loop = False
        if input_ids.shape[0] >= 1 and input_ids.shape[1] >= 2 * _LOOP_MIN_REPEATS:
            tail_len = _LOOP_MAX_PERIOD * _LOOP_MIN_REPEATS
            tail = input_ids[0, -tail_len:].tolist()
            is_loop = self._tail_is_periodic(tail)

        # ── 5. Compute effective temperature ───────────────────────
        # temp_effective = base * (1 - gain * signal), clamped to [floor, base]
        # When signal=0: temp_effective = base (no intervention)
        # When signal>0: temperature drops proportionally to σ-excess
        if is_loop:
            temp_effective = self.base_temperature
            if cascade_signal > 0:
                self._diagnostics.n_loop_releases += 1
            cascade_signal = 0.0
        else:
            temp_effective = self.base_temperature * (1.0 - self.gain * cascade_signal)
            temp_effective = max(self.floor, min(self.base_temperature, temp_effective))

        # ── 6. Apply temperature scaling to logits ─────────────────
        scores_out = scores / temp_effective

        # ── 7. Record diagnostics ──────────────────────────────────
        diag = self._diagnostics
        diag.per_token_entropy.append(entropy)
        diag.per_token_cascade_signal.append(cascade_signal)
        diag.per_token_temperature.append(temp_effective)
        diag.per_token_entropy_std.append(entropy_std)
        diag.per_token_loop.append(is_loop)
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
        n_tokens = len(d.per_token_entropy)
        return {
            "per_token_entropy": d.per_token_entropy,
            "per_token_cascade_signal": d.per_token_cascade_signal,
            "per_token_temperature": d.per_token_temperature,
            "per_token_entropy_std": d.per_token_entropy_std,
            "per_token_loop": d.per_token_loop,
            "ema_trajectories": {str(k): v for k, v in d.ema_trajectories.items()},
            "n_interventions": d.n_interventions,
            "n_loop_releases": d.n_loop_releases,
            "intervention_rate": round(d.n_interventions / n_tokens, 4) if n_tokens else 0.0,
            "max_cascade_signal": round(d.max_cascade_signal, 6),
            "n_tokens": n_tokens,
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
                "deadband": self.deadband,
                "agreement": self.agreement,
                "signal_units": "sigma",   # v2 marker — v1 was raw nats
            },
        }
