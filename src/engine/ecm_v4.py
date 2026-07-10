"""ECM v4 — signal-agnostic cascade detection over pluggable channels.

v2 (src/engine/ecm.py) was never really an *entropy* detector: the EWMA
bank, σ-normalization, deadband, and agreement gate form a general
multi-scale anomaly detector on any scalar trajectory. v4 makes that
explicit:

  CascadeDetector  — the v2 detector math, extracted verbatim, operating
                     on an arbitrary scalar stream. One instance per
                     channel; state identical to v2's (EWMA bank +
                     anomaly-gated EW variance + warmup).
  SignalSource     — produces one scalar per token step.
      EntropySignal   — entropy of the next-token distribution (v2's
                        signal; cheapest channel; logits only).
      DensitySignal   — per-token Spectral Field Density: the effective
                        rank of the current token's activation energy
                        across the alignment-delta subspace, normalized
                        by the layer's global erank (sfd.compute_sfd's
                        per-token body, run online). Hypothesis: smooth-
                        path attacks that keep entropy flat still perturb
                        this trace — collapse (energy concentrating on
                        few delta-directions) registers even when
                        entropy is quiet. Sign convention: the source
                        emits *negative* density so that density
                        COLLAPSE presents to the detector as a RISE,
                        matching the detector's one-sided trigger.
  ECMProcessorV4   — HuggingFace LogitsProcessor pairing each source
                     with its own CascadeDetector and a weight, fusing
                     the per-channel σ-excess signals into a single
                     temperature adjustment. Loop guard and temperature
                     law are v2's, unchanged.

Channel weights:
  weight > 0  — channel participates in fusion (actuates).
  weight == 0 — RECORD-ONLY: the channel is observed and logged in
                diagnostics but never moves the temperature. This is the
                intended first deployment for density: run it alongside
                entropy on benign vs. adversarial prompts and see whether
                the traces separate, before granting it the actuator.

Fusion:
  "max" (default) — the strongest weighted channel drives the response.
                    Conservative-compatible: adding a quiet channel
                    changes nothing.
  "sum"           — weighted sum of channel signals. Compounds; retune
                    gain if you switch.

Displacement channel:
  The third signal from the research harness — per-token divergence
  between instruct (γ=1) and base (γ=0) next-token distributions, whose
  collapse is the hypothesized jailbreak signature — requires a second
  forward pass per token and is not implemented in this runtime path.
  The SignalSource contract accommodates it: a DisplacementSignal would
  observe the two logit sets and emit (negative) KL. See RESEARCH_LEDGER.

Integration mirrors v2: set generate() temperature to 1.0 and pass the
processor via logits_processor. DensitySignal installs forward hooks on
the generation model; call processor.close() (or use it as a context
manager) after generation so hooks never leak. Batch note from v2
applies: per-channel state is shared across the batch — chat (batch=1)
only.

Original concept and v2 formulation: Ostrander (2026).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel
    from src.core.adapter.base import ModelAdapter

logger = logging.getLogger("src")

# ── Fixed internals — MUST match src/engine/ecm.py ──────────────
_VAR_LAMBDA = 0.125      # EW-variance scale for σ estimation (~8-token window)
_STD_FLOOR = 1e-2        # signal units; prevents z-score blowup when σ ≈ 0
_WARMUP_TOKENS = 8       # no interventions before the σ estimate has data
_LOOP_MAX_PERIOD = 8     # detect n-gram loops with period 1..8
_LOOP_MIN_REPEATS = 3    # consecutive periods required to call it a loop


# ── Detector (extracted from ECMProcessor v2, math unchanged) ───

@dataclass
class DetectorStep:
    """One detector update: the raw value in, the state read out."""
    value: float
    signal: float          # σ-excess above deadband; 0 when quiet/warmup
    std: float             # σ estimate used for this step's z-scores
    in_warmup: bool
    z_slopes: list = field(default_factory=list)
    emas: list = field(default_factory=list)


class CascadeDetector:
    """Multi-scale anomaly detector on a scalar trajectory.

    Identical math to ECMProcessor v2 steps 2–3b: a bank of EWMAs at
    dyadic scales; per-scale slopes z-scored against an anomaly-gated
    EW estimate of the stream's own volatility; an agreement gate
    requiring `agreement` scales above `deadband`; signal magnitude is
    the excess of the agreement-th largest slope (the weakest required
    corroborator — conservative by construction).

    Detects one-sided RISES. To detect falls in a quantity, feed its
    negation (see DensitySignal).
    """

    def __init__(self, n_scales: int = 5, deadband: float = 0.75,
                 agreement: int = 2, warmup: int = _WARMUP_TOKENS):
        self.n_scales = n_scales
        self.deadband = max(0.0, float(deadband))
        self.agreement = max(1, min(int(agreement), n_scales))
        self.warmup = max(0, int(warmup))
        # Dyadic decay rates: λ = 0.5, 0.25, ... → windows ≈ 2, 4, 8, ...
        self.lambdas = [0.5 ** (k + 1) for k in range(n_scales)]
        self.reset()

    def reset(self):
        self._emas = [0.0] * self.n_scales
        self._emas_prev = [0.0] * self.n_scales
        self._var_mean = 0.0
        self._var = 0.0
        self._initialized = False
        self._step = 0

    def update(self, value: float) -> DetectorStep:
        """Advance one token step with the channel's scalar observation."""
        # ── EWMA bank ──  (v2 step 2)
        if not self._initialized:
            for k in range(self.n_scales):
                self._emas[k] = value
                self._emas_prev[k] = value
            self._var_mean = value
            self._var = 0.0
            self._initialized = True
        else:
            for k in range(self.n_scales):
                self._emas_prev[k] = self._emas[k]
                lam = self.lambdas[k]
                self._emas[k] = lam * value + (1.0 - lam) * self._emas[k]

        # σ read BEFORE the baseline update; updated only from quiet
        # tokens below — otherwise a sustained climb deflates the very
        # z-scores meant to detect it.  (v2 step 3 preamble)
        std = max(_STD_FLOOR, math.sqrt(max(0.0, self._var)))

        # ── Normalized slopes → agreement-gated signal ──  (v2 step 3)
        z_slopes = [
            (self._emas[k] - self._emas_prev[k]) / std
            for k in range(self.n_scales)
        ]
        ranked = sorted(z_slopes, reverse=True)
        kth = ranked[self.agreement - 1]
        signal = max(0.0, kth - self.deadband)

        in_warmup = self._step < self.warmup
        if in_warmup:
            signal = 0.0

        # ── Anomaly-gated baseline update ──  (v2 step 3b)
        if self._initialized and (in_warmup or signal == 0.0):
            delta = value - self._var_mean
            if not in_warmup:
                delta = max(-3.0 * std, min(3.0 * std, delta))
            self._var_mean += _VAR_LAMBDA * delta
            self._var = (1.0 - _VAR_LAMBDA) * (self._var + _VAR_LAMBDA * delta * delta)

        self._step += 1
        return DetectorStep(
            value=value, signal=signal, std=std, in_warmup=in_warmup,
            z_slopes=z_slopes, emas=list(self._emas),
        )


# ── Signal sources ──────────────────────────────────────────────

class SignalSource:
    """One scalar observation per token step.

    observe() is called once per __call__ of the processor, after the
    model's forward pass for that step (so activation hooks have fired).
    Return None to skip the step (no observation → detector not
    advanced, diagnostics record NaN); return a float otherwise.
    """

    name: str = "signal"

    def observe(self, input_ids, scores) -> Optional[float]:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-generation state. Default: nothing."""

    def close(self) -> None:
        """Release resources (hooks). Default: nothing."""


class EntropySignal(SignalSource):
    """Entropy of the next-token distribution — v2's channel, unchanged.

    Batch note: mean across batch rows, matching v2.
    """

    name = "entropy"

    def observe(self, input_ids, scores) -> float:
        import torch
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1).mean().item()


class DensitySignal(SignalSource):
    """Per-token SFD density, computed online from live activations.

    For the newest token position, projects the hidden state at each
    cached SFD layer onto that layer's delta subspace (V_k), weights by
    singular values, and computes the effective rank of the energy
    distribution normalized by the layer's global erank — exactly the
    per-token body of sfd.compute_sfd, restricted to the current step.

    Emits NEGATIVE density: the detector triggers on rises, and the
    hypothesis of interest is density COLLAPSE (energy concentrating on
    few delta-directions as alignment disengages). Raw density is
    recorded in diagnostics via DetectorStep.value * -1 at read time —
    see ECMProcessorV4.diagnostics_to_dict.

    Owns its hooks: install() registers forward hooks on the generation
    model at the SFD cache's layers; close() removes them. Reads the
    LAST sequence position of each captured activation, which is the
    current token both at prefill (full prompt) and during KV-cached
    decoding (seq dim 1). Batch note: row 0 only, consistent with v2.

    Cost: one k×d matvec per cached layer per token, numpy on CPU.
    """

    name = "density"

    def __init__(self, sfd_cache: dict):
        self._layers = dict(sfd_cache.get("layers", {}))
        self._acts: dict = {}
        self._hooks: list = []

    # ── Hook lifecycle ───────────────────────────────────────────
    def install(self, model: "PreTrainedModel", adapter: "ModelAdapter") -> int:
        """Register hooks at the cached SFD layers. Returns #layers hooked."""
        self.close()
        n_layers = adapter.n_layers(model)

        def _make_hook(layer_idx: int):
            def hook(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                self._acts[layer_idx] = out.detach()
            return hook

        n_hooked = 0
        for li in self._layers:
            if li >= n_layers:
                continue
            try:
                target = adapter.resolve_hook_target(model, "pre_attn_norm", li)
            except (KeyError, AttributeError):
                logger.warning(f"[ECMv4] density: cannot hook layer {li}")
                continue
            self._hooks.append(target.register_forward_hook(_make_hook(li)))
            n_hooked += 1
        if n_hooked == 0:
            logger.warning("[ECMv4] density: no layers hooked — channel will emit None")
        return n_hooked

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._acts.clear()

    def reset(self) -> None:
        self._acts.clear()

    # ── Per-step density ─────────────────────────────────────────
    @staticmethod
    def _density_point(act: np.ndarray, cache) -> Optional[float]:
        """Normalized effective rank of one token's energy in one layer's
        delta subspace. Mirrors sfd.compute_sfd's inner loop."""
        c = cache.V_k @ act
        w = cache.S[: len(c)] * c
        w2 = w * w
        energy = w2.sum()
        if energy < 1e-20:
            return None
        q = w2 / energy
        q_pos = q[q > 1e-10]
        H_t = float(-np.sum(q_pos * np.log(q_pos)))
        erank_t = float(np.exp(H_t))
        return erank_t / cache.erank if cache.erank > 0 else 0.0

    def observe(self, input_ids, scores) -> Optional[float]:
        if not self._acts:
            return None
        total, n_used = 0.0, 0
        for li, cache in self._layers.items():
            act = self._acts.get(li)
            if act is None:
                continue
            # [batch, seq, d] → current token = last position, row 0
            vec = act[0, -1].float().cpu().numpy()
            d = self._density_point(vec, cache)
            if d is not None:
                total += d
                n_used += 1
        if n_used == 0:
            return None
        return -(total / n_used)   # negated: collapse presents as rise


# ── Multi-channel processor ─────────────────────────────────────

@dataclass
class Channel:
    source: SignalSource
    detector: CascadeDetector
    weight: float = 1.0            # 0.0 → record-only (never actuates)


@dataclass
class ChannelDiagnostics:
    per_token_value: list[float] = field(default_factory=list)   # raw signal units
    per_token_signal: list[float] = field(default_factory=list)  # σ-excess
    per_token_std: list[float] = field(default_factory=list)
    n_interventions: int = 0       # steps where this channel's signal > 0
    max_signal: float = 0.0


@dataclass
class V4Diagnostics:
    channels: dict = field(default_factory=dict)   # name → ChannelDiagnostics
    per_token_fused: list[float] = field(default_factory=list)
    per_token_temperature: list[float] = field(default_factory=list)
    per_token_loop: list[bool] = field(default_factory=list)
    n_interventions: int = 0       # steps where the FUSED signal actuated
    n_loop_releases: int = 0
    max_fused_signal: float = 0.0


class ECMProcessorV4:
    """LogitsProcessor fusing multiple cascade channels into one temperature.

    Per token step: every channel observes (entropy from logits, density
    from hooked activations, ...), its own CascadeDetector converts the
    observation into a σ-excess signal, and the weighted per-channel
    signals are fused (max or sum) into the temperature reduction. The
    temperature law, floor, and loop guard are v2's, unchanged — with a
    single entropy channel at weight 1.0 and fusion "max", v4 reproduces
    v2 exactly.

    Use as a context manager or call close() after generation so
    density hooks are removed even on error paths.
    """

    def __init__(
        self,
        channels: list[Channel],
        temperature: float = 0.7,
        gain: float = 0.5,
        floor: float = 0.1,     # ARBITRARY — no derivation (see v2 notes)
        fusion: str = "max",
    ):
        if not channels:
            raise ValueError("ECMProcessorV4 requires at least one channel")
        names = [c.source.name for c in channels]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate channel names: {names}")
        self.channels = channels
        self.base_temperature = temperature
        self.gain = gain
        self.floor = floor
        if fusion not in ("max", "sum"):
            raise ValueError(f"unknown fusion mode: {fusion!r}")
        self.fusion = fusion
        self._nrn_guard = None   # installed via configure_no_repeat()
        self.reset()

    def configure_no_repeat(self, ngram_size: int, prompt_len: int):
        """Install the cooling-gated no-repeat guard for one generation.

        Call after construction, once the prompt length is known and
        before generate(). Replaces the old generate-kwargs
        no_repeat_ngram_size, which was unconditional and included the
        prompt in its window (see ecm_guard module docstring).
        ngram_size < 2 disables the guard.
        """
        from src.engine.ecm_guard import CoolingGatedNoRepeat
        self._nrn_guard = (CoolingGatedNoRepeat(ngram_size, prompt_len)
                           if int(ngram_size or 0) >= 2 else None)

    # ── Lifecycle ────────────────────────────────────────────────
    def reset(self):
        if getattr(self, "_nrn_guard", None) is not None:
            self._nrn_guard.reset()
        for ch in self.channels:
            ch.detector.reset()
            ch.source.reset()
        self._diagnostics = V4Diagnostics(
            channels={ch.source.name: ChannelDiagnostics() for ch in self.channels}
        )

    def close(self):
        for ch in self.channels:
            ch.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ── Loop detection (identical to v2) ─────────────────────────
    @staticmethod
    def _tail_is_periodic(ids: list[int]) -> bool:
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

    # ── Per-token step ───────────────────────────────────────────
    def __call__(self, input_ids: "torch.Tensor", scores: "torch.Tensor") -> "torch.Tensor":
        diag = self._diagnostics

        # ── 1. Observe every channel; advance its detector ───────
        actuating = []   # weighted signals from weight>0 channels
        for ch in self.channels:
            cd = diag.channels[ch.source.name]
            value = ch.source.observe(input_ids, scores)
            if value is None:
                # No observation this step: detector not advanced, so a
                # gap cannot masquerade as a slope when the channel
                # resumes. Recorded as NaN.
                cd.per_token_value.append(float("nan"))
                cd.per_token_signal.append(0.0)
                cd.per_token_std.append(float("nan"))
                continue
            step = ch.detector.update(value)
            cd.per_token_value.append(step.value)
            cd.per_token_signal.append(step.signal)
            cd.per_token_std.append(step.std)
            if step.signal > 0:
                cd.n_interventions += 1
            cd.max_signal = max(cd.max_signal, step.signal)
            if ch.weight > 0.0:
                actuating.append(ch.weight * step.signal)

        # ── 2. Fuse ──────────────────────────────────────────────
        if not actuating:
            fused = 0.0
        elif self.fusion == "max":
            fused = max(actuating)
        else:
            fused = sum(actuating)

        # ── 3. Loop guard (identical to v2) ──────────────────────
        is_loop = False
        if input_ids.shape[0] >= 1 and input_ids.shape[1] >= 2 * _LOOP_MIN_REPEATS:
            tail_len = _LOOP_MAX_PERIOD * _LOOP_MIN_REPEATS
            tail = input_ids[0, -tail_len:].tolist()
            is_loop = self._tail_is_periodic(tail)

        # ── 4. Temperature (identical law to v2) ─────────────────
        if is_loop:
            temp_effective = self.base_temperature
            if fused > 0:
                diag.n_loop_releases += 1
            fused = 0.0
        else:
            temp_effective = self.base_temperature * (1.0 - self.gain * fused)
            temp_effective = max(self.floor, min(self.base_temperature, temp_effective))

        # ── 5. Diagnostics ───────────────────────────────────────
        diag.per_token_fused.append(fused)
        diag.per_token_temperature.append(temp_effective)
        diag.per_token_loop.append(is_loop)
        if fused > 0:
            diag.n_interventions += 1
        diag.max_fused_signal = max(diag.max_fused_signal, fused)

        # ── 6. Cooling-gated no-repeat guard (see ecm_guard) ─────
        # Armed only while cooling (and for n steps after); scans
        # generated tokens only. Quiet generations apply zero bans,
        # keeping the quiet ECM path behaviorally identical to the
        # plain control.
        if self._nrn_guard is not None:
            self._nrn_guard.observe(temp_effective, self.base_temperature)
            if self._nrn_guard.armed:
                banned = self._nrn_guard.banned(input_ids[0].tolist())
                if banned:
                    scores[0, banned] = float("-inf")

        return scores / temp_effective

    # ── Diagnostics access ───────────────────────────────────────
    def get_diagnostics(self) -> V4Diagnostics:
        return self._diagnostics

    def diagnostics_to_dict(self) -> dict:
        d = self._diagnostics
        n_tokens = len(d.per_token_temperature)

        channels = {}
        for ch in self.channels:
            cd = d.channels[ch.source.name]
            values = cd.per_token_value
            # density is stored negated (collapse-as-rise); report raw
            if ch.source.name == "density":
                values = [(-v if v == v else v) for v in values]  # NaN-safe
            channels[ch.source.name] = {
                "per_token_value": values,
                "per_token_signal": cd.per_token_signal,
                "per_token_std": cd.per_token_std,
                "n_interventions": cd.n_interventions,
                "max_signal": round(cd.max_signal, 6),
                "weight": ch.weight,
                "record_only": ch.weight == 0.0,
                "detector": {
                    "n_scales": ch.detector.n_scales,
                    "deadband": ch.detector.deadband,
                    "agreement": ch.detector.agreement,
                    "warmup": ch.detector.warmup,
                },
            }

        out = {
            "version": "v4",
            "mode": "live",     # actuation record — replay audit lives in result["ecm"]
            "channels": channels,
            "per_token_fused_signal": d.per_token_fused,
            "per_token_temperature": d.per_token_temperature,
            "per_token_loop": d.per_token_loop,
            "n_interventions": d.n_interventions,
            "n_loop_releases": d.n_loop_releases,
            "n_ngram_bans": (self._nrn_guard.n_bans
                             if getattr(self, "_nrn_guard", None) else 0),
            "intervention_rate": round(d.n_interventions / n_tokens, 4) if n_tokens else 0.0,
            "max_cascade_signal": round(d.max_fused_signal, 6),
            "n_tokens": n_tokens,
            "config": {
                "base_temperature": self.base_temperature,
                "gain": self.gain,
                "floor": self.floor,
                "fusion": self.fusion,
                "signal_units": "sigma",
            },
        }
        # v2-compatible aliases so existing frontend/coupling tooling
        # (which reads per_token_entropy / per_token_cascade_signal)
        # keeps working against v4 diagnostics.
        ent = channels.get("entropy")
        if ent is not None:
            out["per_token_entropy"] = ent["per_token_value"]
            out["per_token_entropy_std"] = ent["per_token_std"]
        out["per_token_cascade_signal"] = d.per_token_fused
        return out


# ── Config-driven factory ───────────────────────────────────────

def build_processor_from_config(analyzer, temperature: float) -> ECMProcessorV4:
    """Build an ECMProcessorV4 from engine config + a live Analyzer.

    Channels come from `ecm_channels` (comma list). "entropy" needs
    nothing; "density" precomputes (or reuses) the analyzer's SFD cache
    and installs hooks on pipeline.active_model — the caller must
    close() the processor after generation.

    Detector parameters (`ecm_n_scales`, `ecm_deadband`, `ecm_agreement`)
    are shared across channels; per-channel weights come from
    `ecm_<name>_weight`. Density defaults to weight 0.0 — record-only —
    per the measure-before-actuate ordering.
    """
    from src.engine import config as engine_config

    n_scales = int(engine_config.get("ecm_n_scales"))
    deadband = float(engine_config.get("ecm_deadband"))
    agreement = int(engine_config.get("ecm_agreement"))
    warmup = int(engine_config.get("ecm_warmup") or _WARMUP_TOKENS)

    def _detector() -> CascadeDetector:
        return CascadeDetector(n_scales=n_scales, deadband=deadband,
                               agreement=agreement, warmup=warmup)

    requested = [s.strip() for s in
                 str(engine_config.get("ecm_channels") or "entropy").split(",")
                 if s.strip()]

    channels: list[Channel] = []
    for name in requested:
        if name == "entropy":
            w = float(engine_config.get("ecm_entropy_weight"))
            channels.append(Channel(EntropySignal(), _detector(), weight=w))
        elif name == "density":
            w = float(engine_config.get("ecm_density_weight"))
            try:
                if analyzer._sfd_cache is None:
                    from src.engine.sfd import precompute_sfd_cache
                    analyzer._sfd_cache = precompute_sfd_cache(analyzer)
                src = DensitySignal(analyzer._sfd_cache)
                n = src.install(analyzer.pipeline.active_model, analyzer.adapter)
                logger.info(f"[ECMv4] density channel: {n} layers hooked, "
                            f"weight={w}{' (record-only)' if w == 0.0 else ''}")
                channels.append(Channel(src, _detector(), weight=w))
            except Exception as e:
                logger.warning(f"[ECMv4] density channel unavailable: {e}")
        else:
            logger.warning(f"[ECMv4] unknown channel {name!r} — skipped")

    if not channels:
        logger.warning("[ECMv4] no channels resolved — falling back to entropy")
        channels = [Channel(EntropySignal(), _detector(), weight=1.0)]

    return ECMProcessorV4(
        channels=channels,
        temperature=temperature,
        gain=float(engine_config.get("ecm_gain")),
        floor=float(engine_config.get("ecm_floor")),
        fusion=str(engine_config.get("ecm_fusion") or "max"),
    )
