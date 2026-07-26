"""Parity and behavior tests for ecm_v4 — runnable without torch.

1. CascadeDetector parity: fuzz against a pure-python transcription of
   ECMProcessor v2's state recursion (ecm.py steps 2-3b). Any drift in
   the extraction shows up as a signal mismatch.
2. DensitySignal math parity: _density_point vs. sfd.compute_sfd on the
   same activations/cache (compute_sfd is pure numpy).
3. ECMProcessorV4 semantics: v2-equivalence with a single entropy
   channel; record-only channels never actuate; None observations skip
   the detector; loop guard releases; fusion modes.

torch is stubbed with a minimal shim (softmax/log_softmax over a small
vocab) so the processor path runs on CPU-only environments.

Run:  python tools/test_ecm_v4.py
"""
from __future__ import annotations

import math
import random
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Minimal torch shim (enough for EntropySignal + processor) ────


class _T:
    """Tiny tensor wrapper over numpy for the shim."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float64)

    @property
    def shape(self):
        return self.a.shape

    def float(self):
        return _T(self.a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def detach(self):
        return self

    def exp(self):
        return _T(np.exp(self.a))

    def sum(self, dim=-1):
        return _T(self.a.sum(axis=dim))

    def mean(self):
        return _T(self.a.mean())

    def item(self):
        return float(self.a)

    def tolist(self):
        return self.a.astype(int).tolist()

    def __getitem__(self, idx):
        return _T(self.a[idx])

    def __mul__(self, other):
        o = other.a if isinstance(other, _T) else other
        return _T(self.a * o)

    def __neg__(self):
        return _T(-self.a)

    def __truediv__(self, other):
        o = other.a if isinstance(other, _T) else other
        return _T(self.a / o)


def _log_softmax(t, dim=-1):
    x = t.a
    m = x.max(axis=dim, keepdims=True)
    z = x - m
    lse = np.log(np.exp(z).sum(axis=dim, keepdims=True))
    return _T(z - lse)


fake_torch = types.ModuleType("torch")
fake_torch.log_softmax = _log_softmax
fake_torch.Tensor = _T
sys.modules.setdefault("torch", fake_torch)

from src.engine.ecm_v4 import (  # noqa: E402
    CascadeDetector, Channel, DensitySignal, ECMProcessorV4, EntropySignal,
    _STD_FLOOR, _VAR_LAMBDA, _WARMUP_TOKENS,
)


# ── 1. Detector parity vs. v2 transcription ──────────────────────

class V2Reference:
    """Pure-python transcription of ECMProcessor v2 steps 2-3b
    (src/engine/ecm.py lines 252-309), minus torch/entropy/loop/temp."""

    def __init__(self, n_scales=5, deadband=0.75, agreement=2):
        self.n_scales = n_scales
        self.deadband = max(0.0, float(deadband))
        self.agreement = max(1, min(int(agreement), n_scales))
        self.lambdas = [0.5 ** (k + 1) for k in range(n_scales)]
        self._emas = [0.0] * n_scales
        self._emas_prev = [0.0] * n_scales
        self._var_mean = 0.0
        self._var = 0.0
        self._initialized = False
        self._step = 0

    def step(self, entropy):
        if not self._initialized:
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

        entropy_std = max(_STD_FLOOR, math.sqrt(max(0.0, self._var)))
        z = [(self._emas[k] - self._emas_prev[k]) / entropy_std
             for k in range(self.n_scales)]
        kth = sorted(z, reverse=True)[self.agreement - 1]
        signal = max(0.0, kth - self.deadband)
        in_warmup = self._step < _WARMUP_TOKENS
        if in_warmup:
            signal = 0.0
        if self._initialized and (in_warmup or signal == 0.0):
            delta = entropy - self._var_mean
            if not in_warmup:
                delta = max(-3.0 * entropy_std, min(3.0 * entropy_std, delta))
            self._var_mean += _VAR_LAMBDA * delta
            self._var = (1.0 - _VAR_LAMBDA) * (self._var + _VAR_LAMBDA * delta * delta)
        self._step += 1
        return signal, entropy_std


def test_detector_parity():
    rng = random.Random(0)
    for trial in range(50):
        n_scales = rng.choice([3, 5, 7])
        deadband = rng.choice([0.0, 0.5, 0.75, 1.5])
        agreement = rng.randint(1, n_scales)
        det = CascadeDetector(n_scales, deadband, agreement)
        ref = V2Reference(n_scales, deadband, agreement)

        # Trace with regimes: flat, noise, ramp (cascade), spike, decay
        trace = []
        v = rng.uniform(1.0, 5.0)
        for _ in range(200):
            r = rng.random()
            if r < 0.05:
                v += rng.uniform(1.0, 3.0)      # spike
            elif r < 0.25:
                v += rng.uniform(0.05, 0.3)     # ramp
            else:
                v += rng.gauss(0, 0.15)         # jitter
            v = max(0.0, v * rng.uniform(0.97, 1.0))
            trace.append(v)

        for t, x in enumerate(trace):
            step = det.update(x)
            sig_ref, std_ref = ref.step(x)
            assert abs(step.signal - sig_ref) < 1e-12, \
                f"trial {trial} tok {t}: signal {step.signal} != {sig_ref}"
            assert abs(step.std - std_ref) < 1e-12, \
                f"trial {trial} tok {t}: std {step.std} != {std_ref}"
    print("PASS  detector parity (50 fuzzed traces x 200 tokens, bit-exact)")


# ── 2. Density math parity vs. compute_sfd ────────────────────────

def test_density_parity():
    from src.engine.sfd import SFDLayerCache

    rng = np.random.default_rng(1)
    d, k, n_tok = 64, 8, 24
    layers = {}
    for li in (9, 11):
        V_k = rng.normal(size=(k, d)).astype(np.float32)
        S = np.sort(np.abs(rng.normal(size=k)).astype(np.float32))[::-1] + 0.1
        p = (S / S.sum()).astype(np.float64)
        H = float(-np.sum(p * np.log(p)))
        layers[li] = SFDLayerCache(
            V_k=V_k, S=S, erank=float(np.exp(H)), spectral_entropy=H,
            norm_entropy=H / np.log(k), log_volume=float(np.sum(np.log(S))),
            # Renamed from stable_rank/frob_norm: these derive from the
            # TRUNCATED top-k spectrum, so they are not the exact quantities
            # required (see the note on SFDLayerCache in src/engine/sfd.py).
            trunc_stable_rank=float(np.sum(S**2) / S[0]**2),
            trunc_frob_norm=float(np.sqrt(np.sum(S**2))),
        )
    cache = {"layers": layers, "mean_erank": float(np.mean(
        [c.erank for c in layers.values()]))}
    acts = {li: rng.normal(size=(n_tok, d)).astype(np.float32)
            for li in layers}

    # Reference: offline compute_sfd (pure numpy — imports engine.result)
    from src.engine.sfd import compute_sfd
    ref = compute_sfd(acts, cache)

    # Online: DensitySignal._density_point per token, averaged like v4
    sig = DensitySignal(cache)
    for t in range(n_tok):
        vals = [sig._density_point(acts[li][t], layers[li]) for li in layers]
        vals = [v for v in vals if v is not None]
        online = sum(vals) / len(vals)
        assert abs(online - ref.per_token_density[t]) < 1e-9, \
            f"tok {t}: {online} != {ref.per_token_density[t]}"
    print(f"PASS  density parity ({n_tok} tokens x {len(layers)} layers "
          f"vs compute_sfd)")


# ── 3. Processor semantics ────────────────────────────────────────

def _fake_step(vocab_entropy_target, vocab=50, seq_len=20):
    """Build (input_ids, scores) shims. Entropy is controlled by a
    temperature on random logits — approximate, monotone in target."""
    ids = _T(np.arange(seq_len, dtype=np.float64).reshape(1, -1))
    logits = np.random.default_rng(int(vocab_entropy_target * 1e6) % 2**31) \
        .normal(size=(1, vocab)) / max(vocab_entropy_target, 1e-3)
    return ids, _T(logits)


class ScriptedSignal(EntropySignal):
    """Deterministic scalar script; None = no observation."""

    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.i = 0

    def observe(self, input_ids, scores):
        v = self.script[self.i % len(self.script)]
        self.i += 1
        return v

    def reset(self):
        self.i = 0


def test_v2_equivalence_single_entropy_channel():
    """v4 with one entropy channel (weight 1, fusion max) must produce the
    exact temperature trace of the v2 law on the same entropy series."""
    script = [2.0] * 12 + [2.0 + 0.5 * i for i in range(10)] + [7.0] * 8
    src = ScriptedSignal("entropy", script)
    proc = ECMProcessorV4(
        channels=[Channel(src, CascadeDetector(5, 0.75, 2), 1.0)],
        temperature=0.7, gain=0.5, floor=0.1, fusion="max")
    ref = V2Reference(5, 0.75, 2)

    ids = _T(np.arange(40, dtype=np.float64).reshape(1, -1))
    scores = _T(np.zeros((1, 8)))
    for t, e in enumerate(script):
        out = proc(ids, scores)
        sig_ref, _ = ref.step(e)
        temp_ref = max(0.1, min(0.7, 0.7 * (1.0 - 0.5 * sig_ref)))
        temp_v4 = proc.get_diagnostics().per_token_temperature[-1]
        assert abs(temp_v4 - temp_ref) < 1e-12, f"tok {t}: {temp_v4} != {temp_ref}"
        assert np.allclose(out.a, scores.a / temp_ref)
    d = proc.diagnostics_to_dict()
    assert d["per_token_entropy"] == d["channels"]["entropy"]["per_token_value"]
    assert d["per_token_cascade_signal"] == d["per_token_fused_signal"]
    print("PASS  v4 single-entropy channel == v2 temperature law "
          f"({d['n_interventions']} interventions on scripted cascade)")


def test_record_only_never_actuates():
    quiet = ScriptedSignal("entropy", [2.0] * 60)
    loud = ScriptedSignal("density", [-1.0] * 10 + [-1.0 + 0.3 * i for i in range(50)])
    proc = ECMProcessorV4(
        channels=[Channel(quiet, CascadeDetector(), 1.0),
                  Channel(loud, CascadeDetector(), 0.0)],   # record-only
        temperature=0.7, gain=0.5, floor=0.1)
    ids = _T(np.arange(40, dtype=np.float64).reshape(1, -1))
    scores = _T(np.zeros((1, 8)))
    for _ in range(60):
        proc(ids, scores)
    d = proc.diagnostics_to_dict()
    assert d["channels"]["density"]["n_interventions"] > 0, \
        "record-only channel should still DETECT"
    assert d["n_interventions"] == 0, "record-only channel must not ACTUATE"
    assert all(t == 0.7 for t in d["per_token_temperature"])
    assert d["channels"]["density"]["record_only"] is True
    # raw density reported un-negated
    assert d["channels"]["density"]["per_token_value"][0] == 1.0
    print(f"PASS  record-only: density detected "
          f"{d['channels']['density']['n_interventions']} anomalies, "
          f"actuated 0, temperature pinned at base")


def test_none_observation_skips_detector():
    class Gappy(ScriptedSignal):
        def observe(self, input_ids, scores):
            v = super().observe(input_ids, scores)
            return None if self.i % 3 == 0 else v

    src = Gappy("density", [-1.0] * 30)
    proc = ECMProcessorV4(
        channels=[Channel(src, CascadeDetector(), 1.0)],
        temperature=0.7, gain=0.5, floor=0.1)
    ids = _T(np.arange(40, dtype=np.float64).reshape(1, -1))
    scores = _T(np.zeros((1, 8)))
    for _ in range(30):
        proc(ids, scores)
    d = proc.diagnostics_to_dict()
    vals = d["channels"]["density"]["per_token_value"]
    n_nan = sum(1 for v in vals if v != v)
    assert n_nan == 10 and len(vals) == 30
    assert d["n_interventions"] == 0
    print(f"PASS  None observations: {n_nan}/30 gaps recorded as NaN, "
          f"no spurious signals")


def test_loop_guard_releases():
    script = [2.0] * 12 + [6.0] * 10   # cascade while looping
    src = ScriptedSignal("entropy", script)
    proc = ECMProcessorV4(
        channels=[Channel(src, CascadeDetector(), 1.0)],
        temperature=0.7, gain=0.5, floor=0.1)
    # periodic tail: period 2 repeated
    ids = _T(np.array([[7, 3] * 12], dtype=np.float64))
    scores = _T(np.zeros((1, 8)))
    for _ in script:
        proc(ids, scores)
    d = proc.diagnostics_to_dict()
    assert d["n_loop_releases"] > 0
    assert all(t == 0.7 for t in d["per_token_temperature"])
    print(f"PASS  loop guard: {d['n_loop_releases']} releases, "
          f"never cooled into the loop")


def test_fusion_modes():
    a = ScriptedSignal("entropy", [2.0] * 12 + [6.0] * 5)
    b = ScriptedSignal("density", [-1.0] * 12 + [1.0] * 5)
    for mode in ("max", "sum"):
        pa = ScriptedSignal("entropy", a.script)
        pb = ScriptedSignal("density", b.script)
        proc = ECMProcessorV4(
            channels=[Channel(pa, CascadeDetector(), 1.0),
                      Channel(pb, CascadeDetector(), 1.0)],
            temperature=0.7, gain=0.5, floor=0.1, fusion=mode)
        ids = _T(np.arange(40, dtype=np.float64).reshape(1, -1))
        scores = _T(np.zeros((1, 8)))
        for _ in range(17):
            proc(ids, scores)
        d = proc.diagnostics_to_dict()
        fused = d["per_token_fused_signal"][-1]
        s1 = d["channels"]["entropy"]["per_token_signal"][-1]
        s2 = d["channels"]["density"]["per_token_signal"][-1]
        expect = max(s1, s2) if mode == "max" else s1 + s2
        assert abs(fused - expect) < 1e-12
    print("PASS  fusion: max and sum match per-channel signals exactly")


if __name__ == "__main__":
    test_detector_parity()
    test_density_parity()
    test_v2_equivalence_single_entropy_channel()
    test_record_only_never_actuates()
    test_none_observation_skips_detector()
    test_loop_guard_releases()
    test_fusion_modes()
    print("\nAll ecm_v4 tests passed.")
