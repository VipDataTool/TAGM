# ECM v2 — signal reformulation

Drop-in changes to Entropic Cascade Mitigation. The v1 detector fired
on any single-scale entropy rise measured in raw nats; per-token entropy
in natural text is intrinsically spiky, so v1 detected spikes, not
cascades, and nats-scale slopes with gain=2.0 pinned the temperature at
the floor. v2 detects the actual cascade signature: entropy rising
coherently across scales, measured against the sequence's own volatility.

## What changed

**Signal pipeline** (`src/engine/ecm.py`)
- Entropy via `log_softmax` (numerically stable; no clamped logs).
- EWMA slopes are normalized to σ-units by a running EW estimate of
  entropy standard deviation (λ=0.125, ~8-token window, floored at
  0.01 nats). The signal is now model- and vocab-independent.
- **Anomaly-gated baseline**: the σ-tracker updates only from tokens
  where the signal is quiet (winsorized to ±3σ), and is frozen while
  the signal fires. Without this, a sustained entropy climb inflates
  the variance estimate and deflates the very z-scores meant to detect
  it — in synthetic tests an unguarded tracker cut the detected signal
  by ~20×. The EMAs still adapt during a freeze, so when the cascade
  resolves the signal releases and baseline learning resumes.
- **Deadband** (`ecm_deadband`, default 0.75σ): slope jitter below this
  never counts. Word-boundary entropy noise lives here. Synthetic
  calibration (Gaussian jitter σ=0.4 vs. a 0.25 nats/token climb,
  gain=0.5, agreement=2):

  | deadband | benign rate | cascade rate | cascade max (σ) | temp min |
  |---|---|---|---|---|
  | 0.30 | 22.3% | 48.6% | 2.76 | 0.100 |
  | 0.50 | 8.0% | 33.3% | 2.09 | 0.100 |
  | **0.75** | **2.0%** | **25.0%** | **1.82** | **0.100** |
  | 1.00 | 0.0% | 19.4% | 1.57 | 0.152 |
  | 1.50 | 0.0% | 0.0% | 0.00 | 0.700 |

  0.75 keeps benign texture essentially free while a genuine cascade
  still drives the temperature to the floor. Real-text jitter has more
  structure than the synthetic — re-check with the tuning protocol.
- **Agreement** (`ecm_agreement`, default 2): at least this many scales
  must exceed the deadband simultaneously before the signal fires. The
  magnitude is the excess of the agreement-th largest slope — the
  weakest required corroborator. Setting agreement=1 reproduces v1's
  any-scale behavior.
- **Warmup**: no interventions for the first 8 tokens (the σ estimate
  is degenerate before it has data).
- **Loop guard**: if the recent token tail is periodic (period 1–8,
  ≥3 consecutive repeats), the processor never cools — a loop is low,
  stable entropy and would otherwise read as "success." Releases are
  counted in `n_loop_releases`.

**Generation backstop** (`src/service/chat.py`)
- `ecm_no_repeat_ngram` (default 4) sets `no_repeat_ngram_size` while
  ECM is active, preventing the sampler from completing n-gram loops
  seeded during cooled steps. Set to 0 to disable.

**Gain semantics changed.** `ecm_gain` is now temperature reduction per
σ of excess signal (default 0.5), not per nat. v1 values do not
transfer. A persisted v1 `ecm_config.json` is detected on load (no
`_ecm_version` field) and its gain is dropped in favor of the v2
default; other keys carry over. The file is rewritten as v2 on the
next save.

**Diagnostics** now include `per_token_entropy_std`, `per_token_loop`,
`n_loop_releases`, `intervention_rate`, and `config.signal_units:
"sigma"` (the v2 marker). The chat badge shows the intervention rate
and loop-release count; the per-turn session summary
(`ecm_summary`) carries both new scalars.

## Tuning protocol

1. Run a benign prompt with ECM on. Check the badge (or
   `intervention_rate` in the diagnostics). Above ~10–15% means the
   detector is tracking texture, not instability → raise `deadband`
   or `agreement`.
2. Run a prompt that reliably destabilizes generation. If ECM never
   bites, lower `deadband` first, then raise `gain`.
3. Watch `frac_at_floor` in the report script. More than ~5% of tokens
   at the floor means the gain is too hot.

## A/B report script

```bash
python tools/ecm_ab_report.py diag_benign.json diag_adversarial.json --out report.png
```

Accepts either bare `ecm_diagnostics` objects or full chat "done"
events. Prints a summary table (intervention rate, fraction at floor,
loop releases, entropy/temperature stats) with tuning warnings, and
renders a three-panel figure per file: entropy + EWMA bank, effective
temperature (loop-guard tokens shaded), and the cascade signal.

## Known interaction

`top_p` runs after the processor, so cooling also shrinks the nucleus;
the effective intervention is stronger than the temperature trace alone
suggests. Intentional, but remember it when tuning.

## Batch caveat (unchanged)

Entropy is averaged across the batch and one shared temperature applies
to all rows. Correct for chat (batch=1). Do not reuse in a batched path
without making the state per-row.
