# Build Log — ECM v4: Signal-Agnostic Cascade Detection

**Date:** 2026-07-04
**Scope:** Extract the v2 entropy-cascade detector into a signal-agnostic core, define a pluggable `SignalSource` abstraction, implement the SFD density channel as a live runtime signal, and wire the multi-channel processor into the chat generation path.
**Files touched:** `src/engine/ecm_v4.py` (new), `src/engine/config.py`, `src/service/chat.py`, `src/app.py`, `tools/test_ecm_v4.py` (new).
**Status:** All parity and behavior tests pass. One item deliberately left unverified (live hook path against a real model — see §9).

---

## 1. Motivating unification

The v2 processor (`src/engine/ecm.py`) presents itself as an *entropy* detector, but nothing in its detection machinery is entropy-specific. The EWMA bank at dyadic scales, the σ-normalization against an anomaly-gated running variance, the deadband, and the cross-scale agreement gate together form a general multi-scale anomaly detector over *any* scalar trajectory. Entropy is merely the scalar v2 happens to feed it.

The research thread that led here identified two additional per-token scalars already computed elsewhere in TAGM:

- **SFD density** (`sfd.compute_sfd`) — the effective rank of a token's activation energy projected into the alignment-delta subspace, normalized by the layer's global erank. Hypothesis: smooth-path attacks that keep output entropy provably flat still perturb this trace (density *collapse* — energy concentrating on few delta-directions — as alignment disengages).
- **Rank displacement / instruct-vs-base divergence** (`sfd.compute_rank_displacement`) — a per-token meter of how much alignment is load-bearing; hypothesized jailbreak signature is *displacement collapse*.

v4 makes the unification explicit: one detector class, N channels, one actuator. Entropy becomes the cheapest of three channels rather than the architecture.

**Why build this now rather than a bespoke density detector:** the v2 detector's design properties (conservatism via the agreement-th-largest slope, robustness via σ-units, evasion resistance via multi-scale tracking) were argued for and tuned once. Re-deriving them per signal would fork the math and the tuning. Extracting the detector means every future channel inherits the v2 guarantees for free, and Propositions 3–7 from the v2 analysis hold for whichever scalar is plugged in.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │  ECMProcessorV4  (HF LogitsProcessor)         │
                    │                                              │
 logits ──────────▶│  EntropySignal ──▶ CascadeDetector ──┐        │
                    │                                     │ weight │
 hooked            │  DensitySignal ──▶ CascadeDetector ──┤──fuse──┼──▶ temperature
 activations ─────▶│   (−density)                        │ (max/  │    (v2 law,
                    │                                     │  sum)  │     v2 loop guard)
                    │  [DisplacementSignal — future] ──────┘        │
                    └──────────────────────────────────────────────┘
```

Components, in the new `src/engine/ecm_v4.py`:

| Component | Role |
|---|---|
| `CascadeDetector` | v2 steps 2–3b extracted verbatim: EWMA bank, σ-normalization, deadband, agreement gate, warmup, anomaly-gated baseline update. `update(value) → DetectorStep`. |
| `SignalSource` | Contract: one scalar per token step via `observe(input_ids, scores)`; `None` = no observation. Plus `reset()`/`close()` lifecycle. |
| `EntropySignal` | v2's entropy computation, unchanged (log_softmax → −Σp·logp, batch-mean). |
| `DensitySignal` | Online per-token SFD. Owns its own forward hooks. Emits **negative** density. |
| `Channel` | `(source, detector, weight)`. `weight == 0.0` ⇒ record-only. |
| `ECMProcessorV4` | Observes all channels, fuses weighted signals, applies v2's temperature law and loop guard, records per-channel + fused diagnostics. |
| `build_processor_from_config` | Factory reading engine config; resolves channels, precomputes/reuses the analyzer's SFD cache, installs density hooks on `pipeline.active_model`. |

---

## 3. Decisions and reasons

### 3.1 Leave `ecm.py` (v2) completely untouched

**Decision:** v4 lives in a new module; v2 remains byte-identical.
**Reason:** v2 is the published baseline. The ablation harness (`tools/ecm_ablation_v2.py`), the A/B report, and the coupling analysis all reference v2 behavior. Any refactor that routes v2 through new code — however "equivalent" — invalidates the comparison baseline and forces re-running ablations to trust it. Instead, equivalence is *proven externally*: a v4 processor configured with a single entropy channel at weight 1.0 reproduces v2's temperature trace bit-exactly (test §8.3). v2 can be retired later on evidence, not on refactoring faith.
**Cost accepted:** ~120 lines of duplicated detector math between `ecm.py` and `ecm_v4.py`, guarded by the parity fuzz test. If the two ever drift, the test catches it.

### 3.2 One `CascadeDetector` instance *per channel*, not shared

**Decision:** each channel gets its own detector state.
**Reason:** the σ-normalization is the point of the design — signals are measured against *the sequence's own volatility in that signal*. Entropy lives in nats with word-boundary spikes; density lives in [0, ~1] with entirely different noise structure. Sharing a variance tracker across them would let the noisier signal inflate σ and deafen the quieter one. Independent state also keeps the deadband semantics meaningful per channel ("0.75σ of *this* signal's jitter").
**Consequence:** detector hyperparameters (`n_scales`, `deadband`, `agreement`) are currently *shared values but independent state*. Per-channel hyperparameters are a config extension away if density turns out to need a different deadband; I did not add those knobs preemptively (see 3.9).

### 3.3 Density is negated at the source

**Decision:** `DensitySignal.observe()` returns `−density`; diagnostics un-negate at serialization time.
**Reason:** the detector is deliberately one-sided — it triggers on *rises* (v2's cascade = entropy rising coherently across scales). The density hypothesis is about *collapse*. Three options existed: (a) make the detector two-sided, (b) add a per-channel sign flag in the processor, (c) negate in the source. Option (a) doubles the detector's false-positive surface and changes v2 semantics for the entropy channel; option (b) spreads sign logic across two classes. Option (c) keeps the detector's contract pure ("detects rises, full stop") and localizes the domain knowledge (*collapse is the anomaly*) in the one class that knows what density means. The serialization layer reports raw density so humans reading diagnostics never see the trick.

### 3.4 Record-only channels: `weight == 0.0` observes but never actuates

**Decision:** a zero-weight channel runs its detector and logs everything, but is excluded from fusion. Density **defaults to 0.0**.
**Reason:** this is the measure-before-actuate ordering from the research plan made structural rather than procedural. The density-collapse hypothesis is exactly that — a hypothesis. Coupling an unvalidated signal to the actuator risks a detector that fires on benign prose and quietly degrades generation quality, which is worse than no detector because it erodes trust in the whole mechanism. With record-only defaults, flipping `ecm_version` to v4 changes *nothing* about generation behavior (entropy channel reproduces v2) while producing paired entropy/density traces on every generation. Once benign vs. adversarial traces demonstrably separate, `ecm_density_weight` is raised — a one-config change, no code.
**Design detail:** record-only channels still increment their own `n_interventions` counter (they *detect*), but never the fused counter (they don't *actuate*). The test suite asserts both directions.

### 3.5 `None` observations skip the detector entirely

**Decision:** if a source returns `None` (e.g., hooks missed a step), the channel's detector is **not advanced**; the step is logged as NaN.
**Reason:** the alternative — advancing the detector with a filled value (last value, zero, mean) — creates artificial slopes. Worse, a *gap followed by resumption* would present the detector with a jump between the pre-gap EMA state and the post-gap value, which is precisely the rise signature it hunts. Freezing detector state across gaps means the EMA bank resumes from where the signal actually was, and a hook hiccup cannot masquerade as a cascade. NaN in the value trace (rather than omission) keeps all per-token arrays index-aligned with the temperature trace, which downstream tooling (`ecm_coupling.py` tail-alignment) depends on.

### 3.6 Fusion is `max` by default

**Decision:** fused signal = max over weighted per-channel signals; `sum` available by config.
**Reason:** `max` is conservative-compatible in the same sense as v2's agreement-th-largest-slope rule: adding a quiet channel changes nothing, and the intervention magnitude is always attributable to one identifiable channel (diagnosable). `sum` compounds — two channels each below their individual trigger threshold could jointly actuate — which is a legitimate design (weak multi-channel corroboration) but changes the meaning of `gain` and would require retuning. Defaulting to the mode that preserves v2's tuning is the lower-risk choice; `sum` is left available for the research harness.

### 3.7 `DensitySignal` owns its hooks; cleanup lives in the generation thread's `finally`

**Decision:** the source installs/removes its own forward hooks rather than reusing the analyzer's `ActivationCapture`; `chat.py` closes the processor inside `_generate()`'s `finally`.
**Reasons:**
- *Not reusing `ActivationCapture`:* the analyzer's capture installs hooks for signal layers + attention + final norm + optionally full trajectory — far more than density needs — and its lifecycle is tied to `analyze_prompt`, which runs *after* generation in the chat flow. Density needs exactly the SFD cache's layers, hooked for the duration of one `model.generate()` call. A ~25-line dedicated hook set is simpler than threading generation-lifetime semantics through a class designed for single-forward-pass analysis.
- *Cleanup location:* the streaming architecture runs `model.generate()` in a worker thread while the main thread yields SSE events. If the client disconnects, the SSE generator can be abandoned by the framework — any cleanup placed after the streaming loop may never run. The worker thread, however, always finishes its `try/finally` because `generate()` runs to completion (or raises) regardless of the client. Placing `ecm_processor.close()` there covers success, error, and disconnect paths. Leaked hooks would otherwise fire on every subsequent forward pass of the model — analysis passes included — silently corrupting the density channel's next generation *and* wasting compute.
- *Thread-safety note:* hooks write `self._acts` from the generation thread, and the LogitsProcessor reads it from the same thread (processors run inside `generate()`). No cross-thread access, no locking needed. `MODEL_LOCK` already serializes generation against analysis, so hooks cannot fire during an interleaved analysis pass.

### 3.8 Current-token extraction: always the last sequence position, row 0

**Decision:** `act[0, -1]`.
**Reason:** two regimes occur during `generate()`. The prefill forward passes the full prompt — activations are `[batch, prompt_len, d]`, and the distribution the processor is about to shape belongs to the *last* prompt position. Every subsequent step under KV cache passes only the new token — `[batch, 1, d]` — where the last position is trivially the current one. `[-1]` is correct in both without branching. Row 0 mirrors v2's documented batch limitation (state shared across batch; chat is batch=1); the module docstring carries the same warning forward.

### 3.9 Config surface kept minimal; v2 remains the default

**Decision:** five new keys (`ecm_version`, `ecm_channels`, `ecm_entropy_weight`, `ecm_density_weight`, `ecm_fusion`), detector params shared with v2's existing keys; `ecm_version` defaults to `"v2"`.
**Reason:** the codebase has an explicit anti-knob-sprawl stance (`ecm.py`: "Fixed internals (not exposed to the UI to limit knob sprawl)"). Per-channel detector hyperparameters, per-channel gains, and density-layer overrides were all considered and deferred — each is a one-line addition *when evidence demands it*, whereas speculative knobs are permanent tuning burden. Defaulting to v2 means merging this change is behaviorally inert: nobody's running system changes until they opt in.

### 3.10 Diagnostics keep v2-compatible aliases

**Decision:** v4's `diagnostics_to_dict()` emits `per_token_entropy`, `per_token_entropy_std` (from the entropy channel), and `per_token_cascade_signal` (= fused) alongside the new `channels` structure.
**Reason:** three consumers read v2's diagnostic keys today: the frontend summary in `app.py` (reads `n_interventions`, `n_tokens`, `max_cascade_signal`, `intervention_rate` — all preserved at top level), the coupling tool (`ecm_coupling.py` correlates `per_token_entropy`/`per_token_cascade_signal` against analyzer ridges), and the ablation reporter. Breaking those for a schema aesthetic would force lockstep updates across tools that are themselves part of the evidence chain. The aliases cost a few redundant bytes per response and buy zero-migration compatibility; the `"version": "v4"` marker lets consumers distinguish when they care.

### 3.11 Displacement channel: specified, not implemented

**Decision:** the third signal (instruct-vs-base divergence, γ=1 vs γ=0) is documented in the module header with its contract sketch but not built.
**Reason:** it requires a second forward pass per token — a structural change to the generation loop (double compute, or a hybrid partial-delta approximation whose fidelity is itself an open question flagged in the research notes: the delta store holds projection deltas for q,k only, so γ=0 is an *approximate* base). That is a research-harness feature with its own validation burden, not a runtime patch. Bolting it in now would couple this change's correctness to an unvalidated approximation. The `SignalSource` contract was shaped so it drops in later: a `DisplacementSignal` observing two logit sets and emitting −KL (collapse-as-rise, same convention as density) requires no processor changes.

---

## 4. Runtime data flow (one token step, v4 active, channels = entropy + density)

1. `model.generate()` runs the forward pass for the step. Density hooks fire, stashing `[batch, seq, d]` hidden states for each SFD layer into `DensitySignal._acts`.
2. HF calls `ECMProcessorV4.__call__(input_ids, scores)`.
3. `EntropySignal.observe` computes distribution entropy from `scores`. Its detector updates → σ-excess signal `s_e`.
4. `DensitySignal.observe` projects each hooked layer's last-position activation onto that layer's `(V_k, S)`, computes normalized effective rank, averages across layers, returns the negation. Its detector updates → `s_d`.
5. Fusion: `fused = max(w_e·s_e, w_d·s_d)` over channels with `w > 0`. With the default `w_d = 0`, `fused = s_e` exactly — v2 behavior.
6. Loop guard (unchanged from v2): periodic tail ⇒ force base temperature, count a release, zero the signal.
7. Temperature law (unchanged): `T = clamp(base·(1 − gain·fused), floor, base)`; return `scores / T`.
8. Per-channel values/signals/σ, fused signal, temperature, and loop flags are appended to diagnostics.

Per-token added cost of the density channel: one `k×d` matvec per cached layer plus a `d`-vector device→CPU copy — negligible next to the forward pass. No second forward pass anywhere in this path.

---

## 5. Integration changes

**`src/engine/config.py`** — new keys with in-file rationale comments; `ecm_density_weight: 0.0` carries the record-only reasoning inline so the next reader doesn't flip it casually.

**`src/service/chat.py`** —
- `generate_chat_response_streaming(..., analyzer=None)`: optional keyword so every existing call site remains valid.
- Version dispatch: `ecm_version == "v4"` *and* analyzer present ⇒ factory; otherwise v2 with an explicit warning if v4 was requested but the analyzer is missing (fail loud in logs, degrade gracefully in behavior).
- Hook cleanup in `_generate()`'s `finally` (reason in 3.7), guarded by `hasattr(ecm_processor, "close")` so the v2 path is untouched.
- Final log line rewritten against the serialized dict rather than dataclass attributes, because `V4Diagnostics` has no `per_token_entropy` attribute — the old code would have raised `AttributeError` on the v4 path. Version-tagged: `[ECM v4] 3/128 tokens intervened…`.

**`src/app.py`** — one line: the chat route passes `state.analyzer` through. The route's phase-2 analysis block reads only preserved top-level diagnostic keys, so it needed no changes.

---

## 6. The SFD cache dependency

`build_processor_from_config` reuses `analyzer._sfd_cache` if present, else calls `precompute_sfd_cache(analyzer)` and stores it back on the analyzer. Reasons: (a) the cache is a per-model-load artifact (SVDs of `[ΔW_Q; ΔW_K]` per layer) already cached for the analysis path — recomputing per generation would add seconds of SVD to the first chat message for no benefit; (b) storing it back means the analysis path also benefits if chat warmed it first; (c) `clear_caches()` on model reload already invalidates it, so staleness is handled by existing machinery. The factory touches the private `_sfd_cache` attribute directly, matching how `analyzer._compute_sfd` itself manages it; if that grates, promoting a `get_sfd_cache()` accessor on `Analyzer` is a trivial follow-up.

Note the cache-selection subtlety inherited from config: with `sfd_use_signal_layers: False` (default), density hooks land on `sfd_layer_start..end` (default 9–16), not the analyzer's middle-third signal layers. That is the same layer set the offline SFD uses, so online and offline traces are directly comparable — which is the property the validation experiment needs.

---

## 7. Testing strategy under an offline constraint

The build environment has no network and no torch. Rather than ship untested, the module was structured so the *math* is pure Python/numpy (torch appears only inside `observe`/`__call__` runtime paths, imported lazily), making the core fully testable here:

- **Reference-transcription parity:** `V2Reference` in the test file is a line-by-line pure-python transcription of `ecm.py` steps 2–3b (the state recursion). `CascadeDetector` is fuzzed against it — 50 random hyperparameter draws × 200-token traces containing flat/jitter/ramp/spike/decay regimes — asserting *bit-exact* (`<1e-12`) agreement on both signal and σ at every step. This is the strongest available evidence that the extraction changed nothing.
- **Torch shim:** a ~60-line fake `torch` module (numpy-backed tensors with `log_softmax`, `exp`, `sum`, indexing) lets `EntropySignal` and the full `ECMProcessorV4.__call__` path run unmodified.
- **`compute_sfd` as its own oracle:** the offline SFD implementation is pure numpy, so the online `_density_point` is tested directly against it on random caches/activations (agreement `<1e-9` per token across layers).

---

## 8. Test results

`python tools/test_ecm_v4.py` — all pass, first run:

| # | Test | Assertion | Result |
|---|---|---|---|
| 8.1 | Detector parity | signal & σ bit-exact vs v2 transcription, 50×200 fuzz | PASS |
| 8.2 | Density parity | online per-token density == `compute_sfd`, 24 tok × 2 layers | PASS |
| 8.3 | v2 equivalence | single entropy channel reproduces v2 temperature law exactly on a scripted cascade (17 interventions); aliases mirror channel data | PASS |
| 8.4 | Record-only | density detected 49 anomalies, actuated 0; temperature pinned at base; raw (un-negated) values in diagnostics | PASS |
| 8.5 | Gap handling | 10/30 `None` steps recorded as NaN; zero spurious signals on resumption | PASS |
| 8.6 | Loop guard | 10 releases during a scripted cascade over a period-2 tail; never cooled into the loop | PASS |
| 8.7 | Fusion | `max` and `sum` match per-channel signals exactly | PASS |

Additionally: `py_compile` clean on all four touched source files; new config keys resolve with correct defaults.

---

## 9. Known limitations / not verified here

1. **Live hook path untested against a real model.** The adapter resolution (`resolve_hook_target(model, "pre_attn_norm", li)`), tensor shapes under a real KV cache, and dtype/device behavior were verified by reading `hooks.py`'s identical usage, not by execution. **Required smoke test on the GPU box:** set `ecm_version: v4`, generate once, confirm `channels.density.per_token_value` is populated and plausibly tracks an offline SFD run on the same text (tail-aligned — ECM covers generated positions only).
2. **Batch = 1 assumption carried forward** from v2, now in two places (batch-mean entropy, row-0 density). Documented in the module docstring; a batched path needs per-row detector state, same as v2's caveat.
3. **Detector hyperparameters shared across channels.** Density's natural volatility may want its own deadband; deferred pending trace data (3.2, 3.9).
4. **Density hooks observe the *active* model.** If `inference_class` is toggled to the base model, density is measured on base activations projected into the delta subspace — geometrically meaningful but a different quantity than the instruct-side hypothesis. Not guarded against; flag if base-model chat + v4 becomes a real workflow.
5. **Displacement channel absent by design** (3.11).

---

## 10. Immediate next steps

1. GPU smoke test per §9.1.
2. Run the existing benign/adversarial prompt sets through v4 with density record-only; export diagnostics.
3. Extend `ecm_coupling.py` (or a sibling) to correlate the density trace against entropy and against analyzer ridge positions — the Option-B separation question: *does density move where entropy is blind?*
4. If separation holds: titrate `ecm_density_weight` upward with the same intervention-rate discipline used for v2's gain.
5. Then, and only then, the displacement channel in the research harness (second forward pass at γ=0), validated once against a true base-model load before trusting the projection-delta approximation.
