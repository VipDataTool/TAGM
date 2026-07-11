# Build Log — Sycophancy Signature Module + ECM Module v3 (Run-Time Replay)

**Date:** 2026-07-10
**Scope:** Two deliverables in one change set. (1) A new analytical module, `syco_signature`, testing whether sycophantic pivots are low-divergence events (constructive interference: instruct and base checkpoints agree at the cave-in) or high-divergence events (delta-resident, like refusals). (2) ECM module v3, which moves the cascade-detector hyperparameters into the module as Run-time parameters and replays raw traces on every Run, removing the collection-time coupling that made the Configuration-panel workflow misleading.
**Files touched:** `src/engine/modules/syco_signature.py` (new), `src/templates/syco_lexicon.csv` (new), `static/js/syco_module_ui.js` (new), `src/engine/modules/ecm.py` (rewritten, v3.0.0), `static/js/ecm_module_ui.js` (rewritten), `src/engine/modules/base.py` (one DISPLAY_ORDER line), `static/js/main.js` (parameter tooltip attribute), `static/index.html` (one script include).
**Files explicitly not touched:** `src/engine/ecm_analysis.py`, `src/engine/ecm_v4.py`, `src/engine/ecm_harvest.py`, `src/engine/analyzer.py`, `src/engine/app_core.py`, `src/engine/config.py`, `src/engine/session.py`, `src/service/chat.py`. No collection-path or inference-path code changed; the `result["ecm"]` block attached during analysis is produced exactly as before (the module simply no longer depends on it).
**Status:** Code complete, syntax-verified, exercised against synthetic sessions (§7). Requires a real-session smoke test (§8).

---

## 1. Motivation

### 1.1 Sycophancy signature

Hypothesis under test (H1, constructive interference): sycophancy is a fine-tuning artifact converging with patterns the base model already carries — two waveforms in phase. If true, then at a sycophantic pivot token the instruct and base checkpoints *agree*: per-token KL is low and flat exactly where the model caves, and both checkpoints load agreement tokens. This is the mirror image of the refusal signature, where the checkpoints fight and KL spikes (max 13.77 in the ECM paper's jailbreak data). The falsifying outcome (H2) is that pivots spike like refusal boundaries — sycophancy would be delta-resident and therefore reachable by ECM-style intervention.

Either outcome is a result. H1 grounds a taxonomy of alignment failures by residence (delta vs intersection), each requiring different monitoring; H2 extends ECM's reach to a failure mode it was assumed to be blind to.

### 1.2 ECM module footgun

The v2 module read the frozen `result["ecm"]` replay block computed at *collection* time from Configuration-panel values. Consequence: the intended workflow — run prompts, open the module, adjust parameters, Run, get results — silently did not work for detector parameters. Changing deadband in Configuration changed nothing until a full re-analysis; the module's own parameters only controlled aggregation cosmetics. Users were, accurately, picking values at random.

---

## 2. Architecture

### 2.1 syco_signature (detection only; no inference, no actuation)

```
session_results (collected by the normal pipeline)
      │  requires: category "*:response", per_token_kl, tokens
      │  optional: ltp.counterfactual_tokens + base_counterfactual_tokens
      ▼
┌──────────────────────────────────────────────────────────────┐
│ syco_signature.run(session_results, params)                  │
│                                                              │
│ 1. detokenize tokens → char spans → scan lexicon phrases     │
│    (multi-token safe; "You're" tokenization irrelevant)      │
│ 2. lexical pivots = phrase-start token positions             │
│    structural pivots = sentence-terminal tokens OUTSIDE      │
│    pivot windows (the within-response baseline)              │
│ 3. suppression_ratio = agg(KL@pivots) / agg(KL@structural)  │
│    H1 predicts << 1 ; H2 predicts >= 1                       │
│ 4. agreement mass (if LTP): sum p over lexicon 'mass' tokens │
│    in stored top-k, per checkpoint, at each pivot            │
│    interference_index = min(mass_instruct, mass_base)        │
│ 5. matched pairs via ecm_harvest.source_prompt              │
│ 6. worksheet JSONL for manual caved/held labeling            │
└──────────────────────────────────────────────────────────────┘
```

No new extraction was needed: per-position top-k alternatives with probabilities already exist for both checkpoints (`ltp.counterfactual_tokens`, `base_counterfactual_tokens` — same extractor, same normalization, per the analyzer's own comment). The module is one file plus a lexicon CSV.

### 2.2 ECM module v3 (replay at Run)

```
BEFORE (v2):  Configuration panel ──▶ attach_ecm_analysis at collection
              ──▶ frozen result["ecm"] ──▶ module reads it
              (module Run cannot change detector behavior)

AFTER (v3):   raw traces stay on results:
                per_token_stress · per_token_kl ·
                sfd.per_token_density (negated) ·
                ecm_harvest entropy trace
              module Run ──▶ replay_trace(...) with MODULE params
              (scales, deadband, agreement, warmup — widgets)
              ──▶ fresh channels + coverage report per Run
```

`ecm_analysis.replay_trace` is imported and reused unchanged — the replay math has exactly one implementation. The collection-time `result["ecm"]` block continues to exist for other consumers (topology channel registry, exports); the module simply computes its own.

---

## 3. Design decisions

1. **Suppression ratio, not a cascade channel, as the primary syco instrument.** The CascadeDetector is one-sided (detects rises). H1's prediction is an anomalous *absence* — flat KL where decision points normally spike. A flat trace has no slope for a slope-detector to z-score, so the instrument is a contrast between two populations of positions within the same response.
2. **Structural pivots exclude lexical-pivot windows.** The baseline must not contain the thing being contrasted against it.
3. **Agreement mass is top-k truncated and reported as a lower bound.** Adequate: an agreement token that matters at a pivot is in the top-8. The cap is a parameter (`mass_top_k`) and the caveat ships in the output.
4. **`caved` is a manual label.** Auto-judging sycophancy with a model invites the failure mode into the measurement. The module exports `syco_worksheet.jsonl` and reads the label back if present; it never judges.
5. **No ECM dependency in syco.** Entropy-at-pivot (which would have required Cascade Mitigation ON at harvest) was replaced by top-1 alternative probability from the counterfactuals. Control sessions are genuinely ECM-free.
6. **Lexicon as a template CSV, not a parameter.** `ModuleParameter` types are int/float/bool/select; free text does not fit. `syco_lexicon.csv` follows the concept_atoms template pattern, loaded via `set_project_root()`; a select chooses the preset ('default', 'strict'). 'phrase' rows drive pivot scanning; 'mass' rows drive agreement-mass matching.
7. **ECM detector params default to the engine defaults (5 / 0.75σ / 2 / warmup 4)** and are clamped sanely (agreement ≤ n_scales). The results carry a `detector` block naming exactly what THIS replay used, rendered as a banner in the UI.
8. **Coverage is explicit.** Every channel reports available/missing counts with human-readable reasons ("KL divergence checkbox was off when this result was analyzed"). Skipped results are listed with reasons. The silent-drop footgun documented in ECM_REFERENCE.md is not reproduced.
9. **Teacher-forcing caveat is carried in the output**, not just in prose: response-phase traces are re-analysis of generated text, matching the ECM paper's methodology, which keeps the refusal-spike vs sycophancy-flat contrast apples-to-apples.

---

## 4. Parameters

### syco_signature
| name | type | default | function |
|---|---|---|---|
| lexicon_preset | select | default | phrase/mass sets from syco_lexicon.csv |
| pivot_window | int | 2 | ± tokens per lexical pivot in the KL sample |
| aggregate | select | median | pivot/structural summary before the ratio |
| response_only | bool | true | analyze only `*:response` records |
| mass_top_k | int (adv) | 8 | truncation cap for agreement mass |
| include_per_record | bool | true | per-record details + worksheet |

### ecm (v3)
| name | type | default | function |
|---|---|---|---|
| n_scales | int | 5 | EWMA bank size, dyadic horizons ~2–32 tok |
| deadband | float | 0.75 | ignore-below threshold in σ |
| agreement | int | 2 | scales that must corroborate; signal = weakest |
| warmup | int | 4 | calibration-only tokens per trace |
| signal_threshold | float | 0.0 | 'fired' floor for aggregation |
| include_per_record / include_traces / strip_token_limit | — | true / true / 0 | output size controls |

All descriptions rewritten to state units, direction of effect, and when to move the knob; `main.js` now mirrors each description into a hover tooltip on the widget.

---

## 5. UI

Both renderers use the MI/CFT visual language: collapsible `.mod-results-header` sections, responsive `.mod-summary` stat grids, `.mod-tbl` tables (100% width, sticky headers), and `title` tooltips on every metric, column, and section. The ECM v2 renderer's ad-hoc inline-styled cards and tables (the left-hugging partition) are gone. ECM shows a REPLAY banner (Run-time detector settings + n replayed) and a Channel Coverage section; syco shows a verdict banner that color-codes the suppression ratio (green << 1 → H1, red >= 1 → H2, orange ≈ 1 → underpowered) and states the reading in words.

---

## 6. Session recipe for the sycophancy experiment

KL divergence ON · Harvest responses ON (64 tokens, seed 42) · LTP ON (agreement mass) · SFD optional · **Cascade Mitigation OFF** (clean control — now actually possible). Prompt set 4×10: `neutral-fact`, `pressured-fact`, `pressured-opinion`, `flattery-hook`; each pressured-fact prompt has a neutral twin asking the same fact without the assertion. After the run: open Modules → Sycophancy Signature → Run; hand-label `caved` in `syco_worksheet.jsonl`; re-run for labeled aggregates.

Predictions: ratio << 1 on caved responses ≈ H1; ratio >= 1 with pivot spikes ≈ H2; interference_index high at caved pivots = both waveforms in phase; mass_i high with mass_b low = fine-tuning-dominant (interference story wrong in the other direction); caved ≈ 0 = prompts too weak for a 0.5B model — fix prompts before theory.

---

## 7. Verification performed (synthetic sessions, CPU-only)

- Both modules parse (`ast.parse`), import, auto-discover metadata, and produce JSON-serializable output.
- syco: planted H1 signature recovered (suppression ratio 0.0071 vs neutral ≈ 1); multi-token phrase "You're right" found across token boundaries; interference index computed from both checkpoints' alternatives (0.5 with planted masses 0.6/0.5); record missing per_token_kl reported in `skipped` with reason; categories aggregate with `:response` suffix stripped; matched pairs keyed on `source_prompt`.
- ecm v3: per-channel coverage counts and reasons correct across mixed results (kl missing on one, sfd on most, entropy on non-harvest); planted KL spike fires; live actuation summary preserved; per-record traces present for strips. **Footgun test:** deadband 0.75→2.5 + agreement 2→4 at Run reduced mean peak σ 208.3→50.9 on the same session with no re-analysis.
- JS: `node --check` clean on both UI files; renderModuleResults chain composes (main → ecm → syco).

Known synthetic-only artifact: near-constant traces ride the σ floor (1e-2), producing large z-scores by construction; real 64-token responses do not behave this way.

## 8. Smoke test required (GPU / real session)

1. 4×10 session per §6; confirm both modules validate and Run from the Modules tab.
2. ECM: flip deadband between Runs; confirm the REPLAY banner and results change; confirm coverage matches the checkboxes actually used.
3. syco: inspect "no pivots" records; extend `syco_lexicon.csv` with observed cave-in phrasings; confirm worksheet round-trip (label → re-run → labeled aggregates).
4. Cross-check: replay an existing jailbreak session through ECM v3 and confirm the structural-pivot spike exemplars still peak where the paper says they do.

---

## 9. Addendum (2026-07-10, later): cooling-gated no-repeat guard

**Discovered by data, not review.** A seed-matched benign pair diverged from control at token #4 with ZERO interventions. Mechanism: the ECM path installed HF's `no_repeat_ngram_size=4` — unconditional (polices warm steps) and sequence-global (n-gram window includes the PROMPT). The prompt contained "tallest mountain in the"; at "The tallest mountain in", the guard banned " the", the exact token the plain control sampled. This breaks the causal attribution the paper's Table 2 relies on ("nothing else varies") — the guard varied.

**Fix (files touched):** `src/engine/ecm_guard.py` (new — `CoolingGatedNoRepeat` + pure `banned_next_tokens`), `src/engine/ecm.py` and `src/engine/ecm_v4.py` (guard state, `configure_no_repeat(ngram_size, prompt_len)`, application in `__call__` before temperature scaling, `n_ngram_bans` diagnostic), `src/engine/ecm_harvest.py` and `src/service/chat.py` (drop the generate kwarg; call `configure_no_repeat(nrn, input_len)`).

**Semantics now:** armed only while T_eff < T_base and for `ngram_size` steps after release; n-gram window = generated tokens only; ngram_size < 2 disables; quiet generations apply zero bans → the quiet ECM pipeline is behaviorally identical to plain control by construction. `ecm_no_repeat_ngram` config key unchanged in name, changed in meaning (update ECM_REFERENCE.md's parameter row accordingly).

**Verified (CPU):** pure-logic unit tests — HF-semantics parity on generated ids, prompt-echo never banned even when armed, cooled loop-completion banned, release window decays after n steps; plus a token-for-token replay of the benign confound through the exact processor guard block showing zero masks on the quiet path. NOT verified here: torch masking line under real generation.

**Smoke test additions:** (5) re-run the benign matched pair — ECM and plain must now be token-identical when interventions are zero; (6) confirm `n_ngram_bans` appears in harvest diagnostics and is 0 on quiet runs; (7) regenerate the paper's Table 2/3 matched pairs under the corrected control before publication.

---

## 10. Addendum (2026-07-11): syco_signature v0.2.0

Data-driven patches from the first two real runs. (1) `min_pivot_pos` (default 2): pos 0–1 pivots carry prompt→response transition KL (median 5.9σ vs 0.07 at pos 2+) and are excluded from the KL ratio — but retained in a separate `early_pivots` list contributing MASS only, since agreement mass is position-safe and opening caves ("Yes, ...") are the most common cave form. (2) Window-max mass anchoring: agreement mass and interference now scan the forward pivot window per position (interference = max over positions of min(m_i, m_b) at the SAME position); recovers mass missed when phrases start with non-mass tokens ("You're right"). (3) Lexicon: bare "yes," plus retraction phrases (i was wrong / i was mistaken / i stand corrected / upon reflection) in the default preset. Validated against sessions 004241 and 024630: pressured-fact interference 0.023 → 0.053 pooled (0.093 on mid-response pivots alone); flip-unwarranted retains ratio 0.44 with nonzero shared mass while flip-warranted interference is ~0 — the deference/correction geometric split.
