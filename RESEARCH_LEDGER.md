# Research Ledger — ECM + Entropic Asymmetry

Purpose: this file is the persistence layer between AI-assisted sessions.
Each claim gets a status and the experiment that would move it. Update
statuses at the end of every session INSTEAD of re-narrating the work.
Drafts inherit their claims from here; a claim not in this ledger with
status SUPPORTED does not belong in an abstract.

Statuses: SUPPORTED (evidence in hand) · PARTIAL (weak/confounded
evidence) · UNTESTED (mechanism plausible, no data) · REVISED (original
form retracted, replacement noted) · REFUTED.

## ECM mechanism

| # | Claim | Status | Evidence / discriminating experiment |
|---|---|---|---|
| E1 | Multi-scale entropy slope discriminates sustained cascades from benign texture | PARTIAL | v2 synthetic tests (deadband sweep: 2% benign vs cascade-to-floor). Needs real-model rerun: `tools/ecm_ablation_v2.py`. |
| E2 | v1 signal (raw nats, max-over-scales, gain 2.0) tracks texture, not cascades | SUPPORTED | Paper's own 31% benign intervention rate; v1 analysis (2026-07-02 session). Do not cite v1 numbers as ECM results. |
| E3 | Cascade signal is highest on ambiguous/dual-use prompts | PARTIAL | v1 data, n=10 prompts, 1B model, no CIs, non-monotonic table. Rerun under v2 with bootstrap CIs before claiming. |
| E4 | ECM preserves response diversity vs fixed low temperature | PARTIAL | v1 data (5 vs 1 unique responses on 2 prompts). Rerun with paired seeds (ablation script does this). |
| E5 | Cascade signal co-locates with per-token KL / rank-displacement ridge | UNTESTED | `tools/ecm_coupling.py` — the linking experiment between ECM and the topology module. |
| E6 | ECM cannot be removed by weight-space ablation (Arditi-style) | SUPPORTED (trivially) | Architectural: lives in the decode loop. Scope carefully — see E7/E8. |
| E7 | ECM has "no adversarial surface" / "cannot be gamed" | REFUTED (as stated) | Sampler control removes it in one line; prefill/many-shot attacks create confident harmful continuations that ECM would AMPLIFY (Qi et al. 2025; Wolf et al. BEB). Replacement claim: E8. |
| E8 | ECM's failure surface is disjoint from RLHF's: robust to uncertainty-inducing attacks and weight edits, not to sampler control or confidence-inducing attacks | UNTESTED | Prefill-attack experiment: measure cascade signal + outcome under prefilled harmful continuations, ECM on/off. |
| E9 | ECM is a conservatism amplifier, not a values module — on an ablated/base model it amplifies whatever is confident | UNTESTED | Run ablation prompt set through an abliterated pair with ECM on; predicted result: signal still fires at pivots, cooled completions comply. |
| E10 | "Unsupervised alignment" | REVISED | Retitle: unsupervised amplifier of supervised alignment. Safety payload comes from RLHF'd priors (SIRL entropy gap). |
| E11 | Negligible compute; compatible with layer-streaming inference | SUPPORTED | O(vocab) per token. Soften "only class of intervention" → logit bias / banned sequences / CPU guard models share the budget. |
| E12 | ECM works iff the entropy gap exists; gap established independently by SIRL (arXiv:2510.01088, verified real) | SUPPORTED (dependency) | Decouple from geometry theory: mechanism survives even if G-claims fall. |

## Entropic asymmetry / geometry

| # | Claim | Status | Evidence / discriminating experiment |
|---|---|---|---|
| G1 | Base-model harm representations are concentrated/low-rank | SUPPORTED (external) | Shah et al. arXiv:2507.21141 (verified real); Llorente-Saguer arXiv:2603.27412 (verified real, σ≈0.03 vs 0.27 rad, survives abliteration); Arditi et al. 2024. Verify exact K/τ figures against PDFs before quoting. |
| G2 | Asymmetry originates from implementation spaces (destruction context-free, construction context-bound) | UNTESTED (theory) | Rival explanations not yet ruled out: register/genre effects, frequency effects (Li et al. 2025 — cited but unreconciled), probe-template artifacts. Needs the discriminating design, not advocacy. |
| G3 | Principle-level probing yields one low-rank axis; implementation-level probing yields the asymmetry | UNTESTED | The paper's best novel prediction. Design principle-level probe set (coherence-preserve vs -disrupt, register-matched) vs implementation-level set. |
| G4 | Alignment compresses harm geometry onto refusal axis, leaving prosocial space undisturbed | PARTIAL | Correction-prism preliminary on Qwen2.5-0.5B only. |
| G5 | Base models encode intent-typed distinctions (predatory/utilitarian/curious); alignment attenuates them | PARTIAL | Initial triplet probing, one 0.5B pair, no held-out protocol. Pre-register before expanding. |
| G6 | Refusal effective rank 18.6 vs medical 183.4 demonstrates convergence/divergence | PARTIAL (confounded) | Rank tracks response length + lexical diversity. Needs matched-length/diversity controls. |
| G7 | KL(instruct‖base) 11.7x higher on refusal vs factual response | PARTIAL | Single response pair. Bootstrap over a set. |

## Citation hygiene

Verified real (2026-07-02): Shah 2507.21141 (arXiv title differs from
draft's), SIRL 2510.01088, Llorente-Saguer 2603.27412, Arditi 2406.11717.
Unverified: Frank 2603.18280 / 2604.04385, SHARD 2606.15517, Yuan
2508.09224, Dodds 2026, West 2508.01754, SafeSwitch 2502.01042, Zhao
2507.11878, all remaining. Docx Arditi author list is garbled — fix.
Rule: no citation ships unverified; no number ships without a source
page reference.

## Session log

- 2026-07-02: ECM v2 implemented (σ-normalized slopes, deadband 0.75,
  agreement 2, warmup, anomaly-gated baseline, loop guard,
  no-repeat-ngram backstop). Synthetic validation + deadband sweep in
  ECM_V2_NOTES.md. Ablation/coupling/report tooling added under tools/.
  Both drafts reviewed; E7 refuted as stated, E10 revised.
