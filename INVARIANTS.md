# TAGM Invariants

Checkable contracts that hold across the whole codebase. Any session —
human or AI — editing TAGM must preserve these, and any new code that
touches the relevant subsystem must be checked against them before it
lands. Each entry names its single source of truth; do not reimplement
the operation elsewhere.

## 1. Probabilities are full-vocabulary softmax

Every probability attached to a token — `instruct_topk`, `base_topk`,
counterfactual alternatives, anything fed to rank displacement — is a
softmax over the **full vocabulary**, computed in **float32**. Never
softmax a top-k slice of logits: that renormalizes over the slice and
inflates the values, and two slices of different sizes (the original
instruct-k+1 vs base-k+5 bug) are not even mutually comparable.

Source of truth: `src/engine/counterfactuals.py`
(`full_softmax`, `top_alternatives`, `decode_alternatives`).

## 2. One tokenization scheme everywhere

Every tokenizer call that feeds a measurement passes
`add_special_tokens=engine_config.get("add_special_tokens")` (default
False — content-only, position-aligned across model families). Pooling
that aggregates token embeddings honors
`engine_config.get("include_first_token")`. This applies to prompt
analysis, the base phase, behavioral comparison, chat-turn analysis,
and **probe embedding** alike — probes and prompts must live in the
same coordinate system or the probe modules measure tokenizer
artifacts.

Probe caches are stamped `pool_version: 2`; caches without that stamp
predate this invariant and are invalidated on load.

## 3. All model access goes through MODEL_LOCK

Every forward pass and every `generate()` on the loaded models acquires
`src/core/locks.py:MODEL_LOCK`. The analyzer holds it across hook
install → forward → extraction → hook removal; the probe embedder holds
it for the whole loop its hook is installed; ablation holds it for the
whole intervened phase (the intervention hooks live on the shared
model). Chat, probe generation, and roundtable turns lock per call.
Model load and reset are refused while an analysis job is in flight
(`app_core.job_active()`).

Rationale: ActivationCapture and ActivationIntervention hooks are
installed on shared `nn.Module`s. An unlocked concurrent forward
silently corrupts captures or runs under someone else's intervention —
no exception is ever raised, which is why the lock is structural rather
than advisory.

## 4. Database access is serialized

`src/core/db.py:Database` owns one SQLite connection shared across
threads. Every statement goes through its locked `execute`/`commit`;
any multi-statement write uses `with db.transaction():`. Iteration over
results uses short-lived page queries — never hold a cursor open across
yields.

## 5. Exact norms for normalization; truncation is labeled

Anything that *divides* by a delta norm uses the exact Frobenius norm
recorded by `DeltaStore.put()` at delta time. Spectral summaries
(`frob_norm`, `stable_rank`, energy shares) are computed against the
exact tensor norm; only `eff_rank` derives from the truncated top-k
spectrum, and is documented as an approximation.

## 6. Every engine_config key has a consumer

If a parameter exists in `config.py:DEFAULTS`, some code path reads it
and it does what its Advanced Parameters description says. Dead knobs
were removed (attention_weighted_pool, export_domain_embeddings,
persist_probe_caches, ltp_overfetch_*); `delta_svd_k` and
`ltp_svd_rank` are wired. Adding a parameter without a consumer — or
removing a consumer without the parameter — violates this.

## 7. Plots draw only real data

No visualization synthesizes data points from summary statistics. A
plot renders what it actually has (estimates, CIs, n) or renders a
`placeholder_plot` explaining what's missing.

## 8. Async jobs always terminate loudly

Any background job the UI waits on (`analyze`, `export`) publishes
exactly one terminal SSE event on every path, including exceptions
(`analyze_done`, `export_ready`/`export_error`). A worker body without
a try/except that publishes the failure event will hang the UI.
