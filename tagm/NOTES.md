# TAGM Translation Notes

Honest log of decisions, deferrals, and open items from the TASM → TAGM
translation.

## Frontend rewiring — sessions 3-5: full TASM-compat backend complete

Full backend rewiring done across three sessions. Every fetch site in
the TASM frontend now has a corresponding TAGM endpoint. 71 endpoints
in `app.py` (was 28 before session 2). All 64 Python files parse clean.

### Session 3 — session management + dashboard

- `GET /api/session/results?page=N&per_page=M` — paginated, each record
  carries `_index` and `_plot_keys`, projected through tasm_compat.
- `GET /api/dashboard` — slim per-record list + session_info aggregate
  (the data-table view).
- `GET /api/results/detail?start=N&count=M` — start+count window of
  full-shape records (used by the terrain visualization which
  chunk-fetches).
- `POST /api/session/remove` — remove indices, reindex remaining.
- `POST /api/session/rerun` — re-analyze prompt indices, append at end.
- `POST /api/session/restore` — restore from disk snapshot. New
  helpers `_snapshot_session_to_disk()` and `_has_session_snapshot()`
  in app.py; snapshot called automatically after each analyze + batch.
- `POST /api/prompts` — append to prompt library CSV.
- `POST /api/user_info` — accept user-info form, store on AppState.

### Session 4 — modules + probes + viz HTML routes

New module: `tagm/service/modules_runner.py` (~250 lines). Singleton
`ModuleRunner` that wraps both measurements and analyses behind TASM's
unified async-job interface — list/run/status/results/reset/log.
Measurement runs reconfigure the orchestrator temporarily, re-analyze
the whole session against the named measurement, restore the prior
selection. Background threads update state.

Endpoints added:
- `GET /api/modules` — combined measurement+analysis list with status.
- `POST /api/modules/upload_template` — saves CSV to templates dir.
- `POST /api/modules/{name}/run` — starts an async run.
- `GET /api/modules/{name}/status` — polls run state.
- `GET /api/modules/{name}/results` — fetches results, with
  fallback to session.analyses[name] or session prompts'
  measurement entries when no explicit run was made.
- `POST /api/modules/{name}/reset` — clears module state.
- `GET /api/modules/{name}/download_log` — log JSON.
- `GET /api/engine_config` + `POST` + `POST /api/engine_config/reset`
  — engine-wide param dict (defaults match TASM's chat/SFD/LTP knobs).
- `POST /api/probe_set/apply` (form upload, file) + `GET .../apply_status`
  — probe template upload that embeds via the loaded pipeline using
  the real `EmbeddingGenerator` + `GenerationParams` + `ProbeStore.put`
  path. Caught and corrected three real API mismatches by reading the
  actual source: `parse_template_csv` takes a Path (not text);
  `EmbeddingGenerator.generate` takes a `GenerationParams` object;
  template fields are `cells`/`columns`/`rows` (not `probes`).
- `GET /api/probe_set/status` — currently active probe set info.
- `POST /api/probe_set/clear_caches` — clears SFD precompute + probe sets.
- HTML routes: `/chat`, `/domain_surface_viz`, `/correction_manifold_viz`,
  `/correction_heatmap_viz`, `/correction_backscatter_viz`.

### Session 5 — chat + plots + export

New modules:
- `tagm/service/chat.py` (~120 lines). Wraps `pipeline.instruct_model.generate()`
  (or base if toggled). Single `_chat_lock` to prevent generation +
  analysis from interleaving on shared model parameters. Streaming
  intentionally NOT enabled: TASM's chat returned full responses in
  one JSON payload and chat.html doesn't implement streaming-receive.
- `tagm/service/plots.py` (~430 lines). Server-side matplotlib
  rendering of 12 plot keys matching TASM's `_plot_keys_for_result`:
  signed_attribution, stress_per_token, distribution_metrics,
  amplitude_trajectory, heatmap, ltp_profiles, ltp_tension_magnitudes,
  ltp_dual_trajectory, ltp_summary_stats, ltp_profile_heatmap,
  sfd_density, rank_displacement. Each handler takes a TASM-shape
  result dict; returns PNG bytes. Lazy matplotlib import; consistent
  dark theme matching TASM's UI palette. Graceful empty-plot
  fallbacks for missing data.

Endpoints added:
- `POST /api/set_inference_model` — toggle chat between instruct/base.
- `POST /api/chat` — generation + optional analyze flags. Tags
  resulting prompt records with `role: user|assistant` in metadata.
- `GET /api/plots/individual/{index}/{plot_key}` — per-prompt PNG.
- `GET /api/plots/{plot_key}` — aggregate (currently falls back to
  rendering against prompt 0; main.js doesn't actually call this path
  but parity was kept).
- `POST /api/export` — prepare export to a temp file.
- `GET /api/export/download` — stream the prepared file (or generate
  fresh if /api/export wasn't called first).

Requirements updated: `matplotlib` and `aiofiles` added.

### Endpoint coverage

Verified end-to-end: every `/api/*` URL appearing in `static/js/main.js`,
`static/chat.html`, and the five `_viz.html` pages maps to a TAGM
endpoint. The two endpoints that previously returned 404
(`/api/session/results`, `/api/dashboard`) now exist and return the
TASM-shape data the frontend reads. The `_tagm_native` field on each
TASM-shape result preserves the raw TAGM measurement dict for
inspection.

### What hasn't been browser-tested

None of this has touched a real browser. Syntax is right, every API
shape was cross-checked against what main.js actually parses, but the
first end-to-end run will surface small mismatches — most likely a
field name in `tasm_compat.py` that doesn't quite match a
specific TASM read site, or a measurement parameter that the engine
config UI exposes by a different name. Those are minor fixes once a
browser session surfaces them; none would change the architecture.

## Frontend — TASM UI extracted into modular files

Background: an earlier session redesigned the frontend from scratch into
a five-tab single-page app. That was the wrong work — the brief was to
keep TASM's UI and just point it at the new backend with cleaner code
organization. The redesign is gone.

Current state: TASM's `index.html` (4,340 lines) restored and split into
three files:

- `static/index.html` — 231 lines. Same DOM, same layout, same look as
  TASM. Inline `<style>` and inline `<script>` blocks replaced with
  `<link>` and `<script src>` references to the extracted files.
- `static/css/main.css` — 205 lines, the full extracted style block
  verbatim.
- `static/js/main.js` — 3,902 lines, the full extracted script block
  verbatim. Loaded as a regular (non-module) script so the 127 inline
  `onclick=` / `onchange=` event handlers in the HTML continue to find
  their referenced functions on the global scope.

The five `_viz.html` pages and `domain_surface.jsx` are restored to
their original TASM form. They're each small enough (180-470 lines)
that further modularization wouldn't be a clear improvement; left as
single files.

Rebrand applied throughout: `TASM` → `TAGM`, `tasm_chime` → `tagm_chime`,
`tasm_session.zip` → `tagm_session.json.gz`, `tasm.log` → `tagm.log`,
title updated to `TAGM v0.1 — Transformer Alignment Geometric Metrology`.
Verified zero remaining `TASM`/`tasm` references in the static tree
except the obvious case-sensitive ones inside the JS that shouldn't
change (variable names, comments — all changed; nothing is left).

What this session did NOT do, kept honestly out of scope:

- Repointing backend calls from TASM endpoints to TAGM endpoints.
  The 49 `fetch()` sites in `main.js` still reference the old
  `/api/load_model`, `/api/config`, `/api/modules/*`, `/api/probe_set/*`
  endpoints. Those don't exist in TAGM. The UI will load and render
  correctly; calls that hit dropped or renamed endpoints will fail at
  request time. That's the next session's work — backend-call rewiring
  with the data-shape adjustments needed where TAGM's payloads
  differ from TASM's.

- Reshaping data handling for measurement results. TAGM's
  MeasurementResult schema (scalars / per_token / per_layer /
  per_layer_per_token / objects) is not what TASM's UI parses today;
  result-rendering code needs adjustment.

- Anything about the `domain_surface.jsx` build pipeline. Currently
  no React runtime is loaded; the JSX won't run as-is. To address
  next session: either drop a CDN React script tag or rewrite that
  one component in vanilla JS.

## Contract correction — capture is user-set, not measurement-driven

The initial build had measurements declare `capture_requirements()` that
the orchestrator unioned into a CaptureConfig. That conflated two
distinct concerns:

  - *Capture selection* (pipeline-level, user's choice):
    which layers and hook points are recorded during forward passes.
  - *Measurement scope* (per-measurement, scope parameter):
    which of the captured layers a given measurement aggregates over.

This turn's refactor corrects it. The contract is now:

  - The user sets one CaptureConfig, up front, via `POST /api/capture`.
    This is the pipeline's capture — not derived from measurement selection.
  - Each measurement declares a `CaptureExpectation` (what the capture
    must provide for the measurement to run at all) — e.g.
    "needs `pre_attn_norm` hidden at one or more layers" or "needs
    attention_weights at `attn_output`." The orchestrator validates
    each measurement's expectation against the active CaptureConfig
    before dispatch.
  - Measurement parameters like `layers` are *scope* parameters:
    empty means "use everything that's captured and has the required
    deltas;" non-empty narrows aggregation to that subset. They never
    affect what gets captured.
  - All ten measurements use a shared `resolve_scope_layers()` helper
    that intersects (a) what's in the ActivationStore at the relevant
    hook point, (b) the optional scope parameter, (c) what has the
    required deltas in the DeltaStore. That helper replaces a mix of
    ad-hoc loops and eliminates the `layers=[]` ambiguity.
  - Each measurement records `layers_requested`, `layers_used`, and
    `scope_resolution` in its result's `parameters` dict. Exports
    therefore carry the exact set of layers the measurement actually
    aggregated over — no hidden defaults.

Files changed in this pass:

  - `tagm/measurement/requirements.py` — new `CaptureExpectation`
    replaces old `CaptureRequirement`. Adds `validate_expectation()`.
  - `tagm/measurement/base.py` — `capture_expectation(params)` abstract
    method replaces `capture_requirements(adapter, params)`.
  - `tagm/measurement/scope.py` — new shared helper module.
  - All 10 modules in `tagm/measurement/modules/` — version 0.2.0.
  - `tagm/service/orchestration.py` — new `set_capture_config()` method;
    `configure_measurements()` validates against CaptureConfig and
    returns a structured report including per-measurement violations.
  - `tagm/app.py` — new `POST /api/capture` endpoint; `POST /api/configure`
    now requires capture to be set first (returns 409 if not).

Migration note for frontend: the old `/api/config` endpoint still works
via the compat shim but will 409 until `POST /api/capture` has been
called. Frontends should present capture selection as a separate step
before measurement selection.

## Pre-flight audit fixes (applied after initial build)

Issues found during a review pass on the generated code, all fixed:

1. `from tagm.analysis import __init__ as _analysis_init` in app.py — bogus
   explicit dunder import. Replaced with `import tagm.analysis`.

2. Monkey-patched `PairRunResult.instruct_result_as_runresult` from inside
   orchestration.py — removed. Added a proper
   `RunResult.from_pair(pair)` classmethod on `tagm.core.pipeline.RunResult`
   and use it directly.

3. `api_run_analysis` read the request body twice
   (`await request.json() if await request.body() else {}`). Starlette's
   request stream isn't reliably re-readable; the second read returns
   empty. Replaced with a single `await request.body()` followed by
   `json.loads` when non-empty.

4. Llama3 adapter's `parse_projection_key` called `_ROLE_FROM_KEY.get(...)`
   twice in a conditional expression. Refactored to assign once, guard
   against None, then return.

5. Silent registration failures (an empty registry looking like
   "no measurements available" with no cause). Added explicit assertions
   at app import time that raise `RuntimeError` if either the
   measurement or analysis registry is empty after the import side-effects
   fire.

6. Frontend/backend API surface mismatch. The 4340-line `static/index.html`
   was lifted from TASM and calls 38 distinct endpoints; only 4 match
   TAGM's backend directly. Added a compatibility shim
   (`tagm/service/compat_shim.py`) that:
   - Forwards 12 renamed-only endpoints to their TAGM equivalents
     (e.g. `/api/load_model → /api/load`, `/api/config → /api/configure`).
   - Returns HTTP 501 with a structured migration hint for 22 endpoints
     whose behavior changed or was dropped in the migration
     (e.g. `/api/chat`, `/api/engine_config`, `/api/plots/*`,
     `/api/probe_set/*`, `/api/set_inference_model`, `/api/user_info`).
   The shim keeps the frontend functional for load, configure, analyze,
   and session operations. Visualization pages that depend on the
   missing endpoints will show the migration-hint JSON; they need a
   frontend rewrite to consume TAGM's session schema directly.

## Structural choices that deviate from the spec

**`ModelStructure` attached to `RunResult`.** The measurement spec
assumes `compute()` gets `(run_result, adapter, delta_store, params,
probes)` and that's enough. In practice measurements like
`last_position_attribution` need adapter-derived scalars (head counts,
head_dim) *and* a handle back to the model (for `lm_head` /
`unembedding_weight` / live `o_proj` weights). Rather than force
measurements to re-run `adapter.attention_heads(model)` every prompt or
fish a model reference out of `store.adapter`, the Pipeline now
populates a `ModelStructure` dataclass on every `RunResult` and also
sets `run_result.pipeline` as a back-reference. Measurements use those
two fields freely. Clean; just not what the spec said.

**Analysis modules as peers, not in a submodule.** I started with
`tagm/analysis/modules/` mirroring `tagm/measurement/modules/` and
backed it out. TASM's `engine/modules/` was a flat directory for
analysis modules and it's a smaller, clearer layout. Measurement
modules kept their submodule because there are ten of them and the
enclosing directory was getting crowded.

**`depends_on` as a class attribute on measurements.** The spec doesn't
specify inter-measurement dependency. `AmplitudeDerivedMetrics` wants
the heatmap produced by `AmplitudeTrajectory`; `RankDisplacement` wants
the counterfactual tokens from `LateralTensionProfile`. I added
`depends_on: tuple[str, ...]` as an optional class attribute and
resolved it via Kahn's algorithm in the orchestrator. Dependency
outputs are injected into `params` under the key `_dependencies`, which
is a small API wart — a cleaner version would pass them as a separate
kwarg to `compute()`, but that would have required widening the
signature of every measurement. Revisitable.

**`PairRunResult.instruct_result_as_runresult`.** Monkey-patched onto
the class from inside the orchestrator, so the orchestrator can treat
pair and single runs uniformly. Ugly but contained. A proper solution
is a `RunResult.from_pair(pair)` classmethod; I can lift that in a
one-line follow-up.

## Measurements: what's named differently from TASM

- TASM's **signed_attr** → TAGM's **signed_attribution_to_last** with
  an explicit semantic note in the field spec: "Index i is the
  contribution of token i to the LAST position's correction, averaged
  over selected layers. This is NOT a per-token correction." Per the
  TASM forensic audit finding #1.
- TASM's **per_token_spectral_rank** → TAGM's **sublayer_rank** with
  semantic note: "exp(Shannon entropy) — NOT an SVD rank." Per audit
  finding #3.
- TASM's **displacement_field** module → TAGM's
  **correction_field_topology** analysis. The algorithm is the same;
  the name matches what the outputs describe.

## Measurements: math faithfulness notes

- **LastPositionAttribution**: preserved byte-for-byte from TASM's
  `Analyzer._extract_signed_attribution`. The `proof1_threshold`
  parameter is now user-visible (was a constant in `engine_config.py`).

- **LateralTensionProfile**: full translation of TASM's `compute_ltp`
  including the instruct+base dual banks, PRC computation, SVD-
  truncated `dW_V` option, and the dual-PCA-trajectory output. I
  pulled the `PRC_THRESHOLD = 0.02` constant out of the inner function
  into a user parameter (it was silently affecting `n_directional`).

- **SpectralFieldDensity**: module-scoped SVD cache keyed by
  `(instruct_id, base_id, dtype, layers, k)`. The cache survives for
  the module lifetime, not the pipeline lifetime — a real difference
  from TASM, which cleared it on model unload. Call
  `tagm.measurement.modules.spectral_field_density.clear_sfd_caches()`
  manually if you unload a pair and reload a different one at the
  same pipeline instance. Or just restart the process; memory
  economics favor it.

- **RankDisplacement**: kendalltau via scipy with a graceful fallback
  (tau = 0.0 per position) if scipy is missing. TASM hard-required
  scipy; TAGM treats it as a soft optional.

- **AmplitudeDerivedMetrics**: has no capture requirements of its own;
  reads `amplitude_trajectory`'s heatmap output via the dependency
  mechanism. This is the one measurement that would crash immediately
  if dependency injection broke.

## Analyses: what was abbreviated vs. fully translated

- **ComparativeAnalysis**: fully translated. Bootstrap CIs, Cohen's d
  with bootstrap CIs, optimal-threshold scans, pairwise per-category
  comparisons. Default metric list spans 14 fields across all six
  Wave-1 measurements.

- **MIReadiness, MIInstrumentation**: TASM's
  `mechanistic_interpretability.py` was 774 lines combining readiness
  diagnostics with per-token instrumentation tables. I split into two
  focused analyses.

- **TokenVariance**: fully translated.

- **CorrectionHeatmap**: fully translated. TAGM's per-token alignment
  contract eliminates TASM's off-by-one bug in per-cell drill-down.

- **CorrectionManifold**: core translated (signature z-scoring,
  pairwise distance, single-linkage agglomerative clustering, optional
  2D PCA). **Abbreviated**: TASM also produced k-medoids alternatives
  and ESC-style trajectory overlays; those are not in this initial
  drop. The output shape includes `signatures_z` and
  `distance_matrix`, so the frontend can re-derive alternative
  projections without re-running the analysis.

- **CorrectionBackscatter**: fully translated. Per-category mean
  matrices + discriminative-sublayer ranking.

- **DomainSurface**: **abbreviated**. TASM's was 1055 lines; most of
  that was plot generation. I preserved the core polar-projection
  algorithm (angular = best subject probe, radial = stress or density
  or escalation score) and the per-category occupancy summaries. What
  I did not translate: kernel density estimation for the heatmap
  overlay, and the trajectory-through-time visualizations. Those read
  from the `points` array in the result, so they can live in the
  frontend or a second-pass analysis module.

- **CorrectionFieldTopology**: translated with the core layer-tension
  trajectory and attn/MLP transition; TASM also had a "flow arrows"
  feature that rendered as 2D vector arrows over the layer stack,
  which I skipped.

## Service layer

- **Sequential base-phase for single-prompt analyses**: the
  orchestrator currently uses `Pipeline.run_pair_batch([prompt], ...)`
  as a degenerate single-entry batch when base data is needed and the
  base model is not loaded. This is correct but redundantly loads and
  unloads the base model per prompt. For workflows that do many
  single-prompt analyses in a row, either keep the base model loaded
  (`POST /api/load_base` — not yet exposed on the API surface) or
  batch them through `POST /api/batch`.

- **`POST /api/analyze` concurrent mode**: requires that the base
  model be loaded at call time. There is no explicit `load_base`
  endpoint in this initial drop; adding it is six lines in
  `tagm/app.py`. Deferred because for the Codespace single-user
  workflow that TAGM targets, batch mode is the norm.

- **Probe requirement resolution in the orchestrator**: `get()`
  currently passes `parameters={}` to the ProbeStore lookup, which
  means a measurement cannot request probes generated with different
  generator parameters than the default. For v1, probes are looked up
  by (template_id, capture_signature, model_pair_id); if you want
  richer parameter-aware lookup, extend `ProbeRequirement` to carry
  parameter constraints.

## Open spec items (documented in the design package as deferred)

- **Cache invalidation policy**: not implemented. On-disk artifacts
  accumulate; `GET /api/status` reports disk usage. A weekly cleanup
  pass would be a user preference, not an instrument concern.

- **Session schema migrations**: schema_version = 1 everywhere. The
  `load_session` function dispatches on version, but there are no
  migrations to dispatch to yet.

- **`run_pair_batch` base-data extractor signature**: each measurement
  contributes a dict-valued `base_extract` return; the orchestrator
  merges them by key. Two measurements that both produce
  `per_position_base_alts` (LTP and RD both do) will clobber each
  other. In practice the data is identical — both derive from
  `torch.topk(base_logits[i], k+overfetch)` — so the clobber is safe,
  but it's not ideal. A principled solution is to have the
  orchestrator dedupe extractions across measurements; deferred.

## Thing I'd change if I were doing it again

The `_dependencies` key on `params` is a hack. Measurements with
inter-measurement dependencies should receive an explicit `deps` kwarg.
It would not break the existing measurements — just add
`deps=None` to `MeasurementModule.compute()` and change the two
dependency-consuming measurements to read from it. Quick follow-up.

## TASM behaviors intentionally dropped

- **Global `engine_config` module with 30+ tunables**: replaced by
  per-measurement `ModuleParameter` declarations. No process-wide
  mutable state. Default values are declared on the parameters, not
  hidden in a config file.

- **Hook name strings like `layer_5_h` / `layer_5_traj_attn`**: the
  `ActivationStore` is addressed by `(layer, hook_point_name,
  capture_type)` tuples; there are no string keys.

- **`model_manager.state.signal_layers = middle_third_by_default`**:
  TAGM makes layer selection a user choice on every measurement that
  cares. No "signal layers" as a global concept.

- **`session_dir` side-effects during analysis module runs**: TASM's
  analysis modules wrote cache files to disk as a side effect of
  `run()`. TAGM's analyses are pure functions of session + params;
  caching is the session export's job.

- **In-process base-log-softmax caching for KL**: preserved as an
  optional field on `BatchBaseCache` (`base_log_softmax`), but no
  measurement in this initial drop requests it. KL-based measurements
  are a natural extension — add a `KLDivergence` measurement class
  that `needs_base_logits=True` and reads `base_cache['base_log_softmax']`.
