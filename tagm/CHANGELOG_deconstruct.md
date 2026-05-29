# Changelog — Unify analysis onto one async contract

_2026-05-29_

## Changed

Analysis had two completion contracts: batch was async (submit → `batch_done`
SSE → refetch), while a single prompt was synchronous (inline JSON result).
The synchronous path had already grown a timeout-recovery polling branch to
survive long requests — a sign the contract had outgrown "sync." Both paths
already shared the `_analyze_prompt_list` core; now they share the **contract**
too. The synchronous path and its recovery hack are gone. A single prompt is
just a one-item job.

**One terminal event.** `batch_done` → **`analyze_done`**, published by both
endpoints (and kept in the broker snapshot for reconnect replay). Its payload
carries the full outcome so a failed prompt can never finish silently:

    { ok, n_results, n_prompts, n_errors, error }

- `ok=False` → fatal/infrastructure error; nothing produced.
- `ok=True, n_results=0` → the job ran but every prompt failed (`error` = first
  failure message).
- `ok=True, n_errors>0` → partial: some succeeded, some failed.

The client surfaces all three loudly (the error path was built first, since it
is where a careless async conversion turns a failure into a silent one).

**Backend (`app_core.py`).** New shared `_start_analysis_job(prompts, flags, *,
deconstruct, n_prompts, progress)` guards, spawns the worker thread, saves, and
publishes exactly one `analyze_done`. `api_analyze_handler` and
`api_analyze_batch_handler` are now thin: parse input, read flags (single keeps
`trajectory_default=True` and stays console-quiet unless deconstructing; batch
keeps `trajectory_default=False`), and delegate. The batch-only
`_batch_running`/`_batch_lock` guard is generalized to a job-level
`_job_running`/`_job_lock` covering all analysis (distinct from the per-inference
`_analysis_lock`, which is unchanged). Removed the now-unused `run_in_threadpool`
import.

**Frontend (`main.js`).** `analyzePrompt` and `analyzeBatch` are thin callers of
a shared `_submitAnalysis(url, fd, btn, errorElId)` + `_onAnalyzeComplete(evt,
…)`. The completion waiter is registered **before** the POST so a fast job can't
finish before the client is listening; a rejected start leaves a harmless
resolver that the next event drains. Completion refetches full results (for
`_plot_keys`/detail) and the slim dashboard, then chimes — once, after the table
reflects the new rows. (This also fixes a latent bug: a single prompt run after
a batch used to render against a stale `dashResults`.)

**Removed dead config.** The **Backend timeout** / **poll every**
(`cfgBackendTimeout`, `cfgPollInterval`) fields existed only to feed the deleted
recovery branch; with it gone they were misleading dials wired to nothing.

## Trade-off

A single prompt now refetches results like batch instead of appending the one
inline result (`sessionResults.push`). At typical scale (tens–low-hundreds of
prompts) this is unnoticeable; if very large sessions feel slow on each
single-prompt run, a delta-fetch (`/api/results/detail?start=prev&count=new`)
would restore the append without reintroducing a second contract.

## Files touched

- `src/engine/app_core.py`
- `src/service/events.py`
- `static/js/main.js`
- `static/index.html`

---

# Changelog — Notification chime timing on prompt completion

_2026-05-29_

## Fixed

The "Notification chime" was supposed to sound when a prompt finishes and the
dashboard reflects the new result. Two paths were wrong:

- **Single-prompt analysis didn't chime at all** on normal success — the
  chime existed only in the timeout-recovery branch, so an ordinary completed
  prompt finished silently. Added `playChime()` right after the dashboard
  update (`renderDataTable` / `dtGoPage`) in `analyzePrompt`.
- **Batch analysis chimed too early** — it fired the instant `batch_done`
  arrived, before the results were fetched and the table re-rendered. Moved
  the chime to after `renderDataTable()` so it lands once the dashboard shows
  the new rows.

The timeout-recovery path already chimed after the table update and is
unchanged. The chime remains gated by the existing toggle.

## Files touched

- `static/js/main.js`

---

# Changelog — Comparative Analysis: legend/color bug fixes

_2026-05-29_

## Fixed

Bugs in the category legend/color handling shared across the comparative
plots (trajectory overlay, the 2×2 grid, the LTP and SFD panels — all route
through `_cat_legend`):

- **Missing legend entries.** The legend is now built correct-by-construction
  from the categories actually plotted, ordered by `CAT_ORDER`, so a present
  category with a unique color (e.g. `harmful`, vermillion) always gets its
  own row instead of silently dropping out.
- **Duplicate-color swatches.** `benign`/`baseline` share a color, as do
  `jailbreak`/`adversarial`; the legend used to emit two identically-colored
  rows. Color-aliased categories now collapse to a single canonical entry.
- **Line vs. legend color mismatch.** Plotted lines resolved a missing
  category to `#888` while the legend resolved it to `#999999`, so an
  uncategorized line didn't match its swatch. All color resolution now goes
  through one helper (`_cat_color`), so a line's color always equals its
  legend swatch. Unordered insertion-order legends are now CAT_ORDER-sorted.

## Not changed (legibility, not bugs)

Left for a separate pass if wanted: the attn/MLP sawtooth (amplitude sums 3
weight roles for attn vs 2 for mlp — a structural artifact, splitting the
series or averaging over roles would fix it), per-category mean overlays, and
labeling extreme-trajectory prompts.

## Files touched

- `src/engine/comparative.py`

---

# Changelog — New "Probe Activity" Tab (replaces redundant Circuit Decomposition)

_2026-05-29_

## Summary

Replaced the Circuit Decomposition tab — which rendered a single tile that
was a pixel copy of the main heatmap (the backend only computes the primary
circuit + baseline, so "all circuits side-by-side" had nothing to show) —
with **Probe Activity**: a global ranking of which probe terms are most
connected to the correction field across the dataset, inverting the per-cell
Cell Composition view. Same data, transposed; no new analysis.

## Added

- **Probe Activity tab**: ranks every active probe term by mean
  <code>|response|</code> over the current scope (session aggregate, class, or
  single prompt — follows the existing class/prompt selector). Shows term,
  mean |response| (the ranking key), signed mean response, and direction
  (aligned / anti-aligned / orthogonal, using the same scope-relative band as
  the rest of the viz).
- **Cell filter**: clicking a heatmap cell filters the ranking to that cell's
  probes (same click wiring as Cell Composition); a "show all probes" link
  clears it. Cell clicks no longer yank you off Probe Activity or Cell
  Composition — the selection just filters in place.
- **Stopword toggle** (default on): hides filler terms using a per-probe
  `is_stopword` flag the module now attaches.
- **Sortable columns** on the Probe Activity table (Term, |response|,
  response, Direction) — click to sort, click again to reverse. Also made
  the **Direction** column sortable in **Cell Composition**, where it had
  been a plain header; both tabs now sort on all data columns, with string
  columns (Term/Probe, Direction) defaulting to ascending and the rest to
  descending.

## Module (`correction_prism.py`)

- Reads the shared `templates/stopwords.txt` asset directly (best-effort, via
  `_load_stopword_set`) and tags each probe with `is_stopword` using the same
  rule the probe generator applies (`len<3 OR in stoplist`).
- **Deliberately self-contained**: it does *not* import the probe generator's
  loader — it reads the same data file with its own small parse, so the
  modules stay independently removable. A few duplicated lines, traded for
  zero cross-module code coupling. If the asset is missing, stopword flagging
  no-ops; the module never hard-depends on it.

## Removed (dead code)

- The Circuit Decomposition tab, its `renderDecomp` function, the now-unused
  `CIRC_ORDER` state, and the orphaned `.decomp-grid` / `.decomp-tile` /
  `.mini-legend` CSS (including print rules). Fixed the stale print selector
  (`#tab-decomp` → `#tab-probe`) and the cell-panel empty-state text.

## Files touched

- `src/engine/modules/correction_prism.py`
- `static/correction_prism_viz.html`

---

# Changelog — Module Rename: Correction Prism → Probe-Basis Decomposition

_2026-05-29_

## Summary

Renamed the module's user-facing name from "Correction Prism" to
**Probe-Basis Decomposition** (drops the optical metaphor for a term that
names the operation: decomposing the prompt's correction field onto the
probe basis). The rename is **display-only** — every internal identifier
keeps the `correction_prism` slug so saved sessions, exports, routes, and
the viz URL are unaffected.

## Changed (user-visible)

- Module `display_name` → "Probe-Basis Decomposition".
- "Prism Metric" parameter label → "Decomposition Metric" (param *key*
  `prism_metric` unchanged).
- Viz page `<title>`, header name, and the config chip label.
- Glossary: the "Prism" entry → "Probe basis"; "prism metric" / "prism
  direction" / "excluded from the prism" reworded to decomposition /
  probe-basis vocabulary. (Also fixed a stale "excite or oppose" phrase
  missed in the earlier vocabulary sweep → "align with or run against".)
- Validation errors and progress messages ("Prism requires…",
  "Computing prism response…") reworded.
- Inline renderer strings (failure banner, heatmap title, metric label).
- Export filenames `prism_*` → `probe_basis_decomp_*`.
- Module docstring title; removed a dangling reference to a
  never-existent `correction_prism_spec.md`.

## Kept stable (internal identifiers — deliberate)

- Module `name = "correction_prism"` slug; the `"module"` field in
  result dicts; API routes (`/api/modules/correction_prism/*`); the viz
  route + filename (`/correction_prism_viz`); persisted module filenames.
- `prism_metric` param key; code identifiers (`prism_dirs`,
  `_prism_response`, `CorrectionPrismModule`, JS `renderCorrectionPrismResults`
  / `popoutCorrectionPrism` / `window.__prism*`, `#prism-*` element ids).

Flipping the slug would break the saved sessions/exports already keyed on
`correction_prism`, plus the routes and viz URL — so the label moved and
the key stayed, same split as the JSON-field rename. A full slug
migration is available as a separate opt-in if ever wanted.

## Files touched

- `src/engine/modules/correction_prism.py`
- `static/correction_prism_viz.html`
- `static/js/main.js`

---

# Changelog — Correction Prism: Configurable Orthogonality Band + Vocabulary

_2026-05-29_

## Summary

Replaces the hardcoded `±0.05` directional cutoff in the Correction
Prism with a **scale-relative, user-configurable band**, and renames the
direction vocabulary from excited/opposed/neutral to
**aligned / anti-aligned / orthogonal** (which is what a signed cosine
actually reports — no causal/energetic claim).

The old fixed cutoff sat ~20× above the entire per-probe response range
at 0.5B scale, so every probe classified as orthogonal (the 0/0/all
degeneracy). The band is now `pct% × mean(|response|)` over active
probes, computed per scope, so it self-calibrates to the run's
magnitudes. `pct=0` gives a pure sign classifier; larger `pct` widens
the orthogonal class.

## Added

- **`directional_threshold_pct`** module parameter (float, default 10,
  range 0–200), surfaced in the Correction Prism card. Half-width of the
  orthogonality band as a percentage of the mean `|response|`.
- Popout viz reads the parameter from `config` and recomputes the band
  per scope (single-prompt / class) via a new `scopeThreshold` helper
  that mirrors the module's gross-mean approach.

## Changed

- **Threshold semantics**: `signed_thresh` is now
  `directional_threshold_pct/100 × mean(|response|)` over active probes,
  not a fixed `0.05`. Computed per scope — the module uses the
  session-aggregate per-probe means; the viz uses the displayed prompt's
  (or class's) responses. Strict boundary (`|response| > band`), so
  `pct=0` cleanly yields sign-only classification with only exact zeros
  left orthogonal. Per-subject counts use a cell-scale band
  (`pct% × mean(|cell value|)`) since they classify cell values.
- **Vocabulary rename** (display + JSON + CSS, everywhere):
  - direction strings `excited`/`opposed` → `aligned`/`anti-aligned`
    (`orthogonal` unchanged).
  - result-dict keys `n_excited`/`n_opposed`/`n_neutral` →
    `n_aligned`/`n_anti_aligned`/`n_orthogonal` (snake_case keys;
    hyphenated `"anti-aligned"` display string).
  - CSS vars `--excited`/`--opposed` → `--aligned`/`--anti-aligned`.
  - glossary entry rewritten: orthogonal is "within the band
    (`|response| ≤ band`)", not exact perpendicularity, and documents
    the `pct=0` sign-classifier behavior.
  - module docstring / UI description reworded off excite/oppose.
- **Inline per-subject `mean_signed` tint** no longer uses a hardcoded
  `±0.05`; it tints against the same configurable cell-scale band the
  per-subject `N aligned`/`N anti-aligned` counts use, so the tint and
  the counts in a row agree. `pct=0` → sign-only tint.

## Dead-code removal + empirical-constant annotation

- Removed `correction_backscatter_v2_spec.md` — a dead state file for the
  backscatter module (a removed precursor to the Correction Prism).
  Cleaned the now-dangling references: three explanatory comments in the
  prism viz no longer cite "the backscatter popout," and the README
  probe-system list now names Correction Prism in its place.
- Annotated the `empirical_p < 0.05` significance test (random-projection
  control, `main.js`) as an intentional statistical alpha — explicitly
  commented as NOT a tunable/removable display constant. Unrelated `0.05`
  values in other modules (arditi tolerance, MI effect cutoff) were left
  untouched as out of scope.

## Payload-compatibility note

- The result-dict keys changed (`n_excited` → `n_aligned`, etc.). Any
  external reader or saved session keyed on the old names must use the
  new ones. This is a deliberate clean break to keep the schema and the
  UI vocabulary consistent.

## Verified (against the live 27-prompt export)

- Gross mean |response| over active probes ≈ 0.0024 (vs. the old fixed
  0.05). Old band → (0, 0, 500). New band: `pct=0` → 314/186/0;
  `pct=10` → 302/177/21; `pct=100` → 85/85/330 — monotonic, self-
  calibrated, no degeneracy.
- Module compiles; both viz files parse; no stale `excited`/`opposed`/
  `0.05` left in the prism code paths.

## Files touched

- `src/engine/modules/correction_prism.py`
- `static/correction_prism_viz.html`
- `static/js/main.js`

---

# Changelog — Prompt Deconstruction + Analyze-Pipeline Unification

_2026-05-29_

## Summary

Adds an opt-in **Deconstruct** mode that expands each prompt into its
prefix ladder — `I` / `I like` / `I like cake` / `I like cake.` — and
analyzes every rung as its own record, so a prompt's contextual field
can be tracked one unit at a time against the fixed probe lattice.
Punctuation earns its own rung. Each rung record carries `family_index`
and `rung_index` so the ladder regroups in the data table and
downstream modules.

Along the way, the duplicated single-prompt and batch analyze paths were
collapsed onto one shared analysis core (a single prompt is now just a
one-element batch), which is what lets Deconstruct hook the pipeline in
exactly one place.

The toggle is runtime-only — it resets on reset/reboot and is not
persisted as configuration.

## Added

- **`src/engine/deconstruct.py`** (new) — pure prompt segmentation.
  - `segment_prompt(text)` returns the prefix ladder as literal prefixes
    of the original string (cut at unit boundaries), so the final rung
    is byte-identical to the input — the model never sees a re-spaced or
    re-tokenized variant.
  - Policy: a punctuation **run** is its own rung (`...` is one rung);
    intra-word apostrophes and hyphens stay attached (`don't`,
    `well-being` are single units).
  - `expand_prompts(prompts, enabled)` expands a prompt list into rung
    records (no-op when disabled); each rung gains `rung_index` and a
    provisional `family_local`.
- **Deconstruct checkbox** in the analyze control panel
  (`static/index.html`, `id="cfgDeconstruct"`). Applies to both single
  and batch since it flows through the shared form-data builder.
  Runtime-only (no `saveConfig`).
- **`family_index` / `rung_index`** persisted as indexed columns on the
  results table (not blob-only) so they surface in the data table
  without decompression.
- **`Fam` and `Rung` columns** in the data table (`static/js/main.js`) —
  two separate integer columns surfacing `family_index` and
  `rung_index` directly; blank (`--`) for non-ladder rows.
- **`Database.next_family_base(session_id)`** — assigns collision-free
  family ids (`MAX(family_index)+1`), robust against the idx reindexing
  that `remove_results` performs.
- Single-analyze response now includes **`n_rungs`** so the frontend
  knows when one prompt produced several records.

## Changed

- **Unified analyze pipeline** (`src/engine/app_core.py`). The
  duplicated analysis body in `api_analyze_handler` and
  `api_analyze_batch_handler` is replaced by one shared core,
  `_analyze_prompt_list(prompts, flags, *, deconstruct, progress)`, plus
  a shared `_read_analyze_flags(form, *, trajectory_default)`. The two
  endpoints remain as thin wrappers preserving their response contracts:
  single is synchronous and returns the result inline; batch is async
  and returns "started". Deconstruct + family/rung assignment happen
  once, in the core.
- **Single-prompt path, on deconstruct**: stores all rungs and returns
  the final rung (the complete prompt) as the inline result; the data
  table refreshes from the server (using `n_rungs`) so it shows the full
  ladder rather than just the inline result.
- **`db.py`** schema/IO updated for the two new columns: `CREATE TABLE`
  DDL, `_migrate_columns` (in-place `ALTER TABLE ADD COLUMN` on existing
  databases — upgrades automatically on boot), `_extract_scalars`,
  `ResultsList.append`, `Database.insert_result`, and
  `get_dashboard_rows`.

## Behavior preserved

- The console (`progressLog`) is **not** cleared on the analyze path;
  `clearLog()` remains limited to model load and explicit reset. A
  deconstructed run appends per-rung progress; an ordinary single run
  stays quiet, as before.
- Deconstruct OFF is a true no-op: prompts are analyzed exactly as
  before and `family_index`/`rung_index` are stored as `NULL`
  (NULL = "not part of a ladder").
- `ltp_svd_rank` is passed uniformly with default `0` — identical to the
  batch path's prior implicit behavior.

## Migration notes

- Existing `~/.tagm/tagm.db` files upgrade in place on first boot via
  `_migrate_columns`; no manual step. Pre-existing rows get `NULL` for
  the two new columns.

## Known item (deferred, by design)

- A decimal splits under the current punctuation rule
  (`3.5` → `3` / `3.` / `3.5`), because `.` is treated as punctuation
  regardless of context. Decoupled from everything else; gluing
  digit-period-digit is a one-line regex change in `deconstruct.py`.

## Notes / not yet verified

- All touched Python compiles; both result INSERTs balance at 26
  columns/placeholders/values; `main.js` parses. The new data path was
  integration-tested against a real SQLite DB (plain row → NULL; a
  deconstructed prompt → family 0, rungs 0–3 regrouping in order; a
  subsequent call → families 1, 2 with no collision).
- A live forward pass was **not** runnable in the dev environment
  (no torch). First thing to confirm on the Codespace: a single prompt
  with Deconstruct checked produces N rows sharing one family with
  sequential rungs, and the period lands on its own rung.

## Files touched

- `src/engine/deconstruct.py` (new)
- `src/core/db.py`
- `src/engine/app_core.py`
- `static/js/main.js`
- `static/index.html`
