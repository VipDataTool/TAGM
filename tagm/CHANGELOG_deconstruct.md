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
