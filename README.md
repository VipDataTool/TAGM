# TAGM — Transformer Alignment Geometric Metrology

TASM-compatible backend for measuring alignment geometry in
instruction-tuned language models. Extracts per-prompt stress
signatures, lateral tension profiles, spectral field density,
and rank displacement from weight-delta-mediated forward passes.

## Quick start (GitHub Codespaces)

1. Open in Codespaces (a 4-core / 16GB machine is comfortable;
   install dependencies per the local quick start below)
2. `bash start.sh` from the repo root
3. Open the forwarded port 8000

Inside a container the server binds `0.0.0.0`, because the port is reached
through a forwarding proxy that connects from outside the container's
loopback — a `127.0.0.1` bind makes the forwarded URL resolve but serve
nothing. Detection is by `CODESPACES` / `CODESPACE_NAME` /
`REMOTE_CONTAINERS` / `DEVCONTAINER` or `/.dockerenv`.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
bash start.sh
```

Open http://localhost:8000.

Outside a container the bind is `127.0.0.1`. This API is unauthenticated and
can load models and read files, so exposing it on the network is deliberate:
`TAGM_HOST=0.0.0.0 bash start.sh`. `TAGM_HOST` overrides the default either
way, and `--host` / `--port` are also accepted by `python -m src`.

## Dependencies

Core runtime (all in `requirements.txt`):

- **torch** — model inference (CPU-only recommended for Codespaces)
- **transformers** + **accelerate** + **safetensors** — HuggingFace model loading
- **fastapi** + **uvicorn** — HTTP server
- **python-multipart** — form data parsing (file uploads, analyze requests)
- **numpy** + **scipy** — numerical computation
- **matplotlib** — server-side plot rendering
- **scikit-learn** — PCA for domain surface module
- **huggingface-hub** — model downloads
- **starlette** — imported directly for `iterate_in_threadpool`, so it is
  declared explicitly rather than relied on transitively via fastapi

(System resource reporting reads `/proc/meminfo` directly — no psutil
needed. Static files are served by Starlette directly — no aiofiles.)

### Version pinning

`requirements.txt` specifies bounded version ranges, not bare package
names. Each lower bound is justified in a comment next to the pin by the
API that requires it — most importantly `transformers>=4.56`, which is
the first release accepting `from_pretrained(dtype=...)`. Older releases
accept only `torch_dtype=` and ignore the unrecognised kwarg, so the
model would silently load at a different precision and every delta,
stress score and LTP magnitude would change with no error raised.

Ranges are not a lockfile. For a reproducible environment, freeze after
the first successful install:

```bash
pip install -r requirements.txt
pip freeze > requirements.lock.txt
```

and use `pip install -r requirements.lock.txt` on other machines and in
CI. Note that `start.sh` only installs dependencies when `uvicorn` is
missing entirely — it does not detect or correct version drift in an
environment that already has the packages.

**sqlite3** ships with Python's standard library — no additional
install needed for the database layer.

Optional: set `HF_TOKEN` environment variable for faster HuggingFace
downloads and to avoid rate limits.

## Architecture

```
src/engine/           TASM-native computation engine
  analyzer.py         Per-prompt extraction (stress, attribution, LTP, SFD)
  result.py           Flat PromptResult dataclass + serialization
  hooks.py            Adapter-based activation capture → flat key dict
  ltp.py              Lateral Tension Profile computation
  sfd.py              Spectral Field Density + Rank Displacement
  config.py           Runtime parameters (frontend: Advanced Parameters)
  session.py          DB-backed session (SQLite via core/db.py)
  app_core.py         Core API endpoint handlers
  statistics.py       Bootstrap CIs, Cohen's d, threshold optimization
  modules/            TASM analysis modules (auto-discovered at startup)

src/core/             Model + data infrastructure
  adapter/            Model-family abstraction (Qwen2, Llama3)
  pipeline.py         Model loading, delta computation, forward passes
  deltas/             Weight delta computation from disk + spectral profiling
  cache.py            Disk cache management (~/.tagm/cache/)
  db.py               SQLite persistence layer
  locks.py            MODEL_LOCK — serializes all model access

src/api/              Route modules mounted by app.py
  _state.py           Shared app state (analyzer, session, module runner)
  modules.py          Module run/status/results/cancel routes
  probes.py           Probe set apply + diagnostics
  roundtable.py       Roundtable session routes
  hep.py              HEP activation routes
  ecm_config.py       ECM parameter routes

src/probes/           Probe CSV loading + per-depth embedding cache
src/service/          Chat, plots, events, export
src/templates/        Probe generator template CSVs
src/app.py            FastAPI app, route registration, static mounts
src/__main__.py       Entry point: python -m src
```

## Data management

All persistent data lives in a single SQLite database at
`~/.tagm/tagm.db` (override with the `TAGM_DB_PATH` environment
variable). The database uses WAL mode for concurrent read/write
and zlib-compressed blobs for result storage.

### What's stored

**Model registry** — the model pairs table (replaces `models.json`).
Add models via the Configuration tab or `POST /api/models`.

**Sessions** — each model-load creates a new session. Results are
stored individually as compressed blobs with indexed scalar columns
for fast dashboard queries.

**Results** — each prompt analysis is a single INSERT, not a full-file
rewrite. Key scalar metrics (`stress_score`, `net_correction`,
`kl_divergence`, `delta_scale`, etc.) are extracted into dedicated
columns so the dashboard can query 1000+ results without
decompressing any blobs.

**Config** — UI preferences in a key-value store (replaces
`ui_config.json`), plus the HEP activation state. Engine parameters
(Advanced Parameters) live in memory and reset to defaults on restart
— by design, since changing them invalidates collected data anyway
(the UI's Apply flow resets the session for the same reason).

**Prompts** — the prompt library (mirrors `prompts.csv`).

### Schema migration

On first startup the database bootstraps automatically. `_migrate_columns`
in `db._bootstrap()` compares `PRAGMA table_info` against the expected
column set and issues `ALTER TABLE ADD COLUMN` for anything missing, so a
database written by an older build is upgraded in place on open with no
manual step. Existing rows read NULL in the new columns.

A one-time import of legacy JSON files (`models.json`,
`datasets/current/results.json`, `ui_config.json`) also runs if those files
are present, renaming the originals to `*.migrated`. It is idempotent and
does nothing on a database that has no such files beside it.

The database path can be overridden for testing or multi-instance
setups:

```bash
TAGM_DB_PATH=/tmp/test.db bash start.sh
```

## How it works

1. Load a model pair → Pipeline computes weight deltas from disk
2. Click Analyze → form-data flags go to the engine
3. Engine installs hooks (adapter resolves WHERE), runs one forward pass
4. Extraction functions read activations + deltas, write to flat PromptResult
5. `result_to_dict()` → compressed INSERT into SQLite → frontend reads via API

The adapter tells us where to hook. The delta store tells us what
changed. The extraction functions read both and write to flat fields.
Results are persisted on every `add_result()` — there is no separate
save step.

## Data tab

The grid renders from `GET /api/dashboard` — the indexed columns only, no
blob decompression — so it stays responsive at a few thousand records. It is
for scanning and selection:

- **Sort** by clicking a header: ascending, again descending, a third time
  back to record order (the default). Nulls sort to the bottom in both
  directions, and ties break on record index so the order is stable.
- **Filter** matches prompt, category and role. Not the numeric columns,
  where a typed `0.1` returns scattershot hits.
- **Hover a header** for the field's definition. A `°` marks a column whose
  definition carries a caveat — that `stress_score` is not comparable across
  models, or that `n_directional` is a count and so scales with sequence
  length. The Field Glossary card above the grid lists all of them, and both
  are generated from `static/js/common/fields.js` so they cannot disagree.

Per-record depth is the **record card**: select rows, click *View Selected*,
and a popout renders each full record — metrics, the server-rendered plots
from `/api/plots/individual/{index}/{plot_key}`, the LTP section, the full
top-k table and the per-token attribution table. That path fetches full
records from `/api/session/results`, so it sees the per-token arrays the grid
cannot show.

## Probe system

Probes enable the Domain Surface, Correction Field Topology, and
Probe-Basis Decomposition (slug: `correction_prism`) modules.

### Generation

The **Probe Generator** module (Modules tab) queries the instruct
model to build a discriminative vocabulary fingerprint per
class × subclass cell. It reads a template CSV, queries the model
N times per cell, counts token frequencies, then applies two-axis
deduplication (cross-class and cross-subclass) so that each
surviving term is unique to its lattice cell.

Templates live in `src/templates/`. Each template defines a
subject × subclass lattice.

### Embedding

The **Auto-Embed After Generation** checkbox (enabled by default) in
the Probe Generator parameters automatically embeds and activates the
generated probe set when generation completes. The probes are embedded
through adapter-mediated hooks at configurable depths (L50, L75 by
default), cached to `probe_cache/`, and activated — all as part of the
same Run. No extra clicks needed.

If auto-embed is unchecked, or if it fails, the **Embed & Activate
Probe Set** button appears in the results panel as a manual fallback.

### Manual apply

Alternatively, apply any probe CSV via the Configuration tab →
Probe Set → Choose File → Apply. This is useful when working with
hand-curated probe sets or probe files generated in a previous
session.

### Probe diagnostics

The **Probe Diagnostics** popout (↗ button on the Probe Generator
card) inspects lattice properties of the active probe set: cell
coverage, sample terms per cell, cross-class/cross-level collisions,
and embedding-tier metrics when a probe cache exists for the loaded
model.

### Cross-model probe workflow

The same probe CSV produces separate per-model embedding caches, and the
lattice geometry (subjects, levels) provides the shared coordinate system
for comparing cell-aggregated outputs across models. The cache key is
`(probe_file, model_id, depths, projected)`, so switching models re-embeds
rather than silently reusing another model's vectors.

Not every metric survives the crossing. `stress_score` carries a
`1/sqrt(d_in)` factor and `ltp.mean_M` / `mean_V` scale with `‖ΔW_V‖`, the
unembedding row norms and hidden size — none of them are dimensionless, so
they compare within a model, not across. `n_negative_tokens` and
`ltp.n_directional` are counts that grow with sequence length; divide by
`seq_len` before comparing prompts of different lengths, because the result
dict reports the raw count.

## API

`POST /api/analyze` with form data:

```
prompt, category, compute_kl, compute_ltp, compute_sfd,
compute_trajectory, full_capture, capture_responses,
compute_ecm, harvest_responses, ecm_harvest_tokens,
ltp_k, ltp_layer_strategy, ltp_svd_rank, deconstruct
```

Booleans are read by `_read_analyze_flags` in `engine/app_core.py`, which is
shared by the single and batch paths; the only difference between them is
that `compute_trajectory` defaults on for a single prompt and off for a
batch. Response harvesting is gated on `harvest_responses` with
`ecm_harvest_tokens > 0`; `compute_ecm` then selects whether the harvest runs
under ECM regulation or as a plain control.

Analysis is **asynchronous**: the endpoint returns
`{"ok": true, "started": true, "n_prompts": N}` immediately and
announces completion exactly once via the `analyze_done` event on the
`GET /api/events` SSE stream (payload:
`{ok, n_results, n_prompts, n_errors, error, cancelled}`). Fetch results
via the session endpoints below. `POST /api/analyze_batch` (CSV upload)
follows the same contract. Counterfactual probabilities everywhere are
full-vocabulary softmax values, comparable across the instruct/base
pair and against `instruct_topk` / `base_topk`.

### Cancellation

Long jobs — a few-hundred-prompt batch, or a module that sweeps every
layer — can be stopped from the UI. Two endpoints:

```
POST /api/analyze/cancel              Stop the running analysis job
POST /api/modules/{name}/cancel       Stop a running module
```

Both return `{"ok": true, "cancelling": true}`, or
`{"ok": false, "error": ...}` if nothing is running. Note what that
`ok` means: **the request was accepted**, not that the work has stopped.

**Cancellation is cooperative.** There is no safe way to interrupt a
Python thread inside a torch op, so both paths set a flag that the
running code notices at its next checkpoint and unwinds by exception —
which means `finally` blocks still run, so model weights are restored
and hooks removed on the way out. Stopping is therefore not
instantaneous; worst case is one unit of work:

- Analysis job: checkpoints between prompts, so at most one prompt
  (one forward pass, plus generation if harvesting is on).
- Modules: checkpoints at every `progress()` report, plus explicit
  polls in the long-running loops of `routing_ablation`,
  `concept_atoms`, `probe_generator` and `correction_prism` — at most
  one generation or one forward pass in those.

The two paths differ in **what happens to the work already done**:

- A cancelled **analysis job keeps** every prompt that finished. Each
  one is a complete, already-persisted record, so `analyze_done` fires
  with `ok: true`, `cancelled: true` and the kept count in
  `n_results`. The UI reports "Analysis cancelled — N records kept".
- A cancelled **module discards** its results. A half-finished
  analysis is not a result, and keeping it invites quoting numbers
  from an aborted run. The module goes to `cancelled` with no results
  to fetch.

Module status values are now:

```
idle | running | completed | partial | cancelled | error
```

`partial` means the module returned a usable result set *and* an error
(an optional late stage failed); `cancelled` means the user stopped it
and is styled neutrally in the UI, not as a failure. Cancellation is
also reported live on the SSE stream: a `module_status` event with
`{name, status: "cancelled", elapsed}`, and a `progress` event with
stage `"cancelled"` for the analysis job.

### Session endpoints

```
GET  /api/dashboard                  Scalar-only rows (no decompression)
GET  /api/session/results            Paginated full results
GET  /api/results/detail             Paginated full results with plot keys
POST /api/session/restore            Restore most recent session from DB
POST /api/session/clear_all          Reset session
POST /api/session/remove             Remove specific result indices
POST /api/session/rerun              Re-analyze specific prompts
```

### Model management

```
GET  /api/models                     List registered model pairs
POST /api/models                     Add or update a model pair
POST /api/load_model                 Load a model pair into memory
```

### Export

```
POST /api/export                     Start a background export job
GET  /api/export/download            Download the finished archive
```

`POST /api/export` returns immediately; the `export_ready` SSE event
(or `export_error` on failure) signals completion. The artifact is a
**zip**: `session.json.gz` (the session object with `session_id`,
`model`, `n_results`, `results` — minus the three `per_token_*_emb`
fields, which are split into `embeddings_*.csv.gz` companions, keyed
back by `_embedding_files` / `_embedding_dims`), plus per-module
reports under `modules/`. Tooling that read the old single-file
gzipped JSON needs to read `session.json.gz` inside the zip and, if it
used embeddings, join the CSVs.

## Indexed result columns

These scalar fields are stored in dedicated SQLite columns and
queryable without decompressing the result blob:

```
prompt, category, seq_len, stress_score, net_correction,
entropy, top2_share, middle_share, interior_cv, kl_divergence,
n_negative_tokens, has_negative_tokens, delta_scale,
full_capture_enabled, family_index, rung_index,
ltp_mean_m, ltp_mean_v, ltp_max_prc, ltp_n_directional,
sfd_density_mean, rd_mean_tau, rd_mean_overlap,
role, inst_top1, inst_top1_p, base_top1, base_top1_p
```

All other fields (per-token arrays, heatmaps, trajectories, LTP
profiles, full top-k lists, etc.) are stored in the compressed blob and
loaded on demand when the frontend requests a specific result.

`inst_top1` / `base_top1` and their probabilities are the **rank-1 entry
only**. The full top-k lists stay in the blob; the record card reads them
from a full record. Anything the grid displays has to be a column here,
because `/api/dashboard` never decompresses a blob.

Adding a column means editing five places in `core/db.py`: the `CREATE
TABLE`, both `INSERT` statements, `_extract_scalars`, the `get_dashboard_rows`
SELECT, and the `expected` dict in `_migrate_columns` — the last so existing
databases gain it via `ALTER TABLE` on next open. Rows written before the
column existed read NULL; they are not backfilled, because that would mean
decompressing every blob at startup.
