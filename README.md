# TAGM — Transformer Alignment Gradient Metrology

TASM-compatible instrument for measuring alignment geometry in
instruction-tuned language models. TAGM loads a base/instruct model
pair, computes the weight deltas between them, and extracts per-prompt
stress signatures, lateral tension profiles, spectral field density,
and rank displacement from delta-mediated forward passes — all through
a browser dashboard backed by a FastAPI server and a SQLite store.

Supported model families: **Qwen 2.x** and **Llama 3.x** (adapter-based
— new families are added by implementing a `ModelAdapter` subclass).

## Quick start (GitHub Codespaces)

1. Open the repo in Codespaces (a 4-core / 16 GB machine is comfortable)
2. `bash start.sh` from the repo root
3. Open the forwarded port → http://localhost:8000

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
bash start.sh
```

Open http://localhost:8000.

`start.sh` cleans stale bytecode, auto-installs missing dependencies,
and execs `python -m src`. You can also run the server module directly:

```bash
python -m src --host 0.0.0.0 --port 8000            # defaults
python -m src --port 9000 --log-level debug --reload # development
```

Host and port are also configurable via the `TAGM_HOST` / `TAGM_PORT`
environment variables. Per-request access logging is off by default
(the frontend polls status endpoints every ~2 s); pass `--access-log`
to enable it. Application logs go to `tagm.log`, downloadable from the
UI or via `GET /api/log/download`.

## Dependencies

Core runtime (all in `requirements.txt`):

- **torch** — model inference (CPU-only build recommended for Codespaces)
- **transformers** + **accelerate** + **safetensors** — HuggingFace model loading
- **huggingface-hub** — model downloads
- **fastapi** + **uvicorn** — HTTP server + SSE event stream
- **python-multipart** — form data parsing (file uploads, analyze requests)
- **numpy** + **scipy** — numerical computation
- **matplotlib** — server-side plot rendering
- **scikit-learn** — PCA for the domain surface module

**sqlite3** ships with Python's standard library — no additional
install needed for the database layer. System resource reporting reads
`/proc/meminfo` directly (no psutil), and static files are served by
Starlette (no aiofiles).

Optional: set `HF_TOKEN` for faster HuggingFace downloads and to avoid
rate limits.

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
  comparative.py      Batch comparative plots (category overlays, scatters)
  visualizations.py   Per-prompt and batch plot rendering
  deconstruct.py      Prefix-ladder expansion for deconstruction runs
  counterfactuals.py  Full-vocabulary counterfactual probabilities
  ecm.py / ecm_v4.py  Entropic Cascade Mitigation (v2 entropy-only, v4 multi-channel)
  ecm_analysis.py     Replay-mode ECM trace analysis on stored results
  ecm_harvest.py      ECM-regulated response harvesting during analysis
  ablation.py         Ablation experiment machinery
  interventions.py    Activation/weight intervention primitives
  qk_intervention.py  QK-space interventions
  attention_calibration.py  Attention calibration analyses
  modules/            TASM analysis modules (auto-discovered at startup)

src/core/             Model + data infrastructure
  adapter/            Model-family abstraction (Qwen 2.x, Llama 3.x)
  pipeline.py         Model loading, delta computation, forward passes,
                      instruct/base inference toggle
  deltas/             Weight delta computation from disk + spectral profiling
  cache.py            Disk cache management (~/.tagm/cache/)
  db.py               SQLite persistence layer
  locks.py            Global model lock

src/probes/           Probe CSV loading + per-depth embedding cache
src/service/          Chat (SSE-streamed), SSE event broker, plots, export
src/templates/        Probe-generator lattice templates (subject × subclass CSVs)
static/               Dashboard UI + module visualization pages
roundtable_templates/ Batch templates for the Roundtable LMA module
tools/                Standalone research harnesses (run outside the server)
```

### Analysis modules

Modules are auto-discovered from `src/engine/modules/` at startup — any
file defining a `TASMModule` subclass registers itself. Currently:

| Slug | Display name |
|---|---|
| `arditi_benchmarks` | Arditi Benchmark Analyses |
| `comparative_analysis` | Comparative Analysis |
| `concept_atoms` | Concept Atom Explorer |
| `correction_field_topology` | Correction Field Topology |
| `correction_prism` | Probe-Basis Decomposition |
| `domain_surface` | Domain Surface Geometry |
| `ecm` | ECM — Entropic Cascade Mitigation |
| `harm_direction` | Harm Direction (SFD) |
| `harm_trajectory` | Harm Trajectory |
| `mechanistic_interpretability` | MI Readiness Analysis |
| `mi_instrumentation` | MI Instrumentation |
| `model_dialogue` | Model Dialogue Interface |
| `probe_generator` | Probe Generator |
| `roundtable_lma` | Roundtable LMA |
| `routing_ablation` | Routing Ablation Experiment |
| `syco_signature` | Sycophancy Signature |
| `token_pair_coupling` | Token Pair Coupling |

Several modules ship dedicated visualization pages served at the root:
`/domain_surface_viz`, `/correction_prism_viz`,
`/correction_field_topology_viz`, `/probe_diagnostic_viz`,
`/template_maker`, `/roundtable`, and `/chat`.

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

With **Deconstruct** enabled, a single prompt expands into its prefix
ladder; every rung is analyzed and stored with `family_index` /
`rung_index` so records regroup downstream.

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

### Performance at scale

| Operation | Old (JSON) | New (SQLite) |
|---|---|---|
| Add result #1000 | Rewrite ~50 MB file | Single INSERT (~5 ms) |
| Dashboard load (1000 rows) | Parse entire file + Python loop | SQL query, no decompression (~7 ms) |
| Page of 20 results | Parse entire file | `LIMIT 20 OFFSET n` (~19 ms) |
| Storage (1000 prompts) | ~50 MB raw JSON | ~23 MB (zlib compressed) |
| Crash safety | Corrupted file | SQLite WAL transactions |

### Schema migration

On first startup the database bootstraps automatically. If legacy
JSON files exist (`models.json`, `datasets/current/results.json`,
`ui_config.json`), they are auto-migrated into the database and
renamed to `*.migrated`. The migration is idempotent — files that
have already been migrated are skipped.

When new indexed columns are added in code updates, the
`_migrate_columns` step in `db._bootstrap()` detects missing columns
via `PRAGMA table_info` and issues `ALTER TABLE ADD COLUMN` to
upgrade existing databases in place. No manual intervention needed.

The database path can be overridden for testing or multi-instance
setups:

```bash
TAGM_DB_PATH=/tmp/test.db bash start.sh
```

## High-Efficiency Pipeline (HEP)

For disk/RAM-constrained hosts (Codespaces), the High-Efficiency
Pipeline switches the delta store from the in-memory backend to
memory-mapped files under `~/.tagm/cache/deltas/*.tagm`, clears the
HuggingFace cache, and optionally evicts the base model cache after
delta computation.

- `POST /api/hep/initialize` — clear HF cache, switch to mmap backend
- `POST /api/hep/deactivate` — remove mmap deltas, return to memory mode
- `GET /api/hep/status` — backend, mmap file size, disk/RAM headroom

HEP state persists in the database and is restored on restart.

## Entropic Cascade Mitigation (ECM)

ECM is an adaptive sampling processor that tracks output-distribution
entropy across a bank of dyadic-scale EWMAs during generation. When
entropy rises coherently across scales (the cascade signature), it
reduces effective temperature proportionally; a loop guard releases
temperature if the token tail turns periodic. No auxiliary model, no
trained discriminator — the model's own distributional uncertainty is
the only signal.

Two processor versions are selectable via `ecm_version`: **v2**
(entropy-only, `engine/ecm.py`) and **v4** (pluggable multi-channel,
`engine/ecm_v4.py`, e.g. entropy + density channels with configurable
fusion). ECM applies to chat generation when `ecm_active` is on, can
be replayed analytically over stored results (`compute_ecm`), and can
harvest ECM-regulated responses during analysis
(`harvest_responses` / `ecm_harvest_tokens`). All parameters are
tunable from the Advanced Parameters panel.

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
subject × subclass lattice; the **Template Maker** page
(`/template_maker`) builds new ones, and custom templates can be
uploaded via `POST /api/modules/upload_template`.

### Embedding

The **Auto-Embed After Generation** checkbox (enabled by default) in
the Probe Generator parameters automatically embeds and activates the
generated probe set when generation completes. The probes are embedded
through adapter-mediated hooks at configurable depths (L50, L75 by
default), cached to `probe_cache/` at the project root, and activated
— all as part of the same Run. No extra clicks needed.

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
model. Cross-model comparison works by reusing one probe CSV across
model pairs: each model gets its own embedding cache, and the lattice
geometry (subjects, levels) provides the shared coordinate system for
comparing cell-aggregated outputs.

## Roundtable LMA

The Roundtable module runs a configurable Language Model Array over
the loaded model. Two paths through the same infrastructure:

1. **Interactive** — Run with no template opens a chat workspace at
   `/roundtable`: type an inquiry, select personas, apply methods and
   tools, manage stages step by step.
2. **Batch** — upload a CSV template (see `roundtable_templates/`);
   columns are stages (PANEL / ANALYSIS / TOOL), rows are agent seeds,
   cells are JSON dicts, and the pipeline marches through columns left
   to right.

## Chat

The Model Dialogue Interface (`/chat`) generates from the active model
— toggle between the instruct and base weights with
`POST /api/set_inference_model`. Generation is streamed over SSE.
Chat turns can optionally be analyzed and recorded into the session,
and ECM diagnostics are returned in the response when ECM is active.

## API

`POST /api/analyze` with form data:

```
prompt, category, compute_kl, compute_ltp, compute_sfd,
compute_trajectory, compute_ecm, full_capture, capture_responses,
harvest_responses, ecm_harvest_tokens,
ltp_k, ltp_layer_strategy, ltp_svd_rank, deconstruct
```

Analysis is **asynchronous**: the endpoint returns
`{"ok": true, "started": true, "n_prompts": N}` immediately and
announces completion exactly once via the `analyze_done` event on the
`GET /api/events` SSE stream (payload:
`{ok, n_results, n_prompts, n_errors, error}`). Fetch results via the
session endpoints below. `POST /api/analyze_batch` (CSV upload) follows
the same contract, with trajectories defaulting off for batch volume.
Counterfactual probabilities everywhere are full-vocabulary softmax
values, comparable across the instruct/base pair and against
`instruct_topk` / `base_topk`.

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
POST /api/set_inference_model        Toggle instruct/base for generation
POST /api/reset                      Unload models, reset pipeline
```

### Modules

```
GET  /api/modules                    List discovered modules + parameters
POST /api/modules/{name}/run         Start a module run (background thread)
GET  /api/modules/{name}/status      Poll run status
GET  /api/modules/{name}/results     Fetch module output
POST /api/modules/{name}/reset       Clear module state
GET  /api/modules/{name}/download_log
```

Probe-set application (`POST /api/probe_set/apply`) and probe
embedding (`POST /api/modules/probe_generator/embed_active`) follow a
background-thread + status-polling pattern with matching
`*_status` GET endpoints.

### Events, config, misc

```
GET  /api/events                     SSE stream (model_loaded, model_error,
                                     analyze_done, export_ready, export_error,
                                     progress, ...)
GET  /api/status                     Pipeline/session state
GET/POST /api/engine_config          Advanced Parameters (+ /reset)
GET/POST /api/config                 UI preferences (DB-backed)
GET/POST /api/prompts                Prompt library
GET  /api/templates                  List probe templates (+ /{name}, /save)
GET  /api/health                     Liveness check
GET  /api/plots/{plot_key}           Batch comparative plots (PNG)
GET  /api/plots/individual/{index}/{plot_key}   Per-result plots
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
reports under `modules/`. Embedding float precision is configurable
via the `embeddingPrecision` option (4–17 significant digits).
Tooling that read the old single-file gzipped JSON needs to read
`session.json.gz` inside the zip and, if it used embeddings, join the
CSVs.

## Indexed result columns

These scalar fields are stored in dedicated SQLite columns and
queryable without decompressing the result blob:

```
prompt, category, seq_len, stress_score, net_correction,
entropy, top2_share, middle_share, interior_cv, kl_divergence,
n_negative_tokens, has_negative_tokens, delta_scale,
full_capture_enabled, family_index, rung_index,
ltp_mean_m, ltp_mean_v, ltp_max_prc, ltp_n_directional,
sfd_density_mean, rd_mean_tau, rd_mean_overlap
```

All other fields (per-token arrays, heatmaps, trajectories, LTP
profiles, topk lists, etc.) are stored in the compressed blob and
loaded on demand when the frontend requests a specific result.

## Research tools

`tools/` contains standalone harnesses that run outside the server,
plus the prompt sets they consume:

- `benchmark_harness.py` — offline cascade detection over prompt sets
  (bridges the analyzer's per-token metrics with the v4 CascadeDetector)
- `stress_hnorm_harness.py` — validity check: does `stress` measure the
  alignment-delta subspace, or just residual-stream norm ‖h‖?
- `genre_confound_analysis.py` — effective rank, angular concentration,
  and transfer proximity from benchmark JSONL output
- `ecm_ablation_v2.py`, `ecm_ab_report.py`, `ecm_coupling.py`,
  `test_ecm_v4.py` — ECM ablation, A/B reporting, and channel-coupling
  studies
