# TAGM — Transformer Alignment Gradient Metrology

TAGM is a measurement instrument for studying what alignment tuning
(instruction tuning / RLHF) actually changed inside a language model.

It works on a **model pair**: a base checkpoint and its
instruction-tuned sibling (e.g. Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct).
TAGM computes the weight differences between the two, then runs prompts
through the instruct model with hooks installed and measures — token by
token, layer by layer — how much each prompt's processing pushes into
the subspace that alignment tuning modified. The result is a set of
per-prompt geometric signatures you can compare across prompt
categories (benign vs. harmful, for instance) with bootstrap statistics.

Everything runs through a browser dashboard backed by a FastAPI server
and a SQLite store. Supported model families: **Qwen 2.x** and
**Llama 3.x**, via an adapter layer — adding a family means
implementing one `ModelAdapter` subclass.

TAGM is a rewrite of an earlier instrument called TASM; you will see
that name throughout the source. Metric definitions and module
interfaces are TASM-compatible by design.

## What TAGM measures

Plainly: alignment tuning changes a model's weight matrices. The
difference between the instruct and base weights (the "delta") defines
a set of directions in the model's internal space. When a prompt's
activations line up strongly with those directions, alignment tuning
is doing something to that prompt. TAGM quantifies that, several ways:

**Stress** — for each token, how strongly its hidden state projects
into the attention deltas: `‖h · ΔWᵀ‖ / ‖ΔW‖_F`, summed over the Q, K,
and V projections at the signal layers (the middle third of the model
by default). The prompt's stress score is the per-token mean.

**Signed attribution** — decomposes each attention head's
alignment-delta output at the final position into signed per-token
contributions: which tokens pushed the correction forward, which pushed
against it. The decomposition is exact by construction, and every
head's sum is checked against the delta norm on every run (recorded as
`proof1_checks`). Summary scalars derived from it: `net_correction`
(the signed total), counts of negative-contribution tokens, and
distribution-shape metrics (`entropy`, `top2_share`, `middle_share`,
`interior_cv` — is the correction concentrated at the prompt's
boundaries or spread through its interior?).

**Lateral Tension Profile (LTP)** — per-token profiles of how the
alignment deltas laterally displace the residual stream, with
counterfactual token alternatives at each position. Summary scalars:
mean offset magnitude (M), variance (V), coverage, and the count of
directional tokens.

**Spectral Field Density (SFD)** — SVDs the concatenated Q/K deltas
per layer once at load time, then measures how much of each token's
activation lands inside that low-rank delta subspace. The truncation
rank is configurable (fixed k, hidden-dim ratio, or cumulative-energy
threshold).

**Rank Displacement (RD)** — compares the instruct and base models'
ranked counterfactual candidates at each position: Kendall's tau,
candidate overlap, promoted/demoted probability mass. High displacement
means the two checkpoints disagree about what belongs at that position.

**Behavioral comparison** — KL divergence between the instruct and
base next-token distributions, plus top-k predictions from both.
Counterfactual probabilities everywhere are full-vocabulary softmax
values, so they are comparable across the pair.

**Delta spectra** — at load time each delta gets a spectral profile:
effective rank (`exp` of the entropy of its normalized singular
values), stable rank, and top-k energy shares. Low effective rank
means tuning made a surgical, few-direction correction; high means it
reshaped the whole subspace.

None of these metrics is presented as a validated safety measure.
They are instrument readings; the statistics module (bootstrap CIs,
Cohen's d, AUROC in the MI modules, threshold sweeps) exists so you
can test whether they separate your prompt categories, and several of
the bundled tools exist specifically to attack the metrics' validity
(see [Research tools](#research-tools)).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
bash start.sh
```

Open http://localhost:8000, register a model pair in the Configuration
tab (or `POST /api/models`), load it, and analyze a prompt.

Notes:

- The CPU-only torch build keeps the install small; TAGM defaults to
  CPU inference and is comfortable on a 4-core / 16 GB machine with
  sub-1B model pairs.
- `start.sh` cleans stale `__pycache__` directories, installs
  requirements if `uvicorn` is missing, and execs `python -m src`.
- In GitHub Codespaces, follow the same steps in the terminal. (The
  checked-in `.devcontainer/` config predates the current repo layout
  — its `postCreateCommand` references directories that no longer
  exist — so don't rely on it to install dependencies.)
- Optional: `export HF_TOKEN=hf_...` for faster HuggingFace downloads
  and to avoid rate limits.

### Running the server directly

```bash
python -m src --host 0.0.0.0 --port 8000             # defaults
python -m src --port 9000 --log-level debug --reload  # development
```

Host and port can also be set with `TAGM_HOST` / `TAGM_PORT`.
Per-request access logging is off by default because the frontend
polls status endpoints every couple of seconds; pass `--access-log`
to turn it on. Application logs go to `tagm.log` (downloadable via
`GET /api/log/download`).

## Dependencies

All in `requirements.txt`:

- **torch** — model inference
- **transformers**, **accelerate**, **safetensors**, **huggingface-hub**
  — model loading and downloads
- **fastapi**, **uvicorn**, **python-multipart** — HTTP server, SSE
  event stream, form/file parsing
- **numpy**, **scipy** — numerical computation and statistics
- **matplotlib** — server-side plot rendering
- **scikit-learn** — PCA for the Domain Surface module

The database layer is stdlib `sqlite3` — nothing extra to install.
System resource reporting reads `/proc/meminfo` directly (no psutil),
and static files are served by Starlette (no aiofiles).

## Architecture

```
src/engine/           Computation engine
  analyzer.py         Per-prompt extraction: one hooked forward pass →
                      stress, signed attribution, trajectories, LTP, SFD
  result.py           Flat PromptResult dataclass + serialization
  hooks.py            Adapter-based activation capture
  ltp.py              Lateral Tension Profile
  sfd.py              Spectral Field Density + Rank Displacement
  counterfactuals.py  Full-vocabulary counterfactual alternatives
  deconstruct.py      Prefix-ladder expansion (see below)
  statistics.py       Bootstrap CIs, Cohen's d, threshold optimization
  comparative.py      Cross-prompt/category comparison plots
  visualizations.py   Per-prompt plot rendering
  config.py           Runtime parameters (frontend: Advanced Parameters)
  session.py          DB-backed session
  app_core.py         Core API endpoint handlers
  ecm.py / ecm_v4.py  Entropic Cascade Mitigation (see below)
  ecm_analysis.py     Replay ECM detection over stored traces
  ecm_harvest.py      ECM-regulated response harvesting
  ablation.py, interventions.py, qk_intervention.py,
  attention_calibration.py
                      Intervention machinery used by the ablation and
                      calibration modules
  modules/            Analysis modules, auto-discovered at startup

src/core/             Model + data infrastructure
  adapter/            Model-family abstraction (qwen2, llama3)
  pipeline.py         Model pair lifecycle: load, delta computation,
                      instruct/base inference toggle
  deltas/             Delta computation, DeltaStore, spectral profiling
  cache.py            On-disk cache layout (~/.tagm/cache/)
  db.py               SQLite persistence layer
  locks.py            Global model lock

src/probes/           Probe CSV loading + per-depth embedding cache
src/service/          Chat (SSE-streamed), event broker, plots, export
src/templates/        Probe lattice templates (subject × subclass CSVs)
static/               Dashboard UI + module visualization pages
roundtable_templates/ Batch templates for the Roundtable LMA module
tools/                Standalone research harnesses (run outside the server)
stable/               Pinned known-good build (see Stable snapshot)
```

## How an analysis run works

1. **Load a model pair.** The Pipeline loads the instruct model, then
   computes the weight deltas by streaming the base model's weights
   from its safetensors files one tensor at a time — the full base
   model is never instantiated just to diff weights, which is the main
   memory-discipline decision in the codebase. Deltas land in a
   `DeltaStore` addressed by `(layer, role)` where role is one of
   q / k / v / o / gate / up / down, and each delta gets a spectral
   profile. Every model load starts a fresh session in the database.
   (The base model *is* loaded on demand later if you request
   base-model comparisons or base-model chat.)
2. **Analyze.** The engine installs hooks (the adapter resolves the
   family-specific module paths), runs **one forward pass** through the
   instruct model, and the extraction functions read the captured
   activations plus the delta store to fill a flat `PromptResult`.
   Optional flags add KL/top-k comparison against the base model, LTP,
   SFD, full-trajectory capture, and ECM trace collection.
3. **Persist.** `result_to_dict()` → one compressed row inserted into
   SQLite. Results are saved as they are produced; there is no
   separate save step.
4. **Read.** The dashboard queries indexed scalar columns; full
   records decompress on demand.

Analysis endpoints are **asynchronous**: `POST /api/analyze` returns
`{"ok": true, "started": true, "n_prompts": N}` immediately, and a
single `analyze_done` event on the `GET /api/events` SSE stream
announces completion. `POST /api/analyze_batch` (CSV upload) follows
the same contract; batch runs default trajectory capture off for
volume.

### Deconstruction

With **Deconstruct** enabled, one prompt expands into its prefix
ladder — `I` / `I like` / `I like cake` / `I like cake.` — and each
rung is analyzed as its own prompt, tagged with `family_index` /
`rung_index` so records regroup later. Two deliberate rules: a
punctuation run gets its own rung (punctuation shapes context-building,
so it isn't glued to the preceding word), and rungs are literal
prefixes of the original string — never rebuilt from pieces — so
tokenization artifacts don't masquerade as measurements.

## Analysis modules

Modules live in `src/engine/modules/` and are auto-discovered at
startup: any file there defining a `TASMModule` subclass registers
itself and appears in the Modules tab. They run in background threads
over collected session data (`POST /api/modules/{name}/run`, then poll
`/status` and fetch `/results`).

| Slug | Display name | What it does |
|---|---|---|
| `arditi_benchmarks` | Arditi Benchmark Analyses | Refusal-direction experiments following Arditi et al. 2024: ablation (project the direction out, does refusal drop?), addition, and cross-method comparison |
| `comparative_analysis` | Comparative Analysis | Cross-prompt aggregates, category separability, batch plots — the primary module for judging whether metrics separate your categories |
| `concept_atoms` | Concept Atom Explorer | Difference-of-means concept directions from a CSV registry, orthogonality heatmap vs. the refusal direction, as a go/no-go check before ablation |
| `correction_field_topology` | Correction Field Topology | Builds the terrain payload for the 3D counterfactual-plane visualization (client-side Three.js) |
| `correction_prism` | Probe-Basis Decomposition | Decomposes correction signals in the active probe basis |
| `domain_surface` | Domain Surface Geometry | Embeds probes and prompts into a shared PCA space; maps per-token RD/ASM/SFD signals onto the probe-defined domain surface |
| `ecm` | ECM — Entropic Cascade Mitigation | Re-runs cascade detection over stored traces with module-level detector parameters — iterate on detector settings without re-running inference |
| `harm_direction` | Harm Direction (SFD) | Difference-of-means "harm direction" in SFD spectral space, analogous to the residual-stream refusal direction |
| `harm_trajectory` | Harm Trajectory | Cumulative harm-adjacency signal across each prompt's token sequence from SFD connectivity + LTP strain |
| `mechanistic_interpretability` | MI Readiness Analysis | Data-quality evaluation addressing MI-community review points: AUROC, length-confound analysis, and related checks |
| `mi_instrumentation` | MI Instrumentation | Produces citable MI measurements from session data (as opposed to evaluating readiness) |
| `model_dialogue` | Model Dialogue Interface | Launches the chat window; turns can be analyzed and recorded into the session |
| `probe_generator` | Probe Generator | Builds a discriminative probe vocabulary from the loaded model (see Probe system) |
| `roundtable_lma` | Roundtable LMA | Configurable multi-persona Language Model Array, interactive or CSV-batch (see below) |
| `routing_ablation` | Routing Ablation Experiment | Tests whether the SFD harm direction survives refusal ablation and has independent causal effect |
| `syco_signature` | Sycophancy Signature | Tests whether sycophantic pivots are low-divergence (base model already agreed) or delta-resident (alignment installed them) |
| `token_pair_coupling` | Token Pair Coupling | Mines strong (actual → counterfactual) token redirection pairs across the session |

Several modules ship dedicated pages served at the root:
`/domain_surface_viz`, `/correction_prism_viz`,
`/correction_field_topology_viz`, `/probe_diagnostic_viz`,
`/template_maker`, `/roundtable`, and `/chat`.

## Data management

All persistent data lives in one SQLite database at `~/.tagm/tagm.db`
(override with the `TAGM_DB_PATH` environment variable). The database
runs in WAL mode, so readers don't block the writer.

What's stored:

- **Model registry** — registered base/instruct pairs (replaces the
  old `models.json`).
- **Sessions** — one row per model load.
- **Results** — one row per analyzed prompt: the full record as a
  zlib-compressed JSON blob, plus ~20 key scalar metrics copied into
  dedicated indexed columns (listed below). Adding a result is a
  single INSERT; the dashboard reads only the scalar columns, so
  listing and paginating large sessions never decompresses blobs.
  Under the old JSON-file storage, every added result rewrote the
  whole results file and every dashboard load parsed it — the SQLite
  layer exists to remove exactly that.
- **Config** — UI preferences and the HEP activation state. Engine
  parameters (the Advanced Parameters panel) deliberately live in
  memory and reset on restart: changing them invalidates collected
  data, so the UI's Apply flow resets the session for the same reason.
- **Prompts** — the prompt library (mirrors `prompts.csv`).

Migration is automatic and idempotent: on startup, legacy JSON files
(`models.json`, `datasets/current/results.json`, `ui_config.json`) are
imported and renamed with a `.migrated` suffix, and when a code update
adds new indexed columns, the bootstrap detects them via
`PRAGMA table_info` and issues `ALTER TABLE ADD COLUMN` on existing
databases. No manual steps.

```bash
TAGM_DB_PATH=/tmp/test.db bash start.sh   # isolated database
```

### Indexed result columns

Queryable without decompressing any blob:

```
prompt, category, seq_len, stress_score, net_correction,
entropy, top2_share, middle_share, interior_cv, kl_divergence,
n_negative_tokens, has_negative_tokens, delta_scale,
full_capture_enabled, family_index, rung_index,
ltp_mean_m, ltp_mean_v, ltp_max_prc, ltp_n_directional,
sfd_density_mean, rd_mean_tau, rd_mean_overlap
```

Everything else (per-token arrays, heatmaps, trajectories, LTP
profiles, top-k lists…) lives in the compressed blob and loads on
demand when the frontend opens a specific result.

### Other on-disk locations

- `~/.tagm/cache/` — presets, SVD precomputes, mmap delta files
- `<repo>/probe_cache/` — probe embedding caches (JSON, one per
  probe-file × model × depth)
- `<repo>/probe_config.json` — which probe set is active, for which
  model, at which depths
- `<repo>/datasets/current/` — per-module output files
- `~/.tagm/token_pair_cache.json` — Token Pair Coupling's cache

## High-Efficiency Pipeline (HEP)

For disk- and RAM-constrained hosts (Codespaces in particular), HEP
trades speed for headroom:

- switches the delta store to memory-mapped files under
  `~/.tagm/cache/deltas/*.tagm` instead of holding deltas in RAM,
- clears the HuggingFace download cache (existing mmap delta files are
  kept — they're expensive to recompute and valid for reuse),
- optionally evicts the base model's HF cache after deltas are
  computed, since the base weights have served their purpose.

`POST /api/hep/initialize` turns it on, `POST /api/hep/deactivate`
removes the mmap files and returns to memory mode, and
`GET /api/hep/status` reports the backend, mmap file size, and current
disk/RAM headroom. HEP state is persisted in the database and restored
on restart.

## Entropic Cascade Mitigation (ECM)

ECM is an experimental sampling regulator, off by default
(`ecm_active`). During generation it tracks the entropy of the output
distribution across a bank of exponential moving averages at dyadic
time scales. When entropy rises *coherently across scales* — the
cascade signature, uncertainty compounding instead of resolving — it
reduces the effective temperature in proportion to the excess. Design
constraints, per the source: no auxiliary model, no trained
discriminator, no new weights; the model's own distributional
uncertainty is the only signal. A loop guard refuses to cool when the
recent token tail is periodic (cooling into a loop entrenches it), and
an optional `no_repeat_ngram` backstop covers loops seeded during
cooled steps.

Two processor versions are selectable (`ecm_version`): **v2** is
entropy-only; **v4** adds pluggable channels (e.g. entropy + density)
with configurable weights and fusion. The source documents why v1 was
replaced: raw-nats slopes at the fastest scale fired on roughly half of
healthy tokens, pinning temperature at the floor — v2's σ-normalized,
agreement-gated formulation detects cascades rather than spikes.

ECM shows up in three places:

1. **Live** — regulates chat generation when active; diagnostics come
   back with the response.
2. **Replay** — `compute_ecm` on analysis runs the detector over
   stored traces without actuating anything; the ECM module re-runs
   detection with its own parameters so you can iterate on detector
   settings without re-running inference.
3. **Harvest** — `harvest_responses` / `ecm_harvest_tokens` generate a
   short response during analysis (seeded for causal comparability by
   default) and analyze it as its own record.

All parameters are in the Advanced Parameters panel. The config file
is candid about which values are principled and which are arbitrary
(the temperature floor is labeled `ARBITRARY — no derivation`).

## Probe system

Probes are short texts arranged in a subject × subclass lattice. Their
embeddings give the correction-signal metrics a semantic coordinate
system: the Domain Surface, Correction Field Topology, and Probe-Basis
Decomposition modules all require an active probe set.

**Generation.** The Probe Generator module builds a probe set from the
loaded model itself: it reads a template CSV (a subject column plus
subclass columns; templates live in `src/templates/`, and
`/template_maker` builds new ones), queries the instruct model N times
per lattice cell, counts term frequencies, filters stopwords and
low-frequency terms, then applies a **global uniqueness** filter: any
term that appears in more than one cell — same row, same column, or
anywhere else — is removed from all of them. What survives is
vocabulary unique to each cell: the model's own discriminative
fingerprint for that region of the lattice. Cross-class,
cross-subclass, and diagonal collision counts are reported as
diagnostics.

**Embedding.** A probe set becomes usable once embedded: each probe
text is run through the instruct model with a hook at a configurable
depth, and the captured vectors are cached to
`<repo>/probe_cache/`. Depths come from the CSV's meta row if present,
otherwise from engine config, with hard fallbacks of 0.50 and 0.75 of
model depth. The active set is recorded in `probe_config.json` —
which file, which model, which depths — and consumer modules resolve
their caches through that record rather than by filename guessing.
With **Auto-Embed After Generation** checked (the default), generation,
embedding, and activation happen in one Run; if it's unchecked or
fails, an **Embed & Activate Probe Set** button appears as a manual
fallback. You can also apply any hand-made probe CSV from the
Configuration tab.

**Diagnostics.** The Probe Diagnostics popout (↗ on the Probe
Generator card) inspects the active set: cell coverage, sample terms,
collision counts, and embedding-tier metrics when a cache exists for
the loaded model.

**Cross-model comparison.** One probe CSV, applied to several model
pairs, produces a separate embedding cache per model — the lattice
(subjects × levels) is the shared coordinate system for comparing
cell-aggregated outputs across models.

## Roundtable LMA and chat

**Roundtable** (`/roundtable`) runs a configurable Language Model
Array over the loaded model — multiple persona-seeded agents working
through staged inquiries. Two paths through the same infrastructure:
interactive (Run with no template opens a chat workspace where you
pick personas, apply methods and tools, and manage stages by hand) and
batch (upload a CSV template — see `roundtable_templates/` — where
columns are stages of type PANEL / ANALYSIS / TOOL, rows are agent
seeds, and cells are JSON dicts; the pipeline marches through the
columns). The module credits its origin: Ostrander (2024),
ClownCar.AI / alice.ipynb LMA.

**Chat** (`/chat`) talks to the loaded pair directly. Generation
streams over SSE (so long generations don't hit proxy timeouts), and
`POST /api/set_inference_model` toggles between the instruct and base
weights — the base model loads at toggle time, not per message. Turns
can optionally be analyzed and recorded into the session, and ECM
diagnostics ride along when ECM is active.

## API

The full surface is in `src/app.py`; highlights:

`POST /api/analyze` — form fields:

```
prompt, category, compute_kl, compute_ltp, compute_sfd,
compute_trajectory, compute_ecm, full_capture, capture_responses,
harvest_responses, ecm_harvest_tokens,
ltp_k, ltp_layer_strategy, ltp_svd_rank, deconstruct
```

Returns immediately; completion arrives once as the `analyze_done`
event on `GET /api/events` (payload
`{ok, n_results, n_prompts, n_errors, error}`). Prompts are capped at
5000 characters. `POST /api/analyze_batch` takes a CSV upload
(`prompt` and optional `category` columns — see `prompts.csv` for the
shape) under the same contract.

```
# Session
GET  /api/dashboard                  Scalar-only rows (no blob decompression)
GET  /api/session/results            Paginated full results
GET  /api/results/detail             Paginated full results + plot keys
POST /api/session/restore            Restore most recent session from DB
POST /api/session/clear_all          Reset session
POST /api/session/remove             Remove specific result indices
POST /api/session/rerun              Re-analyze specific prompts

# Models
GET  /api/models                     List registered pairs
POST /api/models                     Add or update a pair
POST /api/load_model                 Load a pair (async; model_loaded event)
POST /api/set_inference_model        Toggle instruct/base for generation
POST /api/reset                      Unload and reset the pipeline

# Modules
GET  /api/modules                    Discovered modules + parameters
POST /api/modules/{name}/run         Start a run (background thread)
GET  /api/modules/{name}/status      Poll
GET  /api/modules/{name}/results     Fetch output
POST /api/modules/{name}/reset       Clear module state
GET  /api/modules/{name}/download_log

# Probes
POST /api/probe_set/apply            Embed + activate a probe CSV
GET  /api/probe_set/apply_status     Poll the background embed
GET  /api/probe_set/status           Active-set info
POST /api/probe_set/clear_caches
POST /api/modules/probe_generator/embed_active        (+ _status)
GET  /api/probe_diagnostic

# Config, events, misc
GET  /api/events                     SSE stream (model_loaded, model_error,
                                     analyze_done, export_ready, export_error,
                                     progress, …)
GET  /api/status                     Pipeline/session state
GET/POST /api/engine_config          Advanced Parameters (+ POST .../reset)
GET/POST /api/config                 UI preferences (DB-backed)
GET/POST /api/prompts                Prompt library
GET  /api/templates                  Probe templates (+ /{name}, POST /save)
GET  /api/health                     Liveness
GET  /api/plots/{plot_key}           Batch comparative plots (PNG)
GET  /api/plots/individual/{index}/{plot_key}
GET  /api/hep/status                 (+ POST /initialize, /deactivate)

# Chat & roundtable
POST /api/chat                       SSE-streamed generation
GET/POST /api/chat/config
/api/roundtable/*                    Participants, topic, methods,
                                     interactive session, batch runs

# Export
POST /api/export                     Start a background export job
GET  /api/export/download            Download the finished archive
```

### Export format

`POST /api/export` returns immediately; the `export_ready` SSE event
(or `export_error`) signals completion. The artifact is a **zip**
containing:

- `session.json.gz` — the session object (`session_id`, `model`,
  `n_results`, `results`), minus the three per-token embedding fields
  (`per_token_domain_emb`, `per_token_escalation_emb`,
  `per_token_final_emb`), which are split out into
  `embeddings_domain.csv.gz`, `embeddings_escalation.csv.gz`, and
  `embeddings_final.csv.gz`, keyed back by `_embedding_files` /
  `_embedding_dims` on each result;
- per-module reports under `modules/`.

The export streams results from the database one at a time, so peak
memory is proportional to one result rather than the session.
Embedding float precision is configurable via the request's
`embeddingPrecision` option (clamped to 4–17 significant digits,
default 12). Tooling that consumed the old single-file gzipped JSON
needs to read `session.json.gz` inside the zip and, if it used
embeddings, join the CSVs.

## Stable snapshot

The repo head tracks active development and may occasionally carry
regressions. A known-good build is checked in at
`stable/tagm-stable-0.9.6.zip` as a fallback. If the current code
misbehaves, unzip it into a separate directory and run `bash start.sh`
from there. Because schema migration only ever *adds* columns, a
database touched by newer code generally still works under the older
build — but if you want experiments isolated, point the fallback at
its own store with `TAGM_DB_PATH`.

## Research tools

`tools/` holds standalone harnesses that run outside the server, plus
the prompt sets they consume. Two of them exist specifically to try to
break TAGM's own metrics — they're part of the honesty budget:

- `stress_hnorm_harness.py` — asks whether `stress` actually measures
  the alignment-delta subspace or is just activation magnitude ‖h‖ in
  disguise; recomputes stress from scratch and compares against raw
  residual-stream norms at the same positions and layers.
- `genre_confound_analysis.py` — effective rank, angular
  concentration, and transfer proximity over benchmark output, probing
  whether category separation is a genre/style confound.
- `benchmark_harness.py` — offline cascade detection over prompt sets;
  bridges the analyzer's per-token metrics with the v4 CascadeDetector.
- `ecm_ablation_v2.py`, `ecm_ab_report.py`, `ecm_coupling.py`,
  `test_ecm_v4.py` — ECM ablation studies, A/B reporting, and
  channel-coupling analysis.
