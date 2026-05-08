# TAGM — Transformer Alignment Geometric Metrology

TASM-compatible backend for measuring alignment geometry in
instruction-tuned language models. Extracts per-prompt stress
signatures, lateral tension profiles, spectral field density,
and rank displacement from weight-delta-mediated forward passes.

## Quick start (GitHub Codespaces)

1. Open in Codespaces — the devcontainer installs all dependencies
   automatically (CPU-only PyTorch, ~2 min build)
2. `cd tagm && bash start.sh`
3. Open http://localhost:8000

The `.devcontainer/devcontainer.json` configures a 4-core / 16GB
Codespace with Python 3.11 and all required packages.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
bash start.sh
```

Open http://localhost:8000.

## Dependencies

Core runtime (all in `requirements.txt`):

- **torch** — model inference (CPU-only recommended for Codespaces)
- **transformers** + **accelerate** + **safetensors** — HuggingFace model loading
- **fastapi** + **uvicorn** — HTTP server
- **python-multipart** — form data parsing (file uploads, analyze requests)
- **numpy** + **scipy** — numerical computation
- **matplotlib** — server-side plot rendering
- **scikit-learn** — PCA for domain surface module
- **aiofiles** — async static file serving
- **psutil** — system resource monitoring
- **huggingface-hub** — model downloads

**sqlite3** ships with Python's standard library — no additional
install needed for the database layer.

Optional: set `HF_TOKEN` environment variable for faster HuggingFace
downloads and to avoid rate limits.

## Architecture

```
tagm/engine/          TASM-native computation engine
  analyzer.py         Per-prompt extraction (stress, attribution, LTP, SFD)
  result.py           Flat PromptResult dataclass + serialization
  hooks.py            Adapter-based activation capture → flat key dict
  ltp.py              Lateral Tension Profile computation
  sfd.py              Spectral Field Density + Rank Displacement
  config.py           30 runtime parameters (frontend: Advanced Parameters)
  session.py          DB-backed session (SQLite via core/db.py)
  app_core.py         Core API endpoint handlers
  statistics.py       Bootstrap CIs, Cohen's d, threshold optimization
  modules/            TASM analysis modules (auto-discovered at startup)

tagm/core/            Model + data infrastructure
  adapter/            Model-family abstraction (Qwen2, Llama3)
  pipeline.py         Model loading, delta computation, forward passes
  deltas/             Weight delta computation from disk + spectral profiling
  cache.py            Disk cache management (~/.tagm/cache/)
  db.py               SQLite persistence layer

tagm/probes/          Probe CSV loading + per-depth embedding cache
tagm/service/         Chat, plots, export
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

**Config** — UI preferences and engine configuration in a key-value
store (replaces `ui_config.json`).

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

## Probe system

Probes enable the Domain Surface, Correction Heatmap, Correction
Manifold, and Correction Backscatter modules.

### Generation

The **Probe Generator** module (Modules tab) queries the instruct
model to build a discriminative vocabulary fingerprint per
class × subclass cell. It reads a template CSV, queries the model
N times per cell, counts token frequencies, then applies two-axis
deduplication (cross-class and cross-subclass) so that each
surviving term is unique to its lattice cell.

Templates live in `tagm/templates/`. Each template defines a
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

See `CROSS_MODEL.md` for the full cross-model comparison methodology.
In brief: the same probe CSV produces separate per-model embedding
caches, and the lattice geometry (subjects, levels) provides the
shared coordinate system for comparing cell-aggregated outputs
across models.

## API

Same as TASM. `POST /api/analyze` with form data:

```
prompt, category, compute_kl, compute_ltp, compute_sfd,
compute_trajectory, full_capture, capture_responses,
ltp_k, ltp_layer_strategy, ltp_svd_rank
```

Returns a flat result dict the frontend reads directly.

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
GET  /api/export                     Download session as gzipped JSON
```

The export format is unchanged — a gzipped JSON object with
`session_id`, `model`, `n_results`, and `results` array. Compatible
with any tooling that reads the previous export format.

## Indexed result columns

These scalar fields are stored in dedicated SQLite columns and
queryable without decompressing the result blob:

```
prompt, category, seq_len, stress_score, net_correction,
entropy, top2_share, middle_share, interior_cv, kl_divergence,
n_negative_tokens, has_negative_tokens, delta_scale,
full_capture_enabled, ltp_mean_m, ltp_mean_v, ltp_max_prc,
ltp_n_directional, sfd_density_mean, rd_mean_tau, rd_mean_overlap
```

All other fields (per-token arrays, heatmaps, trajectories, LTP
profiles, topk lists, etc.) are stored in the compressed blob and
loaded on demand when the frontend requests a specific result.
