# TAGM — Transformer Alignment Geometric Metrology

A research-grade mechanistic-interpretability instrument: load an
instruct/base transformer pair, run prompts through it with parameterized
capture, extract a family of correction-field measurements, and produce
session-level analyses suitable for comparative plots and publishable
figures.

TAGM is the production rebuild of the experimental TASM prototype. The
measurements it produces are the same (within floating-point tolerance);
the architecture underneath is three clean layers with stable contracts
instead of accumulated scaffolding.

## Design in one paragraph

Three layers with one-way dependencies. The **instrument layer**
(`tagm/core/`) loads models, computes weight deltas tensor-at-a-time
from disk, and runs parameterized forward passes that populate a
canonical activation store. The **measurement layer**
(`tagm/measurement/`) declares per-module capture and probe requirements;
the framework unions them, runs one forward pass, and dispatches results
to each module. The **consumption layer** (`tagm/analysis/`,
`tagm/service/`, `static/`) consumes session records through a stable
schema — analysis modules aggregate across prompts, the FastAPI service
surfaces everything, the frontend renders. Adding a model family is one
new adapter. Adding a measurement is one new measurement class. Adding
an analysis is one new analysis class. Nothing else changes.

## Install and run

```bash
# 1. Set up a Python 3.10+ environment (3.12 recommended)
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set a HuggingFace token to speed up and de-rate-limit downloads
export HF_TOKEN=hf_yourTokenHere

# 4. Run
./start.sh
#   — or equivalently: python -m tagm
```

Open http://localhost:8000 in a browser.

## Design principles

- **Toaster, not spa.** Set the parameters, run the analysis, export,
  turn off. No background workers, no persistent state beyond loaded
  model + caches.
- **No hidden parameters.** Every value that affects a measurement is
  model-intrinsic, user-set, or documented. Resolved parameter values
  travel with the result so exported sessions are fully reproducible.
- **Single-pass capture, multi-consumer extraction.** One forward pass
  per prompt fills the activation store; many measurements read from
  it. Measurements never trigger their own forward passes silently.
- **Stdlib + minimal dependencies.** PyTorch, Transformers, NumPy,
  SciPy, FastAPI, safetensors, huggingface-hub, psutil. That's it.
- **Per-token alignment contract.** Every per-token array is length
  `seq_len`, indexed by raw token position, with NaN sentinels for
  undefined positions. The framework validates this before merging
  results into the session.

## Architecture

```
CONSUMPTION                                                    
  tagm/analysis/  comparative, mi_readiness, mi_instrumentation,
                  token_variance, correction_heatmap,           
                  correction_manifold, correction_backscatter,  
                  domain_surface, correction_field_topology     
  tagm/service/   FastAPI app, orchestrator, session record,   
                  gzipped-JSON export                          
  static/         multi-page frontend (index + vizzes)         
        ↓ reads from session via stable schema                 
MEASUREMENT                                                    
  tagm/measurement/modules/  last_position_attribution,        
                             stress_score, amplitude_trajectory,
                             amplitude_derived_metrics,        
                             lateral_tension_profile,          
                             spectral_field_density,           
                             rank_displacement,                
                             probe_projection, per_token_embedding,
                             backscatter_projection            
  tagm/probes/               template parser, generator, store 
        ↓ reads adapter + activation store + delta store       
INSTRUMENT                                                     
  tagm/core/adapter/   ModelAdapter base + Qwen2 + Llama3      
  tagm/core/capture/   CaptureConfig, ActivationStore, hook    
                        installer                              
  tagm/core/deltas/    DeltaStore, compute_from_disk, spectral 
  tagm/core/pipeline.py                                        
  tagm/core/cache.py                                           
```

## A typical workflow

1. `POST /api/load` with a model pair (from `models.json` or a custom
   pair id) — loads instruct, detects family adapter, streams base
   weights from disk, computes deltas, runs spectral precomputes.

2. `POST /api/configure` with a list of measurements plus their
   parameters — the orchestrator validates the selection and unions
   capture requirements.

3. (Optional) `POST /api/probes/generate` with a template name and
   depth layers — generates probe embeddings, writes to the content-
   addressed probe store.

4. `POST /api/analyze` for a single prompt, or `POST /api/batch` for
   a list — runs forward pass(es), dispatches to measurements,
   validates the per-token contract, merges into the session.

5. `POST /api/analysis/{name}` to run a post-session analysis —
   comparative, manifold, heatmap, etc.

6. `GET /api/session/export` to download the gzipped-JSON session.

See `NOTES.md` for translation notes, deferred items, and design
decisions made during the TASM → TAGM migration.

## Model pairs

Declared in `models.json` at the repo root. Each entry is
`{id, base, instruct, name}`. Custom pairs can also be passed to
`POST /api/load` by providing `instruct_id` and `base_id` directly.

## Caches

Everything TAGM persists lives under `~/.tagm/cache/` (override with
`TAGM_CACHE_DIR`):

- `presets/`: saved CaptureConfig presets (JSON).
- `probes/`: content-addressed probe sets (`.npz` + `.json` sidecar).
- `svd/`: reserved for disk-backed SVD precomputes.
- `sessions/`: exported session records (`.json.gz`).

`GET /api/cache_usage` (part of `/api/status`) reports per-subdir byte
totals.

## License

TBD — accompanies the forthcoming paper.
