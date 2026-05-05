# TAGM — Transformer Alignment Geometric Metrology

TASM-compatible backend rebuilt with clean architecture. Drop-in
replacement: same API, same data shapes, same frontend — better
internals.

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

Optional: set `HF_TOKEN` environment variable for faster HuggingFace
downloads and to avoid rate limits.

## Architecture

```
tagm/engine/     TASM-native computation engine
  analyzer.py    Per-prompt extraction (stress, attribution, LTP, SFD)
  result.py      Flat PromptResult dataclass + serialization
  hooks.py       Adapter-based activation capture → flat key dict
  ltp.py         Lateral Tension Profile computation
  sfd.py         Spectral Field Density + Rank Displacement
  config.py      30 runtime parameters (frontend: Advanced Parameters)
  session.py     Session storage (flat result dicts)
  app_core.py    Core API endpoint handlers
  statistics.py  Bootstrap CIs, Cohen's d, threshold optimization
  modules/       TASM analysis modules (auto-discovered at startup)

tagm/core/       Model infrastructure
  adapter/       Model-family abstraction (Qwen2, Llama3)
  pipeline.py    Model loading, delta computation, forward passes
  deltas/        Weight delta computation from disk + spectral profiling
  cache.py       Disk cache management

tagm/probes/     Probe CSV loading + per-depth embedding cache
tagm/service/    Chat, plots, export
```

## How it works

1. Load a model pair → Pipeline computes weight deltas from disk
2. Click Analyze → form-data flags go to the engine
3. Engine installs hooks (adapter resolves WHERE), runs one forward pass
4. Extraction functions read activations + deltas, write to flat PromptResult
5. result_to_dict() → JSON → frontend renders directly

The adapter tells us where to hook. The delta store tells us what changed.
The extraction functions read both and write to flat fields.

## Probe system

Probes enable the domain surface, correction heatmap, correction manifold,
and correction backscatter modules.

**Generation**: The Probe Generator module queries the instruct model to
build a discriminative vocabulary fingerprint per class × subclass cell.

**Embedding**: Probe terms are embedded through adapter-mediated hooks
(model-family agnostic) at configurable depths (L50, L75 by default).

**Apply flow**: Configuration tab → Probe Set → Choose File → Apply.
The backend embeds at both depths and caches to `probe_cache/`.

## API

Same as TASM. `POST /api/analyze` with form data:

```
prompt, category, compute_kl, compute_ltp, compute_sfd,
compute_trajectory, full_capture, capture_responses,
ltp_k, ltp_layer_strategy, ltp_svd_rank
```

Returns flat result dict the frontend reads directly.
