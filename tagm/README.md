# TAGM — Transformer Alignment Geometric Metrology

TASM-compatible backend rebuilt with clean architecture. Drop-in
replacement: same API, same data shapes, same frontend — better
internals.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./start.sh
```

Open http://localhost:8000.

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

tagm/core/       Model infrastructure
  adapter/       Model-family abstraction (Qwen2, Llama3)
  pipeline.py    Model loading, delta computation, forward passes
  deltas/        Weight delta computation from disk + spectral profiling
  cache.py       Disk cache management

tagm/analysis/   Post-session analysis modules (read from session)
tagm/probes/     Probe template system
tagm/service/    Chat, plots, export, module runner
```

## How it works

1. Load a model pair → Pipeline computes weight deltas from disk
2. Click Analyze → form-data flags go to the engine
3. Engine installs hooks (adapter resolves WHERE), runs one forward pass
4. Extraction functions read activations + deltas, write to flat PromptResult
5. result_to_dict() → JSON → frontend renders directly

No CaptureConfig. No MeasurementResult. No orchestrator. No translation shim.
The adapter tells us where to hook. The delta store tells us what changed.
The extraction functions read both and write to flat fields. Done.

## API

Same as TASM. `POST /api/analyze` with form data:

```
prompt, category, compute_kl, compute_ltp, compute_sfd,
compute_trajectory, full_capture, capture_responses,
ltp_k, ltp_layer_strategy, ltp_svd_rank
```

Returns flat result dict the frontend reads directly.

## License

TBD — accompanies the forthcoming paper.
