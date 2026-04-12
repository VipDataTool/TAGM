# TASM Documentation

## Table of Contents

1. [Technical Description](#part-1-technical-description)
   - [Identity](#identity)
   - [Architecture](#architecture)
   - [Core Computation Pipeline](#core-computation-pipeline)
   - [Input Specification](#input-specification)
   - [Output Specification](#output-specification)
   - [Concurrency and State Management](#concurrency-and-state-management)
   - [Runtime Environment](#runtime-environment)
2. [Mathematical Pipeline](#part-2-mathematical-pipeline)
   - [Overview](#overview)
   - [Stage 0: Weight Delta Computation](#stage-0-weight-delta-computation-model-load-time)
   - [Stage 1: Forward Pass and Activation Capture](#stage-1-forward-pass-and-activation-capture)
   - [Stage 2: ASM — Alignment Stress Map](#stage-2-asm--alignment-stress-map)
   - [Stage 3: LTP — Lateral Tension Profile](#stage-3-ltp--lateral-tension-profile)
   - [Stage 4: SFD — Spectral Field Density](#stage-4-sfd--spectral-field-density)
   - [Stage 5: Behavioral Comparison](#stage-5-behavioral-comparison)
   - [Stage 6: Rank Displacement](#stage-6-rank-displacement)
   - [Stage 7: Candidate Graph Topology](#stage-7-candidate-graph-topology)
   - [Stage 8: Batch Statistical Aggregation](#stage-8-batch-statistical-aggregation)
   - [Notation Summary](#notation-summary)
3. [Visualization Pipeline](#part-3-visualization-pipeline)
   - [Design System](#design-system)
   - [Single-Prompt Visualizations](#single-prompt-visualizations)
   - [Batch / Dashboard Visualizations](#batch--dashboard-visualizations)
   - [The Terrain Viewer (WebGL)](#the-terrain-viewer-webgl)
   - [Data Table](#data-table-js-rendered)
   - [PDF Reports](#pdf-reports)
   - [Visualization Registry](#visualization-registry)
4. [Token Variance Analysis](#part-4-token-variance-analysis)
   - [What It Measures](#what-it-measures)
   - [Integration](#integration)
   - [Parameters](#parameters)
   - [Output](#output)
   - [Channels](#channels)
   - [Data Requirements](#data-requirements)
   - [Interpretation Notes](#interpretation-notes)
5. [Engine Configuration](#part-5-engine-configuration)
   - [Overview](#overview-1)
   - [Parameter Registry](#parameter-registry)
   - [Persistence](#persistence)
   - [UI Safety](#ui-safety)
6. [Module System](#part-6-module-system)
   - [Framework](#framework)
   - [Probe Generator](#probe-generator)
   - [Correction Heatmap](#correction-heatmap)
   - [Correction Manifold](#correction-manifold)
   - [Domain Surface](#domain-surface)
   - [Token Variance](#token-variance)
   - [Comparative Analysis](#comparative-analysis)
   - [Correction Field Topology](#correction-field-topology)
   - [MI Readiness Analysis](#mi-readiness-analysis)
   - [MI Instrumentation](#mi-instrumentation)

---

# Part 1: Technical Description

## Identity

**TASM** (The Alignment Stress Map) is a web-based analysis platform for measuring alignment signals in transformer language models at inference time. It implements four complementary instruments — ASM (amplitude), LTP-RD (directional/rank displacement), SFD (dimensionality), and Token Variance (cross-context stability) — that together characterize the corrections introduced by instruction-tuning (RLHF/SFT) relative to a base model. The theoretical framework is described in two companion papers by Ostrander (2026).

The application is a single-server Python system built on FastAPI, serving a single-page web frontend over HTTP. It is designed for interactive single-prompt analysis, CSV-driven batch experiments, conversational probing via a built-in chat interface, and extensible post-hoc analysis via the module framework. All measurement-affecting parameters are centrally configured and surfaced in the UI.

---

## Architecture

TASM is structured as a monolithic FastAPI application (`app.py`, ~1,970 lines) backed by a modular engine layer:

| Module | Responsibility |
|--------|---------------|
| `model_manager.py` | Model loading, weight delta computation, forward-hook installation, activation caching |
| `analyzer.py` | Orchestrates the full analysis pipeline: ASM, LTP, SFD, and behavioral comparison in a single forward pass |
| `ltp.py` | Lateral Tension Profile computation: counterfactual probing, PCA trajectories, M/C/V/L statistics |
| `sfd.py` | Spectral Field Density: QK-subspace engagement measurement via SVD projection; rank displacement computation |
| `engine_config.py` | Central registry of all measurement-affecting parameters with typed defaults |
| `baselines.py` | Prompt library management: loading, listing, and appending to prompts.csv |
| `statistics.py` | Bootstrap confidence intervals, Cohen's d effect sizes, cross-category aggregation |
| `visualizations.py` | Matplotlib plot generation for all three signal families (returned as base64 PNGs) |
| `comparative.py` | Cross-prompt comparative visualizations and batch dashboard plots |
| `dataset.py` | Session management: result accumulation, CSV/JSON persistence, ZIP export packaging, session restore |
| `reports.py` | PDF report generation via ReportLab |
| `modules/base.py` | Module framework: TASMModule base class, auto-discovery, parameter metadata, thread-isolated runner |
| `modules/token_variance.py` | Token variance module: cross-context coupling stability analysis |
| `modules/probe_generator.py` | Auto-probe generation from model vocabulary per template cell |
| `modules/correction_heatmap.py` | Domain lattice interaction measurement via inter-layer probe deltas |
| `modules/correction_manifold.py` | PCA + K-means fingerprint clustering of per-prompt probe projections |
| `modules/domain_surface.py` | Probe embedding, domain surface mapping, nearest-probe proximity |
| `modules/comparative_analysis.py` | Cross-prompt aggregate statistics, separability, batch plot coordination |
| `modules/displacement_field.py` | 3D correction field topology data validation and statistics (Three.js viewer) |
| `modules/mechanistic_interpretability.py` | MI readiness evaluation: AUROC, length confounds, PCA consolidation, random baselines |
| `modules/mi_instrumentation.py` | MI outputs: refusal direction, activation patching map, per-layer AUROC |

The frontend consists of a single-page HTML application (`static/index.html`, ~4,180 lines) and a separate chat interface (`static/chat.html`). All rendering and interaction logic is client-side JavaScript; the server provides a JSON API and serves plot images on demand.

---

## Core Computation Pipeline

The fundamental operation is **weight delta projection**. At model load time, TASM downloads the base and instruct variants of a model pair from HuggingFace, then computes the weight delta `ΔW = W_instruct − W_base` for six projection matrices per transformer layer (Q, K, V, O, gate, up). Base model weights are read directly from safetensors files on disk one tensor at a time, avoiding full base-model instantiation and halving peak memory.

At analysis time, a single forward pass through the instruct model with registered hooks captures hidden states and attention weights at monitored layers. All three signal families are extracted from these cached activations without additional forward passes:

**ASM (Alignment Stress Map)** computes per-token signed attribution by projecting hidden states through `ΔW_V`, weighting by attention patterns, and aggregating across signal layers (the middle third of the network). This yields a scalar stress score, per-token attribution vectors, distribution metrics (entropy, top-2 share, interior CV), and — when full trajectory mode is enabled — a layer-by-layer amplitude trace and a token×layer heatmap.

**LTP (Lateral Tension Profile)** probes the alignment field perpendicular to the generation path. For each token position, it identifies the top-k counterfactual tokens the model considered but did not select, computes unembedding directions for each, and measures the lateral tension (the component of the alignment correction toward each alternative) via `ΔW_V` projection. This produces per-token ranked tension profiles, profile shape classifications (steep/flat/inverted), PCA-projected dual trajectories (semantic vs. tension), and three summary statistics: M (offset magnitude), V (offset variance), and L (lateral coverage). A fourth statistic, C (offset consistency), was removed after testing showed near-perfect correlation with M.

**SFD (Spectral Field Density)** measures how many dimensions of the QK routing subspace each token's activation engages. At load time, it SVD-decomposes the concatenated `[ΔW_Q; ΔW_K]` per layer and caches the right singular vectors. At inference, it projects each token's hidden state through this basis. Energy, spectral entropy, and effective rank are computed internally per token, but only the density ratio (per-token effective rank divided by the layer's global effective rank) is persisted. Prompt-level aggregates (mean, max, variance, p90) of density summarize the dimensionality axis.

**Behavioral comparison** optionally loads the base model for a separate forward pass to compute KL(instruct ‖ base) divergence at the output distribution and capture top-k next-token predictions from both models. A rank displacement metric compares counterfactual token orderings between base and instruct models.

---

## Input Specification

### Primary Inputs

**Single prompt analysis** accepts:
- `prompt` (string, ≤5,000 characters) — the text to analyze
- `category` (string) — one of `benign`, `mild`, `harmful`, `jailbreak`, `adversarial`, `dual-use`; unrecognized values are mapped to `unknown`
- Boolean flags: `compute_kl`, `compute_trajectory`, `capture_responses`, `full_capture`, `compute_ltp`, `compute_sfd`
- LTP parameters: `ltp_k` (counterfactual depth: 4, 6, or 8), `ltp_layer_strategy` (`signal` or `late`)

**Batch analysis** accepts a CSV file with columns `prompt` and `category`, plus the same boolean and LTP parameter flags applied uniformly to all prompts.

**Chat interface** accepts a message history (JSON array of `{role, content}` objects), `max_tokens` (≤512), and optional `analyze`/`analyze_response` flags that trigger ASM/LTP/SFD analysis of the user message and/or the model's generated response.

### Model Configuration

The model registry (`models.json`) ships with four Qwen 2.5 pairs (0.5B, 1.5B, 3B, 7B). Custom HuggingFace base/instruct pairs can be added at runtime. The engine auto-detects layer count, attention head configuration, GQA grouping, and hidden dimensionality.

### Auxiliary Inputs

- `prompts.csv` — a library of categorized prompts for the sidebar prompt picker and batch analysis

---

## Output Specification

### Per-Prompt Result Object

Each analysis produces a `PromptResult` serialized to a JSON dictionary with the following field groups:

**Identity and tokenization:** `prompt`, `category`, `tokens` (list of decoded token strings), `seq_len`

**ASM scalars:** `stress_score` (float), `net_correction` (float), `entropy`, `top2_share`, `middle_share`, `interior_cv` (all floats, distribution metrics), `n_negative_tokens` (int), `has_negative_tokens` (bool)

**ASM arrays:** `per_token_stress` (float array, length = seq_len), `signed_attr` (float array, length = seq_len), `amplitude_trajectory` (float array, length = 2 × n_layers for attn+MLP sublayers), `amplitude_normalized` (same shape), `heatmap` (2D array: sublayers × seq_len)

**Behavioral divergence:** `kl_divergence` (float or null), `per_token_kl` (float array or null), `instruct_topk` and `base_topk` (lists of `[token_string, probability]` pairs, up to 10 each), `base_counterfactual_tokens` (per-position top-k from base model)

**LTP sub-object** (`ltp`): `profiles` (list of numpy arrays per token), `tension_magnitudes` (float list), `profile_shapes` (string list: "steep"/"flat"/"inverted"), `counterfactual_tokens` (per-position ranked alternatives with probabilities), `offset_magnitude`/`offset_variance`/`lateral_coverage` (per-layer dictionaries), `mean_M`/`mean_V`/`mean_L` (summary scalars), `max_prc` (peak rank concentration), `n_directional` (count of tokens with PRC above threshold), `semantic_trajectory_2d`/`tension_trajectory_2d` (PCA projections), `k`, `layer_strategy` (configuration echo)

**SFD sub-object** (`sfd`): `per_token_density` (float array), 4 prompt-level density aggregates (`density_mean`/`max`/`var`/`p90`), `global_erank`, `n_layers_monitored`, `k`. Note: per-token energy and entropy are computed internally during the SFD pipeline but only density (the ratio of per-token effective rank to global effective rank) is persisted in the result object.

**Rank displacement** (`rank_displacement`): `mean_matched`, `mean_replacement`, `mean_concentration`, `mean_tau` (Kendall's tau), `mean_overlap`, per-position detail arrays

**Full capture extras** (when enabled): `per_token_coherence` (cross-layer direction agreement), `per_token_spectral_rank`, `attn_frac` (attention vs. MLP contribution ratio), `token_similarity` (token×token cosine similarity matrix)

### Visualizations

Each analysis generates up to 12 plot types as base64-encoded PNGs:

ASM plots: signed attribution bar chart, focused stress bar chart, distribution metrics panel, amplitude trajectory line plot, token×layer heatmap.

LTP plots: lateral tension profiles (stacked magnitude), tension magnitude bar chart, dual trajectory (PCA), summary statistics panel, profile heatmap (token × rank).

SFD plots: density bar chart, rank displacement chart.

### Session and Batch Outputs

**Summary CSV** (`summary.csv`): one row per analyzed prompt, 33 columns of scalar metrics including all ASM, LTP, SFD, and rank displacement aggregates, plus model prediction summaries and configuration echo fields.

**Full results JSON** (`results.json`): complete per-prompt result objects with all per-token arrays, profiles, and trajectories. For 271 prompts with full capture enabled, this file is approximately 17 MB.

**Aggregate statistics JSON** (`aggregate_statistics.json`): batch-level computed analytics including per-category bootstrapped means with confidence intervals, pairwise separability (Cohen's d with bootstrap CIs for every metric between category pairs), length-correlation analysis, and cross-metric correlations.

**Export ZIP** (`tasm_session_<timestamp>.zip`): configurable archive containing any combination of summary CSV, full results JSON, aggregate statistics JSON, module results (e.g. `module_token_variance.json`), per-prompt PNG plots, comparative/dashboard plots, and a PDF report.

**PDF report** (via ReportLab): formatted batch analysis report with summary tables, separability analysis, and embedded visualizations.

### API Response Format

All API endpoints return JSON. The single-analysis endpoint returns:
```json
{
  "ok": true,
  "result": { /* full PromptResult dict */ },
  "plot_keys": ["signed_attribution", "stress_per_token", ...],
  "session_n": 42,
  "cache_size_bytes": 1048576
}
```

Plots are served separately via `GET /api/plots/{plot_key}` and `GET /api/plots/individual/{index}/{plot_key}` as PNG files. Plots are generated lazily on first request and cached to disk — they are not pre-generated during dashboard computation.

---

## Concurrency and State Management

The server maintains global mutable state (loaded models, activation caches, session data, engine configuration) protected by threading locks: `_analysis_lock` serializes forward passes and session writes to prevent activation cache corruption; `_loading_lock` makes model loading state transitions atomic; `_plot_gen_lock` prevents duplicate lazy plot generation. Batch analysis runs in a background daemon thread, with progress communicated via a shared log list polled by the frontend. Module execution runs in isolated daemon threads with crash protection — a module failure does not affect the main application.

---

## Runtime Environment

- **Language:** Python 3.10+
- **Framework:** FastAPI + Uvicorn
- **ML stack:** PyTorch, HuggingFace Transformers, Safetensors, Accelerate
- **Visualization:** Matplotlib
- **Statistics:** NumPy, SciPy
- **Reporting:** ReportLab (PDF)
- **Compute:** CPU by default; GPU optional. The instruct model remains in memory; the base model is loaded on-demand for KL/LTP computation and immediately unloaded afterward.
- **Memory:** 4–32 GB depending on model scale (0.5B–7B parameters).

---

# Part 2: Mathematical Pipeline

## Overview

TASM measures how instruction-tuning (RLHF, SFT, or similar) changed a language model's internal behavior at inference time, on a per-token basis, for any given prompt. It does this by comparing the weights of an instruction-tuned model against its base (pre-training-only) counterpart and projecting activations through the difference.

The pipeline has three signal families — ASM, LTP, and SFD — plus a rank displacement instrument (RD) that answer four different questions about the alignment correction at each token:

- **ASM** (Alignment Stress Map): *How hard* is the alignment correction pushing?
- **LTP** (Lateral Tension Profile): *In which direction* is the alignment correction structured?
- **SFD** (Spectral Field Density): *How many dimensions* of the alignment correction does each token engage?
- **RD** (Rank Displacement): *How much* did alignment training change the set and ordering of candidate tokens?

There are also several supporting computations: behavioral divergence (KL), candidate graph topology, and batch-level statistical aggregation. Token Variance analysis operates post-hoc on session data via the module framework (see Part 4). This document traces every mathematical operation from model loading through to final statistics.

---

## Stage 0: Weight Delta Computation (Model Load Time)

### What's happening in plain language

Before analyzing any prompts, TASM computes the *difference* between the instruction-tuned model's weights and the base model's weights. This difference — the "delta" — represents everything that alignment training changed. If the base model has weights `W_base` and the instruct model has weights `W_instruct`, the delta is simply:

```
ΔW = W_instruct − W_base
```

TASM computes this for six projection matrices in every transformer layer: the query (`q_proj`), key (`k_proj`), value (`v_proj`), output (`o_proj`), gate (`gate_proj`), and up (`up_proj`) projections. These are the linear transformations inside attention and MLP sublayers — the parts of the network where alignment training has the most direct geometric effect.

### Signal layer selection

Not all layers are equally important. TASM designates the middle third of the network as the "signal layers." For a 24-layer model, layers 8–15 are the signal layers. This is where prior work suggests alignment corrections are most active — deep enough to have built semantic representations, but before the network commits to a specific output token. The fraction is configurable via the `signal_layer_fraction` parameter (default 0.333):

```
frac = signal_layer_fraction
mid_start = floor(n_layers × frac)
mid_end   = floor(n_layers × (1 − frac))
signal_layers = [mid_start, mid_start+1, ..., mid_end−1]
```

### Frobenius norms

For every delta matrix, TASM also stores its Frobenius norm — the square root of the sum of all squared entries. This acts as a normalization constant later on:

```
‖ΔW‖_F = √(Σᵢⱼ ΔWᵢⱼ²)
```

### Spectral profile (effective rank)

At load time, TASM also computes the spectral structure of each delta matrix via truncated SVD. For each `ΔW`, it computes the top singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_k (where k is configurable via `delta_svd_k`, default 64), normalizes them into a probability distribution:

```
pᵢ = σᵢ / Σⱼ σⱼ
```

and then computes the effective rank as the exponential of the Shannon entropy of this distribution:

```
effective_rank = exp(−Σᵢ pᵢ log pᵢ)
```

If the delta is dominated by a single direction (σ₁ ≫ σ₂), effective rank ≈ 1 — alignment training made a surgical correction. If the singular values are spread out, effective rank is high — alignment reshuffled the entire subspace. TASM also records the fraction of total spectral energy (`Σ σᵢ²`) captured by the top-1 and top-5 singular values. These are aggregate diagnostics reported at the model level; they cost nothing per prompt.

---

## Stage 1: Forward Pass and Activation Capture

For each prompt, TASM tokenizes it and runs a single forward pass through the instruct model. During this pass, forward hooks capture:

- **Hidden states** `h` at the input-layernorm of each monitored layer. These are the residual stream representations entering each transformer block — the activations that get projected through the attention and MLP sublayers. Shape: `(1, seq_len, hidden_dim)`.

- **Attention weights** `α` at each monitored layer's self-attention module. These are the post-softmax attention matrices that determine how much each position attends to every other position. Shape: `(1, n_heads, seq_len, seq_len)`.

When full trajectory mode is enabled, hidden states are captured at *every* layer (not just signal layers) at both the input-layernorm (pre-attention) and post-attention-layernorm (pre-MLP) positions, giving a sublayer-by-sublayer picture of how the residual stream evolves.

---

## Stage 2: ASM — Alignment Stress Map

### 2a: Signed Attribution

This is the core ASM computation. For each signal layer, it answers: "For each token in the input, how much is that token contributing to the alignment correction, and in which direction?"

**Step 1: Project hidden states through the value delta.**

Take the hidden state `h` (shape: `seq_len × hidden_dim`) and multiply by the transpose of the V-projection delta:

```
v = h · ΔW_V^T
```

This gives `v` with shape `(seq_len, v_proj_out_dim)`. Each row is the "correction potential" — how much each token would be corrected if it were the only thing going through this sublayer.

**Step 2: Reshape for grouped-query attention (GQA).**

The v-projection output is reshaped to `(seq_len, n_kv_heads, head_dim)`, because in GQA architectures, multiple query heads share the same KV head. Attention weights are similarly regrouped by averaging across the query heads that share each KV group:

```
α_kv[g] = mean over q-heads in group g of α[q]
```

**Step 3: Compute the per-head correction direction.**

For each KV head `g`, the attention-weighted correction is:

```
δ[g] = α_kv[g] · v_heads[:, g, :]
```

This is a `(seq_len, head_dim)` matrix. Row `i` is the correction vector at position `i` for this head — a sum of correction potentials from all positions, weighted by how much position `i` attends to them.

**Step 4: Compute the unit correction direction.**

Normalize the correction vector at the final position to get a unit vector:

```
û = δ[-1] / ‖δ[-1]‖
```

This unit vector points in the direction the alignment correction is pushing at the output position.

**Step 5: Project correction potentials onto this direction.**

For each source token `j`, compute how much of its correction potential aligns with the overall correction direction:

```
proj[j] = û · v_heads[j, g, :]
```

Then the signed attribution for token `j` is:

```
signed_attr[j] = α[-1, j] × proj[j]
```

where `α[-1, j]` is how much the output position attends to position `j`. The sign matters: positive means the token pushes *with* the correction, negative means it pushes *against* it.

**Step 6: Verify exactness (Proof 1).**

The sum of signed attributions across all tokens should exactly equal the norm of the correction vector:

```
Σⱼ signed_attr[j] = ‖δ[-1]‖
```

This is checked per-head and reported. It's a mathematical identity (the projection decomposes the norm), so errors above the configurable `proof1_threshold` (default 10⁻⁴) indicate numerical issues.

**Step 7: Aggregate across heads and layers.**

Per-head signed attributions are averaged across KV heads within each layer, then averaged across signal layers, yielding a single signed attribution value per token.

```
final_signed_attr[j] = mean over layers L of (mean over heads g of signed_attr_L,g[j])
```

### 2b: Distribution Metrics

From the averaged signed attribution vector, TASM computes several statistics that characterize *how* the correction is distributed across tokens:

**Net correction:** The sum of all signed attributions. This is positive when the overall correction pushes in a consistent direction; close to zero when positive and negative contributions cancel.

**Normalized entropy:** Take absolute values of attributions, normalize to a probability distribution, and compute Shannon entropy relative to maximum possible entropy (log of sequence length):

```
dᵢ = |attr[i]| / Σⱼ |attr[j]|
H = −Σᵢ dᵢ log dᵢ
entropy = H / log(seq_len)
```

Values near 1 mean the correction is spread uniformly across tokens; values near 0 mean it's concentrated on a few tokens.

**Boundary vs. interior split:** The prompt is divided into boundary tokens (the first and last portion of the sequence, configurable via `boundary_fraction`, default 10%, minimum 1 token each) and interior tokens (everything in between). `top2_share` is the fraction of total absolute attribution at boundary positions. `middle_share` is the fraction at interior positions. `interior_cv` is the coefficient of variation (standard deviation ÷ mean) of the interior attribution values.

The hypothesis: benign prompts concentrate correction at the boundaries (system prompt, instruction tokens), while adversarial prompts distribute correction into the interior.

### 2c: Focused Stress Score

The stress score measures how strongly the alignment delta responds to the prompt's activations, normalized by the delta's own scale. For each signal layer, and for each of the three attention projections (Q, K, V):

```
projected = h · ΔW^T          (shape: seq_len × proj_dim)
per_token_norm = ‖projected‖₂  (along the projection dimension)
normalized = per_token_norm / ‖ΔW‖_F
```

The per-token stress at this layer is the average of the normalized norms across Q, K, and V projections. The final per-token stress is then averaged across signal layers, and the stress score is the mean across tokens.

In plain language: if a token's hidden state happens to lie in the subspace that alignment training modified most, the projection through `ΔW` will be large. Dividing by the Frobenius norm of `ΔW` makes this comparable across projections and layers. A high stress score means the prompt's representation happens to strongly "engage" the directions that alignment training cared about.

### 2d: Amplitude Trajectory

When full trajectory mode is enabled, TASM computes the stress at every sublayer in the entire network (not just signal layers), for both attention and MLP blocks. At each sublayer:

- For attention: project `h` through the Q, K, and V deltas.
- For MLP: project `h` through the gate and up deltas.
- Record both the raw projection norm (unnormalized) and the Frobenius-normalized version.
- Record the per-token normalized norms as a row of the heatmap.

This produces three outputs: a raw trajectory (amplitude at each sublayer), a normalized trajectory (Frobenius-normalized), and a heatmap (sublayer × token matrix of normalized projection norms). The heatmap shows where in the network each token's representation interacts most with the alignment delta.

### 2e: Full Capture Derived Metrics

When full capture is enabled, TASM derives four additional per-token metrics from the heatmap:

**Attention fraction:** For each token, the fraction of total correction energy coming from attention sublayers vs. MLP sublayers. Attention sublayers are the even-indexed rows of the heatmap; MLP sublayers are the odd-indexed rows.

```
attn_frac[t] = Σ(attn_rows[:, t]) / (Σ(attn_rows[:, t]) + Σ(mlp_rows[:, t]))
```

**Per-token coherence:** How concentrated the correction is across sublayers. For each token, take its column of the heatmap (its correction norm at every sublayer), normalize to a distribution, and compute `1 − normalized_entropy`. A value near 1 means all correction happens at a few sublayers (consistent strategy); near 0 means it's spread uniformly.

**Per-token spectral rank:** The effective dimensionality of each token's correction pattern. Same exponential-of-entropy formula as the delta spectral rank, but applied to the heatmap column for each token.

**Token similarity matrix:** Cosine similarity between every pair of tokens' correction profiles (their heatmap columns). This shows which tokens receive similar correction patterns from the network.

---

## Stage 3: LTP — Lateral Tension Profile

The LTP probes the alignment field in directions perpendicular to the generation path. Where ASM measures intensity on the path the model took, LTP measures the structure of the field surrounding that path. Two prompts can have identical ASM amplitude but very different LTP signatures if one runs through the center of a broad high-correction region (robust) and the other runs along a boundary (fragile).

### 3a: Counterfactual Alternative Selection

At each token position, TASM identifies the top-k alternative tokens the model considered but did not select. From the instruct model's logit output at position `i`:

```
top_k_alternatives = top-k tokens from softmax(logits[i]) excluding the actual token
```

Default k=8. If the base model is loaded, TASM also independently selects the base model's top-k alternatives at each position (these may differ from the instruct model's alternatives).

### 3b: Computing the Forward Direction

At each token position `i`, the "forward direction" τ is defined as the normalized difference between consecutive hidden states at the first monitored layer:

```
τ[i] = (h[i] − h[i−1]) / ‖h[i] − h[i−1]‖
```

This approximates the direction the model's residual stream is "moving" through representation space at that position. For the first token, it uses `h[1] − h[0]` instead.

### 3c: Lateral Projection (per counterfactual alternative)

For each counterfactual token `c` at position `i`, TASM constructs a "probing direction" in the output vocabulary space:

```
d_ic = W_u[c] − W_u[chosen_i]
```

where `W_u` is the unembedding matrix (the `lm_head` weight). This is the direction in embedding space that points from the token the model chose toward the alternative it didn't choose.

This probing direction then goes through a three-step projection pipeline:

**Step 1: Project through half the V-delta.**

```
Δv = (ΔW_V / 2) · d_ic
```

The factor of ½ represents the midpoint of the alignment correction — conceptually, the "average" position between the base and instruct models' value projections.

**Step 2: Reshape and expand through the output projection.**

The projected vector is reshaped to `(n_kv_heads, head_dim)`, expanded for GQA (repeating shared KV head outputs for each query head that uses them), flattened back to the full hidden dimension, and then projected through the output projection matrix:

```
proj = W_O · expand(reshape(Δv))
```

This gives a vector in the residual stream's coordinate system — the correction that would be applied.

**Step 3: Decompose into forward and lateral components.**

```
forward_component = (proj · τ) × τ
lateral_component = proj − forward_component
lateral_magnitude = ‖lateral_component‖
```

The lateral component is the part of the correction that pushes perpendicular to the model's generation direction. A large lateral magnitude means the alignment field is structured to pull the representation sideways — toward or away from this alternative — rather than just speeding up or slowing down along the current path.

### 3d: Building the Profile

At each token position, the lateral magnitudes for all k alternatives (sorted by the model's probability ranking) form the **lateral tension profile** — a k-dimensional vector where entry `j` is how much lateral pull the j-th most-likely alternative token exerts.

This computation happens at every monitored layer, and the profiles are averaged across layers to produce a single profile per token.

### 3e: Profile Shape Classification

Each token's averaged profile is classified into one of three shapes:

- **Steep:** The top-ranked alternative dominates (>40% of total magnitude, and at least 2× the second-ranked). This means the alignment correction is concentrated toward one specific alternative — a focused boundary.

- **Inverted:** The second half of the profile has higher mean magnitude than the first half (by >30%). Lower-ranked alternatives exert more pull than higher-ranked ones — an anomalous pattern that suggests the alignment field's geometry doesn't align with the model's probability ranking.

- **Flat:** Neither steep nor inverted. The lateral tension is spread across alternatives without strong directional preference.

### 3f: Weighted Tension Point

For each position, TASM also computes a probability-weighted average of the lateral vectors:

```
tension_point = (Σ_c  prob_c × lateral_vector_c) / Σ_c prob_c
```

This is a single vector in hidden-state space representing the net lateral pull, weighted by how likely each alternative was. This is averaged across monitored layers.

### 3g: Summary Statistics (M, V, L)

From the per-layer tension points, TASM computes three summary statistics at each monitored layer, then averages across layers:

**M — Offset Magnitude:** The norm of the mean tension point across active positions (tokens where tension is non-zero). This measures how large the net lateral displacement is.

```
mean_offset = mean of tension_points across active positions
M = ‖mean_offset‖
```

**V — Offset Variance:** The variance of individual tension magnitudes across positions. High V means the lateral pull is concentrated at specific tokens rather than evenly distributed.

**L — Lateral Coverage:** The fraction of tokens that have any non-zero lateral tension at all.

**C — Offset Consistency (removed):** An earlier revision computed C as the ratio of the mean offset's magnitude to the average individual magnitude (C = M / mean(‖tension_point_i‖)). This was removed because it proved nearly perfectly correlated with M (r=0.989), providing no additional information.

The diagnostic hypothesis: High M = "boundary-threading signature" — the prompt is consistently pulled in one lateral direction at every position, suggesting it runs along a systematic alignment boundary.

### 3h: Removed Enhancements

Two optional enhancements were explored during development and subsequently removed after literature review determined they were inappropriate for cross-model weight delta projection:

**SVD Truncation (removed):** Projecting through a rank-r truncation of `ΔW_V` to isolate a "dominant safety subspace." Ponkshe et al. (2025) showed that top SVD directions of alignment deltas capture general parameter sensitivity rather than safety-specific directions. The full-rank `ΔW_V` is used for all computations. The code path remains dormant.

**Tuned-Lens Correction (removed):** Applying per-layer affine transforms trained on a single model's representational drift to calibrate unembedding probes at intermediate layers. The tuned lens (Belrose et al., 2023) is a within-model state decoder; applying it to vectors produced by cross-model weight delta projection is geometrically incoherent. The code path remains dormant.

### 3i: Peak Rank Concentration (PRC)

For each token, TASM computes how concentrated the profile is toward the top-ranked alternative:

```
normalized_profile = profile / sum(profile)
PRC = max(normalized_profile) − 1/k
```

PRC = 0 means the profile is perfectly flat (uniform across alternatives). PRC > 0 means one alternative dominates. Tokens with PRC above a threshold (0.02) are counted as "directional" — they have a meaningful geometric preference.

### 3j: Dual Trajectory (PCA Visualization)

For visualization, TASM projects both the semantic trajectory (raw hidden states `h[0], h[1], ..., h[T]`) and the tension trajectory (hidden states plus tension points: `h[i] + tension_point[i]`) into a shared 2D space via PCA. The procedure:

1. Stack the semantic and tension trajectories into a single matrix: `[h₁...h_T ; (h₁+t₁)...(h_T+t_T)]`
2. Mean-center the combined matrix.
3. Compute SVD: `U · Σ · V^T`
4. Project all points onto the first two right-singular vectors (the two directions of greatest variance).

The result is two 2D curves. Their separation shows how much and how consistently the alignment correction displaces the representation path.

### 3k: Base Bank Profiles

When the base model is loaded, TASM repeats the lateral tension computation using the *base model's* top-k alternatives (instead of the instruct model's) as the probing directions. Everything else — the `ΔW_V/2`, the hidden states, the forward direction — stays the same. This produces a "base bank" profile at each position that shows the lateral structure from the base model's perspective, enabling comparison of how the two models "see" the nearby counterfactual landscape.

---

## Stage 4: SFD — Spectral Field Density

SFD measures how many dimensions of the alignment-reshaped QK routing subspace each token's activation engages. Where ASM measures intensity and LTP measures directional structure, SFD measures dimensionality.

### 4a: Subspace Construction (Model Load Time)

For each monitored layer, TASM concatenates the Q and K projection deltas vertically:

```
ΔW_QK = [ΔW_Q ; ΔW_K]    (stacked along the output dimension)
```

This concatenated matrix represents the combined "routing topology" that alignment training reshaped — how the model decides which tokens attend to which other tokens.

TASM performs truncated SVD on this concatenated delta:

```
ΔW_QK ≈ U · diag(σ₁..σ_k) · V_k^T
```

The number of components k is configurable via `sfd_svd_k` (default 16). The layer range defaults to layers 9–15 but can be configured via `sfd_layer_start`/`sfd_layer_end`, or set to match the ASM signal layers via the `sfd_use_signal_layers` flag. All three parameters are accessible in the Advanced Parameters panel.

The right singular vectors `V_k` (shape: `k × hidden_dim`) define the k most important directions in the input space that alignment training modified for routing purposes. The singular values `σ₁..σ_k` quantify how much each direction was modified.

TASM also computes per-layer global measures from the singular value spectrum: effective rank (same exp-entropy formula), spectral entropy, log-volume (sum of log singular values), stable rank (`‖ΔW_QK‖²_F / σ₁²`), and Frobenius norm.

### 4b: Per-Token Measurement (Forward Pass)

For each token at each monitored layer, the activation vector `h` is projected into the cached subspace:

```
c = V_k · h          (k-dimensional coefficient vector)
w = σ ⊙ c            (elementwise weighting by singular values)
```

The coefficient vector `c` tells you how much of the activation lies along each alignment-modified direction. Weighting by `σ` emphasizes the directions that alignment training changed most.

From this weighted projection, three measures are computed:

**Energy:** The squared norm of the weighted projection — total activation energy in the alignment subspace.

```
energy = ‖w‖² = Σᵢ (σᵢ · cᵢ)²
```

**Spectral entropy:** How evenly the energy is distributed across the k directions.

```
qᵢ = wᵢ² / energy    (normalized energy distribution)
H_t = −Σᵢ qᵢ log qᵢ
```

**Effective rank of the token:**

```
erank_t = exp(H_t)
```

**Density ratio:** The token's effective rank divided by the layer's global effective rank. Values > 1 mean the token engages more dimensions of the alignment subspace than average; values < 1 mean it engages fewer.

```
density = erank_t / erank_global
```

### 4c: Prompt-Level Aggregation

Per-token energy, entropy, and density measures are averaged across monitored layers internally. However, only the density ratio is persisted in the result object. The prompt-level output includes four density statistics: mean, max, variance, and 90th percentile.

---

## Stage 5: Behavioral Comparison

### 5a: KL Divergence

When the base model is loaded, TASM computes the KL divergence from the instruct model's output distribution to the base model's at each token position:

```
KL(instruct ‖ base)[i] = Σ_v  p_instruct(v|i) × (log p_instruct(v|i) − log p_base(v|i))
```

where the sum is over the entire vocabulary. This is computed exactly (no sampling) from the full logit vectors. The headline number is the KL at the final position (the next-token prediction), but per-token KL is stored for the full sequence.

### 5b: Top-k Predictions

TASM captures the top-k (default 10) most probable next tokens and their softmax probabilities from both the instruct and base models. These are human-readable diagnostic outputs: they show what each model would actually generate.

---

## Stage 6: Rank Displacement

Rank displacement compares how the instruct and base models order the same set of counterfactual alternatives at each token position. It decomposes the displacement into three pools:

**Matched:** Tokens that appear in *both* models' top-k. Displacement = absolute difference in probabilities between the two models.

**Promoted:** Tokens that appear only in the instruct model's top-k. These are alternatives that instruction-tuning elevated into consideration. Displacement = the instruct probability (full value, since the base model didn't rank them highly enough to appear).

**Demoted:** Tokens that appear only in the base model's top-k. These are alternatives that instruction-tuning suppressed. Displacement = the base probability.

At each position, total displacement is the sum of all three pools:

```
total_disp = matched_disp + promoted_mass + demoted_mass
replacement_ratio = (promoted_mass + demoted_mass) / total_disp
```

A high replacement ratio means the instruct and base models consider largely *different* alternatives — alignment training didn't just re-rank candidates, it replaced them entirely.

**Legacy metrics:** For backward compatibility, TASM also computes Kendall's τ (rank correlation) and Jaccard overlap between the shared tokens' rankings. Kendall's τ is computed at every position where at least 2 tokens appear in both models' top-k (configurable via `rd_min_shared`, default 2). Positions with fewer than 2 shared candidates receive τ = 0.0. The `per_position_tau` array is always the same length as the `tokens` array — every position gets a value.

---

## Stage 7: Candidate Graph Topology

This is a post-hoc analysis of the counterfactual candidate sets. For each prompt, TASM builds a graph of all unique candidate tokens across all positions, tracking which positions each candidate appears at and whether it was promoted, demoted, or matched at each.

Key metrics:

- **Contested fraction:** How many positions have *both* promotions and demotions (the instruct model is actively reshuffling alternatives, not just ignoring them).
- **Dual-role candidates:** How many unique tokens appear as promoted at some positions and demoted at others.
- **Role switches:** How many times a candidate's status changes between adjacent positions (promoted → demoted or vice versa).
- **Switch rate:** Role switches per position.

---

## Stage 8: Batch Statistical Aggregation

When multiple prompts have been analyzed, TASM computes aggregate statistics across the full set.

### 8a: Per-Category Summary

Results are grouped by category (benign, harmful, jailbreak, etc.). For each category and each metric in the registry (15 metrics spanning ASM, KL, LTP, SFD, and rank displacement), TASM computes a **bootstrap confidence interval** (configurable: `n_bootstrap` resamples, default 5,000; `ci_level`, default 0.95):

1. Draw a sample with replacement of the same size as the group.
2. Compute the mean of the sample.
3. Repeat for n_bootstrap iterations.
4. The estimate is the mean of the original data; the CI bounds are the α/2 and (1−α/2) percentiles of the bootstrap distribution.

The seed is fixed at 42 for reproducibility.

### 8b: Pairwise Separability

For each pair of (benign-ish category, target category), TASM computes **Cohen's d** — the standardized mean difference — for every metric:

```
d = |mean_A − mean_B| / s_pooled

where s_pooled = √((s_A² + s_B²) / 2)
```

Cohen's d is also bootstrapped (5,000 resamples, same procedure but resampling both groups independently in each iteration) to get confidence intervals on the effect size itself.

### 8c: Optimal Classification Threshold

For each metric and each benign-vs-target pair, TASM does a brute-force threshold sweep (configurable via `threshold_steps`, default 500 evenly-spaced candidate thresholds between the min and max observed values). At each threshold, it evaluates accuracy in both directions ("harmful ≥ threshold" and "harmful < threshold") and reports the threshold, accuracy, and direction that maximizes classification accuracy.

### 8d: Correlation Diagnostics

**Stress vs. KL correlation:** Pearson r between stress_score and kl_divergence across all prompts that have both. This tests whether the structural signal (how much the delta responds) correlates with the behavioral signal (how different the output distributions are).

**Length correlations:** For every metric, Pearson r between the metric value and sequence length. This is a diagnostic for measurement artifacts — if a metric correlates with length, it may be measuring tokenization effects rather than alignment signal.

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| `h` | Hidden state (residual stream activation) at a given layer |
| `ΔW_V` | Value projection weight delta (instruct − base) |
| `ΔW_Q`, `ΔW_K` | Query and key projection weight deltas |
| `α` | Attention weight matrix (post-softmax) |
| `W_O` | Output projection weight (from the instruct model) |
| `W_u` | Unembedding matrix (lm_head weight) |
| `τ` | Forward direction (normalized consecutive hidden-state difference) |
| `d_ic` | Unembedding probing direction (alternative token − chosen token) |
| `V_k` | Top-k right singular vectors of the concatenated QK delta |
| `σ` | Singular values |
| `seq_len` | Number of tokens in the input |
| `n_layers` | Total transformer layers in the model |
| `n_kv_heads` | Number of key/value heads (GQA) |
| `head_dim` | Dimension per attention head |
| `hidden_dim` | Full hidden state dimension (`n_heads × head_dim`) |

---

# Part 3: Visualization Pipeline

## Overview

TASM's visualization system operates across three rendering layers: server-side Matplotlib plots (rendered to base64 PNGs), client-side JavaScript-rendered interactive components, and a client-side Three.js WebGL 3D terrain viewer. All visuals share a unified dark theme based on Material Design's #121212 surface with an Okabe-Ito colorblind-safe palette, following Tufte-inspired data-ink ratio principles.

This section covers every visualization in the system, organized by scope (single prompt vs. batch) and signal family (ASM, LTP, SFD), with attention to the design language, the data each visualization encodes, and the rendering mechanics.

---

## Design System

### Color Palette

TASM uses the Okabe-Ito palette, recommended by Nature Methods for categorical scientific data. It is designed to be distinguishable by people with all common forms of color vision deficiency:

| Category | Color | Hex | Role |
|----------|-------|-----|------|
| Benign | Blue | #0072B2 | Safe/baseline prompts |
| Mild | Amber | #E69F00 | Low-risk prompts |
| Harmful | Vermillion | #D55E00 | Dangerous prompts |
| Jailbreak | Reddish purple | #CC79A7 | Adversarial attacks |
| Adversarial | Reddish purple | #CC79A7 | Same hue as jailbreak |

Marker shapes provide redundant encoding for accessibility: benign = circle, mild = square, harmful = diamond, jailbreak = triangle-up.

Profile shape classifications use their own color subset: steep = amber (#E69F00), flat = sky blue (#56B4E9), inverted = vermillion (#D55E00).

### Dark Theme

The background hierarchy follows Material Design elevation:

| Surface | Hex | Usage |
|---------|-----|-------|
| #121212 | Figure and page background (not pure black — prevents OLED bleed) |
| #1E1E1E | Axes/panel background (one elevation step up) |
| #252525 | Card/legend background |

Text uses three opacity tiers of white: primary (#DEE2E6, 87%), secondary (#9CA3AF, 60%), and muted (#6B7280, 38%). Grid lines are #333333 at 30% opacity. Axis spines are #404040. This hierarchy keeps data prominent while providing structural context without visual competition.

### Continuous Colormap

The TASM sequential colormap progresses through: dark background → deep blue → teal → light blue → yellow-green → amber → vermillion. This is a custom 8-stop map designed to be perceptually uniform in the dark theme context while avoiding problematic green-red transitions.

### Matplotlib RC Overrides

All plots apply a shared set of 22 RC parameter overrides at import time. Key choices: top and right spines are removed (Tufte: maximize data-ink ratio), grid lines are thin (0.5pt) and barely visible (30% alpha), all text is minimum 12-13pt, figure DPI is 200 for sharp rendering on retina displays, and the font stack is Arial → Helvetica → DejaVu Sans.

### Rendering Pipeline

All Matplotlib plots follow the same lifecycle: create figure → render content → call `fig_to_base64()` → close figure. The function writes the figure to a BytesIO buffer as PNG with tight bounding box, encodes to base64, and closes the figure to free memory. The base64 string is stored in the session (for individual prompt plots) or served directly to the frontend (for dashboard plots). On export, base64 strings are decoded back to PNG files for inclusion in the ZIP archive.

---

## Single-Prompt Visualizations

These are generated per prompt analysis, organized into four categories in the frontend: ASM Core (3 plots), ASM Detail (2 JS-rendered components), LTP (3 plots), SFD (2 plots). Each has a scope of "prompt" and renders server-side unless marked as type "js." Additional LTP JS components (counterfactual table) and LTP plots (dual trajectory, profile heatmap, summary stats) are also generated but may be displayed separately from the main visualization groups.

### ASM Core

**Signed Attribution Bar Chart** (`signed_attribution`)

A vertical bar chart with one bar per input token. Positive bars are green (#009E73), negative bars are vermillion (#D55E00). The zero line is drawn as a thin gray horizontal rule. Token labels are rotated 50° along the x-axis.

An annotation box in the upper-right corner reports the net correction (sum of all attributions) and the count of negative tokens. The figure width scales linearly with token count (0.65 inches per token, minimum 9 inches) to prevent label crowding on long prompts.

This is the primary diagnostic view for a single prompt: it shows at a glance which tokens drive the alignment correction and whether any tokens push against it.

**Per-Token Stress Bar Chart** (`stress_per_token`)

A uniform-color (sky blue, #56B4E9) vertical bar chart showing the focused stress score at each token position. Same adaptive width as the attribution chart. An annotation shows the mean stress score.

Where signed attribution shows *direction*, this shows *magnitude* — how strongly the alignment delta responds to each token's representation, regardless of sign. The two views are complementary: a token with high stress but low signed attribution is engaging the alignment subspace without contributing net correction at the output position.

**Token × Layer Heatmap** (`heatmap`)

A 2D image plot with tokens on the x-axis and sublayer depth on the y-axis, using the TASM sequential colormap. The y-axis is labeled with four landmarks: Early, Mid (⅓), Late (⅔), and Final. A colorbar on the right shows the normalized sensitivity scale.

This is the most information-dense single-prompt visualization. Boundary-concentrated correction (bright columns at edges, dark in the middle) is the expected benign signature. Interior-distributed correction (bright columns at unexpected positions, or horizontal bright bands at specific depths) may indicate adversarial patterns. The heatmap also reveals which network depths contribute most to the correction — middle-layer attention peaks are the canonical safety-relevant signal.

### ASM Detail (JS-Rendered)

**Attribution Table** (`token_table`, type: js)

An HTML table rendered client-side from the raw per-token data. Each row shows: token text, signed attribution value (colored green or red), a proportional-width bar showing relative attribution magnitude, stress score, and a proportional-width bar showing relative stress magnitude.

Bars are normalized within-prompt: the maximum absolute attribution sets the scale for attribution bars, and the maximum stress sets the scale for stress bars. This allows visual comparison of tokens within a single prompt but not across prompts.

The component is wrapped in a collapsible "feature" container with a header describing what the table shows and a legend explaining the encoding.

**Model Predictions** (`model_predictions`, type: js)

Rendered client-side when response capture is enabled. Shows the instruct model's and base model's top-k next-token predictions side by side, with token text and probability for each. This is a direct behavioral readout — what each model would actually generate next.

### LTP Visualizations

**Lateral Tension Profiles — Stacked Bar** (`ltp_profiles`)

A stacked bar chart where each bar is a token position and the segments within each bar represent the k counterfactual alternatives (default k=8), ordered by the instruct model's probability ranking. Segment heights encode lateral tension magnitude. Colors use the viridis colormap sampled at k evenly-spaced points from 0.2 to 0.95.

Above each bar, non-flat profile shapes are annotated with small markers: upward triangles (^) for steep profiles (amber), downward triangles (v) for inverted profiles (vermillion). Flat profiles get no marker. Only the first 4 ranks are labeled in the legend to avoid clutter.

This is the primary LTP diagnostic: you can see at a glance which tokens have strong lateral tension, how that tension is distributed across alternatives (steep = concentrated on one alternative, flat = spread evenly), and where anomalous inverted profiles occur.

**Lateral Tension Magnitude Bar Chart** (`ltp_tension_magnitudes`)

A bar chart of the net tension point magnitude at each token position. Unlike the stacked profiles (which show per-alternative structure), this shows the resultant vector magnitude — how much net lateral pull exists after probability-weighted averaging of all counterfactual directions.

Bars are colored by profile shape classification: amber for steep, sky blue for flat, vermillion for inverted. A dashed horizontal line shows the mean magnitude. An annotation reports the numeric mean. A three-patch legend identifies the shape colors.

**LTP Summary Statistics** (`ltp_summary_stats`)

Two horizontal bar panels showing the prompt's M and V statistics. Very small values (< 0.001) switch to scientific notation in the label. C (consistency) is excluded because it is nearly perfectly correlated with M (r=0.989). L (coverage) is excluded because it is consistently 1.0 — every token has non-zero lateral tension, so the metric conveys no information.

**LTP Profile Heatmap** (`ltp_profile_heatmap`)

A 2D image plot with tokens on the y-axis and counterfactual rank on the x-axis. Uses the TASM sequential colormap. Columns are labeled R1 through Rk. This is the LTP analog of the ASM heatmap — a dense view that reveals patterns invisible in the bar charts.

Look for: vertical bright columns (one alternative consistently dominates across all tokens), horizontal bright rows (specific tokens with unusually high tension), and diagonal patterns (rank-position correlations that suggest systematic lateral structure).

### SFD Visualizations

**Per-Token QK Density** (`sfd_density`)

Bar chart (amber, #E69F00) showing the density ratio at each token — how many dimensions of the QK routing subspace the token's activation engages relative to the global effective rank. Values above 1.0 mean the token engages more dimensions than average. An annotation shows mean, max, and global erank.

**Rank Displacement** (`rank_displacement`)

Bar chart of per-position Kendall's τ between the base and instruct models' counterfactual alternative rankings. Bars are color-coded by agreement level: green (τ > 0.5, models agree), amber (0 < τ ≤ 0.5, weak agreement), vermillion (τ ≤ 0, models disagree or rank inversely). Horizontal reference lines at 0 (no correlation) and 1 (perfect agreement). Y-axis is fixed at [-1.1, 1.1].

An annotation reports mean τ, mean overlap percentage, and the count of comparable positions out of total positions. Kendall's τ is computed at positions where at least 2 candidates appear in both models' top-k (configurable via `rd_min_shared`). Positions with fewer shared candidates receive τ = 0.0. The `per_position_tau` array is always position-aligned with the `tokens` array.

---

## Batch / Dashboard Visualizations

These are generated when a plot is first requested after batch analysis. The dashboard endpoint returns aggregate statistics and a list of available plot keys instantly — no matplotlib generation occurs during Refresh Dashboard. Individual plots are generated lazily on first request via `GET /api/plots/{key}`, cached to disk as PNG files, and served from cache on subsequent requests. A generation lock prevents duplicate work on concurrent requests. The frontend shows "Loading {key}.png…" while a plot generates.

Plots are organized into "proven" (Analysis tab) and "experimental" categories. They operate on the full collection of prompt results.

### Proven Visualizations

**Effect Sizes — Forest Plot** (`separability`)

A horizontal forest plot showing Cohen's d for each proven metric (net correction, interior share, boundary share, entropy, stress score, interior CV), sorted by effect size magnitude descending.

Each metric gets: a point estimate dot (sized for emphasis), a horizontal CI whisker line, and a right-aligned label showing `d = X.XX  YY%`. Dots and whiskers are colored by effect size tier: green for large (d ≥ 0.8), amber for medium (d ≥ 0.5), vermillion for small (d < 0.5).

Vertical reference lines are drawn at d = 0.5 (medium, dashed) and d = 0.8 (large, dotted). The zero line is also drawn for reference. Y-axis is inverted so the strongest effects appear at the top. The left spine is removed for a cleaner look.

**Category Distributions — Strip Plot** (`batch_summary`)

Four side-by-side panels (net correction, interior share, stress score, entropy) showing per-category distributions as jittered strip plots with mean markers and CI bars. This replaces traditional box plots following Allen et al. (2021) guidance that strip plots better convey individual data points and avoid the misleading quartile summaries box plots provide at small n.

For each category in each panel: small transparent dots are scattered at jittered x-positions (the jitter is seeded deterministically per category-metric pair). Strip points are synthetically generated from the bootstrap CI parameters (normal distribution centered at the estimate with spread = CI width / 3.5) when the original raw values aren't available in the aggregate. A large opaque circle with a white border marks the mean estimate. A thick vertical line through the mean shows the 95% bootstrap CI.

**Separability Scatters** (`key_scatters`)

A two-panel scatter plot: entropy vs. net correction (left) and stress score vs. net correction (right). Points are colored by category and shaped by category marker (redundant encoding). Each category with n ≥ 3 gets a 95% confidence ellipse overlay.

The ellipses are computed from the 2×2 covariance matrix of the two plotted variables. The eigendecomposition gives the principal axes; the semi-axis lengths are scaled by √(χ²₂(0.95) × λ) = √(5.991 × eigenvalue). This is the standard bivariate-normal 95% containment ellipse. For non-normal data it's an approximation, but it still provides a useful visual summary of each group's location and spread.

**Proof 1 Verification** (`proof1_summary`)

Two-panel diagnostic. Left panel: histogram of log₁₀(error) across all proof-1 exactness checks from all prompts and heads. Right panel: two bars showing the exact (error < `proof1_threshold`, default 10⁻⁴) and inexact fractions, with percentage labels.

This is a computational integrity check, not a statistical inference — it validates that the mathematical decomposition (signed attribution sums to the correction norm) is numerically exact to machine precision.

### Experimental Visualizations

**Trajectory Overlay** (`exp_trajectory_overlay`)

All prompts' normalized amplitude trajectories superimposed on a single plot, each colored by category. Vertical dashed lines mark the early→mid and mid→late transitions. Category legend included.

This is a raw exploratory view — no aggregation or smoothing. If categories cluster into different trajectory shapes (e.g., adversarial prompts peaking earlier or higher), it's visible immediately. But with many prompts, visual overplotting can obscure patterns.

**Difference from Benign Baseline** (`exp_difference_from_benign`)

For each non-benign category, the mean trajectory is computed and the benign mean trajectory is subtracted. The resulting difference curves are plotted with fill-between shading. This isolates where each category deviates from baseline at each network depth.

**Discriminative Sublayers** (`exp_discriminative_sublayers`)

A horizontal bar chart ranking the top-15 sublayers by their adversarial-minus-benign amplitude difference. Each bar is labeled with its layer number, sublayer type (Attn or MLP), and network region (Early/Middle/Late). Colors encode region: green for early, reddish purple for middle, sky blue for late.

**Full Scatter Grid** (`exp_metric_scatters`)

A 2×3 grid of scatter plots showing five pairwise metric combinations (stress vs. KL, entropy vs. net correction, boundary vs. interior share, stress vs. net correction, entropy vs. stress). The sixth cell is empty. Points colored by category. This is the exhaustive view — it includes weak and redundant metric pairs that the proven scatter panel omits.

**Behavioral Comparison** (`exp_behavioral_comparison`)

A grouped bar chart with one group per prompt showing the instruct model's and base model's top-1 next-token probabilities side by side (sky blue for instruct, vermillion for base). Prompt labels are truncated and word-wrapped. The gap between paired bars shows behavioral divergence.

**LTP Category Comparison** (`exp_ltp_category_comparison`)

Three side-by-side box plots (offset magnitude, offset consistency, offset variance) with one box per category. Standard Tukey box plots with IQR boxes, median lines, 1.5×IQR whiskers, and outlier points. Boxes are colored by category and drawn at 75% opacity.

**LTP M vs. Stress** (`exp_ltp_m_vs_stress`)

Scatter plot of LTP offset magnitude (y) against ASM stress score (x), points colored by category. Tests whether LTP captures information orthogonal to ASM — prompts that separate on M but not on stress validate the framework's claim that directional structure carries signal beyond amplitude.

**Profile Shape Distribution** (`exp_ltp_profile_shapes`)

A grouped bar chart showing the fraction of tokens classified as steep, flat, or inverted, for each category. Each category gets three bars (one per shape). Colors match the shape palette.

**SFD Category Comparison** (`exp_sfd_category_comparison`)

Three box plots (QK density, spectral entropy, QK energy) by category. Same Tukey box plot format as the LTP comparison.

**SFD vs. ASM** (`exp_sfd_vs_asm`)

Scatter of SFD QK density against ASM interior share, colored by category. Tests whether the SFD dimensionality axis provides orthogonal information to the ASM distributional metrics.

**Rank Displacement by Category** (`exp_rank_displacement`)

Two box plots (Kendall τ, token overlap) by category. Shows whether the degree of base-vs-instruct counterfactual reshuffling varies systematically by prompt type.

---

## The Terrain Viewer (WebGL)

The most complex visualization is a client-side Three.js 3D terrain that renders per-prompt probability displacement between the base and instruct models' counterfactual candidate sets. It is rendered entirely in JavaScript using WebGL and operates on data fetched from the `/api/results/detail` endpoint.

### Geometry

The terrain is a 3D surface with dimensions tokens × 17 columns. The central column (column 8) is the "spine" — it represents the chosen token at each position. Columns 1–7 (left of spine) show the base model's counterfactual candidates; columns 9–16 (right of spine) show the instruct model's candidates. This gives a "dual-bank" layout where the left bank is the base model's view and the right bank is the instruct model's view.

**Vertex height** at each grid point encodes the displacement magnitude for that candidate at that token position. For candidates that appear in both models' top-k, the height is the absolute probability difference. For promoted candidates (instruct-only), the height is the full instruct probability. For demoted candidates (base-only), the height is the full base probability. The spine height at each position is a weighted average of surrounding displacement magnitudes.

Height is scaled by a constant factor (HSCALE = 180) and a global min/max normalization across all magnitudes in the current prompt.

### Color Encoding

Vertex colors use a dual-ramp scheme with three regions:

- **Base bank (columns 0–7):** Cool tones (blues/teals) that increase in brightness with displacement magnitude.
- **Spine (column 8):** A distinct color ramp that marks the generation path.
- **Instruct bank (columns 9–16):** Warm tones (ambers/vermillions) that increase in brightness with displacement magnitude.

Colors are further modulated by two factors: KL divergence at each token position (adds a brightness boost proportional to behavioral divergence), and edge fade (columns far from the spine are blended toward the background color using a power-1.6 falloff based on distance from the spine). This keeps visual attention on the structurally important central columns while providing context at the periphery.

### Rendering Modes

The viewer supports two render modes toggled by the user:

- **Surface mode:** A triangulated mesh with Phong shading (shininess 30, emissive gray for ambient fill). Grid lines are overlaid as line segments with brightness that decreases with distance from the spine. The spine column gets bright wireframe lines; peripheral columns get dim wireframe lines.

- **Points mode:** Each grid point is rendered as a size-attenuated particle. Point size decreases with distance from the spine (power-0.8 falloff). Non-visible tokens (below-median displacement) get tiny points (size 0.01 vs. 0.06).

### Labels

Every grid vertex gets a text label rendered as a Canvas-textured sprite. Labels show the candidate token text. Spine labels (column 8) are 1.5× larger, colored in sky blue with a dark stroke for contrast. Non-spine labels are colored in off-white with dark stroke. Each label has a colored underline bar whose length is proportional to the KL divergence at that position, using the appropriate color ramp for its bank.

A median-magnitude filter can dim or hide tokens with below-median total displacement to reduce visual clutter in large prompts.

### Interactivity

The terrain supports: orbit camera controls (drag to rotate, scroll to zoom, right-drag to pan), category/prompt selection via dropdown menus, a slideshow mode that auto-advances through prompts at configurable speed, a bank toggle (dual/instruct-only/base-only), render mode toggle (surface/points), a data filter (all tokens/above-median only), label size scaling, and auto-rotate with configurable RPM. The camera defaults to an elevated perspective looking down at the terrain at an angle. Auto-rotation pauses when the user drags the camera and resumes on release.

### Launch Menu

Before rendering, a configuration modal allows the user to set: category filter (restrict to one category or load all), record limit (cap the number of prompts loaded), token limit (truncate per-prompt arrays before terrain transform), prompt character limit (dropdown label length), auto-rotate toggle, and rotation speed. These defaults are persisted in the Display section of the Configuration tab.

The fetch loop terminates early when the record limit is reached rather than downloading all data and truncating client-side. When a category filter is active, records are filtered during fetch so the loop can stop as soon as enough matching prompts are collected.

### Data Flow

The terrain viewer receives pre-computed displacement profiles from the rank displacement computation. Each prompt's data consists of: token list, and per-token rows containing [base_displacement_profile (8 values), instruct_displacement_profile (8 values), kl_at_this_position]. These are fetched from the server via the `/api/results/detail` endpoint, which returns the `rank_displacement.instruct_disp_profiles`, `rank_displacement.base_disp_profiles`, and `per_token_kl` arrays.

---

## Data Table (JS-Rendered)

The dashboard includes a paginated sortable data table rendered client-side. It displays 24 columns (including a checkbox selector) spanning all signal families:

**Identity:** Index, prompt text (truncated), category, role, token count.

**ASM:** Stress score, net correction, entropy, interior share, interior CV, boundary share, KL divergence, negative token count. Stress, net correction, entropy, interior share, density, and Kendall τ columns include heatmap-style background tinting that colors cells proportional to their value.

**Behavioral:** Instruct top-1 token and probability, base top-1 token and probability.

**LTP:** Max PRC, directional token count, mean M.

**SFD:** Density.

**Rank displacement:** Kendall τ, overlap.

The table supports pagination (25 rows per page), column-width customization via inline style, and checkbox selection of individual prompts for batch operations (rerun, remove).

---

## PDF Reports

TASM generates two PDF report types using ReportLab, both following a consistent professional template.

### Template Structure

Both reports share: a cover page with the TASM branding (title in blue, horizontal rule, metadata table with analyst name, organization, model, and timestamp), a page header (accent-colored rule, report title on the left, timestamp on the right), and a page footer (organization/analyst on the left, page number on the right).

The body uses a professional style system with named styles for titles (Helvetica Bold 22pt), H1 (Helvetica Bold 14pt), H2 (Helvetica Bold 11pt, accent blue), body text (Helvetica 10pt, 14pt leading), and monospace (Courier 10pt for prompt text and code).

### Single Prompt Report

Contains: cover page, apparatus description (explaining what TASM measures), the prompt text, a metrics table (all scalar metrics including LTP when available), a per-token attribution table (token, signed attribution, stress), and all applicable plots (signed attribution, stress, distribution metrics, amplitude trajectory, heatmap, and all LTP plots when enabled). Each plot is preceded by a title and a plain-language description of what the plot shows and what to look for.

### Batch Report

Contains: cover page, apparatus description, a category summary table (n, average length, stress, entropy, boundary/interior share, net correction, negative token rate, M — all with bootstrap point estimates), a separability analysis table (Cohen's d, 95% CI, best threshold accuracy for each metric), all comparative/dashboard plots with descriptions, and a per-prompt results table (prompt text truncated to 60 characters, category, token count, and key scalars for every prompt in the batch).

Tables use a consistent style: light gray header row, thin gray grid borders, 10pt Helvetica, 5pt cell padding, word-wrap enabled for prompt text columns.

---

## Visualization Registry

The frontend maintains a central registry of all 29 visualizations with metadata:

| Key | Category | Type | Scope | Needs |
|-----|----------|------|-------|-------|
| signed_attribution | ASM Core | plot | prompt | — |
| stress_per_token | ASM Core | plot | prompt | — |
| heatmap | ASM Core | plot | prompt | — |
| amplitude_trajectory | ASM Detail | plot | batch | — |
| distribution_metrics | ASM Detail | plot | prompt | — |
| token_table | ASM Detail | js | prompt | — |
| model_predictions | ASM Detail | js | prompt | — |
| ltp_tension_magnitudes | LTP | plot | prompt | ltp |
| ltp_profiles | LTP | plot | prompt | ltp |
| ltp_profile_heatmap | LTP | plot | prompt | ltp |
| ltp_summary_stats | LTP | plot | prompt | ltp |
| counterfactual_table | LTP | js | prompt | ltp |
| sfd_density | SFD | plot | prompt | sfd |
| rank_displacement | SFD | plot | prompt | sfd |
| separability | Batch Analysis | plot | batch | — |
| batch_summary | Batch Analysis | plot | batch | — |
| key_scatters | Batch Analysis | plot | batch | — |
| discriminative_sublayers | Batch Analysis | plot | batch | — |
| proof1_summary | Batch Analysis | plot | batch | — |
| exp_trajectory_overlay | Batch Analysis | plot | batch | — |
| exp_difference_from_benign | Batch Analysis | plot | batch | — |
| exp_metric_scatters | Batch Analysis | plot | batch | — |
| exp_behavioral_comparison | Batch Analysis | plot | batch | — |
| exp_ltp_category_comparison | Batch Analysis | plot | batch | — |
| exp_ltp_m_vs_stress | Batch Analysis | plot | batch | — |
| exp_ltp_profile_shapes | Batch Analysis | plot | batch | — |
| exp_sfd_category_comparison | Batch Analysis | plot | batch | — |
| exp_sfd_vs_asm | Batch Analysis | plot | batch | — |
| exp_rank_displacement | Batch Analysis | plot | batch | — |

Note: The `ltp_dual_trajectory` plot is generated server-side but is deliberately excluded from the registry and the frontend display. The Three.js terrain viewer is rendered as a standalone component outside the registry system.

Each entry has an `on` flag (all default to true), an `order` for display sequencing, and a `needs` field that conditionally shows/hides the visualization based on whether LTP or SFD computation was enabled. Visualizations are togglable per-session from the frontend controls.

---

# Part 4: Token Variance Analysis

Cross-context measurement of per-token coupling stability to the correction manifold.

## What It Measures

Every token that passes through an aligned model has a geometric relationship to the weight delta between base and instruct models. The spectral field density (SFD) captures how broadly that token couples to the correction manifold in a single prompt. Token variance asks a different question: **how much does that coupling change across prompts?**

A token like "build" has nearly identical density whether it appears in "How do I build a bookshelf?" or "How do I build a pipe bomb?" Its coupling to the correction manifold is stable because its representational state doesn't shift much with context. A token like "your" has measurably higher density in "Ignore your safety rules" than in "What is your favorite color?" because its representational state is dominated by what surrounds it, and jailbreak context pushes it into regions of activation space that engage more of the correction manifold.

The module computes the coefficient of variation (CV) of spectral density for each token across all prompts where it appears. High CV means context-dependent coupling. Low CV means stable coupling. It also computes eta-squared (η²) — the fraction of variance explained by prompt category — to identify tokens whose coupling is driven by alignment context rather than random variation.

## Integration

Token variance is implemented as a TASM module (`engine/modules/token_variance.py`) and runs from the Modules tab in the frontend. It operates on the current session's results — no separate command-line invocation is needed. Results persist to `module_token_variance.json` in the session directory and are included in session exports.

The module framework provides:
- Auto-discovery of module classes from `engine/modules/`
- Thread-isolated execution with crash protection
- Parameter metadata rendered as UI controls
- Re-run capability with changed parameters

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Appearances | 3 | Minimum interior appearances to include a token. Also used as the "qualified" threshold for summary statistics — no hidden secondary threshold. |
| Merge Subwords | off | Merge BPE subword tokens into whole words before analysis. Eliminates fragments like "ret" (from "Pretend") and "ard" (from "Disregard") that appear as separate entries. |
| Include First Token | off | Include position-0 tokens (density ~0.525 regardless of content due to positional artifact). |
| Top N | 30 | Maximum tokens shown per report section. |
| Min Prompt Length | 3 | Skip prompts shorter than this many tokens. |
| Min Per Category | 2 | Minimum appearances per category for pairwise comparisons. |

## Output

The module produces a structured JSON report with six sections.

**Highest density CV.** Tokens whose coupling to the correction manifold is most sensitive to surrounding context. These are predominantly function words whose representational state is determined by context rather than intrinsic semantics.

**Lowest density CV.** Tokens with stable coupling regardless of context. Content words with concrete semantics that maintain a consistent relationship to the correction manifold. Note: tokens appearing in only one category will show zero CV by definition — stability by isolation, not intrinsic property. Merge Subwords eliminates BPE fragments from this list.

**High eta-squared.** Tokens where the largest fraction of density variance is explained by prompt category (requires appearance in ≥2 categories). η² = 1.0 means 100% of variance is between-category; η² near 0 means category membership doesn't predict density.

**Cross-category profiles.** For tokens appearing in ≥3 categories, shows mean density and stress broken down by category.

**Pairwise category comparisons.** For each pair (benign/jailbreak, benign/adversarial, benign/harmful, harmful/jailbreak), shows the density shift for tokens appearing in both categories with sufficient sample size (controlled by Min Per Category). Positive diff means higher density in the second category.

**Summary statistics.** Distribution statistics (mean, percentiles) of density CV across qualified tokens, counts of context-stable versus context-dependent tokens (using P25/P75 thresholds), and overall token/prompt counts.

## Channels

The module computes variance statistics across three SFD channels simultaneously:

- **Density** (primary): spectral density ratio — the token's effective rank relative to the global effective rank
- **Stress**: per-token stress score from ASM
- **Energy**: spectral energy — the squared norm of the weighted projection into the QK subspace

CV and eta-squared are computed independently for each channel. The density channel is used for sorting, classification (stable vs. dependent), and pairwise comparisons.

## Data Requirements

Reads from the current TASM session. Each prompt entry must contain:

- `tokens`: list of token strings
- `per_token_stress`: list of floats
- `sfd.per_token_density`: list of floats
- `sfd.per_token_energy`: list of floats
- `category`: string (optional, used for cross-category breakdowns and eta-squared)

Prompts missing SFD data are skipped with a count reported in the summary.

## Interpretation Notes

Cross-context variance is not noise. It measures how the correction field responds differently to the same token depending on what surrounds it. This is a direct consequence of the attention-weighted projection: the delta acts on the residual stream state, which is context-dependent. Tokens whose representational state varies with context will show variable coupling to the correction manifold.

The systematic elevation of function word density in jailbreak contexts reflects the fact that jailbreak framing pushes the entire residual stream into regions of activation space that engage more of the correction manifold. The function words absorb this context shift because their representations are dominated by surrounding tokens rather than intrinsic semantics.

Content words with stable density are candidates for baseline calibration. Their consistent coupling provides a reference point against which context-dependent shifts can be measured.

---

# Part 5: Engine Configuration

## Overview

All measurement-affecting parameters are centralized in `engine/engine_config.py` and surfaced in the frontend's Configuration → Advanced Parameters panel. The defaults reproduce the original hardcoded behavior exactly. Changing any parameter requires a full application reset (model unload, session clear, cache invalidation) to ensure measurement comparability.

## Parameter Registry

| Parameter | Default | Module | What it controls |
|-----------|---------|--------|-----------------|
| `signal_layer_fraction` | 0.333 | model_manager | Middle third for ASM signal layers |
| `sfd_use_signal_layers` | false | sfd | Whether SFD uses ASM's layer range |
| `sfd_layer_start` | 9 | sfd | SFD layer range start (when not using signal layers) |
| `sfd_layer_end` | 16 | sfd | SFD layer range end, exclusive |
| `sfd_svd_k` | 16 | sfd | SVD components for per-token spectral projection |
| `sfd_svd_seed` | 42 | sfd | Random seed for SVD (torch.svd_lowrank uses randomized algorithm) |
| `serialization_precision` | 8 | global | Decimal places for JSON transport of measurement values |
| `rd_min_shared` | 2 | sfd | Min shared candidates for Kendall tau |
| `boundary_fraction` | 0.1 | analyzer | Interior/boundary token split |
| `response_topk` | 10 | analyzer | Next-token predictions captured per model |
| `proof1_threshold` | 1e-4 | analyzer | Attribution decomposition exactness threshold |
| `delta_svd_k` | 64 | model_manager | Weight delta spectral summary rank |
| `n_bootstrap` | 5000 | statistics | Bootstrap resamples for CIs |
| `ci_level` | 0.95 | statistics | Confidence interval width |
| `threshold_steps` | 500 | statistics | Classification threshold search granularity |
| `min_valid_separability` | 5 | statistics | Min values for separability analysis |
| `min_samples_d` | 2 | statistics | Min samples per group for Cohen's d |
| `ltp_overfetch_first` | 1 | ltp | Extra candidates on first fetch pass |
| `ltp_overfetch_second` | 5 | ltp | Wider fetch if first pass insufficient |
| `disc_sublayers_top_n` | 15 | comparative | Sublayers in discriminative plot |
| `domain_embedding_layer_frac` | 0.50 | modules | Hidden-state capture depth for domain embeddings (0.0–1.0) |
| `domain_escalation_layer_frac` | 0.75 | modules | Separate depth for escalation-level probe matching |
| `include_first_token` | false | modules | Include position-0 token in per-token domain embeddings |
| `export_domain_embeddings` | false | modules | Include per-token domain embeddings in session JSON export |
| `probe_projection_space` | false | modules | Project embeddings through o_proj delta before probe matching |
| `attention_weighted_pool` | false | modules | Attention-weighted pooling instead of uniform mean-pool |
| `persist_probe_caches` | true | modules | Probe caches and active probe selection survive server restarts |
| `chat_temperature` | 0.7 | chat | Sampling temperature for chat generation |
| `chat_top_p` | 0.9 | chat | Top-p (nucleus) sampling for chat generation |
| `chat_max_tokens` | 512 | chat | Maximum tokens per chat response |

## Persistence

Engine configuration is persisted to `engine_config.json` alongside the application. On startup, saved values are loaded automatically. The Reset to Defaults button deletes the persisted file so the next startup uses defaults.

## UI Safety

The Advanced Parameters panel is locked by default. The user must check an acknowledgment checkbox to unlock it. Edits accumulate locally without being sent to the server. The Apply button shows the count of pending changes and lists each old → new value in a confirmation dialog. Applying triggers a full application reset: model unloaded, session cleared, all caches invalidated. This ensures data collected under different settings is never mixed in the same session.

---

# Part 6: Module System

## Framework

TASM includes an extensible post-collection analysis framework. Modules operate on session data (the list of result dictionaries accumulated from prompt analysis) and produce structured JSON output without affecting the core instrument pipeline. They cannot trigger model inference and they cannot modify session results.

The framework is defined in `engine/modules/base.py` and consists of three components:

**ModuleParameter** is a dataclass that declares a single user-configurable parameter with a name, display name, description, type (`int`, `float`, `bool`, `select`, `text`, `textarea`, `file`), default value, and optional constraints (min/max for numeric types, options list for selects). The frontend renders UI controls directly from this metadata — no per-module frontend code is required.

**TASMModule** is the base class that all modules inherit from. Each subclass declares its identity (name, display name, description, version), its data requirements (minimum session results, whether it needs SFD/LTP/RD data), and its parameter list. It implements two methods: `validate()`, which checks whether the module can run given the current session state and parameters, and `run()`, which executes the analysis and returns a JSON-serializable dictionary.

**ModuleRunner** handles discovery, lifecycle, and execution. At startup it scans `engine/modules/` for any Python file containing a TASMModule subclass, instantiates each, and registers it. Module execution happens in isolated daemon threads — if a module crashes, it does not affect the main application or other modules. The runner tracks per-module state (idle, running, completed, error), persists results to `module_{name}.json` in the session directory, and supports re-running with changed parameters and resetting to idle.

Modules are auto-discovered. To add a new module, place a Python file in `engine/modules/` that defines a class inheriting from TASMModule and implement `run()`. No registration code or frontend changes are needed.

There are currently ten modules spanning four functional categories: probe pipeline (Probe Generator, Correction Heatmap, Correction Manifold, Domain Surface), cross-context analysis (Token Variance), batch evaluation (Comparative Analysis), visualization support (Correction Field Topology), and MI evaluation (MI Readiness Analysis, MI Instrumentation).

---

## Probe Generator

| | |
|---|---|
| **Module name** | `probe_generator` |
| **Version** | 0.3.0 |
| **Min results** | 0 (does not read session data) |
| **Data requirements** | Loaded model (uses the active instruct or base model for inference) |

The Probe Generator creates discriminative vocabulary sets by sampling the loaded model's own output distribution. It is the entry point to the probe pipeline — its output CSV is consumed by the Correction Heatmap, Correction Manifold, and Domain Surface modules.

### Process

1. Reads a template CSV defining a class × subclass lattice (any CSV with a `subject` column and one or more subclass columns; everything beyond `subject` and `anchor_id` is treated as a subclass axis).
2. For each cell in the lattice, constructs a prompt steered toward that cell using a configurable template with `{class}`, `{subclass}`, `{seeds}`, and `{word_count}` placeholders.
3. Queries the loaded model N times per cell (default 50), generating responses with temperature=0.9, top-p=0.95, and repetition penalty=1.1.
4. Tokenizes each response: extracts alphabetic words of 3+ characters that encode to a single token in the model's vocabulary. Words appearing in the optional stopword file are excluded.
5. Applies a frequency filter: tokens appearing fewer than the minimum threshold (default 3) times across all queries for that cell are discarded.
6. Cross-class deduplication: for each subclass column, tokens that appear in more than one class are removed.
7. Cross-subclass deduplication: for each class, tokens that appear in more than one subclass column are removed.
8. Exports a probe CSV with discriminative vocabulary per cell, preserving the template's column structure. Optionally exports an inference catalog CSV logging every model query and response.

The two-axis deduplication is the discriminative filter. A token must be unique to its cell along both axes. What survives is a vocabulary fingerprint specific to each cell in the lattice.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| Template File | file | (none) | Template CSV defining the class × subclass lattice |
| Output File Name | text | `auto_probes.csv` | Filename for the generated probe CSV |
| Queries Per Cell | int | 50 | Model queries per class × subclass cell (10–200) |
| Max Tokens Per Response | int | 256 | Maximum tokens generated per query (64–512) |
| Minimum Frequency | int | 3 | Minimum token appearances to survive filtering (1–20) |
| Prompt Template | textarea | (default template) | Prompt sent to the model, with `{class}`, `{subclass}`, `{seeds}`, `{word_count}` placeholders |
| Export Inference Catalog | bool | off | Save a CSV log of every query and response |
| Stopword File | file | `templates/stopwords.txt` | Words to exclude before discriminative filtering |

### Output

Returns a dictionary containing: template and output filenames, subject and level lists, per-class statistics (raw token count, discriminative token count, top 20 words), cross-class and cross-subclass shared token counts, total discriminative tokens, an optional per-cell inference catalog, and a stopword audit trail (which stopwords were loaded, how many times each was filtered).

---

## Correction Heatmap

| | |
|---|---|
| **Module name** | `correction_heatmap` |
| **Version** | 0.1.0 |
| **Min results** | 1 |
| **Data requirements** | Active probe set with cached embeddings at both depths; `per_token_final_emb` in session results |

The Correction Heatmap measures how strongly each prompt's tokens interact with the correction field across the probe lattice. It produces an aggregate heatmap (mean interaction intensity per subject × subclass cell), per-prompt heatmaps, per-cell variance, and per-cell token specificity rankings.

### Process

1. Loads probe embeddings from cache at two depths (default L50 and L75, overridden by template `_meta` rows if present).
2. Computes probe deltas: L2-normalize each depth's embedding, then subtract (L75_normalized − L50_normalized), re-normalize. This captures the direction of inter-layer refinement for each probe.
3. For each prompt, takes per-token final-layer hidden states (L2-normalized) and computes the dot product of each token against each probe delta.
4. Aggregates projections per cell (subject × subclass) using the selected projection method.
5. Computes per-cell variance across prompts, and ranks tokens within each cell by cell-specificity z-score — tokens that activate a particular cell differently from their global average score highest.

### Projection methods

- **abs** (default): absolute value of the dot product — linear magnitude of interaction.
- **squared**: squared dot product — energy measure, suppresses weak interactions and amplifies strong ones.
- **signed**: raw dot product — preserves sign, cells can go negative, revealing systematic anti-alignment.

### Output

Returns: aggregate heatmap (session mean), per-cell variance grid, per-subject summary statistics (mean/max activation, mean variance), per-cell token detail (top 10 most cell-specific tokens with z-scores and per-category breakdowns), probe file metadata, and category breakdown.

---

## Correction Manifold

| | |
|---|---|
| **Module name** | `correction_manifold` |
| **Version** | 0.3.0 |
| **Min results** | 10 |
| **Data requirements** | Active probe set with cached embeddings; `per_token_final_emb` in session results |

The Correction Manifold projects each prompt through the same probe delta lattice as the Heatmap to produce a per-prompt fingerprint vector (one energy value per cell), then discovers natural clusters and reduces to 2D for visualization. The manifold and heatmap are two views of the same projection: the heatmap is the tabular view, the manifold is the spatial view.

### Process

1. Computes per-prompt fingerprints using the same probe delta projection as the Correction Heatmap (energy per cell).
2. PCA reduces fingerprint vectors to 2D for visualization.
3. K-means clustering on fingerprint vectors. When k=0 (default), auto-selects k by testing k=2 through 8 and choosing the k with the highest silhouette score. K-means uses k-means++ initialization with 10 restarts.
4. Compares discovered clusters against human category labels. Reports cluster-to-category accuracy, binary (safe/risk) accuracy, per-cluster category distribution, and category centroids in the 2D space.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| Clusters (k) | int | 0 | Number of K-means clusters. 0 = auto-select via silhouette score (tries k=2..8) |

### Output

Returns: per-prompt cluster assignments, 2D PCA coordinates, silhouette score, cluster-to-category accuracy, binary accuracy, per-cluster category distributions, category centroids, PCA variance explained, and data formatted for the interactive popout visualization.

---

## Domain Surface

| | |
|---|---|
| **Module name** | `domain_surface` |
| **Version** | 0.2.0 |
| **Min results** | 10 |
| **Data requirements** | Active probe set with cached embeddings; `domain_embedding` in session results; SFD data; Rank Displacement data |

The Domain Surface maps per-token correction signals onto a subject-matter domain surface. It embeds probes and session prompts into a shared PCA space, assigns each token to its nearest probe subject and escalation level, and merges per-token ASM/SFD/RD metrics to reveal how alignment training treats the same token across different topics and discourse frames.

### Process

1. Loads pre-computed prompt-level domain embeddings (mean hidden states at the configured domain layer) and cached probe embeddings at the domain depth.
2. When a separate escalation layer is configured (default L75 vs L50), loads escalation-layer probe embeddings for split-depth ring assignment: the subject (angular) axis uses the domain layer, the escalation level (radial) axis uses the escalation layer.
3. Co-fits PCA across prompt and probe embeddings jointly to produce a shared 2D coordinate space.
4. Builds per-token observations: for each token across all prompts, collects the prompt's 2D coordinate, the token's per-token stress, SFD density, and rank displacement tau, the nearest probe subject (by cosine similarity), and the nearest escalation level.
5. If Token Variance results are available, uses eta-squared (category-dependence) to weight content words over function words when selecting the top N tokens for display.
6. Computes stratification statistics: token counts per subject and per escalation level.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| Top Tokens | int | 30 | Number of most-frequent tokens to include (5–100) |
| Min Appearances | int | 2 | Minimum token appearances across prompts (1–20) |

### Output

Returns: 2D PCA coordinates for all prompts and all probes, per-token observation records (prompt index, subject assignment, level assignment, stress, density, tau), ordered token list, token CV statistics, stratification summaries by subject and level, probe anchor points with coordinates, and PCA variance explained.

---

## Token Variance

Token Variance is documented separately in Part 4 of this document.

---

## Comparative Analysis

| | |
|---|---|
| **Module name** | `comparative_analysis` |
| **Version** | 1.0.0 |
| **Min results** | 2 |
| **Data requirements** | None beyond basic session results |

The Comparative Analysis module computes cross-prompt aggregate statistics and category separability. It is the primary batch-level analysis module and serves as the data backend for the dashboard and batch visualizations.

### Process

1. Reconstitutes scalar-mode PromptResult objects from session result dictionaries.
2. Calls `aggregate_batch()` from `engine/statistics.py`, which computes: per-category bootstrapped metric estimates with 95% CIs (5,000 resamples), Cohen's d effect sizes for every metric between each benign-ish/target category pair (with bootstrapped CIs), optimal classification thresholds (brute-force sweep), length correlation diagnostics, and stress-vs-KL correlation.
3. Caches the aggregate statistics to `aggregate_statistics.json` in the session directory. On subsequent runs with the same number of prompts, returns the cached version unless Force Recompute is enabled.
4. Determines which batch visualization plots are available (15 plot keys spanning proven and experimental categories).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| Force Recompute | bool | off | Recompute statistics even if cached results exist |

### Output

Returns: the full aggregate statistics object (per-category summaries, separability analysis, correlations, thresholds), available plot keys, prompt count, category list, and per-category detail.

---

## Correction Field Topology

| | |
|---|---|
| **Module name** | `correction_field_topology` |
| **Version** | 1.0.0 |
| **Min results** | 1 |
| **Data requirements** | Either rank displacement displacement profiles or LTP profiles in session results |

The Correction Field Topology module validates session data for the 3D displacement field terrain viewer and computes aggregate topology statistics. The visualization itself renders client-side via Three.js; this module provides the data validation, summary statistics, and launch parameter surface for the frontend.

### Process

1. Iterates session results to count prompts with displacement profiles (from rank displacement), LTP-fallback profiles, and base model candidate data. Prompts lacking both are skipped.
2. Computes aggregate topology statistics: per-category displacement magnitude distributions, asymmetry between instruct and base banks, token count statistics (total, mean, max across prompts).
3. Packages launch parameters (category filter, record limit, token limit, prompt label length, auto-rotate toggle and speed) into the result for the frontend to read.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| Category Filter | select | all | Restrict to one category or show all |
| Record Limit | int | 100 | Maximum prompts loaded into the visualization (1–2000) |
| Token Limit | int | 20 | Maximum tokens rendered per prompt terrain (4–100) |
| Prompt Label Length | int | 50 | Character limit for dropdown labels (20–200) |
| Auto-Rotate | bool | off | Spin terrain on launch |
| Rotate Speed (RPM) | float | 0.3 | Auto-rotation speed (0.1–2.0) |

### Output

Returns: prompt counts (total, with displacement, with LTP fallback, with base candidates, skipped), per-category breakdowns, token statistics, displacement and asymmetry aggregate statistics, and launch parameters.

---

## MI Readiness Analysis

| | |
|---|---|
| **Module name** | `mechanistic_interpretability` |
| **Version** | 1.0.0 |
| **Min results** | 10 |
| **Data requirements** | Session results with labeled categories (at least 3 safe and 3 risk) |

The MI Readiness Analysis module evaluates session data against mechanistic interpretability community evaluation standards. It addresses specific gaps identified in independent review of the ASM framework, providing the quantitative evidence needed for MI venue submissions.

### Process

1. Extracts a feature matrix from session results: up to 15 scalar metrics (ASM, LTP, SFD, RD) per prompt, plus binary labels (safe/risk based on configurable safe category list) and sequence lengths.
2. Computes cross-validated AUROC using a custom logistic regression classifier (gradient descent with L2 regularization, configurable learning rate, iterations, and regularization strength) across k folds (default 5).
3. Computes AUROC using sequence length alone, then computes length-residualized AUROC (regressing out length from all features via ordinary least squares).
4. Computes per-metric univariate AUROC to identify which individual metrics carry discriminative signal.
5. Runs PCA on the feature space to determine effective dimensionality (number of components needed for the target cumulative variance, default 95%). Names PCA axes by their top-loading features.
6. Detects metric redundancy: computes the full Pearson correlation matrix and flags pairs with |r| above the threshold (default 0.80).
7. Runs a random projection baseline: generates random unit vectors of matching dimensionality, projects all prompts, computes AUROC, and repeats for N trials (default 10). The gap between the real AUROC and the random baseline mean validates that the weight-delta projection outperforms arbitrary directions.
8. Analyzes per-category stress distributions (means and standard deviations).
9. Generates a readiness scorecard with five criteria: Discrimination (AUROC), Length Confound (residualized AUROC), Weight-Delta Specificity (delta above random), Metric Efficiency (redundant pairs), and Sample Size (minimum class size). Each criterion is rated strong/adequate/weak with thresholds. An overall readiness rating (ready, near-ready, or gaps) summarizes the assessment.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| CV Folds | int | 5 | Cross-validation folds for AUROC (3–10) |
| Random Projection Trials | int | 10 | Random baselines to average (5–50) |
| Redundancy |r| Threshold | float | 0.80 | Pearson |r| above which metric pairs are flagged (0.5–0.99) |
| Random Seed | int | 42 | Seed for reproducibility (0–99999) |
| PCA Variance Target | float | 0.95 | Cumulative variance fraction for effective dimensionality (0.80–0.99) |
| Logistic Regression LR | float | 0.1 | Learning rate for internal classifier (0.001–1.0) |
| Logistic Regression Iterations | int | 500 | Gradient descent iterations |
| L2 Regularization | float | 0.01 | L2 regularization strength (0.0–1.0) |
| Safe Categories | select | benign,mild | Categories treated as safe (class 0) |

### Output

Returns: AUROC results (full, length-only, residualized), per-metric AUROC table, PCA analysis (components, effective dimensionality, named axes), metric redundancy pairs, random projection baseline (mean/std/max AUROC, delta above random), category stress analysis, MI readiness scorecard with per-criterion ratings, and overall readiness assessment.

---

## MI Instrumentation

| | |
|---|---|
| **Module name** | `mi_instrumentation` |
| **Version** | 1.0.0 |
| **Min results** | 10 |
| **Data requirements** | Session results with `per_token_final_emb` (full capture), `per_layer_signed_attr`, labeled categories |

The MI Instrumentation module produces the actual mechanistic interpretability measurements a researcher would cite. Where MI Readiness evaluates data quality, this module generates MI outputs: a refusal direction, an activation patching priority map, per-layer AUROC curves, and a random projection control.

### Process

**1. Refusal Direction Extraction.** Computes the empirical refusal direction as the normalized difference between mean-pooled final-layer embeddings of risk and safe prompts: `refusal_dir = normalize(mean_risk − mean_safe)`. Projects all prompts onto this direction via dot product and computes AUROC. Compares against stress score AUROC. Reports per-category mean cosine similarity with the refusal direction, separation (risk mean − safe mean), and standard deviations.

**2. Per-Layer AUROC Curve.** For each signal layer, computes the mean absolute signed attribution across all tokens for each prompt, then computes AUROC of that per-layer scalar against the safe/risk labels. Reports AUROC, per-class means, and risk/safe ratio for each layer. Identifies the layer with the highest discrimination power.

**3. Activation Patching Priority Map.** Builds a (layer × position) matrix of correction intensity by combining per-layer signed attribution with per-token stress (elementwise product). Averages across all prompts. Identifies the top 20 intervention points (layer-position pairs with highest combined intensity) and computes layer-marginal statistics (mean and max intensity per layer). Exposes the raw matrix for visualization.

**4. Random Projection Control.** Generates N random unit vectors of the same dimensionality as the embedding space (default 50 trials), projects all prompt embeddings onto each, and computes AUROC for each. Reports the distribution of random AUROCs (mean, std, max, 95th percentile). Computes empirical p-values for both the refusal direction AUROC and the stress score AUROC — the fraction of random trials that match or exceed each. This validates that the measured signals are not artifacts of high-dimensional geometry.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| Safe Categories | str | benign,mild | Comma-separated categories treated as safe (class 0) |
| Random Projection Trials | int | 50 | Number of random directions for the projection control |
| Random Seed | int | 42 | Seed for reproducible random projections |
| Max Tokens (Patching Map) | int | 30 | Maximum token positions in the patching priority map |

### Output

Returns: refusal direction statistics (AUROC, stress AUROC, separation, per-category cosines, norms), per-layer AUROC curve (per-layer AUROC, means, ratios), patching priority map (top intervention points, layer marginals, raw matrix), random projection control (random AUROC distribution, deltas, empirical p-values), and a summary comparing all measured AUROCs against the random baseline.
