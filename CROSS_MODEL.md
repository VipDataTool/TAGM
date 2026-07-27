# Cross-Model Comparison Notes

When running the same prompt set across Qwen 2.5 and Llama 3, you have
three nominal differences that interact with TAGM's measurement
substrate:

1. **Different tokenizers.** Qwen 2.5 uses a 151,936-token tokenizer;
   Llama 3.2 uses 128,256. The same prompt becomes different
   `tokens` lists with different `seq_len`. Per-token arrays
   (`signed_attr`, `per_token_stress`, `per_token_density`,
   `ltp.profiles`) live on incomparable indices.
2. **Different hidden dimensions.** Qwen 2.5 0.5B has hidden_size 896;
   Llama 3.2 1B has 2048. Frobenius norms of weight deltas scale
   roughly with the dimension, so raw `delta_scale` and amplitude
   numbers aren't directly comparable.
3. **Different layer counts.** Qwen 2.5 0.5B is 24 layers; Llama 3.2
   1B is 16. The "middle third" of the network is at different
   absolute depths. Trajectory plots overlaid raw will not align.

The framework has measurements at three tiers of comparability. Use
each at the right tier.

## Tier 1 — directly comparable across models

These are dimensionless, normalized, or geometric in a way that
factors out the model-specific scale.

| Metric | Why it survives the cross-model crossing |
|--------|------------------------------------------|
| `entropy` | Normalized to `log(seq_len)` — pure shape statistic on the attribution distribution. |
| `top2_share`, `middle_share` | Fractions of total absolute attribution. Dimensionless. |
| `interior_cv` | Coefficient of variation — dimensionless by construction. |
| `n_negative_tokens / seq_len` | Sign distribution as a fraction. **Divide manually** — the result dict and the stats registry report the raw count, which grows with prompt length. |
| `ltp.max_prc` | Concentration measure; dimensionless. |
| `n_directional / seq_len` | Fraction of directional tokens. **Divide manually** — `ltp_n_dir` in the registry/forest plot is the raw count. |
| `ltp.profile_shapes` distribution | Counts of steep/flat/inverted as fractions of total tokens. |
| `sfd.density_mean`, `density_max`, `density_p90` | Density is per-token erank divided by *that layer's* delta erank, averaged across layers — normalized within-model already (`global_erank` is reported alongside, not used as the divisor). |
| `rank_displacement.mean_tau` | Kendall τ ∈ [-1, 1]. Pure correlation. |
| `rank_displacement.mean_overlap` | Jaccard fraction. |
| `rank_displacement.replacement_ratio` | Mass fraction. (Valid as of the counterfactual-normalization fix: alternative probabilities are full-vocabulary softmax masses, identically normalized on the instruct and base sides.) |

For the headline cross-model comparison ("does Qwen show the same
benign-vs-jailbreak separability as Llama?"), use Cohen's d *within
each model* and then compare the d values across models. A Cohen's d
of 1.2 in Qwen and 1.4 in Llama means both models have a strong,
similar separation; the d-values themselves are dimensionless.

## Tier 2 — comparable after normalization

These need an explicit divisor before they're cross-model meaningful.

| Metric | Normalize by |
|--------|--------------|
| `stress_score` | `delta_scale` (already in result) — gives a unit-free "stress per unit alignment scale" |
| `ltp.mean_M`, `ltp.mean_V` | `delta_scale` (M) / `delta_scale²` (V). Only the trajectory direction τ is unit-normalized; the lateral vector itself is `W_O · (½ΔW_V · (W_u[alt] − W_u[chosen]))` minus its τ-component, so its magnitude scales with ‖ΔW_V‖, the unembedding row norms, and hidden size. NOT dimensionless — an earlier revision of this doc misfiled these in Tier 1. |
| `net_correction` | `delta_scale × seq_len` — accounts for both delta magnitude and prompt length |
| `kl_divergence` | Memory-heavy on long batches (the base phase caches a seq × vocab log-softmax per prompt) — prefer small subsamples; then look at *category-level differences*, not raw KL values |
| `amplitude_normalized` | Already Frobenius-normalized; still has different sublayer counts across models — see Tier 3 |
| Per-layer signed attribution | Express as fraction of the layer's amplitude before comparing layers across models |

`spectral_summary.mean_eff_rank` is technically dimensionless but
bounded by `delta_svd_k = 64`; if one model has many low-rank deltas
and another has many high-rank ones, the comparison is meaningful but
the cap matters. For a fair comparison, also report
`std_eff_rank` and the attention-vs-MLP split.

## Tier 3 — not directly comparable

| Metric | Why |
|--------|-----|
| Per-token arrays (`per_token_stress`, `signed_attr`, `per_token_density`, `per_token_kl`, `ltp.tension_magnitudes`, etc.) | Different tokenizations. Position 7 in Qwen ≠ position 7 in Llama. |
| `tokens` list itself | Same. |
| `amplitude_trajectory` raw values | Different hidden dimensions and different layer counts. |
| `heatmap` | Same. |
| `proof1_checks.error` magnitudes | Numerical-precision artifact, not a measurement. |
| `instruct_topk` / `base_topk` token strings | Different vocabularies. The *behavioral* divergence is comparable; the literal tokens aren't. |

For per-token arrays you want to compare, the path is to align by
*content*, not by index. Run a probe set (see below) to project both
models' per-token signals into a shared lattice space, then compare
cell-aggregated values across models. That's exactly what the
**Domain Surface** and **Correction Prism** modules are for.

## Probe sets and cross-model alignment

A probe cache is bound to `(probe_file, model_id, depths,
projected)`. So the same `cybersecurity_5x5.csv` produces three
distinct caches when applied to three different models — each cache
contains that probe lattice's terms embedded *through that model's
own residual stream*. The lattice geometry (what subjects, what
levels) is shared; the embedding coordinates are per-model.

The cross-model claim becomes: *do the same prompts land in the
same probe cells across models?* The answer comes from running
Domain Surface or Correction Prism on each model's session
independently and then comparing the cell-aggregated outputs.

Practical workflow for cross-model probe analysis:

1. Apply your probe set under model A. `probe_cache/` gets one entry
   tagged `__<safe_model_A>__L50.json` and `...L75.json`.
2. Run the 700-prompt batch under A. Run Domain Surface / Correction
   Prism. Save outputs.
3. Load model B. Apply the *same* CSV. New cache entries appear,
   tagged for model B. The active probe set's `model_id` field
   updates.
4. Run the batch under B. Run the modules. Save outputs.
5. Compare the *cell aggregates* (per-cell signed scalars or
   per-cell density signatures) across models. These are in a
   shared (subject, level) coordinate system — the lattice is
   shared even though embedding coordinates aren't.

If a finding is robust, the cell-level signature should be similar
across models even though the per-token arrays driving the
aggregates are not directly comparable.

## What "the finding replicates across models" should mean

For a benchmark intended to validate cross-model alignment-geometry
claims, I'd require at minimum:

1. **Direction of effect** is the same. If category A has higher
   stress than category B in Qwen, it should in Llama too.
2. **Effect-size ordering** is preserved. The rank-ordering of
   Cohen's d values across metrics should largely agree.
3. **Tier-1 metrics** show the same magnitudes (within bootstrap
   CIs) — these are the truly dimensionless ones.
4. **Tier-2 metrics** show the same *normalized* magnitudes once
   you divide out `delta_scale` etc.
5. **Probe cell signatures** are correlated across models when
   computed on the same probe lattice and the same prompts.

Disagreement at tier 1 or 5 is the strongest signal that something
is model-specific. Disagreement only at tier 2 might be a
normalization issue rather than a real difference.

## A note on the SmolLM2 + TinyLlama options

If Meta access is a blocker, SmolLM2 is the cleanest open
substitute — it's a clean instruct/base pair built explicitly on
Llama architecture, distributed by HuggingFace, no gating. The
cross-architecture claim becomes "Qwen 2 vs. Llama 3" if you have
Meta access, "Qwen 2 vs. Llama-architecture (SmolLM2)" if you don't.
The latter is still a real cross-architecture test; only the
specific pretraining recipe and parameter count differ.

TinyLlama-Chat is multi-stage tuned (SFT + DPO + others, not pure
RLHF), so its alignment-geometry signature will be noisier and
qualitatively different. Useful as a sanity check (does TAGM's
signal exist at all in non-RLHF tuning?) but not as a primary
cross-architecture reference.
