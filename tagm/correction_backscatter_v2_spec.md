# Correction Backscatter v2 — Specification

Status: design proposal · supersedes the math in `correction_backscatter.py` v0.3.x · keeps the module name and storage paths

## 1. What this document is

A precise statement of (a) what the existing Correction Backscatter module computes, (b) what we want it to compute under the contextual-field framing, and (c) where the existing implementation has to change. Written before code so the math is reviewable on its own.

The user's framing, paraphrased: *"The contextual snapshot of a prompt is irreducibly relational — defined by the arrangement of tokens, not just the tokens themselves. I want to project that snapshot through the QK and OV correction fields and see which probes feel it."*

## 2. Notation

For one transformer layer ℓ in a Qwen-2-style architecture:
- `d` is the model hidden size (1536 for Qwen 2.5 1.5B).
- `d_kv` is the K/V projection out-dim under GQA (256 for Qwen 2.5 1.5B; smaller than `d` because there are fewer KV heads than Q heads).
- `T` is the prompt length in tokens.
- `X^ℓ ∈ ℝ^{T×d}` is the residual stream entering layer ℓ's attention block (after pre-attn norm, since that's where the projections actually consume their input).
- `W_Q^ℓ ∈ ℝ^{d×d}`, `W_K^ℓ ∈ ℝ^{d×d_kv}`, `W_V^ℓ ∈ ℝ^{d×d_kv}`, `W_O^ℓ ∈ ℝ^{d_kv×d}` are the four projection weight matrices. (For non-GQA models `d_kv = d` and nothing here breaks.)
- Superscript `I` denotes the instruct (fine-tuned) model; `B` the base model. So `W_Q^{ℓ,I}` and `W_Q^{ℓ,B}` are the instruct and base query weights at layer ℓ.
- `ΔW_Q^ℓ := W_Q^{ℓ,I} − W_Q^{ℓ,B}` and analogously for K, V, O. These are stored in `pipeline.delta_store`.
- `p_k ∈ ℝ^d` is one probe's embedding (k indexes the probe lattice). `P ∈ ℝ^{N×d}` is the stacked probe matrix loaded from the active-set cache.
- `s_k ∈ {1..n_subj}, t_k ∈ {1..n_levels}` are the probe's subject and subclass tags from the CSV.

Subscript on an attention matrix means a specific term in a sum, never a softmax index.

## 3. What the existing module computes

The current `_compute_energies_for_suffix` does, for each layer ℓ and each suffix `s ∈ {q,k,v,o}`:

```
proj_p[k]      = ‖ p_k · ΔW_s^{ℓ,T} ‖_2 / ‖ΔW_s^ℓ‖_F           probe energy per layer
proj_t[i]      = ‖ x_i · ΔW_s^{ℓ,T} ‖_2 / ‖ΔW_s^ℓ‖_F           token energy per layer
```

then averages those over `signal_layers` to give a per-suffix `probe_energies` (shape `[N]`) and one `token_energies` per prompt (shape `[T_prompt]`).

The "qk" and "qkv" composites are arithmetic means of the per-suffix scalars:

```
qk  = (q + k) / 2
qkv = (q + k + v) / 3
```

The cell value at `(subject s, level t)` for a given prompt is then formed by:

```
cell[s,t] = aggregate_over_tokens( token_energies · cell_probe_energy[s,t] )
```

where `cell_probe_energy[s,t]` is the mean of `probe_energies[k]` over probes in cell `(s,t)`, and `aggregate_over_tokens` is mean / max / sum (user choice).

### Three things to notice about that

**Token order is invisible.** The token-energies sum is a sum of scalar norms, one per token, computed independently. Reordering the tokens in the prompt would not change `token_energies`. The "snapshot" being projected is a bag of tokens, not a context.

**Q+K is not QK.** The "qk" composite is the average of the Q-only and K-only energies. The actual QK circuit is the bilinear form `W_Q W_K^T` whose effective behaviour is governed by interactions between Q and K, not their separate norms. A delta concentrated on Q-only and a delta concentrated on K-only could produce identical "qk" composites but completely different routing changes.

**Probe ↔ token coupling is multiplicative-of-norms.** The cell value multiplies a probe-norm by a token-norm. This is rotation-blind: a probe and a token vector that point in opposite directions in the delta's image produce the same cell value as ones that align. There's no signed coupling.

These aren't bugs in the v0.3 sense — the docstring is honest that it measures "intensity (norm), not direction." But for the *contextual response* question the user is now asking, all three are blockers.

## 4. What v2 should compute

### 4.1 The thing being projected: prompt attention deltas

For one prompt and layer ℓ, define the four contextual-field tensors:

```
δQ^ℓ := X^ℓ · ΔW_Q^ℓ        ∈ ℝ^{T × d}        Q-shift of every token
δK^ℓ := X^ℓ · ΔW_K^ℓ        ∈ ℝ^{T × d_kv}     K-shift
δV^ℓ := X^ℓ · ΔW_V^ℓ        ∈ ℝ^{T × d_kv}     V-shift
δO^ℓ := (X^ℓ · W_V^{ℓ,B}) · ΔW_O^ℓ
        + (X^ℓ · ΔW_V^ℓ)   · W_O^{ℓ,I}        O-shift, see §4.2
```

These are the per-prompt, per-layer images of the residual stream under the four delta operators. They depend on token *arrangement* through `X^ℓ` — which is what the layers above ℓ produced from the full prompt, not from individual tokens.

The full QK and OV first-order changes:

```
ΔA_QK^ℓ := (X^ℓ · ΔW_Q^ℓ) · (X^ℓ · W_K^{ℓ,I})^T
         + (X^ℓ · W_Q^{ℓ,B}) · (X^ℓ · ΔW_K^ℓ)^T
                                   ∈ ℝ^{T×T}     pre-softmax score change
ΔY_OV^ℓ := softmax(A_B^ℓ / √d_kv) · δO^ℓ
                                   ∈ ℝ^{T×d}     residual-stream content change
```

`A_B^ℓ` is the base-model attention score matrix `(X^ℓ W_Q^{ℓ,B})(X^ℓ W_K^{ℓ,B})^T`. We use the base softmax weights so that `ΔY_OV^ℓ` is "what would have changed about the residual stream if only V/O had been corrected, at the routing pattern the base model would have used." This gives a clean OV-isolated quantity. (See §4.2 for why the alternative — using the corrected softmax — mixes QK and OV signals.)

### 4.2 First-order vs full-delta of products

For the QK circuit the full quantity is `Δ(W_Q W_K^T) = W_Q^I W_K^{I,T} − W_Q^B W_K^{B,T}`. By the product rule:

```
Δ(W_Q W_K^T) = ΔW_Q · W_K^{I,T} + W_Q^B · ΔW_K^T
             = ΔW_Q · W_K^{B,T} + W_Q^I · ΔW_K^T
             = (linear term) + ΔW_Q · ΔW_K^T
                                  └────── second-order, much smaller
```

The two linear forms are equivalent up to which model "anchors" each side. We use the asymmetric form `ΔW_Q · W_K^{I,T} + W_Q^B · ΔW_K^T` (instruct on the K side, base on the Q side, then the dual term). This is symmetric in what it weights and means "change in attention score for a token treated as a query against an unchanged-on-this-side key, plus change in attention score as a key against an unchanged-on-the-other-side query." The second-order term `ΔW_Q · ΔW_K^T` is dropped for v2; we can re-add it as a diagnostic if it turns out to be non-negligible.

### 4.3 Probe response

A probe `p_k` is a single vector. Its response to a prompt's correction field decomposes per filter:

**Q-filter** — how strongly is this probe's direction excited by the Q-shift, weighted by the prompt's K geometry:
```
r_QK[k] := p_k^T · (X^ℓ ΔW_Q^ℓ)^T · (X^ℓ W_K^{ℓ,I})  ·  ?
```
This needs to collapse to a scalar. Two natural choices:

(a) **Field-magnitude response**: project the probe through the QK operator on both sides, then take Frobenius inner product with the prompt's `ΔA_QK^ℓ`:
```
r_QK^ℓ[k] := < p_k^T · ΔW_Q^ℓ · W_K^{ℓ,I,T} · p_k , 1 >  (a scalar — bilinear self-form)
              + symmetric K term
```
This is "how much would the probe's attention to itself change under correction."

(b) **Field-pattern response**: signed inner product between the probe pushed through the Q-side operator and the prompt's K-shifted geometry, summed over positions:
```
r_QK^ℓ[k] := Σ_{i,j} ΔA_QK^ℓ[i,j] · ⟨ p_k W_Q^{ℓ,I,T} , X_i^ℓ W_Q^{ℓ,I,T} ⟩
                                  · ⟨ p_k W_K^{ℓ,I,T} , X_j^ℓ W_K^{ℓ,I,T} ⟩
```
This says "how much does the probe live in the (query × key) directions where this prompt's attention was rewired."

We use **(b)** for the headline cell value. (a) is a degenerate case of (b) (it's what you get when `X^ℓ = p_k 1_T`, i.e. the prompt is "this probe repeated T times"); (b) is the actual context-aware question.

For OV the analog is simpler because OV produces a vector per token, not a `T×T` matrix:
```
r_OV^ℓ[k] := Σ_i ⟨ p_k , ΔY_OV^ℓ[i] ⟩
```
"How much of the probe's direction is written into the residual stream by the prompt's OV-corrected content, summed across positions."

### 4.4 Layer aggregation

`r_QK^ℓ[k]` and `r_OV^ℓ[k]` are scalars per layer. Aggregation across `signal_layers` is a sum (not a mean), normalized by `‖ΔW_filter^ℓ‖_F` per layer to keep unit-equivariance:

```
r_QK[k] := Σ_ℓ  r_QK^ℓ[k] / N_QK^ℓ          where N_QK^ℓ := ‖ΔW_Q^ℓ‖_F · ‖W_K^{ℓ,I}‖_F + ‖W_Q^{ℓ,B}‖_F · ‖ΔW_K^ℓ‖_F
r_OV[k] := Σ_ℓ  r_OV^ℓ[k] / N_OV^ℓ          where N_OV^ℓ := ‖softmax(A_B^ℓ)‖_F · (‖W_V^{ℓ,B}‖_F · ‖ΔW_O^ℓ‖_F + ‖ΔW_V^ℓ‖_F · ‖W_O^{ℓ,I}‖_F)
```

Sum-not-mean because the contextual response *should* compound across layers — a probe that gets pulled strongly at layer 4 and again at layer 14 has been excited twice.

### 4.5 Cell value

Cell `(s, t)` for filter `f ∈ {Q, K, V, O, QK, OV}` and one prompt is:

```
cell[s,t,f] := aggregate_{k : (s_k, t_k) = (s,t)}  r_f[k]
```

with the aggregation rule (mean / max / sum) inherited from the existing parameter. Across multiple prompts in a batch we average `cell[s,t,f]` over prompts, exactly as v0.3 does.

The Q, K, V, O filters use the simpler per-suffix versions:
```
r_Q^ℓ[k] := Σ_i  ⟨ p_k · ΔW_Q^ℓ , X_i^ℓ · ΔW_Q^ℓ ⟩    (single-side response)
```
plus analogous expressions for K, V, O. These are kept for diagnostic continuity with v0.3 — they say "is the correction concentrated on this side of the attention path?" — but the **headline filters are QK and OV**, because those are the circuit-level questions the framing asks.

### 4.6 Sign convention and interpretation

`r_QK[k]` and `r_OV[k]` are signed. Positive means the probe lies in a direction the correction is *pushing toward* in this prompt; negative means *pushing away from*. A probe that's strongly aligned with the field but in the opposite direction is just as informative as one aligned with it — both indicate the field "knows about" the probe's concept, even if they push in opposite directions. The headline visualization should therefore show **|r|** for intensity and a separate sign panel for direction.

This is the bit that the v0.3 norm-based formulation cannot say.

## 5. Where the data comes from

| Quantity | v0.3 source | v2 source |
|----------|-------------|-----------|
| Probe embeddings `P` | `probe_cache/<...>.json` via active-set | unchanged |
| Per-prompt token vectors `X^ℓ` | `session_results[i]["per_token_final_emb"]` (final layer only) | **new**: hook on each `signal_layer`'s pre-attn-norm output, captured at module-run time by re-running the prompts through the instruct model |
| Instruct weights `W_Q^I` etc. | not used | **new**: read from `pipeline.instruct_model.model.layers[ℓ].self_attn.{q,k,v,o}_proj.weight` |
| Base weights `W_Q^B` etc. | not used | **new**: derived as `W_*^I − ΔW_*` (the delta_store has the deltas; the base weights aren't kept post-load, so reconstruction is a subtraction) |
| Deltas `ΔW_Q` etc. | `pipeline.delta_store` | unchanged |

The only genuinely new infrastructure is per-layer hook capture of `X^ℓ`. Two ways to do it:

1. **Re-run on demand** at the start of `run()`, replaying every prompt through the loaded instruct model with hooks on each `signal_layer`. Latency ~ `O(n_prompts × n_layers)` forward passes, dominated by the prompts (each prompt is a single forward, the hooks cost almost nothing). For 5 prompts × 28 layers on Qwen 2.5 1.5B, this is single-digit seconds. **Recommended.**

2. **Capture during initial analyze** by extending the analyzer to persist `per_layer_pre_attn_resid: list[list[list[float]]]` of shape `[n_layers][T][d]` in `session_results`. Storage cost: 28 × T × 1536 × 4 bytes per prompt; for T=128 that's 22 MB per prompt. Across 100 prompts: 2.2 GB. Persisted to disk on every analyze. **Not recommended** unless a follow-on module also needs it.

## 6. Module shape

```python
class CorrectionBackscatterV2Module(TASMModule):
    name = "correction_backscatter"
    version = "2.0.0"

    parameters = [
        ModuleParameter("primary_filter", default="qk",
                         options=["q","k","v","o","qk","ov"]),
        ModuleParameter("aggregation",   default="mean",
                         options=["mean","max","sum"]),
        ModuleParameter("show_signed",   default=True),   # split |r| and sign(r) panels
    ]

    def run(self, session_results, params, progress=None):
        active = get_active_probe_set(...)               # unchanged from v1
        active.validate_against(self._pipeline)          # unchanged
        P = load probe matrix via active.cache_path(...)  # unchanged

        # 1. Re-run prompts under hooks → X[ℓ][prompt_idx] tensors.
        per_prompt_X = self._capture_layer_residuals(session_results)

        # 2. For each (layer, filter), form ΔA / ΔY tensors and probe responses.
        responses = {f: np.zeros((n_prompts, n_probes)) for f in FILTERS}
        for ℓ in signal_layers:
            for prompt_idx, X in per_prompt_X[ℓ].items():
                responses["q"][prompt_idx]  += self._r_q(X, ℓ, P)
                responses["k"][prompt_idx]  += self._r_k(X, ℓ, P)
                responses["v"][prompt_idx]  += self._r_v(X, ℓ, P)
                responses["o"][prompt_idx]  += self._r_o(X, ℓ, P)
                responses["qk"][prompt_idx] += self._r_qk(X, ℓ, P)
                responses["ov"][prompt_idx] += self._r_ov(X, ℓ, P)

        # 3. Build cell heatmaps (signed) and intensity heatmaps (|r|).
        for f in FILTERS:
            agg, intensity, sign_panel = self._heatmap(responses[f], probe_cells, n_subj, n_levels)
            ...

        # 4. Output structure mirrors v0.3 but with `signed_aggregate` + `sign_panel` next
        #    to `aggregate` for each filter.
```

The output JSON gains `signed_aggregate`, `sign_panel`, and `filter_norms` (the per-filter normalizers `N_QK^ℓ` etc.). The frontend gets one extra toggle: "Show sign" alongside the existing aggregation dropdown.

## 7. Side-by-side: v0.3 vs v2

| Question | v0.3 answer | v2 answer |
|----------|-------------|-----------|
| What's projected? | Static probe vectors and bag-of-token vectors, independently | Probe vectors and the prompt's *attention pattern* through Q/K/V/O |
| Token order matters? | No | Yes (order changes `X^ℓ` and `softmax(A_B^ℓ)`) |
| QK means what? | Mean of Q-only and K-only norms | First-order change in attention scores `ΔA_QK^ℓ` projected onto probe directions |
| OV is computed? | Not at all (v0.3 has only `qkv = (q+k+v)/3`) | Yes, as a separate filter `softmax(A_B) · δO` |
| Direction or magnitude? | Magnitude only (norms) | Both — signed inner products, with `\|r\|` and `sign(r)` reported separately |
| Layer aggregation | Mean of per-layer norms | Sum of per-layer signed inner products, each Frobenius-normalized |
| Per-prompt source vector | Final-layer residual (one per token) | Per-layer pre-attention residual, captured on demand |
| Cell value semantics | "intensity of correction × intensity of token" | "alignment of probe direction with the contextual field this prompt creates at this layer" |
| Cost | One pass over deltas | One forward pass per prompt (re-run with hooks) + same delta arithmetic |

## 8. What stays identical

- `ActiveProbeSet` resolver and the recent v2 cache binding.
- Probe lattice CSV format and the cell layout.
- Per-prompt / per-category accumulation structure in the output JSON.
- The interactive HTML viewer's overall shape (just gains a sign toggle).
- The `_PipelineState` plumbing for delta access.

## 9. What the user sees change

- The "QK" and "QKV" entries in the projection dropdown become "QK (routing field)" and "OV (object field)". `QKV` goes away — it never had a circuit-level interpretation.
- A new "Show sign" toggle.
- The headline numbers will be on a different scale than v0.3 (these are inner products, not norms), so saved sessions from v0.3 won't be numerically comparable. The module log records `version: "2.0.0"` so the difference is auditable.
- First run on a session takes a few seconds longer because of the per-prompt re-forward.

## 10. Open questions for review

1. **Symmetric or asymmetric anchoring for the first-order QK?** §4.2 picks asymmetric `(ΔW_Q, W_K^I) + (W_Q^B, ΔW_K)`. The fully-symmetric alternative averages all four pairings. The asymmetric one is cheaper and has a clean "instruct-as-anchor" interpretation; the symmetric one is more pedantic. Probably doesn't matter empirically but worth a yes/no.

2. **Base softmax or instruct softmax in `ΔY_OV`?** §4.1 uses `softmax(A_B^ℓ)` so the OV signal is isolated. The alternative is `(softmax(A_I) − softmax(A_B)) · (X W_V^I) W_O^I + softmax(A_B) · δO` — i.e. give the OV channel credit for content that gets routed differently because of QK changes too. That mixes the channels, which is exactly what we want to avoid.

3. **Should Q-only, K-only, V-only, O-only filters survive at all?** They're diagnostically useful but they aren't what the framing asks for. Keeping them as "advanced view" is cheap. Removing them simplifies the UI. Lean keep.

4. **Hook target.** Pre-attn-norm output is the right place because that's what the projections see. Confirm the adapter exposes this — `qwen2.py` should have a `pre_attn_norm` hook point already (used by the probe embedding flow). If yes, no adapter changes needed.
