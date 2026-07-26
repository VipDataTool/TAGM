"""Analyzer: TASM-compatible computation engine using TAGM's pipeline.

Single forward pass through the instruct model. Extraction functions
read from the flat activation cache and the adapter-mediated delta store.
Everything writes to one flat PromptResult.

Usage:
    analyzer = Analyzer(pipeline)
    result = analyzer.analyze_prompt(
        "What is recursion?", category="benign",
        compute_ltp=True, compute_sfd=True)
    d = result_to_dict(result)
"""
from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from src.engine import config as engine_config
from src.engine.counterfactuals import top_alternatives, decode_alternatives
from src.engine.hooks import ActivationCapture
from src.engine.result import (
    PromptResult, LTPResult, SFDResult,
    ltp_result_to_dict, result_to_dict,
)

if TYPE_CHECKING:
    from src.core.pipeline import Pipeline

logger = logging.getLogger("src")


class Analyzer:
    """TASM-compatible analyzer backed by TAGM's pipeline.

    Holds a Pipeline reference for model/adapter/delta/tokenizer access.
    Caches SFD and SVD precomputes for the model's lifetime.
    """

    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline
        self._capture = ActivationCapture()
        self._svd_caches: dict = {}
        self._sfd_cache = None
        self._signal_layers: list[int] | None = None

    # ── Structural accessors ────────────────────────────────────────

    @property
    def adapter(self):
        return self.pipeline.adapter

    @property
    def delta_store(self):
        return self.pipeline.delta_store

    @property
    def tokenizer(self):
        return self.pipeline.tokenizer

    @property
    def instruct_model(self):
        return self.pipeline.instruct_model

    @property
    def base_model(self):
        return self.pipeline.base_model

    @property
    def device(self):
        return self.pipeline.device

    @property
    def n_layers(self) -> int:
        return self.adapter.n_layers(self.instruct_model)

    @property
    def n_heads(self) -> int:
        return self.adapter.attention_heads(self.instruct_model)[0]

    @property
    def n_kv_heads(self) -> int:
        return self.adapter.attention_heads(self.instruct_model)[1]

    @property
    def head_dim(self) -> int:
        return self.adapter.head_dim(self.instruct_model)

    @property
    def hidden_size(self) -> int:
        return self.adapter.hidden_size(self.instruct_model)

    @property
    def signal_layers(self) -> list[int]:
        """Middle third of model layers — same heuristic as TASM."""
        if self._signal_layers is None:
            frac = engine_config.get("signal_layer_fraction")
            n = self.n_layers
            mid_start = int(n * frac)
            mid_end = int(n * (1.0 - frac))
            self._signal_layers = list(range(mid_start, mid_end))
        return self._signal_layers

    def clear_caches(self):
        """Clear LTP/SFD caches (e.g. on model reload)."""
        self._svd_caches.clear()
        self._sfd_cache = None
        self._signal_layers = None

    # ── Main pipeline ───────────────────────────────────────────────

    def analyze_prompt(
        self,
        prompt: str,
        category: str = "",
        compute_kl: bool = False,
        compute_full_trajectory: bool = True,
        capture_responses: bool = False,
        full_capture: bool = False,
        compute_ltp: bool = False,
        compute_sfd: bool = False,
        ltp_k: int = 8,
        ltp_layer_strategy: str = "signal",
        ltp_svd_rank: int = 0,
        response_topk: int | None = None,
        base_cache: dict | None = None,
    ) -> PromptResult:
        """Run the full analysis pipeline in a SINGLE forward pass.

        Per-request flags control what gets computed — same contract as
        TASM's Analyzer.analyze_prompt(). No pre-configuration needed.
        """
        if response_topk is None:
            response_topk = engine_config.get("response_topk")

        result = PromptResult(prompt=prompt, category=category)
        result.signal_layer_indices = list(self.signal_layers)
        result.full_capture_enabled = full_capture

        # Determine which extra layers to hook
        ltp_layers = None
        if compute_ltp and ltp_layer_strategy == "late":
            late_start = 2 * self.n_layers // 3
            ltp_layers = list(range(late_start, self.n_layers))

        domain_frac = max(0, min(1, engine_config.get("domain_embedding_layer_frac") or 0.50))
        domain_layer = max(0, min(self.n_layers - 1, int(domain_frac * self.n_layers)))
        esc_frac = max(0, min(1, engine_config.get("domain_escalation_layer_frac") or 0.75))
        escalation_layer = max(0, min(self.n_layers - 1, int(esc_frac * self.n_layers)))

        need_attn_output = full_capture

        # Install hooks and run forward pass.
        #
        # Everything from here until the matching remove() runs inside a
        # try/finally.  Without it, any exception in the ~90 lines of
        # measurement below leaves capture hooks attached to the SHARED
        # instruct model: app_core catches per-prompt exceptions and carries
        # on, so subsequent chat generations, probe embeddings and module runs
        # would then execute with live capture hooks holding activation
        # tensors.  That is exactly the interference MODEL_LOCK exists to
        # prevent: all model access is serialized through it precisely so one
        # caller cannot leave the shared model in a modified state.
        self._capture.install(
            self.instruct_model, self.adapter,
            signal_layers=self.signal_layers,
            full_trajectory=compute_full_trajectory or full_capture,
            ltp_layers=ltp_layers,
            domain_layer=domain_layer,
            escalation_layer=escalation_layer,
            output_attentions=need_attn_output,
        )
        try:
            return self._analyze_hooked(
                result, prompt, compute_kl, compute_full_trajectory,
                capture_responses, full_capture, compute_ltp, compute_sfd,
                ltp_k, ltp_layer_strategy, ltp_svd_rank, response_topk,
                base_cache, need_attn_output, domain_layer, escalation_layer,
            )
        finally:
            # Idempotent: remove() clears the handle list, and the normal path
            # below has already called it.  clear() too — remove() detaches the
            # hooks but does not drop the captured activation tensors, so on
            # the exception path they would stay referenced on the shared
            # ActivationCapture until the next prompt overwrote them.
            self._capture.clear()
            self._capture.remove()

    def _analyze_hooked(
        self, result, prompt, compute_kl, compute_full_trajectory,
        capture_responses, full_capture, compute_ltp, compute_sfd,
        ltp_k, ltp_layer_strategy, ltp_svd_rank, response_topk,
        base_cache, need_attn_output, domain_layer, escalation_layer,
    ) -> PromptResult:
        """Body of analyze_prompt that runs with capture hooks installed.

        Split out solely so the caller can guarantee hook removal in a
        `finally`; the logic is unchanged.
        """
        tokens, inputs, model_out = self._capture.forward(
            self.instruct_model, self.tokenizer, prompt,
            output_attentions=need_attn_output,
        )

        result.tokens = tokens
        result.seq_len = len(tokens)
        seq_len = result.seq_len

        # Instruct top-k predictions
        if capture_responses or compute_kl:
            logits = model_out.logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            topk = torch.topk(probs, min(response_topk, probs.shape[0]))
            precision = engine_config.get("serialization_precision")
            result.instruct_topk = [
                (self.tokenizer.decode(idx.item()).strip(),
                 round(p.item(), precision))
                for idx, p in zip(topk.indices, topk.values)
            ]

        # Core extractions (all read from self._capture.activations/attn_weights)
        self._extract_signed_attribution(result, seq_len)
        self._extract_stress_score(result, seq_len)

        # Model-intrinsic scale
        signal_norms = []
        for li in self.signal_layers:
            for role in ("q", "k", "v"):
                fn = self.delta_store.frob_norm(li, role) if self.delta_store.has(li, role) else None
                if fn is not None:
                    signal_norms.append(fn)
        result.delta_scale = float(np.mean(signal_norms)) if signal_norms else 1.0
        result.spectral_summary = self.delta_store.aggregate_spectral_summary()

        if compute_full_trajectory or full_capture:
            self._extract_amplitude_trajectory(result, seq_len)

        if full_capture and result.heatmap is not None:
            self._compute_full_capture_metrics(result, seq_len)

        # LTP
        if compute_ltp:
            precomputed_base_alts = None
            base_logits = None
            if base_cache is not None:
                precomputed_base_alts = base_cache.get("per_position_base_alts")
            elif self.base_model is not None:
                with torch.no_grad():
                    base_out = self.base_model(**inputs)
                    base_logits = base_out.logits
                    del base_out

            svd_cache = None
            if ltp_svd_rank and ltp_svd_rank > 0:
                svd_cache = self._svd_caches.get(ltp_svd_rank)
                if svd_cache is None:
                    from src.engine.ltp import precompute_svd_cache
                    svd_cache = precompute_svd_cache(self, rank=ltp_svd_rank)
                    self._svd_caches[ltp_svd_rank] = svd_cache

            result.ltp = self._compute_ltp(
                model_out.logits, tokens, inputs["input_ids"],
                k=ltp_k, layer_strategy=ltp_layer_strategy,
                svd_rank=ltp_svd_rank, svd_cache=svd_cache,
                base_logits=base_logits,
                precomputed_base_alts=precomputed_base_alts,
            )

        # SFD
        if compute_sfd:
            result.sfd = self._compute_sfd(seq_len)

        # Domain embeddings
        self._extract_domain_embeddings(
            result, seq_len, domain_layer, escalation_layer)

        # Free activations before base-model pass
        self._capture.clear()
        self._capture.remove()

        # Base-model comparison (KL, topk, counterfactuals)
        needs_base = compute_kl or capture_responses or compute_ltp
        has_cache = base_cache is not None
        has_base = self.base_model is not None

        if needs_base and has_cache:
            self._apply_base_cache(result, model_out, compute_kl, capture_responses, base_cache)
        elif needs_base and has_base:
            self._compute_behavioral_comparison(
                result, instruct_logits=model_out.logits,
                compute_kl=compute_kl, capture_base=capture_responses,
                topk=response_topk,
            )

        # Rank displacement
        if (compute_sfd or compute_ltp) and result.ltp is not None:
            self._compute_rank_displacement(result)

        return result

    # ── Extraction functions ────────────────────────────────────────

    def _extract_signed_attribution(self, result: PromptResult, seq_len: int):
        """Signed attribution with per-layer detail and proof-1 checks.

        Reads: activations["layer_N_h"], attn_weights["layer_N_attn"]
        Uses: delta_store.v_delta(layer_idx)
        """
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        n_heads = self.n_heads
        heads_per_kv = n_heads // n_kv_heads

        layer_attrs = []
        n_missing_attn = 0

        for layer_idx in self.signal_layers:
            h_key = f"layer_{layer_idx}_h"
            a_key = f"layer_{layer_idx}_attn"

            if h_key not in self._capture.activations or a_key not in self._capture.attn_weights:
                if a_key not in self._capture.attn_weights:
                    n_missing_attn += 1
                continue

            h = self._capture.activations[h_key][0, :seq_len].float()
            alpha = self._capture.attn_weights[a_key][0, :, :seq_len, :seq_len].float()

            dw_v = self.delta_store.v_delta_or_none(layer_idx)
            if dw_v is None:
                continue

            v = torch.matmul(h, dw_v.float().T)
            v_heads = v.view(seq_len, n_kv_heads, head_dim)

            alpha_grouped = alpha.view(n_kv_heads, heads_per_kv, seq_len, seq_len)
            alpha_kv = alpha_grouped.mean(dim=1)

            head_attrs = []
            layer_amp = 0.0
            for kv_head in range(n_kv_heads):
                v_h = v_heads[:, kv_head, :]
                a_h = alpha_kv[kv_head]
                delta = torch.matmul(a_h, v_h)
                d_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                u_hat = delta / d_norm
                proj = torch.matmul(u_hat, v_h.T)
                signed = a_h * proj
                head_attrs.append(signed[-1, :])
                layer_amp += d_norm[-1].item()

                # NOTE: this check is an ALGEBRAIC IDENTITY, not a validation.
                # With u_hat = delta/||delta|| and signed_j = a_j (u_hat . v_j),
                #   sum_j a_j (u_hat . v_j) = u_hat . (sum_j a_j v_j)
                #                           = u_hat . delta = ||delta||
                # holds for ANY a and v.  It therefore passes regardless of
                # whether dw_v, the head reshape or the GQA grouping is
                # correct, and must not be read as evidence that the
                # attribution decomposition was verified.  Retained only as a
                # float-accumulation sanity check; "exact" means the sum did
                # not lose precision, nothing more.
                attr_sum = signed[-1, :].sum().item()
                delta_norm_val = d_norm[-1].item()
                error = abs(attr_sum - delta_norm_val)
                result.proof1_checks.append({
                    "layer": layer_idx, "head": kv_head,
                    "attr_sum": round(attr_sum, 8),
                    "delta_norm": round(delta_norm_val, 8),
                    "error": float(f"{error:.2e}"),
                    "exact": error < engine_config.get("proof1_threshold"),
                    "identity_only": True,
                })

            layer_attr = torch.stack(head_attrs).mean(dim=0)
            layer_attrs.append(layer_attr)
            result.per_layer_amplitude[layer_idx] = layer_amp / n_kv_heads
            result.per_layer_signed_attr[layer_idx] = layer_attr.float().numpy().tolist()

        if not layer_attrs:
            # Every attribution-derived field (entropy, top2_share,
            # middle_share, interior_cv, net_correction, n_negative_tokens,
            # per_layer_amplitude) keeps its dataclass default of 0.0 here.
            # Those zeros are indistinguishable from measured zeros once they
            # reach bootstrap_ci, so record WHY they are absent instead of
            # returning silently.  The usual cause is attention weights not
            # being captured: need_attn_output is driven by full_capture,
            # which defaults to False.
            result.attribution_unavailable = (
                "no attention weights captured "
                f"({n_missing_attn}/{len(self.signal_layers)} signal layers "
                "missing); enable full_capture to compute signed attribution"
                if n_missing_attn else "no signal layer had both activations "
                                       "and a v-projection delta"
            )
            logger.warning("[ATTR] %s", result.attribution_unavailable)
            return

        avg_attr = torch.stack(layer_attrs).mean(dim=0).float().numpy()
        result.signed_attr = avg_attr
        # EXTENSIVE: a sum over token positions.  Unlike n_directional this
        # does not shift the MEAN with length (the per-token terms are
        # roughly zero-centred, so the sum stays near zero) — but its SD grows
        # as sqrt(seq_len).  Measured: SD 4.4 at 20 tokens, 12.6 at 160.
        #
        # That is heteroscedasticity, not a spurious effect: it inflates the
        # pooled SD that Cohen's d divides by, so it costs power and violates
        # the equal-variance assumption whenever the two groups differ in
        # average length.  The sum IS the definition of "net" correction, so
        # it is left as-is rather than silently redefined — but treat
        # net_correction comparisons between groups of different typical
        # length with suspicion, and check length_correlations first.
        result.net_correction = float(avg_attr.sum())
        result.n_negative_tokens = int(sum(1 for a in avg_attr if a < 0))
        result.has_negative_tokens = result.n_negative_tokens > 0

        # Distribution metrics
        attr_abs = np.abs(avg_attr)
        total = attr_abs.sum()
        attr_dist = attr_abs / total if total > 0 else np.ones_like(attr_abs) / len(attr_abs)

        ent = -np.sum(attr_dist * np.log(attr_dist + 1e-10))
        max_ent = np.log(seq_len) if seq_len > 1 else 1.0
        result.entropy = float(ent / max_ent)

        n = len(attr_dist)
        boundary = max(1, round(engine_config.get("boundary_fraction") * n))
        if n >= 2:
            result.top2_share = float(attr_dist[:boundary].sum() + attr_dist[-boundary:].sum())
            interior = attr_dist[boundary:-boundary] if n > 2 * boundary else np.array([])
            if len(interior) > 0:
                result.middle_share = float(interior.sum())
                imean = interior.mean()
                result.interior_cv = float(interior.std() / imean) if imean > 0 else 0.0
            else:
                result.middle_share = float(1.0 - result.top2_share)
                result.interior_cv = 0.0
        else:
            result.top2_share = 1.0
            result.middle_share = 0.0
            result.interior_cv = 0.0

    def _extract_stress_score(self, result: PromptResult, seq_len: int):
        """Per-token stress: ||h @ dW_p^T|| / ||dW_p||_F for p in {q,k,v}."""
        # Accumulator must live on the same device as the activations it sums.
        # A bare torch.zeros() is CPU-only and happens to work solely because
        # Pipeline.device defaults to "cpu"; it raises on any GPU/MPS run.
        per_token_total = torch.zeros(seq_len, device=self.device)
        n_layers = 0

        for layer_idx in self.signal_layers:
            key = f"layer_{layer_idx}_h"
            if key not in self._capture.activations:
                continue
            h = self._capture.activations[key]

            # Accumulate this layer separately so it can be averaged over the
            # roles that ACTUALLY contributed.  Summing over roles and then
            # dividing only by the layer count meant a layer holding 2 of 3
            # deltas contributed 2/3 the magnitude of a complete one — the
            # same aggregation asymmetry as the amplitude trajectory — and a
            # layer with NO usable deltas still incremented n_layers, silently
            # deflating the score in proportion to how many deltas were
            # missing.  Values are now per-role, i.e. ~1/3 of previously
            # stored scores when all three roles are present.
            layer_total = torch.zeros(seq_len, device=self.device)
            n_roles = 0
            for role in ("q", "k", "v"):
                dw = self.delta_store.get_or_none(layer_idx, role)
                if dw is None:
                    continue
                fnorm = self.delta_store.frob_norm(layer_idx, role)
                if dw.shape[1] == h.shape[2] and fnorm > 0:
                    projected = torch.matmul(h[0, :seq_len].float(), dw.float().T)
                    layer_total += projected.norm(dim=-1) / fnorm
                    n_roles += 1
            if n_roles == 0:
                continue
            per_token_total += layer_total / n_roles
            n_layers += 1

        if n_layers > 0:
            per_token_total /= n_layers

        result.stress_score = float(per_token_total.mean().item())
        result.per_token_stress = per_token_total.cpu().numpy()

    def _extract_amplitude_trajectory(self, result: PromptResult, seq_len: int):
        """Amplitude trajectory across all sublayers (attn + MLP per layer)."""
        raw_traj = []
        norm_traj = []
        heatmap_rows = []

        for layer_idx in range(self.n_layers):
            for sublayer_type in ("attn", "mlp"):
                key = f"layer_{layer_idx}_traj_{sublayer_type}"
                if key not in self._capture.activations:
                    raw_traj.append(0.0)
                    norm_traj.append(0.0)
                    heatmap_rows.append(np.zeros(seq_len))
                    continue

                h = self._capture.activations[key]

                if sublayer_type == "attn":
                    roles = ("q", "k", "v")
                else:
                    roles = ("gate", "up")

                raw_sum = 0.0
                norm_sum = 0.0
                n_roles = 0
                # Device-matched for the same reason as in _extract_stress_score.
                per_tok = torch.zeros(seq_len, device=self.device)

                for role in roles:
                    dw = self.delta_store.get_or_none(layer_idx, role)
                    if dw is None:
                        continue
                    fnorm = self.delta_store.frob_norm(layer_idx, role)
                    if dw.shape[1] == h.shape[2] and fnorm > 0:
                        h_slice = h[0, :seq_len].float()
                        projected = torch.matmul(h_slice, dw.float().T)
                        pn = projected.norm(dim=-1)
                        raw_sum += pn.mean().item()
                        norm_sum += (pn / fnorm).mean().item()
                        per_tok += pn / fnorm
                        n_roles += 1

                # MEAN over roles, not sum.  attn aggregates 3 roles (q,k,v)
                # and mlp only 2 (gate,up), so summing made the two sublayer
                # types differ by a constant ~3/2 factor that had nothing to
                # do with the model.  Interleaved on one axis (index =
                # 2*layer + {0:attn, 1:mlp}) that produced a period-2 sawtooth
                # roughly 6x larger than the real layer-to-layer variation,
                # burying the depth trend the trajectory exists to show.
                #
                # Worse, the factor is multiplicative, so it did NOT cancel in
                # comparative.plot_difference_from_benign — that subtraction is
                # index-wise, so the artifact modulated the difference and made
                # attention sublayers look ~1.5x more discriminative than MLP
                # sublayers purely from role count.
                #
                # Dividing makes the two types genuinely comparable. Values are
                # now attn/3 and mlp/2 relative to previously stored sessions.
                if n_roles:
                    raw_sum /= n_roles
                    norm_sum /= n_roles
                    per_tok = per_tok / n_roles

                raw_traj.append(raw_sum)
                norm_traj.append(norm_sum)
                heatmap_rows.append(per_tok.float().cpu().numpy())

        result.amplitude_trajectory = raw_traj
        result.amplitude_normalized = norm_traj
        result.heatmap = np.array(heatmap_rows)

    def _compute_full_capture_metrics(self, result: PromptResult, seq_len: int):
        """Coherence, spectral rank, attn/MLP split, and similarity from heatmap."""
        hm = result.heatmap
        if hm is None or hm.size == 0:
            return
        n_sub, n_tok = hm.shape
        if n_tok != seq_len or n_sub < 2:
            return

        # Attn vs MLP fraction
        attn_rows = hm[0::2]
        mlp_rows = hm[1::2]
        attn_sum = attn_rows.sum(axis=0)
        mlp_sum = mlp_rows.sum(axis=0)
        total = attn_sum + mlp_sum
        result.attn_frac = np.where(total > 0, attn_sum / total, 0.5)

        # Per-token coherence
        profiles = hm.T
        coherence = np.zeros(n_tok)
        for i in range(n_tok):
            p = profiles[i]
            total_p = p.sum()
            if total_p > 0:
                normed = p / total_p
                ent = -np.sum(normed * np.log(normed + 1e-10))
                max_ent = np.log(n_sub) if n_sub > 1 else 1.0
                coherence[i] = 1.0 - (ent / max_ent)
        result.per_token_coherence = coherence

        # Per-token spectral rank
        precision = engine_config.get("serialization_precision")
        spectral_ranks = []
        for i in range(n_tok):
            p = profiles[i]
            nonzero = p[p > 1e-8]
            if len(nonzero) > 1:
                normed = nonzero / nonzero.sum()
                ent = -np.sum(normed * np.log(normed))
                spectral_ranks.append(round(float(np.exp(ent)), precision))
            else:
                spectral_ranks.append(1.0)
        result.per_token_spectral_rank = spectral_ranks

        # Token × token similarity
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        normed_profiles = profiles / norms
        result.token_similarity = normed_profiles @ normed_profiles.T

    def _extract_domain_embeddings(self, result: PromptResult, seq_len: int,
                                    domain_layer: int, escalation_layer: int):
        """Capture per-token embeddings at domain and escalation depths."""
        skip_first = not engine_config.get("include_first_token")
        start_pos = 1 if skip_first else 0
        precision = engine_config.get("serialization_precision")
        use_projection = engine_config.get("probe_projection_space")

        def _project_and_normalize(raw_tok, layer_idx):
            tok = raw_tok
            if use_projection:
                delta = self.delta_store.o_delta_or_none(layer_idx)
                if delta is not None:
                    tok = torch.matmul(
                        torch.tensor(raw_tok, dtype=torch.float32),
                        delta.float().cpu().T
                    ).numpy()
            norms = np.linalg.norm(tok, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            return tok / norms

        # Subject layer
        de_key = f"layer_{domain_layer}_h"
        de_act = self._capture.activations.get(de_key)
        if de_act is not None and seq_len > 1:
            raw_tok = de_act[0, start_pos:seq_len].float().cpu().numpy()
            result.per_token_domain_emb = _project_and_normalize(raw_tok, domain_layer).tolist()
            result.per_token_domain_offset = start_pos

            emb = raw_tok.mean(axis=0)
            norm = float(np.linalg.norm(emb))
            if norm > 1e-12:
                emb = emb / norm
            result.domain_embedding = [round(float(x), precision) for x in emb]

        # Escalation layer
        if escalation_layer != domain_layer:
            esc_key = f"layer_{escalation_layer}_h"
            esc_act = self._capture.activations.get(esc_key)
            if esc_act is not None and seq_len > 1:
                esc_raw = esc_act[0, start_pos:seq_len].float().cpu().numpy()
                result.per_token_escalation_emb = _project_and_normalize(
                    esc_raw, escalation_layer).tolist()
        else:
            result.per_token_escalation_emb = result.per_token_domain_emb

        # Final norm hidden state
        final_act = self._capture.activations.get("final_norm_h")
        if final_act is not None and seq_len > 1:
            final_raw = final_act[0, start_pos:seq_len].float().cpu().numpy()
            norms = np.linalg.norm(final_raw, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            result.per_token_final_emb = (final_raw / norms).tolist()

    # ── LTP / SFD (adapter-mediated) ────────────────────────────────

    def _compute_ltp(self, logits, tokens, input_ids,
                     k=8, layer_strategy="signal", svd_rank=0, svd_cache=None,
                     base_logits=None, precomputed_base_alts=None) -> LTPResult:
        """Compute Lateral Tension Profile using adapter-mediated delta access.

        This wraps TASM's LTP computation with adapter-based model access.
        The core math is identical; only the model-structure queries differ.
        """
        from src.engine.ltp import compute_ltp
        return compute_ltp(
            self, logits, tokens, input_ids,
            k=k, layer_strategy=layer_strategy,
            svd_cache=svd_cache, svd_rank=svd_rank,
            base_logits=base_logits,
            precomputed_base_alts=precomputed_base_alts,
        )

    def _compute_sfd(self, seq_len: int) -> Optional[SFDResult]:
        """Compute Spectral Field Density from cached activations."""
        from src.engine.sfd import compute_sfd, precompute_sfd_cache

        try:
            if self._sfd_cache is None:
                self._sfd_cache = precompute_sfd_cache(self)
            layer_acts = {}
            for layer_idx in self._sfd_cache["layers"]:
                act_key = f"layer_{layer_idx}_h"
                act = self._capture.activations.get(act_key)
                if act is not None:
                    layer_acts[layer_idx] = act[0, :seq_len].float().cpu().numpy()
            if layer_acts:
                return compute_sfd(layer_acts, self._sfd_cache)
        except Exception as e:
            logger.warning(f"[SFD] Computation failed: {e}")
        return None

    def _compute_rank_displacement(self, result: PromptResult):
        """Rank displacement between instruct and base counterfactual orderings."""
        from src.engine.sfd import compute_rank_displacement
        try:
            inst_cf = result.ltp.counterfactual_tokens if result.ltp else []
            base_cf = result.base_counterfactual_tokens
            if inst_cf and base_cf:
                result.rank_displacement = compute_rank_displacement(inst_cf, base_cf)
                rd = result.rank_displacement
                if rd and rd.get("mean_tau") is not None:
                    logger.info(f"[RANK] tau={rd['mean_tau']:+.3f}, "
                                f"overlap={rd['mean_overlap']:.3f}")
        except Exception as e:
            logger.warning(f"[RANK] Displacement failed: {e}")

    # ── Base model comparison ───────────────────────────────────────

    def _apply_base_cache(self, result: PromptResult, model_out,
                          compute_kl: bool, capture_responses: bool,
                          base_cache: dict):
        """Populate result from pre-computed base-model cache (sequential mode)."""
        precision = engine_config.get("serialization_precision")

        if capture_responses and "base_topk" in base_cache:
            result.base_topk = base_cache["base_topk"]
        if "base_counterfactual_tokens" in base_cache:
            result.base_counterfactual_tokens = base_cache["base_counterfactual_tokens"]

        if compute_kl and "base_log_softmax" in base_cache:
            base_log_p = base_cache["base_log_softmax"]
            if base_log_p is not None:
                # KL over a ~10^5-entry vocabulary: do the reduction in
                # float32 — bf16/fp16 accumulation visibly distorts it.
                logits_i = model_out.logits[0].float()
                log_p_i = torch.log_softmax(logits_i, dim=-1)
                p_i = torch.softmax(logits_i, dim=-1)
                if not isinstance(base_log_p, torch.Tensor):
                    base_log_p = torch.tensor(base_log_p, device=logits_i.device,
                                              dtype=torch.float32)
                else:
                    base_log_p = base_log_p.float()
                per_tok_kl = (p_i * (log_p_i - base_log_p)).sum(dim=-1)
                result.per_token_kl = per_tok_kl.float().cpu().numpy()
                result.kl_divergence = float(per_tok_kl[-1].item())

    def _compute_behavioral_comparison(self, result: PromptResult,
                                        instruct_logits=None,
                                        compute_kl=False, capture_base=False,
                                        topk=10):
        """KL divergence, base predictions, and counterfactuals (live base model)."""
        precision = engine_config.get("serialization_precision")
        inputs = self.tokenizer(
            result.prompt, return_tensors="pt",
            add_special_tokens=engine_config.get("add_special_tokens"),
        ).to(self.device)

        with torch.no_grad():
            logits_i = instruct_logits[0] if instruct_logits is not None else None
            out_base = self.base_model(**inputs)

            if compute_kl and logits_i is not None:
                logits_if = logits_i.float()
                logits_b = out_base.logits[0].float()
                log_p_i = torch.log_softmax(logits_if, dim=-1)
                log_p_b = torch.log_softmax(logits_b, dim=-1)
                p_i = torch.softmax(logits_if, dim=-1)
                per_tok_kl = (p_i * (log_p_i - log_p_b)).sum(dim=-1)
                result.per_token_kl = per_tok_kl.float().cpu().numpy()
                result.kl_divergence = float(per_tok_kl[-1].item())

            if capture_base:
                logits_base = out_base.logits[0, -1, :]
                probs_base = torch.softmax(logits_base.float(), dim=-1)
                tk = torch.topk(probs_base, min(topk, probs_base.shape[0]))
                result.base_topk = [
                    (self.tokenizer.decode(idx.item()).strip(),
                     round(p.item(), precision))
                    for idx, p in zip(tk.indices, tk.values)
                ]

            # Per-token base counterfactuals — full-vocab softmax via the
            # shared extractor, so masses match the instruct side exactly.
            base_logits_full = out_base.logits[0]
            token_ids = inputs["input_ids"][0]
            ltp_k = result.ltp.k if result.ltp else 8
            base_cf = []
            for i in range(base_logits_full.shape[0]):
                alts = top_alternatives(base_logits_full[i],
                                        token_ids[i].item(), ltp_k)
                base_cf.append(
                    decode_alternatives(alts, self.tokenizer, precision))
            result.base_counterfactual_tokens = base_cf
            del base_logits_full, out_base

    # ── Sequential base phase (batch mode) ──────────────────────────

    def run_base_phase(self, prompts: list[dict], ltp_k: int = 8,
                       compute_kl: bool = False,
                       capture_responses: bool = False,
                       response_topk: int = 10,
                       progress=None) -> list[dict]:
        """Load base model, cache base outputs for all prompts, unload.

        Returns list of base_cache dicts, one per prompt.
        """
        precision = engine_config.get("serialization_precision")

        if progress:
            progress("base_phase", "Loading base model for sequential phase...")

        # If the base model is already resident (e.g. the chat panel's
        # inference-class toggle loaded it), borrow it and leave it loaded
        # afterwards instead of yanking it out from under the other feature.
        base_was_loaded = self.pipeline.base_model is not None
        self.pipeline.load_base(progress=progress)
        if self.base_model is None:
            logger.error("[BASE PHASE] Failed to load base model")
            return [{}] * len(prompts)

        all_caches = []
        try:
            for pi, p in enumerate(prompts):
                prompt_text = p["prompt"]
                if progress and (pi + 1) % 5 == 0:
                    progress("base_phase",
                             f"[Base phase {pi+1}/{len(prompts)}] {prompt_text[:40]}...")

                inputs = self.tokenizer(
                    prompt_text, return_tensors="pt",
                    add_special_tokens=engine_config.get("add_special_tokens"),
                ).to(self.device)
                token_ids = inputs["input_ids"][0]
                seq_len = token_ids.shape[0]

                with torch.no_grad():
                    out_base = self.base_model(**inputs)
                    base_logits = out_base.logits[0]

                cache = {}

                # Per-position base alternatives (for LTP + base bank).
                # Full-vocab softmax via the shared extractor — identical
                # normalization to the instruct side.
                base_alts = []
                base_cf = []
                for i in range(seq_len):
                    alts = top_alternatives(base_logits[i],
                                            token_ids[i].item(), ltp_k)
                    base_alts.append(alts)
                    base_cf.append(
                        decode_alternatives(alts, self.tokenizer, precision))

                cache["per_position_base_alts"] = base_alts
                cache["base_counterfactual_tokens"] = base_cf

                if capture_responses:
                    last_probs = torch.softmax(base_logits[-1].float(), dim=-1)
                    tk = torch.topk(last_probs, min(response_topk, last_probs.shape[0]))
                    cache["base_topk"] = [
                        (self.tokenizer.decode(idx.item()).strip(),
                         round(p_val.item(), precision))
                        for idx, p_val in zip(tk.indices, tk.values)
                    ]

                if compute_kl:
                    # Computed in float32; stored fp16 only to bound memory
                    # (seq × vocab per prompt). _apply_base_cache re-casts to
                    # float32 before the KL reduction.
                    cache["base_log_softmax"] = torch.log_softmax(
                        base_logits.float(), dim=-1).cpu().numpy().astype(np.float16)

                del base_logits, out_base
                all_caches.append(cache)

                if (pi + 1) % 20 == 0:
                    gc.collect()

        finally:
            if base_was_loaded:
                if progress:
                    progress("base_phase",
                             "Base model left loaded (in use by chat).")
            else:
                if progress:
                    progress("base_phase", "Unloading base model...")
                self.pipeline.unload_base()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"[BASE PHASE] Complete: {len(all_caches)} prompts cached")
        if progress:
            progress("base_phase",
                     f"Base phase complete: {len(all_caches)} prompts cached")

        return all_caches
