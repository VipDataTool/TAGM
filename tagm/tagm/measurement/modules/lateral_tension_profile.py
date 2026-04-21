"""LateralTensionProfile (LTP): directional information from the correction field.

For each token position, projects k counterfactual alternatives through
ΔW_V/2 and W_O into residual-stream coordinates, decomposes into forward
(along the path tangent τ) and lateral components, and records the lateral
magnitudes as a "profile." Profile shape, peak concentration, and trajectory
in principal-components space are summary outputs.

Two banks per position:
  - Instruct bank: counterfactuals from the instruct model's top-k logits.
  - Base bank:     counterfactuals from the base model's top-k logits.

Asymmetry is a diagnostic of correction direction. A null instruct-base
pair should produce identical profiles.

Translated from TASM's `engine/ltp.py::compute_ltp`.

Capture expectation:
  Needs pre_attn_norm hidden states at one or more layers AND base logits.
  The user's CaptureConfig picks which layers; this measurement operates
  on layers captured at pre_attn_norm AND with v delta available, narrowed
  by the optional `layers` scope parameter.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from tagm.measurement.base import MeasurementModule
from tagm.measurement.parameters import ModuleParameter
from tagm.measurement.registry import register_measurement
from tagm.measurement.requirements import CaptureExpectation
from tagm.measurement.result import FieldSpec, MeasurementResult, padded_per_token
from tagm.measurement.scope import describe_scope_resolution, resolve_scope_layers

logger = logging.getLogger("tagm")


@register_measurement
class LateralTensionProfile(MeasurementModule):
    name = "lateral_tension_profile"
    display_name = "Lateral Tension Profile"
    description = (
        "Per-token lateral tension from the correction field, decomposed over "
        "k counterfactual alternatives. Produces instruct and base profiles, "
        "directional statistics, and a PCA-projected trajectory."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="layers",
            display_name="Scope layers",
            description=(
                "Optional subset of captured layers to monitor. Empty means "
                "all layers captured at pre_attn_norm that also have v delta "
                "available. Scope parameter; not a capture parameter."
            ),
            kind="layer_list", default=[],
        ),
        ModuleParameter(
            name="k",
            display_name="Counterfactuals per position (k)",
            description="Number of alternative tokens to probe per position.",
            kind="int", default=8, min_value=2, max_value=32,
        ),
        ModuleParameter(
            name="overfetch_first",
            display_name="Overfetch (first pass)",
            description=("Extra candidates beyond k on the first topk call to "
                         "ensure k non-chosen remain."),
            kind="int", default=1, min_value=0, max_value=32, advanced=True,
        ),
        ModuleParameter(
            name="overfetch_second",
            display_name="Overfetch (second pass)",
            description="Wider overfetch if first pass didn't yield k candidates.",
            kind="int", default=5, min_value=0, max_value=64, advanced=True,
        ),
        ModuleParameter(
            name="prc_threshold",
            display_name="Peak Rank Concentration threshold",
            description=("Per-token normalized-profile peak above uniform (1/k) "
                         "to count as 'directional'."),
            kind="float", default=0.02,
            min_value=0.0, max_value=1.0, advanced=True,
        ),
        ModuleParameter(
            name="svd_rank",
            display_name="SVD truncation rank (0 = off)",
            description=("If > 0, project through a rank-r truncated dW_V "
                         "restricted to the dominant singular directions."),
            kind="int", default=0, min_value=0, max_value=64, advanced=True,
        ),
    ]

    def capture_expectation(self, params):
        return CaptureExpectation(
            hook_points_required=("pre_attn_norm",),
            capture_types_required=frozenset({"hidden"}),
            needs_base_logits=True,
            min_layers_captured=1,
        )

    def base_extract(self, pipeline, prompt, base_logits, params):
        """Extract per-position base-model counterfactuals from base_logits."""
        k = int(params.get("k", 8))
        overfetch = int(params.get("overfetch_second", 5))
        tokenizer = pipeline.tokenizer

        inputs = tokenizer(prompt, return_tensors="pt").to(pipeline.device)
        token_ids = inputs["input_ids"][0]
        seq_len = token_ids.shape[0]

        per_position_base_alts = []
        base_cf = []
        for i in range(seq_len):
            chosen_id = token_ids[i].item()
            topk = torch.topk(base_logits[i], k + overfetch)
            probs = torch.softmax(topk.values, dim=-1)
            alts_ids = []
            alts_strs = []
            for j, tid in enumerate(topk.indices.tolist()):
                if tid != chosen_id and len(alts_ids) < k:
                    alts_ids.append((tid, probs[j].item()))
                    alts_strs.append((
                        tokenizer.decode(tid).strip(),
                        round(probs[j].item(), 8),
                    ))
            per_position_base_alts.append(alts_ids)
            base_cf.append(alts_strs)

        return {
            "per_position_base_alts": per_position_base_alts,
            "base_counterfactual_tokens": base_cf,
        }

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        scope = list(params.get("layers") or [])
        k = int(params.get("k", 8))
        overfetch_first = int(params.get("overfetch_first", 1))
        overfetch_second = int(params.get("overfetch_second", 5))
        prc_threshold = float(params.get("prc_threshold", 0.02))
        svd_rank = int(params.get("svd_rank", 0))

        seq_len = run_result.seq_len
        store = run_result.activations
        logits = run_result.logits
        pipeline = run_result.pipeline

        layers = resolve_scope_layers(
            activation_store=store,
            hook_point="pre_attn_norm",
            capture_type="hidden",
            scope=scope,
            required_delta_roles=("v",),
            delta_store=delta_store,
        )

        base_params = {
            "layers_requested": list(scope),
            "layers_used": list(layers),
            "k": k,
            "overfetch_first": overfetch_first,
            "overfetch_second": overfetch_second,
            "prc_threshold": prc_threshold,
            "svd_rank": svd_rank,
            "scope_resolution": describe_scope_resolution(
                scope, layers, "pre_attn_norm"),
        }

        if not layers:
            return self._empty_result(seq_len, k, base_params)

        n_heads = run_result.structure.n_attention_heads
        n_kv_heads = run_result.structure.n_kv_heads
        head_dim = run_result.structure.head_dim
        hidden_size = run_result.structure.hidden_size
        heads_per_kv = n_heads // n_kv_heads

        W_u = adapter.unembedding_weight(pipeline.instruct_model)

        svd_cache = (_svd_truncate_cache(delta_store, layers, svd_rank)
                     if svd_rank > 0 else {})

        token_ids = run_result.token_ids
        log_probs = logits[0]

        # Build instruct-side counterfactuals
        per_position_alts = []
        per_position_chosen = []
        counterfactual_tokens = []

        for i in range(seq_len):
            chosen_id = int(token_ids[i].item())
            per_position_chosen.append(chosen_id)

            topk = torch.topk(log_probs[i], k + overfetch_first)
            topk_ids = topk.indices.tolist()
            probs = torch.softmax(topk.values, dim=-1)

            alts = []
            for j, tid in enumerate(topk_ids):
                if tid != chosen_id and len(alts) < k:
                    alts.append((tid, probs[j].item()))

            if len(alts) < k:
                topk2 = torch.topk(log_probs[i], k + overfetch_second)
                probs2 = torch.softmax(topk2.values, dim=-1)
                have = {a[0] for a in alts}
                for j, tid in enumerate(topk2.indices.tolist()):
                    if tid != chosen_id and tid not in have and len(alts) < k:
                        alts.append((tid, probs2[j].item()))

            per_position_alts.append(alts)
            counterfactual_tokens.append([
                (pipeline.tokenizer.decode(aid).strip(), round(prob, 8))
                for aid, prob in alts
            ])

        # Base-side from batch-mode cache; empty falls back to instruct bank
        per_position_base_alts = []
        if base_cache and base_cache.get("per_position_base_alts"):
            per_position_base_alts = base_cache["per_position_base_alts"]

        # Per-layer accumulators
        all_layer_profiles: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        all_layer_base_profiles: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        all_layer_tension_points: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

        for layer_idx in layers:
            h = store.get(layer_idx, "pre_attn_norm", "hidden")[0]
            if h.shape[0] < seq_len:
                logger.warning(
                    f"[LTP] Activation too short at layer {layer_idx}: "
                    f"h={h.shape[0]} vs seq_len={seq_len}; skipping.")
                continue

            dw_v_full = svd_cache.get(layer_idx, delta_store.get(layer_idx, "v"))
            dw_v_half = dw_v_full * 0.5

            W_O = adapter.o_proj_weight(pipeline.instruct_model, layer_idx)

            def _project_alts(alts_list, chosen_id, tau, return_laterals=False):
                magnitudes = []
                laterals: Optional[list[tuple[float, torch.Tensor]]] = (
                    [] if return_laterals else None)
                for alt_id, alt_prob in alts_list:
                    d_ic = W_u[alt_id] - W_u[chosen_id]
                    delta_val = torch.matmul(dw_v_half, d_ic)
                    expanded = (delta_val
                                .view(n_kv_heads, head_dim)
                                .repeat_interleave(heads_per_kv, dim=0)
                                .reshape(-1))
                    proj = torch.matmul(W_O, expanded)
                    fwd = torch.dot(proj, tau) * tau
                    lateral = proj - fwd
                    magnitudes.append(lateral.norm().item())
                    if return_laterals:
                        laterals.append((alt_prob, lateral))
                while len(magnitudes) < k:
                    magnitudes.append(0.0)
                return magnitudes, laterals

            for i in range(seq_len):
                inst_alts = per_position_alts[i]
                base_alts = (per_position_base_alts[i]
                             if i < len(per_position_base_alts) else inst_alts)
                chosen_id = per_position_chosen[i]

                if not inst_alts:
                    all_layer_tension_points[layer_idx].append(
                        torch.zeros(hidden_size, dtype=h.dtype))
                    all_layer_profiles[layer_idx].append(np.zeros(k, dtype=float))
                    all_layer_base_profiles[layer_idx].append(np.zeros(k, dtype=float))
                    continue

                if i > 0:
                    diff = h[i] - h[i - 1]
                    dn = diff.norm()
                    tau = diff / dn if dn > 1e-8 else torch.zeros_like(h[i])
                else:
                    if seq_len > 1:
                        diff = h[1] - h[0]
                        dn = diff.norm()
                        tau = diff / dn if dn > 1e-8 else torch.zeros_like(h[0])
                    else:
                        tau = torch.zeros_like(h[0])

                instruct_magnitudes, inst_laterals = _project_alts(
                    inst_alts, chosen_id, tau, return_laterals=True)
                base_magnitudes, _ = _project_alts(base_alts, chosen_id, tau)

                weighted_tension = torch.zeros(hidden_size, dtype=h.dtype)
                prob_sum = 0.0
                for alt_prob, lateral in inst_laterals or []:
                    weighted_tension += alt_prob * lateral
                    prob_sum += alt_prob
                if prob_sum > 0:
                    weighted_tension = weighted_tension / prob_sum

                all_layer_tension_points[layer_idx].append(weighted_tension)
                all_layer_profiles[layer_idx].append(
                    np.array(instruct_magnitudes[:k], dtype=float))
                all_layer_base_profiles[layer_idx].append(
                    np.array(base_magnitudes[:k], dtype=float))

        profiles: list[np.ndarray] = []
        base_profiles: list[np.ndarray] = []
        tension_points: list[np.ndarray] = []
        tension_magnitudes: list[float] = []
        profile_shapes: list[str] = []

        for i in range(seq_len):
            lp = [all_layer_profiles[l][i] for l in layers
                  if i < len(all_layer_profiles[l])]
            lbp = [all_layer_base_profiles[l][i] for l in layers
                   if i < len(all_layer_base_profiles[l])]
            lt = [all_layer_tension_points[l][i] for l in layers
                  if i < len(all_layer_tension_points[l])]

            avg_profile = np.mean(lp, axis=0) if lp else np.zeros(k)
            avg_base = np.mean(lbp, axis=0) if lbp else np.zeros(k)
            profiles.append(avg_profile.astype(float))
            base_profiles.append(avg_base.astype(float))

            avg_tension = (torch.stack(lt).mean(dim=0) if lt
                           else torch.zeros(hidden_size))
            tension_points.append(avg_tension.float().cpu().numpy().astype(float))
            tension_magnitudes.append(float(avg_tension.norm().item()))
            profile_shapes.append(_classify_profile(avg_profile))

        offset_magnitude: dict[int, float] = {}
        offset_variance: dict[int, float] = {}
        lateral_coverage: dict[int, float] = {}

        for l in layers:
            points = all_layer_tension_points.get(l, [])
            if not points:
                continue
            mags = [p.norm().item() for p in points]
            non_zero = [idx for idx, m in enumerate(mags) if m > 1e-10]
            lateral_coverage[l] = len(non_zero) / seq_len if seq_len > 0 else 0.0
            if not non_zero:
                offset_magnitude[l] = 0.0
                offset_variance[l] = 0.0
                continue
            active = torch.stack([points[idx] for idx in non_zero])
            mean_offset = active.mean(dim=0)
            offset_magnitude[l] = float(mean_offset.norm().item())
            offset_variance[l] = float(np.var([mags[idx] for idx in non_zero]))

        mean_M = (float(np.mean([offset_magnitude.get(l, 0.0) for l in layers]))
                  if layers else 0.0)
        mean_V = (float(np.mean([offset_variance.get(l, 0.0) for l in layers]))
                  if layers else 0.0)
        mean_L = (float(np.mean([lateral_coverage.get(l, 0.0) for l in layers]))
                  if layers else 0.0)

        prc_values = []
        for p in profiles:
            if len(p) >= 2:
                total = float(np.sum(p))
                if total > 0:
                    normed = p / total
                    prc = float(np.max(normed) - 1.0 / k)
                else:
                    prc = 0.0
            else:
                prc = 0.0
            prc_values.append(prc)
        max_prc = float(max(prc_values)) if prc_values else 0.0
        n_directional = int(sum(1 for v in prc_values if v > prc_threshold))

        semantic_traj, tension_traj = _compute_dual_trajectory_2d(
            store, layers, seq_len, tension_points)

        if base_profiles and profiles:
            n = min(len(base_profiles), len(profiles))
            diff = float(np.mean([
                np.sum(np.abs(profiles[i] - base_profiles[i]))
                for i in range(n)
            ])) if n > 0 else 0.0
            logger.info(
                f"[LTP] mean |instruct - base| across positions = {diff:.6f} "
                f"{'(identical — base bank missing?)' if diff < 1e-10 else '(asymmetric ✓)'}")

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters=base_params,
        )

        result.scalars["mean_M"] = mean_M
        result.scalars["mean_V"] = mean_V
        result.scalars["mean_L"] = mean_L
        result.scalars["max_prc"] = max_prc
        result.scalars["n_directional"] = n_directional
        result.scalars["n_layers_used"] = len(layers)

        result.per_token["tension_magnitude"] = np.array(tension_magnitudes, dtype=float)
        result.per_token["prc"] = np.array(prc_values, dtype=float)

        result.per_layer["offset_magnitude"] = {int(l): v for l, v in offset_magnitude.items()}
        result.per_layer["offset_variance"] = {int(l): v for l, v in offset_variance.items()}
        result.per_layer["lateral_coverage"] = {int(l): v for l, v in lateral_coverage.items()}

        result.objects["profiles"] = [p.tolist() for p in profiles]
        result.objects["base_profiles"] = [p.tolist() for p in base_profiles]
        result.objects["profile_shapes"] = list(profile_shapes)
        result.objects["counterfactual_tokens"] = counterfactual_tokens
        result.objects["semantic_trajectory_2d"] = (
            semantic_traj.tolist() if semantic_traj is not None else [])
        result.objects["tension_trajectory_2d"] = (
            tension_traj.tolist() if tension_traj is not None else [])

        self._annotate(result)
        return result

    def _empty_result(self, seq_len, k, params):
        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters=params,
        )
        result.per_token["tension_magnitude"] = padded_per_token(seq_len)
        result.per_token["prc"] = padded_per_token(seq_len)
        result.scalars["mean_M"] = 0.0
        result.scalars["mean_V"] = 0.0
        result.scalars["mean_L"] = 0.0
        result.scalars["max_prc"] = 0.0
        result.scalars["n_directional"] = 0
        result.scalars["n_layers_used"] = 0
        result.objects["profiles"] = []
        result.objects["base_profiles"] = []
        result.objects["profile_shapes"] = []
        result.objects["counterfactual_tokens"] = []
        result.objects["semantic_trajectory_2d"] = []
        result.objects["tension_trajectory_2d"] = []
        self._annotate(result)
        return result

    def _annotate(self, result: MeasurementResult) -> None:
        for name, desc in (
            ("mean_M", "Mean offset magnitude across monitored layers."),
            ("mean_V", "Mean offset variance across monitored layers."),
            ("mean_L", "Mean lateral coverage (fraction of positions with non-zero tension)."),
            ("max_prc", "Peak Rank Concentration of most-directional position."),
            ("n_directional", "Number of positions with PRC > threshold."),
            ("n_layers_used", "Count of layers actually aggregated."),
        ):
            result.field_specs[name] = FieldSpec(
                name=name, kind="scalar", description=desc, length_invariant=True,
            )
        result.field_specs["tension_magnitude"] = FieldSpec(
            name="tension_magnitude", kind="per_token",
            description="Per-token lateral tension magnitude.", length_invariant=False,
        )
        result.field_specs["prc"] = FieldSpec(
            name="prc", kind="per_token",
            description="Per-token Peak Rank Concentration: max of normalized "
                        "profile minus 1/k. >0 means directional preference.",
            length_invariant=False,
        )
        result.field_specs["offset_magnitude"] = FieldSpec(
            name="offset_magnitude", kind="per_layer",
            description="Per-layer mean-offset magnitude (M).", length_invariant=True,
        )
        result.field_specs["profiles"] = FieldSpec(
            name="profiles", kind="object",
            description="Per-position length-k lateral-magnitude profiles (instruct).",
            length_invariant=False,
        )
        result.field_specs["base_profiles"] = FieldSpec(
            name="base_profiles", kind="object",
            description="Per-position length-k lateral-magnitude profiles (base).",
            length_invariant=False,
        )
        result.field_specs["profile_shapes"] = FieldSpec(
            name="profile_shapes", kind="object",
            description="Per-position profile classification: 'steep' | 'inverted' | 'flat'.",
            length_invariant=False,
        )


# ── Helpers ─────────────────────────────────────────────────────────

def _classify_profile(profile: np.ndarray) -> str:
    if len(profile) < 2 or np.sum(profile) < 1e-10:
        return "flat"
    total = float(np.sum(profile))
    if total <= 0:
        return "flat"
    normed = profile / total
    if normed[0] > 0.4 and (len(normed) < 2 or normed[0] > 2 * normed[1]):
        return "steep"
    first_half = (float(np.mean(normed[:len(normed)//2])) if len(normed) >= 2
                  else normed[0])
    second_half = (float(np.mean(normed[len(normed)//2:])) if len(normed) >= 2
                   else 0.0)
    if second_half > first_half * 1.3:
        return "inverted"
    return "flat"


def _compute_dual_trajectory_2d(store, layers, seq_len, tension_points):
    if seq_len < 2 or not layers:
        return None, None
    layer_idx = layers[0]
    if not store.has(layer_idx, "pre_attn_norm", "hidden"):
        return None, None

    h = store.get(layer_idx, "pre_attn_norm", "hidden")[0].cpu().float().numpy()
    if h.shape[0] < seq_len:
        return None, None

    tension_traj_full = np.zeros_like(h)
    for i in range(seq_len):
        if i < len(tension_points):
            tp = tension_points[i]
            if len(tp) == h.shape[1]:
                tension_traj_full[i] = h[i] + tp
            else:
                tension_traj_full[i] = h[i]
        else:
            tension_traj_full[i] = h[i]

    combined = np.vstack([h, tension_traj_full])
    mean = combined.mean(axis=0)
    centered = combined - mean
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T
        return proj[:seq_len], proj[seq_len:]
    except (np.linalg.LinAlgError, ValueError):
        return None, None


def _svd_truncate_cache(delta_store, layers, rank):
    cache = {}
    for layer_idx in layers:
        dw_v = delta_store.get_or_none(layer_idx, "v")
        if dw_v is None:
            continue
        try:
            U, S, Vt = torch.linalg.svd(dw_v.float(), full_matrices=False)
            r = min(rank, len(S))
            truncated = (U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]).to(dw_v.dtype)
            cache[layer_idx] = truncated
        except Exception as e:
            logger.warning(f"[LTP] SVD truncation failed at layer {layer_idx}: {e}")
            continue
    return cache
