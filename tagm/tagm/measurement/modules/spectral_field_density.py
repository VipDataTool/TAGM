"""SpectralFieldDensity (SFD): per-token subspace engagement.

Measures how many dimensions of the QK correction subspace each token
activation engages — the dimensionality axis alongside intensity (stress)
and selectivity (LTP).

Precompute once per (model pair, layer set, k): SVD of [ΔW_Q ; ΔW_K] per
layer, cached as right singular vectors V_k and singular values S.

Per token: project h through V_k, scale by S, compute the
normalized-energy Shannon entropy, and report density = exp(H_t) / erank_global.

Translated from TASM's `engine/sfd.py`.

Capture expectation:
  Needs pre_attn_norm hidden states at one or more layers. User's
  CaptureConfig picks which layers; this measurement aggregates over
  layers that are captured AND have both q and k deltas, narrowed by
  the optional `layers` scope parameter.
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


# ── SFD layer cache (in-memory, per-pipeline) ───────────────────────

class SFDLayerCache:
    """SVD factors for one layer's concatenated QK delta."""
    def __init__(self, V_k: np.ndarray, S: np.ndarray, erank: float,
                 norm_entropy: float, frob_norm: float):
        self.V_k = V_k
        self.S = S
        self.erank = erank
        self.norm_entropy = norm_entropy
        self.frob_norm = frob_norm


class SFDPairCache:
    """Per-model-pair SFD cache: one layer cache per monitored layer."""
    def __init__(self, k: int):
        self.k = k
        self.layers: dict[int, SFDLayerCache] = {}

    @property
    def mean_erank(self) -> float:
        return float(np.mean([lc.erank for lc in self.layers.values()])) \
            if self.layers else 0.0


# Module-scoped cache keyed by (instruct_id, base_id, dtype, sorted-layer-tuple, k).
_SFD_CACHES: dict[tuple, SFDPairCache] = {}


def _cache_key(delta_store, layers, k):
    m = delta_store.metadata
    return (m.instruct_model_id, m.base_model_id, m.dtype,
            tuple(sorted(layers)), k)


def _get_or_build_cache(delta_store, layers, k, svd_seed=42):
    key = _cache_key(delta_store, layers, k)
    if key in _SFD_CACHES:
        return _SFD_CACHES[key]

    cache = SFDPairCache(k=k)
    torch.manual_seed(svd_seed)

    for layer_idx in layers:
        dw_q = delta_store.get_or_none(layer_idx, "q")
        dw_k = delta_store.get_or_none(layer_idx, "k")
        if dw_q is None or dw_k is None:
            logger.warning(f"[SFD] Layer {layer_idx}: missing Q or K delta; skipping")
            continue

        try:
            dw_qk = torch.cat([dw_q.float().cpu(), dw_k.float().cpu()], dim=0)
            actual_k = min(k, min(dw_qk.shape))
            U, S, V = torch.svd_lowrank(dw_qk, q=actual_k)
            s = S.float().numpy().astype(np.float64)
            v_k = V.float().numpy().astype(np.float32).T

            s_pos = s[s > 1e-10]
            if len(s_pos) == 0:
                continue
            p = s_pos / s_pos.sum()
            H = float(-np.sum(p * np.log(p)))
            erank = float(np.exp(H))
            norm_H = H / np.log(len(s_pos)) if len(s_pos) > 1 else 0.0
            frob = float(np.sqrt(np.sum(s ** 2)))

            cache.layers[layer_idx] = SFDLayerCache(
                V_k=v_k, S=s_pos.astype(np.float32),
                erank=erank, norm_entropy=float(norm_H), frob_norm=frob,
            )
        except Exception as e:
            logger.warning(f"[SFD] Layer {layer_idx} SVD failed: {e}")
            continue

    _SFD_CACHES[key] = cache
    logger.info(f"[SFD] Built cache: {len(cache.layers)} layers, "
                f"k={k}, mean_erank={cache.mean_erank:.2f}")
    return cache


def clear_sfd_caches() -> None:
    """Release all SFD caches. Called by the service layer on model unload."""
    _SFD_CACHES.clear()


def _compute_token(h_vec: np.ndarray, lc: SFDLayerCache) -> tuple[float, float, float]:
    """Return (energy, spectral_entropy, density) for one token at one layer."""
    c = lc.V_k @ h_vec
    w = lc.S[:len(c)] * c
    w2 = w * w
    energy = float(w2.sum())
    if energy < 1e-20:
        return energy, 0.0, 0.0
    q = w2 / energy
    q_pos = q[q > 1e-10]
    H_t = float(-np.sum(q_pos * np.log(q_pos)))
    erank_t = float(np.exp(H_t))
    density = erank_t / lc.erank if lc.erank > 0 else 0.0
    return energy, H_t, density


@register_measurement
class SpectralFieldDensity(MeasurementModule):
    name = "spectral_field_density"
    display_name = "Spectral Field Density"
    description = (
        "Per-token engagement with the QK correction subspace: fraction of "
        "the global effective rank each token activates. Dimensionality axis "
        "of the measurement triple."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="layers",
            display_name="Scope layers",
            description=(
                "Optional subset of captured layers to aggregate over. "
                "Empty means all captured layers that also have q and k deltas. "
                "Scope parameter; not a capture parameter."
            ),
            kind="layer_list", default=[],
        ),
        ModuleParameter(
            name="svd_k",
            display_name="SVD truncation rank (k)",
            description=("Number of right singular vectors retained for "
                         "per-token projection."),
            kind="int", default=16, min_value=1, max_value=256,
            engine_config_key="sfd_svd_k",
        ),
        ModuleParameter(
            name="svd_seed",
            display_name="SVD random seed",
            description="Deterministic seed for torch.svd_lowrank.",
            kind="int", default=42, advanced=True,
            engine_config_key="sfd_svd_seed",
        ),
    ]

    def capture_expectation(self, params):
        return CaptureExpectation(
            hook_points_required=("pre_attn_norm",),
            capture_types_required=frozenset({"hidden"}),
            min_layers_captured=1,
        )

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        scope = list(params.get("layers") or [])
        svd_k = int(params.get("svd_k", 16))
        svd_seed = int(params.get("svd_seed", 42))

        seq_len = run_result.seq_len
        store = run_result.activations

        layers = resolve_scope_layers(
            activation_store=store,
            hook_point="pre_attn_norm",
            capture_type="hidden",
            scope=scope,
            required_delta_roles=("q", "k"),
            delta_store=delta_store,
        )

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={
                "layers_requested": list(scope),
                "layers_used": list(layers),
                "svd_k": svd_k,
                "svd_seed": svd_seed,
                "scope_resolution": describe_scope_resolution(
                    scope, layers, "pre_attn_norm"),
            },
        )

        if not layers:
            result.per_token["density"] = padded_per_token(seq_len)
            result.scalars["density_mean"] = float("nan")
            self._annotate(result)
            return result

        cache = _get_or_build_cache(delta_store, layers, svd_k, svd_seed)
        if not cache.layers:
            result.per_token["density"] = padded_per_token(seq_len)
            result.scalars["density_mean"] = float("nan")
            self._annotate(result)
            return result

        accum_density = np.zeros(seq_len)
        n_layers_used = 0

        for layer_idx, lc in cache.layers.items():
            if not store.has(layer_idx, "pre_attn_norm", "hidden"):
                continue
            h = store.get(layer_idx, "pre_attn_norm", "hidden")
            h_np = h[0, :seq_len].float().cpu().numpy()

            for t in range(seq_len):
                _, _, d = _compute_token(h_np[t], lc)
                accum_density[t] += d
            n_layers_used += 1

        if n_layers_used == 0:
            result.per_token["density"] = padded_per_token(seq_len)
            self._annotate(result)
            return result

        per_density = accum_density / n_layers_used
        result.per_token["density"] = per_density.astype(float)
        result.scalars["density_mean"] = float(np.mean(per_density))
        result.scalars["density_max"] = float(np.max(per_density))
        result.scalars["density_var"] = float(np.var(per_density))
        result.scalars["density_p90"] = float(np.percentile(per_density, 90))
        result.scalars["global_erank"] = float(cache.mean_erank)
        result.scalars["n_layers_used"] = n_layers_used

        self._annotate(result)
        return result

    def _annotate(self, result: MeasurementResult) -> None:
        result.field_specs["density"] = FieldSpec(
            name="density", kind="per_token",
            description=("Per-token effective rank / global effective rank. "
                          "Large values mean the token's activation engages "
                          "many QK correction dimensions."),
            units="ratio", length_invariant=False,
        )
        for name, desc in (
            ("density_mean", "Mean per-token density across the prompt."),
            ("density_max", "Max per-token density in the prompt."),
            ("density_var", "Variance of per-token density."),
            ("density_p90", "90th percentile of per-token density."),
            ("global_erank", "Mean effective rank across used layers."),
            ("n_layers_used", "Count of layers actually aggregated."),
        ):
            result.field_specs[name] = FieldSpec(
                name=name, kind="scalar", description=desc,
                length_invariant=True,
            )
