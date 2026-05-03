"""PromptResult: flat per-prompt data model matching TASM's spec.

Every field is a direct attribute. `result.stress_score` is a float.
`result.signed_attr` is a numpy array. No nesting, no field_specs,
no scalars/per_token/objects decomposition.

`result_to_dict()` serializes to the JSON shape the frontend expects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np


@dataclass
class LTPResult:
    """Per-prompt Lateral Tension Profile results."""
    profiles: List[np.ndarray] = field(default_factory=list)
    base_profiles: List[np.ndarray] = field(default_factory=list)
    tension_points: List[np.ndarray] = field(default_factory=list)
    tension_magnitudes: List[float] = field(default_factory=list)
    profile_shapes: List[str] = field(default_factory=list)
    counterfactual_tokens: List[List[Tuple[str, float]]] = field(default_factory=list)
    offset_magnitude: Dict[int, float] = field(default_factory=dict)
    offset_variance: Dict[int, float] = field(default_factory=dict)
    lateral_coverage: Dict[int, float] = field(default_factory=dict)
    mean_M: float = 0.0
    mean_V: float = 0.0
    mean_L: float = 0.0
    max_prc: float = 0.0
    n_directional: int = 0
    prc_per_token: List[float] = field(default_factory=list)
    semantic_trajectory: Optional[np.ndarray] = None
    tension_trajectory: Optional[np.ndarray] = None
    layer_strategy: str = "signal"
    monitored_layers: List[int] = field(default_factory=list)
    k: int = 8
    svd_rank: int = 0


@dataclass
class SFDResult:
    """Per-prompt Spectral Field Density results."""
    per_token_density: List[float] = field(default_factory=list)
    density_mean: float = 0.0
    density_max: float = 0.0
    density_var: float = 0.0
    density_p90: float = 0.0
    global_erank: float = 0.0
    n_layers_used: int = 0

    def to_dict(self) -> dict:
        return {
            "per_token_density": self.per_token_density,
            "density_mean": self.density_mean,
            "density_max": self.density_max,
            "density_var": self.density_var,
            "density_p90": self.density_p90,
            "global_erank": self.global_erank,
            "n_layers_used": self.n_layers_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SFDResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class PromptResult:
    """Complete analysis result for a single prompt.

    Flat structure. Every consumer — frontend, analysis modules, export —
    reads directly from these fields. This IS the spec.
    """
    prompt: str = ""
    category: str = ""
    tokens: list = field(default_factory=list)
    seq_len: int = 0

    # Amplitude trajectory (all sublayers)
    amplitude_trajectory: list = field(default_factory=list)
    amplitude_normalized: list = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None

    # Stress score (signal layers)
    stress_score: float = 0.0
    per_token_stress: Optional[np.ndarray] = None

    # Signed attribution
    signed_attr: Optional[np.ndarray] = None
    net_correction: float = 0.0
    n_negative_tokens: int = 0
    has_negative_tokens: bool = False
    per_layer_signed_attr: dict = field(default_factory=dict)

    # Distribution metrics
    entropy: float = 0.0
    top2_share: float = 0.0
    middle_share: float = 0.0
    interior_cv: float = 0.0

    # Behavioral divergence
    kl_divergence: Optional[float] = None
    per_token_kl: Optional[np.ndarray] = None

    # Model responses
    instruct_topk: list = field(default_factory=list)
    base_topk: list = field(default_factory=list)
    base_counterfactual_tokens: list = field(default_factory=list)

    # Proof-1 exactness
    proof1_checks: list = field(default_factory=list)

    # Signal layer breakdown
    signal_layer_indices: list = field(default_factory=list)
    per_layer_amplitude: dict = field(default_factory=dict)

    # Full-capture derived metrics
    per_token_coherence: Optional[np.ndarray] = None
    per_token_spectral_rank: Optional[list] = None
    attn_frac: Optional[np.ndarray] = None
    token_similarity: Optional[np.ndarray] = None
    full_capture_enabled: bool = False

    # Model-intrinsic normalization
    delta_scale: float = 0.0
    spectral_summary: dict = field(default_factory=dict)

    # LTP
    ltp: Optional[LTPResult] = None

    # SFD
    sfd: Optional[SFDResult] = None

    # Rank displacement
    rank_displacement: Optional[dict] = None

    # Domain embeddings
    domain_embedding: Optional[list] = None
    per_token_domain_emb: Optional[list] = None
    per_token_escalation_emb: Optional[list] = None
    per_token_final_emb: Optional[list] = None
    per_token_domain_offset: int = 0


def ltp_result_to_dict(ltp: LTPResult) -> dict:
    """Serialize an LTPResult for JSON transport."""
    d = {
        "mean_M": ltp.mean_M,
        "mean_V": ltp.mean_V,
        "mean_L": ltp.mean_L,
        "max_prc": ltp.max_prc,
        "n_directional": ltp.n_directional,
        "n_layers_used": len(ltp.monitored_layers),
        "tension_magnitudes": ltp.tension_magnitudes,
        "prc_per_token": ltp.prc_per_token,
        "offset_magnitude": {str(k): v for k, v in ltp.offset_magnitude.items()},
        "offset_variance": {str(k): v for k, v in ltp.offset_variance.items()},
        "lateral_coverage": {str(k): v for k, v in ltp.lateral_coverage.items()},
        "profiles": [p.tolist() if isinstance(p, np.ndarray) else p
                     for p in ltp.profiles],
        "base_profiles": [p.tolist() if isinstance(p, np.ndarray) else p
                          for p in ltp.base_profiles],
        "profile_shapes": ltp.profile_shapes,
        "counterfactual_tokens": ltp.counterfactual_tokens,
        "layer_strategy": ltp.layer_strategy,
        "k": ltp.k,
    }
    if ltp.semantic_trajectory is not None:
        d["semantic_trajectory_2d"] = ltp.semantic_trajectory.tolist()
    else:
        d["semantic_trajectory_2d"] = []
    if ltp.tension_trajectory is not None:
        d["tension_trajectory_2d"] = ltp.tension_trajectory.tolist()
    else:
        d["tension_trajectory_2d"] = []
    return d


def result_to_dict(r: PromptResult) -> dict:
    """Serialize a PromptResult for JSON transport.

    Output matches the exact field names and shapes that main.js reads.
    This is the single serialization point — no tasm_compat layer.
    """
    d = {
        "prompt": r.prompt,
        "category": r.category,
        "tokens": r.tokens,
        "seq_len": r.seq_len,
        "stress_score": _native(r.stress_score),
        "per_token_stress": r.per_token_stress.tolist() if r.per_token_stress is not None else [],
        "signed_attr": r.signed_attr.tolist() if r.signed_attr is not None else [],
        "net_correction": _native(r.net_correction),
        "n_negative_tokens": _native(r.n_negative_tokens),
        "has_negative_tokens": r.has_negative_tokens,
        "entropy": _native(r.entropy),
        "top2_share": _native(r.top2_share),
        "middle_share": _native(r.middle_share),
        "interior_cv": _native(r.interior_cv),
        "kl_divergence": _native(r.kl_divergence),
        "per_token_kl": _native(r.per_token_kl) if r.per_token_kl is not None else None,
        "instruct_topk": r.instruct_topk,
        "base_topk": r.base_topk,
        "base_counterfactual_tokens": r.base_counterfactual_tokens,
        "proof1_checks": r.proof1_checks,
        "per_layer_signed_attr": {str(k): v for k, v in r.per_layer_signed_attr.items()},
        "amplitude_trajectory": [_native(v) for v in r.amplitude_trajectory],
        "amplitude_normalized": [_native(v) for v in r.amplitude_normalized],
        "heatmap": r.heatmap.tolist() if r.heatmap is not None else [],
        "signal_layer_indices": r.signal_layer_indices,
        "per_layer_amplitude": {str(k): _native(v) for k, v in r.per_layer_amplitude.items()},
        "full_capture_enabled": r.full_capture_enabled,
        "delta_scale": _native(r.delta_scale),
        "spectral_summary": r.spectral_summary,
    }

    # Full-capture derived metrics
    if r.full_capture_enabled:
        d["per_token_coherence"] = r.per_token_coherence.tolist() if r.per_token_coherence is not None else []
        d["per_token_spectral_rank"] = r.per_token_spectral_rank or []
        d["attn_frac"] = r.attn_frac.tolist() if r.attn_frac is not None else []
        d["token_similarity"] = r.token_similarity.tolist() if r.token_similarity is not None else []

    # LTP
    d["ltp"] = ltp_result_to_dict(r.ltp) if r.ltp is not None else None

    # SFD
    d["sfd"] = r.sfd.to_dict() if r.sfd is not None else None

    # Rank displacement
    d["rank_displacement"] = r.rank_displacement

    # Domain embeddings
    d["domain_embedding"] = r.domain_embedding
    d["per_token_domain_emb"] = r.per_token_domain_emb
    d["per_token_escalation_emb"] = r.per_token_escalation_emb
    d["per_token_final_emb"] = r.per_token_final_emb
    d["per_token_domain_offset"] = r.per_token_domain_offset

    return _sanitize(d)


# ── Serialization helpers ────────────────────────────────────────

def _native(v):
    """Coerce numpy/torch scalars to Python natives; NaN/Inf → None."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, np.float32, np.float64)):
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, np.ndarray):
        return [_native(x) for x in v.tolist()]
    if hasattr(v, "item"):
        val = v.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return v


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None. Final pass before JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj
