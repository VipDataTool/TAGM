"""TASM-compat result shaping.

The TASM-derived frontend (`static/js/main.js`) expects per-prompt
results in the flat shape TASM produced. TAGM's native shape is
nested: `prompt.measurements.<name>.scalars.<field>`. This module
converts the TAGM PromptRecord dict into the flat dict main.js
parses.

Only fields actually consumed by main.js are mapped. New TAGM-only
data (full per-layer breakdowns, scope_resolution metadata, etc.)
is preserved under `_tagm_native` so it's still inspectable but
doesn't pollute the flat namespace.

The mapping is mechanical and deterministic. If the result lacks a
measurement (because that measurement wasn't selected for this run),
the corresponding flat fields are simply absent — main.js handles
missing fields gracefully via `r.x ? ... : ...` patterns throughout.
"""
from __future__ import annotations

from typing import Any, Optional


def prompt_record_to_tasm_shape(prec: dict) -> dict:
    """Convert a TAGM PromptRecord dict into TASM's flat result shape.

    Input is `PromptRecord.to_dict()` output:
        {
            "prompt": str,
            "category": str,
            "tokens": [str, ...],
            "seq_len": int,
            "measurements": {
                "stress_score": {"scalars": {...}, "per_token": {...}, ...},
                "last_position_attribution": {...},
                "lateral_tension_profile": {...},
                ...
            },
            "metadata": {...},
        }

    Output is the flat dict shape main.js expects.
    """
    measurements = prec.get("measurements") or {}

    out: dict[str, Any] = {
        "prompt": prec.get("prompt", ""),
        "category": prec.get("category", ""),
        "tokens": list(prec.get("tokens") or []),
        "seq_len": prec.get("seq_len", 0),
    }

    # ── stress_score ────────────────────────────────────────────────
    ss = measurements.get("stress_score") or {}
    ss_scalars = ss.get("scalars") or {}
    ss_per_tok = ss.get("per_token") or {}
    if ss:
        out["stress_score"] = ss_scalars.get("stress_mean")
        out["per_token_stress"] = ss_per_tok.get("stress") or []

    # ── last_position_attribution (TASM called it signed attribution) ──
    lpa = measurements.get("last_position_attribution") or {}
    lpa_scalars = lpa.get("scalars") or {}
    lpa_per_tok = lpa.get("per_token") or {}
    lpa_objects = lpa.get("objects") or {}
    if lpa:
        out["signed_attr"] = lpa_per_tok.get("signed_attribution_to_last") or []
        out["net_correction"] = lpa_scalars.get("net_correction_to_last")
        out["entropy"] = lpa_scalars.get("entropy")
        out["top2_share"] = lpa_scalars.get("top2_share")
        out["middle_share"] = lpa_scalars.get("middle_share")
        out["interior_cv"] = lpa_scalars.get("interior_cv")
        out["n_negative_tokens"] = lpa_scalars.get("n_negative_tokens")
        out["has_negative_tokens"] = lpa_scalars.get("has_negative_tokens")
        out["proof1_checks"] = lpa_objects.get("proof1_checks") or []

    # ── amplitude_trajectory ────────────────────────────────────────
    at = measurements.get("amplitude_trajectory") or {}
    at_scalars = at.get("scalars") or {}
    at_objects = at.get("objects") or {}
    if at:
        out["amplitude_trajectory"] = {
            "raw": at_objects.get("amplitude_raw") or [],
            "normalized": at_objects.get("amplitude_normalized") or [],
            "heatmap": at_objects.get("heatmap") or [],
            "heatmap_shape": at_objects.get("heatmap_shape") or [0, 0],
            "sublayer_labels": at_objects.get("sublayer_labels") or [],
            "mean_raw": at_scalars.get("trajectory_mean_raw"),
            "mean_normalized": at_scalars.get("trajectory_mean_normalized"),
        }

    # ── amplitude_derived_metrics ───────────────────────────────────
    adm = measurements.get("amplitude_derived_metrics") or {}
    adm_per_tok = adm.get("per_token") or {}
    adm_objects = adm.get("objects") or {}
    if adm:
        out["per_token_attn_frac"] = adm_per_tok.get("attn_frac") or []
        out["per_token_coherence"] = adm_per_tok.get("coherence") or []
        out["per_token_sublayer_rank"] = adm_per_tok.get("sublayer_rank") or []
        out["token_similarity"] = adm_objects.get("token_similarity") or []

    # ── lateral_tension_profile ─────────────────────────────────────
    ltp = measurements.get("lateral_tension_profile") or {}
    ltp_scalars = ltp.get("scalars") or {}
    ltp_per_tok = ltp.get("per_token") or {}
    ltp_per_layer = ltp.get("per_layer") or {}
    ltp_objects = ltp.get("objects") or {}
    if ltp:
        out["ltp"] = {
            "mean_M": ltp_scalars.get("mean_M"),
            "mean_V": ltp_scalars.get("mean_V"),
            "mean_L": ltp_scalars.get("mean_L"),
            "max_prc": ltp_scalars.get("max_prc"),
            "n_directional": ltp_scalars.get("n_directional"),
            "n_layers_used": ltp_scalars.get("n_layers_used"),
            "tension_magnitudes": ltp_per_tok.get("tension_magnitude") or [],
            "prc_per_token": ltp_per_tok.get("prc") or [],
            "offset_magnitude": ltp_per_layer.get("offset_magnitude") or {},
            "offset_variance": ltp_per_layer.get("offset_variance") or {},
            "lateral_coverage": ltp_per_layer.get("lateral_coverage") or {},
            "profiles": ltp_objects.get("profiles") or [],
            "base_profiles": ltp_objects.get("base_profiles") or [],
            "profile_shapes": ltp_objects.get("profile_shapes") or [],
            "counterfactual_tokens": ltp_objects.get("counterfactual_tokens") or [],
            "semantic_trajectory_2d": ltp_objects.get("semantic_trajectory_2d") or [],
            "tension_trajectory_2d": ltp_objects.get("tension_trajectory_2d") or [],
        }

    # ── spectral_field_density ──────────────────────────────────────
    sfd = measurements.get("spectral_field_density") or {}
    sfd_scalars = sfd.get("scalars") or {}
    sfd_per_tok = sfd.get("per_token") or {}
    if sfd:
        out["sfd"] = {
            "density_mean": sfd_scalars.get("density_mean"),
            "density_max": sfd_scalars.get("density_max"),
            "density_var": sfd_scalars.get("density_var"),
            "density_p90": sfd_scalars.get("density_p90"),
            "global_erank": sfd_scalars.get("global_erank"),
            "n_layers_used": sfd_scalars.get("n_layers_used"),
            "per_token_density": sfd_per_tok.get("density") or [],
        }

    # ── rank_displacement ───────────────────────────────────────────
    rd = measurements.get("rank_displacement") or {}
    rd_scalars = rd.get("scalars") or {}
    rd_per_tok = rd.get("per_token") or {}
    rd_objects = rd.get("objects") or {}
    if rd:
        out["rank_displacement"] = {
            "mean_disp_per_token": rd_scalars.get("mean_disp_per_token"),
            "mean_replacement": rd_scalars.get("mean_replacement"),
            "mean_tau": rd_scalars.get("mean_tau"),
            "mean_overlap": rd_scalars.get("mean_overlap"),
            "total_displacement": rd_scalars.get("total_displacement"),
            "n_positions": rd_scalars.get("n_positions"),
            "per_position": rd_objects.get("per_position") or [],
            "per_token_disp": rd_per_tok.get("total_disp") or [],
            "per_token_replacement": rd_per_tok.get("replacement_ratio") or [],
            "instruct_disp_profiles": rd_objects.get("instruct_disp_profiles") or [],
            "base_disp_profiles": rd_objects.get("base_disp_profiles") or [],
            "per_position_tau": rd_objects.get("per_position_tau") or [],
            "per_position_overlap": rd_objects.get("per_position_overlap") or [],
        }

    # ── probe_projection ────────────────────────────────────────────
    pp = measurements.get("probe_projection") or {}
    pp_per_tok = pp.get("per_token") or {}
    pp_objects = pp.get("objects") or {}
    if pp:
        out["probe_projection"] = {
            "best_class_idx": pp_per_tok.get("best_class_idx") or [],
            "best_score": pp_per_tok.get("best_score") or [],
            "score_matrix": pp_objects.get("score_matrix") or [],
            "probe_labels": pp_objects.get("probe_labels") or [],
            "per_token_assignment": pp_objects.get("per_token_assignment") or [],
        }

    # ── per_token_embedding ─────────────────────────────────────────
    pte = measurements.get("per_token_embedding") or {}
    pte_objects = pte.get("objects") or {}
    if pte:
        out["per_token_embeddings"] = pte_objects.get("per_token_embeddings") or {}

    # ── backscatter_projection ──────────────────────────────────────
    bp = measurements.get("backscatter_projection") or {}
    bp_objects = bp.get("objects") or {}
    bp_scalars = bp.get("scalars") or {}
    if bp:
        out["backscatter"] = {
            "magnitude_matrix": bp_objects.get("magnitude_matrix") or [],
            "probe_labels": bp_objects.get("probe_labels") or [],
            "sublayer_labels": bp_objects.get("sublayer_labels") or [],
            "n_probes": bp_scalars.get("n_probes"),
            "n_sublayers": bp_scalars.get("n_sublayers"),
            "mean_magnitude": bp_scalars.get("mean_magnitude"),
        }

    # ── KL divergence (lifted from base_cache, when present) ───────
    # TAGM doesn't have a KLDivergence measurement yet; main.js reads
    # `r.kl_divergence` (scalar) and `r.per_token_kl` (array). When
    # the measurement exists, it'll populate these. For now, both are
    # absent and main.js handles that gracefully via null checks.

    # ── base_counterfactual_tokens (used by directional-flow viz) ──
    # Lifted from RD/LTP base_cache when available. Not currently
    # surfaced separately on the prompt record; main.js also uses
    # ltp.counterfactual_tokens which IS present above.

    # ── Native TAGM shape preserved for inspection ──────────────────
    out["_tagm_native"] = {
        "measurements": measurements,
        "metadata": prec.get("metadata") or {},
    }

    return out
