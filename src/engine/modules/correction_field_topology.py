"""
Correction Field Topology Module for TAGM.

Produces a uniform terrain payload for the 3D counterfactual-plane
visualization rendered client-side in Three.js.

─── Architecture: Channel Mapping ────────────────────────────────
The renderer exposes multiple independent visual channels. Each
channel binds to a named measure from the session data. The module
extracts ALL available per-token measures and ships them in a
`channels` dict so the renderer can bind any measure to any
compatible visual encoding without a server round-trip.

Visual channels and their data requirements:
  - height:      per-rank arrays [n_tokens][k] → bank-decomposed
  - spine:       per-token scalar [n_tokens]   → primary signal
  - brightness:  per-token scalar [n_tokens]   → multiplicative modulation
  - bar_length:  per-token scalar [n_tokens]   → [0,1] normalized
  - bar_color:   per-token categorical or scalar
  - font_color:  per-rank categorical [n_tokens][k] → status tags
  - filter:      per-token scalar [n_tokens]   → threshold mask

─── Normalization: The Shoebox ───────────────────────────────────
All per-rank height values are logistic-normalized into a fixed
vertical volume defined by the lattice geometry. The ceiling is
proportional to the bank depth (k × column spacing), making the
terrain volume a bounded "shoebox" regardless of which measure
drives height. The logistic is parameterized per-prompt from the
distribution's min/max so that every prompt fills the same visual
envelope while preserving relative within-prompt contrast.

Per-token scalars (brightness, bars) are min-max normalized to
[0, 1] per-prompt for the same reason: visual consistency across
measures with different native scales.

─── Payload contract ─────────────────────────────────────────────

    {
        "geometry": {                          # lattice constants
            "k": 8,                            # ranks per bank
            "col_spacing": 0.55,               # SCX
            "ceiling_ratio": 1.0,              # ceiling = ratio × k × SCX
        },
        "height_measure": "<measure_name>",    # which bank measure drives height
        "accent_available": bool,
        "available_measures": {
            "bank": [<name>, ...],             # bank-decomposable (height-eligible)
            "scalar": [<name>, ...],           # per-token scalars (bar/brightness)
        },
        "channel_config": {                    # current bindings
            "height": "<bank_measure>",
            "spine": "<bank_measure>",         # same as height (primary)
            "brightness": "<scalar_measure>",
            "bar_length": "<scalar_measure>",
            "bar_color": "status" | "<scalar_measure>",
            "filter": "<scalar_measure>" | "none",
        },
        "prompts": [
            {
                "prompt": str,
                "category": str,
                "tokens": [str, ...],
                "banks": {                     # bank-decomposed measures
                    "<measure>": {
                        "primary": [float, ...],           # per-token aggregate
                        "instruct_bank": [[float,...], ...],# [n_tok][k]
                        "base_bank": [[float,...], ...],
                    },
                    ...
                },
                "scalars": {                   # per-token scalar measures
                    "kl_divergence": [float, ...],
                    "stress": [float, ...],
                    "signed_attr": [float, ...],
                    "sfd_density": [float, ...],
                    ...
                },
                "status": {                    # categorical per-rank data
                    "instruct_status": [[str,...], ...],
                    "base_status": [[str,...], ...],
                },
                "labels": {                    # token text for each bank position
                    "counterfactual_tokens": [...],
                    "base_counterfactual_tokens": [...],
                },
            },
            ...
        ],
        "stats": {...},
        "launch_params": {...},
    }
"""

import logging
import math
import numpy as np
from collections import defaultdict

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")


# ─── Geometry constants ──────────────────────────────────────────
# These define the shoebox volume. They are echoed in the payload
# so the renderer doesn't have to hardcode them.

K = 8               # ranks per bank
COL_SPACING = 0.55   # SCX — lateral spacing between columns
CEILING_RATIO = 1.0  # ceiling = ratio × K × COL_SPACING


# ─── Bank measure registry ──────────────────────────────────────
# Measures that decompose into per-rank arrays and can drive height.

BANK_MEASURES = {
    "rank_displacement": {
        "display_name": "Rank Displacement",
        "accent_capable": True,
        "description": "Per-token probability displacement between base "
                       "and instruct top-k candidates.",
    },
    "ltp_tension": {
        "display_name": "LTP Tension",
        "accent_capable": False,
        "description": "Lateral tension magnitudes from counterfactual "
                       "refinement probing.",
    },
}


# ─── Scalar measure registry ────────────────────────────────────
# Per-token scalars that can drive brightness, bars, filter, etc.

SCALAR_MEASURES = {
    "kl_divergence": {
        "display_name": "KL Divergence",
        "description": "Per-token KL(instruct ∥ base) at each position.",
    },
    "stress": {
        "display_name": "ASM Stress",
        "description": "Per-token alignment stress magnitude.",
    },
    "signed_attr": {
        "display_name": "Signed Attribution",
        "description": "Per-token signed correction attribution. Can be negative.",
        "allow_negative": True,
    },
    "sfd_density": {
        "display_name": "SFD Density",
        "description": "Per-token spectral field density.",
    },
}


# ─── Availability detection ─────────────────────────────────────

def _has_rank_displacement(r):
    rd = r.get("rank_displacement") or {}
    return len(rd.get("instruct_disp_profiles") or []) > 0


def _has_ltp_tension(r):
    ltp = r.get("ltp") or {}
    return len(ltp.get("profiles") or []) > 0


def _has_kl(r):
    kl = r.get("per_token_kl")
    return kl is not None and len(kl) > 0


def _has_stress(r):
    s = r.get("per_token_stress")
    return s is not None and len(s) > 0


def _has_signed_attr(r):
    s = r.get("signed_attr")
    return s is not None and len(s) > 0


def _has_sfd(r):
    sfd = r.get("sfd") or {}
    d = sfd.get("per_token_density")
    return d is not None and len(d) > 0


BANK_CHECKS = {
    "rank_displacement": _has_rank_displacement,
    "ltp_tension": _has_ltp_tension,
}

SCALAR_CHECKS = {
    "kl_divergence": _has_kl,
    "stress": _has_stress,
    "signed_attr": _has_signed_attr,
    "sfd_density": _has_sfd,
}


def _detect_available(session_results):
    """Return dicts of available bank and scalar measures."""
    banks = [m for m, fn in BANK_CHECKS.items()
             if any(fn(r) for r in session_results)]
    scalars = [m for m, fn in SCALAR_CHECKS.items()
               if any(fn(r) for r in session_results)]
    return banks, scalars


# ─── Per-token scalar extractors ─────────────────────────────────
# Each returns a list of floats of length n_tok, or None if data
# is absent from this record.

def _extract_kl(r, n_tok):
    kl = r.get("per_token_kl")
    if kl is None or len(kl) == 0:
        return None
    return [float(kl[i]) if i < len(kl) else 0.0 for i in range(n_tok)]


def _extract_stress(r, n_tok):
    s = r.get("per_token_stress")
    if s is None or len(s) == 0:
        return None
    return [float(s[i]) if i < len(s) else 0.0 for i in range(n_tok)]


def _extract_signed_attr(r, n_tok):
    s = r.get("signed_attr")
    if s is None or len(s) == 0:
        return None
    return [float(s[i]) if i < len(s) else 0.0 for i in range(n_tok)]


def _extract_sfd(r, n_tok):
    sfd = r.get("sfd") or {}
    d = sfd.get("per_token_density")
    if d is None or len(d) == 0:
        return None
    return [float(d[i]) if i < len(d) else 0.0 for i in range(n_tok)]


SCALAR_EXTRACTORS = {
    "kl_divergence": _extract_kl,
    "stress": _extract_stress,
    "signed_attr": _extract_signed_attr,
    "sfd_density": _extract_sfd,
}


# ─── Bank extractors ────────────────────────────────────────────
# Each returns (primary, instruct_bank, base_bank) or None.

def _extract_rd_banks(r, n_tok):
    rd = r.get("rank_displacement") or {}
    i_profs = rd.get("instruct_disp_profiles") or []
    b_profs = rd.get("base_disp_profiles") or []
    per_pos = rd.get("per_position") or []
    if not i_profs:
        return None

    n = min(n_tok, len(i_profs))
    primary = [float(per_pos[i].get("total_disp", 0.0))
               if i < len(per_pos) else 0.0
               for i in range(n)]
    i_bank = [list(i_profs[i]) for i in range(n)]
    b_bank = [list(b_profs[i]) if i < len(b_profs) else [0.0] * K
              for i in range(n)]
    return primary, i_bank, b_bank


def _extract_ltp_banks(r, n_tok):
    ltp = r.get("ltp") or {}
    profs = ltp.get("profiles") or []
    base_profs = ltp.get("base_profiles") or []
    tensions = ltp.get("tension_magnitudes") or []
    if not profs:
        return None

    n = min(n_tok, len(profs))
    primary = [float(tensions[i]) if i < len(tensions) else 0.0
               for i in range(n)]
    i_bank = [list(profs[i]) for i in range(n)]
    b_bank = [list(base_profs[i]) if i < len(base_profs) else [0.0] * K
              for i in range(n)]
    return primary, i_bank, b_bank


BANK_EXTRACTORS = {
    "rank_displacement": _extract_rd_banks,
    "ltp_tension": _extract_ltp_banks,
}


# ─── Status / label extractors ──────────────────────────────────

def _cf_token_name(entry):
    if isinstance(entry, (list, tuple)) and entry:
        return str(entry[0])
    if entry is None:
        return ""
    return str(entry)


def _classify_banks(cf_inst, cf_base, k=K):
    inst_names = [_cf_token_name(e) for e in (cf_inst or [])]
    base_names = [_cf_token_name(e) for e in (cf_base or [])]
    inst_set = set(n for n in inst_names if n)
    base_set = set(n for n in base_names if n)

    i_status = []
    for j in range(k):
        if j < len(inst_names) and inst_names[j]:
            i_status.append("matched" if inst_names[j] in base_set else "promoted")
        else:
            i_status.append("matched")
    b_status = []
    for j in range(k):
        if j < len(base_names) and base_names[j]:
            b_status.append("matched" if base_names[j] in inst_set else "demoted")
        else:
            b_status.append("matched")
    return i_status, b_status


def _extract_status(r, n_tok):
    """Extract promoted/demoted status arrays if counterfactual data exists."""
    ltp = r.get("ltp") or {}
    cf_full = ltp.get("counterfactual_tokens") or []
    base_cf_full = r.get("base_counterfactual_tokens") or []
    if not cf_full and not base_cf_full:
        return [], []

    i_status_all, b_status_all = [], []
    for i in range(n_tok):
        cf_i = cf_full[i] if i < len(cf_full) else []
        bcf_i = base_cf_full[i] if i < len(base_cf_full) else []
        i_st, b_st = _classify_banks(cf_i, bcf_i)
        i_status_all.append(i_st)
        b_status_all.append(b_st)
    return i_status_all, b_status_all


def _extract_labels(r, n_tok):
    """Extract counterfactual token labels for both banks."""
    ltp = r.get("ltp") or {}
    cf = (ltp.get("counterfactual_tokens") or [])[:n_tok]
    base_cf = (r.get("base_counterfactual_tokens") or [])[:n_tok]
    return (
        [list(c) if c else [] for c in cf],
        [list(c) if c else [] for c in base_cf],
    )


# ─── Prompt builder ─────────────────────────────────────────────
# Builds a unified prompt payload with all available measures.

def _build_prompt(r, token_limit, available_banks, available_scalars):
    """Build a single prompt payload with all available measures.

    Returns None if the record has no data for any bank measure
    (we need at least one bank measure for the lattice geometry).
    """
    tokens = (r.get("tokens") or [])[:token_limit]
    if not tokens:
        return None

    # Determine effective n_tok from the height measure's bank data.
    # Try each available bank measure to find one with data.
    n_tok = len(tokens)
    has_any_bank = False

    banks = {}
    for bm in available_banks:
        extractor = BANK_EXTRACTORS.get(bm)
        if not extractor:
            continue
        result = extractor(r, n_tok)
        if result is None:
            continue
        primary, i_bank, b_bank = result
        actual_n = min(n_tok, len(i_bank))
        banks[bm] = {
            "primary": primary[:actual_n],
            "instruct_bank": i_bank[:actual_n],
            "base_bank": b_bank[:actual_n],
        }
        has_any_bank = True
        # Constrain n_tok to the shortest bank measure
        n_tok = min(n_tok, actual_n)

    if not has_any_bank:
        return None

    tokens = tokens[:n_tok]

    # Extract all available scalars
    scalars = {}
    for sm in available_scalars:
        extractor = SCALAR_EXTRACTORS.get(sm)
        if not extractor:
            continue
        result = extractor(r, n_tok)
        if result is not None:
            scalars[sm] = result

    # Status and labels
    i_status, b_status = _extract_status(r, n_tok)
    cf_tokens, base_cf_tokens = _extract_labels(r, n_tok)

    return {
        "prompt": r.get("prompt", ""),
        "category": r.get("category", "unknown"),
        "tokens": tokens,
        "banks": banks,
        "scalars": scalars,
        "status": {
            "instruct_status": i_status,
            "base_status": b_status,
        },
        "labels": {
            "counterfactual_tokens": cf_tokens,
            "base_counterfactual_tokens": base_cf_tokens,
        },
    }


# ─── Summary statistics ─────────────────────────────────────────

def _compute_summary_stats(prompt_payloads, height_measure):
    stats = {
        "n_prompts": len(prompt_payloads),
        "height_measure": height_measure,
        "by_category": {},
        "token_stats": {
            "total_tokens": 0,
            "mean_tokens_per_prompt": 0,
            "max_tokens": 0,
        },
    }

    if not prompt_payloads:
        return stats

    token_counts = []
    for p in prompt_payloads:
        cat = p.get("category") or "unknown"
        n_tok = len(p.get("tokens") or [])
        token_counts.append(n_tok)

        if cat not in stats["by_category"]:
            stats["by_category"][cat] = {"count": 0}
        stats["by_category"][cat]["count"] += 1

    stats["token_stats"]["total_tokens"] = sum(token_counts)
    stats["token_stats"]["mean_tokens_per_prompt"] = round(
        float(np.mean(token_counts)), 1) if token_counts else 0
    stats["token_stats"]["max_tokens"] = max(token_counts) if token_counts else 0

    return stats


# ─── Module ──────────────────────────────────────────────────────

class CorrectionFieldTopologyModule(TASMModule):
    name = "correction_field_topology"
    display_name = "Correction Field Topology"
    description = (
        "3D terrain visualization with configurable channel mapping. "
        "Bank-decomposed measures (rank displacement, LTP tension) drive "
        "the lattice elevation. Per-token scalars (KL, stress, SFD, "
        "signed attribution) can independently drive brightness, bar "
        "length, and data filtering."
    )
    version = "3.0.0"

    min_results = 1
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        # ── Height source ──
        ModuleParameter(
            name="height_measure",
            display_name="Height Measure",
            description="Which bank-decomposed measure drives terrain elevation.",
            type="select",
            default="rank_displacement",
            options=list(BANK_MEASURES.keys()),
        ),

        # ── Channel bindings ──
        ModuleParameter(
            name="brightness_channel",
            display_name="Brightness Channel",
            description="Per-token scalar driving surface brightness modulation.",
            type="select",
            default="kl_divergence",
            options=["none"] + list(SCALAR_MEASURES.keys()),
        ),
        ModuleParameter(
            name="bar_channel",
            display_name="Bar Channel",
            description="Per-token scalar driving underline bar length.",
            type="select",
            default="kl_divergence",
            options=["none"] + list(SCALAR_MEASURES.keys()),
        ),
        ModuleParameter(
            name="filter_channel",
            display_name="Filter Channel",
            description="Per-token scalar for dim/highlight filtering.",
            type="select",
            default="none",
            options=["none"] + list(SCALAR_MEASURES.keys()),
        ),

        # ── Data selection ──
        ModuleParameter(
            name="category",
            display_name="Category Filter",
            description="Restrict prompts to a single category, or show all.",
            type="select",
            default="all",
            options=["all", "benign", "mild", "harmful", "jailbreak",
                     "adversarial", "dual-use", "chat", "model_response"],
        ),
        ModuleParameter(
            name="record_limit",
            display_name="Record Limit",
            description="Maximum prompts to load.",
            type="int",
            default=100,
            min_val=1,
            max_val=2000,
        ),
        ModuleParameter(
            name="token_limit",
            display_name="Token Limit",
            description="Maximum tokens rendered per prompt.",
            type="int",
            default=20,
            min_val=4,
            max_val=100,
        ),

        # ── Visual encoding ──
        ModuleParameter(
            name="palette",
            display_name="Color Palette",
            description="Color ramps applied to the banks and spine.",
            type="select",
            default="dual",
            options=["dual", "vivid", "ember"],
        ),
        ModuleParameter(
            name="brightness_strength",
            display_name="Brightness Strength",
            description="How strongly the brightness channel modulates surface color. "
                        "0 = no effect, 1 = strong.",
            type="float",
            default=0.3,
            min_val=0.0,
            max_val=1.0,
        ),
        ModuleParameter(
            name="accent_mode",
            display_name="Label Accent Encoding",
            description="What the label text color encodes. "
                        "'promoted_demoted' tints by set-membership status. "
                        "'off' uses neutral cream for all labels.",
            type="select",
            default="promoted_demoted",
            options=["promoted_demoted", "off"],
        ),

        # ── Visual subsystem toggles ──
        ModuleParameter(
            name="show_grid",
            display_name="Show Grid Lines",
            description="Render the lattice overlay on the terrain.",
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="show_labels",
            display_name="Show Token Labels",
            description="Render per-vertex token labels.",
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="show_legend",
            display_name="Show Axis Legend",
            description="Render axis markers.",
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="show_spine",
            display_name="Show Spine Ridge",
            description="Render the spine ridge line and spheres.",
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="show_bars",
            display_name="Show Underline Bars",
            description="Render per-token underline bars.",
            type="bool",
            default=True,
        ),

        # ── Label formatting ──
        ModuleParameter(
            name="char_limit",
            display_name="Prompt Label Length",
            description="Character limit for prompt labels in the viewer dropdown.",
            type="int",
            default=50,
            min_val=20,
            max_val=200,
        ),

        # ── Camera motion ──
        ModuleParameter(
            name="camera_preset",
            display_name="Camera Preset",
            description="Initial camera angle. 'overhead' looks down at the "
                        "terrain. 'perspective' views from a low angle across "
                        "the surface.",
            type="select",
            default="overhead",
            options=["overhead", "perspective"],
        ),
        ModuleParameter(
            name="gain",
            display_name="Vertical Gain (%)",
            description="Scales terrain height. 100% = unity, below attenuates, "
                        "above amplifies.",
            type="int",
            default=30,
            min_val=0,
            max_val=200,
        ),
        ModuleParameter(
            name="auto_rotate",
            display_name="Auto-Rotate",
            description="Spin terrain on launch.",
            type="bool",
            default=False,
        ),
        ModuleParameter(
            name="rotate_speed",
            display_name="Rotate Speed (RPM)",
            description="Auto-rotation speed.",
            type="float",
            default=2.0,
            min_val=0.1,
            max_val=2.0,
        ),
        ModuleParameter(
            name="rotate_ccw",
            display_name="Rotate Counter-Clockwise",
            description="Reverse rotation direction.",
            type="bool",
            default=False,
        ),
    ]

    # ── Lifecycle ──

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        height = params.get("height_measure") or "rank_displacement"
        if height not in BANK_MEASURES:
            return False, f"Unknown height measure: {height!r}."

        check = BANK_CHECKS.get(height)
        if check and not any(check(r) for r in session_results):
            display = BANK_MEASURES[height]["display_name"]
            return False, (
                f"Height measure '{display}' has no data in this session. "
                f"Re-run analysis with the corresponding pipeline enabled."
            )
        return True, "OK"

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[CFT] {msg}")

        height_measure = params.get("height_measure") or "rank_displacement"
        category = params.get("category", "all")
        record_limit = int(params.get("record_limit", 100))
        token_limit = int(params.get("token_limit", 20))

        # Detect what's available across the session
        available_banks, available_scalars = _detect_available(session_results)
        prog(f"Available banks: {available_banks}, scalars: {available_scalars}")

        # Category filter
        if category != "all":
            filtered = [r for r in session_results
                        if (r.get("category") or "").lower().strip() == category]
        else:
            filtered = list(session_results)

        # Build unified prompt payloads
        prompts = []
        for r in filtered:
            if len(prompts) >= record_limit:
                break
            payload = _build_prompt(r, token_limit,
                                    available_banks, available_scalars)
            if payload is not None:
                # Must have the selected height measure
                if height_measure in payload["banks"]:
                    prompts.append(payload)

        prog(f"Built {len(prompts)} prompt payloads.")

        if not prompts:
            return {
                "error": (f"No prompts with data for height measure "
                          f"'{height_measure}' after filtering."),
                "height_measure": height_measure,
                "available_measures": {
                    "bank": available_banks,
                    "scalar": available_scalars,
                },
            }

        stats = _compute_summary_stats(prompts, height_measure)

        # Resolve accent mode
        height_spec = BANK_MEASURES[height_measure]
        accent_available = bool(height_spec.get("accent_capable", False))
        requested_accent = params.get("accent_mode", "promoted_demoted")
        if requested_accent == "promoted_demoted" and not accent_available:
            prog("Height measure does not support promoted/demoted accent; "
                 "falling back to off.")
            effective_accent = "off"
        else:
            effective_accent = requested_accent

        # Channel config — echoed so the renderer knows the bindings
        channel_config = {
            "height": height_measure,
            "spine": height_measure,
            "brightness": params.get("brightness_channel", "kl_divergence"),
            "bar_length": params.get("bar_channel", "kl_divergence"),
            "bar_color": "status" if effective_accent == "promoted_demoted" else "none",
            "font_color": "status" if effective_accent == "promoted_demoted" else "none",
            "filter": params.get("filter_channel", "none"),
        }

        result = {
            "geometry": {
                "k": K,
                "col_spacing": COL_SPACING,
                "ceiling_ratio": CEILING_RATIO,
            },
            "height_measure": height_measure,
            "accent_available": accent_available,
            "available_measures": {
                "bank": available_banks,
                "scalar": available_scalars,
            },
            "channel_config": channel_config,
            "prompts": prompts,
            "stats": stats,
            "launch_params": {
                "height_measure": height_measure,
                "category": category,
                "record_limit": record_limit,
                "token_limit": token_limit,
                "palette": params.get("palette", "dual"),
                "brightness_strength": float(
                    params.get("brightness_strength", 0.3)),
                "accent_mode": effective_accent,
                "show_grid": bool(params.get("show_grid", True)),
                "show_labels": bool(params.get("show_labels", True)),
                "show_legend": bool(params.get("show_legend", True)),
                "show_spine": bool(params.get("show_spine", True)),
                "show_bars": bool(params.get("show_bars", True)),
                "char_limit": int(params.get("char_limit", 50)),
                "auto_rotate": bool(params.get("auto_rotate", False)),
                "rotate_speed": float(params.get("rotate_speed", 2.0)),
                "rotate_ccw": bool(params.get("rotate_ccw", False)),
                "gain": int(params.get("gain", 100)),
                "camera_preset": params.get("camera_preset", "overhead"),
                "channel_config": channel_config,
            },
        }

        prog("Topology analysis complete.")
        return result
