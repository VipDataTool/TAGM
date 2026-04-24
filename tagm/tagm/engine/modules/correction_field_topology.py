"""
Correction Field Topology Module for TAGM.

Produces a uniform terrain payload for the 3D counterfactual-plane
visualization rendered client-side in Three.js. The module validates
that the selected measure has data in the session, builds a single
payload shaped to the renderer's contract, and returns an enumeration
of measures the session can actually support.

─── Payload contract ─────────────────────────────────────────────
Each run emits exactly one measure's data. The renderer consumes a
uniform shape regardless of which measure was selected:

    {
        "measure": "<measure_name>",            # selected measure identifier
        "measure_class": "bank" | "scalar",     # geometry class (see below)
        "target_max": <float>,                  # per-measure calibration
        "accent_available": bool,               # promoted/demoted tags present?
        "available_measures": [<name>, ...],    # what this session supports
        "prompts": [
            {
                "prompt": str,
                "category": str,
                "tokens": [str, ...],           # length = n_tokens
                "per_token_kl": [float, ...],   # for color modulation
                "primary": [float, ...],        # length = n_tokens
                "instruct_bank": [[float, ...], ...],   # [n_tokens][k]; empty for scalar
                "base_bank": [[float, ...], ...],       # [n_tokens][k]; empty for scalar
                "instruct_status": [[str, ...], ...],   # per-rank: "promoted"|"matched"
                "base_status": [[str, ...], ...],       # per-rank: "demoted"|"matched"
                "counterfactual_tokens": [[str, ...], ...],
                "base_counterfactual_tokens": [[str, ...], ...],
            },
            ...
        ],
        "stats": {...},                         # summary stats for UI readouts
        "launch_params": {...},                 # echoed back for the viewer UI
    }

─── Accent encoding ───────────────────────────────────────────────
The instruct_status and base_status arrays carry the set-membership
classification from the base-vs-instruct top-k comparison. Each
instruct-side candidate is either "matched" (also present in base
top-k) or "promoted" (RLHF introduced it). Each base-side candidate
is either "matched" or "demoted" (RLHF pushed it out). Only rank
displacement provides this data, since it alone compares the two
models' candidate distributions directly. LTP and the scalar-only
measures emit empty status arrays and set accent_available to False;
the client falls back to the kl_brightness accent mode in that case.

─── Measure classes ──────────────────────────────────────────────
Bank-native measures carry per-rank profiles for both banks and
produce the full dual-bank terrain:
    - "rank_displacement": RD magnitude, uses instruct/base_disp_profiles
                           with per_position[t].total_disp as primary
    - "ltp_tension":       LTP tension, uses profiles/base_profiles with
                           tension_magnitudes as primary

Scalar-only measures carry per-token primary values with empty bank
arrays. The renderer draws the spine ridge alone and suppresses the
bank terrain:
    - "asm_stress":        per_token_stress as primary
    - "sfd_density":       sfd.per_token_density as primary
    - "per_token_kl":      per_token_kl as primary

Only the selected measure's data travels to the renderer; non-selected
measure data stays on the server. This keeps the viewer's memory
footprint bounded as the measure set grows.

─── Absent from this module ──────────────────────────────────────
Token Variance is a cross-prompt aggregate and does not map onto the
per-prompt terrain payload. If future work exposes TV as a tint or
filter layer, it belongs alongside the palette selector rather than
in the measure enumeration.
"""

import logging
import numpy as np
from collections import defaultdict

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")


# ─── Measure registry ───────────────────────────────────────────
# Each measure declares its geometry class, the renderer calibration
# target (natural magnitude at which HSCALE=180 produces a legible
# terrain), and a human-readable label for the UI dropdown.

MEASURES = {
    "rank_displacement": {
        "display_name": "Rank Displacement",
        "class": "bank",
        "target_max": 1.0,
        "accent_capable": True,
        "description": "Per-token probability displacement between base "
                       "and instruct top-k candidates.",
    },
    "ltp_tension": {
        "display_name": "LTP Tension",
        "class": "bank",
        "target_max": 0.008,
        "accent_capable": False,
        "description": "Lateral tension magnitudes from counterfactual "
                       "refinement probing.",
    },
    "asm_stress": {
        "display_name": "ASM Stress",
        "class": "scalar",
        "target_max": 1.0,
        "accent_capable": False,
        "description": "Per-token correction stress from ASM signal layers. "
                       "Spine ridge only.",
    },
    "sfd_density": {
        "display_name": "SFD Density",
        "class": "scalar",
        "target_max": 1.0,
        "accent_capable": False,
        "description": "Spectral field density per token. Spine ridge only.",
    },
    "per_token_kl": {
        "display_name": "Per-token KL",
        "class": "scalar",
        "target_max": 5.0,
        "accent_capable": False,
        "description": "KL divergence between base and instruct distributions. "
                       "Spine ridge only.",
    },
}


# ─── Measure availability detection ─────────────────────────────

def _has_rank_displacement(r):
    rd = r.get("rank_displacement") or {}
    profs = rd.get("instruct_disp_profiles") or []
    return len(profs) > 0


def _has_ltp_tension(r):
    ltp = r.get("ltp") or {}
    profs = ltp.get("profiles") or []
    return len(profs) > 0


def _has_asm_stress(r):
    pts = r.get("per_token_stress")
    return pts is not None and len(pts) > 0


def _has_sfd_density(r):
    sfd = r.get("sfd") or {}
    ptd = sfd.get("per_token_density") or []
    return len(ptd) > 0


def _has_per_token_kl(r):
    kl = r.get("per_token_kl")
    return kl is not None and len(kl) > 0


AVAILABILITY_CHECKS = {
    "rank_displacement": _has_rank_displacement,
    "ltp_tension": _has_ltp_tension,
    "asm_stress": _has_asm_stress,
    "sfd_density": _has_sfd_density,
    "per_token_kl": _has_per_token_kl,
}


def _detect_available_measures(session_results):
    """Return the list of measures with data in at least one record."""
    available = []
    for measure_name, check_fn in AVAILABILITY_CHECKS.items():
        if any(check_fn(r) for r in session_results):
            available.append(measure_name)
    return available


# ─── Per-measure payload builders ───────────────────────────────
# Each builder receives a single session record and returns either a
# prompt payload dict or None (record lacks the required data and
# should be skipped). Builders never fall back to other measures.

def _cf_token_name(entry):
    """Extract the token string from a counterfactual record.

    Counterfactual entries may be either a raw string or a [token, prob]
    pair; this helper normalizes to the token string for set membership.
    """
    if isinstance(entry, (list, tuple)) and entry:
        return str(entry[0])
    if entry is None:
        return ""
    return str(entry)


def _classify_banks(cf_inst, cf_base, k=8):
    """Compute per-rank status tags for the two banks at one token position.

    Each instruct-side candidate is "matched" if its token also appears
    in the base-side top-k, otherwise "promoted". Each base-side
    candidate is "matched" if its token also appears in the instruct
    top-k, otherwise "demoted". Returns a pair of lists of length k.
    """
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


def _build_rd_prompt(r, token_limit):
    rd = r.get("rank_displacement") or {}
    i_profs = rd.get("instruct_disp_profiles") or []
    b_profs = rd.get("base_disp_profiles") or []
    per_pos = rd.get("per_position") or []
    if not i_profs:
        return None

    tokens = (r.get("tokens") or [])[:token_limit]
    n_tok = min(len(tokens), len(i_profs))
    tokens = tokens[:n_tok]

    primary = [float(per_pos[i].get("total_disp", 0.0))
               if i < len(per_pos) else 0.0
               for i in range(n_tok)]
    instruct_bank = [list(i_profs[i]) for i in range(n_tok)]
    base_bank = [list(b_profs[i]) if i < len(b_profs) else []
                 for i in range(n_tok)]

    ltp = r.get("ltp") or {}
    cf_full = ltp.get("counterfactual_tokens") or []
    base_cf_full = r.get("base_counterfactual_tokens") or []

    instruct_status = []
    base_status = []
    for i in range(n_tok):
        cf_i = cf_full[i] if i < len(cf_full) else []
        bcf_i = base_cf_full[i] if i < len(base_cf_full) else []
        i_stat, b_stat = _classify_banks(cf_i, bcf_i, k=8)
        instruct_status.append(i_stat)
        base_status.append(b_stat)

    cf = cf_full[:n_tok]
    base_cf = base_cf_full[:n_tok]

    return {
        "prompt": r.get("prompt", ""),
        "category": r.get("category", "unknown"),
        "tokens": tokens,
        "per_token_kl": _extract_kl(r, n_tok),
        "primary": primary,
        "instruct_bank": instruct_bank,
        "base_bank": base_bank,
        "instruct_status": instruct_status,
        "base_status": base_status,
        "counterfactual_tokens": [list(c) for c in cf],
        "base_counterfactual_tokens": [list(c) for c in base_cf],
    }


def _build_ltp_prompt(r, token_limit):
    ltp = r.get("ltp") or {}
    profs = ltp.get("profiles") or []
    base_profs = ltp.get("base_profiles") or []
    tensions = ltp.get("tension_magnitudes") or []
    if not profs:
        return None

    tokens = (r.get("tokens") or [])[:token_limit]
    n_tok = min(len(tokens), len(profs))
    tokens = tokens[:n_tok]

    primary = [float(tensions[i]) if i < len(tensions) else 0.0
               for i in range(n_tok)]
    instruct_bank = [list(profs[i]) for i in range(n_tok)]
    base_bank = [list(base_profs[i]) if i < len(base_profs) else []
                 for i in range(n_tok)]

    cf = (ltp.get("counterfactual_tokens") or [])[:n_tok]
    base_cf = (r.get("base_counterfactual_tokens") or [])[:n_tok]

    return {
        "prompt": r.get("prompt", ""),
        "category": r.get("category", "unknown"),
        "tokens": tokens,
        "per_token_kl": _extract_kl(r, n_tok),
        "primary": primary,
        "instruct_bank": instruct_bank,
        "base_bank": base_bank,
        "instruct_status": [],
        "base_status": [],
        "counterfactual_tokens": [list(c) for c in cf],
        "base_counterfactual_tokens": [list(c) for c in base_cf],
    }


def _build_scalar_prompt(r, token_limit, scalar_field_fn):
    """Shared builder for scalar-only measures.

    scalar_field_fn(r) -> list[float] or None
    """
    scalar = scalar_field_fn(r)
    if scalar is None or len(scalar) == 0:
        return None

    tokens = (r.get("tokens") or [])[:token_limit]
    n_tok = min(len(tokens), len(scalar))
    tokens = tokens[:n_tok]

    primary = [float(scalar[i]) for i in range(n_tok)]

    return {
        "prompt": r.get("prompt", ""),
        "category": r.get("category", "unknown"),
        "tokens": tokens,
        "per_token_kl": _extract_kl(r, n_tok),
        "primary": primary,
        "instruct_bank": [],
        "base_bank": [],
        "instruct_status": [],
        "base_status": [],
        "counterfactual_tokens": [],
        "base_counterfactual_tokens": [],
    }


def _extract_kl(r, n_tok):
    kl = r.get("per_token_kl") or []
    return [float(kl[i]) if i < len(kl) else 0.0 for i in range(n_tok)]


def _build_asm_stress_prompt(r, token_limit):
    return _build_scalar_prompt(r, token_limit, lambda rec: rec.get("per_token_stress"))


def _build_sfd_density_prompt(r, token_limit):
    def _get(rec):
        sfd = rec.get("sfd") or {}
        return sfd.get("per_token_density")
    return _build_scalar_prompt(r, token_limit, _get)


def _build_kl_prompt(r, token_limit):
    return _build_scalar_prompt(r, token_limit, lambda rec: rec.get("per_token_kl"))


PROMPT_BUILDERS = {
    "rank_displacement": _build_rd_prompt,
    "ltp_tension": _build_ltp_prompt,
    "asm_stress": _build_asm_stress_prompt,
    "sfd_density": _build_sfd_density_prompt,
    "per_token_kl": _build_kl_prompt,
}


# ─── Summary statistics ─────────────────────────────────────────

def _compute_summary_stats(prompt_payloads):
    """Lightweight summary for UI readouts.

    Reports counts, token statistics, and per-category primary-signal
    magnitude. No asymmetry statistics: the previous version computed
    those but no downstream consumer read them.
    """
    stats = {
        "n_prompts": len(prompt_payloads),
        "by_category": {},
        "token_stats": {
            "total_tokens": 0,
            "mean_tokens_per_prompt": 0,
            "max_tokens": 0,
        },
        "primary_stats": None,
    }

    if not prompt_payloads:
        return stats

    token_counts = []
    primary_by_cat = defaultdict(list)
    all_primary = []

    for p in prompt_payloads:
        cat = p.get("category") or "unknown"
        n_tok = len(p.get("tokens") or [])
        token_counts.append(n_tok)

        if cat not in stats["by_category"]:
            stats["by_category"][cat] = {"count": 0, "token_counts": []}
        stats["by_category"][cat]["count"] += 1
        stats["by_category"][cat]["token_counts"].append(n_tok)

        for v in (p.get("primary") or []):
            primary_by_cat[cat].append(v)
            all_primary.append(v)

    stats["token_stats"]["total_tokens"] = sum(token_counts)
    stats["token_stats"]["mean_tokens_per_prompt"] = round(
        float(np.mean(token_counts)), 1)
    stats["token_stats"]["max_tokens"] = max(token_counts)

    for cat, info in stats["by_category"].items():
        counts = info.pop("token_counts")
        info["mean_tokens"] = round(float(np.mean(counts)), 1) if counts else 0

    if all_primary:
        arr = np.array(all_primary)
        per_cat = {}
        for cat, vals in primary_by_cat.items():
            a = np.array(vals)
            per_cat[cat] = {
                "mean": round(float(a.mean()), 6),
                "median": round(float(np.median(a)), 6),
                "max": round(float(a.max()), 6),
            }
        stats["primary_stats"] = {
            "overall": {
                "mean": round(float(arr.mean()), 6),
                "median": round(float(np.median(arr)), 6),
                "max": round(float(arr.max()), 6),
            },
            "by_category": per_cat,
        }

    return stats


# ─── Module ─────────────────────────────────────────────────────

class CorrectionFieldTopologyModule(TASMModule):
    name = "correction_field_topology"
    display_name = "Correction Field Topology"
    description = (
        "3D terrain visualization of the counterfactual plane. The user "
        "selects which per-token measure drives the spine ridge and — for "
        "bank-native measures — the flanking rank profiles. Bank-native: "
        "rank displacement, LTP tension. Scalar-only: ASM stress, SFD "
        "density, per-token KL."
    )
    version = "2.0.0"

    min_results = 1
    # Per-measure requirements are enforced in validate() below, not by
    # the coarse requires_* flags on the base class.
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    parameters = [
        # ── Signal source ──
        ModuleParameter(
            name="measure",
            display_name="Measure",
            description="Which per-token measure drives the terrain geometry. "
                        "The UI should restrict options to measures the session "
                        "has data for.",
            type="select",
            default="rank_displacement",
            options=list(MEASURES.keys()),
        ),

        # ── Data selection ──
        ModuleParameter(
            name="category",
            display_name="Category Filter",
            description="Restrict prompts to a single category, or show all.",
            type="select",
            default="all",
            options=["all", "benign", "mild", "harmful", "jailbreak",
                     "adversarial", "dual-use"],
        ),
        ModuleParameter(
            name="record_limit",
            display_name="Record Limit",
            description="Maximum prompts to load. Higher = slower but more complete.",
            type="int",
            default=100,
            min_val=1,
            max_val=2000,
        ),
        ModuleParameter(
            name="token_limit",
            display_name="Token Limit",
            description="Maximum tokens rendered per prompt. Higher = more detail "
                        "but heavier.",
            type="int",
            default=20,
            min_val=4,
            max_val=100,
        ),

        # ── Visual encoding of the signal ──
        ModuleParameter(
            name="palette",
            display_name="Color Palette",
            description="Color ramps applied to the banks and spine.",
            type="select",
            default="dual",
            options=["dual", "vivid", "ember"],
        ),
        ModuleParameter(
            name="height_scale",
            display_name="Vertical Exaggeration",
            description="Multiplier on primary signal magnitude when "
                        "computing terrain height.",
            type="int",
            default=180,
            min_val=50,
            max_val=400,
        ),
        ModuleParameter(
            name="kl_boost",
            display_name="KL Brightness Boost",
            description="How strongly KL divergence modulates color brightness "
                        "on the terrain surface. 0 = no effect, 1 = strong.",
            type="float",
            default=0.3,
            min_val=0.0,
            max_val=1.0,
        ),
        ModuleParameter(
            name="accent_mode",
            display_name="Label Accent Encoding",
            description="What the label underlines and text tints encode. "
                        "'promoted_demoted' requires an accent-capable measure "
                        "(currently rank_displacement). 'kl_brightness' tints "
                        "labels by KL divergence as in prior versions. 'off' "
                        "leaves all labels in the neutral cream color.",
            type="select",
            default="promoted_demoted",
            options=["promoted_demoted", "kl_brightness", "off"],
        ),

        # ── Visual subsystem toggles ──
        ModuleParameter(
            name="data_filter",
            display_name="Token Filter",
            description="Dim tokens below a threshold. 'highkl' keeps tokens "
                        "with above-mean KL; 'hotspots' keeps tokens with "
                        "above-median primary signal.",
            type="select",
            default="all",
            options=["all", "highkl", "hotspots"],
        ),
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
            description="Render '← base' and 'instruct →' axis markers.",
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="show_spine",
            display_name="Show Spine Ridge",
            description="Render the cream ridge line and spheres along the "
                        "prompt-token axis.",
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
            name="auto_rotate",
            display_name="Auto-Rotate",
            description="Spin terrain around vertical axis on launch.",
            type="bool",
            default=False,
        ),
        ModuleParameter(
            name="rotate_speed",
            display_name="Rotate Speed (RPM)",
            description="Auto-rotation speed in revolutions per minute.",
            type="float",
            default=0.3,
            min_val=0.1,
            max_val=2.0,
        ),
    ]

    # ── Lifecycle ──

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        measure = params.get("measure") or "rank_displacement"
        if measure not in MEASURES:
            return False, f"Unknown measure: {measure!r}."

        check = AVAILABILITY_CHECKS[measure]
        n_with_data = sum(1 for r in session_results if check(r))
        if n_with_data == 0:
            display = MEASURES[measure]["display_name"]
            return False, (
                f"Selected measure '{display}' has no data in this session. "
                f"Re-run analysis with the corresponding pipeline enabled, "
                f"or pick a measure from the available list."
            )
        return True, "OK"

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[CFT] {msg}")

        measure = params.get("measure") or "rank_displacement"
        spec = MEASURES[measure]
        category = params.get("category", "all")
        record_limit = int(params.get("record_limit", 100))
        token_limit = int(params.get("token_limit", 20))

        prog(f"Building payload for measure '{spec['display_name']}'...")

        # Category filter first, then record limit.
        if category != "all":
            filtered = [r for r in session_results
                        if (r.get("category") or "").lower().strip() == category]
        else:
            filtered = list(session_results)

        # Build per-prompt payloads.
        builder = PROMPT_BUILDERS[measure]
        prompts = []
        for r in filtered:
            if len(prompts) >= record_limit:
                break
            payload = builder(r, token_limit)
            if payload is not None:
                prompts.append(payload)

        prog(f"Built {len(prompts)} prompt payloads.")

        if not prompts:
            return {
                "error": (f"No prompts with data for measure "
                          f"'{spec['display_name']}' after filtering."),
                "measure": measure,
                "available_measures": _detect_available_measures(session_results),
            }

        stats = _compute_summary_stats(prompts)

        # Resolve accent_mode against the measure's accent capability. If
        # the user picked promoted_demoted for a non-capable measure, fall
        # back to kl_brightness silently rather than failing the run.
        requested_accent = params.get("accent_mode", "promoted_demoted")
        accent_available = bool(spec.get("accent_capable", False))
        if requested_accent == "promoted_demoted" and not accent_available:
            prog("Selected measure does not support promoted/demoted accent; "
                 "falling back to kl_brightness.")
            effective_accent = "kl_brightness"
        else:
            effective_accent = requested_accent

        result = {
            "measure": measure,
            "measure_class": spec["class"],
            "target_max": spec["target_max"],
            "accent_available": accent_available,
            "available_measures": _detect_available_measures(session_results),
            "prompts": prompts,
            "stats": stats,
            "launch_params": {
                "measure": measure,
                "category": category,
                "record_limit": record_limit,
                "token_limit": token_limit,
                "palette": params.get("palette", "dual"),
                "height_scale": int(params.get("height_scale", 180)),
                "kl_boost": float(params.get("kl_boost", 0.3)),
                "accent_mode": effective_accent,
                "data_filter": params.get("data_filter", "all"),
                "show_grid": bool(params.get("show_grid", True)),
                "show_labels": bool(params.get("show_labels", True)),
                "show_legend": bool(params.get("show_legend", True)),
                "show_spine": bool(params.get("show_spine", True)),
                "char_limit": int(params.get("char_limit", 50)),
                "auto_rotate": bool(params.get("auto_rotate", False)),
                "rotate_speed": float(params.get("rotate_speed", 0.3)),
            },
        }

        prog("Topology analysis complete.")
        return result
