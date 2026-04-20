"""TokenVariance: per-token coupling stability across prompt contexts.

Direct port of TASM's `engine/modules/token_variance.py`, reading TAGM's
per-token measurement outputs (stress_score, spectral_field_density,
last_position_attribution) and emitting the JSON shape the UI's
`renderTokenVarianceResults` renderer reads:

    {
      "summary": {
        "n_prompts", "n_skipped", "n_tokens_analyzed", "n_qualified",
        "qualified_threshold", "categories",
        "density_cv": {"p10","p25","p50","p75","p90","eta_sq_mean","eta_sq_median"},
        "stress_cv":  {...},
        "attr_cv":    {...},
        "context_stable":   {"threshold","tokens","count"},
        "context_dependent":{"threshold","tokens","count"},
      },
      "highest_cv": [{token, n, n_cats, density_cv, stress_cv,
                      density_mean, stress_mean, eta_sq_density, ...}, ...],
      "lowest_cv":  [...],
      "high_eta":   [...],
      "pairwise":   { "<cat_a>_vs_<cat_b>":
                      [{token, mean_a, n_a, mean_b, n_b, diff}, ...] },
    }

The algorithm is faithful to TASM — same channels (density/stress/attr),
same pairwise category comparisons, same CV-percentile-based
context-stable / context-dependent partitions.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter


# Channel registry: display_key -> (measurement_name, per_token_field)
_CHANNELS = {
    "density": ("spectral_field_density",    "density"),
    "stress":  ("stress_score",              "stress"),
    "attr":    ("last_position_attribution", "signed_attribution_to_last"),
}
_CHANNEL_KEYS = ("density", "stress", "attr")

# Category pairs the UI shows comparisons for
_CAT_PAIRS = [
    ("benign", "jailbreak"),
    ("benign", "adversarial"),
    ("benign", "harmful"),
    ("harmful", "jailbreak"),
]
_CAT_ORDER = ("benign", "mild", "dual-use", "harmful", "adversarial",
              "jailbreak")


@register_analysis
class TokenVariance(AnalysisModule):
    name = "token_variance"
    display_name = "Token Variance"
    description = (
        "Measures how each token's coupling to the correction manifold "
        "varies across prompt contexts. Identifies context-dependent "
        "vs context-stable tokens and computes per-category profiles "
        "and pairwise category comparisons."
    )
    version = "1.0.0"

    depends_on_measurements = ("stress_score", "spectral_field_density",
                               "last_position_attribution")

    parameters = [
        ModuleParameter(
            name="min_appearances",
            display_name="Min appearances",
            description="Skip tokens seen fewer times than this.",
            kind="int", default=3, min_value=2, max_value=50,
        ),
        ModuleParameter(
            name="include_first",
            display_name="Include first-position tokens",
            description=(
                "Position-0 tokens are normally excluded because their "
                "attribution has a positional artifact; turn on if you "
                "want them anyway."
            ),
            kind="bool", default=False,
        ),
        ModuleParameter(
            name="top_n",
            display_name="Top N per section",
            description="Max tokens per table in the UI.",
            kind="int", default=30, min_value=5, max_value=100,
        ),
        ModuleParameter(
            name="min_seq_len",
            display_name="Min prompt length",
            description="Skip prompts shorter than this many tokens.",
            kind="int", default=3, min_value=1, max_value=20,
        ),
        ModuleParameter(
            name="min_per_cat",
            display_name="Min appearances per category",
            description=(
                "Minimum per-category count for a token to show up in "
                "pairwise category comparisons."
            ),
            kind="int", default=2, min_value=1, max_value=10,
        ),
    ]

    # ── check_dependencies: soft — any one of the three measurements
    # having per-token data is enough to produce partial output ──
    def check_dependencies(self, session):
        prompts = session.get("prompts") or []
        if not prompts:
            return [f"Analysis '{self.name}' needs at least one prompt."]
        for p in prompts:
            meas = p.get("measurements") or {}
            for mname, _ in _CHANNELS.values():
                pt = ((meas.get(mname) or {}).get("per_token") or {})
                if pt:
                    return []
        return [
            f"Analysis '{self.name}' requires at least one of these "
            f"per-token measurements: "
            f"{', '.join(sorted({m for m, _ in _CHANNELS.values()}))}. "
            f"None of the session prompts have any of them."
        ]

    def run(self, session, params, probes=None, context=None):
        min_app = int(params.get("min_appearances", 3))
        include_first = bool(params.get("include_first", False))
        top_n = int(params.get("top_n", 30))
        min_seq_len = int(params.get("min_seq_len", 3))
        min_per_cat = int(params.get("min_per_cat", 2))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={
                "min_appearances": min_app,
                "include_first": include_first,
                "top_n": top_n,
                "min_seq_len": min_seq_len,
                "min_per_cat": min_per_cat,
            },
        )

        prompts = session.get("prompts") or []

        # ── Extract per-token contexts across all prompts ──
        token_contexts, skipped, cats_seen = _extract_contexts(
            prompts, min_seq_len=min_seq_len)

        # ── Analyze ──
        analyzed = _analyze(token_contexts,
                            min_appearances=min_app,
                            exclude_first=not include_first)

        # ── Build output in UI wire format ──
        output = _build_output(analyzed, n_prompts=len(prompts),
                               skipped=skipped, cats_seen=cats_seen,
                               top_n=top_n, min_app=min_app,
                               min_per_cat=min_per_cat)

        # Stuff the whole output into objects for flattener promotion.
        result.objects.update(output)
        # Also surface a few scalars for the inspector.
        result.scalars["n_prompts"] = len(prompts)
        result.scalars["n_tokens_analyzed"] = output["summary"][
            "n_tokens_analyzed"]
        result.scalars["n_qualified"] = output["summary"]["n_qualified"]
        result.scalars["n_skipped"] = skipped
        return result


# ── Extraction ─────────────────────────────────────────────────

def _extract_contexts(prompts, min_seq_len=3):
    """Collect per-token (channel-value, category, position) contexts
    across all prompts. Returns (token_contexts, n_skipped, cats_seen).
    """
    contexts: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    cats: set[str] = set()

    for p in prompts:
        tokens = p.get("tokens") or []
        seq_len = p.get("seq_len") or len(tokens)
        if seq_len < min_seq_len or not tokens:
            skipped += 1
            continue

        meas = p.get("measurements") or {}
        # Pull per-token arrays for each channel. Missing channels are
        # tolerated — the token will simply lack that channel's stats.
        channel_arrays: dict[str, list] = {}
        for ch_key, (mname, field) in _CHANNELS.items():
            per_tok = ((meas.get(mname) or {}).get("per_token") or {})
            arr = per_tok.get(field)
            if arr is None:
                continue
            channel_arrays[ch_key] = arr

        if not channel_arrays:
            skipped += 1
            continue

        category = (p.get("category") or "").strip()
        if category:
            cats.add(category)
        prompt_text = (p.get("prompt") or "")[:60]

        for i, tok in enumerate(tokens):
            tok = (tok or "").strip()
            if not tok:
                continue
            entry = {
                "pos_frac": i / max(seq_len - 1, 1),
                "is_first": (i == 0),
                "is_last": (i == len(tokens) - 1),
                "category": category,
                "prompt": prompt_text,
            }
            got_any = False
            for ch_key in _CHANNEL_KEYS:
                arr = channel_arrays.get(ch_key)
                if arr is None or i >= len(arr):
                    entry[ch_key] = None
                    continue
                v = arr[i]
                entry[ch_key] = (None if v is None else float(v))
                if entry[ch_key] is not None:
                    got_any = True
            if got_any:
                contexts[tok].append(entry)

    return contexts, skipped, cats


# ── Stats helpers ──────────────────────────────────────────────

def _safe_cv(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    m = np.nanmean(arr)
    if not np.isfinite(m) or abs(m) < 1e-12:
        return 0.0
    return float(np.nanstd(arr) / abs(m))


def _channel_stats(values):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "range": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "cv": _safe_cv(arr),
        "range": float(np.nanmax(arr) - np.nanmin(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def _eta_squared(values_by_group):
    """One-way between-group η² (variance explained by grouping)."""
    groups = [np.asarray([v for v in g if v is not None], dtype=float)
              for g in values_by_group.values()]
    groups = [g for g in groups if g.size >= 1]
    if len(groups) < 2:
        return 0.0
    all_vals = np.concatenate(groups)
    if all_vals.size == 0:
        return 0.0
    grand = np.nanmean(all_vals)
    ss_total = float(np.nansum((all_vals - grand) ** 2))
    if ss_total < 1e-15:
        return 0.0
    ss_between = sum(len(g) * (np.nanmean(g) - grand) ** 2 for g in groups)
    return float(ss_between / ss_total)


def _analyze(token_contexts, min_appearances=3, exclude_first=True):
    """Per-token channel stats + η² + per-category profiles."""
    results = []
    for tok, contexts in token_contexts.items():
        if exclude_first:
            contexts = [c for c in contexts if not c["is_first"]]
        if len(contexts) < min_appearances:
            continue

        cats = {c["category"] for c in contexts if c["category"]}

        channel_stats: dict[str, dict] = {}
        for ch in _CHANNEL_KEYS:
            vals = [c[ch] for c in contexts if c.get(ch) is not None]
            channel_stats[ch] = _channel_stats(vals)

        by_cat: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list))
        for c in contexts:
            cat = c.get("category") or ""
            if not cat:
                continue
            for ch in _CHANNEL_KEYS:
                v = c.get(ch)
                if v is not None:
                    by_cat[cat][ch].append(v)

        cat_profiles: dict[str, dict] = {}
        for cat, ch_lists in by_cat.items():
            cat_profiles[cat] = {
                ch: {
                    "mean": float(np.mean(v)) if v else 0.0,
                    "std": float(np.std(v)) if v else 0.0,
                    "n": len(v),
                }
                for ch, v in ch_lists.items()
            }

        eta_sq: dict[str, float] = {}
        for ch in _CHANNEL_KEYS:
            groups = {cat: ch_lists.get(ch, [])
                       for cat, ch_lists in by_cat.items()}
            eta_sq[ch] = _eta_squared(groups)

        results.append({
            "token": tok,
            "n": len(contexts),
            "n_cats": len(cats),
            "cats": sorted(cats),
            "channels": channel_stats,
            "cat_profiles": cat_profiles,
            "eta_squared": eta_sq,
        })
    return results


def _token_row(r):
    """Flatten an analysis entry into the row shape the UI consumes."""
    row = {
        "token": r["token"],
        "n": r["n"],
        "n_cats": r["n_cats"],
        "cats": r["cats"],
    }
    for ch in _CHANNEL_KEYS:
        for stat in ("mean", "std", "cv", "range", "min", "max"):
            row[f"{ch}_{stat}"] = r["channels"][ch][stat]
        row[f"eta_sq_{ch}"] = r["eta_squared"][ch]
    for cat in _CAT_ORDER:
        if cat in r["cat_profiles"]:
            for ch in _CHANNEL_KEYS:
                if ch in r["cat_profiles"][cat]:
                    row[f"{ch}_{cat}_mean"] = r["cat_profiles"][cat][ch]["mean"]
                    row[f"{ch}_{cat}_n"] = r["cat_profiles"][cat][ch]["n"]
    return row


def _pairwise_deltas(results, cat_a, cat_b, channel="density",
                      min_per_cat=2, top_n=None):
    """Per-token mean difference on a channel, filtered to tokens that
    appeared at least `min_per_cat` times in BOTH categories."""
    pairs = []
    for r in results:
        cp = r.get("cat_profiles") or {}
        pa_entry = cp.get(cat_a, {}).get(channel)
        pb_entry = cp.get(cat_b, {}).get(channel)
        if not pa_entry or not pb_entry:
            continue
        if pa_entry["n"] < min_per_cat or pb_entry["n"] < min_per_cat:
            continue
        pairs.append({
            "token": r["token"],
            "mean_a": pa_entry["mean"], "n_a": pa_entry["n"],
            "mean_b": pb_entry["mean"], "n_b": pb_entry["n"],
            "diff": pb_entry["mean"] - pa_entry["mean"],
        })
    pairs.sort(key=lambda x: x["diff"], reverse=True)
    if top_n and len(pairs) > top_n * 2:
        return pairs[:top_n] + pairs[-top_n:]
    return pairs


# ── Output shaping ─────────────────────────────────────────────

def _build_output(results, n_prompts, skipped, cats_seen,
                  top_n, min_app, min_per_cat):
    qualified = [r for r in results if r["n"] >= min_app]

    by_den_desc = sorted(results,
                          key=lambda x: x["channels"]["density"]["cv"],
                          reverse=True)
    by_den_asc = sorted(results,
                         key=lambda x: x["channels"]["density"]["cv"])
    by_eta_desc = sorted(
        [r for r in results if r["n_cats"] >= 2 and r["n"] >= min_app],
        key=lambda x: x["eta_squared"]["density"], reverse=True)

    summary: dict = {
        "n_prompts": n_prompts,
        "n_skipped": skipped,
        "n_tokens_analyzed": len(results),
        "n_qualified": len(qualified),
        "qualified_threshold": min_app,
        "categories": sorted(cats_seen),
    }

    if qualified:
        for ch in _CHANNEL_KEYS:
            cvs = np.array([r["channels"][ch]["cv"] for r in qualified])
            etas = np.array([r["eta_squared"][ch] for r in qualified])
            p10, p25, p50, p75, p90 = [float(x) for x in
                                        np.percentile(cvs, [10, 25, 50, 75, 90])]
            summary[f"{ch}_cv"] = {
                "mean": float(np.mean(cvs)),
                "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
                "eta_sq_mean": float(np.mean(etas)),
                "eta_sq_median": float(np.median(etas)),
            }

        den_cvs = [r["channels"]["density"]["cv"] for r in qualified]
        if len(den_cvs) >= 4:
            p25v = float(np.percentile(den_cvs, 25))
            p75v = float(np.percentile(den_cvs, 75))
            stable_tokens = [r["token"] for r in qualified
                              if r["channels"]["density"]["cv"] <= p25v]
            dep_tokens = [r["token"] for r in qualified
                           if r["channels"]["density"]["cv"] >= p75v]
            summary["context_stable"] = {
                "threshold": p25v,
                "tokens": stable_tokens[:top_n],
                "count": len(stable_tokens),
            }
            summary["context_dependent"] = {
                "threshold": p75v,
                "tokens": dep_tokens[:top_n],
                "count": len(dep_tokens),
            }

    pairwise: dict = {}
    for ca, cb in _CAT_PAIRS:
        deltas = _pairwise_deltas(results, ca, cb,
                                   min_per_cat=min_per_cat, top_n=top_n)
        if deltas:
            pairwise[f"{ca}_vs_{cb}"] = deltas

    out = {
        "summary": summary,
        "highest_cv": [_token_row(r) for r in by_den_desc[:top_n]],
        "lowest_cv": [_token_row(r) for r in by_den_asc[:top_n]],
        "high_eta": [_token_row(r) for r in by_eta_desc[:top_n]],
        "pairwise": pairwise,
        # Kept for debugging / TAGM-native consumers
        "all_tokens_count": len(results),
    }
    return _round_floats(out)


def _round_floats(x, decimals=6):
    if isinstance(x, dict):
        return {k: _round_floats(v, decimals) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_floats(v, decimals) for v in x]
    if isinstance(x, float):
        return round(x, decimals)
    return x
