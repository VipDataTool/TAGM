"""Token Variance: cross-context per-token coupling stability.

Ported from TASM's engine/modules/token_variance.py. Full computation
preserved; data reading adapted to TAGM's native session schema.
Output shape matches TASM so the existing UI renderer works unchanged.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Optional

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter


CHANNELS = ["density", "stress", "attr"]

CH_ABBREV = {
    "density": "den", "stress": "str", "attr": "att",
}

CAT_ORDER = ["benign", "mild", "dual-use", "harmful", "adversarial", "jailbreak"]


@register_analysis
class TokenVariance(AnalysisModule):
    """Cross-context token variance analysis."""

    name = "token_variance"
    display_name = "Token Variance"
    description = (
        "Measures how each token's coupling to the correction manifold "
        "varies across prompt contexts. Identifies context-dependent vs. "
        "context-stable tokens and computes per-category profiles."
    )
    version = "1.0.0"

    min_results = 10

    depends_on_measurements = (
        "stress_score", "last_position_attribution", "spectral_field_density",
    )

    parameters = [
        ModuleParameter(
            name="min_appearances",
            display_name="Min Appearances",
            description="Minimum interior appearances to include a token",
            kind="int",
            default=3,
            min_value=2,
            max_value=50,
        ),
        ModuleParameter(
            name="merge_subwords",
            display_name="Merge Subwords",
            description="Merge BPE subword tokens into whole words before analysis",
            kind="bool",
            default=False,
        ),
        ModuleParameter(
            name="include_first",
            display_name="Include First Token",
            description="Include position-0 tokens (usually excluded due to positional artifact)",
            kind="bool",
            default=False,
        ),
        ModuleParameter(
            name="top_n",
            display_name="Top N",
            description="Maximum tokens shown per report section",
            kind="int",
            default=30,
            min_value=5,
            max_value=100,
        ),
        ModuleParameter(
            name="min_seq_len",
            display_name="Min Prompt Length",
            description="Skip prompts shorter than this many tokens",
            kind="int",
            default=3,
            min_value=1,
            max_value=20,
        ),
        ModuleParameter(
            name="min_per_cat",
            display_name="Min Per Category",
            description="Minimum appearances per category for pairwise comparisons",
            kind="int",
            default=2,
            min_value=1,
            max_value=10,
        ),
    ]

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        prompts = session.get("prompts") or []

        merge = params.get("merge_subwords", False)
        min_app = params.get("min_appearances", 3)
        include_first = params.get("include_first", False)
        top_n = params.get("top_n", 30)
        min_seq_len = params.get("min_seq_len", 3)
        min_per_cat = params.get("min_per_cat", 2)

        token_contexts, skipped = _extract_token_contexts(
            prompts, min_seq_len=min_seq_len, merge_subwords=merge)

        results = _analyze(token_contexts,
                           min_appearances=min_app,
                           exclude_first=not include_first)

        return _build_output(results, prompts, skipped, top_n,
                             min_app=min_app, min_per_cat=min_per_cat)


# ─── Data extraction (reads TAGM native schema) ──────────────────

def _extract_token_contexts(prompts, min_seq_len=3, merge_subwords=False):
    """Extract per-token channel measurements across all prompts.

    Reads from TAGM's nested measurement schema:
      prompt.measurements.<measurement_name>.per_token.<field>
    """
    token_contexts = defaultdict(list)
    skipped = 0

    for d in prompts:
        tokens = d.get("tokens")
        if not tokens:
            skipped += 1
            continue

        ms = d.get("measurements") or {}

        # Extract per-token arrays from TAGM nested structure
        pts = ((ms.get("stress_score") or {}).get("per_token") or {}).get("stress")
        ptd = ((ms.get("spectral_field_density") or {}).get("per_token") or {}).get("density")
        sa = ((ms.get("last_position_attribution") or {}).get("per_token") or {}).get("signed_attribution_to_last")

        if not pts or not ptd or not sa:
            skipped += 1
            continue

        seq_len = d.get("seq_len", len(tokens))
        if seq_len < min_seq_len:
            continue

        category = d.get("category", "")
        prompt = (d.get("prompt") or "")[:60]

        if merge_subwords:
            channel_lists = {
                "density": ptd, "stress": pts, "attr": sa,
            }
            m_tokens, m_channels, m_positions = _merge_subword_tokens(
                tokens, channel_lists)
            n_merged = len(m_tokens)
            for j, tok in enumerate(m_tokens):
                orig_pos = m_positions[j]
                token_contexts[tok].append({
                    "stress": m_channels["stress"][j],
                    "density": m_channels["density"][j],
                    "attr": m_channels["attr"][j],
                    "pos_frac": orig_pos / max(seq_len - 1, 1),
                    "is_first": (orig_pos == 0),
                    "is_last": (j == n_merged - 1),
                    "category": category,
                    "prompt": prompt,
                })
        else:
            for i, tok in enumerate(tokens):
                if i >= len(pts) or i >= len(ptd):
                    continue
                token_contexts[tok.strip()].append({
                    "stress": pts[i],
                    "density": ptd[i],
                    "attr": sa[i] if i < len(sa) else 0,
                    "pos_frac": i / max(seq_len - 1, 1),
                    "is_first": (i == 0),
                    "is_last": (i == len(tokens) - 1),
                    "category": category,
                    "prompt": prompt,
                })

    return token_contexts, skipped


# ─── Computation (unchanged from TASM) ────────────────────────────

def _merge_subword_tokens(tokens, channel_lists):
    """Merge BPE subword tokens into whole words."""
    if not tokens:
        return [], {ch: [] for ch in channel_lists}, []

    words = []
    for i, tok in enumerate(tokens):
        valid = all(i < len(v) for v in channel_lists.values())
        if not valid:
            continue
        if i == 0 or tok.startswith(" ") or tok.startswith("\n"):
            words.append({
                "text": tok, "first_pos": i, "last_pos": i,
                "values": {ch: [v[i]] for ch, v in channel_lists.items()},
            })
        else:
            if words:
                words[-1]["text"] += tok
                words[-1]["last_pos"] = i
                for ch in channel_lists:
                    words[-1]["values"][ch].append(channel_lists[ch][i])
            else:
                words.append({
                    "text": tok, "first_pos": i, "last_pos": i,
                    "values": {ch: [v[i]] for ch, v in channel_lists.items()},
                })

    merged_tokens = [w["text"].strip() for w in words]
    merged_channels = {
        ch: [float(np.mean(w["values"][ch])) for w in words]
        for ch in channel_lists
    }
    merged_positions = [w["first_pos"] for w in words]
    return merged_tokens, merged_channels, merged_positions


def _safe_cv(values):
    m = np.mean(values)
    if abs(m) < 1e-12:
        return 0.0
    return float(np.std(values) / abs(m))


def _channel_stats(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "cv": _safe_cv(arr),
        "range": float(np.max(arr) - np.min(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _eta_squared(values_by_group):
    groups = [np.array(v) for v in values_by_group.values() if len(v) >= 1]
    if len(groups) < 2:
        return 0.0
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    if ss_total < 1e-15:
        return 0.0
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    return float(ss_between / ss_total)


def _channel_correlations(context_arrays):
    corrs = {}
    for ch_a, ch_b in combinations(CHANNELS, 2):
        a, b = context_arrays[ch_a], context_arrays[ch_b]
        if len(a) < 3:
            corrs[f"{ch_a}__{ch_b}"] = None
            continue
        std_a, std_b = np.std(a), np.std(b)
        if std_a < 1e-12 or std_b < 1e-12:
            corrs[f"{ch_a}__{ch_b}"] = 0.0
        else:
            corrs[f"{ch_a}__{ch_b}"] = float(np.corrcoef(a, b)[0, 1])
    return corrs


def _analyze(token_contexts, min_appearances=3, exclude_first=True):
    results = []
    for tok, contexts in token_contexts.items():
        if exclude_first:
            contexts = [c for c in contexts if not c["is_first"]]
        if len(contexts) < min_appearances:
            continue

        arrays = {ch: np.array([c[ch] for c in contexts]) for ch in CHANNELS}
        cats = set(c["category"] for c in contexts if c["category"])
        channel_stats = {ch: _channel_stats(arrays[ch]) for ch in CHANNELS}

        by_cat = defaultdict(lambda: defaultdict(list))
        for c in contexts:
            if c["category"]:
                for ch in CHANNELS:
                    by_cat[c["category"]][ch].append(c[ch])

        cat_profiles = {}
        for cat, ch_lists in by_cat.items():
            cat_profiles[cat] = {
                ch: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                for ch, v in ch_lists.items()
            }

        eta_sq = {}
        for ch in CHANNELS:
            groups = {cat: ch_lists[ch] for cat, ch_lists in by_cat.items()
                      if ch in ch_lists}
            eta_sq[ch] = _eta_squared(groups)

        correlations = _channel_correlations(arrays)

        results.append({
            "token": tok,
            "n": len(contexts),
            "n_cats": len(cats),
            "cats": sorted(cats),
            "channels": channel_stats,
            "cat_profiles": cat_profiles,
            "eta_squared": eta_sq,
            "correlations": correlations,
        })

    return results


def _pairwise_deltas(results, cat_a, cat_b, channel="density",
                     min_per_cat=2, top_n=None):
    pairs = []
    for r in results:
        cp = r["cat_profiles"]
        if cat_a in cp and cat_b in cp:
            if channel in cp[cat_a] and channel in cp[cat_b]:
                pa, pb = cp[cat_a][channel], cp[cat_b][channel]
                if pa["n"] >= min_per_cat and pb["n"] >= min_per_cat:
                    pairs.append({
                        "token": r["token"],
                        "mean_a": pa["mean"], "n_a": pa["n"],
                        "mean_b": pb["mean"], "n_b": pb["n"],
                        "diff": pb["mean"] - pa["mean"],
                    })
    pairs.sort(key=lambda x: x["diff"], reverse=True)
    if top_n and len(pairs) > top_n * 2:
        return pairs[:top_n] + pairs[-top_n:]
    return pairs


# ─── Output (TASM shape, unchanged) ──────────────────────────────

def _build_output(results, prompts, skipped, top_n,
                  min_app=3, min_per_cat=2):
    """Build the structured output dict matching TASM's shape."""

    cats = set()
    for d in prompts:
        c = d.get("category", "")
        if c:
            cats.add(c)

    by_cv_desc = sorted(results, key=lambda x: x["channels"]["density"]["cv"],
                        reverse=True)
    by_cv_asc = sorted(results, key=lambda x: x["channels"]["density"]["cv"])

    qualified = [r for r in results if r["n"] >= min_app]

    summary = {
        "n_prompts": len(prompts),
        "n_skipped": skipped,
        "n_tokens_analyzed": len(results),
        "n_qualified": len(qualified),
        "qualified_threshold": min_app,
        "categories": sorted(cats),
    }

    if qualified:
        for ch in CHANNELS:
            cvs = [r["channels"][ch]["cv"] for r in qualified]
            etas = [r["eta_squared"][ch] for r in qualified]
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
            p25 = float(np.percentile(den_cvs, 25))
            p75 = float(np.percentile(den_cvs, 75))
            stable = [r["token"] for r in qualified
                      if r["channels"]["density"]["cv"] <= p25]
            dependent = [r["token"] for r in qualified
                         if r["channels"]["density"]["cv"] >= p75]
            summary["context_stable"] = {
                "threshold": p25,
                "tokens": stable[:top_n],
                "count": len(stable),
            }
            summary["context_dependent"] = {
                "threshold": p75,
                "tokens": dependent[:top_n],
                "count": len(dependent),
            }

    cat_pairs = [
        ("benign", "jailbreak"), ("benign", "adversarial"),
        ("benign", "harmful"), ("harmful", "jailbreak"),
    ]
    pairwise = {}
    for cat_a, cat_b in cat_pairs:
        deltas = _pairwise_deltas(results, cat_a, cat_b,
                                  min_per_cat=min_per_cat, top_n=top_n)
        if deltas:
            pairwise[f"{cat_a}_vs_{cat_b}"] = deltas

    def _round_dict(d, decimals=6):
        if isinstance(d, dict):
            return {k: _round_dict(v, decimals) for k, v in d.items()}
        if isinstance(d, list):
            return [_round_dict(v, decimals) for v in d]
        if isinstance(d, float):
            return round(d, decimals)
        return d

    return _round_dict({
        "summary": summary,
        "highest_cv": [_token_row(r) for r in by_cv_desc[:top_n]],
        "lowest_cv": [_token_row(r) for r in by_cv_asc[:top_n]],
        "high_eta": [_token_row(r) for r in
                     sorted([r for r in results if r["n_cats"] >= 2 and r["n"] >= min_app],
                            key=lambda x: x["eta_squared"]["density"],
                            reverse=True)[:top_n]],
        "cross_category": [_token_row(r) for r in
                           sorted([r for r in results if r["n_cats"] >= 3 and r["n"] >= min_app],
                                  key=lambda x: x["channels"]["density"]["cv"],
                                  reverse=True)[:top_n]],
        "pairwise": pairwise,
        "all_tokens": [_token_row(r) for r in by_cv_desc],
    })


def _token_row(r):
    """Flatten a result entry for output."""
    row = {
        "token": r["token"],
        "n": r["n"],
        "n_cats": r["n_cats"],
        "cats": r["cats"] if isinstance(r["cats"], list) else sorted(r["cats"]),
    }
    for ch in CHANNELS:
        for stat in ("mean", "std", "cv", "range", "min", "max"):
            row[f"{ch}_{stat}"] = r["channels"][ch][stat]
    for ch in CHANNELS:
        row[f"eta_sq_{ch}"] = r["eta_squared"][ch]
    for cat in CAT_ORDER:
        if cat in r["cat_profiles"]:
            for ch in CHANNELS:
                if ch in r["cat_profiles"][cat]:
                    row[f"{ch}_{cat}_mean"] = r["cat_profiles"][cat][ch]["mean"]
                    row[f"{ch}_{cat}_n"] = r["cat_profiles"][cat][ch]["n"]
    return row
