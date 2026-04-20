"""TokenVariance: cross-context per-token coupling stability.

For each token string that appears in at least min_appearances prompts,
computes the coefficient of variation (CV) of its per-token density,
stress, and attribution values across its occurrences. High CV means
context-dependent coupling (function words, pronouns); low CV means
stable coupling (content verbs, nouns).

Translated from TASM's `engine/modules/token_variance.py`. Output
table has per-token stats plus per-category aggregation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from tagm.analysis.base import AnalysisModule, AnalysisResult
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

# Channels this analysis tracks. Each maps to (measurement_name, per_token_field).
_CHANNELS = {
    "density": ("spectral_field_density", "density"),
    "stress": ("stress_score", "stress"),
    "attr": ("last_position_attribution", "signed_attribution_to_last"),
}


@register_analysis
class TokenVariance(AnalysisModule):
    name = "token_variance"
    display_name = "Token Variance"
    description = (
        "Cross-context stability of per-token coupling measures (density, "
        "stress, attribution). Separates context-dependent from context-stable "
        "tokens and computes per-category profiles."
    )
    version = "0.1.0"

    depends_on_measurements = (
        "stress_score", "last_position_attribution",
    )

    parameters = [
        ModuleParameter(
            name="min_appearances",
            display_name="Minimum appearances",
            description="Tokens with fewer occurrences are excluded.",
            kind="int", default=3, min_value=2, max_value=50,
        ),
        ModuleParameter(
            name="include_first_token",
            display_name="Include position-0 tokens",
            description="Position 0 often shows positional artifacts unrelated to content.",
            kind="bool", default=False, advanced=True,
        ),
        ModuleParameter(
            name="top_n",
            display_name="Top-N per report",
            description="Maximum tokens shown per ranked list.",
            kind="int", default=30, min_value=5, max_value=200,
        ),
    ]

    def run(self, session, params, probes=None):
        min_appearances = int(params.get("min_appearances", 3))
        include_first = bool(params.get("include_first_token", False))
        top_n = int(params.get("top_n", 30))

        result = AnalysisResult(
            analysis_name=self.name,
            analysis_version=self.version,
            parameters={"min_appearances": min_appearances,
                        "include_first_token": include_first,
                        "top_n": top_n},
        )

        prompts = session.get("prompts") or []
        if not prompts:
            return result

        # token_text -> channel -> list of (prompt_idx, position, category, value)
        token_obs: dict[str, dict[str, list]] = defaultdict(
            lambda: {ch: [] for ch in _CHANNELS})

        for prompt_idx, p in enumerate(prompts):
            tokens = p.get("tokens") or []
            category = p.get("category") or "uncategorized"
            if not tokens:
                continue
            ms = p.get("measurements") or {}

            channel_arrays: dict[str, list] = {}
            for ch, (mname, field) in _CHANNELS.items():
                mr = ms.get(mname) or {}
                arr = (mr.get("per_token") or {}).get(field) or []
                channel_arrays[ch] = arr

            start = 0 if include_first else 1
            for pos in range(start, len(tokens)):
                tok = tokens[pos]
                if not tok.strip():
                    continue
                for ch, arr in channel_arrays.items():
                    if pos < len(arr) and arr[pos] is not None:
                        try:
                            v = float(arr[pos])
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(v) or np.isinf(v):
                            continue
                        token_obs[tok][ch].append({
                            "prompt_idx": prompt_idx,
                            "position": pos,
                            "category": category,
                            "value": v,
                        })

        # Compute per-token stats
        per_token_stats: list[dict[str, Any]] = []
        for tok, ch_obs in token_obs.items():
            n_app = max(len(obs) for obs in ch_obs.values())
            if n_app < min_appearances:
                continue
            entry: dict[str, Any] = {"token": tok, "n_appearances": n_app}
            for ch, obs in ch_obs.items():
                vals = np.array([o["value"] for o in obs], dtype=float)
                if len(vals) < 2:
                    entry[f"{ch}_mean"] = float(vals.mean()) if len(vals) else 0.0
                    entry[f"{ch}_std"] = 0.0
                    entry[f"{ch}_cv"] = 0.0
                    continue
                mean = float(vals.mean())
                std = float(vals.std(ddof=1))
                entry[f"{ch}_mean"] = mean
                entry[f"{ch}_std"] = std
                entry[f"{ch}_cv"] = std / abs(mean) if mean != 0 else 0.0
            per_token_stats.append(entry)

        # Rankings
        def _top(key, reverse=True):
            sorted_list = sorted(
                per_token_stats, key=lambda t: abs(t.get(key, 0)),
                reverse=reverse)
            return sorted_list[:top_n]

        rankings = {
            "most_stable_density": _top("density_cv", reverse=False),
            "most_variable_density": _top("density_cv", reverse=True),
            "most_stable_stress": _top("stress_cv", reverse=False),
            "most_variable_stress": _top("stress_cv", reverse=True),
            "highest_mean_attr": _top("attr_mean", reverse=True),
            "lowest_mean_attr": _top("attr_mean", reverse=False),
        }

        result.objects["per_token_stats"] = per_token_stats
        result.objects["rankings"] = rankings
        result.scalars["n_tokens_analyzed"] = len(per_token_stats)
        result.scalars["n_tokens_observed"] = len(token_obs)
        result.scalars["n_prompts"] = len(prompts)
        return result
