"""Token Pair Coupling — Strongly Coupled Prompt–Counterfactual Token Pairs.

At each position in a prompt, the actual token is T.  The instruct and
base models each predict a ranked set of counterfactual candidates — what
they think *should* be at that position.  When a model assigns high
probability to a candidate C != T, the pair (T -> C) is a strong
interaction: the model wants to redirect that position.

This module extracts all (prompt_token, counterfactual) pairs above a
configurable threshold from both model variants, records them with full
context, and accumulates observations in a persistent JSON cache across
sessions.  Over many sessions, recurring pairs reveal where and how
models consistently disagree with prompt content.

Data source: reads only from the session result dicts — specifically
ltp.counterfactual_tokens (instruct) and base_counterfactual_tokens
(base).  No cross-module dependencies.

Cache location: ~/.tagm/token_pair_cache.json
"""
from __future__ import annotations

import json
import math
import time
import logging
from pathlib import Path
from typing import Any, Optional, Callable

from src.engine.modules.base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")

_CACHE_PATH = Path.home() / ".tagm" / "token_pair_cache.json"


class TokenPairCoupling(TASMModule):
    name = "token_pair_coupling"
    display_name = "Token Pair Coupling"
    version = "0.2.0"
    description = (
        "Identifies strongly coupled (prompt token -> counterfactual) pairs "
        "where models predict a different token at a given position with "
        "high probability. Observations accumulate in a persistent cache "
        "across sessions."
    )

    parameters = [
        ModuleParameter(
            name="min_interaction_score",
            display_name="Minimum interaction score",
            description=(
                "Minimum probability for a counterfactual candidate to be "
                "recorded as a coupled pair. A candidate at probability 0.10 "
                "means the model assigns 10% chance to that token instead of "
                "the actual prompt token at that position."
            ),
            type="float", default=0.04, min_val=0.001, max_val=0.5,
        ),
        ModuleParameter(
            name="top_k_depth",
            display_name="Top-k depth",
            description=(
                "How deep into each model's ranked candidate list to look "
                "for interactions.  Must not exceed the k used during analysis."
            ),
            type="int", default=8, min_val=3, max_val=20,
        ),
        ModuleParameter(
            name="position_bins",
            display_name="Position bins",
            description=(
                "Number of bins for discretizing token position within the "
                "prompt (early / mid / late).  Used for aggregation."
            ),
            type="int", default=3, min_val=2, max_val=10,
        ),
    ]

    def __init__(self):
        super().__init__()
        self._cache = None
        self._model_id = None

    def set_pipeline(self, pipeline) -> None:
        """Record the current model ID for tagging observations."""
        try:
            self._model_id = pipeline.instruct_model_id
        except Exception:
            self._model_id = None

    # ── Cache I/O ───────────────────────────────────────────────────

    def _load_cache(self) -> list:
        if self._cache is not None:
            return self._cache
        if _CACHE_PATH.exists():
            try:
                with open(_CACHE_PATH, "r") as f:
                    self._cache = json.load(f)
                    if not isinstance(self._cache, list):
                        self._cache = []
            except Exception as e:
                logger.warning(f"[TOKEN_PAIR] Cache load failed: {e}")
                self._cache = []
        else:
            self._cache = []
        return self._cache

    def _save_cache(self, cache: list) -> None:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=1)
        self._cache = cache

    def _get_cache_summary(self) -> dict:
        cache = self._load_cache()
        if not cache:
            return {
                "n_observations": 0, "n_unique_pairs": 0,
                "n_sessions": 0, "models": [], "categories": [],
            }
        pairs = set()
        sessions = set()
        models = set()
        categories = set()
        for obs in cache:
            pairs.add((obs.get("prompt_token"), obs.get("counterfactual")))
            sessions.add(obs.get("session_id", ""))
            models.add(obs.get("model", ""))
            categories.add(obs.get("category", ""))
        return {
            "n_observations": len(cache),
            "n_unique_pairs": len(pairs),
            "n_sessions": len(sessions),
            "models": sorted(models - {""}),
            "categories": sorted(categories - {""}),
        }

    @staticmethod
    def reset_cache() -> dict:
        if _CACHE_PATH.exists():
            try:
                with open(_CACHE_PATH, "r") as f:
                    old = json.load(f)
                n = len(old) if isinstance(old, list) else 0
            except Exception:
                n = 0
            _CACHE_PATH.unlink()
            return {"cleared": True, "observations_removed": n}
        return {"cleared": False, "observations_removed": 0}

    # ── Core computation ────────────────────────────────────────────

    def run(self, session_results: list[dict], params: dict,
            progress: Optional[Callable] = None) -> dict:
        """Extract (prompt_token -> counterfactual) pairs from session results.

        For each position in each prompt, looks at both the instruct and
        base model's counterfactual candidate lists.  Any candidate with
        probability >= min_interaction_score is recorded as a coupled pair
        with the actual prompt token at that position.
        """
        min_score = params.get("min_interaction_score", 0.04)
        k_depth = params.get("top_k_depth", 8)
        n_bins = params.get("position_bins", 3)

        cache = self._load_cache()
        session_id = f"s_{int(time.time())}"
        new_observations = []
        n_processed = 0
        n_skipped = 0

        for ri, result in enumerate(session_results):
            if progress and ri % 10 == 0:
                progress(f"Processing {ri + 1}/{len(session_results)}...")

            # Extract per-position counterfactual tokens
            ltp = result.get("ltp")
            instruct_cf = ltp.get("counterfactual_tokens", []) if isinstance(ltp, dict) else []
            base_cf = result.get("base_counterfactual_tokens", [])

            if not instruct_cf and not base_cf:
                n_skipped += 1
                continue

            prompt = result.get("prompt", "")
            category = result.get("category", "")
            model = self._model_id or result.get("model", "")
            tokens = result.get("tokens", [])

            # Process both model variants
            cf_sources = []
            if instruct_cf:
                cf_sources.append(("instruct", instruct_cf))
            if base_cf:
                cf_sources.append(("base", base_cf))

            for variant, cf_list in cf_sources:
                n_positions = min(len(cf_list), len(tokens)) if tokens else len(cf_list)

                for pos in range(n_positions):
                    candidates = cf_list[pos][:k_depth]
                    if not candidates:
                        continue

                    # The actual prompt token at this position
                    prompt_token = tokens[pos] if pos < len(tokens) else ""
                    if not prompt_token:
                        continue

                    # Position bin
                    pos_bin = min(int(pos / max(n_positions, 1) * n_bins), n_bins - 1)
                    pos_label = ["early", "mid", "late"][pos_bin] if n_bins == 3 else f"bin_{pos_bin}"

                    # Record all candidates above threshold
                    for cf_token, cf_prob in candidates:
                        if cf_prob < min_score:
                            continue
                        if cf_token == prompt_token:
                            continue  # skip self-coupling

                        new_observations.append({
                            "prompt_token": prompt_token,
                            "counterfactual": cf_token,
                            "score": round(cf_prob, 6),
                            "variant": variant,
                            "position": pos,
                            "position_bin": pos_label,
                            "prompt": prompt[:200],
                            "category": category,
                            "model": model,
                            "session_id": session_id,
                            "timestamp": time.time(),
                        })

            n_processed += 1

        # Deduplicate against existing cache
        existing_keys = set()
        for obs in cache:
            k = (obs.get("prompt_token"), obs.get("counterfactual"),
                 obs.get("position"), obs.get("variant"),
                 obs.get("prompt", "")[:80], obs.get("model"))
            existing_keys.add(k)

        n_dupes = 0
        for obs in new_observations:
            k = (obs["prompt_token"], obs["counterfactual"],
                 obs["position"], obs["variant"],
                 obs["prompt"][:80], obs["model"])
            if k in existing_keys:
                n_dupes += 1
                continue
            existing_keys.add(k)
            cache.append(obs)

        self._save_cache(cache)

        if progress:
            progress(f"Done: {len(new_observations) - n_dupes} new pairs "
                     f"from {n_processed} prompts "
                     f"({n_dupes} duplicates skipped)")

        # ── Aggregate results ───────────────────────────────────────

        pair_agg = {}
        for obs in cache:
            key = (obs["prompt_token"], obs["counterfactual"])
            if key not in pair_agg:
                pair_agg[key] = {
                    "prompt_token": obs["prompt_token"],
                    "counterfactual": obs["counterfactual"],
                    "count": 0,
                    "scores": [],
                    "categories": set(),
                    "models": set(),
                    "variants": set(),
                    "position_bins": [],
                    "prompts": [],
                }
            entry = pair_agg[key]
            entry["count"] += 1
            entry["scores"].append(obs["score"])
            entry["categories"].add(obs["category"])
            entry["models"].add(obs["model"])
            entry["variants"].add(obs["variant"])
            entry["position_bins"].append(obs["position_bin"])
            if len(entry["prompts"]) < 5:
                entry["prompts"].append(obs["prompt"][:100])

        pair_table = []
        for key, entry in pair_agg.items():
            scores = entry["scores"]
            pair_table.append({
                "prompt_token": entry["prompt_token"],
                "counterfactual": entry["counterfactual"],
                "count": entry["count"],
                "mean_score": round(sum(scores) / len(scores), 6),
                "max_score": round(max(scores), 6),
                "categories": sorted(entry["categories"]),
                "models": sorted(entry["models"]),
                "variants": sorted(entry["variants"]),
                "position_tendency": _mode(entry["position_bins"]),
                "example_prompts": entry["prompts"],
            })

        pair_table.sort(key=lambda x: (-x["count"], -x["mean_score"]))

        cache_summary = self._get_cache_summary()

        return {
            "session_pairs_found": len(new_observations),
            "session_pairs_added": len(new_observations) - n_dupes,
            "duplicates_skipped": n_dupes,
            "prompts_processed": n_processed,
            "prompts_skipped": n_skipped,
            "skip_reason": "missing LTP or base counterfactual data" if n_skipped else None,
            "cache_summary": cache_summary,
            "top_pairs": pair_table[:50],
            "all_pairs_count": len(pair_table),
            "parameters": {
                "min_interaction_score": min_score,
                "top_k_depth": k_depth,
                "position_bins": n_bins,
            },
        }


def _mode(values: list) -> str:
    if not values:
        return ""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)
