"""Token Pair Coupling — Strongly Interacting Token Pairs in the Correction Field.

Identifies per-position token pairs where the alignment correction field
actively redirects probability mass: tokens the base model favored that
the instruct model suppresses (demoted), and tokens the instruct model
promotes that the base model didn't favor (promoted).  The pairs
(demoted → promoted) are the field's vocabulary-level action.

Observations accumulate in a persistent JSON cache across sessions,
models, and prompt categories.  Over many sessions, recurring pairs
separate from noise and compose into correction pathways — chains of
redirections that describe the field's strategy.

Data source: reads only from the session result dicts.  No cross-module
dependencies.  Requires LTP (for instruct counterfactuals) and base
model data (for base counterfactuals) to be present in the results.

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
    version = "0.1.0"
    description = (
        "Identifies strongly interacting token pairs from the correction "
        "field — tokens the base model favored that the instruct model "
        "suppresses, paired with tokens the instruct model promotes. "
        "Observations accumulate in a persistent cache across sessions."
    )

    parameters = [
        ModuleParameter(
            key="min_interaction_score",
            label="Minimum interaction score",
            description=(
                "Minimum geometric-mean probability for a (demoted, promoted) "
                "pair to be recorded.  Lower values capture weaker interactions "
                "but increase noise.  Score = sqrt(p_base(demoted) × p_instruct(promoted))."
            ),
            type="float", default=0.04, min=0.001, max=0.5,
        ),
        ModuleParameter(
            key="top_k_depth",
            label="Top-k depth",
            description=(
                "How deep into each model's ranked candidate list to look "
                "for interactions.  Must not exceed the k used during analysis."
            ),
            type="int", default=8, min=3, max=20,
        ),
        ModuleParameter(
            key="position_bins",
            label="Position bins",
            description=(
                "Number of bins for discretizing token position within the "
                "prompt (early / mid / late).  Used for aggregation."
            ),
            type="int", default=3, min=2, max=10,
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
        """Load the persistent cache from disk."""
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
        """Write the cache to disk."""
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=1)
        self._cache = cache

    def _get_cache_summary(self) -> dict:
        """Summary statistics for the current cache."""
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
            pairs.add((obs.get("source"), obs.get("dest")))
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
        """Clear the persistent cache.  Returns summary of what was cleared."""
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
        """Process session results and merge new observations into the cache.

        For each result that has both instruct and base counterfactual
        tokens, identifies (demoted → promoted) pairs at each position
        and records those exceeding the interaction threshold.
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

            if not instruct_cf or not base_cf:
                n_skipped += 1
                continue

            prompt = result.get("prompt", "")
            category = result.get("category", "")
            model = self._model_id or result.get("model", "")
            n_positions = min(len(instruct_cf), len(base_cf))
            tokens = result.get("tokens", [])

            for pos in range(n_positions):
                i_alts = instruct_cf[pos][:k_depth]
                b_alts = base_cf[pos][:k_depth]

                if not i_alts or not b_alts:
                    continue

                # Build token → probability maps
                inst_map = {t: p for t, p in i_alts}
                base_map = {t: p for t, p in b_alts}

                # Identify promoted and demoted tokens
                promoted = set(inst_map) - set(base_map)  # in instruct, not in base
                demoted = set(base_map) - set(inst_map)    # in base, not in instruct

                if not promoted or not demoted:
                    continue

                # Position bin (0 = early, n_bins-1 = late)
                pos_bin = min(int(pos / n_positions * n_bins), n_bins - 1)
                pos_label = ["early", "mid", "late"][pos_bin] if n_bins == 3 else f"bin_{pos_bin}"

                # Context token at this position
                ctx_token = tokens[pos] if pos < len(tokens) else ""

                # Generate all (demoted → promoted) pairs above threshold
                for src in demoted:
                    src_prob = base_map[src]
                    for dst in promoted:
                        dst_prob = inst_map[dst]
                        # Interaction score: geometric mean of the two probabilities
                        score = math.sqrt(src_prob * dst_prob)
                        if score < min_score:
                            continue

                        new_observations.append({
                            "source": src,
                            "dest": dst,
                            "score": round(score, 6),
                            "source_prob": round(src_prob, 6),
                            "dest_prob": round(dst_prob, 6),
                            "position": pos,
                            "position_bin": pos_label,
                            "context_token": ctx_token,
                            "prompt": prompt[:200],  # truncate for storage
                            "category": category,
                            "model": model,
                            "session_id": session_id,
                            "timestamp": time.time(),
                        })

            n_processed += 1

        # Merge into persistent cache (deduplicate)
        # Key: (source, dest, position, prompt_snippet, model)
        existing_keys = set()
        for obs in cache:
            k = (obs.get("source"), obs.get("dest"), obs.get("position"),
                 obs.get("prompt", "")[:80], obs.get("model"))
            existing_keys.add(k)

        n_dupes = 0
        for obs in new_observations:
            k = (obs["source"], obs["dest"], obs["position"],
                 obs["prompt"][:80], obs["model"])
            if k in existing_keys:
                n_dupes += 1
                continue
            existing_keys.add(k)
            cache.append(obs)

        self._save_cache(cache)

        if progress:
            progress(f"Done: {len(new_observations) - n_dupes} new pairs from {n_processed} prompts ({n_dupes} duplicates skipped)")

        # ── Build summary results ───────────────────────────────────

        # Aggregate pairs across all observations (including historical)
        pair_agg = {}
        for obs in cache:
            key = (obs["source"], obs["dest"])
            if key not in pair_agg:
                pair_agg[key] = {
                    "source": obs["source"],
                    "dest": obs["dest"],
                    "count": 0,
                    "scores": [],
                    "categories": set(),
                    "models": set(),
                    "position_bins": [],
                    "prompts": [],
                }
            entry = pair_agg[key]
            entry["count"] += 1
            entry["scores"].append(obs["score"])
            entry["categories"].add(obs["category"])
            entry["models"].add(obs["model"])
            entry["position_bins"].append(obs["position_bin"])
            if len(entry["prompts"]) < 5:  # keep up to 5 example prompts
                entry["prompts"].append(obs["prompt"][:100])

        # Build ranked pair table
        pair_table = []
        for key, entry in pair_agg.items():
            scores = entry["scores"]
            pair_table.append({
                "source": entry["source"],
                "dest": entry["dest"],
                "count": entry["count"],
                "mean_score": round(sum(scores) / len(scores), 6),
                "max_score": round(max(scores), 6),
                "categories": sorted(entry["categories"]),
                "models": sorted(entry["models"]),
                "position_tendency": _mode(entry["position_bins"]),
                "example_prompts": entry["prompts"],
            })

        pair_table.sort(key=lambda x: (-x["count"], -x["mean_score"]))

        # Cache summary
        cache_summary = self._get_cache_summary()

        return {
            "session_pairs_found": len(new_observations),
            "session_pairs_added": len(new_observations) - n_dupes,
            "duplicates_skipped": n_dupes,
            "prompts_processed": n_processed,
            "prompts_skipped": n_skipped,
            "skip_reason": "missing LTP or base counterfactual data" if n_skipped else None,
            "cache_summary": cache_summary,
            "top_pairs": pair_table[:50],  # top 50 for display
            "all_pairs_count": len(pair_table),
            "parameters": {
                "min_interaction_score": min_score,
                "top_k_depth": k_depth,
                "position_bins": n_bins,
            },
        }


def _mode(values: list) -> str:
    """Most common value in a list."""
    if not values:
        return ""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)
