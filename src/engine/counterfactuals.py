"""Counterfactual candidate extraction — single source of truth.

Every "top-k alternatives to the chosen token" computation in TAGM goes
through this module. Previously three call sites (LTP instruct alts,
base-phase alts, live behavioral comparison) each re-implemented the
extraction with ``softmax(topk(logits).values)`` — a softmax over only
the fetched candidates, which (a) inflated the reported probabilities
relative to the true full-vocabulary distribution and (b) normalized
the instruct side over k+1 candidates but the base side over k+5, so
rank-displacement mass comparisons were systematically biased.

Invariant enforced here:
    All probabilities attached to counterfactual tokens are
    FULL-VOCABULARY softmax values, computed in float32.

Fetching k+1 candidates is always sufficient: the chosen token can
appear at most once in the top-(k+1), so at least k non-chosen
alternatives remain (vocabulary size permitting). The old two-stage
overfetch fallback existed only to paper over the per-fetch
renormalization and is gone.
"""
from __future__ import annotations

from typing import List, Tuple

import torch


def full_softmax(logits_row: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary softmax of one logit row, in float32."""
    return torch.softmax(logits_row.float(), dim=-1)


def top_alternatives(logits_row: torch.Tensor, chosen_id: int,
                     k: int) -> List[Tuple[int, float]]:
    """Up to ``k`` (token_id, probability) alternatives to ``chosen_id``.

    Probabilities are true full-vocab softmax values (float32), so they
    are directly comparable across positions, across the instruct/base
    pair, and against ``instruct_topk`` / ``base_topk``.
    """
    probs = full_softmax(logits_row)
    n = min(k + 1, probs.shape[-1])
    topk = torch.topk(probs, n)
    alts: List[Tuple[int, float]] = []
    for tid, p in zip(topk.indices.tolist(), topk.values.tolist()):
        if tid != chosen_id and len(alts) < k:
            alts.append((tid, float(p)))
    return alts


def decode_alternatives(alts: List[Tuple[int, float]], tokenizer,
                        precision: int) -> List[Tuple[str, float]]:
    """Render (token_id, prob) pairs as (decoded_string, rounded_prob)."""
    return [(tokenizer.decode(tid).strip(), round(p, precision))
            for tid, p in alts]
