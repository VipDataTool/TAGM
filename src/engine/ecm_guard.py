"""Cooling-gated no-repeat n-gram guard for ECM processors.

Replaces the HuggingFace `no_repeat_ngram_size` generate kwarg on the
ECM path. That off-the-shelf constraint was the wrong scope on both
axes for the job §3.3 gives it ("stop the sampler from completing
repetition loops seeded during cooled steps"):

  1. It was UNCONDITIONAL — it policed every step, including fully
     warm, zero-intervention steps where no loop could have been
     seeded by cooling.
  2. It was SEQUENCE-GLOBAL — its n-gram window included the PROMPT,
     so a response that naturally echoes four words of the question
     ("tallest mountain in the...") had its natural continuation
     banned. Discovered 2026-07-10 in a seed-matched pair: the benign
     control and ECM responses diverged at token #4 with ZERO
     interventions, breaking the causal attribution that Table 2 of
     the working paper relies on.

This guard is scoped exactly to the design intent:

  - ARMED only while the processor is actually cooling (T_eff below
    base) and for `ngram_size` steps afterward — long enough for a
    loop seeded at the tail of a cooled run to try to complete.
  - GENERATED TOKENS ONLY — the prompt is excluded from the n-gram
    window, so prompt echoes are never banned.
  - QUIET PATH IS A NO-OP by construction: a generation with zero
    interventions applies zero bans, making the quiet ECM pipeline
    behaviorally identical to the plain control.

Batch note: like the rest of the ECM actuation path, this operates on
batch row 0 (chat/harvest run batch=1 — see the batch note in ecm.py's
module docstring before reusing in any batched path).
"""
from __future__ import annotations


def banned_next_tokens(gen_ids: list, n: int) -> list:
    """Token ids banned as the next token under no-repeat-n-gram
    semantics, computed over generated ids only.

    Mirrors HF's NoRepeatNGramLogitsProcessor: the next token c is
    banned iff (last n-1 generated tokens) + c already occurs as an
    n-gram in gen_ids.
    """
    if n < 2 or len(gen_ids) < n:
        return []
    prefix = tuple(gen_ids[-(n - 1):])
    banned = set()
    for i in range(len(gen_ids) - n + 1):
        if tuple(gen_ids[i:i + n - 1]) == prefix:
            banned.add(gen_ids[i + n - 1])
    return sorted(banned)


class CoolingGatedNoRepeat:
    """Per-generation guard state. Construct (or reset) once per
    generation; call observe() every step AFTER the effective
    temperature is known, then read `armed` / banned().
    """

    def __init__(self, ngram_size: int = 0, prompt_len: int = 0):
        self.n = int(ngram_size or 0)
        self.prompt_len = max(0, int(prompt_len or 0))
        self.steps_since_cooled = None   # None = never cooled yet
        self.n_bans = 0                  # total tokens banned (diagnostics)

    def reset(self):
        self.steps_since_cooled = None
        self.n_bans = 0

    def observe(self, temp_effective: float, base_temperature: float):
        """Advance arming state with this step's effective temperature."""
        if temp_effective < base_temperature - 1e-9:
            self.steps_since_cooled = 0
        elif self.steps_since_cooled is not None:
            self.steps_since_cooled += 1

    @property
    def armed(self) -> bool:
        return (self.n >= 2
                and self.steps_since_cooled is not None
                and self.steps_since_cooled <= self.n)

    def banned(self, full_ids: list) -> list:
        """Banned next-token ids for this step, or [] when disarmed.

        full_ids is the whole sequence (prompt + generated) for the
        batch row; the prompt prefix is excluded here.
        """
        if not self.armed:
            return []
        out = banned_next_tokens(list(full_ids[self.prompt_len:]), self.n)
        self.n_bans += len(out)
        return out
