"""Prompt deconstruction: expand a prompt into its prefix ladder.

A "deconstructed" prompt — ``I like cake.`` — becomes an ordered ladder
of prefix-prompts:

    I
    I like
    I like cake
    I like cake.

Each rung is a genuine prompt the inference pipeline analyzes on its
own (its own forward pass, its own result record). Reading the ladder
back later shows how the prompt's contextual field develops one unit at
a time against the fixed probe lattice.

Two policy decisions are baked in here:

1. **Punctuation runs earn their own rung.** The period in ``I like
   cake.`` is rung 3, not glued onto ``cake``. Punctuation shapes how
   the model builds context, so collapsing it into the preceding word
   would hide that step. A *run* of punctuation (``...``) is one rung,
   not three.

2. **Rungs are literal prefixes of the original string.** We find cut
   points at unit boundaries and take ``text[:cut]`` — we never rebuild
   the string from pieces. So the final rung is byte-identical to the
   input and the model never sees a re-spaced or re-tokenized variant
   that would make us measure a tokenization artifact instead of the
   real effect.

Intra-word apostrophes and hyphens stay attached to their word
(``don't`` and ``well-being`` are each one unit), so we don't
manufacture noise rungs like ``don'``.
"""
from __future__ import annotations

import re

# A unit is either:
#   \w[\w'-]*   a word: a word-char followed by word-chars, intra-word
#               apostrophes, or intra-word hyphens (keeps don't, well-being whole)
#   [^\w\s]+    a punctuation run: one or more non-word, non-space chars
# Whitespace is a separator and never begins a unit, so prefixes cut at
# a unit's end naturally exclude any trailing space.
_UNIT = re.compile(r"\w[\w'\-]*|[^\w\s]+", re.UNICODE)


def segment_prompt(text: str) -> list[str]:
    """Return the prefix ladder for ``text`` as a list of strings.

    Each rung is ``text[:cut]`` for successive unit-boundary cut points.
    ``rungs[-1]`` equals ``text`` up to any trailing whitespace. Returns
    an empty list for text with no units (empty / whitespace-only); the
    caller decides how to handle that.
    """
    if not text:
        return []
    cuts = [m.end() for m in _UNIT.finditer(text)]
    return [text[:c] for c in cuts]


def expand_prompts(prompts: list[dict], enabled: bool) -> list[dict]:
    """Expand a list of prompt-dicts into rung records.

    ``prompts`` is the list the analyze path already builds: dicts with
    at least ``prompt`` and ``category``. When ``enabled`` is False this
    is a no-op (returns the list unchanged) so the off path costs
    nothing.

    When enabled, each source prompt expands into its ladder. Every rung
    inherits the source dict's other fields (category, etc.), and gains:

        rung_index    0-based position within the ladder
        family_local  the source prompt's position in THIS call

    ``family_local`` is provisional — the ingest path turns it into the
    session-stable ``family_index`` by adding a per-session base, so
    families never collide across separate analyze calls. A source
    prompt with no units (empty/whitespace) is preserved as a single
    rung so it isn't silently dropped.
    """
    if not enabled:
        return list(prompts)

    out: list[dict] = []
    for src_i, p in enumerate(prompts):
        text = p.get("prompt") or ""
        rungs = segment_prompt(text) or [text]
        for rung_i, rung_text in enumerate(rungs):
            q = dict(p)
            q["prompt"] = rung_text
            q["rung_index"] = rung_i
            q["family_local"] = src_i
            out.append(q)
    return out
