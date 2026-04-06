"""
Probe Generator Module for TASM.

Expands a seed probe file into a dense probe set by using the loaded
instruct model to generate domain-specific vocabulary.  For each
subject × grammatical level cell, prompts the model to list words,
filters for single-token results (no BPE splits), deduplicates, and
exports a new probe CSV.

Standalone utility module: does not require session results.
"""

import os
import csv
import json
import logging
import re

import numpy as np

from .base import TASMModule, ModuleParameter
from .domain_surface import _discover_probe_files, _detect_level_cols

logger = logging.getLogger("tasm")


# ─── Prompt templates per grammatical category ──────────────────

GRAMMAR_PROMPTS = {
    "nouns": "List {n} single-word nouns related to {subject}. One word per line. Only real English words. No phrases, no explanations.",
    "verbs": "List {n} single-word verbs related to {subject}. One word per line. Only real English words. No phrases, no explanations.",
    "adjectives": "List {n} single-word adjectives related to {subject}. One word per line. Only real English words. No phrases, no explanations.",
    "adverbs": "List {n} single-word adverbs related to {subject}. One word per line. Only real English words. No phrases, no explanations.",
}

# Fallback for non-grammar level names
DEFAULT_PROMPT = "List {n} single words that are {level} related to {subject}. One word per line. Only real English words. No phrases, no explanations."


def _extract_words(text):
    """Extract single words from model output, one per line."""
    words = []
    for line in text.strip().split("\n"):
        line = line.strip()
        # Strip numbering like "1. " or "- "
        line = re.sub(r"^[\d]+[.\)]\s*", "", line)
        line = re.sub(r"^[-•*]\s*", "", line)
        line = line.strip().lower()
        # Single word only, alphabetic
        if line and " " not in line and line.isalpha() and len(line) > 1:
            words.append(line)
    return words


def _is_single_token(word, tokenizer):
    """Check if a word tokenizes to exactly one token (excluding specials)."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    return len(ids) == 1


def _generate_words(model, tokenizer, subject, level, n_words=60,
                    max_new_tokens=512, device=None):
    """Generate a list of domain words using the model."""
    template = GRAMMAR_PROMPTS.get(level.lower(), DEFAULT_PROMPT)
    prompt_text = template.format(n=n_words, subject=subject.replace("_", " "),
                                  level=level)

    # Format as chat
    messages = [{"role": "user", "content": prompt_text}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
    except Exception:
        text = prompt_text

    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
        )

    # Decode only the generated part
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _extract_words(response)


class ProbeGeneratorModule(TASMModule):
    name = "probe_generator"
    display_name = "Probe Generator"
    description = (
        "Expands a seed probe file into a dense probe set using the "
        "loaded model. For each subject × grammatical category, generates "
        "domain vocabulary and filters for single-token words. Exports "
        "as a new probe CSV ready for embedding."
    )
    version = "0.1.0"

    min_results = 0  # Doesn't need session results
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    def __init__(self):
        super().__init__()
        self._model = None
        self._tokenizer = None
        self._probe_files = []

    def set_project_root(self, root):
        self._project_root = root
        self._probe_files = _discover_probe_files(root)

    def set_model(self, model, tokenizer):
        """Provide access to the loaded instruct model."""
        self._model = model
        self._tokenizer = tokenizer

    @property
    def parameters(self):
        options = self._probe_files if self._probe_files else ["probes_grammar.csv"]
        return [
            ModuleParameter(
                name="source_file",
                display_name="Source Probe File",
                description="Seed probe file to expand (provides subjects and levels)",
                type="select",
                default=options[0] if options else "probes_grammar.csv",
                options=options,
            ),
            ModuleParameter(
                name="words_per_cell",
                display_name="Words Per Cell",
                description="Target number of words to generate per subject × level",
                type="int",
                default=50,
                min_val=10,
                max_val=200,
            ),
            ModuleParameter(
                name="include_seeds",
                display_name="Include Seed Words",
                description="Include the original seed words from the source file",
                type="select",
                default="yes",
                options=["yes", "no"],
            ),
        ]

    def validate(self, session_results, params):
        if self._model is None or self._tokenizer is None:
            return False, "Model not loaded. Load a model first."
        source = params.get("source_file", "")
        if source and self._project_root:
            path = os.path.join(self._project_root, source)
            if not os.path.exists(path):
                return False, f"Source file not found: {source}"
        return True, "OK"

    def run(self, session_results, params, progress=None):
        source_file = params.get("source_file", "probes_grammar.csv")
        words_per_cell = int(params.get("words_per_cell", 50))
        include_seeds = params.get("include_seeds", "yes") == "yes"

        csv_path = os.path.join(self._project_root, source_file)
        level_cols, level_names = _detect_level_cols(csv_path)
        if not level_cols:
            raise RuntimeError(f"No level columns found in {source_file}")

        # Read source subjects and seed words
        subjects = []
        seed_words = {}  # {(subject, level_idx): [words]}
        anchor_ids = {}  # {subject: anchor_id_prefix}

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row.get("subject", "").strip()
                if not subj:
                    continue
                if subj not in subjects:
                    subjects.append(subj)
                aid = row.get("anchor_id", "").strip()
                if aid:
                    anchor_ids[subj] = aid.rsplit("_", 1)[0] if "_" in aid else aid

                for li, col in enumerate(level_cols):
                    text = row.get(col, "").strip()
                    if text:
                        words = [w.strip().lower() for w in text.split()
                                 if w.strip().isalpha()]
                        seed_words.setdefault((subj, li), []).extend(words)

        if progress:
            progress(f"Source: {len(subjects)} subjects × {len(level_cols)} levels")

        # Generate expanded vocabulary
        n_cells = len(subjects) * len(level_cols)
        cell_idx = 0
        expanded = {}  # {(subject, level_idx): [words]}

        for si, subj in enumerate(subjects):
            for li, col in enumerate(level_cols):
                cell_idx += 1
                key = (subj, li)
                level_name = level_names[li] if li < len(level_names) else col

                if progress:
                    progress(f"[{cell_idx}/{n_cells}] {subj} × {level_name}")

                # Generate
                try:
                    raw_words = _generate_words(
                        self._model, self._tokenizer,
                        subj, level_name,
                        n_words=words_per_cell + 20,  # request extra for filtering
                    )
                except Exception as e:
                    logger.warning(f"[PROBEGEN] Generation failed for {subj}/{level_name}: {e}")
                    raw_words = []

                # Filter: single-token only, deduplicate
                seen = set()
                filtered = []
                seeds = seed_words.get(key, [])

                # Include seeds first if requested
                if include_seeds:
                    for w in seeds:
                        wl = w.lower()
                        if wl not in seen and _is_single_token(wl, self._tokenizer):
                            filtered.append(wl)
                            seen.add(wl)

                # Add generated words
                for w in raw_words:
                    wl = w.lower()
                    if wl not in seen and _is_single_token(wl, self._tokenizer):
                        filtered.append(wl)
                        seen.add(wl)
                    if len(filtered) >= words_per_cell:
                        break

                expanded[key] = filtered

                if progress:
                    progress(f"[{cell_idx}/{n_cells}] {subj} × {level_name}: "
                             f"{len(raw_words)} generated, {len(filtered)} kept "
                             f"(single-token)")

        # Export CSV
        stem = os.path.splitext(source_file)[0]
        if stem.endswith("_probes"):
            stem = stem[:-7]  # strip "_probes"
        out_name = f"{stem}_auto_probes.csv"
        out_path = os.path.join(self._project_root, out_name)

        # Group by subject: one row per subject, cells contain space-separated words
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["subject", "anchor_id"] + level_cols
            writer.writerow(header)

            for subj in subjects:
                aid = anchor_ids.get(subj, subj[:4])
                # Split into multiple rows if too many words per cell
                # (probe loader creates one probe per row × level)
                max_words_per_row = 3  # matches original probe format
                all_cells = {li: expanded.get((subj, li), [])
                             for li in range(len(level_cols))}
                max_rows = max(1, max(
                    (len(v) + max_words_per_row - 1) // max_words_per_row
                    for v in all_cells.values()))

                for ri in range(max_rows):
                    row = [subj, f"{aid}_{ri+1:02d}"]
                    for li in range(len(level_cols)):
                        words = all_cells[li]
                        start = ri * max_words_per_row
                        end = start + max_words_per_row
                        chunk = words[start:end]
                        row.append(" ".join(chunk))
                    writer.writerow(row)

        # Summary stats
        total_words = sum(len(v) for v in expanded.values())
        single_token_pct = 100  # all are filtered to single-token

        output = {
            "source_file": source_file,
            "output_file": out_name,
            "subjects": subjects,
            "levels": level_names,
            "words_per_cell": {
                f"{subj}/{level_names[li]}": len(expanded.get((subj, li), []))
                for subj in subjects
                for li in range(len(level_cols))
            },
            "total_words": total_words,
            "total_probes": total_words // 3 + 1,  # approx rows
            "single_token_filter": True,
        }

        if progress:
            progress(f"Exported {out_name}: {total_words} words, "
                     f"{len(subjects)} subjects × {len(level_cols)} levels")

        return output
