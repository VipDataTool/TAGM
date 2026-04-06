"""
Probe Generator Module for TASM.

Generates a discriminative probe set by statistically sampling the
loaded model's output distribution per subject domain.

Process:
  1. Read subjects and seed terms from a source probe CSV
  2. For each subject, query the model N times using seed context
  3. Tokenize all responses, count per-subject token frequencies
  4. Remove tokens that appear in more than one subject
  5. Export as *_auto_probes.csv

The cross-subject deduplication is the only filter.  Tokens shared
between domains carry no discriminative signal and are removed.
What survives is the model-specific vocabulary fingerprint for each
domain.
"""

import os
import csv
import json
import logging
import re
import time
from collections import Counter

from .base import TASMModule, ModuleParameter
from .domain_surface import _discover_probe_files, _detect_level_cols

logger = logging.getLogger("tasm")


def _tokenize_response(text, tokenizer):
    """Extract whole words from model output, keep only single-token words."""
    words = Counter()
    raw_words = re.findall(r'[a-zA-Z]+', text.lower())
    for w in raw_words:
        if len(w) < 2:
            continue
        # Check if this word is a single token
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            words[w] += 1
    return words


def _build_prompt(subject, seeds, tokenizer):
    """Build a prompt that steers generation toward a domain."""
    seed_str = ", ".join(seeds[:15])
    prompt = f"Tell me about {subject.replace('_', ' ')}. Keywords: {seed_str}."
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    return text


class ProbeGeneratorModule(TASMModule):
    name = "probe_generator"
    display_name = "Probe Generator"
    description = (
        "Generates a discriminative probe set by sampling the model's "
        "output distribution per subject. Queries the model N times "
        "per subject, counts token frequencies, removes tokens shared "
        "across subjects. Result: a model-specific vocabulary fingerprint "
        "for each domain."
    )
    version = "0.2.0"

    min_results = 0
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
                description="Seed probe file (provides subjects and context terms)",
                type="select",
                default=options[0] if options else "probes_grammar.csv",
                options=options,
            ),
            ModuleParameter(
                name="queries_per_subject",
                display_name="Queries Per Subject",
                description=(
                    "Number of times to query the model per subject. "
                    "More queries = more stable vocabulary distribution."
                ),
                type="int",
                default=50,
                min_val=10,
                max_val=200,
            ),
            ModuleParameter(
                name="max_new_tokens",
                display_name="Max Tokens Per Response",
                description="Maximum tokens the model generates per query",
                type="int",
                default=256,
                min_val=64,
                max_val=512,
            ),
            ModuleParameter(
                name="min_frequency",
                display_name="Minimum Frequency",
                description=(
                    "Minimum times a token must appear across all queries "
                    "for a subject to be kept. Filters sampling noise."
                ),
                type="int",
                default=3,
                min_val=1,
                max_val=20,
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
        n_queries = int(params.get("queries_per_subject", 50))
        max_tokens = int(params.get("max_new_tokens", 256))
        min_freq = int(params.get("min_frequency", 3))

        csv_path = os.path.join(self._project_root, source_file)
        level_cols, level_names = _detect_level_cols(csv_path)

        # ── Read subjects and seed terms from source ──
        subjects = []
        seeds = {}

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row.get("subject", "").strip()
                if not subj:
                    continue
                if subj not in subjects:
                    subjects.append(subj)
                for col in level_cols:
                    text = row.get(col, "").strip()
                    if text:
                        for w in text.split():
                            wl = w.strip().lower()
                            if wl.isalpha() and len(wl) > 1:
                                seeds.setdefault(subj, []).append(wl)

        for subj in seeds:
            seeds[subj] = list(dict.fromkeys(seeds[subj]))

        if progress:
            progress(f"Loaded {len(subjects)} subjects from {source_file}")

        # ── Sample model output distribution per subject ──
        import torch
        device = next(self._model.parameters()).device

        subject_vocab = {}
        t0 = time.time()

        for si, subj in enumerate(subjects):
            subj_seeds = seeds.get(subj, [subj.replace("_", " ")])
            vocab = Counter()

            for qi in range(n_queries):
                if progress and (qi + 1) % 5 == 0:
                    progress(f"[{si+1}/{len(subjects)}] {subj}: "
                             f"query {qi+1}/{n_queries}")

                prompt_text = _build_prompt(subj, subj_seeds, self._tokenizer)
                inputs = self._tokenizer(prompt_text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.95,
                        repetition_penalty=1.1,
                    )

                gen_ids = out[0][inputs["input_ids"].shape[1]:]
                response = self._tokenizer.decode(gen_ids,
                                                   skip_special_tokens=True)
                word_counts = _tokenize_response(response, self._tokenizer)
                vocab.update(word_counts)

            subject_vocab[subj] = vocab

            if progress:
                progress(f"[{si+1}/{len(subjects)}] {subj}: "
                         f"{len(vocab)} unique tokens from {n_queries} queries")

        elapsed_sampling = time.time() - t0

        # ── Frequency filter ──
        for subj in subjects:
            subject_vocab[subj] = Counter({
                tok: cnt for tok, cnt in subject_vocab[subj].items()
                if cnt >= min_freq
            })

        # ── Cross-subject deduplication ──
        token_subjects = Counter()
        for subj in subjects:
            for tok in subject_vocab[subj]:
                token_subjects[tok] += 1

        shared_tokens = {tok for tok, cnt in token_subjects.items() if cnt > 1}
        discriminative = {}
        for subj in subjects:
            discriminative[subj] = Counter({
                tok: cnt for tok, cnt in subject_vocab[subj].items()
                if tok not in shared_tokens
            })

        # ── Export CSV ──
        stem = os.path.splitext(source_file)[0]
        if stem.endswith("_probes"):
            stem = stem[:-7]
        out_name = f"{stem}_auto_probes.csv"
        out_path = os.path.join(self._project_root, out_name)

        words_per_cell = 3

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["subject", "anchor_id"] + level_cols
            writer.writerow(header)

            for subj in subjects:
                all_words = [tok for tok, _ in
                             discriminative[subj].most_common()]
                n_levels = len(level_cols)

                level_words = [[] for _ in range(n_levels)]
                for i, w in enumerate(all_words):
                    level_words[i % n_levels].append(w)

                max_rows = max(1, max(
                    (len(lw) + words_per_cell - 1) // words_per_cell
                    for lw in level_words) if level_words else 1)

                aid_base = subj[:4]
                for ri in range(max_rows):
                    row = [subj, f"{aid_base}_{ri+1:03d}"]
                    for li in range(n_levels):
                        start = ri * words_per_cell
                        end = start + words_per_cell
                        chunk = level_words[li][start:end]
                        row.append(" ".join(chunk))
                    writer.writerow(row)

        # ── Build results ──
        per_subject = {}
        for subj in subjects:
            raw_count = len(subject_vocab[subj])
            disc_count = len(discriminative[subj])
            top_words = [tok for tok, _ in discriminative[subj].most_common(20)]
            per_subject[subj] = {
                "raw_tokens": raw_count,
                "discriminative_tokens": disc_count,
                "top_20": top_words,
            }

        total_raw = sum(len(subject_vocab[s]) for s in subjects)
        total_disc = sum(len(discriminative[s]) for s in subjects)
        total_shared = len(shared_tokens)

        output = {
            "source_file": source_file,
            "output_file": out_name,
            "subjects": subjects,
            "levels": level_names,
            "queries_per_subject": n_queries,
            "max_new_tokens": max_tokens,
            "min_frequency": min_freq,
            "elapsed_seconds": round(time.time() - t0, 1),
            "sampling_seconds": round(elapsed_sampling, 1),
            "total_raw_tokens": total_raw,
            "total_shared_removed": total_shared,
            "total_discriminative": total_disc,
            "per_subject": per_subject,
            "shared_tokens": sorted(shared_tokens),
        }

        if progress:
            progress(f"Exported {out_name}: {total_disc} discriminative tokens "
                     f"({total_shared} shared removed) across "
                     f"{len(subjects)} subjects")

        return output
