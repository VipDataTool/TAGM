"""
Probe Generator Module for TASM.

Generates a discriminative probe set by statistically sampling the
loaded model's output distribution per class × subclass cell.

Process:
  1. Read classes and subclasses from a source probe CSV template
  2. For each (class, subclass) cell, query the model N times
     using a prompt steered toward that specific cell
  3. Tokenize all responses, count per-cell token frequencies
  4. Frequency filter: discard tokens below minimum threshold
  5. Cross-class dedup: for each subclass, remove tokens that
     appear in more than one class
  6. Cross-subclass dedup: for each class, remove tokens that
     appear in more than one subclass
  7. Export as *_auto_probes.csv

The two-axis deduplication is the discriminative filter.  A token
must be unique to its cell — unique to its class along the subclass
axis, and unique to its subclass along the class axis.  What survives
is the model-specific vocabulary fingerprint for each cell in the
class × subclass lattice.
"""

import os
import csv
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
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            words[w] += 1
    return words


def _build_cell_prompt(cls, subclass, seeds, tokenizer, word_count=200):
    """Build a prompt steered toward a specific class × subclass cell."""
    cls_label = cls.replace("_", " ")
    sub_label = subclass.replace("_", " ")
    seed_str = ", ".join(seeds[:15]) if seeds else cls_label

    prompt = (
        f"Provide a comprehensive {word_count}-word description of "
        f"{sub_label} within context of and/or related to {cls_label}. "
        f"Examples include: {seed_str}. "
        f"Provide only the response and nothing more."
    )
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
        "output distribution per class × subclass cell. Queries the "
        "model N times per cell, counts token frequencies, removes "
        "tokens shared across classes and across subclasses. "
        "Result: a model-specific vocabulary fingerprint for each "
        "cell in the class × subclass lattice."
    )
    version = "0.3.0"

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
                description="Template probe file (provides classes and subclasses)",
                type="select",
                default=options[0] if options else "probes_grammar.csv",
                options=options,
            ),
            ModuleParameter(
                name="queries_per_cell",
                display_name="Queries Per Cell",
                description=(
                    "Number of times to query the model per class × subclass cell. "
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
                    "for a cell to be kept. Filters sampling noise."
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
        n_queries = int(params.get("queries_per_cell", 50))
        max_tokens = int(params.get("max_new_tokens", 256))
        min_freq = int(params.get("min_frequency", 3))

        csv_path = os.path.join(self._project_root, source_file)
        level_cols, level_names = _detect_level_cols(csv_path)
        n_subclasses = len(level_cols)

        # ── Read classes and per-cell seed terms from template ──
        classes = []
        cell_seeds = {}  # (class, subclass_col) -> [seed words]

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row.get("subject", "").strip()
                if not cls:
                    continue
                if cls not in classes:
                    classes.append(cls)
                for col in level_cols:
                    text = row.get(col, "").strip()
                    if text:
                        key = (cls, col)
                        cell_seeds.setdefault(key, [])
                        for w in text.split():
                            wl = w.strip().lower()
                            if wl.isalpha() and len(wl) > 1:
                                cell_seeds[key].append(wl)

        # Deduplicate seed lists preserving order
        for key in cell_seeds:
            cell_seeds[key] = list(dict.fromkeys(cell_seeds[key]))

        n_classes = len(classes)
        total_cells = n_classes * n_subclasses

        if progress:
            progress(f"Loaded {n_classes} classes × {n_subclasses} subclasses "
                     f"= {total_cells} cells from {source_file}")

        # ── Sample model output distribution per cell ──
        import torch
        device = next(self._model.parameters()).device

        cell_vocab = {}  # (class, subclass_col) -> Counter
        t0 = time.time()
        cell_idx = 0

        for cls in classes:
            for col in level_cols:
                cell_idx += 1
                sub_label = col.replace("_", " ")
                seeds = cell_seeds.get((cls, col), [cls.replace("_", " ")])
                vocab = Counter()

                for qi in range(n_queries):
                    if progress and (qi + 1) % 5 == 0:
                        progress(f"[{cell_idx}/{total_cells}] "
                                 f"{cls} × {sub_label}: "
                                 f"query {qi+1}/{n_queries}")

                    prompt_text = _build_cell_prompt(
                        cls, col, seeds, self._tokenizer,
                        word_count=max_tokens)
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
                    response = self._tokenizer.decode(
                        gen_ids, skip_special_tokens=True)
                    word_counts = _tokenize_response(
                        response, self._tokenizer)
                    vocab.update(word_counts)

                cell_vocab[(cls, col)] = vocab

                if progress:
                    progress(f"[{cell_idx}/{total_cells}] "
                             f"{cls} × {sub_label}: "
                             f"{len(vocab)} unique tokens")

        elapsed_sampling = time.time() - t0

        # ── Frequency filter ──
        for key in cell_vocab:
            cell_vocab[key] = Counter({
                tok: cnt for tok, cnt in cell_vocab[key].items()
                if cnt >= min_freq
            })

        # ── Cross-class dedup ──
        # For each subclass, remove tokens appearing in >1 class
        cross_class_shared = set()
        for col in level_cols:
            token_classes = Counter()
            for cls in classes:
                for tok in cell_vocab.get((cls, col), {}):
                    token_classes[tok] += 1
            for tok, cnt in token_classes.items():
                if cnt > 1:
                    cross_class_shared.add(tok)

        # ── Cross-subclass dedup ──
        # For each class, remove tokens appearing in >1 subclass
        cross_subclass_shared = set()
        for cls in classes:
            token_subclasses = Counter()
            for col in level_cols:
                for tok in cell_vocab.get((cls, col), {}):
                    token_subclasses[tok] += 1
            for tok, cnt in token_subclasses.items():
                if cnt > 1:
                    cross_subclass_shared.add(tok)

        all_shared = cross_class_shared | cross_subclass_shared

        # ── Apply filters ──
        discriminative = {}
        for cls in classes:
            for col in level_cols:
                key = (cls, col)
                discriminative[key] = Counter({
                    tok: cnt for tok, cnt in cell_vocab.get(key, {}).items()
                    if tok not in all_shared
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

            for cls in classes:
                col_words = {}
                for col in level_cols:
                    col_words[col] = [
                        tok for tok, _ in
                        discriminative[(cls, col)].most_common()
                    ]

                max_rows = max(1, max(
                    (len(col_words[col]) + words_per_cell - 1) // words_per_cell
                    for col in level_cols
                ))

                aid_base = cls[:4]
                for ri in range(max_rows):
                    row = [cls, f"{aid_base}_{ri+1:03d}"]
                    for col in level_cols:
                        start = ri * words_per_cell
                        end = start + words_per_cell
                        chunk = col_words[col][start:end]
                        row.append(" ".join(chunk))
                    writer.writerow(row)

        # ── Build results ──
        per_class = {}
        for cls in classes:
            raw_count = sum(len(cell_vocab.get((cls, col), {}))
                           for col in level_cols)
            disc_count = sum(len(discriminative[(cls, col)])
                            for col in level_cols)
            merged = Counter()
            for col in level_cols:
                merged.update(discriminative[(cls, col)])
            top_words = [tok for tok, _ in merged.most_common(20)]
            per_class[cls] = {
                "raw_tokens": raw_count,
                "discriminative_tokens": disc_count,
                "top_20": top_words,
            }

        total_raw = sum(
            len(cell_vocab.get((cls, col), {}))
            for cls in classes for col in level_cols
        )
        total_disc = sum(
            len(discriminative[(cls, col)])
            for cls in classes for col in level_cols
        )

        output = {
            "source_file": source_file,
            "output_file": out_name,
            "subjects": classes,
            "levels": level_names,
            "queries_per_cell": n_queries,
            "queries_per_subject": n_queries,  # backward compat for frontend
            "max_new_tokens": max_tokens,
            "min_frequency": min_freq,
            "elapsed_seconds": round(time.time() - t0, 1),
            "sampling_seconds": round(elapsed_sampling, 1),
            "total_cells": total_cells,
            "total_raw_tokens": total_raw,
            "total_shared_removed": len(all_shared),
            "cross_class_shared": len(cross_class_shared),
            "cross_subclass_shared": len(cross_subclass_shared),
            "total_discriminative": total_disc,
            "per_subject": per_class,  # backward compat for frontend
            "shared_tokens": sorted(all_shared),
        }

        if progress:
            progress(f"Exported {out_name}: {total_disc} discriminative tokens "
                     f"({len(cross_class_shared)} shared across classes, "
                     f"{len(cross_subclass_shared)} shared across subclasses) "
                     f"across {total_cells} cells")

        return output
