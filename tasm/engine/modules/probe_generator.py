"""
Probe Generator Module for TASM.

Generates a discriminative probe set by statistically sampling the
loaded model's output distribution per class × subclass cell.

Process:
  1. Read a template CSV — any CSV with a 'subject' column and one
     or more subclass columns (everything after subject/anchor_id)
  2. For each (class, subclass) cell, query the model N times
     using a prompt steered toward that specific cell
  3. Tokenize all responses, count per-cell token frequencies
  4. Frequency filter: discard tokens below minimum threshold
  5. Cross-class dedup: for each subclass, remove tokens that
     appear in more than one class
  6. Cross-subclass dedup: for each class, remove tokens that
     appear in more than one subclass
  7. Export to the specified output path

The two-axis deduplication is the discriminative filter.  A token
must be unique to its cell — unique to its class along the subclass
axis, and unique to its subclass along the class axis.  What survives
is the model-specific vocabulary fingerprint for each cell in the
class × subclass lattice.

No dependencies on other TASM modules.  Reads any conforming CSV.
"""

import os
import csv
import logging
import re
import time
from collections import Counter

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")

FIXED_COLS = {"subject", "anchor_id"}
META_TAG = "_meta"


# ─── Template Parsing (self-contained) ────────────────────────

def _parse_template(csv_path):
    """Parse a template CSV into classes, subclass columns, and per-cell seeds.

    Returns:
        classes:      list of class names (ordered by first appearance)
        subclass_cols: list of subclass column names
        cell_seeds:   dict of (class, subclass_col) -> [seed words]
    """
    classes = []
    subclass_cols = []
    cell_seeds = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty or headerless CSV: {csv_path}")

        # Subclass columns = everything that isn't subject or anchor_id
        subclass_cols = [
            col for col in reader.fieldnames
            if col.strip().lower() not in FIXED_COLS
        ]
        if not subclass_cols:
            raise RuntimeError(
                f"No subclass columns found in {csv_path}. "
                f"Need columns beyond 'subject' and 'anchor_id'."
            )

        for row in reader:
            cls = row.get("subject", "").strip()
            if not cls or cls == META_TAG:
                continue
            if cls not in classes:
                classes.append(cls)

            for col in subclass_cols:
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

    return classes, subclass_cols, cell_seeds


# ─── Tokenization ─────────────────────────────────────────────

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


# ─── Prompt Construction ──────────────────────────────────────

DEFAULT_PROMPT_TEMPLATE = (
    "Provide a comprehensive {word_count}-word description of "
    "{subclass} within context of and/or related to {class}. "
    "Examples include: {seeds}. "
    "Provide only the response and nothing more."
)


def _build_cell_prompt(cls, subclass, seeds, tokenizer,
                       word_count=200, template=None):
    """Build a prompt steered toward a specific class × subclass cell."""
    cls_label = cls.replace("_", " ")
    sub_label = subclass.replace("_", " ")
    seed_str = ", ".join(seeds[:15]) if seeds else cls_label

    tmpl = template or DEFAULT_PROMPT_TEMPLATE
    prompt = tmpl.format(**{
        "class": cls_label,
        "subclass": sub_label,
        "seeds": seed_str,
        "word_count": word_count,
    })

    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    return text


# ─── Module ───────────────────────────────────────────────────

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
        self._project_root = None

    def set_project_root(self, root):
        self._project_root = root

    def set_model(self, model, tokenizer):
        """Provide access to the loaded instruct model."""
        self._model = model
        self._tokenizer = tokenizer

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="template_file",
                display_name="Template File",
                description=(
                    "CSV template with 'subject' column and one or more "
                    "subclass columns. Defines the class × subclass lattice."
                ),
                type="file",
                default="",
            ),
            ModuleParameter(
                name="output_name",
                display_name="Output File Name",
                description=(
                    "Name for the generated probe file. "
                    "Saved in the project root."
                ),
                type="text",
                default="auto_probes.csv",
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
            ModuleParameter(
                name="prompt_template",
                display_name="Prompt Template",
                description=(
                    "Template sent to the model for each cell. "
                    "Placeholders: {class}, {subclass}, {seeds}, {word_count}"
                ),
                type="textarea",
                default=DEFAULT_PROMPT_TEMPLATE,
            ),
            ModuleParameter(
                name="export_catalog",
                display_name="Export Inference Catalog",
                description=(
                    "Save a CSV log of every model query and response. "
                    "Useful for auditing prompt quality."
                ),
                type="bool",
                default=False,
            ),
        ]

    def validate(self, session_results, params):
        if self._model is None or self._tokenizer is None:
            return False, "Model not loaded. Load a model first."

        template = params.get("template_file", "")
        if not template:
            return False, "No template file selected."

        # Resolve path
        path = template
        if not os.path.isabs(path) and self._project_root:
            path = os.path.join(self._project_root, path)
        if not os.path.exists(path):
            return False, f"Template file not found: {template}"

        output = params.get("output_name", "").strip()
        if not output:
            return False, "No output file name specified."
        if not output.endswith(".csv"):
            return False, "Output file name must end in .csv"

        # Prevent overwriting the template
        template = params.get("template_file", "")
        if output and template:
            tmpl_name = os.path.basename(template)
            if output == tmpl_name:
                return False, f"Output name matches template file '{tmpl_name}'. Use a different name."

        return True, "OK"

    def run(self, session_results, params, progress=None):
        template_file = params.get("template_file", "")
        output_name = params.get("output_name", "auto_probes.csv").strip()
        n_queries = int(params.get("queries_per_cell", 50))
        max_tokens = int(params.get("max_new_tokens", 256))
        min_freq = int(params.get("min_frequency", 3))
        prompt_template = params.get("prompt_template", "").strip() or None
        export_catalog = bool(params.get("export_catalog", False))

        # Resolve template path
        csv_path = template_file
        if not os.path.isabs(csv_path) and self._project_root:
            csv_path = os.path.join(self._project_root, csv_path)

        # ── Parse template ──
        if progress:
            progress("Parsing template...")

        classes, subclass_cols, cell_seeds = _parse_template(csv_path)
        n_classes = len(classes)
        n_subclasses = len(subclass_cols)
        total_cells = n_classes * n_subclasses

        if progress:
            progress(f"Loaded {n_classes} classes × {n_subclasses} subclasses "
                     f"= {total_cells} cells")

        # ── Sample model output distribution per cell ──
        import torch
        device = next(self._model.parameters()).device

        cell_vocab = {}  # (class, subclass_col) -> Counter
        catalog = []     # [{class, subclass, query, prompt, response, tokens}]
        t0 = time.time()
        cell_idx = 0

        for cls in classes:
            for col in subclass_cols:
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
                        word_count=max_tokens,
                        template=prompt_template)
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

                    catalog.append({
                        "class": cls,
                        "subclass": col,
                        "query": qi + 1,
                        "response": response.strip(),
                        "tokens_extracted": sorted(word_counts.keys()),
                    })

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
        cross_class_shared = set()
        for col in subclass_cols:
            token_classes = Counter()
            for cls in classes:
                for tok in cell_vocab.get((cls, col), {}):
                    token_classes[tok] += 1
            for tok, cnt in token_classes.items():
                if cnt > 1:
                    cross_class_shared.add(tok)

        # ── Cross-subclass dedup ──
        cross_subclass_shared = set()
        for cls in classes:
            token_subclasses = Counter()
            for col in subclass_cols:
                for tok in cell_vocab.get((cls, col), {}):
                    token_subclasses[tok] += 1
            for tok, cnt in token_subclasses.items():
                if cnt > 1:
                    cross_subclass_shared.add(tok)

        all_shared = cross_class_shared | cross_subclass_shared

        # ── Apply filters ──
        discriminative = {}
        for cls in classes:
            for col in subclass_cols:
                key = (cls, col)
                discriminative[key] = Counter({
                    tok: cnt for tok, cnt in cell_vocab.get(key, {}).items()
                    if tok not in all_shared
                })

        # ── Export CSV ──
        out_path = output_name
        if not os.path.isabs(out_path) and self._project_root:
            out_path = os.path.join(self._project_root, out_path)

        words_per_cell = 3

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["subject", "anchor_id"] + subclass_cols
            writer.writerow(header)

            for cls in classes:
                col_words = {}
                for col in subclass_cols:
                    col_words[col] = [
                        tok for tok, _ in
                        discriminative[(cls, col)].most_common()
                    ]

                max_rows = max(1, max(
                    (len(col_words[col]) + words_per_cell - 1) // words_per_cell
                    for col in subclass_cols
                ))

                aid_base = cls[:4]
                for ri in range(max_rows):
                    row = [cls, f"{aid_base}_{ri+1:03d}"]
                    for col in subclass_cols:
                        start = ri * words_per_cell
                        end = start + words_per_cell
                        chunk = col_words[col][start:end]
                        row.append(" ".join(chunk))
                    writer.writerow(row)

        # ── Export catalog CSV (optional) ──
        catalog_name = None
        if export_catalog:
            catalog_name = output_name.replace(".csv", "_catalog.csv")
            catalog_path = catalog_name
            if not os.path.isabs(catalog_path) and self._project_root:
                catalog_path = os.path.join(self._project_root, catalog_path)

            with open(catalog_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["class", "subclass", "query", "response",
                                 "tokens_extracted"])
                for entry in catalog:
                    writer.writerow([
                        entry["class"],
                        entry["subclass"],
                        entry["query"],
                        entry["response"],
                        " ".join(entry["tokens_extracted"]),
                    ])

        # ── Per-cell catalog summary for frontend ──
        cell_catalog = {}
        for entry in catalog:
            key = f"{entry['class']}|{entry['subclass']}"
            cell_catalog.setdefault(key, []).append({
                "q": entry["query"],
                "response": entry["response"][:300],
                "n_tokens": len(entry["tokens_extracted"]),
            })

        # ── Build results ──
        per_class = {}
        for cls in classes:
            raw_count = sum(len(cell_vocab.get((cls, col), {}))
                           for col in subclass_cols)
            disc_count = sum(len(discriminative[(cls, col)])
                            for col in subclass_cols)
            merged = Counter()
            for col in subclass_cols:
                merged.update(discriminative[(cls, col)])
            top_words = [tok for tok, _ in merged.most_common(20)]
            per_class[cls] = {
                "raw_tokens": raw_count,
                "discriminative_tokens": disc_count,
                "top_20": top_words,
            }

        total_raw = sum(
            len(cell_vocab.get((cls, col), {}))
            for cls in classes for col in subclass_cols
        )
        total_disc = sum(
            len(discriminative[(cls, col)])
            for cls in classes for col in subclass_cols
        )

        subclass_names = [c.replace("_", " ") for c in subclass_cols]

        output = {
            "template_file": template_file,
            "output_file": output_name,
            "subjects": classes,
            "levels": subclass_names,
            "queries_per_cell": n_queries,
            "queries_per_subject": n_queries,  # backward compat
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
            "per_subject": per_class,  # backward compat
            "shared_tokens": sorted(all_shared),
            "catalog_file": catalog_name,  # None if export disabled
            "catalog": cell_catalog,
            "total_queries": len(catalog),
        }

        if progress:
            progress(f"Exported {output_name}: {total_disc} discriminative "
                     f"tokens ({len(cross_class_shared)} shared across "
                     f"classes, {len(cross_subclass_shared)} shared across "
                     f"subclasses) across {total_cells} cells")

        return output
