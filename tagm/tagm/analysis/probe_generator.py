"""Probe Generator: discriminative vocabulary extraction from the loaded model.

Ported from TASM's engine/modules/probe_generator.py. Queries the loaded
instruct model N times per cell in a class × subclass template, harvests
domain-specific vocabulary from responses, applies frequency filtering
and two-axis discriminative deduplication. Output: a model-specific
vocabulary fingerprint per cell, exported as CSV and optionally auto-
embedded via TAGM's EmbeddingGenerator + ProbeStore.

This module needs live model access (set_pipeline) — it's a tool, not
a passive analysis of session data.
"""
from __future__ import annotations

import csv
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from tagm.analysis.base import AnalysisModule
from tagm.analysis.registry import register_analysis
from tagm.measurement.parameters import ModuleParameter

logger = logging.getLogger("tagm")

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

FIXED_COLS = {"subject", "anchor_id"}
META_TAG = "_meta"

DEFAULT_PROMPT_TEMPLATE = (
    "List {word_count} domain-specific terms, jargon, and technical "
    "vocabulary related to {class} in the context of {subclass}. "
    "Include specialized terminology, contextual phrases, and "
    "insider language. Seed context: {seeds}. "
    "One term per line."
)


# ─── Template Parsing (TASM 5x5 format) ──────────────────────────

def _parse_template(csv_path):
    """Parse a TASM-format template CSV.

    Returns:
        classes:       list of class names (ordered by first appearance)
        subclass_cols: list of subclass column names
        cell_seeds:    dict of (class, subclass_col) -> seed phrase string
    """
    classes = []
    subclass_cols = []
    cell_seeds = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty or headerless CSV: {csv_path}")

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
                    cell_seeds[(cls, col)] = text

    return classes, subclass_cols, cell_seeds


# ─── Tokenization ────────────────────────────────────────────────

def _tokenize_response(text, stopwords=frozenset()):
    """Extract terms from model output."""
    words = Counter()
    raw_words = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text.lower())
    for w in raw_words:
        if len(w) < 3 or w in stopwords:
            continue
        words[w] += 1
    return words


def _load_stopwords(path):
    """Load stopword list from a text file."""
    words = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            w = line.lower().split()[0]
            words.append(w)
    result = frozenset(words)
    logger.info(f"[STOPWORDS] Loaded {len(result)} words from {path}")
    return result, len(result)


# ─── Prompt Construction ─────────────────────────────────────────

def _build_cell_prompt(cls, subclass, seeds, tokenizer,
                       word_count=200, template=None):
    """Build a prompt steered toward a specific class × subclass cell."""
    cls_label = cls.replace("_", " ")
    sub_label = subclass.replace("_", " ")
    seed_str = seeds if seeds else cls_label

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


# ─── Module ──────────────────────────────────────────────────────

@register_analysis
class ProbeGenerator(AnalysisModule):
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
    version = "1.0.0"
    min_results = 0
    depends_on_measurements = ()

    parameters = [
        ModuleParameter(
            name="template_file",
            display_name="Template File",
            description=(
                "CSV template with 'subject' column and one or more "
                "subclass columns. Defines the class × subclass lattice."
            ),
            kind="file",
            default="",
        ),
        ModuleParameter(
            name="output_name",
            display_name="Output File Name",
            description="Name for the generated probe file.",
            kind="text",
            default="auto_probes.csv",
        ),
        ModuleParameter(
            name="queries_per_cell",
            display_name="Queries Per Cell",
            description=(
                "Number of times to query the model per class × subclass cell."
            ),
            kind="int",
            default=50,
            min_value=10,
            max_value=200,
        ),
        ModuleParameter(
            name="max_new_tokens",
            display_name="Max Tokens Per Response",
            description="Maximum tokens the model generates per query.",
            kind="int",
            default=256,
            min_value=64,
            max_value=512,
        ),
        ModuleParameter(
            name="temperature",
            display_name="Temperature",
            description="Sampling temperature. Lower = tighter vocabulary.",
            kind="float",
            default=0.9,
            min_value=0.1,
            max_value=2.0,
        ),
        ModuleParameter(
            name="top_p",
            display_name="Top-P (Nucleus Sampling)",
            description="Nucleus sampling threshold.",
            kind="float",
            default=0.95,
            min_value=0.1,
            max_value=1.0,
        ),
        ModuleParameter(
            name="repetition_penalty",
            display_name="Repetition Penalty",
            description="Penalty for repeating tokens. 1.0 = no penalty.",
            kind="float",
            default=1.1,
            min_value=1.0,
            max_value=2.0,
        ),
        ModuleParameter(
            name="min_frequency",
            display_name="Minimum Frequency",
            description="Minimum appearances to keep a token.",
            kind="int",
            default=3,
            min_value=1,
            max_value=20,
        ),
        ModuleParameter(
            name="prompt_template",
            display_name="Prompt Template",
            description=(
                "Template sent to the model for each cell. "
                "Placeholders: {class}, {subclass}, {seeds}, {word_count}"
            ),
            kind="textarea",
            default=DEFAULT_PROMPT_TEMPLATE,
        ),
        ModuleParameter(
            name="export_catalog",
            display_name="Export Inference Catalog",
            description="Save a CSV log of every model query and response.",
            kind="bool",
            default=False,
        ),
        ModuleParameter(
            name="min_probes_per_cell",
            display_name="Min Probes Per Cell",
            description=(
                "Minimum discriminative tokens required per cell. "
                "Thin cells are automatically re-queried. 0 to disable."
            ),
            kind="int",
            default=5,
            min_value=0,
            max_value=50,
        ),
        ModuleParameter(
            name="max_probes_per_cell",
            display_name="Max Probes Per Cell",
            description=(
                "Truncate each cell to this many probes (highest frequency). "
                "0 to disable."
            ),
            kind="int",
            default=10,
            min_value=0,
            max_value=100,
        ),
        ModuleParameter(
            name="auto_populate_batch",
            display_name="Auto-Populate Queries Per Round",
            description="Additional queries per thin cell per auto-populate round.",
            kind="int",
            default=5,
            min_value=1,
            max_value=50,
        ),
        ModuleParameter(
            name="auto_populate_max_rounds",
            display_name="Auto-Populate Max Rounds",
            description="Maximum re-query rounds for thin cells.",
            kind="int",
            default=3,
            min_value=1,
            max_value=10,
        ),
        ModuleParameter(
            name="skip_dedup",
            display_name="Skip Discriminative Deduplication",
            description=(
                "Bypass two-axis dedup. Keeps every token that passes "
                "the frequency floor. Diagnostic only."
            ),
            kind="bool",
            default=False,
        ),
        ModuleParameter(
            name="auto_apply",
            display_name="Apply Probe Set After Generation",
            description=(
                "Automatically embed and activate the generated probe set."
            ),
            kind="bool",
            default=False,
        ),
    ]

    def __init__(self):
        self._pipeline = None
        self._probe_store = None
        self._progress = None

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline

    def set_probe_store(self, probe_store):
        self._probe_store = probe_store

    def set_progress(self, progress_fn):
        self._progress = progress_fn

    def _prog(self, msg):
        if self._progress:
            self._progress(msg)
        logger.info(f"[PROBE_GEN] {msg}")

    def check_dependencies(self, session: dict) -> list[str]:
        """Probe generator doesn't need session data, but needs a model."""
        if self._pipeline is None or not self._pipeline.loaded:
            return ["Probe Generator requires a loaded model. Load a model first."]
        return []

    def run(self, session: dict, params: dict,
            probes: Optional[dict] = None) -> dict:
        import torch

        template_file = params.get("template_file", "")
        output_name = params.get("output_name", "auto_probes.csv").strip()
        n_queries = int(params.get("queries_per_cell", 50))
        max_tokens = int(params.get("max_new_tokens", 256))
        min_freq = int(params.get("min_frequency", 3))
        temperature = float(params.get("temperature", 0.9))
        top_p = float(params.get("top_p", 0.95))
        rep_penalty = float(params.get("repetition_penalty", 1.1))
        prompt_template = params.get("prompt_template", "").strip() or None
        export_catalog = bool(params.get("export_catalog", False))
        min_probes = int(params.get("min_probes_per_cell", 0))
        max_probes = int(params.get("max_probes_per_cell", 0))
        ap_batch = int(params.get("auto_populate_batch", 5))
        ap_max_rounds = int(params.get("auto_populate_max_rounds", 3))
        skip_dedup = bool(params.get("skip_dedup", False))

        model = self._pipeline.instruct_model
        tokenizer = self._pipeline.tokenizer
        device = self._pipeline.device

        # ── Load stopwords ──
        sw_path = _TEMPLATES_DIR / "stopwords.txt"
        stopwords = frozenset()
        stopword_count = 0
        if sw_path.exists():
            stopwords, stopword_count = _load_stopwords(str(sw_path))
            self._prog(f"Loaded {stopword_count} stopwords")
        stopword_hits: Counter = Counter()

        # ── Resolve template path ──
        csv_path = template_file
        if not os.path.isabs(csv_path):
            candidate = _TEMPLATES_DIR / csv_path
            if candidate.exists():
                csv_path = str(candidate)
            else:
                project_root = _TEMPLATES_DIR.parent.parent
                csv_path = str(project_root / csv_path)

        if not os.path.exists(csv_path):
            return {"error": f"Template file not found: {template_file}"}

        # ── Parse template ──
        self._prog("Parsing template...")
        classes, subclass_cols, cell_seeds = _parse_template(csv_path)
        n_classes = len(classes)
        n_subclasses = len(subclass_cols)
        total_cells = n_classes * n_subclasses
        self._prog(f"Loaded {n_classes} classes × {n_subclasses} subclasses "
                    f"= {total_cells} cells")

        # ── Sample model output distribution per cell ──
        cell_vocab: dict[tuple, Counter] = {}
        catalog: list[dict] = []
        t0 = time.time()
        cell_idx = 0

        for cls in classes:
            for col in subclass_cols:
                cell_idx += 1
                sub_label = col.replace("_", " ")
                seeds = cell_seeds.get((cls, col), cls.replace("_", " "))
                vocab: Counter = Counter()

                for qi in range(n_queries):
                    if (qi + 1) % 5 == 0:
                        self._prog(f"[{cell_idx}/{total_cells}] "
                                   f"{cls} × {sub_label}: "
                                   f"query {qi+1}/{n_queries}")

                    prompt_text = _build_cell_prompt(
                        cls, col, seeds, tokenizer,
                        word_count=max_tokens,
                        template=prompt_template)

                    inputs = tokenizer(prompt_text, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            repetition_penalty=rep_penalty,
                        )
                    gen_ids = out[0][inputs["input_ids"].shape[1]:]
                    response = tokenizer.decode(gen_ids, skip_special_tokens=True)

                    word_counts = _tokenize_response(response, stopwords)
                    vocab.update(word_counts)

                    if stopwords:
                        for w in re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*",
                                            response.lower()):
                            if len(w) >= 3 and w in stopwords:
                                stopword_hits[w] += 1

                    catalog.append({
                        "class": cls,
                        "subclass": col,
                        "query": qi + 1,
                        "response": response.strip(),
                        "tokens_extracted": sorted(word_counts.keys()),
                    })

                cell_vocab[(cls, col)] = vocab
                self._prog(f"[{cell_idx}/{total_cells}] "
                           f"{cls} × {sub_label}: "
                           f"{len(vocab)} unique tokens")

        elapsed_sampling = time.time() - t0

        # ── Dedup pipeline ──
        def _run_dedup(cv):
            filtered = {}
            for key in cv:
                filtered[key] = Counter({
                    tok: cnt for tok, cnt in cv[key].items()
                    if cnt >= min_freq
                })
            if skip_dedup:
                return filtered, set(), set()
            xclass = set()
            for col in subclass_cols:
                tc: Counter = Counter()
                for cls in classes:
                    for tok in filtered.get((cls, col), {}):
                        tc[tok] += 1
                for tok, cnt in tc.items():
                    if cnt > 1:
                        xclass.add(tok)
            xsub = set()
            for cls in classes:
                ts: Counter = Counter()
                for col in subclass_cols:
                    for tok in filtered.get((cls, col), {}):
                        ts[tok] += 1
                for tok, cnt in ts.items():
                    if cnt > 1:
                        xsub.add(tok)
            shared = xclass | xsub
            disc = {}
            for cls in classes:
                for col in subclass_cols:
                    key = (cls, col)
                    disc[key] = Counter({
                        tok: cnt for tok, cnt in filtered.get(key, {}).items()
                        if tok not in shared
                    })
            return disc, xclass, xsub

        discriminative, cross_class_shared, cross_subclass_shared = \
            _run_dedup(cell_vocab)
        all_shared = cross_class_shared | cross_subclass_shared

        # ── Auto-populate thin cells ──
        auto_populate_rounds = 0
        auto_populate_queries = 0

        if min_probes > 0:
            for round_num in range(ap_max_rounds):
                cell_cnts = {(cls, col): len(discriminative[(cls, col)])
                             for cls in classes for col in subclass_cols}
                thin = [(cls, col) for (cls, col), n in cell_cnts.items()
                        if n < min_probes and len(cell_vocab.get((cls, col), {})) > 0]

                if not thin:
                    break

                auto_populate_rounds += 1
                self._prog(f"Auto-populate round {round_num+1}: "
                           f"{len(thin)} cells below {min_probes}")

                for cls, col in thin:
                    sub_label = col.replace("_", " ")
                    seeds = cell_seeds.get((cls, col), cls.replace("_", " "))

                    for qi in range(ap_batch):
                        prompt_text = _build_cell_prompt(
                            cls, col, seeds, tokenizer,
                            word_count=max_tokens,
                            template=prompt_template)

                        inputs = tokenizer(prompt_text, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            out = model.generate(
                                **inputs,
                                max_new_tokens=max_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                repetition_penalty=rep_penalty,
                            )
                        gen_ids = out[0][inputs["input_ids"].shape[1]:]
                        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

                        word_counts = _tokenize_response(response, stopwords)
                        cell_vocab[(cls, col)].update(word_counts)

                        if stopwords:
                            for w in re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*",
                                                response.lower()):
                                if len(w) >= 3 and w in stopwords:
                                    stopword_hits[w] += 1

                        catalog.append({
                            "class": cls,
                            "subclass": col,
                            "query": n_queries + (round_num * ap_batch) + qi + 1,
                            "response": response.strip(),
                            "tokens_extracted": sorted(word_counts.keys()),
                        })
                        auto_populate_queries += 1

                discriminative, cross_class_shared, cross_subclass_shared = \
                    _run_dedup(cell_vocab)
                all_shared = cross_class_shared | cross_subclass_shared

        # ── Truncate to max probes per cell ──
        truncated_count = 0
        if max_probes > 0:
            for cls in classes:
                for col in subclass_cols:
                    key = (cls, col)
                    if len(discriminative[key]) > max_probes:
                        truncated_count += len(discriminative[key]) - max_probes
                        discriminative[key] = Counter(
                            dict(discriminative[key].most_common(max_probes)))

        # ── Cell resolution analysis ──
        cell_counts = {}
        thin_cells = []
        empty_cells = []
        for cls in classes:
            for col in subclass_cols:
                key = (cls, col)
                n = len(discriminative[key])
                sub_label = col.replace("_", " ")
                cell_counts[(cls, sub_label)] = n
                if n == 0:
                    empty_cells.append(f"{cls} × {sub_label}")
                elif min_probes > 0 and n < min_probes:
                    thin_cells.append(f"{cls} × {sub_label} ({n}/{min_probes})")

        counts_list = [cell_counts[k] for k in cell_counts if cell_counts[k] > 0]
        cell_resolution = {
            "total_cells": total_cells,
            "populated_cells": len(counts_list),
            "empty_cells": len(empty_cells),
            "empty_cell_names": empty_cells,
            "min_probes_setting": min_probes,
            "max_probes_setting": max_probes,
            "truncated_probes": truncated_count,
            "cells_below_minimum": len(thin_cells),
            "thin_cell_names": thin_cells,
            "min_count": min(counts_list) if counts_list else 0,
            "max_count": max(counts_list) if counts_list else 0,
            "mean_count": round(sum(counts_list) / len(counts_list), 1) if counts_list else 0,
            "auto_populate_rounds": auto_populate_rounds,
            "auto_populate_queries": auto_populate_queries,
            "per_cell": {f"{cls} × {col.replace('_', ' ')}": cell_counts.get((cls, col.replace('_', ' ')), 0)
                         for cls in classes for col in subclass_cols},
        }

        # ── Export CSV ──
        project_root = _TEMPLATES_DIR.parent.parent
        out_path = str(project_root / output_name)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "subclass", "text"])
            for cls in classes:
                for col in subclass_cols:
                    tokens = [tok for tok, _ in discriminative[(cls, col)].most_common()]
                    for tok in tokens:
                        writer.writerow([cls, col, tok])

        # ── Per-cell catalog ──
        cell_catalog: dict[str, list] = {}
        for entry in catalog:
            key = f"{entry['class']}|{entry['subclass']}"
            cell_catalog.setdefault(key, []).append({
                "q": entry["query"],
                "response": entry["response"],
                "tokens_extracted": list(entry["tokens_extracted"]),
            })

        # ── Export catalog CSV (optional) ──
        catalog_name = None
        if export_catalog:
            catalog_name = output_name.replace(".csv", "_catalog.csv")
            catalog_path = str(project_root / catalog_name)
            with open(catalog_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["class", "subclass", "query", "response",
                                 "tokens_extracted"])
                for key, entries in cell_catalog.items():
                    cls, col = key.split("|", 1)
                    for entry in entries:
                        writer.writerow([
                            cls, col, entry["q"], entry["response"],
                            " ".join(entry["tokens_extracted"]),
                        ])

        # ── Build results (TASM shape for renderer) ──
        per_class: dict[str, Any] = {}
        for cls in classes:
            raw_count = sum(len(cell_vocab.get((cls, col), {}))
                           for col in subclass_cols)
            disc_count = sum(len(discriminative[(cls, col)])
                            for col in subclass_cols)
            merged: Counter = Counter()
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
            "inference_class": self._pipeline.adapter.family_id if self._pipeline else "unknown",
            "subjects": classes,
            "levels": subclass_names,
            "queries_per_cell": n_queries,
            "queries_per_subject": n_queries,
            "max_new_tokens": max_tokens,
            "min_frequency": min_freq,
            "skip_dedup": skip_dedup,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": rep_penalty,
            "elapsed_seconds": round(time.time() - t0, 1),
            "sampling_seconds": round(elapsed_sampling, 1),
            "total_cells": total_cells,
            "total_raw_tokens": total_raw,
            "total_shared_removed": len(all_shared),
            "cross_class_shared": len(cross_class_shared),
            "cross_subclass_shared": len(cross_subclass_shared),
            "total_discriminative": total_disc,
            "per_subject": per_class,
            "shared_tokens": sorted(all_shared),
            "catalog_file": catalog_name,
            "catalog": cell_catalog,
            "total_queries": len(catalog),
            "stopword_file": str(sw_path) if stopwords else None,
            "stopword_count": stopword_count,
            "stopword_hits": dict(stopword_hits.most_common()),
            "stopword_total_filtered": sum(stopword_hits.values()),
            "cell_resolution": cell_resolution,
        }

        self._prog(f"Exported {output_name}: {total_disc} discriminative "
                    f"tokens across {total_cells} cells")

        # ── Auto-apply ──
        auto_apply = bool(params.get("auto_apply", False))
        if auto_apply and self._pipeline and self._probe_store:
            output["auto_apply"] = self._auto_apply_probes(
                output_name, out_path)
        else:
            output["auto_apply"] = None

        return output

    def _auto_apply_probes(self, filename, csv_path):
        """Embed the generated probe set via TAGM's EmbeddingGenerator."""
        from tagm.probes.generator import EmbeddingGenerator, GenerationParams
        from tagm.probes.template import parse_template_csv

        try:
            template = parse_template_csv(Path(csv_path), name=filename)
        except Exception as e:
            logger.warning(f"[PROBE_GEN] Auto-apply: failed to parse template: {e}")
            return {"applied": False, "error": str(e)}

        adapter = self._pipeline.adapter
        n_layers = adapter.n_layers(self._pipeline.instruct_model)

        gen_params = GenerationParams(
            depth_layers={
                "subject": int(n_layers * 0.50),
                "escalation": int(n_layers * 0.75),
            },
            include_final_norm=True,
            filter_stopwords=True,
        )

        self._prog("Auto-apply: embedding probe set...")
        try:
            generator = EmbeddingGenerator(self._pipeline)
            probe_set = generator.generate(
                template, gen_params,
                progress=lambda stage, msg: self._prog(f"Auto-apply: {msg}"))
            self._probe_store.put(probe_set)
        except Exception as e:
            logger.exception(f"[PROBE_GEN] Auto-apply embed failed: {e}")
            return {"applied": False, "error": str(e)}

        self._prog(f"Auto-apply complete: {len(probe_set.probes)} probes "
                    f"at {len(probe_set.depth_labels)} depth(s)")

        return {
            "applied": True,
            "filename": filename,
            "set_id": probe_set.set_id,
            "n_probes": len(probe_set.probes),
            "depths": list(probe_set.depth_labels),
        }
