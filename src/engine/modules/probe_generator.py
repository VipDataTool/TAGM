"""
Probe Generator Module for TASM.

Generates a discriminative probe set by statistically sampling the
loaded model's output distribution per class × subclass cell.

Process:
  1. Read a template CSV — any CSV with a 'subject' column and one
     or more subclass columns (everything after subject/anchor_id).
     Cell values are preserved as seed phrases for prompt steering.
  2. For each (class, subclass) cell, query the model N times
     using a prompt steered toward that specific cell
  3. Extract terms from all responses, count per-cell frequencies
  4. Frequency filter: discard terms below minimum threshold
  5. Discriminative dedup: any term present in more than one cell
     (anywhere in the lattice — same row, same column, or diagonal)
     is removed from all cells. Cross-class / cross-subclass /
     diagonal breakdowns are reported as diagnostics.
  6. Export flat CSV with columns: subject, subclass, text
     One record per probe.

Global uniqueness is the discriminative filter.  A term must be
unique to its cell; what survives is the model-specific vocabulary
fingerprint for each cell in the class × subclass lattice.

No dependencies on other TASM modules.  Reads any conforming CSV.
"""

import os
import csv
import logging
import re
import time
from collections import Counter

from src.engine import config as engine_config
from src.core.locks import MODEL_LOCK

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")

FIXED_COLS = {"subject", "anchor_id"}
META_TAG = "_meta"


def _load_stopwords(path):
    """Load stopword list from a text file.

    Returns (frozenset of words, raw line count).
    Logs every word loaded so filtering is fully transparent.
    Raises FileNotFoundError — callers decide how to handle.
    """
    words = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            w = line.lower().split()[0]  # first token only, ignore inline comments
            words.append(w)
    result = frozenset(words)
    logger.info(f"[STOPWORDS] Loaded {len(result)} words from {path}")
    return result, len(result)


# ─── Template Parsing (self-contained) ────────────────────────

def _parse_template(csv_path):
    """Parse a template CSV into classes, subclass columns, and per-cell seeds.

    Returns:
        classes:      list of class names (ordered by first appearance)
        subclass_cols: list of subclass column names
        cell_seeds:   dict of (class, subclass_col) -> seed phrase string
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
                    cell_seeds[(cls, col)] = text

    return classes, subclass_cols, cell_seeds


# ─── Tokenization ─────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*")
_TERM_LINE_RE = re.compile(r"^[a-zA-Z][a-zA-Z' \-]*[a-zA-Z]$")


def _tokenize_response(text, stopwords=frozenset()):
    """Extract terms, honoring the prompt's one-term-per-line contract.

    Primary path: each response line is a candidate term after list
    markup is stripped (bullets, numbering, trailing punctuation).
    Lines of one to five words survive as WHOLE terms — the multi-word
    vocabulary the prompt explicitly requests is preserved instead of
    being shredded into single words. Single-word terms pass the
    stopword and length filters; a multi-word term is kept if at least
    one of its words is a non-stopword.

    Fallback: a response that doesn't look like a list (fewer than 3
    line-terms extracted) is word-shredded with the historical regex,
    so prose answers still yield vocabulary.
    """
    terms = Counter()
    for raw in text.splitlines():
        t = raw.strip()
        t = re.sub(r"^[\-\*\u2022\u00b7>]+\s*", "", t)   # bullets
        t = re.sub(r"^\d+[\.\):]\s*", "", t)              # numbering
        t = t.strip().strip(".,;:").strip().lower()
        if len(t) < 3 or not _TERM_LINE_RE.match(t):
            continue
        words = t.split()
        if len(words) > 5:
            continue
        if len(words) == 1:
            if len(t) < 3 or t in stopwords:
                continue
            terms[t] += 1
        else:
            if all(w in stopwords for w in words):
                continue
            terms[t] += 1
    if sum(terms.values()) >= 3:
        return terms

    # Fallback: word-shred prose
    words = Counter()
    for w in _WORD_RE.findall(text.lower()):
        if len(w) < 3 or w in stopwords:
            continue
        words[w] += 1
    return words


def _seed_filter(counts, seeds):
    """Remove tokens that appear verbatim in the cell's own seed phrase.

    Seed phrases steer the prompt, and the model reliably echoes them
    back — so seed words masquerade as model vocabulary and, worse, as
    DISCRIMINATIVE vocabulary, since each cell's seeds are unique to it
    by construction. Returns (counts, removed_counter).
    """
    if not seeds:
        return counts, Counter()
    s = seeds.lower()
    seed_words = set(_WORD_RE.findall(s))
    removed = Counter()
    for t in list(counts):
        if t in seed_words or (" " in t and t in s):
            removed[t] += counts[t]
            del counts[t]
    return counts, removed


# ─── Discriminative filtering ─────────────────────────────────

def _dedup(cell_vocab, classes, subclass_cols, min_freq, skip_dedup):
    """Frequency filter + discriminative (global-uniqueness) dedup.

    The discriminative contract is "a term must be unique to its cell."
    The historical two-axis formulation (cross-class within a subclass
    column, cross-subclass within a class) misses DIAGONAL collisions:
    a token appearing in exactly two cells that share neither row nor
    column — e.g. (physics, technical) and (chemistry, casual) — was
    flagged by neither axis and survived in both cells, violating the
    contract. Uniqueness is therefore enforced globally: any token
    present in more than one frequency-filtered cell is removed from
    all of them. The axis sets are still computed as diagnostics, and
    tokens caught only by the global rule are reported separately as
    the diagonal set.

    Returns (discriminative, cross_class, cross_subclass, diagonal),
    where discriminative maps (class, subclass_col) -> Counter.
    """
    filtered = {}
    for key in cell_vocab:
        filtered[key] = Counter({
            tok: cnt for tok, cnt in cell_vocab[key].items()
            if cnt >= min_freq
        })
    if skip_dedup:
        return filtered, set(), set(), set()

    # Global presence: how many cells contain each token
    presence = Counter()
    for key in filtered:
        for tok in filtered[key]:
            presence[tok] += 1
    shared_global = {tok for tok, c in presence.items() if c > 1}

    # Axis diagnostics (subsets of shared_global)
    xclass = set()
    for col in subclass_cols:
        tc = Counter()
        for cls in classes:
            for tok in filtered.get((cls, col), {}):
                tc[tok] += 1
        for tok, cnt in tc.items():
            if cnt > 1:
                xclass.add(tok)
    xsub = set()
    for cls in classes:
        ts = Counter()
        for col in subclass_cols:
            for tok in filtered.get((cls, col), {}):
                ts[tok] += 1
        for tok, cnt in ts.items():
            if cnt > 1:
                xsub.add(tok)
    diagonal = shared_global - (xclass | xsub)

    disc = {}
    for cls in classes:
        for col in subclass_cols:
            key = (cls, col)
            disc[key] = Counter({
                tok: cnt for tok, cnt in filtered.get(key, {}).items()
                if tok not in shared_global
            })
    return disc, xclass, xsub, diagonal


# ─── Prompt Construction ──────────────────────────────────────

DEFAULT_PROMPT_TEMPLATE = (
    "List {word_count} domain-specific terms, jargon, and technical "
    "vocabulary related to {class} in the context of {subclass}. "
    "Include specialized terminology, contextual phrases, and "
    "insider language. Seed context: {seeds}. "
    "One term per line."
)


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
    version = "0.4.0"

    min_results = 0
    requires_sfd = False
    requires_ltp = False
    requires_rd = False

    def __init__(self):
        super().__init__()
        self._pipeline = None
        self._project_root = None

    def set_project_root(self, root):
        self._project_root = root

    def set_pipeline(self, pipeline):
        """Provide access to the loaded pipeline (model, tokenizer, adapter)."""
        self._pipeline = pipeline

    # Legacy compatibility
    def set_model(self, model, tokenizer):
        pass  # use set_pipeline instead

    def set_model_manager(self, mm):
        pass  # use set_pipeline instead

    @property
    def _active_model(self):
        if self._pipeline is not None:
            return self._pipeline.instruct_model
        return None

    @property
    def _active_tokenizer(self):
        if self._pipeline is not None:
            return self._pipeline.tokenizer
        return None

    @property
    def _active_adapter(self):
        if self._pipeline is not None:
            return self._pipeline.adapter
        return None

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="template_file",
                group="Template & Output",
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
                group="Template & Output",
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
                group="Sampling",
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
                group="Sampling",
                display_name="Max Tokens Per Response",
                description="Maximum tokens the model generates per query",
                type="int",
                default=256,
                min_val=64,
                max_val=512,
            ),
            ModuleParameter(
                name="temperature",
                group="Sampling",
                display_name="Temperature",
                description=(
                    "Sampling temperature. Lower values (0.5-0.7) produce "
                    "tighter, more predictable domain vocabulary. Higher "
                    "values (0.9-1.2) produce more diverse but noisier output. "
                    "Affects vocabulary richness per cell."
                ),
                type="float",
                default=0.9,
                min_val=0.1,
                max_val=2.0,
            ),
            ModuleParameter(
                name="top_p",
                group="Sampling",
                advanced=True,
                display_name="Top-P (Nucleus Sampling)",
                description=(
                    "Nucleus sampling threshold. Limits token selection to "
                    "the smallest set whose cumulative probability exceeds "
                    "this value. Lower = more focused, higher = more diverse."
                ),
                type="float",
                default=0.95,
                min_val=0.1,
                max_val=1.0,
            ),
            ModuleParameter(
                name="repetition_penalty",
                group="Sampling",
                advanced=True,
                display_name="Repetition Penalty",
                description=(
                    "Penalty for repeating tokens in generated text. "
                    "1.0 = no penalty. Higher values discourage repetition, "
                    "encouraging more diverse vocabulary per response."
                ),
                type="float",
                default=1.1,
                min_val=1.0,
                max_val=2.0,
            ),
            ModuleParameter(
                name="min_frequency",
                group="Discriminative Filter",
                display_name="Minimum Frequency",
                description=(
                    "Minimum times a token must appear across all queries "
                    "for a token to be kept. Filters sampling noise."
                ),
                type="int",
                default=3,
                min_val=1,
                max_val=20,
            ),
            ModuleParameter(
                name="prompt_template",
                group="Sampling",
                advanced=True,
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
                group="Diagnostics",
                advanced=True,
                display_name="Export Inference Catalog",
                description=(
                    "Save a CSV log of every model query and response. "
                    "Useful for auditing prompt quality."
                ),
                type="bool",
                default=False,
            ),
            ModuleParameter(
                name="stopword_file",
                group="Discriminative Filter",
                display_name="Stopword File",
                description=(
                    "Text file of words to exclude before discriminative "
                    "filtering. One word per line, '#' for comments. "
                    "Leave empty to disable stopword filtering. "
                    "Filtered tokens are logged in output metadata."
                ),
                type="file",
                default="templates/stopwords.txt",
            ),
            ModuleParameter(
                name="min_probes_per_cell",
                group="Cell Resolution",
                display_name="Min Probes Per Cell",
                description=(
                    "Minimum discriminative tokens required per cell after "
                    "deduplication. Thin cells are automatically re-queried "
                    "to reach this threshold. Set to 0 to disable. "
                    "Higher values ensure uniform detector resolution."
                ),
                type="int",
                default=5,
                min_val=0,
                max_val=50,
            ),
            ModuleParameter(
                name="max_probes_per_cell",
                group="Cell Resolution",
                display_name="Max Probes Per Cell",
                description=(
                    "Truncate each cell to this many probes (highest "
                    "frequency first). Ensures uniform detector resolution "
                    "across the lattice — every cell gets equal statistical "
                    "weight. Set to 0 to disable (keep all). Works with "
                    "min probes to define a resolution band."
                ),
                type="int",
                default=10,
                min_val=0,
                max_val=100,
            ),
            ModuleParameter(
                name="auto_populate_batch",
                group="Cell Resolution",
                advanced=True,
                display_name="Auto-Populate Queries Per Round",
                description=(
                    "Number of additional queries to run per thin cell "
                    "per auto-populate round. Only used when min probes "
                    "per cell is set above 0."
                ),
                type="int",
                default=5,
                min_val=1,
                max_val=50,
            ),
            ModuleParameter(
                name="auto_populate_max_rounds",
                group="Cell Resolution",
                advanced=True,
                display_name="Auto-Populate Max Rounds",
                description=(
                    "Maximum number of re-query rounds for thin cells. "
                    "Total extra queries per cell is at most "
                    "batch × rounds. Set lower to cap compute cost."
                ),
                type="int",
                default=3,
                min_val=1,
                max_val=10,
            ),
            ModuleParameter(
                name="skip_dedup",
                group="Discriminative Filter",
                advanced=True,
                display_name="Skip Discriminative Deduplication",
                description=(
                    "Bypass the two-axis dedup filter. Keeps every token "
                    "that passes the frequency floor, including tokens "
                    "shared across classes and across subclasses. Produces "
                    "a raw probe distribution for comparison against the "
                    "discriminative version. Cells will overlap in "
                    "vocabulary — use only for diagnostic runs."
                ),
                type="bool",
                default=False,
            ),
            ModuleParameter(
                name="auto_embed",
                group="Template & Output",
                display_name="Auto-Embed After Generation",
                description=(
                    "Automatically embed and activate the generated probe "
                    "set when generation completes. Embeds at configured "
                    "depths (L50/L75) and caches to probe_cache/. "
                    "Uncheck to generate the CSV only."
                ),
                type="bool",
                default=True,
            ),
        ]

    def validate(self, session_results, params):
        if self._active_model is None or self._active_tokenizer is None:
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

        # Validate stopword file if specified
        sw_file = params.get("stopword_file", "").strip()
        if sw_file:
            sw_path = sw_file
            if not os.path.isabs(sw_path) and self._project_root:
                sw_path = os.path.join(self._project_root, sw_path)
            if not os.path.exists(sw_path):
                return False, f"Stopword file not found: {sw_file}"

        return True, "OK"

    def run(self, session_results, params, progress=None):
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
        auto_embed = bool(params.get("auto_embed", True))

        # No concurrent inference lock needed — module runs in its own thread
        _inf_lock = None

        # ── Load stopwords ──
        sw_file = params.get("stopword_file", "").strip()
        stopwords = frozenset()
        stopword_path = None
        stopword_count = 0
        if sw_file:
            stopword_path = sw_file
            if not os.path.isabs(stopword_path) and self._project_root:
                stopword_path = os.path.join(self._project_root, stopword_path)
            stopwords, stopword_count = _load_stopwords(stopword_path)
            if progress:
                progress(f"Loaded {stopword_count} stopwords from {sw_file}")
        else:
            if progress:
                progress("No stopword file specified — stopword filtering disabled")
            logger.warning("[STOPWORDS] No stopword file specified. Filtering disabled.")

        # Track which tokens get filtered for auditability
        stopword_hits = Counter()  # token -> times filtered
        seed_hits = Counter()      # token -> occurrences removed as seed echo

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

        # Terms requested in the prompt, derived from the token budget:
        # a term plus newline costs roughly 5-8 tokens, so asking for
        # max_tokens terms (the old behavior) requested a list several
        # times longer than the budget could hold.
        ask_terms = max(10, max_tokens // 6)

        # ── Sample model output distribution per cell ──
        import torch
        device = next(self._active_model.parameters()).device

        cell_vocab = {}  # (class, subclass_col) -> Counter
        catalog = []     # [{class, subclass, query, prompt, response, tokens}]
        t0 = time.time()
        cell_idx = 0

        for cls in classes:
            for col in subclass_cols:
                cell_idx += 1
                sub_label = col.replace("_", " ")
                seeds = cell_seeds.get((cls, col), cls.replace("_", " "))
                vocab = Counter()

                for qi in range(n_queries):
                    if progress and (qi + 1) % 5 == 0:
                        progress(f"[{cell_idx}/{total_cells}] "
                                 f"{cls} × {sub_label}: "
                                 f"query {qi+1}/{n_queries}")

                    prompt_text = _build_cell_prompt(
                        cls, col, seeds, self._active_tokenizer,
                        word_count=ask_terms,
                        template=prompt_template)

                    # Acquire inference lock to prevent concurrent model access
                    if _inf_lock:
                        _inf_lock.acquire()
                    try:
                        inputs = self._active_tokenizer(
                            prompt_text, return_tensors="pt",
                            add_special_tokens=engine_config.get("add_special_tokens"),
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with MODEL_LOCK, torch.no_grad():
                            out = self._active_model.generate(
                                **inputs,
                                max_new_tokens=max_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                repetition_penalty=rep_penalty,
                            )
                        gen_ids = out[0][inputs["input_ids"].shape[1]:]
                        response = self._active_tokenizer.decode(
                            gen_ids, skip_special_tokens=True)
                    finally:
                        if _inf_lock:
                            _inf_lock.release()
                    word_counts = _tokenize_response(response, stopwords)
                    word_counts, _seed_rm = _seed_filter(word_counts, seeds)
                    seed_hits.update(_seed_rm)
                    vocab.update(word_counts)

                    # Track stopword hits for audit trail
                    if stopwords:
                        for w in re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", response.lower()):
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

                if progress:
                    progress(f"[{cell_idx}/{total_cells}] "
                             f"{cls} × {sub_label}: "
                             f"{len(vocab)} unique tokens")

        elapsed_sampling = time.time() - t0

        # ── Dedup pipeline (module-level; see _dedup) ──
        # Initial dedup pass
        discriminative, cross_class_shared, cross_subclass_shared, \
            diagonal_shared = _dedup(cell_vocab, classes, subclass_cols,
                                     min_freq, skip_dedup)
        all_shared = (cross_class_shared | cross_subclass_shared
                      | diagonal_shared)

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
                if progress:
                    progress(f"Auto-populate round {round_num+1}: "
                             f"{len(thin)} cells below {min_probes}, "
                             f"querying {ap_batch} more each...")

                for cls, col in thin:
                    sub_label = col.replace("_", " ")
                    seeds = cell_seeds.get((cls, col), cls.replace("_", " "))

                    for qi in range(ap_batch):
                        if progress:
                            progress(f"Auto-populate: {cls} × {sub_label} "
                                     f"(+{qi+1}/{ap_batch})")

                        prompt_text = _build_cell_prompt(
                            cls, col, seeds, self._active_tokenizer,
                            word_count=ask_terms,
                            template=prompt_template)

                        if _inf_lock:
                            _inf_lock.acquire()
                        try:
                            inputs = self._active_tokenizer(
                                prompt_text, return_tensors="pt",
                                add_special_tokens=engine_config.get("add_special_tokens"),
                            )
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            with MODEL_LOCK, torch.no_grad():
                                out = self._active_model.generate(
                                    **inputs,
                                    max_new_tokens=max_tokens,
                                    do_sample=True,
                                    temperature=temperature,
                                    top_p=top_p,
                                    repetition_penalty=rep_penalty,
                                )
                            gen_ids = out[0][inputs["input_ids"].shape[1]:]
                            response = self._active_tokenizer.decode(
                                gen_ids, skip_special_tokens=True)
                        finally:
                            if _inf_lock:
                                _inf_lock.release()
                        word_counts = _tokenize_response(response, stopwords)
                        word_counts, _seed_rm = _seed_filter(word_counts, seeds)
                        seed_hits.update(_seed_rm)
                        cell_vocab[(cls, col)].update(word_counts)

                        if stopwords:
                            for w in re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", response.lower()):
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

                # Re-run full dedup after adding new vocabulary
                discriminative, cross_class_shared, cross_subclass_shared, \
                    diagonal_shared = _dedup(cell_vocab, classes,
                                             subclass_cols, min_freq,
                                             skip_dedup)
                all_shared = (cross_class_shared | cross_subclass_shared
                              | diagonal_shared)

            if progress and auto_populate_rounds > 0:
                final_cnts = {(cls, col): len(discriminative[(cls, col)])
                              for cls in classes for col in subclass_cols}
                still_thin = sum(1 for v in final_cnts.values()
                                 if 0 < v < min_probes)
                progress(f"Auto-populate complete: {auto_populate_rounds} rounds, "
                         f"{auto_populate_queries} extra queries, "
                         f"{still_thin} cells still below {min_probes}")

        # ── Truncate to max probes per cell ──
        truncated_count = 0
        if max_probes > 0:
            for cls in classes:
                for col in subclass_cols:
                    key = (cls, col)
                    if len(discriminative[key]) > max_probes:
                        truncated_count += len(discriminative[key]) - max_probes
                        # Keep top-n by frequency (most_common)
                        discriminative[key] = Counter(
                            dict(discriminative[key].most_common(max_probes)))
            if progress and truncated_count > 0:
                progress(f"Truncated {truncated_count} excess probes "
                         f"(max {max_probes} per cell)")

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
                    empty_cells.append(f"{cls} \u00d7 {sub_label}")
                elif min_probes > 0 and n < min_probes:
                    thin_cells.append(f"{cls} \u00d7 {sub_label} ({n}/{min_probes})")

        if progress and (empty_cells or thin_cells):
            if empty_cells:
                progress(f"Warning: {len(empty_cells)} empty cells: "
                         f"{', '.join(empty_cells[:5])}"
                         f"{'...' if len(empty_cells) > 5 else ''}")
            if thin_cells:
                progress(f"Warning: {len(thin_cells)} cells below minimum "
                         f"({min_probes}): {', '.join(thin_cells[:5])}"
                         f"{'...' if len(thin_cells) > 5 else ''}")

        # Cell resolution stats for output
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
            "auto_populate_batch": ap_batch,
            "auto_populate_max_rounds": ap_max_rounds,
            "per_cell": {f"{cls} \u00d7 {col.replace('_',' ')}": cell_counts.get((cls, col.replace('_',' ')), 0)
                         for cls in classes for col in subclass_cols},
        }

        # ── Export CSV ──
        out_path = output_name
        if not os.path.isabs(out_path) and self._project_root:
            out_path = os.path.join(self._project_root, out_path)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "subclass", "text"])

            for cls in classes:
                for col in subclass_cols:
                    tokens = [
                        tok for tok, _ in
                        discriminative[(cls, col)].most_common()
                    ]
                    for tok in tokens:
                        writer.writerow([cls, col, tok])

        # ── Per-cell catalog (canonical, full data) ──
        # Full responses and full token lists, organized by cell.
        # This is the authoritative audit trail — the JSON module log
        # is the data model, the CSV export is a pure derivation.
        cell_catalog = {}
        for entry in catalog:
            key = f"{entry['class']}|{entry['subclass']}"
            cell_catalog.setdefault(key, []).append({
                "q": entry["query"],
                "response": entry["response"],
                "tokens_extracted": list(entry["tokens_extracted"]),
            })

        # ── Export catalog CSV (optional, derived from cell_catalog) ──
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
                for key, entries in cell_catalog.items():
                    cls, col = key.split("|", 1)
                    for entry in entries:
                        writer.writerow([
                            cls,
                            col,
                            entry["q"],
                            entry["response"],
                            " ".join(entry["tokens_extracted"]),
                        ])

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

        # Determine which model class was used
        inference_class = "unknown"
        if self._pipeline is not None:
            inference_class = "instruct"  # probe generation always uses instruct model

        output = {
            "template_file": template_file,
            "output_file": output_name,
            "inference_class": inference_class,
            "subjects": classes,
            "levels": subclass_names,
            "queries_per_cell": n_queries,
            "queries_per_subject": n_queries,  # backward compat
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
            "cross_diagonal_shared": len(diagonal_shared),
            "total_discriminative": total_disc,
            "per_subject": per_class,  # backward compat
            "shared_tokens": sorted(all_shared),
            "catalog_file": catalog_name,  # None if export disabled
            "catalog": cell_catalog,
            "total_queries": len(catalog),
            # Stopword audit trail
            "stopword_file": sw_file or None,
            "stopword_count": stopword_count,
            "stopwords_loaded": sorted(stopwords) if stopwords else [],
            "stopword_hits": dict(stopword_hits.most_common()),
            "stopword_total_filtered": sum(stopword_hits.values()),
            # Seed-echo audit trail
            "seed_hits": dict(seed_hits.most_common()),
            "seed_total_filtered": sum(seed_hits.values()),
            # Cell resolution analysis
            "cell_resolution": cell_resolution,
        }

        if progress:
            sw_msg = ""
            if stopwords:
                sw_msg = (f", {sum(stopword_hits.values())} stopword "
                          f"occurrences filtered ({len(stopword_hits)} "
                          f"unique from {sw_file})")
            progress(f"Exported {output_name}: {total_disc} discriminative "
                     f"tokens ({len(cross_class_shared)} shared across "
                     f"classes, {len(cross_subclass_shared)} shared across "
                     f"subclasses) across {total_cells} cells{sw_msg}")

        # ── Auto-embed ──
        if auto_embed and self._pipeline is not None and self._pipeline.loaded:
            try:
                from src.probes.io import embed_and_activate_probe_set

                if progress:
                    progress(f"Auto-embedding {output_name}...")

                project_root = self._project_root or os.getcwd()
                embed_result = embed_and_activate_probe_set(
                    self._pipeline, str(project_root), output_name,
                    progress=progress,
                )

                if embed_result.get("applied"):
                    output["auto_embed"] = {
                        "applied": True,
                        "filename": embed_result.get("filename"),
                        "n_probes": embed_result.get("n_probes"),
                        "n_subjects": embed_result.get("n_subjects"),
                        "n_levels": embed_result.get("n_levels"),
                        "depths": embed_result.get("depths", []),
                    }
                    if progress:
                        progress(
                            f"Embedded and activated: "
                            f"{embed_result['n_probes']} probes at depths "
                            f"L{', L'.join(str(d) for d in embed_result.get('depths', []))}")
                else:
                    output["auto_embed"] = {
                        "applied": False,
                        "error": embed_result.get("error", "Unknown error"),
                    }
                    if progress:
                        progress(f"Auto-embed failed: {embed_result.get('error', 'Unknown')}")

            except Exception as e:
                logger.exception("Auto-embed failed")
                output["auto_embed"] = {"applied": False, "error": str(e)}
                if progress:
                    progress(f"Auto-embed error: {e}")
        elif auto_embed:
            output["auto_embed"] = {"applied": False, "error": "No model loaded"}

        return output
