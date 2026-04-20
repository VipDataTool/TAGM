"""Probe template parsing.

Templates are CSV files in tagm/templates/ following TASM's format.
They have two common shapes:

  5x5 grid:   rows = classes, columns = subclasses, cells = comma-separated tokens.
  Two-column: first col = class, second col = comma-separated tokens
              (a "row × 1" degenerate case).

The parser tolerates blank rows, trims whitespace, splits on commas inside
cells, and deduplicates tokens per cell. Stopwords (from
tagm/templates/stopwords.txt) are filtered at probe-set generation time,
not here — the template itself remains the source of record.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TemplateCell:
    """One cell of a template: list of candidate tokens for (row, column)."""
    row: str
    column: str
    tokens: tuple[str, ...]


@dataclass
class ProbeTemplate:
    """A parsed probe template.

    Identity is via `template_id`, a content hash derived from the parsed
    content (rows, columns, tokens) — not the filename. Two templates with
    identical content but different filenames produce the same ID.
    """
    name: str                                # human-readable (often the filename stem)
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    cells: tuple[TemplateCell, ...]
    source_path: Optional[Path] = None
    template_id: str = ""

    def __post_init__(self):
        if not self.template_id:
            # Compute content hash
            content = {
                "rows": list(self.rows),
                "columns": list(self.columns),
                "cells": [
                    {"row": c.row, "column": c.column, "tokens": list(c.tokens)}
                    for c in self.cells
                ],
            }
            blob = json.dumps(content, sort_keys=True).encode()
            object.__setattr__(
                self, "template_id",
                hashlib.sha256(blob).hexdigest()[:16],
            )

    def cell(self, row: str, column: str) -> Optional[TemplateCell]:
        for c in self.cells:
            if c.row == row and c.column == column:
                return c
        return None

    def all_tokens(self) -> list[tuple[str, str, str]]:
        """Return [(row, column, token), ...] with duplicates across cells preserved.

        Each token appears once per cell it belongs to; if the same token
        is in multiple cells (e.g. "weapon" in both {firearms} and
        {weapons}), it's yielded once per cell so the generator can embed
        it separately per cell.
        """
        out = []
        for cell in self.cells:
            for tok in cell.tokens:
                out.append((cell.row, cell.column, tok))
        return out

    def __repr__(self) -> str:
        return (f"ProbeTemplate(name={self.name!r}, id={self.template_id!r}, "
                f"rows={len(self.rows)}, cols={len(self.columns)}, "
                f"cells={len(self.cells)}, "
                f"tokens={sum(len(c.tokens) for c in self.cells)})")


def parse_template_csv(path: Path, name: Optional[str] = None) -> ProbeTemplate:
    """Parse a CSV at `path` into a ProbeTemplate.

    Handles both the 5×5 grid format (first row is column headers) and the
    degenerate two-column format (no column headers; each row is class +
    comma-separated tokens).

    Empty cells are allowed (produce a TemplateCell with empty tokens).
    """
    path = Path(path)
    name = name or path.stem

    rows_data: list[list[str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not any((cell or "").strip() for cell in row):
                continue  # skip blank lines
            rows_data.append([(c or "").strip() for c in row])

    if not rows_data:
        return ProbeTemplate(name=name, rows=(), columns=(), cells=(),
                             source_path=path)

    # Heuristic: if the first cell of row 0 is empty, it's the 5×5 format
    # ("" | col1 | col2 | ...). Otherwise, two-column.
    first_row = rows_data[0]
    is_grid = (len(first_row) >= 2 and not first_row[0])

    if is_grid:
        columns = tuple(c for c in first_row[1:] if c)
        rows_out: list[str] = []
        cells_out: list[TemplateCell] = []
        for row_entry in rows_data[1:]:
            if not row_entry:
                continue
            row_label = row_entry[0]
            if not row_label:
                continue
            rows_out.append(row_label)
            for col_idx, col_label in enumerate(columns, start=1):
                if col_idx >= len(row_entry):
                    tokens: tuple[str, ...] = ()
                else:
                    raw = row_entry[col_idx]
                    tokens = tuple(_split_and_dedupe(raw))
                cells_out.append(TemplateCell(
                    row=row_label, column=col_label, tokens=tokens,
                ))
        return ProbeTemplate(
            name=name, rows=tuple(rows_out), columns=columns,
            cells=tuple(cells_out), source_path=path,
        )

    # Two-column format: row = [class_label, tokens...]
    rows_out: list[str] = []
    cells_out: list[TemplateCell] = []
    for row_entry in rows_data:
        if not row_entry:
            continue
        row_label = row_entry[0]
        if not row_label:
            continue
        rows_out.append(row_label)
        raw = ",".join(row_entry[1:]) if len(row_entry) > 1 else ""
        tokens = tuple(_split_and_dedupe(raw))
        cells_out.append(TemplateCell(row=row_label, column="items",
                                      tokens=tokens))
    return ProbeTemplate(
        name=name, rows=tuple(rows_out), columns=("items",),
        cells=tuple(cells_out), source_path=path,
    )


def _split_and_dedupe(raw: str) -> list[str]:
    """Split a comma-separated token cell into unique, stripped tokens.

    Preserves order of first appearance.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in raw.split(","):
        t = t.strip()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def load_template(name_or_path: str, templates_dir: Optional[Path] = None
                   ) -> ProbeTemplate:
    """Load a template by name (resolves to `templates_dir/<name>.csv`)
    or by path.

    `name_or_path` may be a bare name (e.g. "cybersecurity_5x5") or a
    path to a CSV file. If a bare name, `templates_dir` is required
    (defaults to the package's own templates dir if None).
    """
    if templates_dir is None:
        templates_dir = Path(__file__).parent.parent / "templates"
    p = Path(name_or_path)
    if p.exists() and p.is_file():
        return parse_template_csv(p)
    candidate = Path(templates_dir) / (
        name_or_path if name_or_path.endswith(".csv") else f"{name_or_path}.csv"
    )
    if candidate.exists():
        return parse_template_csv(candidate)
    raise FileNotFoundError(
        f"Probe template not found: '{name_or_path}' "
        f"(searched {templates_dir})")


def load_stopwords(templates_dir: Optional[Path] = None) -> set[str]:
    """Load the stopword set used to filter probe tokens during generation.

    Returns the empty set if stopwords.txt is missing.
    """
    if templates_dir is None:
        templates_dir = Path(__file__).parent.parent / "templates"
    path = Path(templates_dir) / "stopwords.txt"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}
