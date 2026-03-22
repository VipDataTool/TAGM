"""
Prompt library management.

Loads and manages the prompt CSV (prompts.csv) used for batch analysis.
Also provides access to baselines.csv as a source of known-benign prompts.

Length normalization is handled by regression-based residualization in
engine/statistics.py, not by this module.
"""

import csv
from pathlib import Path
from typing import List

BASELINES_FILE = Path(__file__).parent.parent / "baselines.csv"
PROMPTS_FILE = Path(__file__).parent.parent / "prompts.csv"


# ─── Baseline source ─────────────────────────────────────────────

def load_baselines_csv() -> List[str]:
    """Load baseline prompts from baselines.csv (single column: prompt).
    These are known-benign reference prompts. They can be used as a prompt
    source for batch runs but are no longer used for z-score normalization."""
    if not BASELINES_FILE.exists():
        return []
    prompts = []
    with open(BASELINES_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("prompt", "").strip()
            if p:
                prompts.append(p)
    return prompts


# ─── Prompt library (separate from baselines) ────────────────────

def load_prompts_csv() -> list:
    """Load the prompt library from prompts.csv."""
    if not PROMPTS_FILE.exists():
        return []
    prompts = []
    with open(PROMPTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                "prompt": row.get("prompt", ""),
                "category": row.get("category", "benign"),
            })
    return prompts


def get_all_prompts() -> list:
    """Return all prompts from the prompt library."""
    return load_prompts_csv()


def add_prompt(prompt: str, category: str = "benign", baseline: bool = False):
    """Append a prompt to prompts.csv."""
    with open(PROMPTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "category"])
        writer.writerow({
            "prompt": prompt,
            "category": category,
        })
