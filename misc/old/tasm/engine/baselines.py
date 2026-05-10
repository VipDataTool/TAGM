"""
Prompt library management.

Loads and manages the prompt CSV (prompts.csv) used for batch analysis
and the sidebar prompt library.
"""

import csv
from pathlib import Path
from typing import List

PROMPTS_FILE = Path(__file__).parent.parent / "prompts.csv"


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


def add_prompt(prompt: str, category: str = "benign"):
    """Append a prompt to prompts.csv."""
    with open(PROMPTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "category"])
        writer.writerow({"prompt": prompt, "category": category})
