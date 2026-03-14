"""
Length-normalization baselines for addressing the token-length confound.

Reads baseline prompts from prompts.csv (rows with baseline=yes).
User-supplied baselines can be added at runtime.
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

PROMPTS_FILE = Path(__file__).parent.parent / "prompts.csv"


def load_prompts_csv() -> list:
    """Load unified prompt library from prompts.csv."""
    if not PROMPTS_FILE.exists():
        return []
    prompts = []
    with open(PROMPTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                "prompt": row.get("prompt", ""),
                "category": row.get("category", "benign"),
                "baseline": row.get("baseline", "no").strip().lower() in ("yes", "true", "1"),
            })
    return prompts


def get_baseline_prompts() -> List[str]:
    """Return just the baseline prompt texts."""
    return [p["prompt"] for p in load_prompts_csv() if p["baseline"]]


def get_test_prompts() -> list:
    """Return non-baseline prompts as [{prompt, category}]."""
    return [{"prompt": p["prompt"], "category": p["category"]}
            for p in load_prompts_csv() if not p["baseline"]]


def get_all_prompts() -> list:
    """Return all prompts as [{prompt, category, baseline}]."""
    return load_prompts_csv()


def add_prompt(prompt: str, category: str = "benign", baseline: bool = False):
    """Append a prompt to prompts.csv."""
    with open(PROMPTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "category", "baseline"])
        writer.writerow({
            "prompt": prompt,
            "category": category,
            "baseline": "yes" if baseline else "no",
        })


class BaselineManager:
    def __init__(self):
        self.baselines: Dict[int, List[dict]] = {}
        self._builtin_analyzed = False

    def compute_builtin_baselines(self, analyzer, callback=None):
        """Analyze baseline prompts from prompts.csv to establish length norms."""
        if self._builtin_analyzed:
            return

        baseline_prompts = get_baseline_prompts()
        if not baseline_prompts:
            if callback:
                callback("baseline", "No baseline prompts found in prompts.csv")
            return

        if callback:
            callback("baseline", f"Computing baselines from {len(baseline_prompts)} prompts...")

        for i, prompt in enumerate(baseline_prompts):
            result = analyzer.analyze_prompt(prompt, category="baseline",
                                             compute_kl=False,
                                             compute_full_trajectory=False)
            length = result.seq_len
            if length not in self.baselines:
                self.baselines[length] = []

            self.baselines[length].append({
                "entropy": result.entropy,
                "gini": result.gini,
                "top2_share": result.top2_share,
                "middle_share": result.middle_share,
                "interior_cv": result.interior_cv,
                "stress_score": result.stress_score,
                "net_correction": result.net_correction,
            })

            if callback and (i + 1) % 10 == 0:
                callback("baseline", f"  Processed {i+1}/{len(baseline_prompts)} baselines")

        self._builtin_analyzed = True
        if callback:
            lengths = sorted(self.baselines.keys())
            if lengths:
                callback("baseline",
                         f"Baselines ready: {len(baseline_prompts)} prompts across "
                         f"lengths {lengths[0]}-{lengths[-1]}")

    def add_user_baselines(self, prompts: List[str], analyzer, callback=None):
        if callback:
            callback("baseline", f"Analyzing {len(prompts)} user-supplied baselines...")
        for prompt in prompts:
            result = analyzer.analyze_prompt(prompt, category="user_baseline",
                                             compute_kl=False,
                                             compute_full_trajectory=False)
            length = result.seq_len
            if length not in self.baselines:
                self.baselines[length] = []
            self.baselines[length].append({
                "entropy": result.entropy,
                "gini": result.gini,
                "top2_share": result.top2_share,
                "middle_share": result.middle_share,
                "interior_cv": result.interior_cv,
                "stress_score": result.stress_score,
                "net_correction": result.net_correction,
            })

    def get_baseline_for_length(self, length: int, metric: str,
                                 window: int = 3) -> Optional[dict]:
        values = []
        for l in range(length - window, length + window + 1):
            if l in self.baselines:
                values.extend([b[metric] for b in self.baselines[l]])
        if not values:
            return None
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    def normalize_result(self, result, window: int = 3):
        length = result.seq_len
        for metric, attr in [
            ("entropy", "entropy_ln"),
            ("top2_share", "top2_share_ln"),
            ("middle_share", "middle_share_ln"),
            ("stress_score", "stress_score_ln"),
        ]:
            baseline = self.get_baseline_for_length(length, metric, window)
            if baseline and baseline["std"] > 0 and baseline["n"] >= 2:
                raw = getattr(result, metric)
                deviation = (raw - baseline["mean"]) / baseline["std"]
                setattr(result, attr, deviation)

    def get_summary(self) -> dict:
        lengths = sorted(self.baselines.keys())
        counts = {l: len(self.baselines[l]) for l in lengths}
        return {
            "total_prompts": sum(counts.values()),
            "length_range": [min(lengths), max(lengths)] if lengths else [0, 0],
            "coverage": counts,
        }
