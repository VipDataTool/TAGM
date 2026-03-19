"""
Length-normalization baselines — OPTIONAL diagnostic overlay.

Baselines compute σ-deviation from expected values for benign prompts
at similar token lengths. They address the length confound (longer prompts
mechanically have higher interior share) but require running the analyzer
on each baseline prompt, which takes time and assumes the baseline set
is representative.

This module is opt-in:
  - Baselines are NOT computed at model load time.
  - The user enables them via a toggle, which triggers computation.
  - The baseline prompts come from baselines.csv (auditable, one column: prompt).
  - When disabled, all _ln fields are None throughout the pipeline.

The prompt library (prompts.csv) is fully separate from baselines.
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

BASELINES_FILE = Path(__file__).parent.parent / "baselines.csv"
PROMPTS_FILE = Path(__file__).parent.parent / "prompts.csv"


# ─── Baseline source ─────────────────────────────────────────────

def load_baselines_csv() -> List[str]:
    """Load baseline prompts from baselines.csv (single column: prompt).
    This is the auditable source — every prompt in this file is a known-benign
    reference used for length normalization. Edit this file to change coverage."""
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


# ─── Baseline manager ────────────────────────────────────────────

class BaselineManager:
    """Opt-in length-normalization baseline system.

    When enabled=True and baselines have been computed, normalize_result()
    adds _ln fields to PromptResult. When enabled=False, normalize_result()
    is a no-op and all _ln fields remain None.

    The baselines dict maps token_length -> list of metric dicts, computed
    by running the analyzer on every prompt in baselines.csv.
    """

    def __init__(self):
        self.baselines: Dict[int, List[dict]] = {}
        self.enabled: bool = False
        self._computed: bool = False

    def compute_baselines(self, analyzer, callback=None):
        """Analyze all prompts from baselines.csv to establish length norms.
        This is expensive (one forward pass per prompt) and should only run
        when the user explicitly opts in."""
        self.baselines.clear()
        self._computed = False

        baseline_prompts = load_baselines_csv()
        if not baseline_prompts:
            if callback:
                callback("baseline", "No prompts found in baselines.csv")
            return

        if callback:
            callback("baseline", f"Computing baselines from {len(baseline_prompts)} prompts (baselines.csv)...")

        for i, prompt in enumerate(baseline_prompts):
            try:
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
            except Exception as e:
                if callback:
                    callback("warning", f"Baseline prompt {i+1} failed: {str(e)[:60]}")

            if callback and (i + 1) % 10 == 0:
                callback("baseline", f"  Processed {i+1}/{len(baseline_prompts)} baselines")

        self._computed = True
        self.enabled = True
        if callback:
            lengths = sorted(self.baselines.keys())
            if lengths:
                callback("baseline",
                         f"Baselines ready: {len(baseline_prompts)} prompts across "
                         f"token lengths {lengths[0]}-{lengths[-1]} "
                         f"({len(lengths)} unique lengths)")

    def add_user_baselines(self, prompts: List[str], analyzer, callback=None):
        """Add user-supplied baseline prompts on top of baselines.csv."""
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
        """Add _ln fields if baselines are enabled and computed.
        When disabled, this is a no-op — _ln fields stay None."""
        if not self.enabled or not self._computed:
            return

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
            "enabled": self.enabled,
            "computed": self._computed,
            "total_prompts": sum(counts.values()),
            "length_range": [min(lengths), max(lengths)] if lengths else [0, 0],
            "coverage": counts,
            "source": str(BASELINES_FILE),
        }
