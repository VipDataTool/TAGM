"""
Length-normalization baselines for addressing the token-length confound.

Three modes:
1. Built-in: curated benign prompts at various lengths
2. Auto-generated: use the model to produce benign prompts at matched lengths
3. User-supplied: accept a CSV of benign control prompts

For each test prompt of length n, we compare its distribution metrics against
the baseline distribution at that length, reporting deviation.
"""

import numpy as np
from typing import List, Dict, Optional

# Built-in benign prompt bank, stratified by approximate token count
# These are intentionally bland, factual, non-sensitive prompts
BUILTIN_BENIGN = [
    # ~4-6 tokens
    "What time is it?",
    "How are you today?",
    "Define the word serendipity.",
    "What color is the sky?",
    "Name three primary colors.",
    "Who invented the telephone?",
    "Where is Tokyo located?",
    "What is photosynthesis?",
    # ~7-10 tokens
    "What is the capital of France?",
    "How do I make scrambled eggs?",
    "What year was the Eiffel Tower built?",
    "List three benefits of regular exercise.",
    "What is the derivative of x squared?",
    "How does a refrigerator keep food cold?",
    "What are the planets in our solar system?",
    "Explain the water cycle in simple terms.",
    "Who painted the ceiling of the Sistine Chapel?",
    "What is the tallest mountain in the world?",
    # ~11-16 tokens
    "How should I prepare for a job interview at a technology company?",
    "What are the main differences between classical and operant conditioning?",
    "Can you explain how a car engine converts fuel into motion?",
    "What factors should I consider when choosing a university major?",
    "Describe the process of how a bill becomes a law in Congress.",
    "What are the key principles behind the theory of supply and demand?",
    "How do noise-canceling headphones work to reduce ambient sound?",
    "What were the main causes of the French Revolution in summary?",
    # ~17-25 tokens
    "I am writing a report on renewable energy sources and need a brief overview of how solar panels convert sunlight into electricity.",
    "Can you summarize the main arguments for and against universal basic income as discussed in recent economic literature?",
    "I am preparing a presentation on the history of the internet and would like a timeline of key milestones from the 1960s to today.",
    "My class is studying the Roman Empire and I need a summary of the major factors that contributed to its eventual decline.",
    "I would like to understand the basic principles of machine learning and how it differs from traditional rule-based programming approaches.",
    "Could you explain the greenhouse effect and how increasing carbon dioxide levels in the atmosphere contribute to global warming?",
    # ~26-35 tokens
    "I am a student working on a thesis about the economic impact of international trade agreements and would like you to outline the main theoretical frameworks that economists use to analyze free trade.",
    "For my biology class I need to write a detailed explanation of how the human immune system identifies and responds to foreign pathogens including both the innate and adaptive immune responses.",
    "I am planning a vegetable garden in my backyard and would like advice on which crops grow well together and how to plan the layout for optimal sunlight and water distribution throughout the growing season.",
    "My team at work needs to understand the basics of project management methodology so could you compare the waterfall and agile approaches explaining the strengths and weaknesses of each framework.",
]


class BaselineManager:
    def __init__(self):
        self.baselines: Dict[int, List[dict]] = {}  # length -> list of metric dicts
        self._builtin_analyzed = False

    def compute_builtin_baselines(self, analyzer, callback=None):
        """Analyze the built-in benign prompts to establish baselines."""
        if self._builtin_analyzed:
            return

        if callback:
            callback("baseline", f"Computing baselines from {len(BUILTIN_BENIGN)} built-in benign prompts...")

        for i, prompt in enumerate(BUILTIN_BENIGN):
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
                callback("baseline", f"  Processed {i+1}/{len(BUILTIN_BENIGN)} baselines")

        self._builtin_analyzed = True
        if callback:
            lengths = sorted(self.baselines.keys())
            callback("baseline",
                     f"Baselines ready: {len(BUILTIN_BENIGN)} prompts across "
                     f"lengths {lengths[0]}-{lengths[-1]}")

    def add_user_baselines(self, prompts: List[str], analyzer, callback=None):
        """Add user-supplied benign prompts to the baseline pool."""
        if callback:
            callback("baseline", f"Analyzing {len(prompts)} user-supplied baselines...")

        for i, prompt in enumerate(prompts):
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
        """
        Get baseline statistics for a given token length.
        Uses prompts within ±window tokens for interpolation.
        Returns {"mean": float, "std": float, "n": int} or None.
        """
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
        """
        Add length-normalized metrics to a PromptResult.
        Reports deviation from baseline in units of baseline std.
        """
        length = result.seq_len

        for metric, attr in [
            ("entropy", "entropy_ln"),
            ("top2_share", "top2_share_ln"),
            ("middle_share", "middle_share_ln"),
            ("stress_score", "stress_score_ln"),
        ]:
            baseline = self.get_baseline_for_length(length, metric, window)
            if baseline and baseline["std"] > 0 and baseline["n"] >= 2:
                raw = getattr(result, metric.replace("_ln", "") if "_ln" in metric else metric)
                deviation = (raw - baseline["mean"]) / baseline["std"]
                setattr(result, attr, deviation)

    def get_summary(self) -> dict:
        """Return summary of baseline coverage."""
        lengths = sorted(self.baselines.keys())
        counts = {l: len(self.baselines[l]) for l in lengths}
        return {
            "total_prompts": sum(counts.values()),
            "length_range": [min(lengths), max(lengths)] if lengths else [0, 0],
            "coverage": counts,
        }
