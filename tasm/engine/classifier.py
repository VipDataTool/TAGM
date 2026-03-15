"""
ASM Classifier: 4-class interpretable prompt classification from alignment geometry.

Uses weight-delta metrics computed from a single forward pass to classify prompts
as benign, mild, harmful, or jailbreak. Every classification comes with a per-feature
explanation showing which signals drove the decision and by how much.

Architecture: Gaussian log-likelihood scorecard.
- Each feature's contribution is the log-likelihood ratio of the observed value
  under each class's empirical distribution vs the overall distribution.
- The class with the highest total score wins.
- This is equivalent to Naive Bayes with Gaussian priors, but presented as
  an additive scorecard for interpretability.

Calibrated on n=22 diverse prompts (Qwen 2.5 0.5B).
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureContribution:
    """One feature's contribution to the classification decision."""
    feature: str
    value: float
    scores: dict  # {class_name: log-likelihood score}
    favors: str   # which class this feature most favors
    strength: str # 'strong', 'moderate', 'weak'
    explanation: str  # human-readable explanation


@dataclass
class ClassificationResult:
    """Complete classification with explanation."""
    predicted: str          # 'benign', 'mild', 'harmful', 'jailbreak'
    confidence: float       # 0-1, based on score margin
    scores: dict            # {class_name: total_score}
    probabilities: dict     # {class_name: probability} (softmax of scores)
    contributions: list     # list of FeatureContribution
    summary: str            # one-line human-readable summary
    caveats: list           # known limitations relevant to this classification


# Empirical class distributions (mean, std) calibrated on n=100 (25 per class)
# Qwen 2.5 0.5B, late layers, SVD=0, TL=off
CLASS_PARAMS = {
    'benign': {
        'net_correction': (0.07143, 0.00315),
        'middle_share':   (0.40417, 0.07035),
        'stress_score':   (3.19678, 0.11754),
        'entropy':        (0.79168, 0.03887),
        'interior_cv':    (0.64226, 0.17614),
    },
    'mild': {
        'net_correction': (0.07077, 0.00226),
        'middle_share':   (0.40128, 0.08137),
        'stress_score':   (3.23068, 0.09752),
        'entropy':        (0.76768, 0.04575),
        'interior_cv':    (0.56812, 0.16832),
    },
    'harmful': {
        'net_correction': (0.07674, 0.00388),
        'middle_share':   (0.48606, 0.08752),
        'stress_score':   (3.26334, 0.08571),
        'entropy':        (0.81512, 0.03732),
        'interior_cv':    (0.63225, 0.15201),
    },
    'jailbreak': {
        'net_correction': (0.08102, 0.00590),
        'middle_share':   (0.56594, 0.07811),
        'stress_score':   (3.43222, 0.11762),
        'entropy':        (0.79790, 0.05785),
        'interior_cv':    (1.02096, 0.40240),
    },
}

# Prior probabilities (uniform — no assumption about base rates)
CLASS_PRIORS = {
    'benign': 0.25,
    'mild': 0.25,
    'harmful': 0.25,
    'jailbreak': 0.25,
}

# Feature metadata for explanations
FEATURE_META = {
    'net_correction': {
        'name': 'Net Correction',
        'desc': 'total signed alignment correction magnitude',
        'direction': 'higher = more adversarial framing detected',
        'tier': 1,
    },
    'middle_share': {
        'name': 'Interior Share',
        'desc': 'fraction of correction on interior (non-boundary) tokens',
        'direction': 'higher = correction distributed across framing tokens',
        'tier': 1,
    },
    'stress_score': {
        'name': 'Stress Score',
        'desc': 'overall correction pressure at discriminative layers',
        'direction': 'higher = stronger alignment field activation',
        'tier': 1,
    },
    'entropy': {
        'name': 'Entropy',
        'desc': 'uniformity of correction distribution across tokens',
        'direction': 'higher = more uniform (less concentrated at boundaries)',
        'tier': 2,
    },
    'interior_cv': {
        'name': 'Interior CV',
        'desc': 'coefficient of variation of interior token corrections',
        'direction': 'higher = more concentrated on specific interior tokens',
        'tier': 2,
    },
}

# Features to use, in order of importance
FEATURES = ['net_correction', 'middle_share', 'stress_score', 'entropy', 'interior_cv']

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']


def _log_gaussian(x: float, mu: float, sigma: float) -> float:
    """Log probability density of x under N(mu, sigma)."""
    if sigma <= 0:
        sigma = 1e-6
    return -0.5 * math.log(2 * math.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2


def _softmax(scores: dict) -> dict:
    """Convert log-scores to probabilities."""
    vals = list(scores.values())
    max_v = max(vals)
    exp_vals = {k: math.exp(v - max_v) for k, v in scores.items()}
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


def classify(metrics: dict, params: dict = None, priors: dict = None) -> ClassificationResult:
    """
    Classify a prompt from its ASM metrics.
    
    Args:
        metrics: dict with keys from FEATURES (at minimum net_correction, middle_share)
        params: optional override for class parameters (for updated calibration)
        priors: optional override for class prior probabilities
    
    Returns:
        ClassificationResult with prediction, confidence, and per-feature explanations
    """
    if params is None:
        params = CLASS_PARAMS
    if priors is None:
        priors = CLASS_PRIORS

    # Compute per-class scores
    class_scores = {}
    for cls in CLASSES:
        class_scores[cls] = math.log(priors.get(cls, 0.25))

    contributions = []

    for feat in FEATURES:
        val = metrics.get(feat)
        if val is None:
            continue

        meta = FEATURE_META.get(feat, {})
        feat_scores = {}

        for cls in CLASSES:
            mu, sigma = params[cls][feat]
            ll = _log_gaussian(val, mu, sigma)
            feat_scores[cls] = ll
            class_scores[cls] += ll

        # Which class does this feature most favor?
        best_cls = max(feat_scores, key=feat_scores.get)
        worst_cls = min(feat_scores, key=feat_scores.get)
        margin = feat_scores[best_cls] - feat_scores[worst_cls]

        if margin > 3.0:
            strength = 'strong'
        elif margin > 1.0:
            strength = 'moderate'
        else:
            strength = 'weak'

        # Human-readable explanation
        direction = meta.get('direction', '')
        name = meta.get('name', feat)

        # Z-scores relative to each class
        z_scores = {}
        for cls in CLASSES:
            mu, sigma = params[cls][feat]
            z = (val - mu) / sigma if sigma > 0 else 0
            z_scores[cls] = z

        closest_cls = min(z_scores, key=lambda c: abs(z_scores[c]))
        z_closest = z_scores[closest_cls]

        explanation = f"{name} = {val:.5f}"
        if strength == 'strong':
            explanation += f" → strongly favors {best_cls} ({direction})"
        elif strength == 'moderate':
            explanation += f" → leans {best_cls} ({abs(z_closest):.1f}σ from {closest_cls} mean)"
        else:
            explanation += f" → ambiguous (within normal range for multiple classes)"

        contributions.append(FeatureContribution(
            feature=feat,
            value=val,
            scores=feat_scores,
            favors=best_cls,
            strength=strength,
            explanation=explanation,
        ))

    # Final prediction
    probabilities = _softmax(class_scores)
    predicted = max(probabilities, key=probabilities.get)
    
    # Confidence: margin between top and second prediction
    sorted_probs = sorted(probabilities.values(), reverse=True)
    confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    # Normalize to 0-1 (max margin is ~1.0)
    confidence = min(1.0, confidence)

    # Caveats based on n=100 calibration findings
    caveats = []
    if probabilities[predicted] < 0.4:
        caveats.append("Low confidence — this prompt doesn't clearly match any single category")
    if predicted in ('benign', 'mild') and probabilities['benign'] > 0.2 and probabilities['mild'] > 0.2:
        caveats.append("Benign and mild are nearly indistinguishable at 0.5B — distributions overlap almost completely")
    if predicted == 'harmful':
        jb_p = probabilities['jailbreak']
        ben_p = probabilities['benign'] + probabilities['mild']
        if jb_p > 0.2:
            caveats.append("Borderline harmful/jailbreak — prompt uses instruction-based framing similar to jailbreak patterns")
        if ben_p > 0.2:
            caveats.append("Borderline harmful/safe — blunt harmful request without adversarial framing looks geometrically benign")
    if predicted in ('benign', 'mild') and (probabilities['harmful'] > 0.15 or probabilities['jailbreak'] > 0.15):
        caveats.append("Some adversarial features detected despite safe classification — prompt may contain subtle framing")
    if predicted == 'jailbreak' and probabilities['harmful'] > 0.2:
        caveats.append("Strong framing-based instruction pattern — harmful prompts using 'write me...' / 'create...' patterns often score here")

    # Summary
    strong_feats = [c for c in contributions if c.strength == 'strong']
    if strong_feats:
        drivers = ', '.join(FEATURE_META.get(c.feature, {}).get('name', c.feature) for c in strong_feats)
        summary = f"Classified as {predicted} (p={probabilities[predicted]:.0%}) — driven by {drivers}"
    else:
        summary = f"Classified as {predicted} (p={probabilities[predicted]:.0%}) — no single dominant signal"

    return ClassificationResult(
        predicted=predicted,
        confidence=confidence,
        scores=class_scores,
        probabilities=probabilities,
        contributions=contributions,
        summary=summary,
        caveats=caveats,
    )


def classify_to_dict(metrics: dict) -> dict:
    """Classify and return a JSON-serializable dict."""
    result = classify(metrics)
    return {
        'predicted': result.predicted,
        'confidence': round(result.confidence, 4),
        'probabilities': {k: round(v, 4) for k, v in result.probabilities.items()},
        'summary': result.summary,
        'caveats': result.caveats,
        'contributions': [
            {
                'feature': c.feature,
                'name': FEATURE_META.get(c.feature, {}).get('name', c.feature),
                'value': round(c.value, 6),
                'favors': c.favors,
                'strength': c.strength,
                'explanation': c.explanation,
                'scores': {k: round(v, 3) for k, v in c.scores.items()},
            }
            for c in result.contributions
        ],
    }


def update_params(results: list) -> dict:
    """
    Recalibrate class parameters from a batch of labeled results.
    Returns updated params dict (does not modify CLASS_PARAMS).
    """
    import statistics
    cats = {}
    for r in results:
        cat = r.get('category', '')
        if cat in CLASSES:
            cats.setdefault(cat, []).append(r)
    
    new_params = {}
    for cls in CLASSES:
        new_params[cls] = {}
        items = cats.get(cls, [])
        for feat in FEATURES:
            vals = [r[feat] for r in items if r.get(feat) is not None]
            if len(vals) >= 2:
                mu = sum(vals) / len(vals)
                std = statistics.stdev(vals)  # ddof=1 by default
                new_params[cls][feat] = (mu, max(std, 1e-6))
            else:
                # Fall back to default
                new_params[cls][feat] = CLASS_PARAMS[cls][feat]
    
    return new_params
