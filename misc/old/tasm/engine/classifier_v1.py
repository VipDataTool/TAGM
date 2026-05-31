"""
ASM Classifier v1: Gaussian Naive Bayes.

Architecture: Per-feature Gaussian log-likelihood scorecard.
Each feature contributes independently. Class with highest total
log-likelihood wins. Equivalent to Naive Bayes with Gaussian priors.

Features: 5 ASM core metrics
Calibrated on n=100 balanced prompts (Qwen 2.5 0.5B).
Performance: 61% 4-class, 83% binary.

Strengths: Simple, interpretable, no thresholds to tune.
Weaknesses: Benign/mild overlap. Assumes feature independence.
"""

import math

CLASSIFIER_ID = "v1"
CLASSIFIER_NAME = "Gaussian Naive Bayes"
CLASSIFIER_DESC = "Independent Gaussian likelihood per feature. n=100 calibration."

FEATURES = ['net_correction', 'middle_share', 'stress_score', 'entropy', 'interior_cv']
CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

CLASS_PARAMS = {
    'benign': {
        'net_correction': (0.07143, 0.00315), 'middle_share': (0.40417, 0.07035),
        'stress_score': (3.19678, 0.11754), 'entropy': (0.79168, 0.03887),
        'interior_cv': (0.64226, 0.17614),
    },
    'mild': {
        'net_correction': (0.07077, 0.00226), 'middle_share': (0.40128, 0.08137),
        'stress_score': (3.23068, 0.09752), 'entropy': (0.76768, 0.04575),
        'interior_cv': (0.56812, 0.16832),
    },
    'harmful': {
        'net_correction': (0.07674, 0.00388), 'middle_share': (0.48606, 0.08752),
        'stress_score': (3.26334, 0.08571), 'entropy': (0.81512, 0.03732),
        'interior_cv': (0.63225, 0.15201),
    },
    'jailbreak': {
        'net_correction': (0.08102, 0.00590), 'middle_share': (0.56594, 0.07811),
        'stress_score': (3.43222, 0.11762), 'entropy': (0.79790, 0.05785),
        'interior_cv': (1.02096, 0.40240),
    },
}

FEATURE_META = {
    'net_correction': ('Net Correction', 'alignment correction magnitude'),
    'middle_share':   ('Interior Share', 'fraction on interior tokens'),
    'stress_score':   ('Stress Score', 'correction pressure at mid layers'),
    'entropy':        ('Entropy', 'correction distribution uniformity'),
    'interior_cv':    ('Interior CV', 'interior token concentration'),
}


def _log_gaussian(x, mu, sigma):
    if sigma <= 0: sigma = 1e-6
    return -0.5 * math.log(2 * math.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2


def classify(metrics):
    class_scores = {cls: math.log(0.25) for cls in CLASSES}
    contributions = []
    for feat in FEATURES:
        val = metrics.get(feat)
        if val is None: continue
        feat_scores = {}
        for cls in CLASSES:
            mu, sigma = CLASS_PARAMS[cls][feat]
            ll = _log_gaussian(val, mu, sigma)
            feat_scores[cls] = ll
            class_scores[cls] += ll
        best_cls = max(feat_scores, key=feat_scores.get)
        margin = feat_scores[best_cls] - min(feat_scores.values())
        strength = 'strong' if margin > 3.0 else ('moderate' if margin > 1.0 else 'weak')
        name, desc = FEATURE_META.get(feat, (feat, ''))
        contributions.append({
            'feature': feat, 'name': name, 'value': round(val, 6),
            'favors': best_cls, 'strength': strength,
            'scores': {k: round(v, 3) for k, v in feat_scores.items()},
        })

    max_ll = max(class_scores.values())
    exp_s = {k: math.exp(v - max_ll) for k, v in class_scores.items()}
    total = sum(exp_s.values())
    probs = {k: v / total for k, v in exp_s.items()}
    predicted = max(probs, key=probs.get)
    sorted_p = sorted(probs.values(), reverse=True)
    confidence = min(1.0, sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

    caveats = []
    if probs[predicted] < 0.4:
        caveats.append("Low confidence across all classes")
    if predicted in ('benign', 'mild') and abs(probs.get('benign',0) - probs.get('mild',0)) < 0.15:
        caveats.append("Benign/mild indistinguishable at 0.5B")

    strong = [c for c in contributions if c['strength'] == 'strong']
    drivers = ', '.join(c['name'] for c in strong) if strong else 'no dominant signal'
    summary = f"{predicted} (p={probs[predicted]:.0%}) -- {drivers}"

    return {
        'classifier': CLASSIFIER_ID, 'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted, 'confidence': round(confidence, 4),
        'probabilities': {k: round(v, 4) for k, v in probs.items()},
        'summary': summary, 'caveats': caveats, 'contributions': contributions,
    }
