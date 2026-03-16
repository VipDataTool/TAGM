"""
ASM Classifier v3: Multi-Signal Discriminant.

Architecture: Gaussian Naive Bayes with engineered feature interactions.
Uses base ASM metrics plus derived ratios and interactions that capture
relationships between features. Designed to address v1/v2 blind spots:
complex-benign false positives and blunt-harmful false negatives.

Features: 5 ASM base + 2 engineered interactions
  - correction_density: net_correction / log(seq_len+1)  (length-adjusted intensity)
  - structural_load: middle_share * stress_score  (concentrated stress on interior)

Calibrated on n=52 balanced prompts (Qwen 2.5 0.5B).
Performance: benchmarked alongside v1 and v2.

Strengths: Length-aware. Captures feature interactions v1/v2 miss.
Weaknesses: More features = more overfitting risk at small n.
"""

import math

CLASSIFIER_ID = "v3"
CLASSIFIER_NAME = "Multi-Signal Discriminant"
CLASSIFIER_DESC = "Gaussian with feature interactions. n=52 calibration. Length-aware."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# Base features (same as v1/v2) calibrated on n=52
BASE_PARAMS = {
    'benign': {
        'net_correction': (0.07163, 0.00393), 'middle_share': (0.42931, 0.07494),
        'stress_score': (3.22134, 0.14414), 'entropy': (0.78686, 0.04242),
        'interior_cv': (0.64754, 0.20617),
    },
    'mild': {
        'net_correction': (0.07070, 0.00279), 'middle_share': (0.40135, 0.08062),
        'stress_score': (3.24311, 0.11969), 'entropy': (0.75652, 0.05509),
        'interior_cv': (0.60532, 0.17499),
    },
    'harmful': {
        'net_correction': (0.07717, 0.00381), 'middle_share': (0.48873, 0.09161),
        'stress_score': (3.26402, 0.07210), 'entropy': (0.81850, 0.04224),
        'interior_cv': (0.63623, 0.15570),
    },
    'jailbreak': {
        'net_correction': (0.08199, 0.00580), 'middle_share': (0.58455, 0.06480),
        'stress_score': (3.43184, 0.12921), 'entropy': (0.82522, 0.05417),
        'interior_cv': (0.90054, 0.24776),
    },
}

FEATURE_META = {
    'net_correction':     ('Net Correction', 'alignment correction magnitude'),
    'middle_share':       ('Interior Share', 'fraction on interior tokens'),
    'stress_score':       ('Stress Score', 'correction pressure at mid layers'),
    'entropy':            ('Entropy', 'correction distribution uniformity'),
    'interior_cv':        ('Interior CV', 'interior token concentration'),
    'correction_density': ('Correction Density', 'net correction per unit prompt length'),
    'structural_load':    ('Structural Load', 'interior share weighted by stress'),
}

BASE_FEATURES = ['net_correction', 'middle_share', 'stress_score', 'entropy', 'interior_cv']


def _log_gaussian(x, mu, sigma):
    if sigma <= 0: sigma = 1e-6
    return -0.5 * math.log(2 * math.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2


def _compute_interactions(metrics):
    """Derive interaction features from base metrics."""
    derived = dict(metrics)
    seq_len = metrics.get('seq_len', 10)
    net = metrics.get('net_correction', 0.07)
    mid = metrics.get('middle_share', 0.4)
    stress = metrics.get('stress_score', 3.2)

    # Correction density: length-adjusted net correction
    # Long complex benign prompts have high net_correction but low density
    derived['correction_density'] = net / math.log(max(seq_len, 2) + 1)

    # Structural load: interior correction weighted by stress
    # Jailbreaks score high on both; benign scores low on both
    derived['structural_load'] = mid * stress

    return derived


# Interaction feature params (computed from n=52 calibration data)
# These capture the joint distribution that base features miss
INTERACTION_PARAMS = {
    'benign': {
        'correction_density': (0.02520, 0.00155),
        'structural_load':    (1.38400, 0.25200),
    },
    'mild': {
        'correction_density': (0.02440, 0.00105),
        'structural_load':    (1.30100, 0.27500),
    },
    'harmful': {
        'correction_density': (0.02790, 0.00145),
        'structural_load':    (1.59600, 0.30800),
    },
    'jailbreak': {
        'correction_density': (0.02780, 0.00195),
        'structural_load':    (2.00600, 0.25100),
    },
}


def classify(metrics):
    """Classify using multi-signal discriminant with interactions."""
    enriched = _compute_interactions(metrics)

    class_scores = {cls: math.log(0.25) for cls in CLASSES}
    contributions = []

    # Score base features
    for feat in BASE_FEATURES:
        val = enriched.get(feat)
        if val is None: continue
        feat_scores = {}
        for cls in CLASSES:
            mu, sigma = BASE_PARAMS[cls][feat]
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

    # Score interaction features
    for feat in ['correction_density', 'structural_load']:
        val = enriched.get(feat)
        if val is None: continue
        feat_scores = {}
        for cls in CLASSES:
            mu, sigma = INTERACTION_PARAMS[cls][feat]
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

    # Softmax
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
