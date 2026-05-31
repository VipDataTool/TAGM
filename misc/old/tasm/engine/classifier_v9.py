"""
ASM Classifier v9: Three Instrument.

Nearest centroid in a three-axis measurement space:
  ASM — attribution distribution shape (middle_share, interior_cv, net_correction)
  LTP — rank displacement between base and instruct (tau, overlap)
  SFD — QK routing subspace density (density_mean, entropy_mean)

Classifies by weighted Euclidean distance to four category centroids
calibrated from n=53 prompts. Feature weights are Cohen's d values.

Degrades gracefully when SFD or LTP data is unavailable —
uses whatever instruments are present, reports which axes are active.

One path: extract features, compute distances, classify, report everything.
"""

import math


CLASSIFIER_ID = "v9"
CLASSIFIER_NAME = "Three Instrument"

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# Category centroids (mean feature values from n=53 calibration)
CENTROIDS = {
    'benign': {
        'middle_share': 0.4245, 'interior_cv': 0.6355, 'net_correction': 0.0715,
        'ltp_tau': 0.7014, 'ltp_overlap': 0.7897,
        'sfd_density': 0.3722, 'sfd_entropy': 1.7071,
    },
    'mild': {
        'middle_share': 0.4013, 'interior_cv': 0.6053, 'net_correction': 0.0707,
        'ltp_tau': 0.6552, 'ltp_overlap': 0.7612,
        'sfd_density': 0.3804, 'sfd_entropy': 1.7297,
    },
    'harmful': {
        'middle_share': 0.4887, 'interior_cv': 0.6362, 'net_correction': 0.0772,
        'ltp_tau': 0.6260, 'ltp_overlap': 0.7174,
        'sfd_density': 0.3902, 'sfd_entropy': 1.7580,
    },
    'jailbreak': {
        'middle_share': 0.5845, 'interior_cv': 0.9005, 'net_correction': 0.0820,
        'ltp_tau': 0.5851, 'ltp_overlap': 0.6560,
        'sfd_density': 0.3852, 'sfd_entropy': 1.7448,
    },
}

# Feature weights (Cohen's d, benign vs jailbreak, from n=53 calibration)
FEATURE_WEIGHTS = {
    'middle_share': 2.39, 'interior_cv': 1.22, 'net_correction': 2.22,
    'ltp_tau': 0.94, 'ltp_overlap': 1.80,
    'sfd_density': 0.95, 'sfd_entropy': 0.98,
}

FEATURE_AXIS = {
    'middle_share': 'ASM', 'interior_cv': 'ASM', 'net_correction': 'ASM',
    'ltp_tau': 'LTP', 'ltp_overlap': 'LTP',
    'sfd_density': 'SFD', 'sfd_entropy': 'SFD',
}


def _extract_features(metrics):
    """Extract the three-instrument feature vector."""
    features = {}
    available_axes = []

    features['middle_share'] = metrics.get('middle_share', 0)
    features['interior_cv'] = metrics.get('interior_cv', 0)
    features['net_correction'] = metrics.get('net_correction', 0)
    available_axes.append('ASM')

    rd = metrics.get('rank_displacement')
    if rd and rd.get('mean_tau') is not None:
        features['ltp_tau'] = rd['mean_tau']
        features['ltp_overlap'] = rd.get('mean_overlap', 0.5)
        available_axes.append('LTP')
    else:
        ltp = metrics.get('ltp')
        if isinstance(ltp, dict) and ltp.get('mean_C') is not None:
            features['ltp_tau'] = 1.0 - ltp.get('mean_C', 0)
            features['ltp_overlap'] = ltp.get('mean_L', 1.0)
            available_axes.append('LTP*')
        else:
            features['ltp_tau'] = None
            features['ltp_overlap'] = None

    sfd = metrics.get('sfd')
    if isinstance(sfd, dict) and sfd.get('density_mean') is not None:
        features['sfd_density'] = sfd['density_mean']
        features['sfd_entropy'] = sfd.get('entropy_mean', 0.5)
        available_axes.append('SFD')
    else:
        features['sfd_density'] = None
        features['sfd_entropy'] = None

    return features, available_axes


def _weighted_distance(features, centroid, weights):
    """Weighted Euclidean distance, skipping None features."""
    dist_sq = 0.0
    total_w = 0.0
    for feat, val in features.items():
        if val is None:
            continue
        w = weights.get(feat, 1.0)
        c = centroid.get(feat, 0.5)
        dist_sq += w * (val - c) ** 2
        total_w += w
    if total_w == 0:
        return 0.0
    return math.sqrt(dist_sq / total_w)


def _softmax(distances, scale=200.0):
    """Convert distances to probabilities (closer = higher probability).
    Scale controls temperature — higher = more peaked around nearest."""
    neg_sq = {k: -scale * (v ** 2) for k, v in distances.items()}
    max_v = max(neg_sq.values())
    exps = {k: math.exp(v - max_v) for k, v in neg_sq.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def classify(metrics):
    features, available_axes = _extract_features(metrics)
    n_axes = len([a for a in available_axes if '*' not in a])

    # Distance to each category centroid
    distances = {}
    for cat in CLASSES:
        distances[cat] = _weighted_distance(features, CENTROIDS[cat], FEATURE_WEIGHTS)

    # Nearest centroid wins
    predicted = min(distances, key=distances.get)
    probabilities = _softmax(distances)
    confidence = probabilities[predicted]

    # Runner-up for margin
    sorted_cats = sorted(distances, key=distances.get)
    runner = sorted_cats[1] if len(sorted_cats) > 1 else predicted
    margin = distances[runner] - distances[predicted]

    # Stage: one stage, the centroid distance computation
    stages = [{
        'stage': 'centroid',
        'name': f'Nearest Centroid ({n_axes}-axis)',
        'score': round(distances[predicted], 5),
        'threshold': 0,
        'margin': round(margin, 5),
        'confidence': round(confidence, 4),
        'left_class': predicted,
        'right_class': runner,
        'chosen': predicted,
        'explanation': (
            f"d({predicted})={distances[predicted]:.3f}, "
            f"d({runner})={distances[runner]:.3f}, "
            f"margin={margin:.3f} "
            f"[{'+'.join(available_axes)}]"
        ),
    }]

    # Contributions: every feature, its value, deviation from predicted centroid,
    # which centroid it's closest to
    contributions = []
    for feat, val in features.items():
        if val is None:
            continue
        axis = FEATURE_AXIS.get(feat, '?')
        pred_c = CENTROIDS[predicted].get(feat, 0.5)
        dev = val - pred_c
        # Which centroid is this feature closest to?
        feat_dists = {cat: abs(val - CENTROIDS[cat].get(feat, 0.5)) for cat in CLASSES}
        closest = min(feat_dists, key=feat_dists.get)
        feat_margin = sorted(feat_dists.values())
        feat_sep = feat_margin[1] - feat_margin[0] if len(feat_margin) > 1 else 0
        contributions.append({
            'feature': feat, 'name': feat,
            'value': round(val, 4),
            'deviation': round(dev, 4),
            'axis': axis,
            'favors': closest,
            'strength': ('strong' if feat_sep > 0.05
                         else ('moderate' if feat_sep > 0.01 else 'weak')),
        })

    # Caveats
    caveats = []
    if margin < 0.3:
        caveats.append(f"Close to {runner} centroid (margin={margin:.3f})")
    if predicted in ('benign', 'mild') and abs(distances['benign'] - distances['mild']) < 0.5:
        caveats.append("Benign/mild boundary is weak at 0.5B")
    if n_axes < 3:
        missing = []
        if 'LTP' not in available_axes and 'LTP*' not in available_axes:
            missing.append('LTP')
        if 'SFD' not in [a for a in available_axes if not a.endswith('*')]:
            missing.append('SFD')
        if missing:
            caveats.append(f"Missing: {', '.join(missing)}. Operating on {n_axes}/3 instruments.")

    summary = (
        f"{predicted} ({confidence:.0%}) via {n_axes}-instrument centroid. "
        f"Distances: {', '.join(f'{c}={distances[c]:.3f}' for c in sorted_cats)}"
    )

    return {
        'classifier': CLASSIFIER_ID,
        'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted,
        'confidence': round(confidence, 4),
        'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
        'summary': summary,
        'caveats': caveats,
        'stages': stages,
        'contributions': contributions,
        'distances': {k: round(v, 4) for k, v in distances.items()},
        'available_axes': available_axes,
        'n_axes': n_axes,
    }
