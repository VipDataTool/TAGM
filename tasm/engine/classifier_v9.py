"""
ASM Classifier v9: Three Instrument.

Architecture: centroid distance model using the three measurement axes:
  ASM (net disruption) — attribution distribution shape
  LTP (selectivity) — rank displacement between base and instruct
  SFD (subspace) — QK routing density

The classifier measures distance from the benign centroid (the region
of minimal base-instruct divergence) in a normalized feature space.
Each instrument contributes bounded [0,1] features that are directly
comparable without population normalization.

Falls back gracefully when SFD or LTP data is unavailable, using
whatever instruments are present.

  Stage 1: shared binary gate (from classifier_common)
  Stage 2a: safe sub with honest benign/mild capping
  Stage 2b: three-instrument centroid distance for harmful vs jailbreak
"""

import math
import numpy as np
from engine.classifier_common import (
    binary_classify, binary_stage_dict, binary_contributions,
    safe_sub_classify, safe_sub_stage_dict, safe_sub_contributions,
    sigmoid_confidence,
)

CLASSIFIER_ID = "v9"
CLASSIFIER_NAME = "Three Instrument"
CLASSIFIER_DESC = "Centroid distance: ASM (disruption) + LTP (selectivity) + SFD (subspace)."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# ═══ Benign centroids (resting state values from n=56 calibration) ═══
# These define the "origin" — where base and instruct converge.
# Each feature should be near these values for benign prompts.

BENIGN_CENTROID = {
    # ASM axis (attribution distribution shape)
    'middle_share': 0.4245,
    'interior_cv': 0.6355,
    'net_correction': 0.0715,
    # LTP axis (rank displacement)
    'ltp_tau': 0.7014,
    'ltp_overlap': 0.7897,
    # SFD axis (routing density)
    'sfd_density': 0.3722,
    'sfd_entropy': 1.7071,
}

# Jailbreak centroid (maximal deviation)
JAILBREAK_CENTROID = {
    'middle_share': 0.5845,
    'interior_cv': 0.9005,
    'net_correction': 0.0820,
    'ltp_tau': 0.5851,
    'ltp_overlap': 0.6560,
    'sfd_density': 0.3852,
    'sfd_entropy': 1.7448,
}

# Feature weights for centroid distance (Cohen's d from n=53 calibration)
FEATURE_WEIGHTS = {
    'middle_share': 2.39,
    'interior_cv': 1.22,
    'net_correction': 2.22,
    'ltp_tau': 0.94,
    'ltp_overlap': 1.80,
    'sfd_density': 0.95,
    'sfd_entropy': 0.98,
}

# Threshold for harmful vs jailbreak: distance ratio from benign centroid
JAILBREAK_DISTANCE_THRESHOLD = 0.41


def _extract_features(metrics):
    """Extract the three-instrument feature vector from a metrics dict."""
    features = {}
    available_axes = []

    # ─── ASM axis (always available) ───
    features['middle_share'] = metrics.get('middle_share', 0)
    features['interior_cv'] = metrics.get('interior_cv', 0)
    features['net_correction'] = metrics.get('net_correction', 0)
    available_axes.append('ASM')

    # ─── LTP axis (available when rank_displacement present) ───
    rd = metrics.get('rank_displacement')
    if rd and rd.get('mean_tau') is not None:
        features['ltp_tau'] = rd['mean_tau']
        features['ltp_overlap'] = rd.get('mean_overlap', 0.5)
        available_axes.append('LTP')
    else:
        # Fall back to LTP summary stats if available
        ltp = metrics.get('ltp')
        if isinstance(ltp, dict) and ltp.get('mean_C') is not None:
            # Use offset consistency as a proxy (weaker signal)
            features['ltp_tau'] = 1.0 - ltp.get('mean_C', 0)  # invert: low C = high agreement
            features['ltp_overlap'] = ltp.get('mean_L', 1.0)
            available_axes.append('LTP*')  # partial
        else:
            features['ltp_tau'] = None
            features['ltp_overlap'] = None

    # ─── SFD axis (available when sfd result present) ───
    sfd = metrics.get('sfd')
    if isinstance(sfd, dict) and sfd.get('density_mean') is not None:
        features['sfd_density'] = sfd['density_mean']
        features['sfd_entropy'] = sfd.get('entropy_mean', 0.5)
        available_axes.append('SFD')
    else:
        features['sfd_density'] = None
        features['sfd_entropy'] = None

    return features, available_axes


def _weighted_centroid_distance(features, centroid, weights):
    """Compute weighted Euclidean distance from a centroid,
    using only features that are available (not None)."""
    dist_sq = 0.0
    total_weight = 0.0

    for feat, val in features.items():
        if val is None:
            continue
        w = weights.get(feat, 1.0)
        c = centroid.get(feat, 0.5)
        dist_sq += w * (val - c) ** 2
        total_weight += w

    if total_weight == 0:
        return 0.0

    return math.sqrt(dist_sq / total_weight)


def classify(metrics):
    stages = []
    caveats = []

    nc = metrics.get('net_correction', 0)
    ms = metrics.get('middle_share', 0)
    seq_len = metrics.get('seq_len')

    # ═══ Stage 1: Shared binary gate ═══
    is_adversarial, bin_score, bin_conf, bin_caveats = binary_classify(ms, nc, seq_len)
    stages.append(binary_stage_dict(bin_score, is_adversarial, bin_conf, ms, nc))
    contributions = list(binary_contributions(ms, nc))
    caveats.extend(bin_caveats)

    # Extract three-instrument features
    features, available_axes = _extract_features(metrics)
    n_axes = len([a for a in available_axes if '*' not in a])

    if is_adversarial:
        # ═══ Stage 2b: Three-instrument centroid distance ═══
        dist_benign = _weighted_centroid_distance(
            features, BENIGN_CENTROID, FEATURE_WEIGHTS)
        dist_jailbreak = _weighted_centroid_distance(
            features, JAILBREAK_CENTROID, FEATURE_WEIGHTS)

        # Distance ratio: how far toward the jailbreak centroid
        total_dist = dist_benign + dist_jailbreak
        if total_dist > 0:
            ratio = dist_benign / total_dist
        else:
            ratio = 0.5

        predicted = 'jailbreak' if ratio > JAILBREAK_DISTANCE_THRESHOLD else 'harmful'
        margin = abs(ratio - JAILBREAK_DISTANCE_THRESHOLD)
        sub_conf = sigmoid_confidence(margin, scale=6.0)

        leq = '\u2264'
        arrow = '\u2192'
        cmp = '>' if predicted == 'jailbreak' else leq

        stages.append({
            'stage': 'adversarial_sub',
            'name': f'Centroid Distance ({n_axes}-axis)',
            'score': round(ratio, 5),
            'threshold': JAILBREAK_DISTANCE_THRESHOLD,
            'margin': round(ratio - JAILBREAK_DISTANCE_THRESHOLD, 5),
            'confidence': round(sub_conf, 4),
            'left_class': 'harmful',
            'right_class': 'jailbreak',
            'chosen': predicted,
            'explanation': (
                f"d(benign)={dist_benign:.3f}, d(jailbreak)={dist_jailbreak:.3f}, "
                f"ratio={ratio:.3f} "
                f"{cmp} "
                f"{JAILBREAK_DISTANCE_THRESHOLD} {arrow} {predicted} "
                f"[axes: {'+'.join(available_axes)}]"
            ),
        })

        # Contributions from each axis
        for feat, val in features.items():
            if val is None:
                continue
            ben_c = BENIGN_CENTROID.get(feat, 0.5)
            dev = val - ben_c
            axis = ('ASM' if feat in ('middle_share', 'interior_cv', 'net_correction')
                    else 'LTP' if feat.startswith('ltp_')
                    else 'SFD')
            contributions.append({
                'feature': feat, 'name': feat,
                'value': round(val, 4),
                'deviation': round(dev, 4),
                'axis': axis,
                'favors': 'jailbreak' if abs(val - JAILBREAK_CENTROID.get(feat, 0.5)) < abs(dev) else 'harmful',
                'strength': ('strong' if abs(dev) > 0.1
                             else ('moderate' if abs(dev) > 0.03 else 'weak')),
            })

        if margin < 0.05:
            caveats.append("Close to harmful/jailbreak boundary")

    else:
        # ═══ Stage 2a: Shared safe sub ═══
        ent = metrics.get('entropy', 0)
        icv = metrics.get('interior_cv', 0)
        predicted, safe_score, safe_conf, safe_caveat = safe_sub_classify(ent, icv)
        stages.append(safe_sub_stage_dict(predicted, safe_score, safe_conf, ent, icv))
        contributions.extend(safe_sub_contributions(ent, icv))
        caveats.append(safe_caveat)

    overall_conf = bin_conf * stages[-1]['confidence']
    axis_label = f"{n_axes}-instrument" if n_axes > 1 else "ASM-only"

    # Note missing instruments
    if 'LTP' not in available_axes and 'LTP*' not in available_axes:
        caveats.append("LTP data unavailable (no rank displacement)")
    if 'SFD' not in [a for a in available_axes if not a.endswith('*')]:
        caveats.append("SFD data unavailable (no QK density)")
    if n_axes < 3:
        caveats.append(f"Operating on {n_axes}/3 instruments. Full triangulation requires ASM+LTP+SFD.")

    summary = (f"{predicted} ({overall_conf:.0%}) via {axis_label}. "
               f"Axes: {', '.join(available_axes)}")

    return {
        'classifier': CLASSIFIER_ID,
        'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted,
        'confidence': round(overall_conf, 4),
        'probabilities': {},
        'summary': summary,
        'caveats': caveats,
        'stages': stages,
        'contributions': contributions,
        'available_axes': available_axes,
        'n_axes': n_axes,
    }
