"""
ASM Classifier v4: Topographic Model.

Architecture: 3-stage hierarchical decision tree.
REVISED: Uses shared binary stage (5.1), removed fragile LTP entropy
variance axis (5.2), replaced coupling axis with kl_min (5.2),
harmonized confidence (5.3), honest benign/mild handling (5.4).

Original z-normalized population statistics retained for the adversarial
sub-stage only, where they provide meaningful feature scaling.

  Stage 1 (shared): middle_share + weighted net_correction
  Stage 2a (shared): entropy + 0.3*CV (capped confidence)
  Stage 2b: z-normalized stress_max + z-normalized kl_min
    (replaced the original coupling axis which was empirically fragile)
"""

import math
import numpy as np
from engine.classifier_common import (
    binary_classify, binary_stage_dict, binary_contributions,
    safe_sub_classify, safe_sub_stage_dict, safe_sub_contributions,
    sigmoid_confidence, BINARY_CONF_SCALE, SUB_CONF_SCALE,
)

CLASSIFIER_ID = "v4"
CLASSIFIER_NAME = "Topographic Model"
CLASSIFIER_DESC = "Z-normalized fusion with shared binary stage. Revised calibration."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# Population statistics for z-normalization (adversarial sub-stage)
# Retained from n=52 calibration for stress_max and kl_min
POP_STATS = {
    'stress_max':  (3.72, 0.18),   # mean, std across adversarial prompts
    'kl_min':      (0.015, 0.030),  # mean, std — jailbreak has much higher floor
}

# Stage 2b: z(stress_max) + z(kl_min) > threshold => jailbreak
STAGE2B_THRESHOLD = 0.80


def _z(val, feat):
    mu, sigma = POP_STATS.get(feat, (0, 1))
    return (val - mu) / sigma if sigma > 0 else 0.0


def classify(metrics):
    stages = []
    caveats = []
    contributions = []

    nc = metrics.get('net_correction', 0)
    ms = metrics.get('middle_share', 0)
    seq_len = metrics.get('seq_len')

    # ═══ Stage 1: Shared binary ═══
    is_adversarial, bin_score, bin_conf, bin_caveats = binary_classify(ms, nc, seq_len)
    stages.append(binary_stage_dict(bin_score, is_adversarial, bin_conf, ms, nc))
    contributions.extend(binary_contributions(ms, nc))
    caveats.extend(bin_caveats)

    if is_adversarial:
        # ═══ Stage 2b: Adversarial sub — stress ceiling + KL floor ═══
        pts = metrics.get('per_token_stress', [])
        stress_max = max(pts) if pts else metrics.get('stress_score', 0)
        ptk = metrics.get('per_token_kl', [])
        kl_min = min(ptk) if ptk and len(ptk) > 0 else 0

        z_sm = _z(stress_max, 'stress_max')
        z_km = _z(kl_min, 'kl_min')
        s2b_score = z_sm + z_km

        predicted = 'jailbreak' if s2b_score > STAGE2B_THRESHOLD else 'harmful'
        s2b_conf = sigmoid_confidence(abs(s2b_score - STAGE2B_THRESHOLD))

        stages.append({
            'stage': 'adversarial_sub', 'name': 'Z-normalized Stress + KL Floor',
            'score': round(s2b_score, 4), 'threshold': STAGE2B_THRESHOLD,
            'margin': round(s2b_score - STAGE2B_THRESHOLD, 4),
            'confidence': round(s2b_conf, 4),
            'left_class': 'harmful', 'right_class': 'jailbreak',
            'chosen': predicted,
            'explanation': (
                f"z(stress_max={stress_max:.3f})={z_sm:.2f} + "
                f"z(kl_min={kl_min:.5f})={z_km:.2f} = {s2b_score:.3f} "
                f"{'>' if predicted == 'jailbreak' else '≤'} {STAGE2B_THRESHOLD} → {predicted}"
            ),
        })
        contributions.append({
            'feature': 'stress_max', 'name': 'Stress Ceiling',
            'value': round(stress_max, 4), 'z': round(z_sm, 3),
            'favors': 'jailbreak' if z_sm > 0 else 'harmful',
            'strength': 'strong' if abs(z_sm) > 1.5 else ('moderate' if abs(z_sm) > 0.5 else 'weak'),
        })
        contributions.append({
            'feature': 'kl_min', 'name': 'KL Floor',
            'value': round(kl_min, 6), 'z': round(z_km, 3),
            'favors': 'jailbreak' if z_km > 0 else 'harmful',
            'strength': 'strong' if abs(z_km) > 1.5 else ('moderate' if abs(z_km) > 0.5 else 'weak'),
        })

        if abs(s2b_score - STAGE2B_THRESHOLD) < 0.3:
            caveats.append("Borderline harmful/jailbreak")
        if not ptk:
            caveats.append("No KL data — kl_min defaults to 0, biasing toward harmful")
    else:
        # ═══ Stage 2a: Shared safe sub ═══
        ent = metrics.get('entropy', 0)
        icv = metrics.get('interior_cv', 0)
        predicted, safe_score, safe_conf, safe_caveat = safe_sub_classify(ent, icv)
        stages.append(safe_sub_stage_dict(predicted, safe_score, safe_conf, ent, icv))
        contributions.extend(safe_sub_contributions(ent, icv))
        caveats.append(safe_caveat)

    overall_conf = bin_conf * stages[-1]['confidence']
    stage_path = ' → '.join(s['chosen'] for s in stages)
    summary = f"{predicted} via {stage_path} ({overall_conf:.0%})"

    return {
        'classifier': CLASSIFIER_ID, 'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted, 'confidence': round(overall_conf, 4),
        'probabilities': {},
        'summary': summary, 'caveats': caveats,
        'stages': stages, 'contributions': contributions,
    }
