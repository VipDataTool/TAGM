"""
ASM Classifier v6: Tetrahedral.

Architecture: Benign = origin. Three axes rise from it.
REVISED: Uses shared binary gate (5.1), shared safe sub (5.4),
harmonized confidence (5.3). Tetrahedral axis geometry retained
for the adversarial sub-classification.

  Stage 1 (shared binary): interior distribution test
  Stage 2a (shared): entropy + CV (capped confidence)
  Stage 2b (tetrahedral): harmful vs jailbreak via KL entropy axis
    Harmful: high elevation + selective KL (concentrated divergence)
    Jailbreak: high elevation + uniform KL (blanket divergence)

Requires base model for KL features. Falls back to shared safe sub
when classified as safe.
"""

import math
import numpy as np
from engine.classifier_common import (
    binary_classify, binary_stage_dict, binary_contributions,
    safe_sub_classify, safe_sub_stage_dict, safe_sub_contributions,
    sigmoid_confidence,
)

CLASSIFIER_ID = "v6"
CLASSIFIER_NAME = "Tetrahedral"
CLASSIFIER_DESC = "Tetrahedral axes with shared binary gate. Benign = origin."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# KL entropy thresholds for harmful vs jailbreak axis
# Harmful: low KL entropy (concentrated on framing tokens)
# Jailbreak: high KL entropy (uniform across all tokens)
KL_ENT_THRESHOLD = 0.48  # midpoint of the separation range


def _kl_entropy(kl_array):
    kl = np.clip(np.array(kl_array, dtype=float), 1e-10, None)
    s = kl.sum()
    if s <= 0 or len(kl) < 2:
        return None
    p = kl / s
    return float(-np.sum(p * np.log(p)) / np.log(len(kl)))


def classify(metrics):
    contributions = []
    caveats = []
    stages = []

    nc = metrics.get('net_correction', 0)
    ms = metrics.get('middle_share', 0)
    seq_len = metrics.get('seq_len')

    kl_data = metrics.get('per_token_kl')
    has_kl = kl_data is not None and len(kl_data) >= 2
    kl_ent = _kl_entropy(kl_data) if has_kl else None

    layers = ['core']
    if has_kl: layers.append('KL')

    # ═══ Stage 1: Shared binary ═══
    is_adversarial, bin_score, bin_conf, bin_caveats = binary_classify(ms, nc, seq_len)
    stages.append(binary_stage_dict(bin_score, is_adversarial, bin_conf, ms, nc))
    contributions.extend(binary_contributions(ms, nc))
    caveats.extend(bin_caveats)

    if is_adversarial:
        # ═══ Stage 2b: Tetrahedral harmful vs jailbreak ═══
        if kl_ent is not None:
            predicted = 'jailbreak' if kl_ent > KL_ENT_THRESHOLD else 'harmful'
            margin = abs(kl_ent - KL_ENT_THRESHOLD)
            sub_conf = sigmoid_confidence(margin)

            if predicted == 'jailbreak':
                reason = f"Uniform divergence (KL entropy {kl_ent:.3f} > {KL_ENT_THRESHOLD})"
            else:
                reason = f"Selective divergence (KL entropy {kl_ent:.3f} ≤ {KL_ENT_THRESHOLD})"

            contributions.append({
                'feature': 'kl_entropy', 'name': 'KL Entropy',
                'value': round(kl_ent, 5), 'layer': 'KL',
                'favors': 'jailbreak' if kl_ent > 0.50 else 'harmful',
                'strength': ('strong' if margin > 0.15
                             else ('moderate' if margin > 0.06 else 'weak')),
            })
        else:
            # No KL data — use stress as rough proxy
            stress = metrics.get('stress_score', 1.0)
            stress_thr = 3.35
            predicted = 'jailbreak' if stress > stress_thr else 'harmful'
            margin = abs(stress - stress_thr)
            sub_conf = sigmoid_confidence(margin)
            reason = f"Stress proxy = {stress:.3f} {'>' if predicted == 'jailbreak' else '≤'} {stress_thr}"
            caveats.append("No KL data — using stress as rough proxy for harmful/jailbreak")

        stages.append({
            'stage': 'adversarial_sub', 'name': 'Tetrahedral Axes',
            'score': round(kl_ent if kl_ent is not None else metrics.get('stress_score', 0), 5),
            'threshold': KL_ENT_THRESHOLD if kl_ent is not None else 3.35,
            'margin': round(margin, 5),
            'confidence': round(sub_conf, 4),
            'left_class': 'harmful', 'right_class': 'jailbreak',
            'chosen': predicted,
            'explanation': reason,
        })

        if margin < 0.05:
            caveats.append("Borderline harmful/jailbreak — KL entropy in the overlap zone")
    else:
        # ═══ Stage 2a: Shared safe sub ═══
        ent = metrics.get('entropy', 0)
        icv = metrics.get('interior_cv', 0)
        predicted, safe_score, safe_conf, safe_caveat = safe_sub_classify(ent, icv)
        stages.append(safe_sub_stage_dict(predicted, safe_score, safe_conf, ent, icv))
        contributions.extend(safe_sub_contributions(ent, icv))
        caveats.append(safe_caveat)

    overall_conf = bin_conf * stages[-1]['confidence']
    summary = f"{predicted} ({overall_conf:.0%}) [{'+'.join(layers)}]"

    return {
        'classifier': CLASSIFIER_ID, 'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted, 'confidence': round(overall_conf, 4),
        'probabilities': {},
        'summary': summary, 'caveats': caveats,
        'stages': stages, 'contributions': contributions,
        'layers_active': layers,
    }
