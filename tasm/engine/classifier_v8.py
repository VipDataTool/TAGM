"""
ASM Classifier v8: Intrinsic Rules.

Architecture: 2-stage decision tree using self-referencing features.
REVISED: Uses shared binary stage (5.1), shared safe sub (5.4),
harmonized confidence (5.3). KL floor adversarial sub retained
as-is — it was already the strongest separator (13/14 jailbreak).

  Stage 1 (shared): middle_share + weighted net_correction
  Stage 2a (shared): entropy + 0.3*CV (capped confidence)
  Stage 2b (intrinsic): stress_max + 15 * kl_min
    kl_min is the standout intrinsic feature: it measures whether
    the base model disagrees with every token (jailbreak, high floor)
    or just framing tokens (harmful, near-zero floor).

Thresholds are Qwen 2.5 0.5B specific. The KL_FLOOR_WEIGHT = 15
is calibrated to make kl_min swing the score by ~0.5 units, matching
the stress_max variation range.
"""

import math
from engine.classifier_common import (
    binary_classify, binary_stage_dict, binary_contributions,
    safe_sub_classify, safe_sub_stage_dict, safe_sub_contributions,
    sigmoid_confidence,
)

CLASSIFIER_ID = "v8"
CLASSIFIER_NAME = "Intrinsic Rules"
CLASSIFIER_DESC = "Rule-based. Shared binary. KL floor for adversarial sub."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# Adversarial sub-stage
ADV_THRESHOLD = 4.1
ADV_KL_FLOOR_WEIGHT = 15.0


def _extract(metrics):
    nc = metrics.get('net_correction', 0)
    ms = metrics.get('middle_share', 0)
    ic = metrics.get('interior_cv', 0)
    ent = metrics.get('entropy', 0)
    seq_len = metrics.get('seq_len', 1) or 1

    pts = metrics.get('per_token_stress', [])
    if pts and len(pts) > 0:
        stress_mean = sum(pts) / len(pts)
        stress_max = max(pts)
        stress_min = min(pts)
    else:
        stress_mean = metrics.get('stress_score', 0) or 0
        stress_max = stress_mean
        stress_min = stress_mean

    ptk = metrics.get('per_token_kl', [])
    if ptk and len(ptk) > 0:
        kl_min = min(ptk)
        kl_max = max(ptk)
        kl_mean = sum(ptk) / len(ptk)
    else:
        kl_min = 0
        kl_max = metrics.get('kl_divergence', 0)
        kl_mean = kl_max

    return {
        'net_correction': nc, 'middle_share': ms, 'interior_cv': ic,
        'entropy': ent, 'seq_len': seq_len,
        'stress_mean': stress_mean, 'stress_max': stress_max,
        'stress_min': stress_min, 'stress_range': stress_max - stress_min,
        'peak_ratio': stress_max / stress_mean if stress_mean > 0 else 1.0,
        'kl_min': kl_min, 'kl_max': kl_max, 'kl_mean': kl_mean,
        'kl_floor_ratio': kl_min / kl_mean if kl_mean > 0 else 0.0,
    }


def classify(metrics):
    f = _extract(metrics)
    stages = []
    caveats = []

    # ═══ Stage 1: Shared binary ═══
    is_adversarial, bin_score, bin_conf, bin_caveats = binary_classify(
        f['middle_share'], f['net_correction'], f['seq_len'])
    stages.append(binary_stage_dict(
        bin_score, is_adversarial, bin_conf, f['middle_share'], f['net_correction']))
    contributions = list(binary_contributions(f['middle_share'], f['net_correction']))
    caveats.extend(bin_caveats)

    if is_adversarial:
        # ═══ Stage 2b: Stress ceiling + KL floor ═══
        adv_score = f['stress_max'] + ADV_KL_FLOOR_WEIGHT * f['kl_min']
        predicted = 'jailbreak' if adv_score > ADV_THRESHOLD else 'harmful'
        adv_margin = abs(adv_score - ADV_THRESHOLD)
        adv_conf = sigmoid_confidence(adv_margin)

        stages.append({
            'stage': 'adversarial_sub', 'name': 'Stress Ceiling + KL Floor',
            'score': round(adv_score, 6), 'threshold': ADV_THRESHOLD,
            'margin': round(adv_score - ADV_THRESHOLD, 6),
            'confidence': round(adv_conf, 4),
            'left_class': 'harmful', 'right_class': 'jailbreak',
            'chosen': predicted,
            'explanation': (
                f"stress_max({f['stress_max']:.3f}) + 15×kl_min({f['kl_min']:.5f}) "
                f"= {adv_score:.4f} {'>' if predicted == 'jailbreak' else '≤'} "
                f"{ADV_THRESHOLD} → {predicted}"
            ),
        })

        contributions.append({
            'feature': 'stress_max', 'name': 'Stress Ceiling',
            'value': round(f['stress_max'], 4),
            'favors': 'jailbreak' if f['stress_max'] > 3.75 else 'harmful',
            'strength': 'strong' if abs(f['stress_max'] - 3.75) > 0.15 else 'moderate',
        })
        contributions.append({
            'feature': 'kl_min', 'name': 'KL Floor',
            'value': round(f['kl_min'], 6),
            'favors': 'jailbreak' if f['kl_min'] > 0.02 else 'harmful',
            'strength': ('strong' if f['kl_min'] > 0.02
                         else ('moderate' if f['kl_min'] > 0.005 else 'weak')),
        })

        if adv_margin < 0.1:
            caveats.append("Close to the harmful/jailbreak boundary")
    else:
        # ═══ Stage 2a: Shared safe sub ═══
        predicted, safe_score, safe_conf, safe_caveat = safe_sub_classify(
            f['entropy'], f['interior_cv'])
        stages.append(safe_sub_stage_dict(
            predicted, safe_score, safe_conf, f['entropy'], f['interior_cv']))
        contributions.extend(safe_sub_contributions(f['entropy'], f['interior_cv']))
        caveats.append(safe_caveat)

    overall_conf = bin_conf * stages[-1]['confidence']

    # Probabilities from stage confidences
    if is_adversarial:
        sub_conf = stages[-1]['confidence']
        if predicted == 'jailbreak':
            probs = {'benign': 0, 'mild': 0,
                     'harmful': round(1 - sub_conf, 4),
                     'jailbreak': round(sub_conf, 4)}
        else:
            probs = {'benign': 0, 'mild': 0,
                     'harmful': round(sub_conf, 4),
                     'jailbreak': round(1 - sub_conf, 4)}
        leak = round((1 - bin_conf) * 0.5, 4)
        probs['benign'] = leak
        probs['mild'] = leak
        probs[predicted] = round(max(0, probs[predicted] - 2 * leak), 4)
    else:
        sub_conf = stages[-1]['confidence']
        if predicted == 'benign':
            probs = {'benign': round(sub_conf, 4), 'mild': round(1 - sub_conf, 4),
                     'harmful': 0, 'jailbreak': 0}
        else:
            probs = {'benign': round(1 - sub_conf, 4), 'mild': round(sub_conf, 4),
                     'harmful': 0, 'jailbreak': 0}
        leak = round((1 - bin_conf) * 0.5, 4)
        probs['harmful'] = leak
        probs['jailbreak'] = leak
        probs[predicted] = round(max(0, probs[predicted] - 2 * leak), 4)

    summary = (f"{predicted} ({overall_conf:.0%}) via intrinsic rules. "
               f"Key: ms={f['middle_share']:.3f}, "
               f"stress_max={f['stress_max']:.3f}, "
               f"kl_min={f['kl_min']:.5f}")

    return {
        'classifier': CLASSIFIER_ID, 'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted, 'confidence': round(overall_conf, 4),
        'probabilities': probs, 'summary': summary,
        'caveats': caveats, 'stages': stages, 'contributions': contributions,
        'binary': 'adversarial' if is_adversarial else 'safe',
        'diagnostics': {
            'binary_score': round(bin_score, 5),
            'stress_max': round(f['stress_max'], 4),
            'stress_range': round(f['stress_range'], 4),
            'kl_min': round(f['kl_min'], 6),
            'kl_floor_ratio': round(f['kl_floor_ratio'], 4),
            'peak_ratio': round(f['peak_ratio'], 4),
        },
    }
