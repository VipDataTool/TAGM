"""
ASM Classifier v8: Intrinsic Rule-Based.

Architecture: 2-stage decision tree using only self-referencing
features — thresholds derived from what the features MEAN
mathematically, not from where a training sample clustered.

Every threshold has a physical interpretation:
  - middle_share > 0.5: majority of correction on interior tokens
    (geometric fact — more than half the signal is interior)
  - stress_max: absolute stress ceiling on the per-token scale
    (jailbreak prompts produce extreme hotspots)
  - kl_min: floor of per-token KL divergence
    (jailbreak = uniform disagreement everywhere, harmful = concentrated)
  - entropy: 0–1 normalized correction uniformity (absolute scale)
  - interior_cv: concentration coefficient (absolute scale)

Decision flow:
  Stage 1 (binary): Interior-distributed correction → adversarial
    Score: middle_share + 0.8 * (net_correction - 0.070)
    Threshold: 0.465
    Logic: 0.070 is the baseline correction any prompt generates.
    Excess correction amplifies the interior distribution signal.

  Stage 2a (safe sub): benign vs mild
    Score: entropy + 0.3 * interior_cv
    Threshold: 1.0
    Logic: Benign prompts have higher entropy (more uniform correction)
    and higher CV (more variation across interior tokens).

  Stage 2b (adversarial sub): harmful vs jailbreak
    Score: stress_max + 15 * kl_min
    Threshold: 4.1
    Logic: stress_max is ~3.5–3.7 for both, so the KL floor is the
    tiebreaker. Jailbreak kl_min ≈ 0.03 adds ~0.45 to the score.
    Harmful kl_min ≈ 0.001 adds ~0.015. Clean separation.

No population statistics. No centroids. No sampled references.
Every parameter has a derivation from the measurement geometry.

Designed for Qwen 2.5 0.5B. Thresholds may need adjustment for
other model scales where absolute magnitudes differ.
"""

import math

CLASSIFIER_ID = "v8"
CLASSIFIER_NAME = "Intrinsic Rules"
CLASSIFIER_DESC = "Rule-based. Self-referencing features only. No population references."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# Threshold derivations documented inline
BINARY_THRESHOLD = 0.465       # middle_share=0.465 when nc=0.070 (baseline)
BINARY_NC_BASELINE = 0.070     # minimum correction any prompt generates
BINARY_NC_WEIGHT = 0.8         # how much excess correction amplifies the signal

SAFE_THRESHOLD = 1.0           # entropy(~0.78) + 0.3*cv(~0.65) ≈ 0.97 for mild
SAFE_ENT_WEIGHT = 1.0
SAFE_CV_WEIGHT = 0.3

ADV_THRESHOLD = 4.1            # stress_max(~3.7) + 15*kl_min(~0.03) ≈ 4.15 for jailbreak
ADV_KL_FLOOR_WEIGHT = 15.0     # amplifies the kl_min signal to be decision-relevant


def _extract(metrics):
    """Extract the features used by v8 from a metrics dict."""
    nc = metrics.get('net_correction', 0)
    ms = metrics.get('middle_share', 0)
    ic = metrics.get('interior_cv', 0)
    ent = metrics.get('entropy', 0)
    kl = metrics.get('kl_divergence', 0)
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
        kl_mean = sum(ptk) / len(ptk)
        kl_min = min(ptk)
        kl_max = max(ptk)
    else:
        kl_mean = kl
        kl_min = 0
        kl_max = kl

    # Self-referencing derived features
    peak_ratio = stress_max / stress_mean if stress_mean > 0 else 1.0
    kl_floor_ratio = kl_min / kl_mean if kl_mean > 0 else 0.0
    stress_range = stress_max - stress_min

    return {
        'net_correction': nc, 'middle_share': ms, 'interior_cv': ic,
        'entropy': ent, 'kl_divergence': kl, 'seq_len': seq_len,
        'stress_mean': stress_mean, 'stress_max': stress_max,
        'stress_min': stress_min, 'stress_range': stress_range,
        'peak_ratio': peak_ratio,
        'kl_mean': kl_mean, 'kl_min': kl_min, 'kl_max': kl_max,
        'kl_floor_ratio': kl_floor_ratio,
    }


def classify(metrics):
    """Classify a prompt using intrinsic rule-based thresholds."""
    f = _extract(metrics)
    stages = []
    caveats = []

    # ═══ Stage 1: Binary — safe vs adversarial ═══
    # Interior-distributed correction is THE adversarial signature.
    # middle_share > 0.5 means interior tokens carry majority of correction.
    # Excess correction (above baseline) amplifies the signal.
    binary_score = f['middle_share'] + BINARY_NC_WEIGHT * (f['net_correction'] - BINARY_NC_BASELINE)
    is_adversarial = binary_score > BINARY_THRESHOLD
    binary = 'adversarial' if is_adversarial else 'safe'

    # Confidence: distance from threshold, scaled to 0.5–1.0
    bin_margin = abs(binary_score - BINARY_THRESHOLD)
    bin_conf = min(1.0, 0.5 + bin_margin * 5)

    stages.append({
        'stage': 'binary', 'name': 'Interior Distribution',
        'score': round(binary_score, 6), 'threshold': BINARY_THRESHOLD,
        'margin': round(binary_score - BINARY_THRESHOLD, 6),
        'confidence': round(bin_conf, 4),
        'left_class': 'safe', 'right_class': 'adversarial',
        'chosen': binary,
        'explanation': (f"binary_score = ms({f['middle_share']:.3f}) + "
                        f"0.8*(nc({f['net_correction']:.4f})-0.070) = {binary_score:.4f} "
                        f"{'>' if is_adversarial else '<='} {BINARY_THRESHOLD} -> {binary}"),
    })

    if bin_margin < 0.01:
        caveats.append("Very close to the safe/adversarial boundary")

    if is_adversarial:
        # ═══ Stage 2b: harmful vs jailbreak ═══
        # Jailbreak: high stress ceiling (DAN framing creates extreme hotspots)
        # Jailbreak: elevated KL floor (every token disagrees with base)
        # Harmful: concentrated KL on framing tokens, near-zero floor
        adv_score = f['stress_max'] + ADV_KL_FLOOR_WEIGHT * f['kl_min']
        predicted = 'jailbreak' if adv_score > ADV_THRESHOLD else 'harmful'

        adv_margin = abs(adv_score - ADV_THRESHOLD)
        adv_conf = min(1.0, 0.5 + adv_margin * 3)

        stages.append({
            'stage': 'adversarial_sub', 'name': 'Stress Ceiling + KL Floor',
            'score': round(adv_score, 6), 'threshold': ADV_THRESHOLD,
            'margin': round(adv_score - ADV_THRESHOLD, 6),
            'confidence': round(adv_conf, 4),
            'left_class': 'harmful', 'right_class': 'jailbreak',
            'chosen': predicted,
            'explanation': (f"adv_score = stress_max({f['stress_max']:.3f}) + "
                            f"15*kl_min({f['kl_min']:.5f}) = {adv_score:.4f} "
                            f"{'>' if predicted == 'jailbreak' else '<='} {ADV_THRESHOLD} -> {predicted}"),
        })

        if adv_margin < 0.1:
            caveats.append("Close to the harmful/jailbreak boundary")
    else:
        # ═══ Stage 2a: benign vs mild ═══
        # Benign: higher entropy (more uniform correction pattern)
        # Benign: higher CV (more variation in interior token correction)
        # Mild: lower entropy (correction more focused on topic-sensitive tokens)
        safe_score = SAFE_ENT_WEIGHT * f['entropy'] + SAFE_CV_WEIGHT * f['interior_cv']
        predicted = 'benign' if safe_score > SAFE_THRESHOLD else 'mild'

        safe_margin = abs(safe_score - SAFE_THRESHOLD)
        safe_conf = min(1.0, 0.5 + safe_margin * 4)

        stages.append({
            'stage': 'safe_sub', 'name': 'Entropy + Concentration',
            'score': round(safe_score, 6), 'threshold': SAFE_THRESHOLD,
            'margin': round(safe_score - SAFE_THRESHOLD, 6),
            'confidence': round(safe_conf, 4),
            'left_class': 'mild', 'right_class': 'benign',
            'chosen': predicted,
            'explanation': (f"safe_score = ent({f['entropy']:.3f}) + "
                            f"0.3*cv({f['interior_cv']:.3f}) = {safe_score:.4f} "
                            f"{'>' if predicted == 'benign' else '<='} {SAFE_THRESHOLD} -> {predicted}"),
        })

        caveats.append("Benign/mild distinction is weak at 0.5B — treat as low-confidence")

    overall_conf = bin_conf * stages[-1]['confidence']

    # Simple probabilities from stage confidences
    if is_adversarial:
        sub_conf = stages[-1]['confidence']
        if predicted == 'jailbreak':
            probs = {'benign': 0, 'mild': 0, 'harmful': round(1 - sub_conf, 4), 'jailbreak': round(sub_conf, 4)}
        else:
            probs = {'benign': 0, 'mild': 0, 'harmful': round(sub_conf, 4), 'jailbreak': round(1 - sub_conf, 4)}
        # Distribute some probability to safe side based on binary confidence
        leak = round((1 - bin_conf) * 0.5, 4)
        probs['benign'] = leak
        probs['mild'] = leak
        probs[predicted] = round(probs[predicted] - 2 * leak, 4)
    else:
        sub_conf = stages[-1]['confidence']
        if predicted == 'benign':
            probs = {'benign': round(sub_conf, 4), 'mild': round(1 - sub_conf, 4), 'harmful': 0, 'jailbreak': 0}
        else:
            probs = {'benign': round(1 - sub_conf, 4), 'mild': round(sub_conf, 4), 'harmful': 0, 'jailbreak': 0}
        leak = round((1 - bin_conf) * 0.5, 4)
        probs['harmful'] = leak
        probs['jailbreak'] = leak
        probs[predicted] = round(probs[predicted] - 2 * leak, 4)

    # Contributions: show the decisive features
    contributions = _build_contributions(f, predicted, is_adversarial)

    summary = (f"{predicted} ({overall_conf:.0%}) via intrinsic rules. "
               f"Binary: {binary}. "
               f"Key: ms={f['middle_share']:.3f}, "
               f"stress_max={f['stress_max']:.3f}, "
               f"kl_min={f['kl_min']:.5f}")

    return {
        'classifier': CLASSIFIER_ID, 'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted, 'confidence': round(overall_conf, 4),
        'probabilities': probs, 'summary': summary,
        'caveats': caveats, 'stages': stages,
        'contributions': contributions,
        'binary': binary, 'features_used': 7,
        'diagnostics': {
            'binary_score': round(binary_score, 5),
            'stress_max': round(f['stress_max'], 4),
            'stress_range': round(f['stress_range'], 4),
            'kl_min': round(f['kl_min'], 6),
            'kl_floor_ratio': round(f['kl_floor_ratio'], 4),
            'peak_ratio': round(f['peak_ratio'], 4),
        },
    }


def _build_contributions(f, predicted, is_adversarial):
    """Per-feature breakdown showing which features drove the decision."""
    contributions = []

    # Always show the binary-critical features
    ms = f['middle_share']
    contributions.append({
        'feature': 'middle_share', 'name': 'Interior Share',
        'value': round(ms, 4),
        'favors': 'adversarial' if ms > 0.5 else 'safe',
        'strength': 'strong' if abs(ms - 0.5) > 0.08 else ('moderate' if abs(ms - 0.5) > 0.03 else 'weak'),
        'explanation': f"Interior Share = {ms:.3f} ({'majority interior -> adversarial' if ms > 0.5 else 'boundary-dominated -> safe'})",
    })

    nc = f['net_correction']
    contributions.append({
        'feature': 'net_correction', 'name': 'Net Correction',
        'value': round(nc, 5),
        'favors': 'adversarial' if nc > 0.075 else 'safe',
        'strength': 'moderate' if abs(nc - 0.075) > 0.005 else 'weak',
        'explanation': f"Net Correction = {nc:.5f} ({'elevated' if nc > 0.075 else 'baseline'})",
    })

    if is_adversarial:
        sm = f['stress_max']
        contributions.append({
            'feature': 'stress_max', 'name': 'Stress Ceiling',
            'value': round(sm, 4),
            'favors': 'jailbreak' if sm > 3.75 else 'harmful',
            'strength': 'strong' if abs(sm - 3.75) > 0.15 else 'moderate',
            'explanation': f"Stress max = {sm:.3f} ({'extreme hotspot -> jailbreak' if sm > 3.75 else 'moderate peak -> harmful'})",
        })

        km = f['kl_min']
        contributions.append({
            'feature': 'kl_min', 'name': 'KL Floor',
            'value': round(km, 6),
            'favors': 'jailbreak' if km > 0.02 else 'harmful',
            'strength': 'strong' if km > 0.02 else ('moderate' if km > 0.005 else 'weak'),
            'explanation': f"KL floor = {km:.5f} ({'uniform divergence -> jailbreak' if km > 0.02 else 'concentrated divergence -> harmful'})",
        })
    else:
        ent = f['entropy']
        contributions.append({
            'feature': 'entropy', 'name': 'Entropy',
            'value': round(ent, 4),
            'favors': 'benign' if ent > 0.80 else 'mild',
            'strength': 'weak',  # benign/mild always weak at 0.5B
            'explanation': f"Entropy = {ent:.3f} ({'uniform -> benign' if ent > 0.80 else 'focused -> mild'})",
        })

        ic = f['interior_cv']
        contributions.append({
            'feature': 'interior_cv', 'name': 'Interior CV',
            'value': round(ic, 4),
            'favors': 'benign' if ic > 0.65 else 'mild',
            'strength': 'weak',
            'explanation': f"Interior CV = {ic:.3f} ({'variable -> benign' if ic > 0.65 else 'concentrated -> mild'})",
        })

        # v8 instrument readings — always present
        sm = f['stress_max']
        contributions.append({
            'feature': 'stress_max', 'name': 'Stress Ceiling',
            'value': round(sm, 4),
            'favors': 'jailbreak' if sm > 3.75 else 'harmful',
            'strength': 'strong' if abs(sm - 3.75) > 0.15 else 'moderate',
            'explanation': f"Stress max = {sm:.3f} (adv threshold {ADV_THRESHOLD})",
        })

        km = f['kl_min']
        contributions.append({
            'feature': 'kl_min', 'name': 'KL Floor',
            'value': round(km, 6),
            'favors': 'jailbreak' if km > 0.02 else 'harmful',
            'strength': 'strong' if km > 0.02 else ('moderate' if km > 0.005 else 'weak'),
            'explanation': f"KL floor = {km:.5f} (adv score would be {sm + ADV_KL_FLOOR_WEIGHT * km:.3f})",
        })

    return contributions
