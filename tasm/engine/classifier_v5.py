"""
ASM Classifier v5: Model-Intrinsic Topographic.

Architecture: 3-stage hierarchical with layered measurement depth.
REVISED: Uses shared binary stage (5.1), removed 1000x LTP multiplier
from texture stage (5.2), harmonized confidence (5.3), honest
benign/mild handling (5.4).

Three measurement layers, each adding depth when available:
  Core (always): ASM features from a single forward pass
  LTP  (opt-in): Directional entropy texture from LTP profiles
  KL   (opt-in): Per-token behavioral divergence (requires base model)

Stages:
  Stage 1 (shared): middle_share + weighted net_correction
  Stage 2a (shared): entropy + 0.3*CV (capped confidence)
  Stage 2b: layered KL entropy → LTP coupling → stress fallback
"""

import math
import numpy as np
from engine.classifier_common import (
    binary_classify, binary_stage_dict, binary_contributions,
    safe_sub_classify, safe_sub_stage_dict, safe_sub_contributions,
    sigmoid_confidence,
)

CLASSIFIER_ID = "v5"
CLASSIFIER_NAME = "Layered Intrinsic"
CLASSIFIER_DESC = "Layered depth: Core + LTP + KL. Shared binary stage."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# Stage 2b thresholds (natural units, layered)
STAGE2B_KL_THRESHOLD = 0.4587       # kl_entropy [0,1]
STAGE2B_LTP_THRESHOLD = -0.1250     # attr_ltp_corr [-1,1]
STAGE2B_CORE_THRESHOLD = 3.3234     # stress_score


def _compute_kl_entropy(metrics):
    kl = metrics.get('per_token_kl')
    if kl is None: return None
    kl = np.array(kl, dtype=float)
    if len(kl) < 2: return None
    kl = np.clip(kl, 1e-10, None)
    kl_sum = kl.sum()
    if kl_sum <= 0: return None
    probs = kl / kl_sum
    ent = -np.sum(probs * np.log(probs))
    return float(ent / np.log(len(kl)))


def _compute_attr_ltp_corr(metrics):
    signed_attr = metrics.get('signed_attr', [])
    ltp_data = metrics.get('ltp')
    if not signed_attr or not ltp_data: return None
    profiles = ltp_data.get('profiles', [])
    if not profiles: return None
    attr = [abs(a) for a in signed_attr]
    entropies = []
    for p in profiles:
        p = np.array(p, dtype=float); t = p.sum()
        if t > 0:
            normed = p / t; normed = normed[normed > 0]; k = len(p)
            entropies.append(float(-np.sum(normed * np.log(normed)) / np.log(k)) if k > 1 else 1.0)
        else:
            entropies.append(1.0)
    n = min(len(attr), len(entropies))
    if n < 3: return None
    c = np.corrcoef(attr[:n], entropies[:n])[0, 1]
    return float(c) if not np.isnan(c) else None


def classify(metrics):
    stages = []
    caveats = []
    contributions = []
    layers_active = ['core']

    has_ltp = bool(metrics.get('ltp', {}).get('profiles'))
    has_kl = metrics.get('per_token_kl') is not None
    if has_ltp: layers_active.append('LTP')
    if has_kl: layers_active.append('KL')

    nc = metrics.get('net_correction', 0)
    ms = metrics.get('middle_share', 0)
    seq_len = metrics.get('seq_len')

    # ═══ Stage 1: Shared binary ═══
    is_adversarial, bin_score, bin_conf, bin_caveats = binary_classify(ms, nc, seq_len)
    stages.append(binary_stage_dict(bin_score, is_adversarial, bin_conf, ms, nc))
    contributions.extend(binary_contributions(ms, nc))
    caveats.extend(bin_caveats)

    if is_adversarial:
        # ═══ Stage 2b: Layered adversarial sub ═══
        kl_ent = _compute_kl_entropy(metrics) if has_kl else None
        alc = _compute_attr_ltp_corr(metrics) if has_ltp else None
        stress = metrics.get('stress_score', 1.0)

        if kl_ent is not None:
            s2b_score = kl_ent
            s2b_threshold = STAGE2B_KL_THRESHOLD
            predicted = 'jailbreak' if s2b_score > s2b_threshold else 'harmful'
            s2b_conf = sigmoid_confidence(abs(s2b_score - s2b_threshold))
            method = 'KL Entropy'
            explanation = (
                f"KL entropy = {kl_ent:.4f} "
                f"{'>' if predicted == 'jailbreak' else '≤'} {s2b_threshold} → {predicted}. "
                f"{'Uniform divergence (blanket)' if predicted == 'jailbreak' else 'Concentrated divergence (selective)'}"
            )
            contributions.append({
                'feature': 'kl_entropy', 'name': 'KL Entropy',
                'value': round(kl_ent, 5), 'layer': 'KL',
                'favors': 'jailbreak' if kl_ent > s2b_threshold else 'harmful',
                'strength': ('strong' if abs(kl_ent - s2b_threshold) > 0.15
                             else ('moderate' if abs(kl_ent - s2b_threshold) > 0.06 else 'weak')),
            })

        elif alc is not None:
            s2b_score = alc
            s2b_threshold = STAGE2B_LTP_THRESHOLD
            predicted = 'jailbreak' if s2b_score > s2b_threshold else 'harmful'
            s2b_conf = sigmoid_confidence(abs(s2b_score - s2b_threshold))
            method = 'LTP Coupling'
            explanation = (
                f"Attr-LTP coupling = {alc:+.4f} "
                f"{'>' if predicted == 'jailbreak' else '≤'} {s2b_threshold} → {predicted}"
            )
            contributions.append({
                'feature': 'attr_ltp_corr', 'name': 'Attr-LTP Coupling',
                'value': round(alc, 5), 'layer': 'LTP',
                'favors': 'jailbreak' if alc > s2b_threshold else 'harmful',
                'strength': ('strong' if abs(alc - s2b_threshold) > 0.2
                             else ('moderate' if abs(alc - s2b_threshold) > 0.08 else 'weak')),
            })
            caveats.append("No KL data — using LTP coupling (weaker signal)")

        else:
            s2b_score = stress
            s2b_threshold = STAGE2B_CORE_THRESHOLD
            predicted = 'jailbreak' if s2b_score > s2b_threshold else 'harmful'
            s2b_conf = sigmoid_confidence(abs(s2b_score - s2b_threshold))
            method = 'Stress (core)'
            explanation = (
                f"Stress = {stress:.4f} "
                f"{'>' if predicted == 'jailbreak' else '≤'} {s2b_threshold} → {predicted}"
            )
            contributions.append({
                'feature': 'stress_score', 'name': 'Stress Score',
                'value': round(stress, 5), 'layer': 'core',
                'favors': 'jailbreak' if stress > s2b_threshold else 'harmful',
                'strength': ('strong' if abs(stress - s2b_threshold) > 0.15
                             else ('moderate' if abs(stress - s2b_threshold) > 0.05 else 'weak')),
            })
            caveats.append("No KL or LTP data — using stress alone")

        stages.append({
            'stage': 'adversarial_sub', 'name': f'Strategy ({method})',
            'score': round(s2b_score, 5), 'threshold': s2b_threshold,
            'margin': round(s2b_score - s2b_threshold, 5),
            'confidence': round(s2b_conf, 4),
            'left_class': 'harmful', 'right_class': 'jailbreak',
            'chosen': predicted,
            'explanation': explanation,
        })
        if abs(s2b_score - s2b_threshold) < 0.05:
            caveats.append("Borderline harmful/jailbreak")
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
    depth = len(layers_active)
    depth_label = f"{depth}-layer ({'+'.join(layers_active)})"
    summary = f"{predicted} via {stage_path} ({overall_conf:.0%}) [{depth_label}]"

    return {
        'classifier': CLASSIFIER_ID, 'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted, 'confidence': round(overall_conf, 4),
        'probabilities': {},
        'summary': summary, 'caveats': caveats,
        'stages': stages, 'contributions': contributions,
        'layers_active': layers_active,
    }
