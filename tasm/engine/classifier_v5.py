"""
ASM Classifier v5: Model-Intrinsic Topographic.

Architecture: 3-stage hierarchical with layered measurement depth.
All features in model-intrinsic units. No sample-calibrated population stats.

Three measurement layers, each adding depth when available:
  Core (always): ASM features from a single forward pass
  LTP  (opt-in): Directional entropy texture from LTP profiles
  KL   (opt-in): Per-token behavioral divergence (requires base model)

Stages:
  Stage 1 (Elevation): safe vs adversarial
    Core:  correction_per_stress + middle_share
    Both dimensionless. Accuracy: 82.7% on n=52.

  Stage 2a (Texture): benign vs mild
    Core:  entropy [0,1]
    LTP:   + ltp_ent_std * 1000 (directional texture variation)
    KL does not help at this boundary (d < 0.44).
    Accuracy: 69.2% on n=52.

  Stage 2b (Correction Strategy): harmful vs jailbreak
    Layered with fallback:
      KL (best):  kl_entropy -- how uniformly distributed is the behavioral
                  divergence across the sequence? Harmful concentrates KL on
                  framing tokens (low entropy). Jailbreaks spread KL across
                  the full sequence (high entropy). 81.8% on n=63.
      LTP (fallback): attr_ltp_corr -- selective vs saturated correction.
                  73.1% on n=52.
      Core (fallback): stress_score alone. 80.8% on n=52. Not structurally
                  perpendicular but functional.

Instrument context: delta spectral structure reported as diagnostic metadata.

Thresholds in natural units -- intended to be model-portable.
"""

import math
import numpy as np

CLASSIFIER_ID = "v5"
CLASSIFIER_NAME = "Model-Intrinsic Topographic"
CLASSIFIER_DESC = "Stateless, layered depth: Core + LTP + KL. Natural-unit thresholds."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# ═══ Thresholds in natural feature units ═══

# Stage 1: correction_per_stress + middle_share
STAGE1_THRESHOLD = 0.4884

# Stage 2a: entropy + ltp_ent_std * scale
STAGE2A_THRESHOLD = 1.8157
LTP_ENT_STD_SCALE = 1000.0

# Stage 2b: layered
STAGE2B_KL_THRESHOLD = 0.4587       # kl_entropy [0,1], higher = jailbreak
STAGE2B_LTP_THRESHOLD = -0.1250     # attr_ltp_corr [-1,1], higher = jailbreak
STAGE2B_CORE_THRESHOLD = 3.3234     # stress_score, higher = jailbreak

FEATURE_META = {
    'correction_per_stress': ('Correction/Stress', 'net correction relative to field activation'),
    'middle_share':          ('Interior Share', 'fraction on interior tokens'),
    'entropy':               ('Entropy', 'correction distribution uniformity'),
    'ltp_ent_std':           ('LTP Entropy Std', 'variation in directional focus'),
    'attr_ltp_corr':         ('Attr-LTP Coupling', 'attribution-direction correlation'),
    'kl_entropy':            ('KL Entropy', 'uniformity of behavioral divergence across tokens'),
    'stress_score':          ('Stress Score', 'correction pressure at mid layers'),
}


def _compute_ltp_entropies(ltp_data):
    if not ltp_data: return []
    profiles = ltp_data.get('profiles', [])
    if not profiles: return []
    out = []
    for p in profiles:
        p = np.array(p, dtype=float); t = p.sum()
        if t > 0:
            normed = p / t; normed = normed[normed > 0]; k = len(p)
            out.append(float(-np.sum(normed * np.log(normed)) / np.log(k)) if k > 1 else 1.0)
        else:
            out.append(1.0)
    return out


def _compute_attr_ltp_corr(metrics):
    signed_attr = metrics.get('signed_attr', [])
    ltp_data = metrics.get('ltp')
    if not signed_attr or not ltp_data: return None
    entropies = _compute_ltp_entropies(ltp_data)
    attr = [abs(a) for a in signed_attr]
    n = min(len(attr), len(entropies))
    if n < 3: return None
    c = np.corrcoef(attr[:n], entropies[:n])[0, 1]
    return float(c) if not np.isnan(c) else None


def _compute_kl_entropy(metrics):
    """Normalized Shannon entropy of per-token KL divergence distribution.
    [0,1]. Low = KL concentrated on few tokens. High = KL spread uniformly."""
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


def _sigmoid_confidence(score, threshold, scale=2.0):
    margin = abs(score - threshold)
    return 0.5 + 0.5 * math.tanh(margin * scale)


def classify(metrics):
    """Classify using layered model-intrinsic features."""
    stages = []
    caveats = []
    contributions = []
    layers_active = ['core']

    # Detect available layers
    has_ltp = bool(metrics.get('ltp', {}).get('profiles'))
    has_kl = metrics.get('per_token_kl') is not None

    if has_ltp: layers_active.append('LTP')
    if has_kl: layers_active.append('KL')

    spectral = metrics.get('spectral_summary', {})

    # ═══ Stage 1: Elevation ═══
    nc = metrics.get('net_correction', 0)
    stress = metrics.get('stress_score', 1.0)
    ms = metrics.get('middle_share', 0)

    cps = nc / stress if stress > 0 else 0
    s1_score = cps + ms

    is_adversarial = s1_score > STAGE1_THRESHOLD
    s1_conf = _sigmoid_confidence(s1_score, STAGE1_THRESHOLD, scale=8.0)
    s1_chosen = 'adversarial' if is_adversarial else 'safe'

    stages.append({
        'stage': 'elevation', 'name': 'Elevation (Coverage Intensity)',
        'score': round(s1_score, 5), 'threshold': STAGE1_THRESHOLD,
        'margin': round(s1_score - STAGE1_THRESHOLD, 5),
        'confidence': round(s1_conf, 4),
        'left_class': 'safe', 'right_class': 'adversarial',
        'chosen': s1_chosen,
        'explanation': (f"Correction/stress ({cps:.5f}) + interior share ({ms:.3f}) "
                        f"= {s1_score:.4f} ({'>' if is_adversarial else '<='} "
                        f"{STAGE1_THRESHOLD}) -> {s1_chosen} ({s1_conf:.0%})"),
    })

    contributions.append({
        'feature': 'correction_per_stress', 'name': 'Correction/Stress',
        'value': round(cps, 6), 'layer': 'core',
        'favors': 'adversarial' if cps > 0.023 else 'safe',
        'strength': 'strong' if abs(cps - 0.023) > 0.003 else ('moderate' if abs(cps - 0.023) > 0.001 else 'weak'),
    })
    contributions.append({
        'feature': 'middle_share', 'name': 'Interior Share',
        'value': round(ms, 4), 'layer': 'core',
        'favors': 'adversarial' if ms > 0.46 else 'safe',
        'strength': 'strong' if abs(ms - 0.46) > 0.10 else ('moderate' if abs(ms - 0.46) > 0.04 else 'weak'),
    })

    if abs(s1_score - STAGE1_THRESHOLD) < 0.02:
        caveats.append("Close to the safe/adversarial boundary")

    if is_adversarial:
        # ═══ Stage 2b: Correction Strategy ═══
        # Layered: KL > LTP > Core fallback

        kl_ent = _compute_kl_entropy(metrics) if has_kl else None
        alc = _compute_attr_ltp_corr(metrics) if has_ltp else None

        if kl_ent is not None:
            # Best layer: KL entropy
            s2b_score = kl_ent
            s2b_threshold = STAGE2B_KL_THRESHOLD
            predicted = 'jailbreak' if s2b_score > s2b_threshold else 'harmful'
            s2b_conf = _sigmoid_confidence(s2b_score, s2b_threshold, scale=5.0)
            stage_method = 'KL Entropy'

            explanation = (f"KL entropy = {kl_ent:.4f} ({'>' if predicted == 'jailbreak' else '<='} "
                           f"{s2b_threshold}) -> {predicted} ({s2b_conf:.0%}). "
                           f"{'Divergence spread uniformly (blanket response)' if predicted == 'jailbreak' else 'Divergence concentrated on framing tokens (selective suppression)'}")

            contributions.append({
                'feature': 'kl_entropy', 'name': 'KL Entropy',
                'value': round(kl_ent, 5), 'layer': 'KL',
                'favors': 'jailbreak' if kl_ent > s2b_threshold else 'harmful',
                'strength': 'strong' if abs(kl_ent - s2b_threshold) > 0.15 else ('moderate' if abs(kl_ent - s2b_threshold) > 0.06 else 'weak'),
            })

            # Also report coupling as secondary diagnostic if available
            if alc is not None:
                contributions.append({
                    'feature': 'attr_ltp_corr', 'name': 'Attr-LTP Coupling',
                    'value': round(alc, 5), 'layer': 'LTP',
                    'favors': 'jailbreak' if alc > STAGE2B_LTP_THRESHOLD else 'harmful',
                    'strength': 'weak',  # secondary diagnostic
                })

        elif alc is not None:
            # Fallback: LTP coupling
            s2b_score = alc
            s2b_threshold = STAGE2B_LTP_THRESHOLD
            predicted = 'jailbreak' if s2b_score > s2b_threshold else 'harmful'
            s2b_conf = _sigmoid_confidence(s2b_score, s2b_threshold, scale=3.0)
            stage_method = 'LTP Coupling'

            explanation = (f"Attr-LTP coupling = {alc:+.4f} ({'>' if predicted == 'jailbreak' else '<='} "
                           f"{s2b_threshold}) -> {predicted} ({s2b_conf:.0%}). "
                           f"{'Blanket saturation' if predicted == 'jailbreak' else 'Selective suppression'}")

            contributions.append({
                'feature': 'attr_ltp_corr', 'name': 'Attr-LTP Coupling',
                'value': round(alc, 5), 'layer': 'LTP',
                'favors': 'jailbreak' if alc > s2b_threshold else 'harmful',
                'strength': 'strong' if abs(alc - s2b_threshold) > 0.2 else ('moderate' if abs(alc - s2b_threshold) > 0.08 else 'weak'),
            })
            caveats.append("No KL data -- using LTP coupling (73% accuracy vs 82% with KL)")

        else:
            # Core fallback: stress alone
            s2b_score = stress
            s2b_threshold = STAGE2B_CORE_THRESHOLD
            predicted = 'jailbreak' if s2b_score > s2b_threshold else 'harmful'
            s2b_conf = _sigmoid_confidence(s2b_score, s2b_threshold, scale=4.0)
            stage_method = 'Stress (core)'

            explanation = (f"Stress = {stress:.4f} ({'>' if predicted == 'jailbreak' else '<='} "
                           f"{s2b_threshold}) -> {predicted} ({s2b_conf:.0%})")

            contributions.append({
                'feature': 'stress_score', 'name': 'Stress Score',
                'value': round(stress, 5), 'layer': 'core',
                'favors': 'jailbreak' if stress > s2b_threshold else 'harmful',
                'strength': 'strong' if abs(stress - s2b_threshold) > 0.15 else ('moderate' if abs(stress - s2b_threshold) > 0.05 else 'weak'),
            })
            caveats.append("No KL or LTP data -- using stress alone (intensity, not structure)")

        stages.append({
            'stage': 'strategy', 'name': f'Strategy ({stage_method})',
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
        # ═══ Stage 2a: Texture ═══
        ent = metrics.get('entropy', 0)
        ltp_entropies = _compute_ltp_entropies(metrics.get('ltp')) if has_ltp else []
        ltp_es = float(np.std(ltp_entropies)) if len(ltp_entropies) > 1 else 0.0

        s2a_score = ent + ltp_es * LTP_ENT_STD_SCALE

        predicted = 'benign' if s2a_score > STAGE2A_THRESHOLD else 'mild'
        s2a_conf = _sigmoid_confidence(s2a_score, STAGE2A_THRESHOLD, scale=4.0)

        active = 'ASM'
        if has_ltp: active += ' + LTP'

        stages.append({
            'stage': 'texture', 'name': f'Texture ({active})',
            'score': round(s2a_score, 5), 'threshold': STAGE2A_THRESHOLD,
            'margin': round(s2a_score - STAGE2A_THRESHOLD, 5),
            'confidence': round(s2a_conf, 4),
            'left_class': 'mild', 'right_class': 'benign',
            'chosen': predicted,
            'explanation': (f"Entropy ({ent:.4f}) + LTP texture ({ltp_es:.6f} x {LTP_ENT_STD_SCALE:.0f}) "
                            f"= {s2a_score:.4f} ({'>' if predicted == 'benign' else '<='} "
                            f"{STAGE2A_THRESHOLD}) -> {predicted} ({s2a_conf:.0%})"),
        })

        contributions.append({
            'feature': 'entropy', 'name': 'Entropy',
            'value': round(ent, 5), 'layer': 'core',
            'favors': 'benign' if ent > 0.78 else 'mild',
            'strength': 'strong' if abs(ent - 0.78) > 0.05 else ('moderate' if abs(ent - 0.78) > 0.02 else 'weak'),
        })
        if has_ltp and ltp_es > 0:
            contributions.append({
                'feature': 'ltp_ent_std', 'name': 'LTP Entropy Std',
                'value': round(ltp_es, 8), 'layer': 'LTP',
                'favors': 'benign' if ltp_es > 0.0011 else 'mild',
                'strength': 'moderate' if abs(ltp_es - 0.0011) > 0.0003 else 'weak',
            })
        if not has_ltp:
            caveats.append("No LTP data -- texture uses ASM entropy only")

        caveats.append("Benign/mild boundary is inherently weak at 0.5B")

    overall_conf = s1_conf * stages[-1]['confidence']
    stage_path = ' -> '.join(s['chosen'] for s in stages)

    depth = len(layers_active)
    depth_label = f"{depth}-layer ({'+'.join(layers_active)})"
    summary = f"{predicted} via {stage_path} ({overall_conf:.0%}) [{depth_label}]"

    result = {
        'classifier': CLASSIFIER_ID,
        'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted,
        'confidence': round(overall_conf, 4),
        'probabilities': {},
        'summary': summary,
        'caveats': caveats,
        'stages': stages,
        'contributions': contributions,
        'layers_active': layers_active,
    }

    if spectral:
        result['instrument'] = {
            'spectral': spectral,
            'interpretation': _interpret_spectral(spectral),
        }

    return result


def _interpret_spectral(spectral):
    rank = spectral.get('mean_eff_rank', 0)
    attn_rank = spectral.get('attn_mean_rank', 0)
    mlp_rank = spectral.get('mlp_mean_rank', 0)
    top1 = spectral.get('mean_top1_share', 0)
    parts = []
    if rank > 0:
        if rank < 5:
            parts.append(f"Low-rank delta (eff. rank {rank:.1f}): surgical RLHF corrections")
        elif rank < 15:
            parts.append(f"Moderate-rank delta (eff. rank {rank:.1f}): targeted subspace modification")
        else:
            parts.append(f"High-rank delta (eff. rank {rank:.1f}): broad RLHF reshaping")
    if attn_rank > 0 and mlp_rank > 0:
        if attn_rank > mlp_rank * 1.5:
            parts.append(f"Attention-dominant ({attn_rank:.0f} vs MLP {mlp_rank:.0f})")
        elif mlp_rank > attn_rank * 1.5:
            parts.append(f"MLP-dominant ({mlp_rank:.0f} vs attn {attn_rank:.0f})")
    if top1 > 0.5:
        parts.append(f"Rank-1 dominant ({top1:.0%} energy in first SV)")
    return '; '.join(parts) if parts else "Spectral structure within normal range"
