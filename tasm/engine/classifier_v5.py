"""
ASM Classifier v5: Model-Intrinsic Topographic.

Architecture: 3-stage hierarchical with layered measurement depth.
All features expressed in model-intrinsic units (dimensionless ratios,
bounded ranges, or correlation coefficients). No sample-calibrated
population statistics.

Three measurement layers, each adding depth when available:
  Core (always): ASM features from a single forward pass
  LTP  (opt-in): Directional entropy texture from LTP profiles
  KL   (opt-in): Per-token behavioral divergence (requires base model)

Stages:
  Stage 1 (Elevation): safe vs adversarial
    Core:  correction_per_stress + middle_share
    KL:    + mean_kl (overall behavioral divergence)

  Stage 2a (Texture): benign vs mild
    Core:  entropy
    LTP:   + ltp_ent_std * scale (directional texture)
    KL:    + kl_variance (selective vs uniform divergence)

  Stage 2b (Coupling): harmful vs jailbreak
    Core:  (no strong core-only signal at this boundary)
    LTP:   attr_ltp_corr (selective suppression vs blanket saturation)
    KL:    + kl_interior_concentration (where divergence focuses)

Instrument context: delta spectral structure (effective rank profile)
is reported as diagnostic metadata when available.

Calibrated on n=52 balanced prompts (Qwen 2.5 0.5B).
Thresholds in natural units -- intended to be model-portable.
"""

import math
import numpy as np

CLASSIFIER_ID = "v5"
CLASSIFIER_NAME = "Model-Intrinsic Topographic"
CLASSIFIER_DESC = "Stateless, layered depth: Core + LTP + KL. Natural-unit thresholds."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# ═══ Thresholds in natural feature units ═══

# Stage 1: correction_per_stress + middle_share [+ mean_kl * weight]
STAGE1_THRESHOLD = 0.4884
STAGE1_KL_WEIGHT = 0.1   # KL in nats; typical range 0-5

# Stage 2a: entropy [+ ltp_ent_std * scale] [+ kl_variance * weight]
STAGE2A_THRESHOLD = 1.8157
LTP_ENT_STD_SCALE = 1000.0
STAGE2A_KL_WEIGHT = -0.05  # higher KL variance → more mild-like (selective attention)

# Stage 2b: attr_ltp_corr [+ kl features]
STAGE2B_THRESHOLD = -0.1250

FEATURE_META = {
    'correction_per_stress': ('Correction/Stress', 'net correction relative to field activation'),
    'middle_share':          ('Interior Share', 'fraction on interior tokens'),
    'entropy':               ('Entropy', 'correction distribution uniformity'),
    'ltp_ent_std':           ('LTP Entropy Std', 'variation in directional focus'),
    'attr_ltp_corr':         ('Attr-LTP Coupling', 'attribution-direction correlation'),
    'mean_kl':               ('Mean KL', 'average behavioral divergence from base model'),
    'kl_variance':           ('KL Variance', 'selectivity of behavioral divergence'),
    'kl_interior_share':     ('KL Interior Share', 'fraction of divergence on interior tokens'),
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


def _extract_kl_features(metrics):
    """Extract per-token KL features when available."""
    kl = metrics.get('per_token_kl')
    if kl is None:
        return None
    kl = np.array(kl, dtype=float)
    if len(kl) < 2:
        return None
    mean_kl = float(kl.mean())
    kl_var = float(kl.var())
    # Interior share of KL (middle tokens vs boundary tokens)
    if len(kl) >= 3:
        interior_kl = kl[1:-1].sum()
        total_kl = kl.sum()
        kl_int_share = float(interior_kl / total_kl) if total_kl > 0 else 0.5
    else:
        kl_int_share = 0.5
    return {
        'mean_kl': mean_kl,
        'kl_variance': kl_var,
        'kl_interior_share': kl_int_share,
        'max_kl': float(kl.max()),
        'kl_cv': float(kl.std() / mean_kl) if mean_kl > 0 else 0,
    }


def _sigmoid_confidence(score, threshold, scale=2.0):
    margin = abs(score - threshold)
    return 0.5 + 0.5 * math.tanh(margin * scale)


def classify(metrics):
    """Classify using layered model-intrinsic features."""
    stages = []
    caveats = []
    contributions = []
    layers_active = ['core']

    # Detect available measurement layers
    has_ltp = bool(metrics.get('ltp', {}).get('profiles'))
    kl_feats = _extract_kl_features(metrics)
    has_kl = kl_feats is not None

    if has_ltp:
        layers_active.append('LTP')
    if has_kl:
        layers_active.append('KL')

    # Instrument context
    spectral = metrics.get('spectral_summary', {})

    # ═══ Stage 1: Elevation ═══
    nc = metrics.get('net_correction', 0)
    stress = metrics.get('stress_score', 1.0)
    ms = metrics.get('middle_share', 0)

    cps = nc / stress if stress > 0 else 0
    s1_score = cps + ms

    # KL enhancement: mean behavioral divergence lifts adversarial score
    if has_kl:
        s1_score += kl_feats['mean_kl'] * STAGE1_KL_WEIGHT
        contributions.append({
            'feature': 'mean_kl', 'name': 'Mean KL Divergence',
            'value': round(kl_feats['mean_kl'], 5), 'layer': 'KL',
            'favors': 'adversarial' if kl_feats['mean_kl'] > 1.0 else 'safe',
            'strength': 'strong' if kl_feats['mean_kl'] > 2.0 else ('moderate' if kl_feats['mean_kl'] > 0.5 else 'weak'),
        })

    is_adversarial = s1_score > STAGE1_THRESHOLD
    s1_margin = s1_score - STAGE1_THRESHOLD
    s1_conf = _sigmoid_confidence(s1_score, STAGE1_THRESHOLD, scale=8.0)
    s1_chosen = 'adversarial' if is_adversarial else 'safe'

    stages.append({
        'stage': 'elevation', 'name': 'Elevation (Coverage Intensity)',
        'score': round(s1_score, 5), 'threshold': STAGE1_THRESHOLD,
        'margin': round(s1_margin, 5), 'confidence': round(s1_conf, 4),
        'left_class': 'safe', 'right_class': 'adversarial',
        'chosen': s1_chosen,
        'explanation': (f"Score {s1_score:.4f} ({'>' if is_adversarial else '<='} "
                        f"{STAGE1_THRESHOLD}) -> {s1_chosen} ({s1_conf:.0%})"
                        f" [layers: {'+'.join(layers_active)}]"),
    })

    contributions.insert(0, {
        'feature': 'correction_per_stress', 'name': 'Correction/Stress',
        'value': round(cps, 6), 'layer': 'core',
        'favors': 'adversarial' if cps > 0.023 else 'safe',
        'strength': 'strong' if abs(cps - 0.023) > 0.003 else ('moderate' if abs(cps - 0.023) > 0.001 else 'weak'),
    })
    contributions.insert(1, {
        'feature': 'middle_share', 'name': 'Interior Share',
        'value': round(ms, 4), 'layer': 'core',
        'favors': 'adversarial' if ms > 0.46 else 'safe',
        'strength': 'strong' if abs(ms - 0.46) > 0.10 else ('moderate' if abs(ms - 0.46) > 0.04 else 'weak'),
    })

    if abs(s1_margin) < 0.02:
        caveats.append("Close to the safe/adversarial boundary")

    if is_adversarial:
        # ═══ Stage 2b: Coupling ═══
        alc = _compute_attr_ltp_corr(metrics) if has_ltp else None

        if alc is not None:
            s2b_score = alc
            predicted = 'jailbreak' if s2b_score > STAGE2B_THRESHOLD else 'harmful'
            s2b_conf = _sigmoid_confidence(s2b_score, STAGE2B_THRESHOLD, scale=3.0)

            stages.append({
                'stage': 'coupling', 'name': 'Coupling (Selective vs Saturated)',
                'score': round(s2b_score, 5), 'threshold': STAGE2B_THRESHOLD,
                'margin': round(s2b_score - STAGE2B_THRESHOLD, 5),
                'confidence': round(s2b_conf, 4),
                'left_class': 'harmful', 'right_class': 'jailbreak',
                'chosen': predicted,
                'explanation': (f"Coupling = {alc:+.4f} ({'>' if predicted == 'jailbreak' else '<='} "
                                f"{STAGE2B_THRESHOLD}) -> {predicted} ({s2b_conf:.0%}). "
                                f"{'Blanket saturation' if predicted == 'jailbreak' else 'Selective suppression'}"),
            })

            contributions.append({
                'feature': 'attr_ltp_corr', 'name': 'Attr-LTP Coupling',
                'value': round(alc, 5), 'layer': 'LTP',
                'favors': 'jailbreak' if alc > STAGE2B_THRESHOLD else 'harmful',
                'strength': 'strong' if abs(alc - STAGE2B_THRESHOLD) > 0.2 else ('moderate' if abs(alc - STAGE2B_THRESHOLD) > 0.08 else 'weak'),
            })

            if has_kl:
                contributions.append({
                    'feature': 'kl_interior_share', 'name': 'KL Interior Share',
                    'value': round(kl_feats['kl_interior_share'], 4), 'layer': 'KL',
                    'favors': 'jailbreak' if kl_feats['kl_interior_share'] > 0.6 else 'harmful',
                    'strength': 'weak',  # uncalibrated
                })

            if abs(s2b_score - STAGE2B_THRESHOLD) < 0.08:
                caveats.append("Borderline coupling -- mixed selective and saturated characteristics")
        else:
            # No LTP: can't determine sub-class
            predicted = 'harmful'
            caveats.append("No LTP data -- cannot measure coupling, defaulting to harmful")

    else:
        # ═══ Stage 2a: Texture ═══
        ent = metrics.get('entropy', 0)
        ltp_entropies = _compute_ltp_entropies(metrics.get('ltp')) if has_ltp else []
        ltp_es = float(np.std(ltp_entropies)) if len(ltp_entropies) > 1 else 0.0

        s2a_score = ent + ltp_es * LTP_ENT_STD_SCALE

        # KL enhancement: KL variance indicates selective attention (mild-like)
        if has_kl:
            s2a_score += kl_feats['kl_variance'] * STAGE2A_KL_WEIGHT
            contributions.append({
                'feature': 'kl_variance', 'name': 'KL Variance',
                'value': round(kl_feats['kl_variance'], 5), 'layer': 'KL',
                'favors': 'mild' if kl_feats['kl_variance'] > 0.5 else 'benign',
                'strength': 'weak',  # uncalibrated
            })

        predicted = 'benign' if s2a_score > STAGE2A_THRESHOLD else 'mild'
        s2a_conf = _sigmoid_confidence(s2a_score, STAGE2A_THRESHOLD, scale=4.0)

        active_layers = 'ASM'
        if has_ltp: active_layers += ' + LTP'
        if has_kl: active_layers += ' + KL'

        stages.append({
            'stage': 'texture', 'name': f'Texture ({active_layers})',
            'score': round(s2a_score, 5), 'threshold': STAGE2A_THRESHOLD,
            'margin': round(s2a_score - STAGE2A_THRESHOLD, 5),
            'confidence': round(s2a_conf, 4),
            'left_class': 'mild', 'right_class': 'benign',
            'chosen': predicted,
            'explanation': (f"Texture score {s2a_score:.4f} ({'>' if predicted == 'benign' else '<='} "
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

    # Measurement depth indicator
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

    # Attach instrument context if available
    if spectral:
        result['instrument'] = {
            'spectral': spectral,
            'interpretation': _interpret_spectral(spectral),
        }

    return result


def _interpret_spectral(spectral):
    """Human-readable interpretation of the delta spectral structure."""
    rank = spectral.get('mean_eff_rank', 0)
    attn_rank = spectral.get('attn_mean_rank', 0)
    mlp_rank = spectral.get('mlp_mean_rank', 0)
    top1 = spectral.get('mean_top1_share', 0)

    parts = []
    if rank > 0:
        if rank < 5:
            parts.append(f"Low-rank delta (eff. rank {rank:.1f}): RLHF made surgical corrections in a few directions")
        elif rank < 15:
            parts.append(f"Moderate-rank delta (eff. rank {rank:.1f}): RLHF modified a moderate subspace")
        else:
            parts.append(f"High-rank delta (eff. rank {rank:.1f}): RLHF broadly reshaped sublayer representations")

    if attn_rank > 0 and mlp_rank > 0:
        if attn_rank > mlp_rank * 1.5:
            parts.append(f"Attention deltas higher rank ({attn_rank:.0f}) than MLP ({mlp_rank:.0f}): alignment primarily modified attention routing")
        elif mlp_rank > attn_rank * 1.5:
            parts.append(f"MLP deltas higher rank ({mlp_rank:.0f}) than attention ({attn_rank:.0f}): alignment primarily modified value computation")

    if top1 > 0.5:
        parts.append(f"Dominant first singular vector ({top1:.0%}): correction is mostly rank-1")

    return '; '.join(parts) if parts else "Spectral structure within normal range"
