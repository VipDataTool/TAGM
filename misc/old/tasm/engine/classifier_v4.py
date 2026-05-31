"""
ASM Classifier v4: Topographic Model.

Architecture: 3-stage hierarchical decision tree with z-normalized feature fusion.
Each stage measures a different dimension of RLHF training coverage using a
structurally distinct measurement axis:

  Stage 1 (Elevation): z(net_correction) + z(middle_share)
    Measures: Coverage intensity -- how loud is the alignment field?
    Separates: safe vs adversarial
    Calibrated accuracy: 86.5% on n=52

  Stage 2a (Texture): z(ASM_entropy) + z(LTP_entropy_variance)
    Measures: Coverage smoothing -- how much pretraining texture remains?
    Separates: benign vs mild
    Calibrated accuracy: 69.2% on n=52
    Requires LTP data.

  Stage 2b (Coupling): z(attr_ltp_corr) + z(-interestingness_variance)
    Measures: Correction uniformity -- selective suppression vs blanket saturation.
    Harmful prompts have SELECTIVE correction (high coupling between attribution
    and LTP entropy, high variance in interestingness). The field knows which
    tokens matter and pushes those specifically.
    Jailbreaks have SATURATED correction (low coupling, low variance). The field
    pushes every token uniformly hard. No token stands out because the entire
    sequence triggers a blanket response.
    Calibrated accuracy: 73.1% on n=52
    Requires LTP data.

    NOTE: This axis is perpendicular to the magnitude axis used in Stage 1.
    It does not measure how HARD the field pushes (both harmful and jailbreak
    are loud). It measures how SELECTIVELY the field pushes. The 73.1% accuracy
    is an honest reflection of the boundary -- many long-form jailbreaks (DAN,
    NoCensor-GPT, academic framing) use selective token correction like harmful
    prompts because their elaborate framing creates hot and cold tokens. The
    short imperative jailbreaks (Ignore all instructions, Enter developer mode)
    are the ones that produce blanket saturation.

Fusion method: z-normalize each feature against population statistics, then sum.
Calibrated on n=52 balanced prompts (Qwen 2.5 0.5B).
"""

import math
import numpy as np

CLASSIFIER_ID = "v4"
CLASSIFIER_NAME = "Topographic Model"
CLASSIFIER_DESC = "Z-normalized fusion: Elevation + Texture + Coupling. n=52 calibration."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# ═══ Population statistics for z-normalization (from n=52) ═══

POP_STATS = {
    'net_correction':   (0.075374, 0.006157),
    'middle_share':     (0.475985, 0.104088),
    'entropy':          (0.796776, 0.054886),
    'ltp_entropy_var':  (0.0000014307, 0.0000013262),
    'attr_ltp_corr':    (-0.175056, 0.258782),
    'interest_var':     (0.926439, 1.082712),
}

# ═══ Stage thresholds ═══

STAGE1_THRESHOLD = -0.1963     # Elevation: z(nc) + z(ms) > threshold => adversarial
STAGE2A_THRESHOLD = -0.7769    # Texture: z(ent) + z(lev) > threshold => benign
STAGE2B_THRESHOLD = 0.9022     # Coupling: z(alc) + z(-iv) > threshold => jailbreak

# ═══ Score distribution stats for Gaussian confidence ═══

STAGE1_STATS = {
    'safe':        (-1.2660, 1.1451),
    'adversarial': (1.2660, 1.6052),
}
STAGE2A_STATS = {
    'benign': (-0.0603, 1.0783),
    'mild':   (-1.0250, 0.9635),
}
STAGE2B_STATS = {
    'harmful':   (-0.9303, 1.2826),
    'jailbreak': (0.1748, 1.0887),
}

FEATURE_META = {
    'net_correction':    ('Net Correction', 'alignment correction magnitude'),
    'middle_share':      ('Interior Share', 'fraction on interior tokens'),
    'entropy':           ('Entropy', 'correction distribution uniformity'),
    'ltp_entropy_var':   ('LTP Entropy Var', 'variance of directional focus across tokens'),
    'attr_ltp_corr':     ('Attr-LTP Coupling', 'correlation between attribution and directional focus'),
    'interest_var':      ('Interestingness Var', 'variance of per-token interestingness scores'),
}


def _z(val, feat):
    """Z-normalize a value against population statistics."""
    mu, sigma = POP_STATS.get(feat, (0, 1))
    return (val - mu) / sigma if sigma > 0 else 0.0


def _gaussian_confidence(score, threshold, stats_below, stats_above):
    """Confidence from Gaussian overlap at a decision boundary."""
    mu_lo, sigma_lo = stats_below
    mu_hi, sigma_hi = stats_above
    sigma_lo = max(sigma_lo, 1e-6)
    sigma_hi = max(sigma_hi, 1e-6)
    ll_lo = -0.5 * ((score - mu_lo) / sigma_lo) ** 2
    ll_hi = -0.5 * ((score - mu_hi) / sigma_hi) ** 2
    max_ll = max(ll_lo, ll_hi)
    p_lo = math.exp(ll_lo - max_ll)
    p_hi = math.exp(ll_hi - max_ll)
    total = p_lo + p_hi
    if score > threshold:
        return max(0.5, min(1.0, p_hi / total))
    else:
        return max(0.5, min(1.0, p_lo / total))


def _compute_ltp_entropy_variance(ltp_data):
    """Variance of per-token LTP directional entropy."""
    if not ltp_data:
        return None
    profiles = ltp_data.get('profiles', [])
    if not profiles:
        return None
    entropies = []
    for p in profiles:
        p = np.array(p, dtype=float)
        t = p.sum()
        if t > 0:
            normed = p / t
            normed = normed[normed > 0]
            k = len(p)
            entropies.append(float(-np.sum(normed * np.log(normed)) / np.log(k)) if k > 1 else 1.0)
        else:
            entropies.append(1.0)
    if len(entropies) < 2:
        return None
    return float(np.var(entropies))


def _compute_attr_ltp_corr(metrics):
    """Correlation between attribution magnitude and LTP directional entropy."""
    signed_attr = metrics.get('signed_attr', [])
    ltp_data = metrics.get('ltp')
    if not signed_attr or not ltp_data:
        return None
    profiles = ltp_data.get('profiles', [])
    if not profiles:
        return None

    attr = [abs(a) for a in signed_attr]
    entropies = []
    for p in profiles:
        p = np.array(p, dtype=float)
        t = p.sum()
        if t > 0:
            normed = p / t
            normed = normed[normed > 0]
            k = len(p)
            entropies.append(float(-np.sum(normed * np.log(normed)) / np.log(k)) if k > 1 else 1.0)
        else:
            entropies.append(1.0)

    n = min(len(attr), len(entropies))
    if n < 3:
        return None
    c = np.corrcoef(attr[:n], entropies[:n])[0, 1]
    return float(c) if not np.isnan(c) else None


def _compute_interestingness_variance(metrics):
    """Variance of per-token interestingness (z-stress * z-focus)."""
    stress = np.array(metrics.get('per_token_stress', []), dtype=float)
    ltp_data = metrics.get('ltp')
    if len(stress) < 3 or not ltp_data:
        return None
    profiles = ltp_data.get('profiles', [])
    if not profiles:
        return None

    entropies = []
    for p in profiles:
        p = np.array(p, dtype=float)
        t = p.sum()
        if t > 0:
            normed = p / t
            normed = normed[normed > 0]
            k = len(p)
            entropies.append(float(-np.sum(normed * np.log(normed)) / np.log(k)) if k > 1 else 1.0)
        else:
            entropies.append(1.0)

    e = np.array(entropies, dtype=float)
    n = min(len(stress), len(e))
    if n < 3:
        return None
    s, e = stress[:n], e[:n]
    s_z = (s - s.mean()) / (s.std() + 1e-10)
    e_z = (e - e.mean()) / (e.std() + 1e-10)
    interest = s_z * (-e_z)  # high stress AND low entropy = interesting
    return float(np.var(interest))


def classify(metrics):
    """Classify using the topographic model."""
    stages = []
    caveats = []
    contributions = []

    # ═══ Stage 1: Elevation (safe vs adversarial) ═══
    nc = metrics.get('net_correction', 0)
    ms = metrics.get('middle_share', 0)
    z_nc = _z(nc, 'net_correction')
    z_ms = _z(ms, 'middle_share')
    s1_score = z_nc + z_ms

    s1_conf = _gaussian_confidence(s1_score, STAGE1_THRESHOLD,
                                    STAGE1_STATS['safe'], STAGE1_STATS['adversarial'])
    is_adversarial = s1_score > STAGE1_THRESHOLD
    s1_chosen = 'adversarial' if is_adversarial else 'safe'

    stages.append({
        'stage': 'elevation', 'name': 'Elevation (Coverage Intensity)',
        'score': round(s1_score, 4), 'threshold': STAGE1_THRESHOLD,
        'margin': round(s1_score - STAGE1_THRESHOLD, 4),
        'confidence': round(s1_conf, 4),
        'left_class': 'safe', 'right_class': 'adversarial',
        'chosen': s1_chosen,
        'explanation': (f"Elevation z-sum = {s1_score:.3f} "
                        f"({'>' if is_adversarial else '<='} {STAGE1_THRESHOLD}) "
                        f"-> {s1_chosen} ({s1_conf:.0%})"),
    })

    contributions.append({
        'feature': 'net_correction', 'name': 'Net Correction',
        'value': round(nc, 6), 'z': round(z_nc, 3),
        'favors': 'adversarial' if z_nc > 0 else 'safe',
        'strength': 'strong' if abs(z_nc) > 1.5 else ('moderate' if abs(z_nc) > 0.5 else 'weak'),
    })
    contributions.append({
        'feature': 'middle_share', 'name': 'Interior Share',
        'value': round(ms, 6), 'z': round(z_ms, 3),
        'favors': 'adversarial' if z_ms > 0 else 'safe',
        'strength': 'strong' if abs(z_ms) > 1.5 else ('moderate' if abs(z_ms) > 0.5 else 'weak'),
    })

    if abs(s1_score - STAGE1_THRESHOLD) < 0.3:
        caveats.append("Close to the safe/adversarial boundary")

    if is_adversarial:
        # ═══ Stage 2b: Coupling (harmful vs jailbreak) ═══
        # z(attr_ltp_corr) + z(-interestingness_var)
        # Higher = more uniform/saturated = jailbreak-like
        # Lower = more selective/coupled = harmful-like
        alc = _compute_attr_ltp_corr(metrics)
        iv = _compute_interestingness_variance(metrics)
        has_coupling = alc is not None and iv is not None

        if has_coupling:
            z_alc = _z(alc, 'attr_ltp_corr')
            z_neg_iv = -_z(iv, 'interest_var')
            s2b_score = z_alc + z_neg_iv

            s2b_conf = _gaussian_confidence(s2b_score, STAGE2B_THRESHOLD,
                                             STAGE2B_STATS['harmful'], STAGE2B_STATS['jailbreak'])
            predicted = 'jailbreak' if s2b_score > STAGE2B_THRESHOLD else 'harmful'

            stages.append({
                'stage': 'coupling', 'name': 'Coupling (Selective vs Saturated)',
                'score': round(s2b_score, 4), 'threshold': STAGE2B_THRESHOLD,
                'margin': round(s2b_score - STAGE2B_THRESHOLD, 4),
                'confidence': round(s2b_conf, 4),
                'left_class': 'harmful', 'right_class': 'jailbreak',
                'chosen': predicted,
                'explanation': (f"Coupling z-sum = {s2b_score:.3f} "
                                f"({'>' if predicted == 'jailbreak' else '<='} {STAGE2B_THRESHOLD}) "
                                f"-> {predicted} ({s2b_conf:.0%})"),
            })

            contributions.append({
                'feature': 'attr_ltp_corr', 'name': 'Attr-LTP Coupling',
                'value': round(alc, 6), 'z': round(z_alc, 3),
                'favors': 'jailbreak' if z_alc > 0 else 'harmful',
                'strength': 'strong' if abs(z_alc) > 1.5 else ('moderate' if abs(z_alc) > 0.5 else 'weak'),
            })
            contributions.append({
                'feature': 'interest_var', 'name': 'Interestingness Var',
                'value': round(iv, 6), 'z': round(-_z(iv, 'interest_var'), 3),
                'favors': 'jailbreak' if _z(iv, 'interest_var') < 0 else 'harmful',
                'strength': 'strong' if abs(_z(iv, 'interest_var')) > 1.5 else ('moderate' if abs(_z(iv, 'interest_var')) > 0.5 else 'weak'),
            })

            if abs(s2b_score - STAGE2B_THRESHOLD) < 0.5:
                caveats.append("Borderline harmful/jailbreak -- correction has mixed selective and saturated characteristics")
        else:
            # No LTP data -- fall back to adversarial without sub-classification
            predicted = 'harmful'
            caveats.append("No LTP data available -- cannot distinguish harmful from jailbreak on coupling axis, defaulting to harmful")

    else:
        # ═══ Stage 2a: Texture (benign vs mild) ═══
        ent = metrics.get('entropy', 0)
        ltp_data = metrics.get('ltp')
        ltp_ev = _compute_ltp_entropy_variance(ltp_data)
        has_ltp = ltp_ev is not None

        z_ent = _z(ent, 'entropy')
        z_lev = _z(ltp_ev, 'ltp_entropy_var') if has_ltp else 0.0
        s2a_score = z_ent + z_lev

        s2a_conf = _gaussian_confidence(s2a_score, STAGE2A_THRESHOLD,
                                         STAGE2A_STATS['mild'], STAGE2A_STATS['benign'])
        predicted = 'benign' if s2a_score > STAGE2A_THRESHOLD else 'mild'

        stage_name = 'Texture (ASM + LTP)' if has_ltp else 'Texture (ASM only)'
        stages.append({
            'stage': 'texture', 'name': stage_name,
            'score': round(s2a_score, 4), 'threshold': STAGE2A_THRESHOLD,
            'margin': round(s2a_score - STAGE2A_THRESHOLD, 4),
            'confidence': round(s2a_conf, 4),
            'left_class': 'mild', 'right_class': 'benign',
            'chosen': predicted,
            'explanation': (f"Texture z-sum = {s2a_score:.3f} "
                            f"({'>' if predicted == 'benign' else '<='} {STAGE2A_THRESHOLD}) "
                            f"-> {predicted} ({s2a_conf:.0%})"),
        })

        contributions.append({
            'feature': 'entropy', 'name': 'Entropy',
            'value': round(ent, 6), 'z': round(z_ent, 3),
            'favors': 'benign' if z_ent > 0 else 'mild',
            'strength': 'strong' if abs(z_ent) > 1.5 else ('moderate' if abs(z_ent) > 0.5 else 'weak'),
        })
        if has_ltp:
            contributions.append({
                'feature': 'ltp_entropy_var', 'name': 'LTP Entropy Variance',
                'value': round(ltp_ev, 10), 'z': round(z_lev, 3),
                'favors': 'benign' if z_lev > 0 else 'mild',
                'strength': 'strong' if abs(z_lev) > 1.5 else ('moderate' if abs(z_lev) > 0.5 else 'weak'),
            })
        else:
            caveats.append("No LTP data -- texture measurement uses ASM entropy only")

        caveats.append("Benign/mild boundary is weak at 0.5B -- texture signal is subtle")

    overall_conf = s1_conf * stages[-1]['confidence']

    stage_path = ' -> '.join(s['chosen'] for s in stages)
    summary = f"{predicted} via {stage_path} ({overall_conf:.0%})"

    return {
        'classifier': CLASSIFIER_ID,
        'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted,
        'confidence': round(overall_conf, 4),
        'probabilities': {},  # v4 uses stages, not global probabilities
        'summary': summary,
        'caveats': caveats,
        'stages': stages,
        'contributions': contributions,
    }
