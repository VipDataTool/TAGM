"""
ASM Classifier v6: Tetrahedral.

Architecture: Benign = origin. Three axes rise from it.
Four model-intrinsic signals, no sample calibration, no population stats.

  Signal 1 (Elevation): correction_per_stress + middle_share
    Both dimensionless. High = adversarial territory.
    Gates the harmful and jailbreak axes -- they only fire
    when elevation exceeds the safe floor.

  Signal 2 (Terminal KL): KL divergence at the last token
    Measures the instruct model's prediction confidence at the
    sequence-ending position relative to the base model.
    High terminal KL = confident response pathway (benign-like).
    Low terminal KL = uncertain/cautious pathway (mild-like).
    Discovered via punctuation probe experiments: terminal punctuation
    carries strong RLHF signal about how the model intends to respond.

  Signal 3 (KL Entropy): Normalized Shannon entropy of per-token KL
    Measures how uniformly behavioral divergence spreads across tokens.
    Low KL entropy = divergence concentrated on framing tokens (harmful).
    High KL entropy = divergence spread uniformly (jailbreak blanket response).

  Signal 4 (Benign): The origin. No signal = no RLHF excursion = benign.
    Classified by absence. If no axis pulls hard enough, you stay at the
    valley floor where alignment training left no mark.

Geometry: Regular tetrahedron with benign at the center.
  - Mild vertex: low terminal KL, low elevation
  - Harmful vertex: high elevation, low KL entropy (selective)
  - Jailbreak vertex: high elevation, high KL entropy (blanket)
  - Benign: origin (no axis exceeds threshold)

Stateless. All thresholds in natural units.
Requires base model for KL features (Signals 2 & 3).
Falls back to ASM-only elevation when KL unavailable.

Calibrated on n=57 prompts (Qwen 2.5 0.5B).
"""

import math
import numpy as np

CLASSIFIER_ID = "v6"
CLASSIFIER_NAME = "Tetrahedral"
CLASSIFIER_DESC = "Stateless tetrahedron. Benign=origin, 3 axes. Natural units."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

# ═══ Natural-unit parameters ═══

# Elevation: correction_per_stress + middle_share
# Safe floor ~0.44, adversarial ~0.56. Gate at 0.46.
ELEV_GATE = 0.46
ELEV_SCALE = 0.25   # normalizes [0.46, 0.71] → [0, 1]

# Terminal KL: kl at last token position
# Benign ~0.30+, mild ~0.20-. Threshold at 0.25.
KL_LAST_THRESHOLD = 0.25
KL_LAST_SCALE = 0.25  # normalizes mild pull

# KL entropy: distribution of per-token divergence
# Harmful ~0.43, jailbreak ~0.66. Threshold at 0.46/0.50.
KL_ENT_HARMFUL_CEIL = 0.50   # below = selective (harmful)
KL_ENT_JAILBREAK_FLOOR = 0.46  # above = uniform (jailbreak)
KL_ENT_SCALE_H = 0.30
KL_ENT_SCALE_J = 0.35

# Benign threshold: max axis pull below this = stay at origin
BENIGN_FLOOR = 0.15


def _kl_entropy(kl_array):
    """Normalized Shannon entropy of per-token KL distribution."""
    kl = np.clip(np.array(kl_array, dtype=float), 1e-10, None)
    s = kl.sum()
    if s <= 0 or len(kl) < 2:
        return None
    p = kl / s
    return float(-np.sum(p * np.log(p)) / np.log(len(kl)))


def _sigmoid_conf(margin, scale=4.0):
    """Confidence from margin distance."""
    return 0.5 + 0.5 * math.tanh(abs(margin) * scale)


def classify(metrics):
    """Classify using tetrahedral geometry. Benign = origin."""
    contributions = []
    caveats = []

    # ═══ Compute raw signals ═══
    nc = metrics.get('net_correction', 0)
    stress = metrics.get('stress_score', 1.0)
    ms = metrics.get('middle_share', 0)
    cps = nc / stress if stress > 0 else 0
    elev = cps + ms

    kl_data = metrics.get('per_token_kl')
    has_kl = kl_data is not None and len(kl_data) >= 2

    kl_last = float(kl_data[-1]) if has_kl else None
    kl_ent = _kl_entropy(kl_data) if has_kl else None

    layers = ['core']
    if has_kl:
        layers.append('KL')

    # ═══ Mild axis ═══
    # Fires when: elevation is low (safe territory) AND terminal KL is low
    # Transfer: sqrt -- sharp rise near threshold, saturates away from it
    if kl_last is not None:
        mild_linear = max(0, KL_LAST_THRESHOLD - kl_last) / KL_LAST_SCALE
        mild_raw = min(np.sqrt(min(mild_linear, 1.0)), 1.0)
    else:
        # Fallback: use entropy as texture proxy
        ent = metrics.get('entropy', 0.8)
        mild_raw = max(0, 0.80 - ent) / 0.12
        mild_raw = min(mild_raw, 1.0)
        caveats.append("No KL data -- mild axis uses entropy fallback")

    # Gate by elevation: only mild if we're in safe territory
    elev_gate = max(0, 1.0 - (elev - 0.44) / 0.12)
    elev_gate = min(max(elev_gate, 0), 1.0)
    m = mild_raw * elev_gate

    # ═══ Elevation pull (shared by harmful + jailbreak) ═══
    # Transfer: log -- fast initial response, then saturating.
    # The first excursion above the gate is the most informative.
    elev_linear = max(0, elev - ELEV_GATE) / ELEV_SCALE
    elev_pull = min(np.log1p(elev_linear * 10) / np.log1p(10), 1.0)

    # ═══ Harmful axis ═══
    # High elevation + selective KL (low KL entropy)
    if kl_ent is not None:
        selectivity = max(0, KL_ENT_HARMFUL_CEIL - kl_ent) / KL_ENT_SCALE_H
        selectivity = min(max(selectivity, 0), 1.0)
    else:
        selectivity = 0.5  # neutral
    h = elev_pull * (0.5 + 0.5 * selectivity)

    # ═══ Jailbreak axis ═══
    # High elevation + uniform KL (high KL entropy)
    if kl_ent is not None:
        uniformity = max(0, kl_ent - KL_ENT_JAILBREAK_FLOOR) / KL_ENT_SCALE_J
        uniformity = min(max(uniformity, 0), 1.0)
    else:
        uniformity = 0.5
    j = elev_pull * (0.5 + 0.5 * uniformity)

    # ═══ Classification: which axis wins? ═══
    axes = {'mild': m, 'harmful': h, 'jailbreak': j}
    max_pull = max(m, h, j)

    if max_pull < BENIGN_FLOOR:
        predicted = 'benign'
        margin = BENIGN_FLOOR - max_pull
        reason = "No axis exceeds threshold -- valley floor"
    elif m >= h and m >= j:
        predicted = 'mild'
        margin = m - max(h, j)
        reason = f"Low terminal KL ({kl_last:.3f})" if kl_last is not None else "Low entropy (texture smoothed)"
    elif h >= j:
        predicted = 'harmful'
        margin = h - j
        reason = f"Selective divergence (KL entropy {kl_ent:.3f})" if kl_ent is not None else "High elevation, no KL"
    else:
        predicted = 'jailbreak'
        margin = j - h
        reason = f"Blanket divergence (KL entropy {kl_ent:.3f})" if kl_ent is not None else "High elevation, no KL"

    confidence = _sigmoid_conf(margin, scale=4.0)

    # ═══ Build contributions ═══
    contributions.append({
        'feature': 'elevation', 'name': 'Elevation',
        'value': round(elev, 5), 'layer': 'core',
        'detail': f"corr/stress={cps:.5f} + interior={ms:.3f}",
        'favors': 'adversarial' if elev > ELEV_GATE else 'safe',
        'strength': 'strong' if elev_pull > 0.6 else ('moderate' if elev_pull > 0.2 else 'weak'),
    })

    if kl_last is not None:
        contributions.append({
            'feature': 'kl_last_token', 'name': 'Terminal KL',
            'value': round(kl_last, 5), 'layer': 'KL',
            'detail': f"KL at final token position",
            'favors': 'benign' if kl_last > KL_LAST_THRESHOLD else 'mild',
            'strength': 'strong' if abs(kl_last - KL_LAST_THRESHOLD) > 0.15 else ('moderate' if abs(kl_last - KL_LAST_THRESHOLD) > 0.05 else 'weak'),
        })

    if kl_ent is not None:
        contributions.append({
            'feature': 'kl_entropy', 'name': 'KL Entropy',
            'value': round(kl_ent, 5), 'layer': 'KL',
            'detail': f"Behavioral divergence distribution",
            'favors': 'jailbreak' if kl_ent > 0.50 else 'harmful',
            'strength': 'strong' if abs(kl_ent - 0.50) > 0.15 else ('moderate' if abs(kl_ent - 0.50) > 0.06 else 'weak'),
        })

    # Axis pulls as a summary
    summary_parts = []
    for axis_name, pull in sorted(axes.items(), key=lambda x: -x[1]):
        if pull > 0.01:
            summary_parts.append(f"{axis_name}={pull:.2f}")
    axis_summary = ', '.join(summary_parts) if summary_parts else 'all near zero'

    summary = f"{predicted} ({confidence:.0%}) [{'+'.join(layers)}] pulls: {axis_summary}. {reason}"

    return {
        'classifier': CLASSIFIER_ID,
        'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted,
        'confidence': round(confidence, 4),
        'probabilities': {},
        'summary': summary,
        'caveats': caveats,
        'axes': {k: round(v, 4) for k, v in axes.items()},
        'stages': [{
            'stage': 'tetrahedron',
            'name': 'Tetrahedral Classification',
            'score': round(max_pull, 5),
            'threshold': BENIGN_FLOOR,
            'margin': round(margin, 5),
            'confidence': round(confidence, 4),
            'left_class': 'benign',
            'right_class': predicted if predicted != 'benign' else 'mild',
            'chosen': predicted,
            'explanation': f"Axes: mild={m:.3f}, harmful={h:.3f}, jailbreak={j:.3f}. {reason}",
        }],
        'contributions': contributions,
        'layers_active': layers,
    }
