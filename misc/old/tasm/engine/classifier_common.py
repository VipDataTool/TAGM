"""
Shared classifier infrastructure for the geometric classifier variants (v4, v5, v6, v8).

Implements the corrective engineering recommendations from the calibration analysis:
  5.1 — Standardized binary stage (shared across all classifiers)
  5.3 — Harmonized confidence function
  5.4 — Honest benign/mild handling

All classifiers import from here to ensure they share the same
decision boundary, confidence model, and benign/mild caveats.
"""

import math

# ═══ 5.1: Standardized Binary Stage ═══
#
# middle_share + weighted (net_correction - baseline) is the strongest
# binary signal. Net_correction has a floor around 0.070 on Qwen 0.5B
# (minimum correction any prompt generates). Excess correction above
# this floor amplifies the interior-distribution signal.
#
# Threshold 0.465 separates safe from adversarial at the optimal
# Youden's J point on n=56 balanced prompts.
#
# Known limitation: middle_share increases with token count. Prompts
# longer than ~35 tokens may cross the threshold purely from length.
# The classifiers flag this with a caveat rather than silently
# overcorrecting with a model-specific length regression.

BINARY_NC_FLOOR = 0.070       # minimum correction any prompt generates
BINARY_NC_WEIGHT = 0.8        # how much excess correction amplifies
BINARY_THRESHOLD = 0.465      # optimal split on n=56

# Calibration range: the thresholds were set on prompts with 5-31 tokens.
# Outside this range, the binary score is less reliable.
CALIBRATION_SEQ_MIN = 5
CALIBRATION_SEQ_MAX = 35


def binary_score(middle_share, net_correction):
    """Compute the standardized binary score (safe vs adversarial).
    Higher = more adversarial. Threshold = BINARY_THRESHOLD."""
    return middle_share + BINARY_NC_WEIGHT * (net_correction - BINARY_NC_FLOOR)


def binary_classify(middle_share, net_correction, seq_len=None):
    """Run the standardized binary stage.
    Returns (is_adversarial, score, confidence, caveats)."""
    score = binary_score(middle_share, net_correction)
    is_adversarial = score > BINARY_THRESHOLD
    conf = sigmoid_confidence(abs(score - BINARY_THRESHOLD), scale=5.0)

    caveats = []
    if abs(score - BINARY_THRESHOLD) < 0.01:
        caveats.append("Very close to the safe/adversarial boundary")
    if seq_len is not None and seq_len > CALIBRATION_SEQ_MAX:
        caveats.append(
            f"Prompt is {seq_len} tokens (calibration range: {CALIBRATION_SEQ_MIN}-"
            f"{CALIBRATION_SEQ_MAX}). Interior share increases with length — "
            f"this may be a long benign prompt rather than an adversarial one."
        )

    return is_adversarial, score, conf, caveats


def binary_stage_dict(score, is_adversarial, confidence, middle_share, net_correction):
    """Build the standard stage dict for the binary split."""
    chosen = 'adversarial' if is_adversarial else 'safe'
    return {
        'stage': 'binary', 'name': 'Interior Distribution (shared)',
        'score': round(score, 6), 'threshold': BINARY_THRESHOLD,
        'margin': round(score - BINARY_THRESHOLD, 6),
        'confidence': round(confidence, 4),
        'left_class': 'safe', 'right_class': 'adversarial',
        'chosen': chosen,
        'explanation': (
            f"ms({middle_share:.3f}) + 0.8×(nc({net_correction:.5f}) - 0.070) "
            f"= {score:.4f} {'>' if is_adversarial else '≤'} {BINARY_THRESHOLD} → {chosen}"
        ),
    }


def binary_contributions(middle_share, net_correction):
    """Standard contributions for the binary stage."""
    return [
        {
            'feature': 'middle_share', 'name': 'Interior Share',
            'value': round(middle_share, 4),
            'favors': 'adversarial' if middle_share > 0.5 else 'safe',
            'strength': ('strong' if abs(middle_share - 0.5) > 0.08
                         else ('moderate' if abs(middle_share - 0.5) > 0.03 else 'weak')),
        },
        {
            'feature': 'net_correction', 'name': 'Net Correction',
            'value': round(net_correction, 5),
            'favors': 'adversarial' if net_correction > 0.075 else 'safe',
            'strength': ('moderate' if abs(net_correction - 0.075) > 0.005 else 'weak'),
        },
    ]


# ═══ 5.3: Harmonized Confidence ═══
#
# All classifiers use the same confidence function: tanh sigmoid.
# Scale=5.0 for binary stage, scale=4.0 for sub-stages.
# This ensures confidence values are comparable across classifiers.

BINARY_CONF_SCALE = 5.0
SUB_CONF_SCALE = 4.0


def sigmoid_confidence(margin, scale=None):
    """Confidence from margin distance. Consistent across all classifiers.
    scale defaults to SUB_CONF_SCALE (4.0) for sub-stages."""
    if scale is None:
        scale = SUB_CONF_SCALE
    return 0.5 + 0.5 * math.tanh(abs(margin) * scale)


# ═══ 5.4: Honest Benign/Mild Handling ═══
#
# At 0.5B model scale, the benign/mild boundary is not reliably
# separable. Entropy and interior_cv distributions overlap completely.
# Rather than pretending to classify with confidence, we:
#   1. Cap sub-confidence at SAFE_SUB_CONF_CAP
#   2. Add an explicit caveat
#   3. Still make a prediction (it's useful directionally, just not reliable)

SAFE_SUB_CONF_CAP = 0.70


def safe_sub_classify(entropy, interior_cv):
    """Classify benign vs mild with honest confidence capping.
    Returns (predicted, score, confidence, caveat_text)."""
    score = entropy + 0.3 * interior_cv
    threshold = 1.0
    predicted = 'benign' if score > threshold else 'mild'
    raw_conf = sigmoid_confidence(abs(score - threshold))
    conf = min(raw_conf, SAFE_SUB_CONF_CAP)

    caveat = (
        "Benign/mild distinction is unreliable at 0.5B model scale — "
        "entropy and interior CV overlap completely between categories. "
        f"Confidence capped at {SAFE_SUB_CONF_CAP:.0%}."
    )
    return predicted, score, conf, caveat


def safe_sub_stage_dict(predicted, score, confidence, entropy, interior_cv):
    """Build the standard stage dict for the safe sub-classification."""
    threshold = 1.0
    return {
        'stage': 'safe_sub', 'name': 'Entropy + CV (low confidence)',
        'score': round(score, 6), 'threshold': threshold,
        'margin': round(score - threshold, 6),
        'confidence': round(confidence, 4),
        'left_class': 'mild', 'right_class': 'benign',
        'chosen': predicted,
        'explanation': (
            f"ent({entropy:.3f}) + 0.3×cv({interior_cv:.3f}) = {score:.4f} "
            f"{'>' if predicted == 'benign' else '≤'} {threshold} → {predicted} "
            f"(capped at {SAFE_SUB_CONF_CAP:.0%})"
        ),
    }


def safe_sub_contributions(entropy, interior_cv):
    """Standard contributions for the safe sub-stage."""
    return [
        {
            'feature': 'entropy', 'name': 'Entropy',
            'value': round(entropy, 4),
            'favors': 'benign' if entropy > 0.80 else 'mild',
            'strength': 'weak',
        },
        {
            'feature': 'interior_cv', 'name': 'Interior CV',
            'value': round(interior_cv, 4),
            'favors': 'benign' if interior_cv > 0.65 else 'mild',
            'strength': 'weak',
        },
    ]
