"""
ASM Classifier v2: Hierarchical 4-class prompt classification.

Architecture: 2-stage decision tree with Gaussian confidence.

  Stage 1: BINARY — safe (benign+mild) vs adversarial (harmful+jailbreak)
    Score: net_correction - 0.003 * stress_score
    Threshold: 0.064
    Accuracy: 87% on n=100

  Stage 2a: SAFE sub-split — benign vs mild
    Score: entropy + 0.5 * interior_cv
    Threshold: 1.056
    Accuracy: 72% on n=100 (honest: this boundary is weak at 0.5B)

  Stage 2b: ADVERSARIAL sub-split — harmful vs jailbreak
    Score: stress_score + 0.1 * interior_cv
    Threshold: 3.455
    Accuracy: 86% on n=100

Calibrated on n=100 balanced prompts (Qwen 2.5 0.5B).
"""

import math
from dataclasses import dataclass


@dataclass
class StageResult:
    """Result from one decision stage."""
    stage: str
    score: float
    threshold: float
    margin: float
    confidence: float
    left_class: str
    right_class: str
    chosen: str
    explanation: str


@dataclass
class FeatureContribution:
    """Per-feature explanation."""
    feature: str
    name: str
    value: float
    favors: str
    strength: str
    explanation: str
    scores: dict


@dataclass
class ClassificationResult:
    """Complete hierarchical classification."""
    predicted: str
    confidence: float
    probabilities: dict
    stages: list
    summary: str
    caveats: list
    contributions: list


# ═══ Stage definitions (calibrated on n=100) ═══

BINARY_WEIGHTS = {'net_correction': 1.0, 'stress_score': -0.003}
BINARY_THRESHOLD = 0.064
BINARY_SCORE_STATS = {
    'safe':        (0.0607, 0.0025),
    'adversarial': (0.0696, 0.0047),
}

SAFE_WEIGHTS = {'entropy': 1.0, 'interior_cv': 0.5}
SAFE_THRESHOLD = 1.056
SAFE_SCORE_STATS = {
    'mild':   (1.052, 0.055),
    'benign': (1.113, 0.077),
}

ADV_WEIGHTS = {'stress_score': 1.0, 'interior_cv': 0.1}
ADV_THRESHOLD = 3.455
ADV_SCORE_STATS = {
    'harmful':   (3.327, 0.087),
    'jailbreak': (3.534, 0.117),
}

FEATURE_META = {
    'net_correction':  ('Net Correction',  'alignment correction magnitude'),
    'middle_share':    ('Interior Share',   'fraction on interior tokens'),
    'stress_score':    ('Stress Score',     'correction pressure at mid layers'),
    'entropy':         ('Entropy',          'correction distribution uniformity'),
    'interior_cv':     ('Interior CV',      'interior token concentration'),
}

FEATURES = ['net_correction', 'middle_share', 'stress_score', 'entropy', 'interior_cv']
CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

CLASS_PARAMS = {
    'benign': {
        'net_correction': (0.07143, 0.00315),
        'middle_share':   (0.40417, 0.07035),
        'stress_score':   (3.19678, 0.11754),
        'entropy':        (0.79168, 0.03887),
        'interior_cv':    (0.64226, 0.17614),
    },
    'mild': {
        'net_correction': (0.07077, 0.00226),
        'middle_share':   (0.40128, 0.08137),
        'stress_score':   (3.23068, 0.09752),
        'entropy':        (0.76768, 0.04575),
        'interior_cv':    (0.56812, 0.16832),
    },
    'harmful': {
        'net_correction': (0.07674, 0.00388),
        'middle_share':   (0.48606, 0.08752),
        'stress_score':   (3.26334, 0.08571),
        'entropy':        (0.81512, 0.03732),
        'interior_cv':    (0.63225, 0.15201),
    },
    'jailbreak': {
        'net_correction': (0.08102, 0.00590),
        'middle_share':   (0.56594, 0.07811),
        'stress_score':   (3.43222, 0.11762),
        'entropy':        (0.79790, 0.05785),
        'interior_cv':    (1.02096, 0.40240),
    },
}


def _log_gaussian(x, mu, sigma):
    if sigma <= 0:
        sigma = 1e-6
    return -0.5 * math.log(2 * math.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2


def _gaussian_confidence(score, threshold, stats_below, stats_above):
    """Confidence from Gaussian overlap at a decision boundary."""
    mu_lo, sigma_lo = stats_below
    mu_hi, sigma_hi = stats_above
    ll_lo = _log_gaussian(score, mu_lo, sigma_lo)
    ll_hi = _log_gaussian(score, mu_hi, sigma_hi)
    max_ll = max(ll_lo, ll_hi)
    p_lo = math.exp(ll_lo - max_ll)
    p_hi = math.exp(ll_hi - max_ll)
    total = p_lo + p_hi
    if score > threshold:
        return max(0.5, min(1.0, p_hi / total))
    else:
        return max(0.5, min(1.0, p_lo / total))


def _compute_score(metrics, weights):
    return sum(metrics.get(f, 0) * w for f, w in weights.items())


def classify(metrics, **kwargs):
    """Classify a prompt using the hierarchical decision tree."""
    stages = []
    caveats = []

    # ═══ Stage 1: Binary ═══
    s1_score = _compute_score(metrics, BINARY_WEIGHTS)
    s1_margin = s1_score - BINARY_THRESHOLD
    s1_conf = _gaussian_confidence(s1_score, BINARY_THRESHOLD,
                                    BINARY_SCORE_STATS['safe'],
                                    BINARY_SCORE_STATS['adversarial'])
    is_adversarial = s1_score > BINARY_THRESHOLD
    s1_chosen = 'adversarial' if is_adversarial else 'safe'

    stages.append(StageResult(
        stage='binary',
        score=s1_score,
        threshold=BINARY_THRESHOLD,
        margin=s1_margin,
        confidence=s1_conf,
        left_class='safe',
        right_class='adversarial',
        chosen=s1_chosen,
        explanation=(f"Binary score = {s1_score:.5f} "
                     f"({'>' if is_adversarial else '≤'} {BINARY_THRESHOLD}) "
                     f"→ {s1_chosen} ({s1_conf:.0%})"),
    ))

    if abs(s1_margin) < 0.002:
        caveats.append("Very close to the safe/adversarial boundary")

    if is_adversarial:
        # ═══ Stage 2b: harmful vs jailbreak ═══
        s2_score = _compute_score(metrics, ADV_WEIGHTS)
        s2_margin = s2_score - ADV_THRESHOLD
        s2_conf = _gaussian_confidence(s2_score, ADV_THRESHOLD,
                                        ADV_SCORE_STATS['harmful'],
                                        ADV_SCORE_STATS['jailbreak'])
        predicted = 'jailbreak' if s2_score > ADV_THRESHOLD else 'harmful'

        stages.append(StageResult(
            stage='adversarial_sub',
            score=s2_score,
            threshold=ADV_THRESHOLD,
            margin=s2_margin,
            confidence=s2_conf,
            left_class='harmful',
            right_class='jailbreak',
            chosen=predicted,
            explanation=(f"Adversarial score = {s2_score:.4f} "
                         f"({'>' if predicted == 'jailbreak' else '≤'} {ADV_THRESHOLD}) "
                         f"→ {predicted} ({s2_conf:.0%})"),
        ))

        if abs(s2_margin) < 0.05:
            caveats.append("Close to the harmful/jailbreak boundary — may use partial adversarial framing")
    else:
        # ═══ Stage 2a: benign vs mild ═══
        s2_score = _compute_score(metrics, SAFE_WEIGHTS)
        s2_margin = s2_score - SAFE_THRESHOLD
        s2_conf = _gaussian_confidence(s2_score, SAFE_THRESHOLD,
                                        SAFE_SCORE_STATS['mild'],
                                        SAFE_SCORE_STATS['benign'])
        predicted = 'benign' if s2_score > SAFE_THRESHOLD else 'mild'

        stages.append(StageResult(
            stage='safe_sub',
            score=s2_score,
            threshold=SAFE_THRESHOLD,
            margin=s2_margin,
            confidence=s2_conf,
            left_class='mild',
            right_class='benign',
            chosen=predicted,
            explanation=(f"Safe score = {s2_score:.4f} "
                         f"({'>' if predicted == 'benign' else '≤'} {SAFE_THRESHOLD}) "
                         f"→ {predicted} ({s2_conf:.0%})"),
        ))

        caveats.append("Benign/mild distinction is weak at 0.5B — treat as low-confidence")

    overall_conf = s1_conf * stages[-1].confidence

    # 4-class Gaussian probabilities (informational)
    class_scores = {}
    for cls in CLASSES:
        ll = 0.0
        for feat in FEATURES:
            val = metrics.get(feat)
            if val is not None and feat in CLASS_PARAMS[cls]:
                mu, sigma = CLASS_PARAMS[cls][feat]
                ll += _log_gaussian(val, mu, sigma)
        class_scores[cls] = ll
    max_ll = max(class_scores.values())
    exp_s = {k: math.exp(v - max_ll) for k, v in class_scores.items()}
    total = sum(exp_s.values())
    probabilities = {k: v / total for k, v in exp_s.items()}

    if probabilities[predicted] < 0.3:
        gauss_pred = max(probabilities, key=probabilities.get)
        if gauss_pred != predicted:
            caveats.append(f"Gaussian probabilities favor {gauss_pred} but tree selects {predicted}")

    contributions = _build_contributions(metrics, predicted)
    stage_names = ' → '.join(s.chosen for s in stages)
    summary = f"Classified as {predicted} via {stage_names} (overall {overall_conf:.0%})"

    return ClassificationResult(
        predicted=predicted,
        confidence=overall_conf,
        probabilities=probabilities,
        stages=stages,
        summary=summary,
        caveats=caveats,
        contributions=contributions,
    )


def _build_contributions(metrics, predicted):
    contributions = []
    for feat in FEATURES:
        val = metrics.get(feat)
        if val is None:
            continue
        name, desc = FEATURE_META.get(feat, (feat, ''))
        best_cls = None
        best_z = float('inf')
        z_scores = {}
        for cls in CLASSES:
            if feat in CLASS_PARAMS[cls]:
                mu, sigma = CLASS_PARAMS[cls][feat]
                z = abs(val - mu) / sigma if sigma > 0 else 0
                z_scores[cls] = z
                if z < best_z:
                    best_z = z
                    best_cls = cls
        if z_scores:
            sorted_z = sorted(z_scores.items(), key=lambda x: x[1])
            gap = sorted_z[1][1] - sorted_z[0][1] if len(sorted_z) > 1 else 0
            strength = 'strong' if gap > 1.5 else ('moderate' if gap > 0.5 else 'weak')
        else:
            strength = 'weak'
        if strength == 'strong':
            explanation = f"{name} = {val:.5f} → strongly favors {best_cls} ({best_z:.1f}σ)"
        elif strength == 'moderate':
            explanation = f"{name} = {val:.5f} → leans {best_cls} ({best_z:.1f}σ)"
        else:
            explanation = f"{name} = {val:.5f} → ambiguous (within range of multiple classes)"
        feat_scores = {}
        for cls in CLASSES:
            if feat in CLASS_PARAMS[cls]:
                mu, sigma = CLASS_PARAMS[cls][feat]
                feat_scores[cls] = round(_log_gaussian(val, mu, sigma), 3)
        contributions.append(FeatureContribution(
            feature=feat, name=name, value=round(val, 6),
            favors=best_cls or predicted, strength=strength,
            explanation=explanation, scores=feat_scores,
        ))
    return contributions


def classify_to_dict(metrics):
    """Classify and return a JSON-serializable dict."""
    result = classify(metrics)
    return {
        'predicted': result.predicted,
        'confidence': round(result.confidence, 4),
        'probabilities': {k: round(v, 4) for k, v in result.probabilities.items()},
        'summary': result.summary,
        'caveats': result.caveats,
        'stages': [
            {
                'stage': s.stage,
                'score': round(s.score, 6),
                'threshold': s.threshold,
                'margin': round(s.margin, 6),
                'confidence': round(s.confidence, 4),
                'left_class': s.left_class,
                'right_class': s.right_class,
                'chosen': s.chosen,
                'explanation': s.explanation,
            }
            for s in result.stages
        ],
        'contributions': [
            {
                'feature': c.feature,
                'name': c.name,
                'value': c.value,
                'favors': c.favors,
                'strength': c.strength,
                'explanation': c.explanation,
                'scores': c.scores,
            }
            for c in result.contributions
        ],
    }


def update_params(results):
    """Recalibrate class parameters from labeled results."""
    import statistics as st
    cats = {}
    for r in results:
        cat = r.get('category', '')
        if cat in CLASSES:
            cats.setdefault(cat, []).append(r)
    new_params = {}
    for cls in CLASSES:
        new_params[cls] = {}
        items = cats.get(cls, [])
        for feat in FEATURES:
            vals = [r[feat] for r in items if r.get(feat) is not None]
            if len(vals) >= 2:
                new_params[cls][feat] = (sum(vals)/len(vals), max(st.stdev(vals), 1e-6))
            else:
                new_params[cls][feat] = CLASS_PARAMS[cls][feat]
    return new_params
