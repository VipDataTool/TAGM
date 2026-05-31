"""
ASM Classifier v7: Nearest Centroid.

Architecture: Thresholdless nearest-centroid in a 14-dimensional
z-scored feature space (7 raw + 7 self-normalizing ratios).

  Raw features:
    net_correction, middle_share, mean_stress, entropy,
    kl_divergence, mean_kl, interior_cv

  Ratio features (self-normalizing -- resilient to uniform drift):
    middle_share / interior_cv
    net_correction / entropy
    kl_divergence / net_correction
    mean_stress * net_correction
    entropy / middle_share
    interior_cv * middle_share
    net_correction * middle_share

Classification:
  1. Extract 14 features from prompt metrics.
  2. Z-score each feature against population statistics.
  3. Compute Euclidean distance to each class centroid.
  4. Predict the nearest centroid's class.
  5. Confidence from softmax over negative squared distances.

No thresholds, no dead zones, no gating. Every point in the space
has a nearest vertex. Points equidistant from multiple centroids get
low confidence (the honest answer). Points deep inside one cluster
get high confidence.

State: 14 population means, 14 population stds, 4x14 centroid means.
All derived deterministically from labeled data via update_params().

NOTE: This is a population-referenced classifier. The centroids and
z-scoring stats are derived from a calibration dataset. Performance
may degrade if the input distribution shifts significantly.

Calibrated on n=65 balanced prompts (Qwen 2.5 0.5B).
LOO-validated: 87% binary, 52% four-class.
Naive (in-sample): 87% binary, 64% four-class.
"""

import math

CLASSIFIER_ID = "v7"
CLASSIFIER_NAME = "Nearest Centroid"
CLASSIFIER_DESC = "Thresholdless nearest centroid. 14 features (7 raw + 7 ratio). Z-scored."

CLASSES = ['benign', 'mild', 'harmful', 'jailbreak']

FEATURES = [
    'net_correction', 'middle_share', 'mean_stress', 'entropy',
    'kl_divergence', 'mean_kl', 'interior_cv',
    'r_ms_over_ic', 'r_nc_over_ent', 'r_kl_over_nc',
    'r_stress_x_nc', 'r_ent_over_ms', 'r_ic_x_ms', 'r_nc_x_ms',
]

FEATURE_META = {
    'net_correction': ('Net Correction', 'alignment correction magnitude'),
    'middle_share':   ('Interior Share', 'fraction on interior tokens'),
    'mean_stress':    ('Mean Stress', 'per-token correction pressure'),
    'entropy':        ('Entropy', 'correction distribution uniformity'),
    'kl_divergence':  ('KL Divergence', 'instruct-base behavioral gap'),
    'mean_kl':        ('Mean KL', 'per-token behavioral divergence'),
    'interior_cv':    ('Interior CV', 'interior token concentration'),
    'r_ms_over_ic':   ('Share/CV', 'interior share / concentration'),
    'r_nc_over_ent':  ('Corr/Ent', 'correction / entropy'),
    'r_kl_over_nc':   ('KL/Corr', 'divergence / correction'),
    'r_stress_x_nc':  ('Stress*Corr', 'stress-correction interaction'),
    'r_ent_over_ms':  ('Ent/Share', 'entropy / interior share'),
    'r_ic_x_ms':      ('CV*Share', 'concentration-share interaction'),
    'r_nc_x_ms':      ('Corr*Share', 'correction-share interaction'),
}

# === Population statistics (for z-scoring) ===
# Derived from n=65 balanced dataset via update_params()

POPULATION_STATS = {
    'net_correction': (0.07505821814903846, 0.005861739146296918),
    'middle_share': (0.46496769831730766, 0.10300831481676694),
    'mean_stress': (3.2764190323152618, 0.14823956309668124),
    'entropy': (0.79753730495369, 0.052676604376583135),
    'kl_divergence': (0.30780310997596155, 0.20289211392050527),
    'mean_kl': (0.4196310300360155, 0.2628413198453494),
    'interior_cv': (0.6813626802884616, 0.22881616846443356),
    'r_ms_over_ic': (0.730768248511019, 0.20145415130385685),
    'r_nc_over_ent': (0.0942780033854847, 0.0066809274374212526),
    'r_kl_over_nc': (4.0441157532076115, 2.462083896744308),
    'r_stress_x_nc': (0.24653495232974576, 0.028947466000716054),
    'r_ent_over_ms': (1.7914813376562828, 0.37903347874109405),
    'r_ic_x_ms': (0.3300474579517658, 0.16338585058812857),
    'r_nc_x_ms': (0.035388238384173466, 0.010396560044278535),
}

# === Class centroids (in raw feature space) ===
# Z-scoring applied at classify time using POPULATION_STATS

CLASS_CENTROIDS = {
    'benign': {
        'net_correction': 0.07167816162109375, 'middle_share': 0.4187469482421875,
        'mean_stress': 3.207218943521652, 'entropy': 0.7889584716242485,
        'kl_divergence': 0.2833671569824219, 'mean_kl': 0.3402241074417885,
        'interior_cv': 0.6484375, 'r_ms_over_ic': 0.6887003291271722,
        'r_nc_over_ent': 0.09108658237725395, 'r_kl_over_nc': 3.9265642116794024,
        'r_stress_x_nc': 0.23008236705322865, 'r_ent_over_ms': 1.941541723472844,
        'r_ic_x_ms': 0.2771727368235588, 'r_nc_x_ms': 0.030122355557978153,
    },
    'mild': {
        'net_correction': 0.07064280790441177, 'middle_share': 0.39040958180147056,
        'mean_stress': 3.216509971609119, 'entropy': 0.763106550183349,
        'kl_divergence': 0.20491656135110295, 'mean_kl': 0.3359482578350987,
        'interior_cv': 0.5967945772058824, 'r_ms_over_ic': 0.6923950720460877,
        'r_nc_over_ent': 0.09295154306766622, 'r_kl_over_nc': 2.9069735451390897,
        'r_stress_x_nc': 0.2272970745164904, 'r_ent_over_ms': 2.0103027391422006,
        'r_ic_x_ms': 0.24088249136419856, 'r_nc_x_ms': 0.02771459081593682,
    },
    'harmful': {
        'net_correction': 0.0767669677734375, 'middle_share': 0.4775543212890625,
        'mean_stress': 3.2532283842237906, 'entropy': 0.820332545939644,
        'kl_divergence': 0.333038330078125, 'mean_kl': 0.5408733363952664,
        'interior_cv': 0.591217041015625, 'r_ms_over_ic': 0.8511859493410143,
        'r_nc_over_ent': 0.09364988936446707, 'r_kl_over_nc': 4.355007198667075,
        'r_stress_x_nc': 0.249915316644478, 'r_ent_over_ms': 1.7688487365165473,
        'r_ic_x_ms': 0.29131147265434265, 'r_nc_x_ms': 0.03692319989204407,
    },
    'jailbreak': {
        'net_correction': 0.0814208984375, 'middle_share': 0.57781982421875,
        'mean_stress': 3.432463146200619, 'entropy': 0.8199035742406648,
        'kl_divergence': 0.41632080078125, 'mean_kl': 0.46670859173446566,
        'interior_cv': 0.894287109375, 'r_ms_over_ic': 0.6931899670588598,
        'r_nc_over_ent': 0.09950690250241523, 'r_kl_over_nc': 5.058989445349163,
        'r_stress_x_nc': 0.28004741846811454, 'r_ent_over_ms': 1.4315558139006697,
        'r_ic_x_ms': 0.5163959413766861, 'r_nc_x_ms': 0.04727241024374962,
    },
}


def _extract_features(metrics):
    """Extract the 14 features from a metrics dict."""
    nc = metrics.get('net_correction', 0)
    ms = metrics.get('middle_share', 0)
    ic = metrics.get('interior_cv', 0)
    ent = metrics.get('entropy', 0)
    kl = metrics.get('kl_divergence', 0)

    pts = metrics.get('per_token_stress')
    if pts and len(pts) > 0:
        stress = sum(pts) / len(pts)
    else:
        stress = metrics.get('stress_score', 0) or metrics.get('mean_stress', 0)

    ptk = metrics.get('per_token_kl')
    if ptk and len(ptk) > 0:
        mkl = sum(ptk) / len(ptk)
    else:
        mkl = metrics.get('mean_kl', kl)

    return {
        'net_correction': nc, 'middle_share': ms, 'mean_stress': stress,
        'entropy': ent, 'kl_divergence': kl, 'mean_kl': mkl, 'interior_cv': ic,
        'r_ms_over_ic': ms / ic if ic > 0 else 0,
        'r_nc_over_ent': nc / ent if ent > 0 else 0,
        'r_kl_over_nc': kl / nc if nc > 0 else 0,
        'r_stress_x_nc': stress * nc,
        'r_ent_over_ms': ent / ms if ms > 0 else 0,
        'r_ic_x_ms': ic * ms,
        'r_nc_x_ms': nc * ms,
    }


def _z_score(features, stats=None):
    if stats is None:
        stats = POPULATION_STATS
    return {
        k: (features[k] - stats[k][0]) / stats[k][1] if stats[k][1] > 0 else 0
        for k in FEATURES
    }


def _euclidean(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in FEATURES))


def _softmax_distances(dists):
    logits = {k: -(v ** 2) for k, v in dists.items()}
    max_l = max(logits.values())
    exps = {k: math.exp(v - max_l) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def classify(metrics):
    """Classify a prompt by nearest centroid in z-scored feature space."""
    raw = _extract_features(metrics)
    z_input = _z_score(raw)
    z_centroids = {cat: _z_score(c) for cat, c in CLASS_CENTROIDS.items()}

    dists = {cat: _euclidean(z_input, z_centroids[cat]) for cat in CLASSES}
    predicted = min(dists, key=dists.get)
    probabilities = _softmax_distances(dists)
    confidence = probabilities[predicted]

    sorted_cats = sorted(dists, key=dists.get)
    margin = dists[sorted_cats[1]] - dists[sorted_cats[0]] if len(sorted_cats) > 1 else 0

    safe_dist = min(dists['benign'], dists['mild'])
    adv_dist = min(dists['harmful'], dists['jailbreak'])
    binary = 'safe' if safe_dist <= adv_dist else 'adversarial'

    caveats = []
    if margin < 0.3:
        caveats.append(f"Close to {sorted_cats[1]} centroid (margin={margin:.2f})")
    if predicted in ('benign', 'mild'):
        if abs(dists['benign'] - dists['mild']) < 0.5:
            caveats.append("Benign/mild boundary is weak — treat sub-class as low-confidence")

    contributions = _build_contributions(z_input, z_centroids, predicted)
    dist_parts = [f"{c}={dists[c]:.2f}" for c in sorted_cats]
    runner = sorted_cats[1] if len(sorted_cats) > 1 else predicted

    summary = (f"{predicted} ({confidence:.0%}) via nearest centroid. "
               f"Binary: {binary}. Distances: {', '.join(dist_parts)}")

    stages = [{
        'stage': 'centroid', 'name': 'Nearest Centroid',
        'score': round(dists[predicted], 5), 'threshold': 0,
        'margin': round(margin, 5), 'confidence': round(confidence, 4),
        'left_class': predicted, 'right_class': runner, 'chosen': predicted,
        'explanation': (f"Nearest: {predicted} (d={dists[predicted]:.3f}). "
                        f"Runner-up: {runner} (d={dists[runner]:.3f}). "
                        f"Binary: {binary}"),
    }]

    return {
        'classifier': CLASSIFIER_ID, 'classifier_name': CLASSIFIER_NAME,
        'predicted': predicted, 'confidence': round(confidence, 4),
        'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
        'summary': summary, 'caveats': caveats, 'stages': stages,
        'contributions': contributions,
        'distances': {k: round(v, 4) for k, v in dists.items()},
        'binary': binary, 'features_used': len(FEATURES),
    }


def _build_contributions(z_input, z_centroids, predicted):
    contributions = []
    for feat in FEATURES:
        z_val = z_input[feat]
        feat_dists = {cat: abs(z_val - z_centroids[cat][feat]) for cat in CLASSES}
        closest = min(feat_dists, key=feat_dists.get)
        gap = sorted(feat_dists.values())
        feat_margin = gap[1] - gap[0] if len(gap) > 1 else 0
        strength = 'strong' if feat_margin > 1.0 else ('moderate' if feat_margin > 0.4 else 'weak')
        name, desc = FEATURE_META.get(feat, (feat, ''))
        contributions.append({
            'feature': feat, 'name': name,
            'value': round(z_val, 4),
            'raw_value': round((z_val * POPULATION_STATS[feat][1]) + POPULATION_STATS[feat][0], 6),
            'favors': closest, 'strength': strength,
            'explanation': f"{name} (z={z_val:+.2f}) -> {closest} ({strength})",
        })
    return contributions


def update_params(results):
    """Recalibrate population stats and centroids from labeled data."""
    import statistics as st
    global POPULATION_STATS, CLASS_CENTROIDS

    extracted = []
    for r in results:
        cat = r.get('category', '')
        if cat in CLASSES:
            extracted.append((_extract_features(r), cat))

    if len(extracted) < 8:
        return POPULATION_STATS, CLASS_CENTROIDS

    new_stats = {}
    for feat in FEATURES:
        vals = [f[feat] for f, _ in extracted]
        mu = sum(vals) / len(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 1e-6
        new_stats[feat] = (mu, max(sd, 1e-6))

    new_centroids = {}
    for cat in CLASSES:
        cat_feats = [f for f, c in extracted if c == cat]
        if not cat_feats:
            new_centroids[cat] = CLASS_CENTROIDS[cat]
            continue
        new_centroids[cat] = {}
        for feat in FEATURES:
            vals = [f[feat] for f in cat_feats]
            new_centroids[cat][feat] = sum(vals) / len(vals)

    POPULATION_STATS = new_stats
    CLASS_CENTROIDS = new_centroids
    return new_stats, new_centroids
