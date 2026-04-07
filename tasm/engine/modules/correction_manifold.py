"""
Correction Manifold Module for TASM — The Witness Plate.

Constructs a manifold from correction field measurements and probe
geometry using anchor-repulsor design, enabling spatial
characterization of prompts by their alignment signature.

The manifold combines:
  1. Subject domain angle (from domain surface probe proximity)
  2. Probe escalation level (ring, from domain surface probe proximity)
  3. RD replacement ratio (radial position within ring)
  4-5. Signal-driven repulsion from anchor points in locally-oriented
       coordinate frames: Entropy/SFD_e (radial), KL/ASM (tangential)

Each probe anchor (subject × level intersection) pushes tokens outward
via its own rotated coordinate frame. "North" = radially outward from
center. Clusters self-organize from signal patterns without global
attractor geometry.

Pure post-processor: requires session results and a completed
domain_surface module run.

Original concept: Ostrander (2026).
"""

import os
import json
import math
import logging
import numpy as np
from collections import Counter, defaultdict

from .base import TASMModule, ModuleParameter
from .domain_surface import _load_probe_cache, _probe_cache_path, _load_probes

logger = logging.getLogger("tasm")

# The four independent signals (|r| < 0.40 between all pairs)
# Selected from the full set of 8 by correlation analysis
SIGNAL_KEYS = [
    ("entropy",             "Entropy"),      # prompt radial repulsor
    ("kl_divergence",       "KL"),           # prompt tangential repulsor
    ("rd_mean_replacement", "RD_repl"),      # radial position within ring
    ("sfd_energy_mean",     "SFD_e"),        # token radial repulsor (proxy for Entropy)
]

# Full set of 8 signals for reporting
ALL_SIGNAL_KEYS = [
    ("stress_score",        "ASM"),
    ("entropy",             "Entropy"),
    ("gini",                "Gini"),
    ("interior_cv",         "IntCV"),
    ("sfd_energy_mean",     "SFD_e"),
    ("sfd_density_mean",    "SFD_d"),
    ("rd_mean_replacement", "RD_repl"),
    ("rd_mean_overlap",     "RD_ovlp"),
]


# ─── Probe-Level Computation ─────────────────────────────────

def _compute_prompt_probe_stats(domain_surface_data, session_results=None,
                                 probe_embs=None, probes_meta=None,
                                 knn_probes=5):
    """Position prompts on the manifold by kNN against probe embeddings.

    Primary path: each prompt's domain_embedding is compared directly
    against probe embeddings via cosine similarity. The k nearest probes
    define the prompt's angular position (weighted centroid) and level.

    Fallback: if prompt embeddings or probe cache unavailable, falls back
    to token-vote from domain surface observations.

    Returns:
        mean_level: array of mean probe escalation level per prompt
        blended_angle: array of blended subject angle per prompt
        dom_subject: array of dominant subject index per prompt
        n_prompts: number of prompts
        nearest_probes: list of k nearest probe indices per prompt (or None)
    """
    subjects = domain_surface_data.get("subjects", [])
    n_subj = len(subjects)

    # Subject angles: evenly spaced around circle, starting at top
    subj_angles = np.linspace(0, 2 * np.pi, n_subj, endpoint=False) - np.pi / 2

    n_prompts = domain_surface_data.get("n_prompts_used",
                domain_surface_data.get("n_prompts_total", 0))

    mean_level = np.full(n_prompts, 2.0)
    blended_angle = np.zeros(n_prompts)
    dom_subject = np.zeros(n_prompts, dtype=int)
    nearest_probes = None

    # ── Primary path: direct prompt→probe cosine similarity ──
    if (session_results is not None and probe_embs is not None
            and probes_meta is not None):

        probe_mat = np.array(probe_embs, dtype=np.float32)
        # L2-normalize probe matrix
        probe_norms = np.linalg.norm(probe_mat, axis=1, keepdims=True)
        probe_norms[probe_norms < 1e-12] = 1
        probe_mat_n = probe_mat / probe_norms

        used_direct = 0
        nearest_probes = []

        for pi in range(min(n_prompts, len(session_results))):
            r = session_results[pi]
            emb = r.get("domain_embedding")
            if emb is None or len(emb) == 0:
                nearest_probes.append([])
                continue

            # Cosine similarity: prompt embedding vs all probes
            emb_arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb_arr)
            if norm < 1e-12:
                nearest_probes.append([])
                continue
            emb_n = emb_arr / norm

            sims = probe_mat_n @ emb_n  # [n_probes]

            # Top-k nearest probes
            k = min(knn_probes, len(sims))
            top_k = np.argsort(sims)[-k:][::-1]
            top_sims = sims[top_k]
            nearest_probes.append(top_k.tolist())

            # Weighted position from nearest probes
            # Weights: softmax of similarities for smooth blending
            weights = np.exp(top_sims * 10)  # temperature-scaled
            weights /= weights.sum()

            # Angular position: circular weighted mean of probe subjects
            sin_sum = 0
            cos_sum = 0
            level_sum = 0
            subj_counts = Counter()
            for idx, w in zip(top_k, weights):
                if idx < len(probes_meta):
                    pm = probes_meta[idx]
                    si = pm["subj_idx"]
                    li = pm["level"]
                    sin_sum += w * np.sin(subj_angles[si])
                    cos_sum += w * np.cos(subj_angles[si])
                    level_sum += w * li
                    subj_counts[si] += w

            blended_angle[pi] = np.arctan2(sin_sum, cos_sum)
            mean_level[pi] = level_sum
            dom_subject[pi] = subj_counts.most_common(1)[0][0] if subj_counts else 0
            used_direct += 1

        logger.info(f"[MANIFOLD] Direct prompt→probe kNN: {used_direct}/{n_prompts} "
                     f"prompts positioned (k={knn_probes})")

        if used_direct > 0:
            return mean_level, blended_angle, dom_subject, n_prompts, subj_angles, nearest_probes

    # ── Fallback: token-vote from observations ──
    logger.info("[MANIFOLD] Falling back to token-vote for prompt positioning")
    obs = domain_surface_data.get("observations", [])

    prompt_subjects = defaultdict(Counter)
    prompt_levels = defaultdict(Counter)

    for o in obs:
        pi = o[9]   # prompt index
        prompt_subjects[pi][o[13]] += 1  # near_subj_idx
        prompt_levels[pi][o[12]] += 1    # near_level

    for pi in range(n_prompts):
        if pi in prompt_levels:
            total = sum(prompt_levels[pi].values())
            mean_level[pi] = sum(l * n for l, n in prompt_levels[pi].items()) / total

        if pi in prompt_subjects:
            counts = prompt_subjects[pi]
            dom_subject[pi] = counts.most_common(1)[0][0]

            # Circular weighted mean of subject angles
            total = sum(counts.values())
            sin_sum = sum((n / total) * np.sin(subj_angles[si])
                          for si, n in counts.items())
            cos_sum = sum((n / total) * np.cos(subj_angles[si])
                          for si, n in counts.items())
            blended_angle[pi] = np.arctan2(sin_sum, cos_sum)

    return mean_level, blended_angle, dom_subject, n_prompts, subj_angles, None


# ─── Manifold Construction ───────────────────────────────────

def _get_ring(level, n_levels):
    """Map continuous probe level to ring index for n_levels rings.

    level is a continuous value in [0, n_levels-1].
    Maps linearly to ring indices [0, n_levels-1].
    """
    if n_levels <= 1:
        return 0
    t = max(0.0, min(1.0, level / (n_levels - 1)))
    ring = int(t * (n_levels - 0.01))
    return min(ring, n_levels - 1)


def _make_ring_bands(n_rings, ring_gap=0.04, r_inner=0.18, r_outer=0.92):
    """Generate evenly-spaced ring bands for n_rings rings.

    Returns list of {"inner": float, "outer": float} dicts.
    """
    if n_rings <= 0:
        return []
    total_gap = ring_gap * (n_rings - 1)
    usable = r_outer - r_inner - total_gap
    band_width = usable / n_rings

    bands = []
    cursor = r_inner
    for i in range(n_rings):
        bands.append({"inner": round(cursor, 4), "outer": round(cursor + band_width, 4)})
        cursor += band_width + ring_gap
    return bands


def _cell_corners(angle, ring_idx, ring_bands, wedge_half):
    """Compute 4 corner positions for a cell at given angle and ring.

    Corner mapping:
        0 (ASM):     outer-right
        1 (IntCV):   outer-left
        2 (RD_repl): inner-right
        3 (SFD_d):   inner-left

    Returns:
        corners: (4, 2) array of corner positions
    """
    a_left = angle - wedge_half
    a_right = angle + wedge_half
    ri = ring_bands[ring_idx]["inner"]
    ro = ring_bands[ring_idx]["outer"]

    return np.array([
        [ro * np.cos(a_right), ro * np.sin(a_right)],   # ASM: outer-right
        [ro * np.cos(a_left),  ro * np.sin(a_left)],    # IntCV: outer-left
        [ri * np.cos(a_right), ri * np.sin(a_right)],   # RD_repl: inner-right
        [ri * np.cos(a_left),  ri * np.sin(a_left)],    # SFD_d: inner-left
    ])


def _build_manifold(session_results, mean_level, blended_angle, dom_subject,
                     ring_bands, wedge_half, push_strength,
                     n_levels=5, progress=None):
    """Compute 2D manifold positions for all prompts.

    Anchor-repulsor geometry:
        - Subject angle determines angular direction (wedge)
        - Probe level determines ring band (escalation)
        - RD_repl controls radial position within ring
        - Entropy pushes radially from anchor (outward when high)
        - KL pushes tangentially from anchor (clockwise when high)
        Each anchor has a local coordinate frame: "north" = radially
        outward, "east" = tangential clockwise.

    Returns:
        positions: (n, 2) array of manifold positions
        norm_signals: (n, 4) normalized signal values [Entropy, KL, RD_repl, SFD_e]
        raw_signals: (n, 8) all 8 raw signal values
        rings: (n,) ring assignments
    """
    n = len(session_results)

    # Resolve signal values from session results
    _NESTED = {
        "sfd_density_mean":    ("sfd", "density_mean"),
        "sfd_energy_mean":     ("sfd", "energy_mean"),
        "sfd_entropy_mean":    ("sfd", "entropy_mean"),
        "rd_mean_replacement": ("rank_displacement", "mean_replacement"),
        "rd_mean_overlap":     ("rank_displacement", "mean_overlap"),
        "rd_mean_tau":         ("rank_displacement", "mean_tau"),
    }

    def _get_signal(r, key):
        v = r.get(key)
        if v is not None:
            return float(v)
        nested = _NESTED.get(key)
        if nested:
            parent = r.get(nested[0])
            if isinstance(parent, dict):
                v2 = parent.get(nested[1])
                if v2 is not None:
                    return float(v2)
        return 0.0

    # Extract the 4 signals: Entropy[0], KL[1], RD_repl[2], SFD_e[3]
    raw_4 = np.zeros((n, 4))
    for i, r in enumerate(session_results):
        for j, (key, _) in enumerate(SIGNAL_KEYS):
            raw_4[i, j] = _get_signal(r, key)

    # Z-score normalization: each signal measured in standard deviations
    # from its mean, clipped to ±2σ, rescaled to [0,1].
    # This makes the push axes proportionate — each contributes equally
    # to the visual spread regardless of raw range.
    means = raw_4.mean(axis=0)
    stds = raw_4.std(axis=0)
    stds[stds < 1e-12] = 1
    z_4 = (raw_4 - means) / stds
    z_4 = np.clip(z_4, -2, 2)
    norm_4 = (z_4 + 2) / 4  # rescale [-2,2] → [0,1]

    # Extract all 8 signals
    raw_8 = np.zeros((n, 8))
    for i, r in enumerate(session_results):
        for j, (key, _) in enumerate(ALL_SIGNAL_KEYS):
            raw_8[i, j] = _get_signal(r, key)

    # Normalize all 8
    norm_8 = np.zeros_like(raw_8)
    for j in range(8):
        mn, mx = raw_8[:, j].min(), raw_8[:, j].max()
        norm_8[:, j] = (raw_8[:, j] - mn) / (mx - mn) if mx > mn else 0

    # ── Anchor-repulsor position computation ──
    positions = np.zeros((n, 2))
    rings = np.zeros(n, dtype=int)

    for i in range(n):
        ri = _get_ring(mean_level[i], n_levels)
        rings[i] = ri
        inner = ring_bands[ri]["inner"]
        outer = ring_bands[ri]["outer"]

        # RD positions the point within the ring (structural axis)
        rd_frac = norm_4[i, 2]  # RD_repl normalized
        anchor_r = inner + (outer - inner) * rd_frac

        # Subject angle
        angle = blended_angle[i]

        # Anchor position
        ax = anchor_r * np.cos(angle)
        ay = anchor_r * np.sin(angle)

        # Local coordinate frame at this anchor
        rad_x, rad_y = np.cos(angle), np.sin(angle)  # radial outward
        tan_x, tan_y = np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)  # tangential

        # Signal-driven displacement from anchor.
        # (norm - 0.5) maps [0,1] → [-0.5, +0.5].
        # × ring_width × push_strength = displacement in manifold units.
        # push_strength=1.0 → extreme signals reach the ring edge.
        rw = outer - inner
        e_push = (norm_4[i, 0] - 0.5) * rw * push_strength   # Entropy → radial
        k_push = (norm_4[i, 1] - 0.5) * rw * push_strength   # KL → tangential

        positions[i, 0] = ax + rad_x * e_push + tan_x * k_push
        positions[i, 1] = ay + rad_y * e_push + tan_y * k_push

    if progress:
        progress(f"Computed manifold positions for {n} prompts")

    return positions, norm_4, norm_8, raw_8, rings


# ─── Classification ──────────────────────────────────────────


# ─── Category Mean Positions ─────────────────────────────────

def _category_centroids(positions, categories):
    """Compute mean positions per category for visualization reference."""
    cats = np.array(categories)
    centroids = {}
    for c in sorted(set(cats)):
        mask = cats == c
        centroids[c] = {
            "x": round(float(positions[mask, 0].mean()), 4),
            "y": round(float(positions[mask, 1].mean()), 4),
            "n": int(mask.sum()),
        }
    return centroids


# ─── Module Class ────────────────────────────────────────────

class CorrectionManifoldModule(TASMModule):
    """Correction manifold — the witness plate.

    Combines probe geometry (subject × escalation level) with
    correction signals (Entropy, KL, RD, SFD) into a spatial
    manifold using anchor-repulsor design.

    Requires a completed domain_surface module run.
    """

    name = "correction_manifold"
    display_name = "Correction Manifold"
    description = (
        "Constructs the witness plate from correction signals and "
        "probe geometry. Anchor-repulsor design: each probe anchor "
        "pushes tokens outward via locally-oriented signal axes "
        "(Entropy/SFD_e radial, KL/ASM tangential)."
    )
    version = "0.2.0"

    min_results = 20
    requires_sfd = True
    requires_ltp = False
    requires_rd = True

    def set_project_root(self, root):
        """Set project root for probe cache access."""
        self._project_root = root

    def set_session_dir(self, path):
        """Set session directory for loading domain surface output."""
        self._session_dir = path

    @property
    def parameters(self):
        return [
            ModuleParameter(
                name="push_strength",
                display_name="Push Strength",
                description=(
                    "Fraction of ring width that extreme signal values "
                    "can push a token from its anchor. 1.0 = ring edge. "
                    "Values above 1.0 allow crossing into adjacent rings."
                ),
                type="float",
                default=0.5,
                min_val=0.1,
                max_val=1.5,
            ),
            ModuleParameter(
                name="ring_gap",
                display_name="Ring Gap",
                description=(
                    "Gap between escalation ring bands (0.0\u20130.1). "
                    "Larger gaps create clearer ring separation."
                ),
                type="float",
                default=0.04,
                min_val=0.0,
                max_val=0.1,
            ),
            ModuleParameter(
                name="probe_knn",
                display_name="Probe kNN",
                description=(
                    "Number of nearest probes used to position each prompt. "
                    "Prompt embedding is compared against all probes by "
                    "cosine similarity; the top-k define its position."
                ),
                type="float",
                default=5,
                min_val=1,
                max_val=20,
            ),
        ]

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        # Check that domain surface has been run
        if hasattr(self, '_session_dir') and self._session_dir:
            ds_path = os.path.join(self._session_dir, "module_domain_surface.json")
            if not os.path.exists(ds_path):
                return False, (
                    "Domain Surface module must be run first. "
                    "The Correction Manifold requires probe proximity data."
                )

        return True, "OK"

    def run(self, session_results, params, progress=None):
        """Execute correction manifold analysis.

        Requires a completed domain_surface module run for probe
        proximity data. Computes manifold positions from correction
        signals and probe geometry.
        """
        push_strength = params.get("push_strength", 0.5)
        ring_gap = params.get("ring_gap", 0.04)
        probe_knn = int(params.get("probe_knn", 5))

        # ── Load domain surface output ──
        if progress:
            progress("Loading domain surface data...")

        ds_data = self._load_domain_surface(session_results)
        if ds_data is None:
            raise RuntimeError(
                "Domain Surface module output not found. "
                "Run the Domain Surface module first."
            )

        subjects = ds_data.get("subjects", [])
        n_subj = len(subjects)
        if n_subj == 0:
            raise RuntimeError("No subjects found in domain surface data.")

        # ── Load probe embeddings for direct prompt→probe matching ──
        probe_embs = None
        probes_meta = None
        probe_file = ds_data.get("probe_file")

        if probe_file and getattr(self, '_project_root', None):
            if progress:
                progress("Loading probe embeddings for kNN positioning...")
            try:
                from engine import engine_config
                layer_frac = max(0, min(1, engine_config.get("domain_embedding_layer_frac") or 0.50))
                use_proj = engine_config.get("probe_projection_space")
            except Exception:
                layer_frac = 0.50
                use_proj = False

            # Load probe cache
            cache_path = _probe_cache_path(
                self._project_root, probe_file,
                "",  # model_id — scan for any matching cache
                layer_frac, projected=use_proj)

            # Try exact path first, then scan cache dir
            cache_data = _load_probe_cache(cache_path)
            if cache_data is None:
                # Scan for any matching cache file
                cache_dir = os.path.join(self._project_root, "probe_cache")
                if os.path.isdir(cache_dir):
                    stem = os.path.splitext(probe_file)[0]
                    for fn in sorted(os.listdir(cache_dir)):
                        if fn.startswith(stem) and fn.endswith(".json"):
                            candidate = _load_probe_cache(os.path.join(cache_dir, fn))
                            if candidate and candidate.get("embeddings"):
                                cache_data = candidate
                                logger.info(f"[MANIFOLD] Using probe cache: {fn}")
                                break

            if cache_data and cache_data.get("embeddings"):
                probe_embs = cache_data["embeddings"]

                # Build probe metadata: subject_idx and level per probe
                # Probes are stored in CSV iteration order (rows × level_cols)
                csv_path = os.path.join(self._project_root, probe_file)
                if os.path.exists(csv_path):
                    raw_probes = _load_probes(csv_path)
                    # Build subject→index mapping from domain surface subjects
                    subj_idx_map = {s: i for i, s in enumerate(subjects)}
                    probes_meta = []
                    for p in raw_probes:
                        si = subj_idx_map.get(p["subject"], 0)
                        probes_meta.append({
                            "subj_idx": si,
                            "level": p["level"],
                            "subject": p["subject"],
                            "text": p["text"],
                        })

                    if len(probes_meta) != len(probe_embs):
                        logger.warning(
                            f"[MANIFOLD] Probe count mismatch: meta={len(probes_meta)} "
                            f"embs={len(probe_embs)}. Disabling kNN positioning.")
                        probe_embs = None
                        probes_meta = None
                    else:
                        logger.info(f"[MANIFOLD] Loaded {len(probe_embs)} probe "
                                     f"embeddings for kNN positioning")

        # ── Compute probe stats (kNN or fallback) ──
        if progress:
            progress("Computing per-prompt probe statistics...")

        mean_level, blended_angle, dom_subject, n_prompts, subj_angles, nearest_probes = \
            _compute_prompt_probe_stats(
                ds_data, session_results=session_results,
                probe_embs=probe_embs, probes_meta=probes_meta,
                knn_probes=probe_knn)

        if n_prompts != len(session_results):
            logger.warning(
                f"[MANIFOLD] Prompt count mismatch: DS has {n_prompts}, "
                f"session has {len(session_results)}. Using min."
            )
            n_prompts = min(n_prompts, len(session_results))
            session_results = session_results[:n_prompts]
            mean_level = mean_level[:n_prompts]
            blended_angle = blended_angle[:n_prompts]
            dom_subject = dom_subject[:n_prompts]

        # ── Level names from domain surface ──
        level_names = ds_data.get("level_names", ["nouns", "phrase", "question", "instruct", "meta"])
        n_levels = len(level_names)

        # ── Ring geometry (one ring per escalation level) ──
        ring_bands = _make_ring_bands(n_levels, ring_gap=ring_gap, r_inner=0.10)
        wedge_half = np.pi / n_subj * 0.88

        # ── Build manifold ──
        if progress:
            progress(f"Building manifold: {n_levels} rings × {n_subj} subjects...")

        positions, norm_4, norm_8, raw_8, rings = _build_manifold(
            session_results, mean_level, blended_angle, dom_subject,
            ring_bands, wedge_half, push_strength,
            n_levels=n_levels, progress=progress)

        # ── Category labels (for centroids and viz) ──
        categories = []
        for r in session_results:
            cat = r.get("category", "unknown")
            if cat == "dual-use":
                categories.append("d")
            elif cat:
                categories.append(cat[0])
            else:
                categories.append("?")

        # ── Category centroids ──
        centroids = _category_centroids(positions, categories)

        # ── Build visualization data ──
        if progress:
            progress("Building visualization data...")

        prompts_viz = []
        for i in range(n_prompts):
            r = session_results[i]
            prompts_viz.append([
                r.get("prompt", "")[:80],         # 0: prompt text
                categories[i],                     # 1: category code
                round(float(positions[i, 0]), 4),  # 2: x position
                round(float(-positions[i, 1]), 4), # 3: y position (flip for canvas)
                int(dom_subject[i]),               # 4: dominant subject
                int(rings[i]),                     # 5: ring index
                round(float(norm_4[i, 0]), 4),     # 6: Entropy normalized (radial)
                round(float(norm_4[i, 1]), 4),     # 7: KL normalized (tangential)
                round(float(norm_4[i, 2]), 4),     # 8: RD_repl normalized
                round(float(norm_4[i, 3]), 4),     # 9: SFD_e normalized
                round(float(mean_level[i]), 2),    # 10: mean probe level
            ] + [round(float(raw_8[i, j]), 4) for j in range(8)])  # 11-18: raw

        # ── Token-level positions from domain surface observations ──
        tokens_viz = []
        obs = ds_data.get("observations", [])
        if obs:
            # Collect per-token signal ranges for normalization
            tok_asm = [o[6] for o in obs]
            tok_sfd_e = [o[7] for o in obs]
            tok_repl = [o[4] for o in obs]

            # Z-score normalization for token signals (same as prompt level)
            def _zstats(vals):
                n = len(vals)
                m = sum(vals) / n
                s = (sum((v - m)**2 for v in vals) / n) ** 0.5
                return m, s if s > 1e-12 else 1

            asm_m, asm_s = _zstats(tok_asm)
            sfd_e_m, sfd_e_s = _zstats(tok_sfd_e)
            repl_m, repl_s = _zstats(tok_repl)

            def _znorm(val, m, s):
                z = max(-2, min(2, (val - m) / s))
                return (z + 2) / 4  # [-2,2] → [0,1]

            for o in obs:
                n_asm = _znorm(o[6], asm_m, asm_s)
                n_sfd_e = _znorm(o[7], sfd_e_m, sfd_e_s)
                n_repl = _znorm(o[4], repl_m, repl_s)

                # Ring from probe level (now kNN-weighted float)
                t_level = o[12]  # near_level
                t_ri = _get_ring(t_level, n_levels)
                t_inner = ring_bands[t_ri]["inner"]
                t_outer = ring_bands[t_ri]["outer"]

                # Continuous angle from kNN (index 14), fallback to subject snap
                t_si = int(o[13])  # near_subj_idx (dominant, for metadata)
                if len(o) > 14 and o[14] is not None:
                    t_angle = o[14]  # kNN-weighted continuous angle
                else:
                    t_angle = subj_angles[t_si] if t_si < len(subj_angles) else 0

                # RD positions within ring (structural axis)
                rd_frac = max(0, min(1, n_repl))
                anchor_r = t_inner + (t_outer - t_inner) * rd_frac

                # Local coordinate frame
                rad_x, rad_y = np.cos(t_angle), np.sin(t_angle)
                tan_x, tan_y = np.cos(t_angle + np.pi / 2), np.sin(t_angle + np.pi / 2)

                # Signal-driven push from anchor
                rw = t_outer - t_inner
                e_push = (n_sfd_e - 0.5) * rw * push_strength   # SFD_e → radial
                a_push = (n_asm - 0.5) * rw * push_strength      # ASM → tangential

                t_x = anchor_r * np.cos(t_angle) + rad_x * e_push + tan_x * a_push
                t_y = anchor_r * np.sin(t_angle) + rad_y * e_push + tan_y * a_push

                pi = o[9]  # prompt index
                cat = categories[pi] if pi < len(categories) else "?"
                tokens_viz.append([
                    o[0],                          # 0: token text
                    cat,                           # 1: category code
                    round(float(t_x), 4),          # 2: x
                    round(float(-t_y), 4),         # 3: y (flip)
                    t_si,                          # 4: subject index
                    t_ri,                          # 5: ring index
                    round(float(n_repl), 3),       # 6: RD normalized
                    round(float(n_sfd_e), 3),      # 7: SFD_e normalized (radial)
                    round(float(n_asm), 3),        # 8: ASM normalized (tangential)
                    pi,                            # 9: prompt index
                ])

        # Cell corners for visualization
        cells_viz = []
        for si in range(n_subj):
            for ri in range(len(ring_bands)):
                corners = _cell_corners(subj_angles[si], ri, ring_bands, wedge_half)
                center = corners.mean(axis=0)
                cells_viz.append({
                    "s": si, "r": ri,
                    "corners": [[round(float(c[0]), 4), round(float(-c[1]), 4)]
                                for c in corners],
                    "center": [round(float(center[0]), 4),
                               round(float(-center[1]), 4)],
                })

        subj_short = [s.replace("_", " ").title()[:12] for s in subjects]

        # ── Build output ──
        output = {
            # Metadata
            "version": self.version,
            "n_prompts": n_prompts,
            "push_strength": push_strength,

            # Geometry
            "subjects": subjects,
            "subj_short": subj_short,
            "subj_angles": [round(float(a), 4) for a in subj_angles],
            "rings": [
                {"label": level_names[i].title() if i < len(level_names) else f"Ring {i}",
                 "inner": ring_bands[i]["inner"],
                 "outer": ring_bands[i]["outer"]}
                for i in range(len(ring_bands))
            ],
            "repulsor_axes": {
                "prompt": {
                    "radial": {"signal": "Entropy", "color": "#ff6347"},
                    "tangential": {"signal": "KL", "color": "#4fc3f7"},
                },
                "token": {
                    "radial": {"signal": "SFD_e", "color": "#ff6347"},
                    "tangential": {"signal": "ASM", "color": "#4fc3f7"},
                },
            },
            "center_signal": {"name": "RD", "color": "#a78bfa"},

            # Signal metadata
            "signal_keys": [s for _, s in SIGNAL_KEYS],
            "all_signal_keys": [s for _, s in ALL_SIGNAL_KEYS],

            # Per-prompt data
            "prompts": prompts_viz,
            "tokens": tokens_viz,
            "n_tokens": len(tokens_viz),
            "cells": cells_viz,

            # Prompt positioning method
            "prompt_positioning": "probe_knn" if nearest_probes is not None else "token_vote",
            "probe_knn": probe_knn if nearest_probes is not None else None,
            "nearest_probes": nearest_probes,

            "category_centroids": centroids,

            # Escalation stats
            "level_stats": {
                cat: round(float(np.mean([
                    mean_level[i] for i in range(n_prompts)
                    if categories[i] == cat
                ])), 2)
                for cat in sorted(set(categories)) if cat != "?"
            },
        }

        if progress:
            progress(f"Complete: {n_prompts} prompts positioned")

        return output

    def _load_domain_surface(self, session_results):
        """Load domain surface module output from session directory."""
        # Try session directory
        if hasattr(self, '_session_dir') and self._session_dir:
            ds_path = os.path.join(self._session_dir, "module_domain_surface.json")
            if os.path.exists(ds_path):
                try:
                    with open(ds_path) as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"[MANIFOLD] Failed to load domain surface: {e}")

        # Try common paths
        for path in ["module_domain_surface.json",
                      "datasets/current/module_domain_surface.json"]:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception:
                    continue

        return None
