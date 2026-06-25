"""
Harm Direction Module for TAGM.

Computes a harm direction in SFD spectral space using the same
difference-of-means methodology that the Arditi benchmarks use
for the refusal direction in residual stream space.

The refusal direction answers: "which direction in embedding space
does the model move when it refuses?"

The harm direction answers: "which direction in QK routing space
do tokens point when they participate in harm-convergent connectivity?"

These are different questions in different spaces. The refusal
direction is a behavioral signal (what the model does). The harm
direction is a representational signal (what the token means).

Requires: SFD data with per_token_directions (v0.2 SFD patch).

Produces:
  - Fitted harm direction vector in spectral space
  - Train/holdout AUROC for direction quality
  - Per-token projection scores across all session prompts
  - Separability statistics by category
  - Compatibility output for harm_trajectory module v0.2
"""

import logging
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from .base import TASMModule, ModuleParameter

logger = logging.getLogger("tasm")


# ── Category sets ───────────────────────────────────────────────

HARM_CATEGORIES = {"harmful", "jailbreak", "adversarial", "unknown"}
SAFE_CATEGORIES = {"benign", "mild", "safe"}


# ── Fitted direction output ─────────────────────────────────────

@dataclass
class FittedHarmDirection:
    """Output of harm direction fitting."""
    vector: np.ndarray           # unit-normalized, float32
    method: str
    spectral_dim: int            # k (SVD truncation rank)
    n_harm_tokens: int
    n_safe_tokens: int
    n_harm_prompts: int
    n_safe_prompts: int
    train_auroc: float
    holdout_auroc: float
    heldout_prompts: list
    convergence_ratio: float     # harm_angular_var / safe_angular_var


# ── Direction fitting ───────────────────────────────────────────

class SpectralDirectionFitter:
    """Fits a harm direction in SFD spectral space.

    Analogous to DirectionFitter in ablation.py but operates on
    per_token_directions from SFD results rather than
    per_token_final_emb from the residual stream.
    """

    def __init__(self, session_results, harm_cats, safe_cats):
        self.results = session_results
        self.harm_cats = {c.lower().strip() for c in harm_cats}
        self.safe_cats = {c.lower().strip() for c in safe_cats}

    def _extract_directions(self):
        """Extract (direction_vectors, labels, prompt_indices) from
        session data. Each token with a non-empty SFD direction
        contributes one vector."""
        vectors = []
        labels = []
        prompt_indices = []
        prompt_categories = []

        for i, r in enumerate(self.results):
            cat = (r.get("category") or "").lower().strip()
            if cat in self.harm_cats:
                y = 1
            elif cat in self.safe_cats:
                y = 0
            else:
                continue

            sfd = r.get("sfd", {})
            dirs = sfd.get("per_token_directions", [])
            if not dirs:
                continue

            for d in dirs:
                if not d or len(d) == 0:
                    continue
                v = np.array(d, dtype=np.float32)
                if np.linalg.norm(v) < 1e-10:
                    continue
                vectors.append(v / (np.linalg.norm(v) + 1e-10))
                labels.append(y)
                prompt_indices.append(i)

            prompt_categories.append((i, cat, y))

        return vectors, labels, prompt_indices, prompt_categories

    def difference_of_means(self, holdout_frac=0.2, seed=0):
        """Compute harm direction as mean(harm_dirs) - mean(safe_dirs).

        Holdout is at the prompt level (not token level) to prevent
        data leakage from the same prompt appearing in both train
        and test.
        """
        rng = np.random.default_rng(seed)
        vectors, labels, prompt_indices, prompt_cats = \
            self._extract_directions()

        if len(vectors) < 4:
            raise ValueError(
                f"Need at least 4 token directions, got {len(vectors)}. "
                f"Run SFD with per_token_directions enabled."
            )

        vectors = np.stack(vectors)
        labels = np.array(labels, dtype=np.int32)
        prompt_indices = np.array(prompt_indices)

        # Get unique prompts per class for holdout split
        harm_prompts = sorted(set(
            i for i, c, y in prompt_cats if y == 1
        ))
        safe_prompts = sorted(set(
            i for i, c, y in prompt_cats if y == 0
        ))

        if len(harm_prompts) < 2 or len(safe_prompts) < 2:
            raise ValueError(
                f"Need >=2 prompts per class; got "
                f"{len(harm_prompts)} harm, {len(safe_prompts)} safe."
            )

        # Hold out harm prompts for testing
        perm = rng.permutation(len(harm_prompts))
        n_train = max(2, int(round(len(harm_prompts) * (1 - holdout_frac))))
        train_harm = set(
            harm_prompts[j] for j in perm[:n_train]
        )
        held_harm = set(
            harm_prompts[j] for j in perm[n_train:]
        )
        train_safe = set(safe_prompts)

        # Partition token vectors by prompt membership
        train_mask = np.array([
            (labels[i] == 1 and prompt_indices[i] in train_harm) or
            (labels[i] == 0 and prompt_indices[i] in train_safe)
            for i in range(len(vectors))
        ])
        held_mask = np.array([
            prompt_indices[i] in held_harm
            for i in range(len(vectors))
        ])

        X_train = vectors[train_mask]
        y_train = labels[train_mask]
        X_held = vectors[held_mask]
        y_held = labels[held_mask]

        X_harm = X_train[y_train == 1]
        X_safe = X_train[y_train == 0]

        if len(X_harm) < 2 or len(X_safe) < 2:
            raise ValueError("Insufficient tokens after split.")

        # Difference of means
        v = X_harm.mean(axis=0) - X_safe.mean(axis=0)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError(
                "Harm direction has near-zero norm; classes may be "
                "indistinguishable in spectral space."
            )
        v = v / norm

        # Train AUROC
        scores_train = X_train @ v
        train_auroc = _auroc(scores_train, y_train)

        # Holdout AUROC (harm held-out vs all safe)
        holdout_auroc = 0.5
        if len(X_held) > 0:
            safe_scores = X_safe @ v
            held_scores = X_held @ v
            all_scores = np.concatenate([held_scores, safe_scores])
            all_labels = np.concatenate([
                np.ones(len(held_scores)),
                np.zeros(len(safe_scores)),
            ]).astype(int)
            holdout_auroc = _auroc(all_scores, all_labels)

        # Convergence ratio: angular variance of harm vs safe
        # This measures whether harm tokens actually converge
        # (the theoretical prediction)
        harm_cos = X_harm @ v
        safe_cos = X_safe @ v
        harm_angular_var = float(np.var(harm_cos))
        safe_angular_var = float(np.var(safe_cos))
        convergence_ratio = (
            harm_angular_var / (safe_angular_var + 1e-10)
        )

        # Heldout prompt texts
        held_prompts = [
            self.results[i].get("prompt", "")
            for i in sorted(held_harm)
        ]

        return FittedHarmDirection(
            vector=v.astype(np.float32),
            method="difference_of_means",
            spectral_dim=int(v.shape[0]),
            n_harm_tokens=int(len(X_harm)),
            n_safe_tokens=int(len(X_safe)),
            n_harm_prompts=len(harm_prompts),
            n_safe_prompts=len(safe_prompts),
            train_auroc=float(train_auroc),
            holdout_auroc=float(holdout_auroc),
            heldout_prompts=held_prompts,
            convergence_ratio=float(convergence_ratio),
        )


# ── Module ──────────────────────────────────────────────────────

class HarmDirectionModule(TASMModule):
    name = "harm_direction"
    display_name = "Harm Direction (SFD)"
    description = (
        "Computes a harm direction in SFD spectral space using "
        "difference-of-means on per-token spectral direction vectors. "
        "Analogous to the refusal direction but in QK routing space "
        "rather than residual stream space. The refusal direction "
        "measures what the model does. The harm direction measures "
        "what the tokens mean."
    )
    version = "0.1.0"

    min_results = 10
    requires_sfd = True
    requires_ltp = False
    requires_rd = False

    parameters = [
        ModuleParameter(
            name="harm_categories",
            display_name="Harmful Categories",
            description=(
                "Comma-separated session categories treated as harmful."
            ),
            type="str",
            default="harmful,jailbreak,unknown",
        ),
        ModuleParameter(
            name="safe_categories",
            display_name="Safe Categories",
            description=(
                "Comma-separated session categories treated as safe."
            ),
            type="str",
            default="benign,mild",
        ),
        ModuleParameter(
            name="holdout_frac",
            display_name="Holdout Fraction",
            description="Fraction of harm prompts held out for validation.",
            type="float",
            default=0.2,
            min_val=0.05,
            max_val=0.5,
        ),
        ModuleParameter(
            name="seed",
            display_name="Random Seed",
            description="Seed for the train/holdout split.",
            type="int",
            default=42,
        ),
    ]

    def validate(self, session_results, params):
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        # Check for per_token_directions in SFD data
        has_dirs = sum(
            1 for r in session_results
            if r.get("sfd", {}).get("per_token_directions")
        )
        if has_dirs == 0:
            return False, (
                "SFD data does not contain per_token_directions. "
                "Apply the SFD direction persistence patch and "
                "re-run the session with SFD enabled."
            )
        return True, "OK"

    def run(self, session_results, params, progress=None):
        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[HARM_DIR] {msg}")

        harm_cats = {
            c.strip() for c in
            params.get("harm_categories", "harmful,jailbreak,unknown")
            .split(",")
        }
        safe_cats = {
            c.strip() for c in
            params.get("safe_categories", "benign,mild")
            .split(",")
        }
        holdout_frac = params.get("holdout_frac", 0.2)
        seed = params.get("seed", 42)

        n = len(session_results)
        prog(f"Fitting harm direction from {n} prompts...")

        # ── Fit direction ───────────────────────────────────────

        fitter = SpectralDirectionFitter(
            session_results, harm_cats, safe_cats
        )
        fit = fitter.difference_of_means(
            holdout_frac=holdout_frac, seed=seed
        )

        prog(f"Direction fitted: dim={fit.spectral_dim}, "
             f"train AUROC={fit.train_auroc:.3f}, "
             f"holdout AUROC={fit.holdout_auroc:.3f}, "
             f"convergence ratio={fit.convergence_ratio:.4f}")

        # ── Project all tokens onto harm direction ──────────────

        prog("Projecting all tokens onto harm direction...")

        per_prompt_projections = []
        for i, r in enumerate(session_results):
            sfd = r.get("sfd", {})
            dirs = sfd.get("per_token_directions", [])
            tokens = r.get("tokens", [])
            category = r.get("category", "unknown")

            projections = []
            for t, d in enumerate(dirs):
                if not d or len(d) == 0:
                    projections.append(0.0)
                    continue
                v = np.array(d, dtype=np.float32)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-10:
                    projections.append(0.0)
                    continue
                v = v / v_norm
                proj = float(v @ fit.vector)
                projections.append(round(proj, 6))

            per_prompt_projections.append({
                "index": i,
                "prompt": r.get("prompt", ""),
                "category": category,
                "tokens": tokens,
                "projections": projections,
                "mean_projection": round(
                    float(np.mean(projections)) if projections else 0, 6
                ),
                "max_projection": round(
                    float(max(projections)) if projections else 0, 6
                ),
                "n_positive": sum(1 for p in projections if p > 0),
                "n_tokens": len(projections),
            })

        # ── Category separability ───────────────────────────────

        cat_means = defaultdict(list)
        for pp in per_prompt_projections:
            cat_means[pp["category"]].append(pp["mean_projection"])

        category_stats = {}
        for cat in sorted(cat_means.keys()):
            vals = cat_means[cat]
            category_stats[cat] = {
                "n": len(vals),
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "max": round(float(max(vals)), 6),
                "min": round(float(min(vals)), 6),
            }

        # ── Token leaderboard by projection ─────────────────────

        token_projections = defaultdict(list)
        for pp in per_prompt_projections:
            for t, proj in zip(pp["tokens"], pp["projections"]):
                tok = t.strip()
                if tok:
                    token_projections[tok].append(proj)

        token_leaderboard = sorted(
            [
                {
                    "token": tok,
                    "mean_projection": round(
                        float(np.mean(projs)), 6
                    ),
                    "max_projection": round(float(max(projs)), 6),
                    "occurrences": len(projs),
                    "std": round(float(np.std(projs)), 6),
                }
                for tok, projs in token_projections.items()
                if len(projs) > 0
            ],
            key=lambda x: x["mean_projection"],
            reverse=True,
        )[:50]

        prog(f"Complete. Harm direction fitted with "
             f"AUROC={fit.holdout_auroc:.3f}")

        return {
            "direction": {
                "vector": [round(float(x), 8) for x in fit.vector],
                "method": fit.method,
                "spectral_dim": fit.spectral_dim,
                "n_harm_tokens": fit.n_harm_tokens,
                "n_safe_tokens": fit.n_safe_tokens,
                "n_harm_prompts": fit.n_harm_prompts,
                "n_safe_prompts": fit.n_safe_prompts,
                "train_auroc": fit.train_auroc,
                "holdout_auroc": fit.holdout_auroc,
                "convergence_ratio": fit.convergence_ratio,
            },
            "per_prompt_projections": per_prompt_projections,
            "category_stats": category_stats,
            "token_leaderboard": token_leaderboard,
            "heldout_prompts": fit.heldout_prompts,
            "params": {
                "harm_categories": sorted(harm_cats),
                "safe_categories": sorted(safe_cats),
                "holdout_frac": holdout_frac,
                "seed": seed,
            },
            "n_prompts": n,
        }


# ── Helpers ─────────────────────────────────────────────────────

def _auroc(scores, labels):
    """Wilcoxon-Mann-Whitney AUROC. No sklearn dependency."""
    if len(scores) < 4:
        return 0.5
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.5
    s_pos = scores[pos]
    s_neg = scores[neg]
    n_pos = len(s_pos)
    n_neg = len(s_neg)
    # Count concordant pairs
    concordant = sum(
        np.sum(s_neg < sp) + 0.5 * np.sum(s_neg == sp)
        for sp in s_pos
    )
    return float(concordant / (n_pos * n_neg))
