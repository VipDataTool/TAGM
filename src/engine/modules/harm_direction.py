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

from .base import TASMModule, ModuleParameter, classify_category
from src.engine.metrics import (
    HARM_CATEGORIES,
    SAFE_CATEGORIES,
    auroc as _metric_auroc,
    fit_direction_holdout,
)

logger = logging.getLogger("tasm")


# ── Category sets ───────────────────────────────────────────────
# WAS WRONG: this module's private vocabulary treated "unknown" as HARMFUL,
# silently inflating n_harm with unlabelled prompts. HARM_CATEGORIES /
# SAFE_CATEGORIES are now re-exported from src.engine.metrics, which excludes
# "unknown" rather than guessing. n_harm_prompts WILL CHANGE for sessions
# containing unknown-category prompts.


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
    train_auroc: float           # IN-SAMPLE, prompt-level
    holdout_auroc: float         # out-of-fold CV, BOTH classes held out
    heldout_prompts: list
    convergence_ratio: float     # harm_angular_var / safe_angular_var
    cv_p_value: float = None     # label-permutation null on holdout_auroc
    underpowered: bool = False
    n_excluded_prompts: int = 0  # category neither harm nor safe


# ── Direction fitting ───────────────────────────────────────────

class SpectralDirectionFitter:
    """Fits a harm direction in SFD spectral space.

    Analogous to DirectionFitter in ablation.py but operates on
    per_token_directions from SFD results rather than
    per_token_final_emb from the residual stream.
    """

    def __init__(self, session_results, harm_cats=None, safe_cats=None):
        self.results = session_results
        # Optional explicit overrides layered on the canonical taxonomy.
        self.harm_cats = ({c.lower().strip() for c in harm_cats}
                          if harm_cats else None)
        self.safe_cats = ({c.lower().strip() for c in safe_cats}
                          if safe_cats else None)
        self.n_excluded = 0

    def _classify(self, cat):
        return classify_category(cat, harm_override=self.harm_cats,
                                 safe_override=self.safe_cats)

    def _extract_directions(self):
        """Extract token- and prompt-level direction vectors from session data.

        Each token with a non-empty SFD direction contributes one unit vector.
        A prompt-level vector is also produced: the mean of that prompt's unit
        token directions, re-normalized.

        TOKEN DEPENDENCE (was wrong): the previous code emitted one sample PER
        TOKEN and then fitted and scored on those samples as if they were
        i.i.d. Tokens within a prompt are strongly correlated, so the effective
        sample size was the number of PROMPTS, not the number of tokens, and
        every AUROC and variance here was correspondingly over-confident.
        Fitting and scoring now happen on the prompt-level vectors (one row per
        prompt), which removes the within-prompt dependence outright.
        """
        vectors = []
        labels = []
        prompt_indices = []
        prompt_categories = []
        prompt_vectors = []       # one aggregated vector per prompt
        prompt_labels = []
        prompt_ids = []
        self.n_excluded = 0

        for i, r in enumerate(self.results):
            cat = (r.get("category") or "").lower().strip()
            cls = self._classify(cat)
            if cls == "harm":
                y = 1
            elif cls == "safe":
                y = 0
            else:
                self.n_excluded += 1
                continue

            sfd = r.get("sfd", {})
            dirs = sfd.get("per_token_directions", [])
            if not dirs:
                continue

            this_prompt = []
            for d in dirs:
                if not d or len(d) == 0:
                    continue
                v = np.array(d, dtype=np.float32)
                if np.linalg.norm(v) < 1e-10:
                    continue
                v = v / (np.linalg.norm(v) + 1e-10)
                vectors.append(v)
                labels.append(y)
                prompt_indices.append(i)
                this_prompt.append(v)

            if this_prompt:
                pv = np.mean(this_prompt, axis=0)
                n = np.linalg.norm(pv)
                if n > 1e-10:
                    prompt_vectors.append((pv / n).astype(np.float32))
                    prompt_labels.append(y)
                    prompt_ids.append(i)

            prompt_categories.append((i, cat, y))

        self.prompt_vectors = prompt_vectors
        self.prompt_labels = prompt_labels
        self.prompt_ids = prompt_ids
        return vectors, labels, prompt_indices, prompt_categories

    def difference_of_means(self, holdout_frac=0.2, seed=0):
        """Compute harm direction as mean(harm) - mean(safe), prompt-level.

        BROKEN HOLDOUT (was wrong): the previous version held out ONLY the
        positive class — ``train_safe = set(safe_prompts)`` put every safe
        prompt in training, and the "holdout" AUROC then scored those same
        training safe rows (``safe_scores = X_safe @ v``) against the held-out
        harm rows as if the safe side had been held out too. Half the
        comparison was in-sample, which inflates the number.

        Now: both classes are held out, via k-fold CV that refits the
        direction inside each fold and scores out-of-fold, plus a
        label-permutation null through the identical pipeline. Rows are
        one-per-PROMPT (see _extract_directions) so no prompt can appear in
        both train and test, and correlated within-prompt tokens no longer
        masquerade as independent samples.

        NUMBERS CHANGE: ``holdout_auroc`` is now an out-of-fold CV AUROC on
        prompt-level vectors and will differ from — usually be lower than —
        every previously reported value. ``train_auroc`` is likewise now
        prompt-level and in-sample.
        """
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

        # Get unique prompts per class
        harm_prompts = sorted(set(i for i, c, y in prompt_cats if y == 1))
        safe_prompts = sorted(set(i for i, c, y in prompt_cats if y == 0))

        if len(harm_prompts) < 2 or len(safe_prompts) < 2:
            raise ValueError(
                f"Need >=2 prompts per class; got "
                f"{len(harm_prompts)} harm, {len(safe_prompts)} safe."
            )

        # ── Prompt-level design matrix ──────────────────────────
        P = np.stack(self.prompt_vectors)
        py = np.array(self.prompt_labels, dtype=np.int32)
        P_harm = P[py == 1]
        P_safe = P[py == 0]
        if len(P_harm) < 2 or len(P_safe) < 2:
            raise ValueError(
                f"Need >=2 prompts per class with usable SFD directions; got "
                f"{len(P_harm)} harm, {len(P_safe)} safe.")

        cv = fit_direction_holdout(P_harm, P_safe, seed=seed)
        v = np.asarray(cv["direction"], dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError(
                "Harm direction has near-zero norm; classes may be "
                "indistinguishable in spectral space."
            )
        v = (v / norm).astype(np.float32)

        train_auroc = cv["train_auroc"]      # in-sample, prompt-level
        holdout_auroc = cv["cv_auroc"]       # out-of-fold, BOTH classes

        # Convergence ratio: angular variance of harm vs safe. Computed on
        # TOKEN vectors, which is what the theoretical claim is about; note
        # that the tokens are not independent, so treat this as descriptive.
        X_harm = vectors[labels == 1]
        X_safe = vectors[labels == 0]
        harm_cos = X_harm @ v
        safe_cos = X_safe @ v
        harm_angular_var = float(np.var(harm_cos))
        safe_angular_var = float(np.var(safe_cos))
        convergence_ratio = (
            harm_angular_var / (safe_angular_var + 1e-10)
        )

        # Reserve a nominal harm-prompt holdout list for downstream causal
        # tests. It no longer plays any role in the AUROC above.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(harm_prompts))
        n_train = max(2, int(round(len(harm_prompts) * (1 - holdout_frac))))
        held_harm = sorted(harm_prompts[j] for j in perm[n_train:])
        held_prompts = [self.results[i].get("prompt", "") for i in held_harm]

        return FittedHarmDirection(
            vector=v.astype(np.float32),
            method="difference_of_means_prompt_level_cv",
            spectral_dim=int(v.shape[0]),
            n_harm_tokens=int(len(X_harm)),
            n_safe_tokens=int(len(X_safe)),
            n_harm_prompts=len(harm_prompts),
            n_safe_prompts=len(safe_prompts),
            train_auroc=float(train_auroc),
            holdout_auroc=float(holdout_auroc),
            heldout_prompts=held_prompts,
            convergence_ratio=float(convergence_ratio),
            cv_p_value=cv["p_value"],
            underpowered=bool(cv["underpowered"]),
            n_excluded_prompts=int(self.n_excluded),
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
                "Comma-separated session categories treated as harmful. "
                "Leave EMPTY to use the canonical taxonomy in "
                "src/engine/metrics.py (recommended). Note that 'unknown' is "
                "no longer harmful by default — it is excluded."
            ),
            type="str",
            default="",
        ),
        ModuleParameter(
            name="safe_categories",
            display_name="Safe Categories",
            description=(
                "Comma-separated session categories treated as safe. "
                "Leave EMPTY to use the canonical taxonomy in "
                "src/engine/metrics.py (recommended)."
            ),
            type="str",
            default="",
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

        # Empty / whitespace-only lists mean "use the canonical taxonomy".
        harm_cats = {
            c.strip() for c in
            params.get("harm_categories", "").split(",") if c.strip()
        } or None
        safe_cats = {
            c.strip() for c in
            params.get("safe_categories", "").split(",") if c.strip()
        } or None
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
             f"train AUROC={fit.train_auroc:.3f} (IN-SAMPLE), "
             f"CV holdout AUROC={fit.holdout_auroc:.3f} "
             f"(p={fit.cv_p_value}), "
             f"convergence ratio={fit.convergence_ratio:.4f}"
             + (" [UNDERPOWERED]" if fit.underpowered else ""))

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

        prog(f"Complete. Harm direction fitted with out-of-fold CV "
             f"AUROC={fit.holdout_auroc:.3f} (p={fit.cv_p_value})")

        return {
            "direction": {
                "vector": [round(float(x), 8) for x in fit.vector],
                "method": fit.method,
                "spectral_dim": fit.spectral_dim,
                "n_harm_tokens": fit.n_harm_tokens,
                "n_safe_tokens": fit.n_safe_tokens,
                "n_harm_prompts": fit.n_harm_prompts,
                "n_safe_prompts": fit.n_safe_prompts,
                # Prompts whose category is neither harm nor safe are
                # EXCLUDED (previously "unknown" was counted as harm).
                "n_excluded_prompts": fit.n_excluded_prompts,
                # IN-SAMPLE: direction fitted on the rows it scores.
                "train_auroc": fit.train_auroc,
                # Out-of-fold, BOTH classes held out, prompt-level rows.
                "holdout_auroc": fit.holdout_auroc,
                "cv_auroc": fit.holdout_auroc,
                "cv_p_value": fit.cv_p_value,
                "underpowered": fit.underpowered,
                "unit_of_analysis": "prompt",
                "convergence_ratio": fit.convergence_ratio,
            },
            "per_prompt_projections": per_prompt_projections,
            "category_stats": category_stats,
            "token_leaderboard": token_leaderboard,
            "heldout_prompts": fit.heldout_prompts,
            "params": {
                "harm_categories": (sorted(harm_cats) if harm_cats
                                    else sorted(HARM_CATEGORIES)),
                "safe_categories": (sorted(safe_cats) if safe_cats
                                    else sorted(SAFE_CATEGORIES)),
                "holdout_frac": holdout_frac,
                "seed": seed,
            },
            "n_prompts": n,
        }


# ── Helpers ─────────────────────────────────────────────────────

def _auroc(scores, labels):
    """Wilcoxon-Mann-Whitney AUROC.

    WAS WRONG (as a design): one of five divergent copies. Body now delegates
    to the single verified implementation in src.engine.metrics. Behaviour is
    unchanged (same n < 4 guard, same tie handling).
    """
    return _metric_auroc(scores, labels)
