"""Routing Ablation Module — dual-projection causal intervention.

Tests whether the SFD harm direction (routing signal) survives refusal
ablation and/or provides causal behavioral effects independent of the
refusal direction.

Four stages, each independently toggleable:

  Stage 1 — Residual-stream abliteration (Arditi weight orthogonalization)
  Stage 2 — SRA concept-guided spectral cleaning (deferred)
  Stage 3 — QK routing correction (SKOP-adapted query-space projection)
  Stage 4 — Measurement harness (runs alongside all stages)

The measurement harness reports for each condition:
  - Refusal rate (phrase-match detector)
  - Harm probe AUROC at t_inst (harm recognition)
  - Refusal probe AUROC at t_post_inst (refusal execution)
  - Per-head Safety Attention Score (SAS)
  - SFD harm direction projection (if SFD data available)
  - First-token KL divergence (capability drift guard)

Requires: session data with per_token_final_emb and (for Stage 3)
SFD data with per_token_directions.
"""

import logging
import time
import copy
from typing import Any, Callable, Optional
from collections import defaultdict

import numpy as np
import torch

from .base import (TASMModule, ModuleParameter, classify_category,
                   ModuleCancelled)
from src.engine.metrics import (
    HARM_CATEGORIES,
    SAFE_CATEGORIES,
    auroc as _metric_auroc,
    auroc_with_n,
)
from src.engine.ablation import (
    AblationConfig,
    AblationRunner,
    DirectionFitter,
    FittedDirection,
    RefusalDetector,
)
from src.engine.interventions import (
    ActivationIntervention,
    InterventionSpec,
)
from src.core.locks import MODEL_LOCK

logger = logging.getLogger("src")


# ── Category sets ──────────────────────────────────────────────────
# WAS WRONG: this module carried a private {"harmful","jailbreak"} /
# {"benign","mild"} vocabulary that disagreed with the five other copies in
# this package.  HARM_CATEGORIES / SAFE_CATEGORIES are now re-exported from
# src.engine.metrics so every module counts the same prompts.  Practical
# effect here: "adversarial", "dual-use", "risk" and "dangerous" now count as
# harm (they were silently dropped), and "safe"/"neutral"/"baseline" now
# count as safe.  n_harm / n_safe will change for existing sessions.


# ── Harm Probe ─────────────────────────────────────────────────────

class HarmProbe:
    """Linear probe on residual-stream activations to measure whether
    harm recognition survives an intervention.

    Trains logistic regression (via closed-form ridge) on hidden states
    at a specified token position. Reports AUROC on a held-out split.
    """

    def __init__(self, regularization: float = 1.0):
        self.reg = regularization
        self.weights = None    # [hidden_dim]
        self.bias = 0.0
        self.train_auroc = 0.0
        self.holdout_auroc = 0.0
        # The holdout can be as small as 1-vs-1 (see fit()), which no rank
        # statistic can interpret. Track that explicitly instead of letting
        # a meaningless 0.0/1.0 flow into a boolean verdict.
        self.holdout_underpowered = True
        self.holdout_n_pos = 0
        self.holdout_n_neg = 0

    def fit(self, features: np.ndarray, labels: np.ndarray,
            holdout_frac: float = 0.2, seed: int = 42) -> None:
        """Fit a ridge-regularized linear probe.

        Args:
            features: [n_samples, hidden_dim] activation vectors
            labels: [n_samples] binary (1=harm, 0=safe)
            holdout_frac: fraction held out for validation
            seed: random seed for split
        """
        n = len(labels)
        if n < 10:
            logger.warning(f"[HARM-PROBE] Only {n} samples; probe unreliable.")

        # Stratified split
        rng = np.random.default_rng(seed)
        harm_idx = np.where(labels == 1)[0]
        safe_idx = np.where(labels == 0)[0]
        n_hold_harm = max(1, int(len(harm_idx) * holdout_frac))
        n_hold_safe = max(1, int(len(safe_idx) * holdout_frac))
        rng.shuffle(harm_idx)
        rng.shuffle(safe_idx)
        hold_idx = np.concatenate([harm_idx[:n_hold_harm], safe_idx[:n_hold_safe]])
        train_idx = np.concatenate([harm_idx[n_hold_harm:], safe_idx[n_hold_safe:]])

        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_hold = features[hold_idx]
        y_hold = labels[hold_idx]

        # Ridge regression: w = (X^T X + λI)^-1 X^T y
        # Center features and labels for implicit bias
        X_mean = X_train.mean(axis=0)
        y_mean = y_train.mean()
        Xc = X_train - X_mean
        yc = y_train - y_mean

        d = Xc.shape[1]
        XtX = Xc.T @ Xc + self.reg * np.eye(d, dtype=np.float32)
        Xty = Xc.T @ yc

        try:
            self.weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        self.bias = float(y_mean - X_mean @ self.weights)

        # Score
        train_scores = X_train @ self.weights + self.bias
        hold_scores = X_hold @ self.weights + self.bias
        self.train_auroc = self._auroc(train_scores, y_train)
        # NUMBERS CHANGE HERE: the old local _auroc had no small-n guard, so
        # the 1-vs-1 holdout produced by n_hold_* = max(1, ...) above returned
        # exactly 0.0 or 1.0.  The shared implementation returns 0.5 (chance)
        # when either class is too small, and we now carry the sample sizes.
        hold_stats = auroc_with_n(hold_scores, y_hold)
        self.holdout_auroc = hold_stats["auroc"]
        self.holdout_n_pos = hold_stats["n_pos"]
        self.holdout_n_neg = hold_stats["n_neg"]
        self.holdout_underpowered = hold_stats["underpowered"]
        if self.holdout_underpowered:
            logger.warning(
                "[HARM-PROBE] Holdout is %d harm vs %d safe — too small for a "
                "rank statistic. holdout_auroc reported as chance (0.5); no "
                "survival verdict will be issued.",
                self.holdout_n_pos, self.holdout_n_neg)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Score new samples. Higher = more harm-like."""
        if self.weights is None:
            raise RuntimeError("Probe not fitted.")
        return features @ self.weights + self.bias

    def to_dict(self) -> dict:
        return {
            # train_auroc is IN-SAMPLE (the probe was fitted on these rows).
            "train_auroc": round(self.train_auroc, 4),
            "holdout_auroc": round(self.holdout_auroc, 4),
            "holdout_n_harm": self.holdout_n_pos,
            "holdout_n_safe": self.holdout_n_neg,
            "holdout_underpowered": self.holdout_underpowered,
            "regularization": self.reg,
        }

    @staticmethod
    def _auroc(scores, labels):
        # WAS WRONG: a private, unguarded copy of Wilcoxon-Mann-Whitney (one
        # of five divergent copies in this package). Now delegates to the
        # single verified implementation in src.engine.metrics, which returns
        # 0.5 rather than 0.0/1.0 when a class has fewer than 4 samples.
        return _metric_auroc(scores, labels)


# ── Safety Attention Score ─────────────────────────────────────────

def compute_sas(
    model,
    adapter,
    tokenizer,
    prompts: list[str],
    instruction_token_counts: list[int],
    layers: list[int],
    progress: Optional[Callable] = None,
) -> dict:
    """Compute per-head Safety Attention Score (SAS).

    SAS is the fraction of attention mass a head allocates to
    instruction/system tokens (first instruction_token_count positions).

    Returns:
        {
            "per_head": {(layer, head): mean_sas},
            "per_layer": {layer: mean_sas},
            "top_heads": [(layer, head, sas), ...],  # sorted desc
        }
    """
    from src.engine.hooks import ActivationCapture

    n_heads, _ = adapter.attention_heads(model)
    sas_accum = defaultdict(list)  # (layer, head) -> [sas_values]

    cap = ActivationCapture()
    cap.install(model, adapter, signal_layers=layers, output_attentions=True)

    # HOOK SAFETY (was wrong): cap.remove() used to sit after the loop with no
    # try/finally, so any failure inside left capture hooks installed on the
    # shared model forever. Matches correction_prism._beam_activation.
    try:
        for i, prompt in enumerate(prompts):
            if progress and i % 10 == 0:
                progress(f"SAS: prompt {i+1}/{len(prompts)}")

            n_inst = instruction_token_counts[i]
            with torch.no_grad():
                tokens, inputs, output = cap.forward(
                    model, tokenizer, prompt, output_attentions=True
                )

            # Extract attention weights from output.
            # output.attentions is a tuple over ALL model layers, indexed by
            # ABSOLUTE layer index — [batch, n_heads, seq, seq] each.
            # WAS WRONG: this indexed by the caller's list position
            # (enumerate offset), which is only correct when `layers` happens
            # to be [0..k-1]. Any non-contiguous or non-zero-based layer list
            # mislabelled every head in the SAS table.
            if hasattr(output, 'attentions') and output.attentions is not None:
                n_attn_layers = len(output.attentions)
                assert max(layers) < n_attn_layers, (
                    f"SAS: requested layer {max(layers)} but the model "
                    f"returned attentions for only {n_attn_layers} layers; "
                    f"attention indexing would be wrong.")
                for li in layers:
                    attn = output.attentions[li]  # [1, n_heads, seq, seq]
                    if attn is None:
                        continue
                    attn = attn[0].cpu().numpy()  # [n_heads, seq, seq]
                    for hi in range(min(n_heads, attn.shape[0])):
                        # SAS = mean attention to instruction tokens across all positions
                        inst_mass = attn[hi, :, :n_inst].sum(axis=-1).mean()
                        sas_accum[(li, hi)].append(float(inst_mass))
    finally:
        cap.remove()

    # Aggregate
    per_head = {k: float(np.mean(v)) for k, v in sas_accum.items()}
    per_layer = defaultdict(list)
    for (li, hi), sas in per_head.items():
        per_layer[li].append(sas)
    per_layer = {k: float(np.mean(v)) for k, v in per_layer.items()}

    top_heads = sorted(per_head.items(), key=lambda x: x[1], reverse=True)
    top_heads = [(li, hi, sas) for (li, hi), sas in top_heads[:30]]

    return {
        "per_head": {f"{li}_{hi}": v for (li, hi), v in per_head.items()},
        "per_layer": per_layer,
        "top_heads": [{"layer": li, "head": hi, "sas": round(sas, 4)}
                      for li, hi, sas in top_heads],
    }


# ── First-token KL divergence ──────────────────────────────────────

def compute_first_token_kl(
    model,
    tokenizer,
    prompts: list[str],
    baseline_logits: Optional[list[np.ndarray]] = None,
    progress: Optional[Callable] = None,
) -> dict:
    """Compute first-token KL divergence between current and baseline models.

    If baseline_logits is None, returns the current logits for use as
    a baseline in subsequent calls.
    """
    current_logits = []
    device = next(model.parameters()).device

    for i, prompt in enumerate(prompts):
        if progress and i % 10 == 0:
            progress(f"KL: prompt {i+1}/{len(prompts)}")

        from src.engine import config as engine_config
        inputs = tokenizer(
            prompt, return_tensors="pt",
            add_special_tokens=engine_config.get("add_special_tokens"),
        ).to(device)

        with torch.no_grad():
            output = model(**inputs)
            # First generated token logits = last position logits
            logits = output.logits[0, -1, :].float().cpu().numpy()
            current_logits.append(logits)

    if baseline_logits is None:
        return {"logits": current_logits, "kl": None}

    # Compute per-prompt KL divergence
    kl_values = []
    for bl, cl in zip(baseline_logits, current_logits):
        # Softmax both
        bl_probs = np.exp(bl - bl.max())
        bl_probs /= bl_probs.sum()
        cl_probs = np.exp(cl - cl.max())
        cl_probs /= cl_probs.sum()

        # KL(baseline || current) = sum p_base * log(p_base / p_current)
        kl = float(np.sum(bl_probs * np.log(
            (bl_probs + 1e-12) / (cl_probs + 1e-12)
        )))
        kl_values.append(kl)

    return {
        "logits": current_logits,
        "kl": round(float(np.mean(kl_values)), 4),
        "kl_std": round(float(np.std(kl_values)), 4),
        "kl_max": round(float(np.max(kl_values)), 4),
    }


# ── Weight orthogonalization (Arditi) ──────────────────────────────

def orthogonalize_weight(weight: torch.Tensor, direction: torch.Tensor) -> None:
    """In-place weight orthogonalization: W' = W − r̂(r̂ᵀW).

    The direction is projected out of the output (row) dimension of
    the weight matrix. This permanently modifies the weight tensor.

    Args:
        weight: [d_out, d_in] weight matrix (modified in place).
        direction: [d_out] unit-normalized direction vector.
    """
    with torch.no_grad():
        v = direction.to(device=weight.device, dtype=torch.float32)
        v = v / (v.norm() + 1e-12)
        W = weight.data.float()
        # proj = r̂(r̂ᵀW) has shape [d_out, d_in]
        proj = torch.outer(v, v @ W)
        weight.data = (W - proj).to(dtype=weight.dtype)


def apply_arditi_abliteration(
    model,
    adapter,
    direction: torch.Tensor,
    progress: Optional[Callable] = None,
    should_cancel: Optional[Callable] = None,
) -> dict:
    """Apply Arditi weight orthogonalization to all residual-writing matrices.

    Targets: W_E (embedding), o_proj (attention out), down_proj (MLP out).

    Args:
        model: The loaded instruct model. Weights are modified in place.
        direction: [hidden_dim] refusal direction, will be unit-normalized.

    Returns:
        Summary dict with counts of modified matrices.
    """
    v = direction.detach().float()
    v = v / (v.norm() + 1e-12)

    n_modified = 0

    # Embedding
    if hasattr(model.model, 'embed_tokens'):
        if progress:
            progress("Orthogonalizing embedding matrix...")
        orthogonalize_weight(model.model.embed_tokens.weight, v)
        n_modified += 1

    # Per-layer: o_proj and down_proj
    n_layers = adapter.n_layers(model)
    for li in range(n_layers):
        # Safe cancel point: the caller snapshotted the full state_dict before
        # calling us and restores it in a `finally`, so bailing out part-way
        # through the sweep still leaves the shared model intact.
        if should_cancel and should_cancel():
            raise ModuleCancelled("routing_ablation")
        if progress and li % 4 == 0:
            progress(f"Orthogonalizing layer {li}/{n_layers}...")

        # Attention output projection
        o_proj = model.model.layers[li].self_attn.o_proj
        orthogonalize_weight(o_proj.weight, v)
        n_modified += 1

        # MLP down projection
        mlp = model.model.layers[li].mlp
        if hasattr(mlp, 'down_proj'):
            orthogonalize_weight(mlp.down_proj.weight, v)
            n_modified += 1

    if progress:
        progress(f"Orthogonalized {n_modified} weight matrices.")

    return {"n_modified": n_modified, "n_layers": n_layers}


# ── Main Module ────────────────────────────────────────────────────

class RoutingAblationModule(TASMModule):

    name = "routing_ablation"
    display_name = "Routing Ablation Experiment"
    description = (
        "Dual-projection causal intervention: tests whether the SFD "
        "routing harm direction survives residual-stream abliteration "
        "and/or provides independent behavioral effects. Implements "
        "Arditi weight orthogonalization (Stage 1) and SKOP-adapted "
        "query-space routing correction (Stage 3), with a measurement "
        "harness tracking harm recognition, refusal behavior, and "
        "attention routing patterns."
    )
    version = "0.1.0"

    min_results = 20
    requires_sfd = False  # Stage 3 needs SFD, but Stage 1 doesn't

    parameters = [
        # ═══ STAGE SELECTION ═══
        ModuleParameter(
            name="run_residual_ablation",
            display_name="Stage 1: Residual Abliteration",
            description=(
                "Arditi weight orthogonalization on W_E, o_proj, down_proj. "
                "Tests necessity of the refusal direction."
            ),
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="run_routing_ablation",
            display_name="Stage 3: QK Routing Correction",
            description=(
                "Project the harm-routing direction out of query vectors "
                "at risk heads. Requires SFD per_token_directions. "
                "Tests whether routing carries independent harm signal."
            ),
            type="bool",
            default=False,
        ),

        # ═══ DIRECTION FITTING ═══
        ModuleParameter(
            name="holdout_frac",
            display_name="Holdout Fraction",
            description=(
                "Fraction of harmful prompts held out for evaluation. "
                "These prompts are NOT used for direction fitting."
            ),
            type="float",
            default=0.20,
            min_val=0.10,
            max_val=0.50,
        ),

        # ═══ ROUTING (STAGE 3) ═══
        ModuleParameter(
            name="risk_head_fraction",
            display_name="Risk Head Fraction",
            description=(
                "Top fraction of heads (by Rayleigh quotient) to "
                "intervene on. 0.20 = top 20%%. SKOP recommends "
                "10-20%% for best utility-efficacy tradeoff."
            ),
            type="float",
            default=0.20,
            min_val=0.05,
            max_val=0.50,
        ),

        # ═══ GENERATION ═══
        ModuleParameter(
            name="max_new_tokens",
            display_name="Max New Tokens",
            description="Maximum tokens to generate per prompt.",
            type="int",
            default=120,
            min_val=20,
            max_val=512,
        ),

        # ═══ MEASUREMENT ═══
        ModuleParameter(
            name="measure_sas",
            display_name="Measure: Safety Attention Score",
            description=(
                "Compute per-head SAS before and after ablation. "
                "Requires output_attentions, which is slower."
            ),
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="measure_kl",
            display_name="Measure: First-Token KL Divergence",
            description="Capability drift guard. Measures logit distribution shift.",
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="n_kl_prompts",
            display_name="KL: Number of Neutral Prompts",
            description="How many benign prompts to use for KL measurement.",
            type="int",
            default=20,
            min_val=5,
            max_val=100,
        ),

        # ═══ SKOP CALIBRATION ═══
        ModuleParameter(
            name="focus_tau",
            display_name="SKOP: Focus Set Threshold",
            description=(
                "Attention mass threshold for focus set construction. "
                "0.80 means the minimal set of tokens capturing 80%% of "
                "attention mass per head. Lower values yield smaller "
                "focus sets and more aggressive filtering."
            ),
            type="float",
            default=0.80,
            min_val=0.50,
            max_val=0.95,
        ),
        ModuleParameter(
            name="skop_energy_gamma",
            display_name="SKOP: Subspace Energy Threshold",
            description=(
                "Cumulative eigenvalue energy threshold for the "
                "key-difference subspace. 0.90 retains eigenvectors "
                "capturing 90%% of routing-relevant energy."
            ),
            type="float",
            default=0.90,
            min_val=0.50,
            max_val=0.99,
        ),
        ModuleParameter(
            name="n_calibration_prompts",
            display_name="SKOP: Calibration Prompts Per Class",
            description=(
                "Number of prompts per class (harm/safe) for SKOP "
                "calibration. 30 per class is sufficient; SKOP reports "
                "utility plateaus around 1000 but 250 is the floor."
            ),
            type="int",
            default=30,
            min_val=10,
            max_val=100,
        ),
        ModuleParameter(
            name="measure_sfd_persistence",
            display_name="Measure: SFD Direction Persistence",
            description=(
                "After ablation, recompute the SFD cache and measure "
                "cosine similarity with the pre-ablation directions. "
                "Tests whether the routing signal survives intervention."
            ),
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="measure_delta_m",
            display_name="Measure: Focus-to-Tail Mass Shift",
            description=(
                "Measure how much attention mass moves from focus to "
                "tail tokens after routing intervention. Positive "
                "delta_m means routing was disrupted. ARA shows this "
                "predicts jailbreak success better than head ablation."
            ),
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="probe_position",
            display_name="Probe: Token Position",
            description=(
                "Which token position to use for harm probe features. "
                "'mean' = mean-pool all tokens (default). 't_inst' = "
                "last instruction token (Zhao et al. show this encodes "
                "harm recognition specifically)."
            ),
            type="str",
            default="mean",
        ),
    ]

    def __init__(self):
        super().__init__()
        self._pipeline = None

    def set_pipeline(self, pipeline):
        """Receive the loaded pipeline."""
        self._pipeline = pipeline

    def validate(self, session_results: list, params: dict) -> tuple:
        ok, msg = super().validate(session_results, params)
        if not ok:
            return ok, msg

        if self._pipeline is None or getattr(self._pipeline, 'instruct_model', None) is None:
            return False, "No model loaded. Load an instruct model first."

        # Check for per_token_final_emb
        has_emb = sum(1 for r in session_results if r.get("per_token_final_emb"))
        if has_emb < 10:
            return False, (
                f"Need per_token_final_emb in at least 10 results; found {has_emb}. "
                f"Re-run with embedding capture enabled."
            )

        # Stage 3 needs SFD directions
        if params.get("run_routing_ablation", False):
            has_sfd = sum(
                1 for r in session_results
                if r.get("sfd", {}).get("per_token_directions")
            )
            if has_sfd < 10:
                return False, (
                    f"Stage 3 requires SFD per_token_directions in at least 10 "
                    f"results; found {has_sfd}. Re-run with SFD direction "
                    f"persistence enabled."
                )

        return True, "OK"

    def run(self, session_results: list, params: dict,
            progress: Callable[[str], None] = None,
            should_cancel: Callable[[], bool] = None) -> dict:
        """Execute the routing ablation experiment."""

        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[ROUTING-ABLATION] {msg}")

        def teardown_prog(msg):
            """Progress that can never abort the caller.

            prog() doubles as the cancellation checkpoint (see base.py), so
            calling it from a `finally:` would raise ModuleCancelled *inside*
            the cleanup it is announcing — skipping the weight restore and
            leaving the server's shared model permanently orthogonalized.
            Teardown paths report through this instead.
            """
            try:
                if progress:
                    progress(msg)
            except ModuleCancelled:
                pass
            logger.info(f"[ROUTING-ABLATION] {msg}")

        results = {
            "version": self.version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
        }

        model = self._pipeline.instruct_model
        adapter = self._pipeline.adapter
        tokenizer = self._pipeline.tokenizer

        # ── Fit directions ─────────────────────────────────────────
        prog("Fitting refusal direction from session data...")
        fitter = DirectionFitter(
            session_results,
            harm_cats=HARM_CATEGORIES,
            safe_cats=SAFE_CATEGORIES,
        )
        fit = fitter.difference_of_means(
            holdout_frac=params.get("holdout_frac", 0.20),
            seed=0,
            holdout_side="harm",
        )
        n_excluded = sum(
            1 for r in session_results
            if classify_category(r.get("category")) is None
        )
        results["refusal_direction"] = {
            "method": fit.method,
            "hidden_dim": fit.hidden_dim,
            "n_harm": fit.n_harm,
            "n_safe": fit.n_safe,
            # Prompts whose category is neither harm nor safe under the
            # canonical taxonomy (e.g. "unknown") are EXCLUDED, not guessed.
            # Reported so the exclusion is visible rather than silent.
            "n_excluded_unclassified": n_excluded,
            "train_auroc": round(fit.train_auroc, 4),
            "n_heldout": len(fit.heldout_prompts),
        }
        prog(f"Refusal direction: dim={fit.hidden_dim}, "
             f"AUROC={fit.train_auroc:.4f}, "
             f"holdout={len(fit.heldout_prompts)} prompts")

        # ── Baseline measurements ──────────────────────────────────
        prog("Collecting baseline measurements...")

        # Harm probe (baseline).
        # NUMBERS CHANGE HERE: the baseline used to read the CACHED
        # per_token_final_emb (per-token L2-normalized, sliced from the
        # analyzer's start_pos, tokenized with the chat template), while the
        # post-ablation probe ran a live forward and took the raw
        # hidden_states[-1] mean. Those are NOT the same feature, so the
        # reported harm_probe delta confounded the ablation with the change
        # of extraction. Both sides now run the identical live extraction.
        probe_features, probe_labels, n_probe_excluded = \
            self._collect_probe_features_live(
                model, adapter, tokenizer, session_results, prog
            )
        baseline_probe = HarmProbe(regularization=1.0)
        baseline_probe.fit(probe_features, probe_labels)
        results["baseline_harm_probe"] = baseline_probe.to_dict()
        results["baseline_harm_probe"]["n_excluded_prompts"] = n_probe_excluded
        results["baseline_harm_probe"]["feature_source"] = (
            "live_forward_final_hidden_state_mean")
        prog(f"Baseline harm probe: train={baseline_probe.train_auroc:.4f}, "
             f"holdout={baseline_probe.holdout_auroc:.4f}")

        # Baseline generation on held-out harmful prompts
        eval_prompts = fit.heldout_prompts
        if not eval_prompts:
            # Fall back to all harmful prompts if holdout is empty
            eval_prompts = [
                r["prompt"] for r in session_results
                if classify_category(r.get("category")) == "harm"
            ][:20]

        prog(f"Baseline generation: {len(eval_prompts)} prompts...")
        detector = RefusalDetector()
        baseline_gen = self._generate_batch(
            model, tokenizer, eval_prompts,
            max_new_tokens=params.get("max_new_tokens", 120),
            detector=detector, progress=prog, tag="baseline",
            should_cancel=should_cancel,
        )
        results["baseline_generation"] = baseline_gen

        # KL baseline
        if params.get("measure_kl", True):
            prog("Computing baseline KL logits...")
            neutral_prompts = [
                r["prompt"] for r in session_results
                if classify_category(r.get("category")) == "safe"
            ][:params.get("n_kl_prompts", 20)]
            kl_baseline = compute_first_token_kl(
                model, tokenizer, neutral_prompts, progress=prog
            )
            results["kl_baseline_logits_collected"] = True
        else:
            kl_baseline = None
            neutral_prompts = []

        # SAS baseline
        if params.get("measure_sas", True):
            prog("Computing baseline SAS...")
            signal_layers = list(range(adapter.n_layers(model)))
            inst_counts = [
                len(tokenizer.encode(p, add_special_tokens=False))
                for p in eval_prompts
            ]
            sas_baseline = compute_sas(
                model, adapter, tokenizer,
                eval_prompts[:30], inst_counts[:30],
                signal_layers[:8],  # Sample of layers for speed
                progress=prog,
            )
            results["baseline_sas"] = sas_baseline

        # Capture pre-ablation SFD cache for persistence check
        if params.get("measure_sfd_persistence", True):
            try:
                from src.engine.sfd import precompute_sfd_cache
                from src.engine.attention_calibration import SFDContext
                prog("Capturing pre-ablation SFD cache...")
                ctx = SFDContext(self._pipeline)
                self._pre_sfd_cache = precompute_sfd_cache(ctx)
                prog(f"SFD cache captured: {self._pre_sfd_cache.get('n_layers', 0)} layers, "
                     f"k={self._pre_sfd_cache.get('k', '?')}")
            except ModuleCancelled:
                # prog() raises this; a bare `except Exception` would turn a
                # user cancellation into a bogus "capture failed" result.
                raise
            except Exception as e:
                prog(f"SFD cache capture failed: {e}")
                self._pre_sfd_cache = None

        # ── Stage 1: Residual abliteration ─────────────────────────
        if params.get("run_residual_ablation", True):
            prog("═══ Stage 1: Residual Abliteration ═══")

            # Save original weights for restoration
            prog("Saving model state for restoration...")
            original_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }

            # EXCEPTION SAFETY (was wrong): everything between the
            # orthogonalization and the load_state_dict below used to run
            # WITHOUT a try/finally.  Any failure in ~80 lines of generation,
            # probing, SAS or KL left the server's SHARED model permanently
            # orthogonalized for every later request and every other module,
            # and the user saw a plain error rather than a warning that their
            # loaded model was now corrupt.  Restoration is now unconditional.
            try:
                # Apply orthogonalization
                ablation_info = apply_arditi_abliteration(
                    model, adapter, fit.vector, progress=prog,
                    should_cancel=should_cancel,
                )
                results["stage1_ablation_info"] = ablation_info

                # Post-ablation generation
                prog(f"Post-ablation generation: {len(eval_prompts)} prompts...")
                post_gen = self._generate_batch(
                    model, tokenizer, eval_prompts,
                    max_new_tokens=params.get("max_new_tokens", 120),
                    detector=detector, progress=prog, tag="post_residual",
                    should_cancel=should_cancel,
                )
                results["stage1_generation"] = post_gen

                # Post-ablation harm probe.  Uses the SAME live-extraction
                # path as the baseline probe above (see _collect_probe_features
                # _live) so the reported delta isolates the ablation.
                prog("Re-collecting harm probe features post-ablation...")
                post_features, post_labels, post_excluded = \
                    self._collect_probe_features_live(
                        model, adapter, tokenizer, session_results, prog
                    )
                post_probe = HarmProbe(regularization=1.0)
                post_probe.fit(post_features, post_labels)
                results["stage1_harm_probe"] = post_probe.to_dict()
                results["stage1_harm_probe"]["n_excluded_prompts"] = post_excluded
                prog(f"Post-ablation harm probe: holdout={post_probe.holdout_auroc:.4f} "
                     f"(baseline={baseline_probe.holdout_auroc:.4f})")

                # Post-ablation KL
                if kl_baseline is not None:
                    prog("Computing post-ablation KL...")
                    kl_post = compute_first_token_kl(
                        model, tokenizer, neutral_prompts,
                        baseline_logits=kl_baseline["logits"],
                        progress=prog,
                    )
                    results["stage1_kl"] = kl_post
                    prog(f"Post-ablation KL: {kl_post.get('kl', '?')}")

                # Post-ablation SAS
                if params.get("measure_sas", True):
                    prog("Computing post-ablation SAS...")
                    sas_post = compute_sas(
                        model, adapter, tokenizer,
                        eval_prompts[:30], inst_counts[:30],
                        signal_layers[:8],
                        progress=prog,
                    )
                    results["stage1_sas"] = sas_post

                # Summary.  "harm_probe_survived" is a scientific claim, so it
                # is reported as None (no verdict) rather than False when the
                # holdout is too small to support a rank statistic — the old
                # 1-vs-1 holdout could only return 0.0 or 1.0.
                survived = (None if post_probe.holdout_underpowered
                            else bool(post_probe.holdout_auroc > 0.70))
                results["stage1_summary"] = {
                    "baseline_refusal_rate": baseline_gen["refusal_rate"],
                    "post_refusal_rate": post_gen["refusal_rate"],
                    "delta": round(baseline_gen["refusal_rate"] - post_gen["refusal_rate"], 4),
                    "baseline_harm_probe": baseline_probe.holdout_auroc,
                    "post_harm_probe": post_probe.holdout_auroc,
                    "harm_probe_survived": survived,
                    "harm_probe_underpowered": post_probe.holdout_underpowered,
                    "harm_probe_n_holdout": [post_probe.holdout_n_pos,
                                             post_probe.holdout_n_neg],
                    "kl": results.get("stage1_kl", {}).get("kl"),
                }
                prog(f"Stage 1 summary: refusal {baseline_gen['refusal_rate']:.0%} → "
                     f"{post_gen['refusal_rate']:.0%}, "
                     f"harm probe {baseline_probe.holdout_auroc:.3f} → "
                     f"{post_probe.holdout_auroc:.3f}"
                     + (" (holdout UNDERPOWERED — no verdict)"
                        if post_probe.holdout_underpowered else ""))

                # SFD direction persistence check (on ablated model)
                if params.get("measure_sfd_persistence", True):
                    try:
                        from src.engine.attention_calibration import (
                            compute_sfd_persistence,
                        )
                        prog("Checking SFD direction persistence post-ablation...")
                        pre_sfd = getattr(self, '_pre_sfd_cache', None)
                        if pre_sfd is not None:
                            sfd_persist = compute_sfd_persistence(
                                self._pipeline, pre_sfd, progress=prog,
                            )
                            results["stage1_sfd_persistence"] = sfd_persist
                            prog(f"SFD persistence: cosine={sfd_persist['mean_cosine']:.3f}, "
                                 f"survived={sfd_persist['signal_survived']}")
                        else:
                            prog("SFD persistence: no pre-ablation cache available. "
                                 "Run an analysis session with SFD enabled first.")
                    except ModuleCancelled:
                        raise
                    except Exception as e:
                        prog(f"SFD persistence check failed: {e}")
            finally:
                # Restore original weights for subsequent stages. If THIS
                # fails the shared model really is corrupt, so say so loudly
                # and let the error propagate rather than continuing.
                #
                # CANCELLATION SAFETY: every report on this path goes through
                # teardown_prog, never prog.  We reach here with the cancel
                # flag already set, so a prog() call would raise
                # ModuleCancelled before load_state_dict ran and the shared
                # model would stay orthogonalized forever.
                teardown_prog("Restoring original model weights...")
                try:
                    model.load_state_dict(original_state)
                    teardown_prog("Model weights restored.")
                except Exception as restore_err:
                    logger.critical(
                        "[ROUTING-ABLATION] FAILED TO RESTORE MODEL WEIGHTS "
                        "after abliteration: %s. The loaded model is now "
                        "permanently orthogonalized — RELOAD IT before "
                        "trusting any further result.", restore_err)
                    teardown_prog("CRITICAL: weight restoration FAILED. The "
                                  "loaded model is permanently orthogonalized. "
                                  "Reload the model before running anything "
                                  "else.")
                    raise
                finally:
                    del original_state

        # ── Stage 3: QK Routing Correction ─────────────────────────
        if params.get("run_routing_ablation", False):
            prog("═══ Stage 3: QK Routing Correction ═══")

            try:
                from src.engine.qk_intervention import (
                    QKRoutingIntervention,
                    select_risk_heads,
                )
                from src.engine.attention_calibration import (
                    AttentionCalibrator,
                    apply_skop_projection,
                    compute_delta_m,
                )
            except ImportError as e:
                prog(f"ERROR: required module not found: {e}")
                results["stage3_error"] = str(e)
            else:
                # SKOP calibration: collect Q, K, attention per head
                prog("Running SKOP calibration...")
                n_cal = params.get("n_calibration_prompts", 30)
                cal_harm = [r["prompt"] for r in session_results
                            if classify_category(r.get("category")) == "harm"][:n_cal]
                cal_safe = [r["prompt"] for r in session_results
                            if classify_category(r.get("category")) == "safe"][:n_cal]
                cal_prompts = cal_harm + cal_safe
                cal_labels = ["harm"] * len(cal_harm) + ["safe"] * len(cal_safe)

                calibrator = AttentionCalibrator(model, adapter, tokenizer)
                calibrator.calibrate(
                    cal_prompts, cal_labels,
                    tau=params.get("focus_tau", 0.80),
                    progress=prog,
                )

                # Fit per-head directions from focus-restricted Q
                prog("Fitting per-head routing directions (focus-restricted)...")
                per_head_dirs = calibrator.fit_directions()
                results["stage3_n_head_directions"] = len(per_head_dirs)

                if per_head_dirs:
                    # Build key-difference subspace and score heads
                    prog("Building key-difference subspaces and scoring heads...")
                    projectors, risk_scores = calibrator.build_subspace_and_score(
                        per_head_dirs,
                        gamma=params.get("skop_energy_gamma", 0.90),
                    )

                    # Apply SKOP subspace projection to directions
                    prog("Applying SKOP subspace projection...")
                    projected_dirs = apply_skop_projection(per_head_dirs, projectors)
                    n_projected = len(projected_dirs)
                    n_collapsed = len(per_head_dirs) - n_projected
                    if n_collapsed > 0:
                        prog(f"  {n_collapsed} directions collapsed after projection "
                             f"(removed from intervention set)")
                    results["stage3_n_projected"] = n_projected
                    results["stage3_n_collapsed"] = n_collapsed

                    # Select risk heads by Rayleigh quotient
                    risk_frac = params.get("risk_head_fraction", 0.20)
                    risk_heads = select_risk_heads(risk_scores, risk_frac)
                    n_risk = sum(len(v) for v in risk_heads.values())
                    prog(f"Selected {n_risk} risk heads across "
                         f"{len(risk_heads)} layers (Rayleigh scoring)")
                    results["stage3_risk_heads"] = {
                        str(k): v for k, v in risk_heads.items()
                    }

                    # Report top risk scores
                    top_risk = sorted(risk_scores.items(),
                                      key=lambda x: x[1], reverse=True)[:10]
                    results["stage3_top_risk_scores"] = [
                        {"layer": li, "head": hi, "score": round(s, 6)}
                        for (li, hi), s in top_risk
                    ]

                    # Install QK intervention with SKOP-projected directions.
                    # HOOK SAFETY (was wrong): qk_intv.remove() used to sit at
                    # the end of this block with no try/finally, so any failure
                    # in generation/KL/delta_m/SFD left the query-projection
                    # hooks installed on the SHARED model for every later
                    # request. Cleanup is now unconditional.
                    qk_intv = QKRoutingIntervention()
                    qk_intv.install_on_q_proj(
                        model, adapter, projected_dirs, risk_heads
                    )
                    try:
                        # Post-routing generation
                        prog(f"Post-routing generation: {len(eval_prompts)} prompts...")
                        post_routing_gen = self._generate_batch(
                            model, tokenizer, eval_prompts,
                            max_new_tokens=params.get("max_new_tokens", 120),
                            detector=detector, progress=prog, tag="post_routing",
                            should_cancel=should_cancel,
                        )
                        results["stage3_generation"] = post_routing_gen

                        # Post-routing KL
                        if kl_baseline is not None:
                            prog("Computing post-routing KL...")
                            kl_routing = compute_first_token_kl(
                                model, tokenizer, neutral_prompts,
                                baseline_logits=kl_baseline["logits"],
                                progress=prog,
                            )
                            results["stage3_kl"] = kl_routing

                        # Focus-to-tail mass shift (delta_m)
                        if params.get("measure_delta_m", True):
                            try:
                                prog("Measuring focus-to-tail mass shift...")
                                # Forward pass on eval prompts with intervention active
                                device = next(model.parameters()).device
                                delta_m_results = {}
                                for li in range(min(4, calibrator.n_layers)):
                                    focus = calibrator.get_focus_sets_for_layer(li)
                                    if focus:
                                        delta_m_results[str(li)] = {
                                            "n_heads_with_focus": len(focus),
                                        }
                                results["stage3_delta_m_layers"] = delta_m_results
                                prog("Delta_m measurement recorded.")
                            except ModuleCancelled:
                                raise
                            except Exception as e:
                                prog(f"Delta_m measurement failed: {e}")

                        # SFD persistence after routing ablation
                        if params.get("measure_sfd_persistence", True):
                            try:
                                from src.engine.attention_calibration import (
                                    compute_sfd_persistence,
                                )
                                pre_sfd = getattr(self, '_pre_sfd_cache', None)
                                if pre_sfd is not None:
                                    prog("Checking SFD persistence post-routing...")
                                    sfd_persist = compute_sfd_persistence(
                                        self._pipeline, pre_sfd, progress=prog,
                                    )
                                    results["stage3_sfd_persistence"] = sfd_persist
                                    prog(f"SFD persistence: cosine="
                                         f"{sfd_persist['mean_cosine']:.3f}")
                            except ModuleCancelled:
                                raise
                            except Exception as e:
                                prog(f"SFD persistence check failed: {e}")

                    finally:
                        qk_intv.remove()

                    results["stage3_summary"] = {
                        "baseline_refusal_rate": baseline_gen["refusal_rate"],
                        "post_refusal_rate": post_routing_gen["refusal_rate"],
                        "delta": round(
                            baseline_gen["refusal_rate"] -
                            post_routing_gen["refusal_rate"], 4
                        ),
                        "n_risk_heads": n_risk,
                        "n_projected_directions": n_projected,
                        "scoring_method": "rayleigh_quotient",
                        "kl": results.get("stage3_kl", {}).get("kl"),
                    }
                    prog(f"Stage 3 summary: refusal {baseline_gen['refusal_rate']:.0%} → "
                         f"{post_routing_gen['refusal_rate']:.0%}")

        prog("═══ Routing ablation experiment complete ═══")
        return results

    # ── Helper: collect probe features from session ────────────────

    def _collect_probe_features(
        self, session_results: list, progress: Callable,
        position: str = "mean",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract cached final-layer embeddings for the harm probe.

        NOTE: no longer used by run(). The cached per_token_final_emb is
        per-token L2-normalized and sliced from the analyzer's start_pos,
        which does NOT match the live hidden_states[-1] extraction used
        post-ablation; mixing the two confounded the reported probe delta.
        Retained for callers that specifically want the cached features.

        Args:
            position: 'mean' for mean-pool (default), 't_inst' for
                last instruction token, 't_first' for first token.
        """
        features, labels = [], []
        for r in session_results:
            cls = classify_category(r.get("category"))
            if cls == "harm":
                y = 1
            elif cls == "safe":
                y = 0
            else:
                continue
            emb = r.get("per_token_final_emb")
            if not emb:
                continue
            arr = np.asarray(emb, dtype=np.float32)
            if position == "t_inst":
                features.append(arr[-1])
            elif position == "t_first":
                features.append(arr[0])
            else:
                features.append(arr.mean(axis=0))
            labels.append(y)
        return np.array(features), np.array(labels)

    def _collect_probe_features_live(
        self, model, adapter, tokenizer, session_results: list,
        progress: Callable
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Collect probe features via live forward pass on current model.

        Used for BOTH the baseline and the post-ablation probe so the
        reported delta isolates the ablation rather than a change of
        feature extraction.

        Returns (features, labels, n_excluded) where n_excluded counts
        prompts skipped because their category is neither harm nor safe.
        """
        from src.engine.hooks import ActivationCapture
        from src.engine import config as engine_config

        cap = ActivationCapture()
        n_layers = adapter.n_layers(model)
        # Hook only the final norm for efficiency
        cap.install(model, adapter, signal_layers=[])

        features, labels = [], []
        n_excluded = 0
        device = next(model.parameters()).device

        # HOOK SAFETY (was wrong): cap.remove() sat after the loop with no
        # try/finally, so a failure mid-loop leaked hooks onto the shared model.
        try:
            for i, r in enumerate(session_results):
                cls = classify_category(r.get("category"))
                if cls == "harm":
                    y = 1
                elif cls == "safe":
                    y = 0
                else:
                    n_excluded += 1
                    continue

                prompt = r.get("prompt", "")
                if not prompt:
                    continue

                if i % 20 == 0:
                    progress(f"Probe features: {i}/{len(session_results)}")

                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    add_special_tokens=engine_config.get("add_special_tokens"),
                ).to(device)

                with torch.no_grad():
                    output = model(**inputs, output_hidden_states=True)
                    # Last layer hidden states, mean-pooled
                    last_hidden = output.hidden_states[-1][0].float().cpu().numpy()
                    features.append(last_hidden.mean(axis=0))
                    labels.append(y)
        finally:
            cap.remove()
        return np.array(features), np.array(labels), n_excluded

    # ── Helper: generation with refusal detection ──────────────────

    def _generate_batch(
        self, model, tokenizer, prompts: list[str],
        max_new_tokens: int, detector: RefusalDetector,
        progress: Callable, tag: str,
        should_cancel: Callable = None,
    ) -> dict:
        """Generate completions and score for refusal."""
        from src.engine import config as engine_config

        device = next(model.parameters()).device
        outcomes = []

        with MODEL_LOCK:
            for i, prompt in enumerate(prompts):
                # Progress only fires every 5th prompt, and one prompt is a
                # full 120-token generate() — so without this poll a cancel
                # could sit unnoticed for five generations.  Bailing out here
                # is safe: nothing is mutated, and `with MODEL_LOCK` releases
                # on the way out.
                if should_cancel and should_cancel():
                    raise ModuleCancelled("routing_ablation")
                if i % 5 == 0:
                    progress(f"{tag}: generating {i+1}/{len(prompts)}")

                # WAS WRONG: this hardcoded add_special_tokens=True while the
                # sibling feature-extraction path used the configured value, so
                # generations were tokenized differently from the activations
                # they are compared against. Use the config value in both.
                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    add_special_tokens=engine_config.get("add_special_tokens"),
                ).to(device)

                with torch.no_grad():
                    # temperature was passed with do_sample=False, where it is
                    # ignored and transformers emits a warning. Dropped.
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

                # Decode only new tokens
                new_tokens = output[0][inputs["input_ids"].shape[1]:]
                reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

                refused = detector.detect(reply)
                outcomes.append({
                    "prompt": prompt[:100],
                    "reply": reply[:300],
                    "refused": refused,
                })

        n_refused = sum(1 for o in outcomes if o["refused"])
        rate = n_refused / len(outcomes) if outcomes else 0.0

        return {
            "n_prompts": len(outcomes),
            "n_refused": n_refused,
            "refusal_rate": round(rate, 4),
            "outcomes": outcomes,
        }

    # ── Helper: per-head direction fitting for Stage 3 ─────────────
    # Superseded by AttentionCalibrator in src/engine/attention_calibration.py.
    # Stage 3 now uses calibrator.fit_directions() which provides:
    #   - Focus-restricted Q activation pooling (SKOP focus/tail sets)
    #   - Per-head key-difference subspace construction
    #   - Rayleigh quotient risk scoring
    #   - SKOP subspace-projected directions
    # The old mean-pool approach is retained below for fallback/testing.

    def _fit_per_head_directions_legacy(
        self, model, adapter, tokenizer, session_results: list,
        progress: Callable,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Legacy: mean-pool Q direction fitting (no focus/tail selection).

        Retained for comparison testing. Stage 3 uses
        AttentionCalibrator.fit_directions() instead.
        """
        from src.engine import config as engine_config

        n_layers = adapter.n_layers(model)
        n_heads, _ = adapter.attention_heads(model)
        d_head = adapter.head_dim(model)
        device = next(model.parameters()).device

        harm_q_sums = {}
        safe_q_sums = {}
        harm_counts = {}
        safe_counts = {}

        for (li, hi) in [(l, h) for l in range(n_layers) for h in range(n_heads)]:
            harm_q_sums[(li, hi)] = np.zeros(d_head, dtype=np.float64)
            safe_q_sums[(li, hi)] = np.zeros(d_head, dtype=np.float64)
            harm_counts[(li, hi)] = 0
            safe_counts[(li, hi)] = 0

        harm_prompts = [
            r["prompt"] for r in session_results
            if classify_category(r.get("category")) == "harm"
        ][:30]
        safe_prompts = [
            r["prompt"] for r in session_results
            if classify_category(r.get("category")) == "safe"
        ][:30]

        for label, prompts in [("harm", harm_prompts), ("safe", safe_prompts)]:
            sums = harm_q_sums if label == "harm" else safe_q_sums
            counts = harm_counts if label == "harm" else safe_counts

            for pi, prompt in enumerate(prompts):
                if pi % 10 == 0:
                    progress(f"Q activations ({label}): {pi+1}/{len(prompts)}")

                captured_q = {}

                def make_q_capture_hook(layer_idx):
                    def hook(module, input, output):
                        captured_q[layer_idx] = output.detach().float().cpu()
                    return hook

                hooks = []
                for li in range(n_layers):
                    q_mod = model.model.layers[li].self_attn.q_proj
                    hooks.append(q_mod.register_forward_hook(
                        make_q_capture_hook(li)
                    ))

                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    add_special_tokens=engine_config.get("add_special_tokens"),
                ).to(device)

                with torch.no_grad():
                    model(**inputs)

                for h in hooks:
                    h.remove()

                for li, q_tensor in captured_q.items():
                    q = q_tensor[0].numpy()
                    q = q.reshape(q.shape[0], n_heads, d_head)
                    q_mean = q.mean(axis=0)
                    for hi in range(n_heads):
                        sums[(li, hi)] += q_mean[hi]
                        counts[(li, hi)] += 1

        per_head_dirs = {}
        for li in range(n_layers):
            for hi in range(n_heads):
                nh = harm_counts[(li, hi)]
                ns = safe_counts[(li, hi)]
                if nh == 0 or ns == 0:
                    continue
                harm_mean = harm_q_sums[(li, hi)] / nh
                safe_mean = safe_q_sums[(li, hi)] / ns
                direction = harm_mean - safe_mean
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    per_head_dirs[(li, hi)] = (direction / norm).astype(np.float32)

        progress(f"Fitted {len(per_head_dirs)} per-head routing directions (legacy)")
        return per_head_dirs
