"""Concept Atom Module: SRA-style diagnostic and cleaning for refusal directions.

Standalone exploration module that:
  1. Loads a Concept Atom Registry from CSV.
  2. Computes per-atom directions (difference-of-means) across layers.
  3. Computes the raw refusal direction from session data.
  4. Produces an orthogonality heatmap (atoms vs layers vs cosine)
     as a go/no-go diagnostic for ablation safety.
  5. Optionally runs the SRA ridge-regression cleaning step to
     produce a cleaned refusal direction.
  6. Persists the cleaned direction and diagnostics as a JSON
     artifact that the routing ablation module can consume.

Based on SRA (Cristofano, arXiv:2601.08489).

Does not modify weights by default. Does not install persistent hooks.
Does not import from or depend on any other TAGM module.
The routing ablation module reads this module's artifact from disk
if available; otherwise it falls back to raw difference-of-means.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch

from src.engine import config as engine_config
from src.core.locks import MODEL_LOCK
from .base import TASMModule, ModuleParameter

if TYPE_CHECKING:
    pass

logger = logging.getLogger("src")

VALID_ROLES = {"shield", "confound", "target"}
VALID_SIDES = {"positive", "negative"}


# ── CSV Parsing ────────────────────────────────────────────────────

def parse_atom_registry(csv_text: str) -> dict[str, dict]:
    """Parse a concept atom registry from CSV text.

    Returns:
        {atom_name: {"role": str, "positive": [str], "negative": [str]}}
    """
    atoms: dict[str, dict] = {}
    reader = csv.reader(io.StringIO(csv_text))

    for row_num, row in enumerate(reader, 1):
        # Skip empty lines and comments
        if not row or row[0].strip().startswith("#"):
            continue

        # Skip header row if present
        if row_num == 1 and row[0].strip().lower() == "atom_name":
            continue

        if len(row) < 4:
            logger.warning(f"[CAM] Row {row_num}: expected 4 columns, got {len(row)}. Skipping.")
            continue

        name = row[0].strip()
        role = row[1].strip().lower()
        side = row[2].strip().lower()
        prompt = row[3].strip()

        if not name or not prompt:
            continue

        if role not in VALID_ROLES:
            logger.warning(f"[CAM] Row {row_num}: unknown role '{role}'. "
                           f"Expected one of {VALID_ROLES}. Skipping.")
            continue

        if side not in VALID_SIDES:
            logger.warning(f"[CAM] Row {row_num}: unknown side '{side}'. "
                           f"Expected 'positive' or 'negative'. Skipping.")
            continue

        if name not in atoms:
            atoms[name] = {"role": role, "positive": [], "negative": []}

        # Validate consistent role
        if atoms[name]["role"] != role:
            logger.warning(f"[CAM] Row {row_num}: atom '{name}' has inconsistent "
                           f"role '{role}' (expected '{atoms[name]['role']}'). Skipping.")
            continue

        atoms[name][side].append(prompt)

    return atoms


def validate_registry(atoms: dict[str, dict]) -> tuple[bool, str]:
    """Validate a parsed atom registry."""
    if not atoms:
        return False, "Registry is empty. Check CSV format."

    issues = []
    for name, data in atoms.items():
        n_pos = len(data["positive"])
        n_neg = len(data["negative"])
        if n_pos < 2:
            issues.append(f"'{name}': only {n_pos} positive prompts (need at least 2)")
        if n_neg < 2:
            issues.append(f"'{name}': only {n_neg} negative prompts (need at least 2)")

    roles = {d["role"] for d in atoms.values()}
    if "shield" not in roles:
        issues.append("No Shield atoms defined. Add at least one capability atom "
                      "(e.g., Logic, Math, Coding) to detect entanglement.")

    if issues:
        return False, "Registry issues: " + "; ".join(issues)

    return True, "OK"


# ── Atom Direction Computation ─────────────────────────────────────

def compute_atom_directions(
    model,
    adapter,
    tokenizer,
    atoms: dict[str, dict],
    layer_range: list[int],
    progress: Optional[Callable] = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Compute per-atom, per-layer direction vectors via difference-of-means.

    Returns:
        {atom_name: {layer_idx: unit_direction_vector}}
    """
    from src.engine.hooks import ActivationCapture

    device = next(model.parameters()).device
    n_layers = adapter.n_layers(model)
    directions: dict[str, dict[int, np.ndarray]] = {}

    total_prompts = sum(len(d["positive"]) + len(d["negative"]) for d in atoms.values())
    prompt_count = 0

    with MODEL_LOCK:
        for atom_name, data in atoms.items():
            pos_means = defaultdict(list)  # layer -> [activation vectors]
            neg_means = defaultdict(list)

            for side, prompts in [("positive", data["positive"]),
                                  ("negative", data["negative"])]:
                accum = pos_means if side == "positive" else neg_means

                for prompt in prompts:
                    prompt_count += 1
                    if progress and prompt_count % 20 == 0:
                        progress(f"Atom directions: {prompt_count}/{total_prompts}")

                    inputs = tokenizer(
                        prompt, return_tensors="pt",
                        add_special_tokens=engine_config.get("add_special_tokens"),
                    ).to(device)

                    with torch.no_grad():
                        output = model(**inputs, output_hidden_states=True)

                    if hasattr(output, "hidden_states") and output.hidden_states:
                        for li in layer_range:
                            if li < len(output.hidden_states):
                                h = output.hidden_states[li][0].float().cpu().numpy()
                                # Mean-pool across sequence
                                accum[li].append(h.mean(axis=0))

            # Compute direction per layer
            atom_dirs = {}
            for li in layer_range:
                if li not in pos_means or li not in neg_means:
                    continue
                if len(pos_means[li]) < 2 or len(neg_means[li]) < 2:
                    continue
                pos_mean = np.mean(pos_means[li], axis=0).astype(np.float64)
                neg_mean = np.mean(neg_means[li], axis=0).astype(np.float64)
                direction = pos_mean - neg_mean
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    atom_dirs[li] = (direction / norm).astype(np.float32)

            directions[atom_name] = atom_dirs

    if progress:
        progress(f"Computed directions for {len(directions)} atoms "
                 f"across {len(layer_range)} layers")
    return directions


# ── Orthogonality Diagnostics ──────────────────────────────────────

def compute_orthogonality_map(
    refusal_directions: dict[int, np.ndarray],
    atom_directions: dict[str, dict[int, np.ndarray]],
    layer_range: list[int],
) -> dict[int, dict[str, float]]:
    """Compute cosine similarity between refusal direction and each atom per layer.

    Returns:
        {layer_idx: {atom_name: cosine_similarity}}
    """
    ortho_map = {}
    for li in layer_range:
        r = refusal_directions.get(li)
        if r is None:
            continue
        r = r.astype(np.float64)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-10:
            continue
        r = r / r_norm

        layer_cosines = {}
        for atom_name, atom_dirs in atom_directions.items():
            a = atom_dirs.get(li)
            if a is None:
                continue
            a = a.astype(np.float64)
            a_norm = np.linalg.norm(a)
            if a_norm < 1e-10:
                continue
            layer_cosines[atom_name] = float(np.dot(r, a / a_norm))

        ortho_map[li] = layer_cosines

    return ortho_map


def summarize_entanglement(
    ortho_map: dict[int, dict[str, float]],
    atoms: dict[str, dict],
) -> dict:
    """Summarize entanglement from the orthogonality map."""
    shield_cosines = []
    confound_cosines = []
    target_cosines = []

    for li, cosines in ortho_map.items():
        for atom_name, cos in cosines.items():
            role = atoms.get(atom_name, {}).get("role", "")
            if role == "shield":
                shield_cosines.append(abs(cos))
            elif role == "confound":
                confound_cosines.append(abs(cos))
            elif role == "target":
                target_cosines.append(abs(cos))

    max_shield = max(shield_cosines) if shield_cosines else 0.0
    mean_shield = float(np.mean(shield_cosines)) if shield_cosines else 0.0

    if max_shield < 0.10:
        recommendation = ("Direction appears clean. Shield atom entanglement is low. "
                          "Cleaning may not be necessary.")
    elif max_shield < 0.20:
        recommendation = ("Moderate capability entanglement detected. "
                          "Cleaning is recommended to reduce drift risk.")
    else:
        recommendation = ("Significant capability entanglement detected. "
                          "Cleaning is strongly recommended before ablation.")

    return {
        "max_shield_cosine": round(max_shield, 4),
        "mean_shield_cosine": round(mean_shield, 4),
        "max_confound_cosine": round(max(confound_cosines) if confound_cosines else 0.0, 4),
        "mean_target_cosine": round(float(np.mean(target_cosines)) if target_cosines else 0.0, 4),
        "recommendation": recommendation,
    }


# ── Ridge-Regression Cleaning ─────────────────────────────────────

def clean_direction(
    r_dirty: np.ndarray,
    atom_matrix_sc: np.ndarray,
    ridge_lambda: float = 1.0,
) -> np.ndarray:
    """SRA ridge-regression cleaning step.

    Projects the dirty refusal direction onto the null space of the
    Shield+Confound atom subspace (approximately, via ridge regularization).

    Args:
        r_dirty: [d_model] dirty refusal direction.
        atom_matrix_sc: [d_model, K_SC] matrix of Shield+Confound atom
            directions as columns.
        ridge_lambda: regularization strength.

    Returns:
        r_clean: [d_model] cleaned direction, unit-normalized.
    """
    A = atom_matrix_sc.astype(np.float64)
    r = r_dirty.astype(np.float64)

    # Ridge solution: w_hat = (A^T A + lambda I)^-1 A^T r
    AtA = A.T @ A
    reg = ridge_lambda * np.eye(AtA.shape[0])
    try:
        w_hat = np.linalg.solve(AtA + reg, A.T @ r)
    except np.linalg.LinAlgError:
        logger.warning("[CAM] Ridge solve failed (singular matrix). "
                       "Returning dirty direction unchanged.")
        return r_dirty

    # Clean: r_tilde = r - A w_hat
    r_clean = r - A @ w_hat
    norm = np.linalg.norm(r_clean)
    if norm < 1e-10:
        logger.warning("[CAM] Cleaned direction has near-zero norm. "
                       "Ridge step may have removed too much signal. "
                       "Try increasing lambda.")
        return r_dirty

    return (r_clean / norm).astype(np.float32)


def build_sc_matrix(
    atom_directions: dict[str, dict[int, np.ndarray]],
    atoms: dict[str, dict],
    layer_idx: int,
) -> np.ndarray:
    """Build the Shield+Confound atom matrix A_SC for a given layer.

    Returns:
        [d_model, K_SC] matrix with Shield and Confound atom directions
        as columns. Target atoms are excluded.
    """
    columns = []
    for atom_name, atom_dirs in atom_directions.items():
        role = atoms.get(atom_name, {}).get("role", "")
        if role in ("shield", "confound"):
            d = atom_dirs.get(layer_idx)
            if d is not None:
                columns.append(d.astype(np.float64))

    if not columns:
        return np.zeros((0, 0), dtype=np.float64)

    return np.column_stack(columns)


# ── Gamma Calibration ──────────────────────────────────────────────

def calibrate_gamma(
    atom_directions: dict[str, dict[int, np.ndarray]],
    atoms: dict[str, dict],
    layer_range: list[int],
    mode: str = "fixed",
    fixed_value: float = 1.0,
) -> dict[int, float]:
    """Compute per-layer gamma scaling factor.

    Args:
        mode: "fixed" or "semantic_energy".
        fixed_value: gamma value when mode is "fixed".

    Returns:
        {layer_idx: gamma}
    """
    if mode == "fixed":
        return {li: fixed_value for li in layer_range}

    # Semantic energy proxy: gamma proportional to the largest Target atom norm
    target_norms = defaultdict(float)
    for atom_name, atom_dirs in atom_directions.items():
        role = atoms.get(atom_name, {}).get("role", "")
        if role != "target":
            continue
        for li in layer_range:
            d = atom_dirs.get(li)
            if d is not None:
                norm = float(np.linalg.norm(d.astype(np.float64)))
                target_norms[li] = max(target_norms[li], norm)

    if not target_norms:
        logger.warning("[CAM] No Target atoms found for semantic energy proxy. "
                       "Falling back to fixed gamma.")
        return {li: fixed_value for li in layer_range}

    # Normalize to [0.5, 1.0] range
    max_norm = max(target_norms.values())
    if max_norm < 1e-10:
        return {li: fixed_value for li in layer_range}

    gamma = {}
    for li in layer_range:
        n = target_norms.get(li, 0.0)
        gamma[li] = 0.5 + 0.5 * (n / max_norm)

    return gamma


# ── Module ─────────────────────────────────────────────────────────

class ConceptAtomModule(TASMModule):

    name = "concept_atoms"
    display_name = "Concept Atom Explorer"
    description = (
        "SRA-style diagnostic tool for refusal direction quality. "
        "Loads a Concept Atom Registry (CSV of contrastive prompt pairs "
        "labeled as Shield, Confound, or Target), computes per-atom "
        "directions, and measures entanglement between the refusal "
        "direction and capability atoms. Optionally cleans the refusal "
        "direction via ridge regression to remove capability bleed "
        "(Ghost Noise). Produces a JSON artifact that the routing "
        "ablation module can consume for cleaner ablation."
    )

    parameters = [
        # ═══ REGISTRY ═══
        ModuleParameter(
            name="registry_csv",
            display_name="Atom Registry CSV",
            description=(
                "Paste or upload the concept atom registry as CSV text. "
                "Four columns: atom_name, role (shield/confound/target), "
                "side (positive/negative), prompt. Each atom needs at "
                "least 2 prompts per side; 10-15 per side is recommended. "
                "A starter template is available in src/templates/."
            ),
            type="str",
            default="",
        ),

        # ═══ ANALYSIS ═══
        ModuleParameter(
            name="layer_start",
            display_name="Layer Range: Start",
            description=(
                "First layer for atom and refusal direction analysis. "
                "For 16-layer models (Llama 3.2-1B), try 4. "
                "For 28+ layer models (Qwen3-VL), try 15. "
                "Refusal geometry concentrates in the middle-to-upper "
                "third of the model."
            ),
            type="int",
            default=4,
            min_val=0,
            max_val=64,
        ),
        ModuleParameter(
            name="layer_end",
            display_name="Layer Range: End",
            description=(
                "Last layer (exclusive) for analysis. "
                "For 16-layer models, try 14. "
                "For 28+ layer models, try 25. "
                "Avoid the final 1-2 layers where representations "
                "collapse toward the unembedding."
            ),
            type="int",
            default=14,
            min_val=1,
            max_val=64,
        ),
        ModuleParameter(
            name="refusal_holdout_frac",
            display_name="Refusal Direction: Holdout Fraction",
            description=(
                "Fraction of harmful prompts held out when fitting "
                "the refusal direction. Not used for direction fitting; "
                "available for validation."
            ),
            type="float",
            default=0.20,
            min_val=0.0,
            max_val=0.50,
        ),

        # ═══ CLEANING ═══
        ModuleParameter(
            name="run_cleaning",
            display_name="Run SRA Cleaning",
            description=(
                "Run the ridge-regression cleaning step to remove "
                "capability bleed from the refusal direction. If False, "
                "only diagnostics are produced (orthogonality map, atom "
                "statistics). Turn this off for a quick diagnostic pass."
            ),
            type="bool",
            default=True,
        ),
        ModuleParameter(
            name="ridge_lambda",
            display_name="Ridge Lambda",
            description=(
                "Ridge regularization strength for the cleaning step. "
                "Higher values produce more conservative cleaning (less "
                "removal of entangled components). Start with 1.0. "
                "Decrease if Shield cosines remain above 0.05 after "
                "cleaning. Increase if the cleaned direction loses too "
                "much refusal signal."
            ),
            type="float",
            default=1.0,
            min_val=0.01,
            max_val=100.0,
        ),

        # ═══ GAMMA ═══
        ModuleParameter(
            name="gamma_mode",
            display_name="Gamma: Scaling Mode",
            description=(
                "How to scale the ablation strength per layer. "
                "'fixed' uses gamma_value for all layers. "
                "'semantic_energy' scales proportionally to the largest "
                "Target atom norm at each layer (SRA's Semantic Energy "
                "Proxy). Use 'fixed' with gamma=1.0 for full "
                "orthogonalization (Arditi-equivalent). Use gamma < 1.0 "
                "for partial ablation."
            ),
            type="str",
            default="fixed",
        ),
        ModuleParameter(
            name="gamma_value",
            display_name="Gamma: Fixed Value",
            description=(
                "Ablation strength when gamma_mode is 'fixed'. "
                "1.0 = full orthogonalization (Arditi). "
                "0.5-0.9 = partial ablation (attenuate, do not fully "
                "remove). Only used when gamma_mode is 'fixed'."
            ),
            type="float",
            default=1.0,
            min_val=0.1,
            max_val=1.0,
        ),

        # ═══ PREVIEW ═══
        ModuleParameter(
            name="run_preview",
            display_name="Preview: Test Cleaned Direction",
            description=(
                "After cleaning, install the cleaned direction as "
                "temporary inference-time hooks and generate on a small "
                "set of harmful prompts. Shows the effect of the cleaned "
                "ablation without touching weights. Hooks are removed "
                "after the test. The model is unchanged."
            ),
            type="bool",
            default=False,
        ),
        ModuleParameter(
            name="preview_n_prompts",
            display_name="Preview: Number of Prompts",
            description="Harmful prompts to generate on during preview.",
            type="int",
            default=10,
            min_val=3,
            max_val=50,
        ),
        ModuleParameter(
            name="preview_max_tokens",
            display_name="Preview: Max Tokens",
            description="Maximum tokens per preview generation.",
            type="int",
            default=80,
            min_val=20,
            max_val=256,
        ),

        # ═══ EXPORT ═══
        ModuleParameter(
            name="export_weights",
            display_name="Advanced: Export Ablated Weights",
            description=(
                "Apply the cleaned direction permanently to model weights "
                "and save a checkpoint. This is irreversible without "
                "reloading the model. Off by default. Use only for "
                "external benchmarking with tools that cannot use hooks."
            ),
            type="bool",
            default=False,
        ),
    ]

    def __init__(self):
        self._pipeline = None

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline

    def validate(self, session_results: list, params: dict) -> tuple:
        if self._pipeline is None or not self._pipeline.loaded:
            return False, "No model loaded."

        csv_text = params.get("registry_csv", "")
        if not csv_text.strip():
            return False, ("No atom registry provided. Paste CSV text into the "
                           "registry field, or load the starter template from "
                           "src/templates/concept_atoms_starter.csv.")

        atoms = parse_atom_registry(csv_text)
        ok, msg = validate_registry(atoms)
        if not ok:
            return False, msg

        # Check layer range against model
        n_layers = self._pipeline.adapter.n_layers(self._pipeline.instruct_model)
        layer_end = params.get("layer_end", 14)
        if layer_end > n_layers:
            return False, (f"Layer end ({layer_end}) exceeds model layer count "
                           f"({n_layers}). For this model, try layer_end={n_layers - 1}.")

        # Need session data for refusal direction
        has_emb = sum(1 for r in session_results
                      if r.get("per_token_final_emb"))
        if has_emb < 10:
            return False, (f"Need at least 10 results with per_token_final_emb "
                           f"for refusal direction fitting; found {has_emb}. "
                           f"Run an analysis session first.")

        return True, "OK"

    def run(self, session_results: list, params: dict,
            progress: Callable = None) -> dict:
        results = {}
        t0 = time.time()

        def prog(msg):
            if progress:
                progress(msg)
            logger.info(f"[CAM] {msg}")

        model = self._pipeline.instruct_model
        adapter = self._pipeline.adapter
        tokenizer = self._pipeline.tokenizer

        # ── Step 1: Parse registry ─────────────────────────────────
        prog("Parsing atom registry...")
        csv_text = params.get("registry_csv", "")
        atoms = parse_atom_registry(csv_text)

        n_shields = sum(1 for d in atoms.values() if d["role"] == "shield")
        n_confounds = sum(1 for d in atoms.values() if d["role"] == "confound")
        n_targets = sum(1 for d in atoms.values() if d["role"] == "target")

        results["registry"] = {
            "atoms": [
                {
                    "name": name,
                    "role": data["role"],
                    "n_positive": len(data["positive"]),
                    "n_negative": len(data["negative"]),
                }
                for name, data in atoms.items()
            ],
            "n_atoms": len(atoms),
            "n_shields": n_shields,
            "n_confounds": n_confounds,
            "n_targets": n_targets,
        }
        prog(f"Registry: {len(atoms)} atoms ({n_shields} shields, "
             f"{n_confounds} confounds, {n_targets} targets)")

        # ── Step 2: Compute atom directions ────────────────────────
        n_layers = adapter.n_layers(model)
        layer_start = params.get("layer_start", 4)
        layer_end = min(params.get("layer_end", 14), n_layers)
        layer_range = list(range(layer_start, layer_end))
        results["analysis_layers"] = layer_range

        prog(f"Computing atom directions (layers {layer_start}-{layer_end-1})...")
        atom_directions = compute_atom_directions(
            model, adapter, tokenizer, atoms, layer_range, progress=prog,
        )

        # ── Step 3: Compute raw refusal direction ──────────────────
        prog("Fitting raw refusal direction from session data...")
        from src.engine.ablation import DirectionFitter

        harm_cats = {"harmful", "jailbreak", "unknown"}
        safe_cats = {"benign", "mild"}
        fitter = DirectionFitter(session_results, harm_cats, safe_cats)
        fit = fitter.difference_of_means(
            holdout_frac=params.get("refusal_holdout_frac", 0.20),
        )

        # Convert fitted direction to per-layer (same direction at all layers
        # since DirectionFitter works on final-layer embeddings)
        refusal_per_layer = {}
        for li in layer_range:
            refusal_per_layer[li] = fit.vector.astype(np.float32)

        results["refusal_direction"] = {
            "auroc": round(fit.train_auroc, 4),
            "holdout_auroc": round(fit.holdout_auroc, 4) if fit.holdout_auroc else None,
            "n_harm": fit.n_harm,
            "n_safe": fit.n_safe,
        }
        prog(f"Refusal direction: AUROC={fit.train_auroc:.4f}")

        # ── Step 4: Orthogonality diagnostics ──────────────────────
        prog("Computing orthogonality map...")
        ortho_dirty = compute_orthogonality_map(
            refusal_per_layer, atom_directions, layer_range,
        )
        entanglement = summarize_entanglement(ortho_dirty, atoms)
        results["orthogonality_dirty"] = {
            str(li): {name: round(cos, 4) for name, cos in cosines.items()}
            for li, cosines in ortho_dirty.items()
        }
        results["entanglement_summary_dirty"] = entanglement
        prog(f"Entanglement: max Shield |cosine| = {entanglement['max_shield_cosine']:.3f}")
        prog(f"  {entanglement['recommendation']}")

        # ── Step 5: Ridge cleaning ─────────────────────────────────
        cleaned_per_layer = {}
        if params.get("run_cleaning", True):
            prog("Running SRA ridge-regression cleaning...")
            ridge_lambda = params.get("ridge_lambda", 1.0)

            for li in layer_range:
                r_dirty = refusal_per_layer.get(li)
                if r_dirty is None:
                    continue

                A_SC = build_sc_matrix(atom_directions, atoms, li)
                if A_SC.size == 0:
                    prog(f"  Layer {li}: no Shield/Confound atoms available, skipping.")
                    cleaned_per_layer[li] = r_dirty
                    continue

                r_clean = clean_direction(r_dirty, A_SC, ridge_lambda=ridge_lambda)
                cleaned_per_layer[li] = r_clean

            # Post-cleaning orthogonality
            ortho_cleaned = compute_orthogonality_map(
                cleaned_per_layer, atom_directions, layer_range,
            )
            entanglement_cleaned = summarize_entanglement(ortho_cleaned, atoms)

            results["orthogonality_cleaned"] = {
                str(li): {name: round(cos, 4) for name, cos in cosines.items()}
                for li, cosines in ortho_cleaned.items()
            }
            results["entanglement_summary_cleaned"] = entanglement_cleaned
            prog(f"Post-cleaning: max Shield |cosine| = "
                 f"{entanglement_cleaned['max_shield_cosine']:.3f} "
                 f"(was {entanglement['max_shield_cosine']:.3f})")

            cleaning_effective = (entanglement_cleaned["max_shield_cosine"] <
                                  entanglement["max_shield_cosine"] * 0.5)
            results["cleaning_effective"] = cleaning_effective
            if not cleaning_effective:
                prog("WARNING: Cleaning did not substantially reduce Shield "
                     "entanglement. Consider adjusting ridge_lambda or "
                     "adding more Shield atoms.")
        else:
            prog("Cleaning skipped (run_cleaning=False).")
            cleaned_per_layer = refusal_per_layer

        # ── Step 6: Gamma calibration ──────────────────────────────
        gamma_mode = params.get("gamma_mode", "fixed")
        gamma_value = params.get("gamma_value", 1.0)
        gamma_schedule = calibrate_gamma(
            atom_directions, atoms, layer_range,
            mode=gamma_mode, fixed_value=gamma_value,
        )
        results["gamma"] = {
            "mode": gamma_mode,
            "per_layer": {str(li): round(g, 4) for li, g in gamma_schedule.items()},
        }

        # ── Step 7: Preview ────────────────────────────────────────
        if params.get("run_preview", False):
            prog("Running ablation preview with temporary hooks...")
            try:
                from src.engine.interventions import ActivationIntervention
                from src.engine.ablation import RefusalDetector

                # Get harmful prompts from session
                harm_prompts = [
                    r["prompt"] for r in session_results
                    if r.get("category", "").lower() in {"harmful", "jailbreak"}
                ]
                n_preview = min(params.get("preview_n_prompts", 10), len(harm_prompts))
                preview_prompts = harm_prompts[:n_preview]

                if not preview_prompts:
                    prog("No harmful prompts in session for preview.")
                else:
                    # Use the middle layer's direction for the hook
                    mid_layer = layer_range[len(layer_range) // 2]
                    direction = cleaned_per_layer.get(mid_layer)
                    if direction is None:
                        direction = list(cleaned_per_layer.values())[0]

                    gamma = gamma_schedule.get(mid_layer, gamma_value)
                    direction_tensor = torch.tensor(direction, dtype=torch.float32)

                    intv = ActivationIntervention()
                    intv.install_multilayer_ablation(
                        model, adapter,
                        direction=direction_tensor,
                        layers=layer_range,
                        alpha=gamma,
                    )

                    detector = RefusalDetector()
                    device = next(model.parameters()).device
                    preview_results = []

                    with MODEL_LOCK:
                        for i, prompt in enumerate(preview_prompts):
                            if i % 3 == 0:
                                prog(f"Preview generation: {i+1}/{n_preview}")

                            inputs = tokenizer(
                                prompt, return_tensors="pt",
                                add_special_tokens=True,
                            ).to(device)

                            with torch.no_grad():
                                output = model.generate(
                                    **inputs,
                                    max_new_tokens=params.get("preview_max_tokens", 80),
                                    do_sample=False,
                                )

                            new_tokens = output[0][inputs["input_ids"].shape[1]:]
                            reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
                            refused = detector.detect(reply)
                            preview_results.append({
                                "prompt": prompt[:100],
                                "reply": reply[:200],
                                "refused": refused,
                            })

                    intv.remove()

                    n_refused = sum(1 for r in preview_results if r["refused"])
                    results["preview"] = {
                        "n_prompts": n_preview,
                        "n_refused": n_refused,
                        "refusal_rate": round(n_refused / n_preview, 4) if n_preview > 0 else 0,
                        "prompts": preview_results,
                    }
                    prog(f"Preview: {n_refused}/{n_preview} refused "
                         f"({results['preview']['refusal_rate']:.0%})")

            except Exception as e:
                prog(f"Preview failed: {e}")
                results["preview_error"] = str(e)

        # ── Step 8: Persist artifact ───────────────────────────────
        prog("Persisting artifact...")

        precision = engine_config.get("serialization_precision") or 6

        artifact = {
            "module": "concept_atoms",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "registry": results["registry"],
            "analysis_layers": layer_range,
            "dirty_direction": {
                "per_layer": {
                    str(li): [round(float(x), precision) for x in v]
                    for li, v in refusal_per_layer.items()
                },
            },
            "cleaned_direction": {
                "per_layer": {
                    str(li): [round(float(x), precision) for x in v]
                    for li, v in cleaned_per_layer.items()
                },
                "method": "ridge_regression" if params.get("run_cleaning", True) else "none",
                "lambda": params.get("ridge_lambda", 1.0),
            },
            "gamma": results["gamma"],
            "entanglement_dirty": results.get("entanglement_summary_dirty", {}),
            "entanglement_cleaned": results.get("entanglement_summary_cleaned", {}),
        }

        results["_artifact"] = artifact
        results["artifact_size_keys"] = len(json.dumps(artifact))

        # ── Step 9: Export weights (if requested) ──────────────────
        if params.get("export_weights", False):
            prog("WARNING: Applying permanent weight modification...")
            try:
                self._export_ablated_weights(
                    model, adapter, cleaned_per_layer,
                    gamma_schedule, prog,
                )
                results["export"] = {"status": "success"}
            except Exception as e:
                prog(f"Export failed: {e}")
                results["export"] = {"status": "failed", "error": str(e)}

        elapsed = time.time() - t0
        results["elapsed_seconds"] = round(elapsed, 1)
        prog(f"Concept atom analysis complete ({elapsed:.0f}s)")

        return results

    # ── Weight export (advanced, off by default) ───────────────────

    def _export_ablated_weights(
        self, model, adapter, direction_per_layer, gamma_per_layer, progress,
    ):
        """Apply rank-one weight updates permanently. Irreversible."""
        for li, direction in direction_per_layer.items():
            gamma = gamma_per_layer.get(li, 1.0)
            v = torch.tensor(direction, dtype=torch.float32)
            v = v / (v.norm() + 1e-10)

            # Orthogonalize residual-stream writers at this layer
            for role in ["o_proj", "down_proj"]:
                try:
                    target = adapter.resolve_hook_target(model, role, li)
                    W = target.weight.data.float()
                    # W shape: [d_out, d_in]; for residual writers d_out = d_model
                    proj = gamma * torch.outer(v, v @ W)
                    target.weight.data = (W - proj).to(target.weight.dtype)
                except Exception as e:
                    progress(f"  Layer {li} {role}: skipped ({e})")

        progress("Weight export complete. Model is now permanently modified.")
