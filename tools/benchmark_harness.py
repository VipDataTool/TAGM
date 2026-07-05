#!/usr/bin/env python3
"""
Standalone benchmark harness for offline cascade detection on prompt sets.

Bridges two existing TAGM subsystems:
  - The analyzer (per-token SFD density, rank displacement, KL, stress, entropy)
  - The CascadeDetector from ecm_v4 (multi-scale anomaly detection on scalar traces)

The gap this fills: ECM v2/v4 only run during generation (LogitsProcessor).
The analyzer computes rich per-token traces but has no cascade detection.
This harness feeds analyzer traces through CascadeDetector instances offline,
producing the same cascade signals the runtime would — without actuation,
without generation, on any prompt set.

Two modes:
  --mode prompt    Analyze the prompt text itself (genre confound, representation geometry)
  --mode generate  Generate a response, then analyze the response (§9 benchmark)

Usage:
  # Prompt-only analysis (genre confound experiment)
  python tools/benchmark_harness.py \\
    --prompts tools/genre_confound_prompts.csv \\
    --mode prompt \\
    --out benchmark_out/genre_confound

  # Generate + analyze (§9 benchmark battery)
  python tools/benchmark_harness.py \\
    --prompts prompts_200_harm.csv \\
    --mode generate \\
    --runs 10 \\
    --out benchmark_out/harm_200

  # Resume an interrupted run
  python tools/benchmark_harness.py \\
    --prompts prompts_200_harm.csv \\
    --mode generate \\
    --out benchmark_out/harm_200 \\
    --resume

Requires a running TAGM instance with a loaded model pair, OR direct
model loading via --model-pair (not yet implemented — use the TAGM
server for now).

Output: one JSONL file (resumable) + a summary JSON.

Drop this file into tagm/tools/ alongside the existing harness scripts.
"""

import argparse
import csv
import json
import hashlib
import sys
import os
import time
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# TAGM imports — adjust path if running from repo root vs tools/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.engine.ecm_v4 import CascadeDetector
except ImportError:
    # Fallback: inline the detector if ecm_v4 isn't importable standalone
    # (e.g., torch not available). The pure-python reference from the test
    # suite would go here. For now, fail loud.
    raise ImportError(
        "Could not import CascadeDetector from src.engine.ecm_v4. "
        "Ensure you're running from the TAGM repo root, or that torch "
        "is available in your environment."
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    """All knobs in one place. Defaults match ECM v2/v4 production settings."""

    # Detector parameters (shared across channels, matching ecm_v4 defaults)
    n_scales: int = 5
    deadband: float = 0.75
    agreement: int = 2
    warmup: int = 8
    sigma_floor: float = 0.01

    # Analysis flags
    compute_sfd: bool = True
    compute_ltp: bool = True
    compute_kl: bool = True
    full_capture: bool = True

    # Generation parameters (mode=generate only)
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    runs_per_prompt: int = 10

    # Output
    output_dir: str = "benchmark_out"


# ---------------------------------------------------------------------------
# Offline cascade detection
# ---------------------------------------------------------------------------

def run_cascade_detection(
    trace: List[float],
    config: HarnessConfig,
    negate: bool = False,
) -> Dict[str, Any]:
    """
    Feed a per-token scalar trace through a CascadeDetector offline.

    Parameters
    ----------
    trace : list of float
        Per-token scalar values (entropy, density, etc.)
    config : HarnessConfig
        Detector hyperparameters.
    negate : bool
        If True, negate values before feeding (density collapse → rise).

    Returns
    -------
    dict with:
        per_token_signal : list of float — cascade signal per token
        per_token_sigma  : list of float — running σ estimate per token
        n_interventions  : int — tokens where signal > 0
        n_tokens         : int — total tokens processed
        intervention_rate: float — n_interventions / n_tokens
        max_signal       : float — peak cascade signal
        mean_signal      : float — mean over non-zero signals
    """
    detector = CascadeDetector(
        n_scales=config.n_scales,
        deadband=config.deadband,
        agreement=config.agreement,
        warmup=config.warmup,
        sigma_floor=config.sigma_floor,
    )

    signals = []
    sigmas = []

    for value in trace:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            signals.append(float("nan"))
            sigmas.append(float("nan"))
            continue

        v = -value if negate else value
        step = detector.update(v)
        signals.append(step.signal)
        sigmas.append(step.sigma)

    n_tokens = len([s for s in signals if not math.isnan(s)])
    n_interventions = len([s for s in signals if not math.isnan(s) and s > 0])
    nonzero = [s for s in signals if not math.isnan(s) and s > 0]

    return {
        "per_token_signal": signals,
        "per_token_sigma": sigmas,
        "n_interventions": n_interventions,
        "n_tokens": n_tokens,
        "intervention_rate": n_interventions / n_tokens if n_tokens > 0 else 0.0,
        "max_signal": max(nonzero) if nonzero else 0.0,
        "mean_signal": sum(nonzero) / len(nonzero) if nonzero else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-token entropy from logits (if not already in analyzer output)
# ---------------------------------------------------------------------------

def compute_per_token_entropy(logits_list):
    """
    Compute per-token distribution entropy from captured logits.

    Parameters
    ----------
    logits_list : list of array-like
        Per-token logit vectors. Each element is a 1D array of |V| logits.

    Returns
    -------
    list of float — entropy in nats per token.
    """
    import numpy as np

    entropies = []
    for logits in logits_list:
        logits = np.array(logits, dtype=np.float64)
        # Numerically stable log-softmax
        max_l = logits.max()
        shifted = logits - max_l
        log_sum_exp = max_l + np.log(np.exp(shifted).sum())
        log_probs = logits - log_sum_exp
        probs = np.exp(log_probs)
        entropy = -np.sum(probs * log_probs)
        entropies.append(float(entropy))
    return entropies


# ---------------------------------------------------------------------------
# Representation extraction (for genre confound geometry analysis)
# ---------------------------------------------------------------------------

def extract_prompt_representation(result, sfd_cache):
    """
    Extract the prompt-level representation in the ΔW subspace.

    From the hooked activations and the SFD cache, compute the mean-pooled
    hidden state projected into the alignment-delta subspace per layer,
    then average across layers.

    Parameters
    ----------
    result : PromptResult
        The analyzer result containing per-token activations.
    sfd_cache : dict
        The precomputed SFD cache (V_k, S per layer).

    Returns
    -------
    dict with:
        representation : list of float — k-dimensional prompt centroid
            in the alignment-delta subspace
        per_layer : dict of layer_idx → list of float
        density_mean : float — mean SFD density across tokens
    """
    import numpy as np

    if not hasattr(result, "sfd") or result.sfd is None:
        return None

    sfd = result.sfd
    rep = {
        "density_mean": float(sfd.density_mean) if hasattr(sfd, "density_mean") else None,
        "density_per_token": [float(d) for d in sfd.density] if hasattr(sfd, "density") else None,
    }

    # If we have raw per-token projections from the SFD computation,
    # extract the prompt centroid. This depends on the analyzer storing
    # the projected coordinates — check result structure.
    # If not available, the density trace is still useful for cascade
    # detection.

    return rep


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(csv_path: str) -> List[Dict[str, str]]:
    """Load prompts from CSV. Expects at minimum a 'prompt' column."""
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "prompt" not in row:
                raise ValueError(f"CSV must have a 'prompt' column. Found: {list(row.keys())}")
            prompts.append(dict(row))
    return prompts


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

def make_record(
    prompt_row: Dict[str, str],
    run_idx: int,
    mode: str,
    analyzer_result,
    entropy_cascade: Optional[Dict] = None,
    density_cascade: Optional[Dict] = None,
    representation: Optional[Dict] = None,
    response_text: Optional[str] = None,
    elapsed_s: float = 0.0,
) -> Dict[str, Any]:
    """Build one JSONL record from all available data."""
    rec = {
        # Prompt metadata
        "prompt": prompt_row.get("prompt", ""),
        "category": prompt_row.get("category", ""),
        "harmfulness": prompt_row.get("harmfulness", prompt_row.get("category", "")),
        "register": prompt_row.get("register", ""),
        "matched_pair_id": prompt_row.get("matched_pair_id", ""),
        "run_idx": run_idx,
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed_s,

        # Prompt fingerprint for dedup / resume
        "prompt_hash": hashlib.sha256(
            prompt_row.get("prompt", "").encode()
        ).hexdigest()[:16],
    }

    # Analyzer scalars
    if analyzer_result is not None:
        r = analyzer_result
        rec["seq_len"] = getattr(r, "seq_len", None)
        rec["stress_score"] = getattr(r, "stress_score", None)
        rec["net_correction"] = getattr(r, "net_correction", None)
        rec["entropy"] = getattr(r, "entropy", None)
        rec["kl_divergence"] = getattr(r, "kl_divergence", None)
        rec["delta_scale"] = getattr(r, "delta_scale", None)
        rec["n_negative_tokens"] = getattr(r, "n_negative_tokens", None)

        # SFD scalars
        if hasattr(r, "sfd") and r.sfd is not None:
            rec["sfd_density_mean"] = getattr(r.sfd, "density_mean", None)

        # Rank displacement scalars
        if hasattr(r, "rank_displacement") and r.rank_displacement is not None:
            rd = r.rank_displacement
            rec["rd_mean_tau"] = rd.get("mean_tau") if isinstance(rd, dict) else getattr(rd, "mean_tau", None)
            rec["rd_mean_overlap"] = rd.get("mean_overlap") if isinstance(rd, dict) else getattr(rd, "mean_overlap", None)
            rec["rd_replacement_ratio"] = rd.get("replacement_ratio") if isinstance(rd, dict) else getattr(rd, "replacement_ratio", None)

        # Per-token traces (stored for offline re-analysis)
        rec["per_token_kl"] = _safe_list(getattr(r, "per_token_kl", None))
        rec["per_token_stress"] = _safe_list(getattr(r, "per_token_stress", None))

        # Per-token SFD density
        if hasattr(r, "sfd") and r.sfd is not None and hasattr(r.sfd, "density"):
            rec["per_token_density"] = _safe_list(r.sfd.density)

    # Cascade detection results (offline)
    if entropy_cascade is not None:
        rec["entropy_cascade"] = {
            k: v for k, v in entropy_cascade.items()
            if k != "per_token_signal" and k != "per_token_sigma"
        }
        rec["entropy_cascade_per_token"] = {
            "signal": entropy_cascade.get("per_token_signal"),
            "sigma": entropy_cascade.get("per_token_sigma"),
        }

    if density_cascade is not None:
        rec["density_cascade"] = {
            k: v for k, v in density_cascade.items()
            if k != "per_token_signal" and k != "per_token_sigma"
        }
        rec["density_cascade_per_token"] = {
            "signal": density_cascade.get("per_token_signal"),
            "sigma": density_cascade.get("per_token_sigma"),
        }

    # Representation (for genre confound geometry)
    if representation is not None:
        rec["representation"] = representation

    # Generated response (mode=generate only)
    if response_text is not None:
        rec["response_text"] = response_text

    return rec


def _safe_list(arr):
    """Convert numpy array or None to a JSON-safe list."""
    if arr is None:
        return None
    try:
        return [float(x) for x in arr]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed(jsonl_path: str):
    """Load set of (prompt_hash, run_idx) already completed."""
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec.get("prompt_hash", ""), rec.get("run_idx", 0))
                completed.add(key)
            except json.JSONDecodeError:
                continue
    return completed


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_benchmark(
    prompts: List[Dict[str, str]],
    analyzer,
    config: HarnessConfig,
    mode: str = "prompt",
    pipeline=None,
    resume: bool = False,
):
    """
    Run the benchmark harness.

    Parameters
    ----------
    prompts : list of dict
        Loaded from CSV.
    analyzer : Analyzer instance
        The TAGM analyzer with a loaded model pair.
    config : HarnessConfig
        All parameters.
    mode : str
        "prompt" — analyze prompts only (genre confound).
        "generate" — generate responses, then analyze (§9).
    pipeline : Pipeline instance, optional
        Required for mode=generate. The TAGM pipeline with loaded models.
    resume : bool
        If True, skip already-completed (prompt_hash, run_idx) pairs.
    """
    import numpy as np

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    # Resume support
    completed = load_completed(str(jsonl_path)) if resume else set()
    if completed:
        print(f"Resuming: {len(completed)} records already completed.")

    n_runs = config.runs_per_prompt if mode == "generate" else 1
    total = len(prompts) * n_runs
    done = 0
    skipped = 0

    # Precompute SFD cache if needed
    sfd_cache = None
    if config.compute_sfd:
        try:
            from src.engine.sfd import precompute_sfd_cache
            sfd_cache = precompute_sfd_cache(analyzer)
            print("SFD cache precomputed.")
        except Exception as e:
            print(f"Warning: SFD cache precompute failed: {e}")
            print("Density cascade detection will not be available.")

    with open(jsonl_path, "a") as out_f:
        # Balanced loop: run → prompt → avoid order effects
        for run_idx in range(n_runs):
            for prompt_row in prompts:
                prompt_text = prompt_row["prompt"]
                prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

                # Resume check
                if (prompt_hash, run_idx) in completed:
                    skipped += 1
                    done += 1
                    continue

                t0 = time.time()
                response_text = None

                try:
                    if mode == "generate":
                        # Generate response, then analyze the response
                        if pipeline is None:
                            raise RuntimeError("mode=generate requires a pipeline")

                        response_text = _generate_response(
                            pipeline, prompt_text, config
                        )
                        analysis_text = response_text
                    else:
                        # Analyze the prompt text directly
                        analysis_text = prompt_text

                    # Run the analyzer
                    result = analyzer.analyze_prompt(
                        analysis_text,
                        compute_sfd=config.compute_sfd,
                        compute_ltp=config.compute_ltp,
                        compute_kl=config.compute_kl,
                        full_capture=config.full_capture,
                    )

                    # --- Offline cascade detection ---

                    # Channel 1: Entropy
                    entropy_cascade = None
                    per_token_entropy = None

                    # Try to get per-token entropy from the result
                    if hasattr(result, "per_token_entropy") and result.per_token_entropy is not None:
                        per_token_entropy = [float(x) for x in result.per_token_entropy]
                    elif hasattr(result, "instruct_topk") and result.instruct_topk is not None:
                        # Compute from top-k probabilities if full logits aren't stored
                        # This is an approximation — flag it
                        pass

                    if per_token_entropy is not None:
                        entropy_cascade = run_cascade_detection(
                            per_token_entropy, config, negate=False
                        )

                    # Channel 2: Density (negate so collapse → rise)
                    density_cascade = None
                    if (
                        hasattr(result, "sfd")
                        and result.sfd is not None
                        and hasattr(result.sfd, "density")
                        and result.sfd.density is not None
                    ):
                        density_trace = [float(d) for d in result.sfd.density]
                        density_cascade = run_cascade_detection(
                            density_trace, config, negate=True
                        )

                    # Representation extraction (for geometry analysis)
                    representation = None
                    if sfd_cache is not None:
                        representation = extract_prompt_representation(
                            result, sfd_cache
                        )

                    elapsed = time.time() - t0

                    # Build and write the record
                    rec = make_record(
                        prompt_row=prompt_row,
                        run_idx=run_idx,
                        mode=mode,
                        analyzer_result=result,
                        entropy_cascade=entropy_cascade,
                        density_cascade=density_cascade,
                        representation=representation,
                        response_text=response_text,
                        elapsed_s=elapsed,
                    )

                    out_f.write(json.dumps(rec, default=str) + "\n")
                    out_f.flush()

                except Exception as e:
                    elapsed = time.time() - t0
                    # Write error record so we don't retry on resume
                    error_rec = {
                        "prompt_hash": prompt_hash,
                        "run_idx": run_idx,
                        "prompt": prompt_text,
                        "category": prompt_row.get("category", ""),
                        "error": str(e),
                        "elapsed_s": elapsed,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    out_f.write(json.dumps(error_rec) + "\n")
                    out_f.flush()
                    print(f"  ERROR: {e}")

                done += 1
                cat = prompt_row.get("category", "?")
                status = f"[{done}/{total}] run={run_idx} cat={cat}"
                if entropy_cascade:
                    status += f" ent_rate={entropy_cascade['intervention_rate']:.1%}"
                if density_cascade:
                    status += f" den_rate={density_cascade['intervention_rate']:.1%}"
                print(status)

    print(f"\nDone. {done} total, {skipped} skipped (resume).")
    print(f"Results: {jsonl_path}")

    # Write summary
    _write_summary(str(jsonl_path), str(out_dir / "summary.json"))


def _generate_response(pipeline, prompt_text, config):
    """
    Generate a response using the TAGM pipeline.

    Adapt this to your pipeline's generate API. The current stub
    assumes pipeline.generate() or model.generate() — adjust to
    match your setup.
    """
    # This is a stub — the exact API depends on how TAGM's pipeline
    # exposes generation. The chat service uses model.generate() with
    # HF's generate API. Adapt accordingly:
    #
    # tokens = pipeline.tokenizer.encode(prompt_text, return_tensors="pt")
    # output = pipeline.active_model.generate(
    #     tokens,
    #     max_new_tokens=config.max_new_tokens,
    #     temperature=config.temperature,
    #     top_p=config.top_p,
    #     do_sample=True,
    # )
    # response = pipeline.tokenizer.decode(output[0][tokens.shape[1]:])
    # return response
    raise NotImplementedError(
        "Generation stub — implement using your pipeline's generate API. "
        "See the comment in _generate_response() for the pattern."
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _write_summary(jsonl_path: str, summary_path: str):
    """Compute per-category summary statistics from the results JSONL."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "error" not in rec:
                    records.append(rec)
            except json.JSONDecodeError:
                continue

    if not records:
        print("No valid records for summary.")
        return

    # Group by category
    by_cat = {}
    for rec in records:
        cat = rec.get("category", "unknown")
        by_cat.setdefault(cat, []).append(rec)

    summary = {"n_total": len(records), "categories": {}}

    for cat, recs in sorted(by_cat.items()):
        cat_summary = {"n": len(recs)}

        # Analyzer scalars
        for key in ["stress_score", "kl_divergence", "sfd_density_mean",
                     "rd_mean_tau", "rd_mean_overlap", "entropy"]:
            values = [r[key] for r in recs if r.get(key) is not None]
            if values:
                cat_summary[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }

        # Cascade detection summaries
        for channel in ["entropy_cascade", "density_cascade"]:
            rates = [
                r[channel]["intervention_rate"]
                for r in recs
                if r.get(channel) and "intervention_rate" in r[channel]
            ]
            maxsigs = [
                r[channel]["max_signal"]
                for r in recs
                if r.get(channel) and "max_signal" in r[channel]
            ]
            if rates:
                cat_summary[channel] = {
                    "mean_intervention_rate": sum(rates) / len(rates),
                    "max_intervention_rate": max(rates),
                    "mean_max_signal": sum(maxsigs) / len(maxsigs) if maxsigs else 0,
                    "n": len(rates),
                }

        summary["categories"][cat] = cat_summary

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {summary_path}")

    # Print a quick table
    print(f"\n{'Category':<15} {'n':>4} {'stress':>8} {'KL':>8} "
          f"{'density':>8} {'ent_rate':>9} {'den_rate':>9}")
    print("-" * 72)
    for cat, s in sorted(summary["categories"].items()):
        n = s["n"]
        stress = s.get("stress_score", {}).get("mean", float("nan"))
        kl = s.get("kl_divergence", {}).get("mean", float("nan"))
        density = s.get("sfd_density_mean", {}).get("mean", float("nan"))
        ent_rate = s.get("entropy_cascade", {}).get("mean_intervention_rate", float("nan"))
        den_rate = s.get("density_cascade", {}).get("mean_intervention_rate", float("nan"))
        print(f"{cat:<15} {n:>4} {stress:>8.3f} {kl:>8.3f} "
              f"{density:>8.3f} {ent_rate:>8.1%} {den_rate:>8.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TAGM benchmark harness — offline cascade detection on prompt sets"
    )
    parser.add_argument(
        "--prompts", required=True,
        help="Path to prompt CSV (must have 'prompt' column; 'category' recommended)"
    )
    parser.add_argument(
        "--mode", choices=["prompt", "generate"], default="prompt",
        help="'prompt' = analyze prompt text only; 'generate' = generate response + analyze"
    )
    parser.add_argument(
        "--out", default="benchmark_out",
        help="Output directory (default: benchmark_out/)"
    )
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per prompt (mode=generate only, default: 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed (prompt_hash, run_idx) pairs")

    # Detector parameters
    parser.add_argument("--deadband", type=float, default=0.75)
    parser.add_argument("--agreement", type=int, default=2)
    parser.add_argument("--n-scales", type=int, default=5)

    # Analysis flags
    parser.add_argument("--no-sfd", action="store_true", help="Skip SFD computation")
    parser.add_argument("--no-ltp", action="store_true", help="Skip LTP computation")
    parser.add_argument("--no-kl", action="store_true", help="Skip KL computation")

    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    config = HarnessConfig(
        n_scales=args.n_scales,
        deadband=args.deadband,
        agreement=args.agreement,
        compute_sfd=not args.no_sfd,
        compute_ltp=not args.no_ltp,
        compute_kl=not args.no_kl,
        runs_per_prompt=args.runs if args.mode == "generate" else 1,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.out,
    )

    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Category breakdown
    cats = {}
    for p in prompts:
        cat = p.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")

    # --- Connect to TAGM ---
    # Option A: import the analyzer directly (requires model to be loaded)
    # Option B: connect via HTTP API
    #
    # For now, this demonstrates Option A. Adjust imports to match your
    # TAGM installation.

    print("\nConnecting to TAGM...")
    try:
        from src.engine.analyzer import Analyzer
        from src.core.pipeline import Pipeline

        # Check if there's already a loaded pipeline/analyzer in the
        # running TAGM instance. If running standalone, you'll need to
        # load the model pair here:
        #
        # pipeline = Pipeline()
        # pipeline.load_model("meta-llama/Llama-3.2-1B-Instruct",
        #                     "meta-llama/Llama-3.2-1B")
        # analyzer = Analyzer(pipeline)

        print("TAGM modules imported. Ensure a model pair is loaded.")
        print("If running standalone, uncomment the pipeline.load_model() "
              "block in main().")

        # Placeholder — replace with your actual analyzer instance
        analyzer = None
        pipeline = None

        if analyzer is None:
            print("\nNo analyzer instance available. To run:")
            print("  1. Start TAGM: bash start.sh")
            print("  2. Load a model pair in the UI")
            print("  3. Import and run from the TAGM Python environment")
            print("\nOr implement the standalone loading block in main().")
            print("\nExiting. The harness structure is ready — wire it to")
            print("your analyzer and it will produce results.")
            return

        run_benchmark(
            prompts=prompts,
            analyzer=analyzer,
            config=config,
            mode=args.mode,
            pipeline=pipeline if args.mode == "generate" else None,
            resume=args.resume,
        )

    except ImportError as e:
        print(f"Could not import TAGM modules: {e}")
        print("Ensure you're running from the TAGM repo root with")
        print("dependencies installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
