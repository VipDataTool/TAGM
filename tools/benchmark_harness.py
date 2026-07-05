#!/usr/bin/env python3
"""
Standalone benchmark harness for offline cascade detection on prompt sets.

Bridges two TAGM subsystems:
  - The analyzer (per-token SFD density, rank displacement, KL, stress)
  - The CascadeDetector from ecm_v4 (multi-scale anomaly detection)

Drop this in tagm/tools/benchmark_harness.py and run from repo root:

  # Prompt-only analysis (representation geometry)
  python tools/benchmark_harness.py \
    --prompts prompts_200_harm.csv \
    --mode prompt \
    --out benchmark_out/harm_200

  # Resume after interruption
  python tools/benchmark_harness.py \
    --prompts prompts_200_harm.csv \
    --mode prompt \
    --out benchmark_out/harm_200 \
    --resume

  # Generate + analyze (§9 battery)
  python tools/benchmark_harness.py \
    --prompts prompts_200_harm.csv \
    --mode generate \
    --runs 10 \
    --out benchmark_out/harm_200_gen

  # Custom model pair
  python tools/benchmark_harness.py \
    --prompts prompts_200_harm.csv \
    --mode prompt \
    --instruct meta-llama/Llama-3.2-1B-Instruct \
    --base meta-llama/Llama-3.2-1B \
    --out benchmark_out/llama_harm
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.ecm_v4 import CascadeDetector


# ---------------------------------------------------------------------------
# Offline cascade detection — the core bridge
# ---------------------------------------------------------------------------

def run_cascade_detection(trace, n_scales=5, deadband=0.75, agreement=2,
                          warmup=8, sigma_floor=0.01, negate=False):
    """
    Feed a per-token scalar trace through a CascadeDetector offline.

    Parameters
    ----------
    trace : list of float
        Per-token values (density, KL, stress, etc.)
    negate : bool
        If True, negate values before feeding. Use for density
        (collapse = anomaly, but detector fires on rises).

    Returns
    -------
    dict with per-token signals, summary statistics, and
    intervention rate.
    """
    detector = CascadeDetector(
        n_scales=n_scales,
        deadband=deadband,
        agreement=agreement,
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
        sigmas.append(step.std)

    clean = [s for s in signals if not math.isnan(s)]
    firing = [s for s in clean if s > 0]

    return {
        "per_token_signal": signals,
        "per_token_sigma": sigmas,
        "n_interventions": len(firing),
        "n_tokens": len(clean),
        "intervention_rate": len(firing) / len(clean) if clean else 0.0,
        "max_signal": max(firing) if firing else 0.0,
        "mean_signal": sum(firing) / len(firing) if firing else 0.0,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_pair(instruct_id, base_id):
    """Load a model pair via TAGM's pipeline and return (pipeline, analyzer)."""
    from src.core.pipeline import Pipeline
    from src.engine.analyzer import Analyzer

    print(f"Loading model pair:")
    print(f"  instruct: {instruct_id}")
    print(f"  base:     {base_id}")

    pipeline = Pipeline(instruct_id, base_id)
    pipeline.load()
    pipeline.load_base()
    # model loaded via constructor

    analyzer = Analyzer(pipeline)
    print("Model pair loaded.\n")
    return pipeline, analyzer


# ---------------------------------------------------------------------------
# Per-token entropy from a forward pass
# ---------------------------------------------------------------------------

def compute_per_token_entropy(pipeline, text):
    """
    Run a forward pass and compute per-token entropy from logits.

    The analyzer doesn't store per-token entropy, so this is a
    standalone computation. Uses the instruct model.
    """
    import torch
    import numpy as np

    tokenizer = pipeline.tokenizer
    model = pipeline.active_model

    tokens = tokenizer.encode(text, return_tensors="pt",
                              add_special_tokens=False)
    tokens = tokens.to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0]  # [seq_len, vocab]

    # Per-position entropy via log-softmax (numerically stable)
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len]

    return entropy.cpu().numpy().tolist()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(csv_path):
    """Load prompts from CSV. Requires a 'prompt' column."""
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "prompt" not in row:
                raise ValueError(
                    f"CSV must have a 'prompt' column. Found: {list(row.keys())}"
                )
            prompts.append(dict(row))
    return prompts


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed(jsonl_path):
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
# Record builder
# ---------------------------------------------------------------------------

def build_record(prompt_row, run_idx, mode, result,
                 entropy_cascade, density_cascade, kl_cascade,
                 stress_cascade, per_token_entropy,
                 response_text, elapsed):
    """Assemble one JSONL record from all available data."""

    prompt_text = prompt_row.get("prompt", "")
    rec = {
        "prompt": prompt_text,
        "category": prompt_row.get("category", ""),
        "harmfulness": prompt_row.get("harmfulness",
                                      prompt_row.get("category", "")),
        "register": prompt_row.get("register", ""),
        "matched_pair_id": prompt_row.get("matched_pair_id", ""),
        "run_idx": run_idx,
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 2),
        "prompt_hash": hashlib.sha256(prompt_text.encode()).hexdigest()[:16],
    }

    # --- Analyzer scalars ---
    if result is not None:
        for key in ["seq_len", "stress_score", "net_correction",
                     "entropy", "kl_divergence", "delta_scale",
                     "n_negative_tokens", "interior_cv",
                     "top2_share", "middle_share"]:
            val = getattr(result, key, None)
            rec[key] = float(val) if val is not None else None

        # SFD scalars
        if result.sfd is not None:
            for key in ["density_mean", "density_max", "density_var",
                         "density_p90"]:
                val = getattr(result.sfd, key, None)
                rec[f"sfd_{key}"] = float(val) if val is not None else None

        # Rank displacement scalars
        rd = result.rank_displacement
        if rd is not None:
            if isinstance(rd, dict):
                rec["rd_mean_tau"] = rd.get("mean_tau")
                rec["rd_mean_overlap"] = rd.get("mean_overlap")
                rec["rd_replacement_ratio"] = rd.get("replacement_ratio")
            else:
                rec["rd_mean_tau"] = getattr(rd, "mean_tau", None)
                rec["rd_mean_overlap"] = getattr(rd, "mean_overlap", None)
                rec["rd_replacement_ratio"] = getattr(rd, "replacement_ratio", None)

    # --- Per-token traces (for offline re-analysis) ---
    if result is not None:
        rec["per_token_kl"] = _safe_list(result.per_token_kl)
        rec["per_token_stress"] = _safe_list(result.per_token_stress)
        if result.sfd is not None:
            rec["per_token_density"] = _safe_list(
                getattr(result.sfd, "per_token_density", None)
            )

    if per_token_entropy is not None:
        rec["per_token_entropy"] = _safe_list(per_token_entropy)

    # --- Cascade detection (offline) ---
    for name, cascade in [("entropy_cascade", entropy_cascade),
                           ("density_cascade", density_cascade),
                           ("kl_cascade", kl_cascade),
                           ("stress_cascade", stress_cascade)]:
        if cascade is not None:
            # Summary (always stored)
            rec[name] = {
                "n_interventions": cascade["n_interventions"],
                "n_tokens": cascade["n_tokens"],
                "intervention_rate": cascade["intervention_rate"],
                "max_signal": cascade["max_signal"],
                "mean_signal": cascade["mean_signal"],
            }
            # Per-token (stored under a separate key to keep summaries small)
            rec[f"{name}_trace"] = {
                "signal": cascade["per_token_signal"],
                "sigma": cascade["per_token_sigma"],
            }

    if response_text is not None:
        rec["response_text"] = response_text

    return rec


def _safe_list(arr):
    if arr is None:
        return None
    try:
        return [float(x) for x in arr]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_benchmark(prompts, pipeline, analyzer, args):
    """Run the full benchmark."""
    import numpy as np

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    # Resume
    completed = load_completed(str(jsonl_path)) if args.resume else set()
    if completed:
        print(f"Resuming: {len(completed)} records already completed.\n")

    n_runs = args.runs if args.mode == "generate" else 1
    total = len(prompts) * n_runs
    done = 0
    skipped = 0

    # Precompute SFD cache
    sfd_cache = None
    if not args.no_sfd:
        try:
            from src.engine.sfd import precompute_sfd_cache
            sfd_cache = precompute_sfd_cache(analyzer)
            print("SFD cache ready.\n")
        except Exception as e:
            print(f"Warning: SFD cache failed: {e}")
            print("Density cascade detection unavailable.\n")

    with open(jsonl_path, "a") as out_f:
        for run_idx in range(n_runs):
            for prompt_row in prompts:
                prompt_text = prompt_row["prompt"]
                prompt_hash = hashlib.sha256(
                    prompt_text.encode()
                ).hexdigest()[:16]

                if (prompt_hash, run_idx) in completed:
                    skipped += 1
                    done += 1
                    continue

                t0 = time.time()
                response_text = None
                entropy_cascade = None
                density_cascade = None
                kl_cascade = None
                stress_cascade = None
                per_token_entropy = None
                response_text = None

                try:
                    # --- Text to analyze ---
                    if args.mode == "generate":
                        response_text = generate_response(
                            pipeline, prompt_text, args
                        )
                        analysis_text = response_text
                    else:
                        analysis_text = prompt_text

                    # --- Analyzer pass ---
                    result = analyzer.analyze_prompt(
                        analysis_text,
                        compute_sfd=not args.no_sfd,
                        compute_ltp=not args.no_ltp,
                        compute_kl=not args.no_kl,
                        full_capture=True,
                    )

                    # --- Per-token entropy (separate forward pass) ---
                    per_token_entropy = None
                    if not args.no_entropy:
                        try:
                            per_token_entropy = compute_per_token_entropy(
                                pipeline, analysis_text
                            )
                        except Exception as e:
                            print(f"  entropy pass failed: {e}")

                    # --- Cascade detection on all available traces ---

                    entropy_cascade = None
                    if per_token_entropy is not None:
                        entropy_cascade = run_cascade_detection(
                            per_token_entropy,
                            n_scales=args.n_scales,
                            deadband=args.deadband,
                            agreement=args.agreement,
                        )

                    density_cascade = None
                    if (result.sfd is not None
                            and hasattr(result.sfd, "per_token_density")
                            and result.sfd.per_token_density is not None):
                        density_cascade = run_cascade_detection(
                            result.sfd.per_token_density,
                            n_scales=args.n_scales,
                            deadband=args.deadband,
                            agreement=args.agreement,
                            negate=True,  # collapse → rise
                        )

                    kl_cascade = None
                    if result.per_token_kl is not None:
                        kl_cascade = run_cascade_detection(
                            [float(x) for x in result.per_token_kl],
                            n_scales=args.n_scales,
                            deadband=args.deadband,
                            agreement=args.agreement,
                        )

                    stress_cascade = None
                    if result.per_token_stress is not None:
                        stress_cascade = run_cascade_detection(
                            [float(x) for x in result.per_token_stress],
                            n_scales=args.n_scales,
                            deadband=args.deadband,
                            agreement=args.agreement,
                        )

                    elapsed = time.time() - t0

                    rec = build_record(
                        prompt_row=prompt_row,
                        run_idx=run_idx,
                        mode=args.mode,
                        result=result,
                        entropy_cascade=entropy_cascade,
                        density_cascade=density_cascade,
                        kl_cascade=kl_cascade,
                        stress_cascade=stress_cascade,
                        per_token_entropy=per_token_entropy,
                        response_text=response_text,
                        elapsed=elapsed,
                    )

                    out_f.write(json.dumps(rec, default=str) + "\n")
                    out_f.flush()

                except Exception as e:
                    elapsed = time.time() - t0
                    error_rec = {
                        "prompt_hash": prompt_hash,
                        "run_idx": run_idx,
                        "prompt": prompt_text,
                        "category": prompt_row.get("category", ""),
                        "error": str(e),
                        "elapsed_s": round(elapsed, 2),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    out_f.write(json.dumps(error_rec) + "\n")
                    out_f.flush()
                    print(f"  ERROR: {e}")
                    done += 1
                    continue

                done += 1
                cat = prompt_row.get("category", "?")
                rate_parts = []
                if entropy_cascade:
                    rate_parts.append(
                        f"ent={entropy_cascade['intervention_rate']:.0%}"
                    )
                if density_cascade:
                    rate_parts.append(
                        f"den={density_cascade['intervention_rate']:.0%}"
                    )
                if kl_cascade:
                    rate_parts.append(
                        f"kl={kl_cascade['intervention_rate']:.0%}"
                    )
                rates = " ".join(rate_parts) if rate_parts else ""
                print(f"[{done}/{total}] r{run_idx} {cat:<14s} "
                      f"{elapsed:.1f}s {rates}")

    print(f"\nDone. {done} processed, {skipped} skipped (resume).")
    print(f"Results: {jsonl_path}")

    write_summary(str(jsonl_path), str(out_dir / "summary.json"))


# ---------------------------------------------------------------------------
# Generation (mode=generate)
# ---------------------------------------------------------------------------

def generate_response(pipeline, prompt_text, args):
    """Generate a single response using the instruct model."""
    import torch

    tokenizer = pipeline.tokenizer
    model = pipeline.active_model

    # Apply chat template if the tokenizer supports it
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        tokens = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    else:
        tokens = tokenizer.encode(prompt_text, return_tensors="pt")

    tokens = tokens.to(next(model.parameters()).device)

    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )

    # Decode only the new tokens
    response_ids = output[0][tokens.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(jsonl_path, summary_path):
    """Per-category summary statistics."""
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

    by_cat = {}
    for rec in records:
        cat = rec.get("category", "unknown")
        by_cat.setdefault(cat, []).append(rec)

    summary = {"n_total": len(records), "categories": {}}

    for cat, recs in sorted(by_cat.items()):
        cs = {"n": len(recs)}

        for key in ["stress_score", "kl_divergence", "sfd_density_mean",
                     "rd_mean_tau", "rd_mean_overlap", "entropy",
                     "delta_scale"]:
            vals = [r[key] for r in recs if r.get(key) is not None]
            if vals:
                cs[key] = {
                    "mean": round(sum(vals) / len(vals), 4),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "n": len(vals),
                }

        for channel in ["entropy_cascade", "density_cascade",
                         "kl_cascade", "stress_cascade"]:
            rates = [r[channel]["intervention_rate"]
                     for r in recs if r.get(channel)]
            maxsigs = [r[channel]["max_signal"]
                       for r in recs if r.get(channel)]
            if rates:
                cs[channel] = {
                    "mean_rate": round(sum(rates) / len(rates), 4),
                    "max_rate": round(max(rates), 4),
                    "mean_max_sig": round(
                        sum(maxsigs) / len(maxsigs), 4
                    ) if maxsigs else 0,
                    "n": len(rates),
                }

        summary["categories"][cat] = cs

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    header = (f"{'Category':<15} {'n':>4} {'stress':>8} {'KL':>8} "
              f"{'density':>8} {'ent%':>6} {'den%':>6} {'kl%':>6}")
    print(f"\n{header}")
    print("-" * len(header))
    for cat, s in sorted(summary["categories"].items()):
        n = s["n"]
        stress = s.get("stress_score", {}).get("mean", float("nan"))
        kl = s.get("kl_divergence", {}).get("mean", float("nan"))
        density = s.get("sfd_density_mean", {}).get("mean", float("nan"))
        er = s.get("entropy_cascade", {}).get("mean_rate", float("nan"))
        dr = s.get("density_cascade", {}).get("mean_rate", float("nan"))
        kr = s.get("kl_cascade", {}).get("mean_rate", float("nan"))
        print(f"{cat:<15} {n:>4} {stress:>8.3f} {kl:>8.3f} "
              f"{density:>8.3f} {er:>5.0%} {dr:>5.0%} {kr:>5.0%}")

    print(f"\nSummary: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="TAGM benchmark harness — offline cascade detection"
    )

    # Required
    p.add_argument("--prompts", required=True,
                   help="Prompt CSV (needs 'prompt' column)")

    # Mode
    p.add_argument("--mode", choices=["prompt", "generate"],
                   default="prompt")
    p.add_argument("--out", default="benchmark_out")
    p.add_argument("--runs", type=int, default=1,
                   help="Runs per prompt (mode=generate)")
    p.add_argument("--resume", action="store_true")

    # Model pair (optional — uses whatever's loaded if omitted)
    p.add_argument("--instruct", default=None,
                   help="Instruct model HF ID")
    p.add_argument("--base", default=None,
                   help="Base model HF ID")

    # Detector
    p.add_argument("--deadband", type=float, default=0.75)
    p.add_argument("--agreement", type=int, default=2)
    p.add_argument("--n-scales", type=int, default=5)

    # Skip flags
    p.add_argument("--no-sfd", action="store_true")
    p.add_argument("--no-ltp", action="store_true")
    p.add_argument("--no-kl", action="store_true")
    p.add_argument("--no-entropy", action="store_true",
                   help="Skip per-token entropy forward pass")

    # Generation
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)

    args = p.parse_args()

    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    cats = {}
    for pr in prompts:
        cat = pr.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")
    print()

    # Load or connect to model pair
    if args.instruct and args.base:
        pipeline, analyzer = load_model_pair(args.instruct, args.base)
    else:
        # Try importing from a running TAGM instance
        try:
            from src.core.pipeline import Pipeline
            from src.engine.analyzer import Analyzer
            print("Attempting standalone model load...")
            print("Specify --instruct and --base, or load via the TAGM UI.\n")
            print("Example:")
            print("  python tools/benchmark_harness.py \\")
            print("    --prompts prompts_200_harm.csv \\")
            print("    --instruct meta-llama/Llama-3.2-1B-Instruct \\")
            print("    --base meta-llama/Llama-3.2-1B \\")
            print("    --mode prompt --out benchmark_out/harm_200\n")
            sys.exit(1)
        except ImportError as e:
            print(f"Import failed: {e}")
            sys.exit(1)

    run_benchmark(prompts, pipeline, analyzer, args)


if __name__ == "__main__":
    main()
