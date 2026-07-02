"""ECM v2 ablation — resumable, timeout-tolerant.

Conditions: baseline (T=base), fixed_mid (T=0.3), fixed_low (T=0.1),
ecm (adaptive, v2 defaults). Default conditions are just
baseline + ecm — the core contrast. Add the fixed conditions only for
the diversity comparison.

Designed for Codespaces that idle out:
  - Every generation is appended to results.jsonl IMMEDIATELY. Kill it,
    let the Codespace die, whatever — nothing done is lost.
  - On restart it resumes: completed (prompt, condition, run) triples
    are skipped.
  - Loop order is run -> prompt -> condition, so an interrupted job
    leaves a BALANCED dataset (every prompt x condition at runs 0..k)
    instead of five runs of the first half of the prompt list.
  - --summarize-only recomputes the stats from whatever is in the
    JSONL without generating anything.

Staged protocol (recommended):
  Stage A (fits one sitting): --runs 2 --max-tokens 96
      -> cascade-by-category CIs, intervention profile. Moves E1/E3.
  Stage B (resume later):     --runs 5 --max-tokens 96
      -> same command, more runs; resume fills runs 2-4 only.
  Stage C (only if needed):   add --conditions baseline fixed_low ecm
      and --max-tokens 200 on the interesting prompts (--prompts a
      trimmed CSV) for the diversity claim (E4).

Run from the repo root:
    python tools/ecm_ablation_v2.py \
        --model meta-llama/Llama-3.2-1B-Instruct --runs 2 --max-tokens 96
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_prompts(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("prompt")]
    for r in rows:
        r.setdefault("category", "uncategorized")
    return rows


def bootstrap_ci(values: list[float], n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean. Returns (mean, lo, hi)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return mean, lo, hi


def read_done(jsonl: Path) -> tuple[list[dict], set]:
    """Load completed records and their (prompt, condition, run) keys."""
    records, done = [], set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a hard kill — redo it
            records.append(r)
            done.add((r["prompt"], r["condition"], r["run"]))
    return records, done


def summarize(records: list[dict], conditions: list[str],
              runs: int, out_dir: Path) -> None:
    if not records:
        print("No records yet.")
        return

    # 1. Cascade signal by category (ecm condition), bootstrap CI over
    #    prompt-level means (the paper's unit of analysis).
    by_prompt: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        if r["condition"] == "ecm" and "ecm_summary" in r:
            by_prompt[(r["category"], r["prompt"])].append(
                r["ecm_summary"]["mean_cascade_signal"])
    by_cat: dict[str, list[float]] = defaultdict(list)
    for (cat, _), vals in by_prompt.items():
        by_cat[cat].append(sum(vals) / len(vals))

    print("\n== Cascade signal by category (v2, sigma units, 95% bootstrap CI) ==")
    cat_summary = {}
    for cat, vals in sorted(by_cat.items()):
        m, lo, hi = bootstrap_ci(vals)
        cat_summary[cat] = {"mean": m, "ci_lo": lo, "ci_hi": hi,
                            "n_prompts": len(vals)}
        print(f"  {cat:<14} mean={m:.3f}  CI[{lo:.3f}, {hi:.3f}]  "
              f"(n={len(vals)} prompts)")

    # 2. Response diversity: unique responses per prompt per condition.
    uniq: dict[tuple, set] = defaultdict(set)
    n_runs_seen: dict[tuple, int] = defaultdict(int)
    for r in records:
        uniq[(r["condition"], r["prompt"])].add(r["response"])
        n_runs_seen[(r["condition"], r["prompt"])] += 1
    print("\n== Mean unique responses per prompt ==")
    div_summary = {}
    for cond in conditions:
        pairs = [(len(v), n_runs_seen[(c, p)])
                 for (c, p), v in uniq.items() if c == cond]
        if not pairs:
            continue
        m = sum(u for u, _ in pairs) / len(pairs)
        collapsed = sum(1 for u, nr in pairs if u == 1 and nr > 1)
        div_summary[cond] = {"mean_unique": m, "n_collapsed": collapsed,
                             "n_prompts": len(pairs)}
        print(f"  {cond:<9} mean_unique={m:.2f}  "
              f"collapsed_to_1={collapsed}/{len(pairs)} prompts "
              f"(only meaningful once runs >= 2)")

    # 3. ECM intervention profile by category.
    print("\n== ECM intervention profile by category ==")
    ecm_prof = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["condition"] == "ecm" and "ecm_summary" in r:
            s = r["ecm_summary"]
            ecm_prof[r["category"]]["rate"].append(s["intervention_rate"])
            ecm_prof[r["category"]]["mean_temp"].append(s["mean_temp"])
            ecm_prof[r["category"]]["loops"].append(s["n_loop_releases"])
    for cat, d in sorted(ecm_prof.items()):
        n = len(d["rate"])
        print(f"  {cat:<14} rate={sum(d['rate'])/n:.3f}  "
              f"mean_T={sum(d['mean_temp'])/n:.3f}  "
              f"loop_releases={sum(d['loops'])}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {"n_records": len(records),
         "cascade_by_category": cat_summary,
         "diversity": div_summary}, indent=1))
    print(f"\nWrote {out_dir}/summary.json "
          f"(from {len(records)} records)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None,
                    help="HF id or local path of the instruct model")
    ap.add_argument("--prompts", type=Path,
                    default=REPO_ROOT / "tools" / "ablation_prompts.csv")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--base-temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=Path("ablation_v2_out"))
    ap.add_argument("--conditions", nargs="+",
                    default=["baseline", "ecm"],
                    choices=["baseline", "fixed_mid", "fixed_low", "ecm"])
    ap.add_argument("--device", default=None, help="cuda / cpu / mps")
    ap.add_argument("--summarize-only", action="store_true",
                    help="recompute stats from existing results.jsonl")
    ap.add_argument("--no-resume", action="store_true",
                    help="redo everything (default: skip completed work)")
    # v2 ECM parameters — defaults mirror src/engine/config.py
    ap.add_argument("--ecm-gain", type=float, default=0.5)
    ap.add_argument("--ecm-floor", type=float, default=0.1)
    ap.add_argument("--ecm-scales", type=int, default=5)
    ap.add_argument("--ecm-deadband", type=float, default=0.75)
    ap.add_argument("--ecm-agreement", type=int, default=2)
    ap.add_argument("--ecm-no-repeat", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "results.jsonl"
    records, done = read_done(jsonl)
    prompts = load_prompts(args.prompts)

    if args.summarize_only:
        summarize(records, args.conditions, args.runs, args.out)
        return 0

    if args.model is None:
        ap.error("--model is required unless --summarize-only")
    if args.no_resume:
        records, done = [], set()
        jsonl.unlink(missing_ok=True)

    # Done check BEFORE the (slow) model load: if every
    # (prompt, condition, run) triple is complete, say so and summarize.
    planned = {(row["prompt"], cond, run)
               for row in prompts
               for cond in args.conditions
               for run in range(args.runs)}
    remaining = planned - done
    if not remaining:
        print(f"COMPLETE: all {len(planned)} generations for this design "
              f"({len(prompts)} prompts x {len(args.conditions)} conditions "
              f"x {args.runs} runs) are in {jsonl}. Nothing to do.")
        print("To extend: raise --runs, add --conditions, or grow the "
              "prompts CSV, then rerun.")
        summarize(records, args.conditions, args.runs, args.out)
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.engine.ecm import ECMProcessor

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        else "cpu")
    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device).eval()

    total = len(prompts) * len(args.conditions) * args.runs
    todo = total - len(done)
    print(f"{len(prompts)} prompts x {len(args.conditions)} conditions "
          f"x {args.runs} runs = {total} generations "
          f"({len(done)} already done, {todo} to go)\n")

    t_start = time.time()
    n_new = 0
    # run -> prompt -> condition: interruption leaves balanced data.
    with open(jsonl, "a") as fout:
        for run in range(args.runs):
            for row in prompts:
                prompt, category = row["prompt"], row["category"]
                encoded = None  # tokenize lazily, once per prompt
                for cond in args.conditions:
                    if (prompt, cond, run) in done:
                        continue
                    if encoded is None:
                        text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False, add_generation_prompt=True)
                        encoded = tokenizer(
                            text, return_tensors="pt").to(device)

                    torch.manual_seed(args.seed + run)  # paired seeds

                    gen_kwargs = dict(
                        **encoded,
                        max_new_tokens=args.max_tokens,
                        do_sample=True,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.pad_token_id
                                      or tokenizer.eos_token_id,
                    )
                    processor = None
                    if cond == "baseline":
                        gen_kwargs["temperature"] = args.base_temp
                    elif cond == "fixed_mid":
                        gen_kwargs["temperature"] = 0.3
                    elif cond == "fixed_low":
                        gen_kwargs["temperature"] = 0.1
                    elif cond == "ecm":
                        processor = ECMProcessor(
                            temperature=args.base_temp,
                            n_scales=args.ecm_scales,
                            gain=args.ecm_gain,
                            floor=args.ecm_floor,
                            deadband=args.ecm_deadband,
                            agreement=args.ecm_agreement,
                        )
                        gen_kwargs["temperature"] = 1.0
                        gen_kwargs["logits_processor"] = [processor]
                        if args.ecm_no_repeat > 0:
                            gen_kwargs["no_repeat_ngram_size"] = \
                                args.ecm_no_repeat

                    with torch.no_grad():
                        out = model.generate(**gen_kwargs)
                    response = tokenizer.decode(
                        out[0][encoded["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip()

                    rec = {
                        "prompt": prompt, "category": category,
                        "condition": cond, "run": run,
                        "response": response,
                        "n_tokens": int(out.shape[1]
                                        - encoded["input_ids"].shape[1]),
                    }
                    if processor is not None:
                        d = processor.diagnostics_to_dict()
                        rec["ecm"] = d
                        temps = d["per_token_temperature"]
                        sig = d["per_token_cascade_signal"]
                        rec["ecm_summary"] = {
                            "intervention_rate": d["intervention_rate"],
                            "max_cascade_signal": d["max_cascade_signal"],
                            "mean_cascade_signal": (
                                sum(sig) / max(1, len(sig))),
                            "mean_temp": sum(temps) / max(1, len(temps)),
                            "min_temp": min(temps) if temps else None,
                            "n_loop_releases": d["n_loop_releases"],
                        }
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    records.append(rec)
                    done.add((prompt, cond, run))
                    n_new += 1

                    el = time.time() - t_start
                    rate = el / n_new
                    eta = rate * (todo - n_new)
                    print(f"  [{len(done)}/{total}] run{run} "
                          f"{category:<12} {cond:<9} "
                          f"'{prompt[:38]}'  "
                          f"({rate:.0f}s/gen, ETA {eta/60:.0f}m)")

    summarize(records, args.conditions, args.runs, args.out)
    print("\nResumable: rerun the same command to add runs; "
          "--summarize-only to recompute stats without generating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
