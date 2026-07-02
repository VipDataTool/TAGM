"""ECM v2 ablation — reruns the paper's experiment under the v2 mechanism.

Conditions: baseline (T=base), fixed_mid (T=0.3), fixed_low (T=0.1),
ecm (adaptive, v2 defaults from src/engine/config.py unless overridden).

For each prompt x condition x run: generate, record the response, and
(for the ecm condition) capture full ECMProcessor diagnostics. Outputs
per-category cascade-signal means with percentile-bootstrap CIs,
response-diversity counts, and effective-temperature statistics.

Run from the repo root inside the Codespace (needs torch + transformers):

    python tools/ecm_ablation_v2.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --prompts tools/ablation_prompts.csv \
        --runs 5 --max-tokens 200 --out ablation_v2_out

Seeding: run r of every condition uses the same seed, so diversity
differences between conditions are attributable to the condition, not
the noise draw.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="HF id or local path of the instruct model")
    ap.add_argument("--prompts", type=Path,
                    default=REPO_ROOT / "tools" / "ablation_prompts.csv")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--base-temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=Path("ablation_v2_out"))
    ap.add_argument("--conditions", nargs="+",
                    default=["baseline", "fixed_mid", "fixed_low", "ecm"])
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / mps (default: auto)")
    # v2 ECM parameters — defaults mirror src/engine/config.py
    ap.add_argument("--ecm-gain", type=float, default=0.5)
    ap.add_argument("--ecm-floor", type=float, default=0.1)
    ap.add_argument("--ecm-scales", type=int, default=5)
    ap.add_argument("--ecm-deadband", type=float, default=0.75)
    ap.add_argument("--ecm-agreement", type=int, default=2)
    ap.add_argument("--ecm-no-repeat", type=int, default=4)
    args = ap.parse_args()

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

    prompts = load_prompts(args.prompts)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(prompts)} prompts x {len(args.conditions)} conditions "
          f"x {args.runs} runs = "
          f"{len(prompts) * len(args.conditions) * args.runs} generations\n")

    records: list[dict] = []
    t_start = time.time()

    for p_idx, row in enumerate(prompts):
        prompt, category = row["prompt"], row["category"]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        for cond in args.conditions:
            for run in range(args.runs):
                torch.manual_seed(args.seed + run)  # paired across conditions

                gen_kwargs = dict(
                    **inputs,
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
                        gen_kwargs["no_repeat_ngram_size"] = args.ecm_no_repeat
                else:
                    raise ValueError(f"unknown condition {cond}")

                with torch.no_grad():
                    out = model.generate(**gen_kwargs)
                response = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True).strip()

                rec = {
                    "prompt": prompt, "category": category,
                    "condition": cond, "run": run,
                    "response": response,
                    "n_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
                }
                if processor is not None:
                    d = processor.diagnostics_to_dict()
                    rec["ecm"] = d
                    temps = d["per_token_temperature"]
                    rec["ecm_summary"] = {
                        "intervention_rate": d["intervention_rate"],
                        "max_cascade_signal": d["max_cascade_signal"],
                        "mean_cascade_signal": (
                            sum(d["per_token_cascade_signal"])
                            / max(1, len(d["per_token_cascade_signal"]))),
                        "mean_temp": sum(temps) / max(1, len(temps)),
                        "min_temp": min(temps) if temps else None,
                        "n_loop_releases": d["n_loop_releases"],
                    }
                records.append(rec)
            done = len(records)
            total = len(prompts) * len(args.conditions) * args.runs
            print(f"  [{done}/{total}] {category:<12} {cond:<9} "
                  f"'{prompt[:44]}'  ({time.time() - t_start:.0f}s)")

    (args.out / "results.json").write_text(json.dumps(records, indent=1))

    # ── Summaries ───────────────────────────────────────────────
    # 1. Cascade signal by category (ecm condition), bootstrap CI over
    #    prompt-level means (the paper's unit of analysis).
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_prompt: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        if r["condition"] == "ecm":
            by_prompt[(r["category"], r["prompt"])].append(
                r["ecm_summary"]["mean_cascade_signal"])
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
    for r in records:
        uniq[(r["condition"], r["prompt"])].add(r["response"])
    print("\n== Mean unique responses per prompt (of "
          f"{args.runs} runs) ==")
    div_summary = {}
    for cond in args.conditions:
        counts = [len(v) for (c, _), v in uniq.items() if c == cond]
        m = sum(counts) / max(1, len(counts))
        collapsed = sum(1 for c in counts if c == 1)
        div_summary[cond] = {"mean_unique": m, "n_collapsed": collapsed}
        print(f"  {cond:<9} mean_unique={m:.2f}  "
              f"collapsed_to_1={collapsed}/{len(counts)} prompts")

    # 3. ECM intervention profile by category.
    print("\n== ECM intervention profile by category ==")
    ecm_prof = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["condition"] == "ecm":
            s = r["ecm_summary"]
            ecm_prof[r["category"]]["rate"].append(s["intervention_rate"])
            ecm_prof[r["category"]]["mean_temp"].append(s["mean_temp"])
            ecm_prof[r["category"]]["loops"].append(s["n_loop_releases"])
    for cat, d in sorted(ecm_prof.items()):
        n = len(d["rate"])
        print(f"  {cat:<14} rate={sum(d['rate'])/n:.3f}  "
              f"mean_T={sum(d['mean_temp'])/n:.3f}  "
              f"loop_releases={sum(d['loops'])}")

    summary = {"config": vars(args) | {"prompts": str(args.prompts),
                                       "out": str(args.out)},
               "cascade_by_category": cat_summary,
               "diversity": div_summary}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nWrote {args.out}/results.json and {args.out}/summary.json")
    print("Per-generation ECM diagnostics are embedded in results.json — "
          "feed individual ones to tools/ecm_ab_report.py for plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
