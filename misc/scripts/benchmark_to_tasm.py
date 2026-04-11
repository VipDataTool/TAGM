#!/usr/bin/env python3
"""
benchmark_to_tasm.py — Convert standard safety benchmarks to TASM batch CSV format.

Output: prompt,category

Supported benchmarks:
  - HarmBench (text behaviors)
  - AdvBench (harmful behaviors)
  - XSTest (safe + unsafe prompts)
  - JailbreakBench (harmful + benign)

Usage:
  # Step 1: Fetch the raw data (run once)
  python benchmark_to_tasm.py --fetch

  # Step 2: Convert individual benchmarks
  python benchmark_to_tasm.py --source harmbench
  python benchmark_to_tasm.py --source advbench
  python benchmark_to_tasm.py --source xstest
  python benchmark_to_tasm.py --source jailbreakbench

  # Step 3: Build a combined benchmark (recommended for TASM)
  python benchmark_to_tasm.py --source harmbench+xstest
  python benchmark_to_tasm.py --source all

  # Options
  --output FILENAME       Output CSV path (default: tasm_benchmark_{source}.csv)
  --functional standard   HarmBench only: filter by FunctionalCategory (default: standard)
  --safe-label benign     Label to assign to safe/benign prompts (default: benign)
  --skip-context          Skip contextual behaviors that require a ContextString
"""

import argparse
import csv
import os
import sys
import urllib.request
import json

# ─── Raw data URLs ───────────────────────────────────────────────────────────

URLS = {
    "harmbench_test": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
    "harmbench_val": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_val.csv",
    "harmbench_all": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_cache")


# ─── Fetch ───────────────────────────────────────────────────────────────────

def fetch_all():
    """Download raw benchmark CSVs to local cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name, url in URLS.items():
        dest = os.path.join(CACHE_DIR, f"{name}.csv")
        if os.path.exists(dest):
            print(f"  [skip] {name} — already cached")
            continue
        print(f"  [fetch] {name} ← {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  [ok]    → {dest}")
        except Exception as e:
            print(f"  [FAIL]  {name}: {e}")
    
    # JailbreakBench requires the Python package
    jbb_path = os.path.join(CACHE_DIR, "jailbreakbench_harmful.csv")
    jbb_benign_path = os.path.join(CACHE_DIR, "jailbreakbench_benign.csv")
    if not os.path.exists(jbb_path):
        print(f"  [info] JailbreakBench requires the jailbreakbench package.")
        print(f"         Install: pip install jailbreakbench")
        print(f"         Then re-run: python {sys.argv[0]} --fetch")
        try:
            import jailbreakbench as jbb
            # Harmful
            dataset = jbb.read_dataset()
            with open(jbb_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Goal", "Behavior", "Category"])
                for item in dataset.behaviors:
                    w.writerow([item.Goal, item.Behavior, item.Category])
            print(f"  [ok]    → {jbb_path}")
            # Benign
            dataset_b = jbb.read_dataset("benign")
            with open(jbb_benign_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Goal", "Behavior", "Category"])
                for item in dataset_b.behaviors:
                    w.writerow([item.Goal, item.Behavior, item.Category])
            print(f"  [ok]    → {jbb_benign_path}")
        except ImportError:
            print(f"  [skip] jailbreakbench not installed — skipping")
        except Exception as e:
            print(f"  [FAIL] jailbreakbench: {e}")

    print(f"\nCache directory: {CACHE_DIR}")


# ─── Converters ──────────────────────────────────────────────────────────────

def load_cached(name):
    path = os.path.join(CACHE_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run with --fetch first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def convert_harmbench(functional="standard", skip_context=False, split="test"):
    """
    HarmBench → TASM
    Columns: Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID
    Maps: Behavior → prompt, SemanticCategory → category
    """
    if split == "all":
        rows = load_cached("harmbench_all")
    elif split == "val":
        rows = load_cached("harmbench_val")
    else:
        rows = load_cached("harmbench_test")
    
    out = []
    for r in rows:
        # Filter by functional category
        fc = r.get("FunctionalCategory", "").strip()
        if functional and fc != functional:
            # Allow contextual if not skipping
            if skip_context or fc == "copyright":
                continue
            if functional == "standard" and fc == "contextual" and not skip_context:
                pass  # include contextual unless --skip-context
            elif functional != fc:
                continue
        
        behavior = r.get("Behavior", "").strip()
        category = r.get("SemanticCategory", "harmful").strip()
        context = r.get("ContextString", "").strip()
        
        if not behavior:
            continue
        
        # For contextual behaviors, prepend context if present
        if context and not skip_context:
            prompt = f"[Context: {context[:200]}...] {behavior}" if len(context) > 200 else f"[Context: {context}] {behavior}"
        else:
            if context and skip_context:
                continue
            prompt = behavior
        
        out.append({"prompt": prompt, "category": category})
    
    return out


def convert_advbench(safe_label="benign"):
    """
    AdvBench → TASM
    Columns: goal, target
    Maps: goal → prompt, all labeled 'harmful'
    """
    rows = load_cached("advbench")
    out = []
    for r in rows:
        goal = r.get("goal", "").strip()
        if not goal:
            continue
        out.append({"prompt": goal, "category": "harmful"})
    return out


def convert_xstest(safe_label="benign"):
    """
    XSTest → TASM
    Columns: id_v2, type, prompt, note, label
    Maps: prompt → prompt, label/type → category
    
    Safe prompts get safe_label. Unsafe prompts (type starts with 'contrast_')
    get their type as category for granularity.
    """
    rows = load_cached("xstest")
    out = []
    for r in rows:
        prompt = r.get("prompt", "").strip()
        label = r.get("label", "").strip()
        ptype = r.get("type", "").strip()
        
        if not prompt:
            continue
        
        if ptype.startswith("contrast_"):
            # Unsafe prompt — use the type minus 'contrast_' prefix as category
            category = ptype.replace("contrast_", "unsafe_")
        else:
            category = safe_label
        
        out.append({"prompt": prompt, "category": category})
    return out


def convert_jailbreakbench(safe_label="benign"):
    """
    JailbreakBench → TASM
    Columns: Goal, Behavior, Category
    Maps: Goal → prompt, Category → category
    """
    out = []
    
    # Harmful
    path_h = os.path.join(CACHE_DIR, "jailbreakbench_harmful.csv")
    if os.path.exists(path_h):
        with open(path_h, "r") as f:
            for r in csv.DictReader(f):
                goal = r.get("Goal", "").strip()
                cat = r.get("Category", "harmful").strip().lower().replace(" ", "_")
                if goal:
                    out.append({"prompt": goal, "category": cat})
    
    # Benign
    path_b = os.path.join(CACHE_DIR, "jailbreakbench_benign.csv")
    if os.path.exists(path_b):
        with open(path_b, "r") as f:
            for r in csv.DictReader(f):
                goal = r.get("Goal", "").strip()
                if goal:
                    out.append({"prompt": goal, "category": safe_label})
    
    if not out:
        print("Warning: JailbreakBench data not found. Install jailbreakbench and re-run --fetch.")
    
    return out


# ─── Combiners ───────────────────────────────────────────────────────────────

CONVERTERS = {
    "harmbench": lambda args: convert_harmbench(args.functional, args.skip_context),
    "advbench": lambda args: convert_advbench(args.safe_label),
    "xstest": lambda args: convert_xstest(args.safe_label),
    "jailbreakbench": lambda args: convert_jailbreakbench(args.safe_label),
}

def convert_combined(sources, args):
    """Combine multiple benchmark sources, deduplicating by prompt text."""
    all_rows = []
    seen = set()
    
    for source in sources:
        if source not in CONVERTERS:
            print(f"Warning: unknown source '{source}', skipping")
            continue
        rows = CONVERTERS[source](args)
        for r in rows:
            key = r["prompt"].strip().lower()
            if key not in seen:
                seen.add(key)
                all_rows.append(r)
        print(f"  {source}: {len(rows)} prompts ({len(all_rows)} total after dedup)")
    
    return all_rows


# ─── Output ──────────────────────────────────────────────────────────────────

def write_tasm_csv(rows, output_path):
    """Write TASM-format CSV: prompt,category"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "category"])
        w.writeheader()
        w.writerows(rows)
    
    # Summary
    cats = {}
    for r in rows:
        c = r["category"]
        cats[c] = cats.get(c, 0) + 1
    
    print(f"\nWritten: {output_path}")
    print(f"Total prompts: {len(rows)}")
    print(f"Categories ({len(cats)}):")
    for cat in sorted(cats.keys(), key=lambda x: -cats[x]):
        print(f"  {cat:40s} {cats[cat]:5d}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert safety benchmarks to TASM batch CSV format (prompt,category)")
    
    parser.add_argument("--fetch", action="store_true",
                        help="Download raw benchmark data to local cache")
    parser.add_argument("--source", type=str, default=None,
                        help="Benchmark source(s). Options: harmbench, advbench, xstest, "
                             "jailbreakbench, harmbench+xstest, all")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: tasm_benchmark_{source}.csv)")
    parser.add_argument("--functional", type=str, default="standard",
                        help="HarmBench: filter by FunctionalCategory (default: standard)")
    parser.add_argument("--skip-context", action="store_true",
                        help="HarmBench: skip contextual behaviors")
    parser.add_argument("--safe-label", type=str, default="benign",
                        help="Label for safe/benign prompts (default: benign)")
    parser.add_argument("--split", type=str, default="test",
                        help="HarmBench: which split — test, val, or all (default: test)")
    
    args = parser.parse_args()
    
    if args.fetch:
        fetch_all()
        return
    
    if not args.source:
        parser.print_help()
        print("\nExample:")
        print("  python benchmark_to_tasm.py --fetch")
        print("  python benchmark_to_tasm.py --source harmbench+xstest")
        return
    
    # Parse source(s)
    source_str = args.source.lower().strip()
    
    if source_str == "all":
        sources = ["harmbench", "advbench", "xstest", "jailbreakbench"]
    elif "+" in source_str:
        sources = [s.strip() for s in source_str.split("+")]
    else:
        sources = [source_str]
    
    # Convert
    print(f"Converting: {' + '.join(sources)}")
    rows = convert_combined(sources, args)
    
    if not rows:
        print("No prompts converted. Check that data is fetched (--fetch) and source names are correct.")
        return
    
    # Output path
    if args.output:
        output = args.output
    else:
        output = f"tasm_benchmark_{source_str.replace('+', '_')}.csv"
    
    write_tasm_csv(rows, output)


if __name__ == "__main__":
    main()
