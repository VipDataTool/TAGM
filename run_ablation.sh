#!/usr/bin/env bash
# Run the ECM fixed-temperature ablation end-to-end.
# Usage: bash run_ablation.sh
#   or:  bash run_ablation.sh --dry-run   (1 run per condition instead of 3)

set -e

N_RUNS=3
[[ "$1" == "--dry-run" ]] && N_RUNS=1 && echo "Dry run: 1 generation per (prompt, condition)"

python3 -c "
import sys, os
sys.path.insert(0, '.')

# ---- import everything from the ablation file ----
exec(open('ecm_ablation_fixed_temperature.py').read())

# ---- override N_RUNS from bash ----
N_RUNS = ${N_RUNS}

# ---- run ----
print()
print('='*60)
print(f'ECM ABLATION: {len(PROMPTS)} prompts × {len(CONDITIONS)} conditions × {N_RUNS} runs')
print(f'= {len(PROMPTS) * len(CONDITIONS) * N_RUNS} total generations')
print(f'Model: {MODEL_ID}')
print(f'Device: {DEVICE}')
print('='*60)
print()

model, tokenizer = load_model(MODEL_ID, DEVICE)
results = run_ablation(model, tokenizer)

import pandas as pd
from dataclasses import asdict
df = pd.DataFrame([asdict(r) for r in results])
df.to_csv('ecm_ablation_results.csv', index=False)
print(f'\nSaved {len(df)} results to ecm_ablation_results.csv')

# ---- analysis ----
print()
routing_comparison(df)
print()
ecm_telemetry(df)
print()
print('BEHAVIOR SUMMARY:')
print(behavior_summary(df).to_string())
print()
side_by_side(df, 'reframeable', run=0)
side_by_side(df, 'non_reframeable', run=0)

try:
    plot_behavior_grid(df)
except Exception as e:
    print(f'Plot skipped ({e}) — results are in the CSV.')
"
