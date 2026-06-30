#!/usr/bin/env bash
# Run from the TAGM project root:
#   bash run_ablation.sh              # full (3 runs)
#   bash run_ablation.sh --dry-run    # 1 run, quick check
set -e
python ecm_ablation.py "$@"
