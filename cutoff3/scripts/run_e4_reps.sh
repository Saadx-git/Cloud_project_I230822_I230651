#!/usr/bin/env bash
# Final-paper repetitions of E4 (live). Seed 1 runs are the C3 runs (no tag); persistence@100m seed 1 added here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
[ -f results/e4/bursty-persistence-100m/config.json ] || \
  $PY experiments/e4_live.py --capacity 4.0 --runs persistence@100m 2>&1 | tee -a results/e4_reps.log
for s in 2 3; do
  $PY experiments/e4_live.py --capacity 4.0 --seed $s --tag=-s$s \
      --runs hpa,prophet_lstm@100m,prophet_lstm@500m,persistence@100m 2>&1 | tee -a results/e4_reps.log
done
$PY experiments/e4_live.py --capacity 4.0 --seed 2 --tag=-s2 --runs prophet@500m 2>&1 | tee -a results/e4_reps.log
echo ALL_DONE | tee -a results/e4_reps.log
