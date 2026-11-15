#!/usr/bin/env bash
# Final-paper extension of E2: CPU limits below 100m (50m, 25m) to locate where late forecasts start.
set -euo pipefail
cd "$(dirname "$0")/.."
venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "experiments")
import e2_overhead as e2
out = f"{e2.ROOT}/results/e2"
for budget in ["0.05", "0.025"]:
    for model, fast in [("persistence", 0), ("prophet_lstm", 1), ("prophet_lstm", 0)]:
        e2.run(model, fast, budget, 30, 0, "fifa", out)
e2.summarize(out)
PY
