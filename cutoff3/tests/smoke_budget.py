"""Smoke test of the budget-enforced forecaster (normal path, deadline miss -> kill/respawn)."""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scaler")]
from budget import BudgetedForecaster

if __name__ == "__main__":
    trace = sys.argv[1] if len(sys.argv) > 1 else "nasa"
    fc = BudgetedForecaster("prophet_lstm", trace, fast=False, on_miss="kill")
    ds = list(pd.date_range("1995-08-20" if trace == "nasa" else "1998-07-02", periods=20, freq="min"))
    yh = list(np.linspace(20, 40, 20))
    r = fc.forecast(ds, yh, 0.5)
    print("during cold start ->", r["reason"], r["forecast"])
    fc._poll_ready(300)
    print("cold start:", fc.cold_starts[-1])
    for _ in range(3):
        r = fc.forecast(ds, yh, 15.0)
        print("on time:", r["late"], round(r["forecast"], 2), "wall_ms", round(r["wall_ms"], 1), "cpu_ms", round(r["cpu_ms"], 1))
    r = fc.forecast(ds, yh, 0.005)
    print("tight deadline ->", r["reason"], "late", r["late"], "fallback value", r["forecast"])
    fc._poll_ready(300)
    r = fc.forecast(ds, yh, 15.0)
    print("after respawn: late", r["late"], round(r["forecast"], 2), "cold starts so far", len(fc.cold_starts))
    fc.kill()
