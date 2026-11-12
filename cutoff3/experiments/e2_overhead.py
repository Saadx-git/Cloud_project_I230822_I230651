"""E2 - Controller overhead under CPU budgets (host-side orchestrator).

For each (forecaster, implementation, CPU budget) a fresh container of the
scaler image is started with `docker run --cpus=<budget>` - the same CFS quota
mechanism that enforces `resources.limits.cpu` for a Kubernetes Pod - and
experiments/e2_bench_inside.py is executed in it.

Aggregates into results/e2_overhead.json:
  per configuration: cold start, per-decision latency/CPU percentiles, RSS, throttling, refit cost
  late_fraction[model|fast|budget|period] = P(decision latency > period), the share of control
       periods whose forecast would miss the deadline (deadline = period, 'before the next iteration')
  cold_start_periods[...] = cold-start wall time / period (control periods without a model forecast)

Usage: python experiments/e2_overhead.py [--quick]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = "pbscale/scaler:c3"
BUDGETS = ["0.1", "0.25", "0.5", "1.0", "0"]  # cores; "0" = no limit (all host CPUs)
PERIODS = [15, 30, 60]
CONFIGS = [  # (model, fast)
    ("persistence", 0), ("ar10", 0), ("prophet", 0), ("prophet", 1), ("prophet_lstm", 0), ("prophet_lstm", 1)]


def run(model, fast, budget, n, refit, trace, outdir):
    tag = f"{model}-f{fast}-c{budget}-{trace}"
    out = f"{outdir}/raw/{tag}.json"
    if os.path.exists(out):
        return out
    cmd = ["docker", "run", "--rm", "--memory=2g", "-e", "PBSCALE_DATA=/data", "-e", "HOME=/tmp",
           "--user", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{ROOT}/upstream/repo/datasets:/data:ro", "-v", f"{outdir}/raw:/out"]
    if budget != "0":
        cmd += [f"--cpus={budget}"]
    cmd += [IMAGE, "python", "/app/experiments/e2_bench_inside.py", "--model", model, "--trace", trace,
            "--fast", str(fast), "--n", str(n), "--refit", str(refit), "--out", f"/out/{tag}.json"]
    print(" ".join(cmd[-14:]), flush=True)
    subprocess.run(cmd, check=True)
    return out


def summarize(outdir):
    rows, late, cold = [], {}, {}
    for p in sorted(glob.glob(f"{outdir}/raw/*.json")):
        r = json.load(open(p))
        w, c = np.array(r["decisions"]["wall_ms"]), np.array(r["decisions"]["cpu_ms"])
        budget = r["cpu_limit_cores"] or 0
        cs = r["cold_start"]
        row = dict(model=r["model"], fast=r["fast"], trace=r["trace"], budget_cores=budget, n=len(w),
                   lat_p50_ms=float(np.median(w)), lat_p95_ms=float(np.percentile(w, 95)),
                   lat_max_ms=float(w.max()), cpu_p50_ms=float(np.median(c)), cpu_mean_ms=float(c.mean()),
                   cold_load_wall_s=cs["load_wall_ms"] / 1000, cold_first_call_wall_s=cs["first_call_wall_ms"] / 1000,
                   cold_total_wall_s=(cs["load_wall_ms"] + cs["first_call_wall_ms"]) / 1000,
                   cold_cpu_s=(cs["load_cpu_ms"] + cs["first_call_cpu_ms"]) / 1000,
                   rss_mb=cs["rss_mb"], peak_rss_mb=r.get("peak_rss_mb"),
                   throttled_pct=(100 * r["cgroup"]["nr_throttled"] / max(r["cgroup"]["nr_periods"], 1)
                                  if r.get("cgroup") else None),
                   refit=r.get("refit"))
        rows.append(row)
        for P in PERIODS:
            k = f"{r['model']}|{int(r['fast'])}|{budget}|{P}"
            late[k] = float(np.mean(w / 1000 > P))
            cold[k] = row["cold_total_wall_s"] / P
            # share of a period's CPU budget consumed by one forecast
            if budget:
                row[f"budget_share_P{P}_pct"] = 100 * row["cpu_mean_ms"] / (budget * P * 1000)
    json.dump(dict(rows=rows, late_fraction=late, cold_start_periods=cold, periods=PERIODS),
              open(f"{ROOT}/results/e2_overhead.json", "w"), indent=1)
    for r in rows:
        print(f"{r['model']:13s} fast={int(r['fast'])} cpu={r['budget_cores']:<5} p50={r['lat_p50_ms']:9.1f}ms "
              f"p95={r['lat_p95_ms']:9.1f}ms cpu={r['cpu_mean_ms']:8.1f}ms cold={r['cold_total_wall_s']:7.1f}s "
              f"rss={r['rss_mb']:6.0f}MB thr={r['throttled_pct'] if r['throttled_pct'] is None else round(r['throttled_pct'],1)}"
              f" refit={r['refit']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--trace", default="fifa")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    outdir = f"{ROOT}/results/e2"
    os.makedirs(f"{outdir}/raw", exist_ok=True)
    if not a.summarize_only:
        for budget in BUDGETS:
            for model, fast in CONFIGS:
                refit = int(model == "prophet_lstm" and fast == 0 and budget in ("0.1", "0.5", "1.0"))
                run(model, fast, budget, a.n, refit, a.trace, outdir)
    summarize(outdir)
