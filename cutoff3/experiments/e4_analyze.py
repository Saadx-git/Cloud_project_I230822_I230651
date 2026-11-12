"""E4 analysis - per-run provisioning, SLO and controller-overhead metrics from the live runs.

Per trace-minute window i ([t0 + i*step, t0 + (i+1)*step)):
  demand r_i   = offered load (req/s) of the trace minute
  capacity p_i = time-weighted mean of READY replicas in the window * target_per_pod
Then Theta_U/Theta_O/T_U/T_O (pbscale.metrics.provisioning), pod-minutes (ready replicas,
in trace minutes), and request-level SLO: share of requests that failed, timed out or
exceeded the 500 ms latency SLO.  For KEDA runs the scaler's decision log gives
forecast accuracy (t+1, causal, observed in the live loop), per-decision CPU and
latency, late-forecast fraction and cgroup throttling.

Usage: python experiments/e4_analyze.py  -> results/e4_live.json
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pbscale.metrics import accuracy, provisioning  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ready_per_window(rep, t0, step, n):
    rep = rep.sort_values("wall")
    out = np.zeros(n)
    w, r = rep.wall.to_numpy(), rep.ready.fillna(0).to_numpy(float)
    for i in range(n):
        a, b = t0 + i * step, t0 + (i + 1) * step
        grid = np.linspace(a, b, 31)[:-1]
        idx = np.searchsorted(w, grid, side="right") - 1
        out[i] = np.mean(r[np.clip(idx, 0, len(r) - 1)])
    return out


def analyze(d):
    cfg = json.load(open(f"{d}/config.json"))
    load = pd.read_csv(f"{d}/load.csv")
    rep = pd.read_csv(f"{d}/replicas.csv")
    sl = pd.read_csv(f"{ROOT}/results/e4/slice_{cfg['scenario']}.csv")
    n, step, t0 = len(sl), cfg["step"], cfg["t0"]
    demand = sl.y.to_numpy(float) * cfg["scale"]
    ready = ready_per_window(rep, t0, step, n)
    m = provisioning(demand, ready * cfg["target_per_pod"])
    sent = load.sent.sum()
    bad = load.err.sum() + load.timeout.sum() + load.slow.sum()
    m.update(pod_minutes=float(ready.sum()), mean_ready=float(ready.mean()), max_ready=float(ready.max()),
             requests=int(sent), slo_violation_pct=float(100 * bad / sent),
             errors_pct=float(100 * (load.err.sum() + load.timeout.sum()) / sent),
             lat_p95_ms_median_over_s=float(load.lat_p95_ms.median()),
             lat_p95_ms_p95_over_s=float(load.lat_p95_ms.quantile(0.95)))
    out = dict(run=os.path.basename(d.rstrip("/")), policy=cfg["policy"], metrics=m, scaler_ready_after_s=cfg.get("scaler_ready_after_s"),
               series=dict(demand_rps=demand.tolist(), ready=ready.tolist()))
    if os.path.exists(f"{d}/scaler.log"):
        recs = [json.loads(l) for l in open(f"{d}/scaler.log") if l.startswith("{")]
        dec = pd.DataFrame([r for r in recs if r.get("event") == "decision"])
        boot = [r for r in recs if r.get("event") == "boot"]
        if len(dec):
            dec = dec[(dec.k >= 1) & (dec.k <= n)]
            # forecast made at tick k (end of minute k-1) is for minute k
            f = np.full(n, np.nan)
            for k, v in zip(dec.k, dec.desired_rps):
                if k < n:
                    f[k] = v
            obs = np.full(n, np.nan)
            for k, v in zip(dec.k, dec.observed_rps):
                obs[k - 1] = v
            cpu = dec.cpu_ms.dropna()
            out["scaler"] = dict(
                decisions=int(len(dec)), late_pct=float(100 * dec.late.mean()), fallback_pct=float(100 * dec.fallback.mean()),
                reasons=dec.reason.fillna("ok").value_counts().to_dict(),
                forecast_cpu_ms_mean=float(cpu.mean()) if len(cpu) else None,
                forecast_cpu_ms_p95=float(cpu.quantile(.95)) if len(cpu) else None,
                forecast_wall_ms_p50=float(dec.wall_ms.median()), forecast_wall_ms_p95=float(dec.wall_ms.quantile(.95)),
                forecast_wall_ms_max=float(dec.wall_ms.max()),
                container_cpu_ms_per_period_mean=float(dec.cg_usage_ms.mean()) if "cg_usage_ms" in dec else None,
                container_throttled_ms_per_period_mean=float(dec.cg_throttled_ms.mean()) if "cg_throttled_ms" in dec else None,
                container_mem_mb_max=float(dec.cg_mem_mb.max()) if "cg_mem_mb" in dec else None,
                cpu_limit_cores=boot[0]["cgroup"].get("cpu_limit_cores") if boot else None,
                accuracy_vs_offered=accuracy(demand[1:], f[1:]),
                accuracy_vs_observed=accuracy(obs[1:], f[1:]),
                persistence_vs_observed=accuracy(obs[1:], obs[:-1]))
            out["series"]["forecast_rps"] = [None if np.isnan(x) else float(x) for x in f]
            out["series"]["observed_rps"] = [None if np.isnan(x) else float(x) for x in obs]
    return out


if __name__ == "__main__":
    runs = [analyze(d) for d in sorted(glob.glob(f"{ROOT}/results/e4/*/")) if os.path.exists(f"{d}/config.json")]
    json.dump(runs, open(f"{ROOT}/results/e4_live.json", "w"), indent=1)
    for r in runs:
        m = r["metrics"]
        s = r.get("scaler", {})
        print(f"{r['run']:35s} ThU={m['Theta_U']:6.2f} ThO={m['Theta_O']:7.2f} TU={m['T_U']:6.2f} TO={m['T_O']:6.2f} "
              f"podmin={m['pod_minutes']:7.1f} SLOviol={m['slo_violation_pct']:6.2f}% "
              f"late={s.get('late_pct')} cpu={s.get('forecast_cpu_ms_mean')} wall95={s.get('forecast_wall_ms_p95')}")
