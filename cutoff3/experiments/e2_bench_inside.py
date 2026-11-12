"""E2 (inside the container) - controller overhead of one forecaster under a CPU limit.

Run by experiments/e2_overhead.sh as
  docker run --cpus=<budget> -v <datasets>:/data:ro pbscale/scaler:c3 \
      python /app/experiments/e2_bench_inside.py --model prophet_lstm --trace fifa --n 60 --out /out/x.json

Measures, in the same process a controller would use:
  cold start   : import of the model libraries + model load + first call (wall, CPU, RSS)
  per decision : wall latency and CPU time (time.process_time, all threads) of predict_next(),
                 decisions paced every --interval s, inputs = consecutive minutes of the 'bursty' scenario
  cgroup       : CPU usage and CFS throttling of the container over the decision phase
  refit        : (optional) Prophet re-fit on the last 7 days + 1 LSTM epoch, i.e. the cost of
                 keeping the model fresh inside the same budget
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scaler.budget import cgroup_cpu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--trace", default="fifa")
ap.add_argument("--scenario", default="bursty")
ap.add_argument("--fast", type=int, default=0)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--interval", type=float, default=1.0)
ap.add_argument("--refit", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

res = dict(model=a.model, trace=a.trace, fast=bool(a.fast), scenario=a.scenario, n=a.n, interval_s=a.interval)
cg0 = cgroup_cpu()
res["cpu_limit_cores"] = cg0.get("cpu_limit_cores")
w0, c0 = time.perf_counter(), time.process_time()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from pbscale import data  # noqa: E402
from pbscale.forecasters import build  # noqa: E402
proc = psutil.Process()
model = build(a.model, a.trace, bool(a.fast))
t_load_wall, t_load_cpu = time.perf_counter() - w0, time.process_time() - c0
sc = data.scenario_frame(a.scenario)
hist = data.history_before(a.scenario, 1500)
full = pd.concat([hist, sc], ignore_index=True)
k0 = len(hist)
w1, c1 = time.perf_counter(), time.process_time()
model.predict_next(full.ds.iloc[:k0], full.y.to_numpy()[:k0])  # first call (graph build, Stan init ...)
res["cold_start"] = dict(load_wall_ms=1000 * t_load_wall, load_cpu_ms=1000 * t_load_cpu,
                         first_call_wall_ms=1000 * (time.perf_counter() - w1),
                         first_call_cpu_ms=1000 * (time.process_time() - c1),
                         rss_mb=proc.memory_info().rss / 2**20)

lat, cpu = [], []
cg1 = cgroup_cpu()
y = full.y.to_numpy(float)
for i in range(a.n):
    k = k0 + 1 + i
    t_next = time.perf_counter() + a.interval
    w, c = time.perf_counter(), time.process_time()
    model.predict_next(full.ds.iloc[k - 1500:k], y[k - 1500:k])
    lat.append(1000 * (time.perf_counter() - w))
    cpu.append(1000 * (time.process_time() - c))
    time.sleep(max(0.0, t_next - time.perf_counter()))
cg2 = cgroup_cpu()
res["decisions"] = dict(wall_ms=lat, cpu_ms=cpu, rss_mb_after=proc.memory_info().rss / 2**20,
                        peak_rss_mb=proc.memory_info().rss / 2**20 if not hasattr(proc, "memory_full_info")
                        else None)
try:
    import resource
    res["decisions"]["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
except Exception:
    pass
if cg1 and cg2:
    res["cgroup"] = dict(usage_ms=(cg2["usage_usec"] - cg1["usage_usec"]) / 1000,
                         throttled_ms=(cg2["throttled_usec"] - cg1["throttled_usec"]) / 1000,
                         nr_periods=cg2["nr_periods"] - cg1["nr_periods"],
                         nr_throttled=cg2["nr_throttled"] - cg1["nr_throttled"],
                         memory_current_mb=cg2["memory_current"] / 2**20)

if a.refit and a.model in ("prophet", "prophet_lstm"):
    from pbscale.forecasters import ProphetLSTM, ProphetOnly
    tr = data.load_trace(a.trace)
    t_end = pd.Timestamp(data.SCENARIOS[a.scenario].start)
    win = tr[(tr.ds < t_end) & (tr.ds >= t_end - pd.Timedelta(days=7))]
    w, c = time.perf_counter(), time.process_time()
    pm = ProphetOnly(a.trace).fit(win)
    res["refit"] = dict(window_rows=len(win), prophet_wall_s=time.perf_counter() - w,
                        prophet_cpu_s=time.process_time() - c)
    if a.model == "prophet_lstm":
        hy = ProphetLSTM(a.trace, epochs=1)
        w, c = time.perf_counter(), time.process_time()
        hy.m = pm.m
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.layers import LSTM, Dense, Input
        from tensorflow.keras.models import Sequential
        r = win.y.to_numpy(float) - pm._yhat(win.ds)
        s = (r - r.min()) / (r.max() - r.min())
        net = Sequential([Input((1, 1)), LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
        net.compile(optimizer="adam", loss="mean_squared_error")
        net.fit(s[:-1].reshape(-1, 1, 1), s[1:], epochs=1, batch_size=16, verbose=0)
        res["refit"].update(lstm_1epoch_wall_s=time.perf_counter() - w, lstm_1epoch_cpu_s=time.process_time() - c)
res["peak_rss_mb"] = res["decisions"].get("peak_rss_mb")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
d = res["decisions"]
print(f"{a.model} fast={a.fast} cpus={res['cpu_limit_cores']}: cold={res['cold_start']['load_wall_ms']:.0f}ms "
      f"p50={np.median(d['wall_ms']):.1f}ms p95={np.percentile(d['wall_ms'], 95):.1f}ms "
      f"cpu={np.median(d['cpu_ms']):.1f}ms rss={res['cold_start']['rss_mb']:.0f}MB "
      f"refit={res.get('refit')}", flush=True)
