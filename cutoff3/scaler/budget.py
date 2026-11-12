"""Budget-enforced forecasting: the model runs in a separate worker process.

The parent sends one request per control period and waits at most `deadline_s`.
  * on time  -> the forecast is used
  * too late -> the persistence fallback y(t) is used and, with on_miss="kill",
                the worker is SIGKILLed and re-spawned (it may be stuck or
                starved by CFS throttling); with on_miss="abandon" the late
                answer is simply discarded when it eventually arrives.
The worker reports the CPU time it consumed (time.process_time, all threads)
for every forecast, plus its cold-start (import + model load) cost.

The CPU budget itself (e.g. 100m-500m) is enforced by the container runtime:
the scaler Pod runs with resources.limits.cpu = budget (CFS quota), exactly
as any Kubernetes controller would.  `cgroup_cpu()` reads the container's
cpu.stat so that throttling can be reported.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time


def cgroup_cpu() -> dict:
    """cgroup v2 CPU and memory counters of this container (empty if unavailable)."""
    out = {}
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            k, v = line.split()
            if k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec"):
                out[k] = int(v)
        out["memory_current"] = int(open("/sys/fs/cgroup/memory.current").read())
        q = open("/sys/fs/cgroup/cpu.max").read().split()
        out["cpu_limit_cores"] = None if q[0] == "max" else int(q[0]) / int(q[1])
    except (OSError, ValueError):
        pass
    return out


def _worker(conn, name, trace, fast):
    t_wall, t_cpu = time.perf_counter(), time.process_time()
    import numpy as np  # noqa: F401
    import pandas as pd
    from pbscale.forecasters import build
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None
    model = build(name, trace, fast)
    # one warm-up call so that lazy graph construction is part of the cold start
    ds0 = pd.Series(pd.to_datetime(["1998-07-01 00:00"] if trace == "fifa" else ["1995-08-15 00:00"]))
    try:
        model.predict_next(ds0.repeat(1500).reset_index(drop=True), np.ones(1500))
    except Exception:
        pass
    conn.send(("ready", dict(load_wall_ms=1000 * (time.perf_counter() - t_wall),
                             load_cpu_ms=1000 * (time.process_time() - t_cpu),
                             rss_mb=proc.memory_info().rss / 2**20 if proc else None)))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        rid, ds_hist, y_hist = msg
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            f = model.predict_next(pd.Series(pd.to_datetime(ds_hist)), np.asarray(y_hist, float))
            err = None
        except Exception as e:  # report, never crash the controller
            f, err = float("nan"), repr(e)
        conn.send((rid, dict(forecast=f, wall_ms=1000 * (time.perf_counter() - w0),
                             cpu_ms=1000 * (time.process_time() - c0), error=err,
                             rss_mb=proc.memory_info().rss / 2**20 if proc else None)))


class BudgetedForecaster:
    def __init__(self, name, trace, fast=False, on_miss="kill"):
        self.name, self.trace, self.fast, self.on_miss = name, trace, fast, on_miss
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.conn = None
        self.ready = False
        self.cold_starts = []
        self.rid = 0
        self._spawn()

    def _spawn(self):
        parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_worker, args=(child, self.name, self.trace, self.fast), daemon=True)
        self.t_spawn = time.perf_counter()
        self.proc.start()
        self.conn, self.ready = parent, False

    def _poll_ready(self, timeout=0.0):
        """Consume the worker's 'ready' message if it has arrived."""
        if not self.ready and self.conn.poll(timeout):
            try:
                tag, info = self.conn.recv()
            except EOFError:  # worker died during start-up (e.g. OOM-killed): start a new one
                self.kill()
                self._spawn()
                return False
            info["spawn_to_ready_ms"] = 1000 * (time.perf_counter() - self.t_spawn)
            self.cold_starts.append(info)
            self.ready = True
        return self.ready

    def kill(self):
        if self.proc is not None and self.proc.is_alive():
            self.proc.kill()
            self.proc.join(5)

    def forecast(self, ds_hist, y_hist, deadline_s: float) -> dict:
        """Return dict(forecast, late, fallback, wall_ms, cpu_ms, ...).  Never blocks > deadline_s."""
        t0 = time.perf_counter()
        fallback = float(y_hist[-1])
        if not self._poll_ready(0.0):
            # cold start still in progress: we may wait inside the deadline
            if not self._poll_ready(deadline_s):
                return dict(forecast=fallback, late=True, fallback=True, reason="cold_start",
                            wall_ms=1000 * (time.perf_counter() - t0), cpu_ms=None)
        # drain answers abandoned in earlier periods
        while self.conn.poll(0):
            self.conn.recv()
        self.rid += 1
        rid = self.rid
        self.conn.send((rid, [str(d) for d in ds_hist], list(map(float, y_hist))))
        remaining = deadline_s - (time.perf_counter() - t0)
        while remaining > 0 and self.conn.poll(remaining):
            try:
                got, info = self.conn.recv()
            except EOFError:
                break  # worker died: treated as a miss below
            if got == rid:
                ok = info["error"] is None and info["forecast"] == info["forecast"]
                info.update(late=False, fallback=not ok, reason=None if ok else "error",
                            forecast=info["forecast"] if ok else fallback,
                            total_ms=1000 * (time.perf_counter() - t0))
                return info
            remaining = deadline_s - (time.perf_counter() - t0)
        # deadline missed
        if self.on_miss == "kill":
            self.kill()
            self._spawn()
        return dict(forecast=fallback, late=True, fallback=True, reason="deadline",
                    wall_ms=1000 * (time.perf_counter() - t0), cpu_ms=None)
