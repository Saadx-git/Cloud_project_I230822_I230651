"""Budget-aware predictive KEDA external scaler (gRPC).

Control loop (period P = CONTROL_PERIOD_S):
  every P seconds:
    1. observe:   y_t = sum(rate(app_requests_total[P])) from Prometheus  (live req/s)
                  converted to trace units  y_t / LOAD_SCALE  (requests per trace-minute)
    2. forecast:  BudgetedForecaster with deadline DEADLINE_S (default = P);
                  late/failed -> persistence fallback
    3. publish:   desired load for the next period, in live req/s
  KEDA GetMetrics returns the last published value instantly; the HPA that KEDA
  creates computes replicas = ceil(value / TARGET_PER_POD) with the usual
  tolerance and stabilisation, i.e. the same policy as the CPU-based HPA.

Replay clock: POST /start?trace_start=<ISO>&step=<s> (called by the experiment
orchestrator when the load generator starts) maps wall time to trace minutes,
so Prophet is queried at the trace timestamp it was trained for (fixes C2 F4,
the 30-year extrapolation of the released scaler).

Instrumentation: one JSON line per decision on stdout (and /decisions).

Environment
  MODEL=prophet|prophet_lstm|ar10|persistence   TRACE=nasa|fifa   FAST=0|1
  CONTROL_PERIOD_S=15  DEADLINE_S=<P>  ON_MISS=kill|abandon
  PROMETHEUS=http://prometheus.monitoring:9090  QUERY=sum(rate(app_requests_total[15s]))
  LOAD_SCALE=<live req/s per trace req/min>     TARGET_PER_POD=<live req/s per pod at target util>
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import externalscaler_pb2 as pb  # noqa: E402
import externalscaler_pb2_grpc as pbg  # noqa: E402
from budget import BudgetedForecaster, cgroup_cpu  # noqa: E402

E = os.environ.get
MODEL, TRACE, FAST = E("MODEL", "prophet_lstm"), E("TRACE", "fifa"), E("FAST", "0") == "1"
P = float(E("CONTROL_PERIOD_S", "15"))
DEADLINE = float(E("DEADLINE_S", str(P)))
ON_MISS = E("ON_MISS", "kill")
PROM = E("PROMETHEUS", "http://prometheus.monitoring.svc.cluster.local:9090")
QUERY = E("QUERY", "sum(rate(app_requests_total[15s]))")
LOAD_SCALE = float(E("LOAD_SCALE", "1"))
TARGET_PER_POD = float(E("TARGET_PER_POD", "7"))
HISTORY = int(E("HISTORY", "1500"))

state = dict(published=None, decisions=[], clock=None)
lock = threading.Lock()


def log(rec):
    print(json.dumps(rec), flush=True)


def prom_query(q):
    url = f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": q})
    with urllib.request.urlopen(url, timeout=5) as r:
        res = json.load(r)["data"]["result"]
    return float(res[0]["value"][1]) if res else 0.0


def control_loop(fc: BudgetedForecaster):
    import pandas as pd
    ds_hist, y_hist = [], []
    cg_prev = cgroup_cpu()
    k_prev = None
    while True:
        clock = state["clock"]
        if clock is None:  # waiting for /start
            fc._poll_ready(1.0)
            continue
        # tick at the end of each trace minute
        k = int((time.time() - clock["t0"]) // clock["step"])
        if k == k_prev:
            time.sleep(0.2)
            continue
        k_prev = k
        if k < 1:
            continue
        t_obs = pd.Timestamp(clock["trace_start"]) + pd.Timedelta(minutes=k - 1)  # minute just completed
        try:
            rps = prom_query(QUERY)
        except Exception as e:  # observation failure: keep last decision
            log(dict(event="observe_error", k=k, error=repr(e)))
            continue
        y = rps / LOAD_SCALE
        ds_hist.append(t_obs)
        y_hist.append(y)
        ds_hist, y_hist = ds_hist[-HISTORY:], y_hist[-HISTORY:]
        t_dec = time.time()
        r = fc.forecast(ds_hist, y_hist, DEADLINE)
        desired_rps = max(r["forecast"], 0.0) * LOAD_SCALE
        cg = cgroup_cpu()
        rec = dict(event="decision", k=k, wall=t_dec, trace_ts=str(t_obs), observed_rps=rps, observed_trace=y,
                   forecast_trace=r["forecast"], desired_rps=desired_rps,
                   desired_pods=desired_rps / TARGET_PER_POD, late=r["late"], fallback=r["fallback"],
                   reason=r.get("reason"), wall_ms=r.get("wall_ms"), cpu_ms=r.get("cpu_ms"),
                   worker_rss_mb=r.get("rss_mb"), model=MODEL, fast=FAST, period_s=P, deadline_s=DEADLINE)
        if cg and cg_prev:
            rec.update(cg_usage_ms=(cg["usage_usec"] - cg_prev["usage_usec"]) / 1000,
                       cg_throttled_ms=(cg["throttled_usec"] - cg_prev["throttled_usec"]) / 1000,
                       cg_nr_throttled=cg["nr_throttled"] - cg_prev["nr_throttled"],
                       cg_mem_mb=cg["memory_current"] / 2**20, cpu_limit_cores=cg.get("cpu_limit_cores"))
        cg_prev = cg
        with lock:
            state["published"] = desired_rps
            state["decisions"].append(rec)
        log(rec)


class Scaler(pbg.ExternalScalerServicer):
    def IsActive(self, request, context):
        return pb.IsActiveResponse(result=True)

    def StreamIsActive(self, request, context):
        while context.is_active():
            yield pb.IsActiveResponse(result=True)
            time.sleep(P)

    def GetMetricSpec(self, request, context):
        # metric unit = milli-requests/s: KEDA 2.16 uses the integer targetSize, so a fractional
        # per-pod target (e.g. 2.8 req/s) would otherwise be truncated (observed: 2.8 -> 2)
        tgt = 1000 * float(request.scalerMetadata.get("targetPerPod", TARGET_PER_POD))
        return pb.GetMetricSpecResponse(metricSpecs=[pb.MetricSpec(
            metricName="forecast_mrps", targetSize=int(round(tgt)), targetSizeFloat=tgt)])

    def GetMetrics(self, request, context):
        with lock:
            v = state["published"]
        if v is None:  # no decision yet: report the currently observed load (reactive)
            try:
                v = prom_query(QUERY)
            except Exception:
                v = 0.0
        return pb.GetMetricsResponse(metricValues=[pb.MetricValue(
            metricName="forecast_mrps", metricValue=int(round(1000 * v)), metricValueFloat=1000 * float(v))])


class Ctl(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/start":
            with lock:
                state["clock"] = dict(t0=float(q.get("t0", time.time())), trace_start=q["trace_start"],
                                      step=float(q.get("step", P)))
                state["decisions"].clear()
                state["published"] = None
            log(dict(event="start", **state["clock"]))
            self._json(state["clock"])
        else:
            self._json({"error": "unknown"}, 404)

    def do_GET(self):
        if self.path.startswith("/decisions"):
            with lock:
                self._json(dict(decisions=state["decisions"], cold_starts=FC.cold_starts))
        elif self.path.startswith("/healthz"):
            self._json(dict(ok=True, ready=FC.ready))
        else:
            self._json({"error": "unknown"}, 404)


if __name__ == "__main__":
    FC = BudgetedForecaster(MODEL, TRACE, FAST, ON_MISS)
    log(dict(event="boot", model=MODEL, trace=TRACE, fast=FAST, period_s=P, deadline_s=DEADLINE,
             on_miss=ON_MISS, cgroup=cgroup_cpu()))
    threading.Thread(target=control_loop, args=(FC,), daemon=True).start()
    http = ThreadingHTTPServer(("0.0.0.0", 8081), Ctl)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_ExternalScalerServicer_to_server(Scaler(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
