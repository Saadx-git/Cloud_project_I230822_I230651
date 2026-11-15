"""E4 - Live Kubernetes (kind) comparison: HPA vs KEDA predictive scalers on a replayed trace.

Time compression: one trace minute is replayed in STEP=15 wall seconds, and the
control period of every autoscaler is 15 s, i.e. one decision per trace minute.
HPA scale-down stabilisation (75 s) and the Prometheus rate window are compressed
by the same factor (k8s/kind-config.yaml, k8s/hpa.yaml).

Per run:
  1. reset: remove autoscalers, set the Deployment to the replicas needed for minute 0, wait ready
  2. install the autoscaler (HPA or scaler Deployment + ScaledObject with CPU budget)
  3. start the replay clock of the scaler and the open-loop load generator
  4. sample spec/ready replicas every 2 s
  5. collect: load CSV, replica samples, scaler decision log, HPA status
Outputs: results/e4/<run>/...; summary in results/e4_live.json (see e4_analyze.py)

Usage: python experiments/e4_live.py --runs hpa,prophet@500m,prophet_lstm@500m,prophet_lstm@100m
       repetitions: add --seed 2 --tag=-s2 (load-generator seed; seed 1 runs carry no tag)
"""
import argparse
import csv
import json
import math
import os
import string
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbscale import data  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = ["kubectl"]


def sh(args, check=True, capture=True, inp=None):
    r = subprocess.run(args, check=check, capture_output=capture, text=True, input=inp)
    return r.stdout


def reset(n0):
    sh(K + ["delete", "hpa", "work-hpa", "--ignore-not-found"])
    sh(K + ["delete", "scaledobject", "work-predictive", "--ignore-not-found"])
    sh(K + ["delete", "deploy", "pbscale-scaler", "--ignore-not-found", "--wait=true"])
    time.sleep(3)
    sh(K + ["delete", "hpa", "keda-hpa-work-predictive", "--ignore-not-found"])
    sh(K + ["scale", "deploy/work", f"--replicas={n0}"])
    sh(K + ["rollout", "status", "deploy/work", "--timeout=120s"])
    time.sleep(20)  # let metrics-server / Prometheus forget the previous run


def scaler_exec(code):
    return sh(K + ["exec", "deploy/pbscale-scaler", "--", "python", "-c", code])


def install(policy, cfg):
    if policy["kind"] == "hpa":
        sh(K + ["apply", "-f", f"{ROOT}/k8s/hpa.yaml"])
        return
    tmpl = string.Template(open(f"{ROOT}/k8s/scaler.yaml.tmpl").read())
    y = tmpl.substitute(MODEL=policy["model"], TRACE=cfg["trace"], FAST=str(policy.get("fast", 0)),
                        PERIOD=str(int(cfg["step"])), DEADLINE=str(cfg["step"]), BUDGET=policy["budget"],
                        LOAD_SCALE=f"{cfg['scale']:.10f}", TARGET_PER_POD=f"{cfg['target_per_pod']:.4f}")
    open(f"{cfg['outdir']}/scaler.yaml", "w").write(y)
    docs = y.split("\n---\n")
    scaled_object = docs.pop()  # last document: the ScaledObject
    # the ScaledObject is applied only once the gRPC scaler is up; otherwise KEDA's GetMetricSpec call fails and
    # it writes an HPA without the external metric (observed in a first attempt, see README troubleshooting)
    sh(K + ["apply", "-f", "-"], inp="\n---\n".join(docs))
    sh(K + ["rollout", "status", "deploy/pbscale-scaler", "--timeout=300s"])
    t0 = time.time()
    while True:  # wait for the forecaster's cold start (this is itself a measured quantity)
        try:
            out = scaler_exec("import urllib.request,json;print(json.load(urllib.request.urlopen("
                              "'http://127.0.0.1:8081/healthz'))['ready'])").strip()
            if out == "True":
                break
        except subprocess.CalledProcessError:
            pass
        if time.time() - t0 > 900:
            raise RuntimeError("scaler never became ready")
        time.sleep(3)
    cfg["scaler_ready_after_s"] = time.time() - t0
    sh(K + ["apply", "-f", "-"], inp=scaled_object)
    for _ in range(40):  # verify KEDA's HPA really scales on the external forecast metric
        time.sleep(3)
        m = sh(K + ["get", "hpa", "keda-hpa-work-predictive", "-o", "jsonpath={.spec.metrics[*].type}"], check=False)
        if "External" in m:
            break
    else:
        raise RuntimeError(f"KEDA HPA has no External metric: {m!r}")
    cfg["hpa_metrics"] = m
    time.sleep(10)


def sampler(stop, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall", "spec", "ready"])
        while not stop.is_set():
            try:
                out = sh(K + ["get", "deploy", "work", "-o",
                              "jsonpath={.spec.replicas} {.status.readyReplicas}"], check=False).split()
                w.writerow([time.time(), out[0] if out else "", out[1] if len(out) > 1 else 0])
                f.flush()
            except Exception:
                pass
            stop.wait(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="bursty")
    ap.add_argument("--start", default="1998-07-07 18:50")
    ap.add_argument("--minutes", type=int, default=100)
    ap.add_argument("--step", type=float, default=15.0)
    ap.add_argument("--capacity", type=float, required=True, help="pod capacity C (req/s at 100%% of request)")
    ap.add_argument("--peak-pods", type=int, default=10)
    ap.add_argument("--runs", default="hpa,prophet@500m,prophet_lstm@500m,prophet_lstm@100m")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1, help="load-generator seed (repetitions)")
    a = ap.parse_args()

    sc = data.SCENARIOS[a.scenario]
    fr = data.scenario_frame(a.scenario)
    fr = fr[fr.ds >= a.start].head(a.minutes).reset_index(drop=True)
    target = 0.7 * a.capacity
    scale = a.peak_pods * target / fr.y.max()  # live req/s per trace req/min
    base = f"{ROOT}/results/e4"
    os.makedirs(base, exist_ok=True)
    slice_csv = f"{base}/slice_{a.scenario}.csv"
    fr.to_csv(slice_csv, index=False)
    n0 = max(1, math.ceil(fr.y.iloc[0] * scale / target))
    for spec in a.runs.split(","):
        if spec == "hpa":
            pol = dict(kind="hpa", name="hpa")
        else:
            m, b = spec.split("@")
            fast = 1 if m.endswith("+fast") else 0
            m = m.replace("+fast", "")
            pol = dict(kind="keda", model=m, budget=b, fast=fast, name=f"{m}{'-fast' if fast else ''}-{b}")
        run = f"{a.scenario}-{pol['name']}{a.tag}"
        outdir = f"{base}/{run}"
        os.makedirs(outdir, exist_ok=True)
        cfg = dict(trace=sc.trace, step=a.step, scale=scale, target_per_pod=target, capacity=a.capacity,
                   outdir=outdir, policy=pol, scenario=a.scenario, start=a.start, minutes=a.minutes, n0=n0,
                   seed=a.seed)
        print(f"=== {run}: scale={scale:.6f} target/pod={target:.2f} rps n0={n0}", flush=True)
        reset(n0)
        install(pol, cfg)
        stop = threading.Event()
        th = threading.Thread(target=sampler, args=(stop, f"{outdir}/replicas.csv"), daemon=True)
        th.start()
        t0 = time.time() + 3
        cfg["t0"] = t0
        if pol["kind"] == "keda":
            scaler_exec("import urllib.request;urllib.request.urlopen(urllib.request.Request("
                        f"'http://127.0.0.1:8081/start?t0={t0}&trace_start={fr.ds.iloc[0]:%Y-%m-%dT%H:%M:%S}"
                        f"&step={a.step}',method='POST'))")
        time.sleep(max(0, t0 - time.time()))
        sh([sys.executable, f"{ROOT}/loadgen/replay.py", "--trace-csv", slice_csv, "--scale", str(scale),
            "--step", str(a.step), "--seed", str(a.seed), "--out", f"{outdir}/load.csv"], capture=False)
        time.sleep(5)
        stop.set()
        th.join()
        if pol["kind"] == "keda":
            open(f"{outdir}/scaler.log", "w").write(sh(K + ["logs", "deploy/pbscale-scaler"]))
            open(f"{outdir}/hpa.txt", "w").write(sh(K + ["describe", "hpa", "keda-hpa-work-predictive"], check=False))
        else:
            open(f"{outdir}/hpa.txt", "w").write(sh(K + ["describe", "hpa", "work-hpa"], check=False))
        json.dump(cfg, open(f"{outdir}/config.json", "w"), indent=1)
        print(f"=== {run} done", flush=True)
    reset(n0)


if __name__ == "__main__":
    main()
