"""E3 - Trace-driven scaling comparison (HPA vs predictive) on the workload scenarios.

For every scenario the forecasters are run causally over the minute grid (with
1500 minutes of preceding trace as warm-up history), and the resulting
forecasts drive the shared replica policy of pbscale.simulate.  The leaky
(base-paper-aligned) hybrid is included to show how the evaluation defect
propagates into apparent *scaling* quality.

Budget enforcement: if results/e2_overhead.json exists, each predictive signal
is also simulated with the late-forecast probability measured in E2 for a given
CPU budget and control period (late -> persistence fallback), seed-averaged.

Outputs: results/e3_scaling.json, results/e3_timeseries_<scenario>.csv.gz
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pbscale import data, forecasters as F  # noqa: E402
from pbscale.metrics import accuracy  # noqa: E402
from pbscale.simulate import Policy, simulate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEAK_PODS = int(os.environ.get("PEAK_PODS", "10"))  # capacity normalisation: scenario peak needs this many pods
SEEDS = 20

models = {}


def get(trace, name):
    if (trace, name) not in models:
        models[(trace, name)] = F.build(name, trace)
    return models[(trace, name)]


def forecasts_for(key):
    sc = data.SCENARIOS[key]
    hist = data.history_before(key, 1500)
    frame = data.scenario_frame(key)
    full = pd.concat([hist, frame], ignore_index=True)
    k0 = len(hist)
    out = {"oracle": frame.y.to_numpy(float)}
    for name in ["persistence", "moving_avg5", "seasonal_naive_1d", "ar10", "prophet", "prophet_lstm"]:
        out[name] = get(sc.trace, name).offline_series(full)[k0:]
    out["prophet_lstm_LEAKY"] = get(sc.trace, "prophet_lstm").offline_series(full, leaky=True)[k0:]
    return frame, out


def late_table():
    p = f"{ROOT}/results/e2_overhead.json"
    if not os.path.exists(p):
        return {}
    e2 = json.load(open(p))
    return e2.get("late_fraction", {})  # {model|fast|budget|period: fraction}


def main():
    late = late_table()
    results, rows = {}, []
    for key, sc in data.SCENARIOS.items():
        frame, fc = forecasts_for(key)
        y = frame.y.to_numpy(float)
        C = y.max() / (PEAK_PODS * 0.7)
        res = dict(scenario=sc.__dict__, stats=data.describe(y), capacity_per_pod=C, runs={}, accuracy={})
        ts = pd.DataFrame({"ds": frame.ds, "y": y})
        for startup in (0, 1):
            pol = Policy(capacity=C, startup_steps=startup)
            r = simulate(y, pol, "hpa")
            res["runs"][f"hpa|startup={startup}"] = r["metrics"]
            if startup == 0:
                ts["hpa"] = r["ready"]
            for name, f in fc.items():
                r = simulate(y, pol, "predictive", forecast=f)
                res["runs"][f"{name}|startup={startup}"] = r["metrics"]
                if startup == 0:
                    ts[name] = r["ready"]
                    ts[f"fc_{name}"] = f
        for name, f in fc.items():
            if name != "oracle":
                res["accuracy"][name] = accuracy(y[1:], f[1:])
        # budget-enforced variants (late forecasts -> persistence fallback)
        pol = Policy(capacity=C)
        for k, q in late.items():
            model, fast, budget, period = k.split("|")
            base = model if model in fc else None
            if base is None or q is None:
                continue
            ms = []
            for s in range(SEEDS):
                rng = np.random.default_rng(s)
                ms.append(simulate(y, pol, "predictive", forecast=fc[base], late=rng.random(len(y)) < q)["metrics"])
            res["runs"][f"{model}|fast={fast}|budget={budget}|P={period}"] = {
                m: float(np.mean([x[m] for x in ms])) for m in ms[0]} | {"late_fraction": q}
        results[key] = res
        ts.to_csv(f"{ROOT}/results/e3_timeseries_{key}.csv.gz", index=False)
        print(f"== {key}: C={C:.1f}/pod  {res['stats']}")
        for k, m in res["runs"].items():
            if "budget" in k or k.endswith("startup=0"):
                print(f"  {k:45s} ThU={m['Theta_U']:7.2f} ThO={m['Theta_O']:7.2f} TU={m['T_U']:6.2f} "
                      f"TO={m['T_O']:6.2f} pods={m['mean_pods']:6.2f} sat={m['saturated_pct']:5.2f}")
    json.dump(dict(peak_pods=PEAK_PODS, results=results), open(f"{ROOT}/results/e3_scaling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
