"""E5 - Statistical analysis for the final paper (no new measurements; reads results/ only).

(a) E3 paired moving-block bootstrap.  For every scenario and every causal scaler, the per-minute
    under-provisioning u_t, over-provisioning o_t and ready pods n_t are compared with the HPA and
    with the persistence scaler on the *same* minutes.  Minutes are resampled in blocks of B=30
    (moving-block bootstrap, Kuensch 1989) to respect autocorrelation; 2000 resamples; 95 % percentile CI.
(b) Decision equivalence.  Share of minutes in which a scaler requests exactly the same number of ready
    pods as the persistence scaler (E3), and the correlation of the forecasts with y_{t-1}.
(c) Deadline sensitivity (E2 raw latencies).  Late-forecast fraction Pr[latency > D] for deadlines
    D in {0.5, 1, 2, 3, 5, 15} s and every CPU budget, i.e. the deadline at which a budgeted forecaster
    would start falling back to persistence.
(d) E4 repetitions.  Mean, standard deviation and min/max over load-generator seeds per autoscaler,
    plus a paired comparison (same seed) of every autoscaler against the HPA.

Usage: python experiments/e5_statistics.py   -> results/e5_statistics.json, report/tables/e5_*.tex
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"
TAB = f"{ROOT}/report/tables"
B, NBOOT, PEAK = 30, 2000, 10
SCALERS = ["persistence", "moving_avg5", "ar10", "prophet", "prophet_lstm", "prophet_lstm_LEAKY"]
LABEL = {"hpa": "HPA (B1)", "persistence": "Persistence", "moving_avg5": "Moving avg.\\ (5)", "ar10": "AR(10)",
         "prophet": "Prophet (B2)", "prophet_lstm": "Prophet--LSTM", "prophet_lstm_LEAKY": "Prophet--LSTM (leaky)",
         "seasonal_naive_1d": "Seasonal naive"}


def block_idx(n, rng):
    starts = rng.integers(0, n - B + 1, size=int(np.ceil(n / B)))
    return np.concatenate([np.arange(s, s + B) for s in starts])[:n]


def per_minute(y, ready):
    cap = y.max() / (PEAK * 0.7) * 0.7
    p = ready * cap
    d = np.maximum(y, 1.0)
    return 100 * np.maximum(y - p, 0) / d, 100 * np.maximum(p - y, 0) / d, 100.0 * (y > p), ready.astype(float)


def boot_diff(a, b, rng_seed=0):
    """a, b: per-minute series (paired). Returns mean(a)-mean(b) and its 95 % block-bootstrap CI."""
    rng = np.random.default_rng(rng_seed)
    n = len(a)
    d = a - b
    stats = np.array([d[block_idx(n, rng)].mean() for _ in range(NBOOT)])
    return dict(diff=float(d.mean()), lo=float(np.percentile(stats, 2.5)), hi=float(np.percentile(stats, 97.5)),
                p_le0=float(np.mean(stats <= 0)), p_ge0=float(np.mean(stats >= 0)))


def e3_bootstrap():
    out = {}
    for f in sorted(glob.glob(f"{RES}/e3_timeseries_*.csv.gz")):
        sc = re.search(r"e3_timeseries_(\w+)\.csv", f).group(1)
        ts = pd.read_csv(f)
        y = ts.y.to_numpy(float)
        series = {k: per_minute(y, ts[k].to_numpy(float)) for k in ["hpa"] + SCALERS}
        res = {}
        for k in SCALERS:
            res[k] = {}
            for ref in ("hpa", "persistence"):
                if k == ref:
                    continue
                res[k][f"vs_{ref}"] = {
                    m: boot_diff(series[k][i], series[ref][i]) for i, m in enumerate(["Theta_U", "Theta_O", "T_U", "pods"])}
            same = np.mean(ts[k].to_numpy() == ts["persistence"].to_numpy())
            fc = ts[f"fc_{k}"].to_numpy(float)
            res[k]["same_pods_as_persistence_pct"] = float(100 * same)
            res[k]["corr_forecast_with_last_value"] = float(np.corrcoef(fc[1:], y[:-1])[0, 1])
        out[sc] = res
    return out


def deadline_sensitivity():
    out = {}
    for p in sorted(glob.glob(f"{RES}/e2/raw/*.json")):
        r = json.load(open(p))
        w = np.array(r["decisions"]["wall_ms"]) / 1000
        b = float(re.search(r"-c([0-9.]+)-", os.path.basename(p)).group(1))  # requested --cpus (cgroup value is rounded)
        key = f"{r['model']}|fast={r['fast']}|budget={b if b else 0}"
        out[key] = dict(n=len(w), p50=float(np.median(w)), p95=float(np.percentile(w, 95)), max=float(w.max()),
                        late={str(D): float(np.mean(w > D)) for D in (0.5, 1, 2, 3, 5, 15)})
    return out


def e4_repetitions():
    p = f"{RES}/e4_live.json"
    if not os.path.exists(p):
        return {}
    runs = json.load(open(p))
    rows = []
    for r in runs:
        m = re.match(r"(.+?)(?:-s(\d+))?$", r["run"])
        name, seed = m.group(1), int(m.group(2) or 1)
        s = r.get("scaler", {})
        mm = r["metrics"]
        rows.append(dict(config=name, seed=seed, Theta_U=mm["Theta_U"], Theta_O=mm["Theta_O"], T_U=mm["T_U"],
                         T_O=mm["T_O"], pod_minutes=mm["pod_minutes"], slo=mm["slo_violation_pct"],
                         errors=mm["errors_pct"], late=s.get("late_pct"), cpu=s.get("forecast_cpu_ms_mean"),
                         p95=s.get("forecast_wall_ms_p95"), ready_s=r.get("scaler_ready_after_s"),
                         mem=s.get("container_mem_mb_max"),
                         rmse=(s.get("accuracy_vs_observed") or {}).get("RMSE"),
                         rmse_pers=(s.get("persistence_vs_observed") or {}).get("RMSE")))
    df = pd.DataFrame(rows)
    agg = {}
    for c, g in df.groupby("config"):
        agg[c] = dict(n=int(len(g)), seeds=sorted(int(x) for x in g.seed))
        for k in ["Theta_U", "Theta_O", "T_U", "T_O", "pod_minutes", "slo", "errors", "late", "cpu", "p95", "ready_s",
                  "mem", "rmse", "rmse_pers"]:
            v = pd.to_numeric(g[k], errors="coerce").dropna()
            if len(v):
                agg[c][k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                 min=float(v.min()), max=float(v.max()))
    # paired (same seed) differences against the HPA
    paired = {}
    h = df[df.config == "bursty-hpa"].set_index("seed")
    for c, g in df.groupby("config"):
        if c == "bursty-hpa":
            continue
        g = g.set_index("seed")
        common = sorted(set(g.index) & set(h.index))
        if not common:
            continue
        paired[c] = {k: [float(g.loc[s, k] - h.loc[s, k]) for s in common] for k in ["Theta_U", "slo", "pod_minutes"]}
        paired[c]["seeds"] = common
    return dict(runs=rows, aggregate=agg, paired_vs_hpa=paired)


def fmt_ci(d, nd=2):
    return f"{d['diff']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def write_tables(st):
    os.makedirs(TAB, exist_ok=True)
    # (a) bootstrap vs HPA and vs persistence for the causal scalers
    lines = [r"\begin{tabular}{@{}llrrr@{}}\toprule",
             r"Scenario & Scaler & $\Delta\Theta_U$ vs.\ HPA (pp) & $\Delta\Theta_U$ vs.\ persistence (pp) & same pods as pers.\ (\%) \\ \midrule"]
    for sc in ["steady", "diurnal", "ramp", "bursty"]:
        if sc not in st["e3_bootstrap"]:
            continue
        first = True
        for k in ["ar10", "prophet", "prophet_lstm"]:
            r = st["e3_bootstrap"][sc][k]
            lines.append(f"{sc if first else ''} & {LABEL[k]} & {fmt_ci(r['vs_hpa']['Theta_U'])} & "
                         f"{fmt_ci(r['vs_persistence']['Theta_U'])} & {r['same_pods_as_persistence_pct']:.1f} \\\\")
            first = False
        r = st["e3_bootstrap"][sc]["persistence"]
        lines.append(f" & {LABEL['persistence']} & {fmt_ci(r['vs_hpa']['Theta_U'])} & --- & 100.0 \\\\")
        if sc != "bursty":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular}")
    open(f"{TAB}/e5_bootstrap.tex", "w").write("\n".join(lines) + "\n")
    # (c) deadline sensitivity: hybrid as released
    ds = st["deadline"]
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}\toprule",
             r"Forecaster & CPU limit & p95 lat.\ (s) & $D{=}0.5$\,s & $D{=}1$\,s & $D{=}2$\,s & $D{=}3$\,s & $D{=}5$\,s \\ \midrule"]
    for model, fast, lab in [("prophet_lstm", 0, "Prophet--LSTM (released)"), ("prophet_lstm", 1, "Prophet--LSTM (optimised)"),
                             ("prophet", 0, "Prophet (released)"), ("ar10", 0, "AR(10)")]:
        for b in ["0.025", "0.05", "0.1", "0.25", "0.5", "1.0", "0"]:
            key = next((k for k in ds if k.startswith(f"{model}|fast={bool(fast)}|budget=") and
                        float(k.split('=')[-1]) == float(b)), None)
            if key is None:
                continue
            d = ds[key]
            bl = "none" if float(b) == 0 else f"{int(float(b) * 1000)}m"
            lines.append(f"{lab} & {bl} & {d['p95']:.2f} & " +
                         " & ".join(f"{100 * d['late'][D]:.0f}" for D in ("0.5", "1", "2", "3", "5")) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule\end{tabular}"
    open(f"{TAB}/e5_deadline.tex", "w").write("\n".join(lines) + "\n")
    # (d) E4 repetitions
    agg = st["e4"].get("aggregate", {})
    order = [("bursty-hpa", "B1: HPA (CPU 70\\,\\%)"), ("bursty-prophet-500m", "B2: Prophet @500m"),
             ("bursty-persistence-100m", "Persistence @100m"),
             ("bursty-prophet_lstm-100m", "Prophet--LSTM @100m"), ("bursty-prophet_lstm-500m", "Prophet--LSTM @500m")]

    def ms(c, k, nd=2):
        if k not in agg.get(c, {}):
            return "---"
        v = agg[c][k]
        return f"{v['mean']:.{nd}f}" + (f" $\\pm$ {v['sd']:.{nd}f}" if agg[c]["n"] > 1 else "")
    lines = [r"\begin{tabular}{@{}lcllllll@{}}\toprule",
             r"Autoscaler & $n$ & $\Theta_U$ (\%) & $T_U$ (\%) & Pod-min & SLO viol.\ (\%) & Late (\%) & Ready after (s) \\ \midrule"]
    for c, lab in order:
        if c not in agg:
            continue
        lines.append(f"{lab} & {agg[c]['n']} & {ms(c, 'Theta_U')} & {ms(c, 'T_U', 1)} & {ms(c, 'pod_minutes', 0)} & "
                     f"{ms(c, 'slo')} & {ms(c, 'late', 1)} & {ms(c, 'ready_s', 0)} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(f"{TAB}/e5_e4_reps.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    st = dict(e3_bootstrap=e3_bootstrap(), deadline=deadline_sensitivity(), e4=e4_repetitions(),
              params=dict(block=B, n_boot=NBOOT))
    json.dump(st, open(f"{RES}/e5_statistics.json", "w"), indent=1)
    write_tables(st)
    for sc, r in st["e3_bootstrap"].items():
        for k in ["persistence", "ar10", "prophet", "prophet_lstm"]:
            x = r[k]
            print(f"{sc:8s} {k:13s} dThU vs HPA {fmt_ci(x['vs_hpa']['Theta_U'])}  "
                  + (f"vs pers {fmt_ci(x['vs_persistence']['Theta_U'])}  " if 'vs_persistence' in x else "")
                  + f"same={x['same_pods_as_persistence_pct']:.1f}%  corr(fc,y_t-1)={x['corr_forecast_with_last_value']:.4f}")
    for k, d in st["deadline"].items():
        print(f"{k:35s} p95={d['p95']:.2f}s late@1s={d['late']['1']:.2f} late@2s={d['late']['2']:.2f} late@3s={d['late']['3']:.2f}")
    for c, a in st["e4"].get("aggregate", {}).items():
        print(c, a["n"], {k: round(v["mean"], 2) for k, v in a.items() if isinstance(v, dict)})
