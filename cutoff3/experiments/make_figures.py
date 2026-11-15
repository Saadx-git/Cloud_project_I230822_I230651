"""Regenerate every C3 figure and LaTeX table from results/*.json.  Usage: python experiments/make_figures.py"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, FIG, TAB = f"{ROOT}/results", f"{ROOT}/figures", f"{ROOT}/report/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# fixed colour per entity (validated reference categorical palette, light mode)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"prophet_lstm": "#2a78d6", "hpa": "#eb6834", "prophet": "#1baf7a", "ar10": "#6250d6",
       "persistence": "#eda100", "moving_avg5": "#e87ba4", "seasonal_naive_1d": "#008300",
       "prophet_lstm_LEAKY": "#e34948", "oracle": "#52514e"}
LABEL = {"persistence": "Persistence", "moving_avg5": "Moving avg (5)", "seasonal_naive_1d": "Seasonal naive (1 d)",
         "ar10": "AR(10)", "prophet": "Prophet", "prophet_lstm": "Prophet-LSTM (causal)",
         "prophet_lstm_LEAKY": "Prophet-LSTM (leaky, as released)", "hpa": "HPA (CPU, reactive)", "oracle": "Oracle"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png")
    plt.close(fig)
    print("figure", name)


def num(v):
    return f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"


def tex_table(path, header, rows, colspec):
    with open(path, "w") as f:
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("table", os.path.basename(path))


# ---------------------------------------------------------------- E1 accuracy
def fig_accuracy():
    order = ["persistence", "moving_avg5", "seasonal_naive_1d", "ar10", "prophet", "prophet_lstm", "prophet_lstm_LEAKY"]
    acc = {t: json.load(open(f"{R}/e1_accuracy_{t}.json")) for t in ("nasa", "fifa") if os.path.exists(f"{R}/e1_accuracy_{t}.json")}
    if not acc:
        return
    fig, axes = plt.subplots(1, len(acc), figsize=(3.4 * len(acc), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (t, a) in zip(axes, acc.items()):
        base = a["persistence"]["RMSE"]
        vals = [a[m]["RMSE"] / base for m in order]
        ypos = np.arange(len(order))[::-1]
        bars = ax.barh(ypos, vals, height=0.62, color=[COL[m] for m in order], edgecolor="white", linewidth=1)
        bars[-1].set_hatch("////")
        ax.axvline(1, color=MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        for yv, v in zip(ypos, vals):
            ax.text(v * 1.08, yv, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
        ax.set_title(f"{t.upper()} test split ({a['_diag']['test_rows']:,} min)", fontsize=9, color=INK)
        ax.set_xlabel("RMSE relative to persistence (lower is better)")
        ax.set_yticks(ypos, [LABEL[m] for m in order])
        ax.set_xlim(0.3, max(vals) * 2.2)
        ticks = [t for t in (0.5, 1, 2, 5, 10, 20) if t < max(vals) * 2.2]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(axis="y", visible=False)
    save(fig, "fig_e1_causal_vs_leaky")
    rows = []
    for m in order:
        r = [LABEL[m]]
        for t in acc:
            x = acc[t][m]
            r += [f"{x['RMSE']:.2f}" if t == "nasa" else f"{x['RMSE']:.0f}", f"{x['MAE']:.2f}" if t == "nasa" else f"{x['MAE']:.0f}", f"{x['R2']:.3f}"]
        rows.append(r)
    hdr = "Forecaster & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{t.upper()}}}" for t in acc) + " \\\\\n & " + \
          " & ".join("RMSE & MAE & $R^2$" for _ in acc)
    tex_table(f"{TAB}/e1_accuracy.tex", hdr, rows, "l" + "rrr" * len(acc))


# ---------------------------------------------------------------- E2 overhead
def fig_overhead():
    p = f"{R}/e2_overhead.json"
    if not os.path.exists(p):
        return
    df = pd.DataFrame(json.load(open(p))["rows"])
    df["cfg"] = df.model + np.where(df.fast, "+fast", "")
    df["b"] = df.budget_cores.replace(0, np.nan)
    host = df[df.budget_cores == 0].set_index("cfg")
    df = df[df.budget_cores > 0].sort_values("b")
    cfgs = [("prophet_lstm", "Prophet-LSTM (as released)", "-"), ("prophet_lstm+fast", "Prophet-LSTM (optimised)", "--"),
            ("prophet", "Prophet (as released)", "-"), ("prophet+fast", "Prophet (optimised)", "--"),
            ("ar10", "AR(10)", "-"), ("persistence", "Persistence", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    for cfg, lab, ls in cfgs:
        d = df[df.cfg == cfg]
        if d.empty:
            continue
        c = COL[cfg.replace("+fast", "")]
        axes[0].plot(d.b * 1000, d.lat_p95_ms / 1000, ls, color=c, lw=2, marker="o", ms=4, label=lab)
        axes[1].plot(d.b * 1000, d.cold_total_wall_s, ls, color=c, lw=2, marker="o", ms=4, label=lab)
        axes[2].plot(d.b * 1000, d.cpu_mean_ms, ls, color=c, lw=2, marker="o", ms=4, label=lab)
    for ax in axes:
        ax.set_xlabel("controller CPU limit (millicores)")
        ax.set_xscale("log")
        ax.set_xticks([25, 50, 100, 250, 500, 1000], ["25m", "50m", "100m", "250m", "500m", "1000m"])
        ax.minorticks_off()
    for P in (15, 30, 60):
        for ax in axes[:2]:
            ax.axhline(P, color=MUTED, lw=0.8, ls=":")
            ax.text(df.b.min() * 950, P * 1.05, f"P = {P} s", fontsize=7, color=MUTED, va="bottom", ha="left")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("per-decision latency p95 (s)")
    axes[0].set_title("Inference latency vs. control period", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("cold start: import+load+first call (s)")
    axes[1].set_title("Cold start (restart / kill-on-miss)", fontsize=9)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("CPU time per decision (ms)")
    axes[2].set_title("CPU cost per decision", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig_e2_overhead")
    rows = []
    for cfg, lab, _ in cfgs:
        for b in (0.025, 0.05, 0.1, 0.25, 0.5, 1.0):
            d = df[(df.cfg == cfg) & (np.isclose(df.b, b))]
            if d.empty:
                continue
            x = d.iloc[0]
            rows.append([lab, f"{int(b * 1000)}m", num(x.cpu_mean_ms), num(x.lat_p50_ms), num(x.lat_p95_ms),
                         f"{x.cold_total_wall_s:.1f}", f"{x.rss_mb:.0f}", f"{x.throttled_pct:.0f}",
                         f"{x.budget_share_P15_pct:.2f}"])
        if cfg in host.index:
            x = host.loc[cfg]
            rows.append([lab, "none", num(x.cpu_mean_ms), num(x.lat_p50_ms), num(x.lat_p95_ms),
                         f"{x.cold_total_wall_s:.1f}", f"{x.rss_mb:.0f}", "--", "--"])
    tex_table(f"{TAB}/e2_overhead.tex",
              "Forecaster & Limit & CPU/dec. (ms) & p50 (ms) & p95 (ms) & Cold start (s) & RSS (MB) & Thr. (\\%) & Budget@15s (\\%)",
              rows, "llrrrrrrr")
    ref = df[df.refit.notna()]
    if len(ref):
        rr = []
        for _, x in ref.iterrows():
            f = x.refit
            rr.append([f"{int(x.b * 1000)}m", f"{f['window_rows']:,}", f"{f['prophet_wall_s']:.0f}", f"{f['prophet_cpu_s']:.1f}",
                       f"{f.get('lstm_1epoch_wall_s', float('nan')):.0f}", f"{f.get('lstm_1epoch_cpu_s', float('nan')):.1f}"])
        tex_table(f"{TAB}/e2_refit.tex", "Limit & Rows (7 d) & Prophet fit wall (s) & CPU (s) & LSTM 1 epoch wall (s) & CPU (s)",
                  rr, "lrrrrr")


# ---------------------------------------------------------------- E3 simulation
def fig_sim():
    p = f"{R}/e3_scaling.json"
    if not os.path.exists(p):
        return
    e3 = json.load(open(p))["results"]
    sigs = ["hpa", "persistence", "ar10", "prophet", "prophet_lstm", "prophet_lstm_LEAKY", "oracle"]
    rows = []
    for sc, res in e3.items():
        for s in sigs:
            m = res["runs"][f"{s}|startup=0"]
            rows.append([sc if s == "hpa" else "", LABEL[s], f"{m['Theta_U']:.2f}", f"{m['Theta_O']:.1f}", f"{m['T_U']:.1f}",
                         f"{m['T_O']:.1f}", f"{m['mean_pods']:.2f}", f"{m['saturated_pct']:.1f}"])
    sr = []
    for sc, res in e3.items():
        st, meta = res["stats"], res["scenario"]
        sr.append([sc, meta["trace"].upper(), f"{meta['start']} -- {meta['end'][11:]}", meta["kind"], str(st["minutes"]),
                   f"{st['mean']:,.0f}", f"{st['max']:,.0f}", f"{st['cv']:.2f}", f"{st['peak_to_mean']:.2f}",
                   f"{st['p99_abs_step_pct']:.0f}\\%", f"{st['lag1_autocorr']:.3f}"])
    tex_table(f"{TAB}/scenarios.tex", "Scenario & Trace & Window & Type & Length (min) & Mean & Max & CV & Peak/mean & p99 step & ACF(1)",
              sr, "llllrrrrrrr")
    tex_table(f"{TAB}/e3_scaling.tex",
              "Scenario & Signal & $\\Theta_U$ (\\%) & $\\Theta_O$ (\\%) & $T_U$ (\\%) & $T_O$ (\\%) & mean pods & sat. (\\%)",
              rows, "llrrrrrr")
    # time series for the bursty and ramp scenarios
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 2.9))
    for ax, sc in zip(axes, ["bursty", "ramp"]):
        ts = pd.read_csv(f"{R}/e3_timeseries_{sc}.csv.gz", parse_dates=["ds"])
        C = e3[sc]["capacity_per_pod"] * 0.7
        ax.fill_between(ts.ds, ts.y / C, step="mid", color=GRID, label="demand (pods needed)")
        for s in ["hpa", "prophet_lstm", "prophet_lstm_LEAKY"]:
            ax.step(ts.ds, ts[s], where="mid", color=COL[s], lw=1.6 if s != "prophet_lstm_LEAKY" else 1.0,
                    ls="-" if s != "prophet_lstm_LEAKY" else "--", label=LABEL[s])
        ax.set_title({"bursty": "Bursty: WorldCup98 semi-final, 7 Jul 1998 18:00-23:00",
                      "ramp": "Gradual ramp: WorldCup98, 7 Jul 1998 09:51-16:31"}[sc], fontsize=8.5)
        ax.set_ylabel("ready pods")
        ax.tick_params(axis="x", labelrotation=0)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_e3_sim_timeseries")
    # bar chart: under-provisioning vs pods across scenarios
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.1))
    scs = list(e3)
    w = 0.13
    for j, s in enumerate(["hpa", "persistence", "ar10", "prophet", "prophet_lstm", "prophet_lstm_LEAKY"]):
        x = np.arange(len(scs)) + (j - 2.5) * w
        tu = [e3[sc]["runs"][f"{s}|startup=0"]["Theta_U"] for sc in scs]
        po = [e3[sc]["runs"][f"{s}|startup=0"]["mean_pods"] / e3[sc]["runs"]["oracle|startup=0"]["mean_pods"] for sc in scs]
        kw = dict(width=w * 0.9, color=COL[s], label=LABEL[s], edgecolor="white", linewidth=0.5,
                  hatch="////" if "LEAKY" in s else None)
        axes[0].bar(x, tu, **kw)
        axes[1].bar(x, po, **kw)
    for ax in axes:
        ax.set_xticks(np.arange(len(scs)), scs)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("$\\Theta_U$: mean under-provisioning (%)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("pod-minutes relative to oracle")
    axes[1].axhline(1, color=MUTED, lw=0.8, ls="--")
    axes[1].set_ylim(0, 1.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig_e3_sim_bars")


# ---------------------------------------------------------------- E4 live
def fig_live():
    p = f"{R}/e4_live.json"
    if not os.path.exists(p):
        return
    runs = [r for r in json.load(open(p)) if not re.search(r"-s\d+$", r["run"])]  # seed-1 runs; repetitions: fig_e5
    if not runs:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.9))
    d0 = np.array(runs[0]["series"]["demand_rps"])
    tgt = json.load(open(f"{R}/e4/{runs[0]['run']}/config.json"))["target_per_pod"]
    tmin = np.arange(len(d0))
    ax.fill_between(tmin, d0 / tgt, step="mid", color=GRID, label="demand (pods needed at 70 % target)")
    styles = {"hpa": ("hpa", "-"), "prophet-500m": ("prophet", "-"), "prophet_lstm-500m": ("prophet_lstm", "-"),
              "prophet_lstm-100m": ("prophet_lstm", "--"), "persistence-100m": ("persistence", ":")}
    rows = []
    for r in runs:
        name = r["policy"]["name"]
        key, ls = styles.get(name, ("ar10", "-"))
        ax.step(tmin, r["series"]["ready"], where="mid", color=COL[key], ls=ls, lw=1.6, label=f"{name} (ready)")
        m, s = r["metrics"], r.get("scaler", {})
        rows.append([name.replace("_", "\\_"), f"{m['Theta_U']:.2f}", f"{m['Theta_O']:.1f}", f"{m['T_U']:.1f}", f"{m['T_O']:.1f}",
                     f"{m['pod_minutes']:.0f}", f"{m['slo_violation_pct']:.2f}",
                     "--" if not s else f"{s['accuracy_vs_offered']['RMSE']:.2f}",
                     "--" if not s or s.get('forecast_cpu_ms_mean') is None else f"{s['forecast_cpu_ms_mean']:.0f}",
                     "--" if not s else f"{s['forecast_wall_ms_p95']:.0f}",
                     "--" if not s else f"{s['late_pct']:.1f}"])
    ax.set_xlabel("trace minute (replayed at 15 s per minute)")
    ax.set_ylabel("ready pods")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_e4_live")
    tex_table(f"{TAB}/e4_live.tex",
              "Autoscaler & $\\Theta_U$ & $\\Theta_O$ & $T_U$ & $T_O$ & pod-min & SLO viol. (\\%) & RMSE (rps) & CPU/dec. (ms) & p95 lat. (ms) & late (\\%)",
              rows, "lrrrrrrrrrr")


# ---------------------------------------------------------------- E5 statistics (final paper)
def fig_stats():
    p = f"{R}/e5_statistics.json"
    if not os.path.exists(p):
        return
    st = json.load(open(p))
    # (1) deadline sensitivity of the as-released and optimised hybrid
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 2.8))
    Ds = ["0.5", "1", "2", "3", "5", "15"]
    for ax, fast, title in [(axes[0], 0, "Prophet-LSTM as released"), (axes[1], 1, "Prophet-LSTM optimised")]:
        for b, ls, c in [(0.025, "-", "#0b0b0b"), (0.05, "--", "#52514e"), (0.1, "-", COL["prophet_lstm"]), (0.25, "--", COL["prophet_lstm"]), (0.5, ":", COL["prophet_lstm"]), (1.0, "-.", COL["prophet_lstm"])]:
            k = next((k for k in st["deadline"] if k.startswith(f"prophet_lstm|fast={bool(fast)}|budget=")
                      and float(k.split("=")[-1]) == b), None)
            if k is None:
                continue
            ax.plot([float(d) for d in Ds], [100 * st["deadline"][k]["late"][d] for d in Ds], ls=ls, marker="o", ms=3,
                    color=c, label=f"CPU limit {int(b * 1000)}m")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 3, 5, 15], ["0.5", "1", "2", "3", "5", "15"])
        ax.set_xlabel("decision deadline D (s)")
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("late forecasts (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_e5_deadline")
    # (2) E4 repetitions: per-seed dots
    rows = st.get("e4", {}).get("runs", [])
    if rows:
        df = pd.DataFrame(rows)
        order = [("bursty-hpa", "HPA", "hpa"), ("bursty-persistence-100m", "Persist.\n@100m", "persistence"),
                 ("bursty-prophet_lstm-100m", "P-LSTM\n@100m", "prophet_lstm"),
                 ("bursty-prophet_lstm-500m", "P-LSTM\n@500m", "prophet_lstm"),
                 ("bursty-prophet-500m", "Prophet\n@500m", "prophet")]
        order = [o for o in order if o[0] in set(df.config)]
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8))
        for ax, (col, lab, logy) in zip(axes, [("Theta_U", "$\\Theta_U$ (%)", True), ("slo", "SLO violations (%)", True),
                                               ("pod_minutes", "pod-minutes", False)]):
            for i, (c, name, key) in enumerate(order):
                v = df[df.config == c][col].to_numpy(float)
                ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, color=COL[key], s=18, zorder=3)
                ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, lw=1.2)
            ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=7)
            ax.set_ylabel(lab)
            ax.grid(axis="x", visible=False)
            if logy:
                ax.set_yscale("log")
        fig.tight_layout()
        save(fig, "fig_e5_live_reps")


if __name__ == "__main__":
    fig_accuracy()
    fig_overhead()
    fig_sim()
    fig_live()
    fig_stats()
