"""E1 - Train forecasters on the 70 % training split and compare causal vs. leaky accuracy.

Usage: python experiments/e1_train_and_accuracy.py nasa|fifa

Outputs
  models/<trace>-prophet_lstm-{prophet.json,lstm.keras,meta.json}, models/<trace>-ar10.pkl
  results/e1_accuracy_<trace>.json   accuracy (t+1) of every forecaster on the full test split,
                                     plus the hybrid evaluated with the base paper's mis-alignment
  results/e1_forecasts_<trace>.csv.gz  per-row test forecasts (input to E3)

Accuracy is computed in raw units (requests/minute) for both traces; the base
paper reported FIFA on a Min-Max-scaled series, so R^2 is the comparable number.
"""
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbscale import data, forecasters as F  # noqa: E402
from pbscale.metrics import accuracy  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

trace = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/models", exist_ok=True)
os.makedirs(f"{ROOT}/results", exist_ok=True)

df = data.load_trace(trace)
train, test = data.split(df)
print(f"{trace}: {len(df)} rows, train {len(train)} ({train.ds.min()} .. {train.ds.max()}), "
      f"test {len(test)} ({test.ds.min()} .. {test.ds.max()})", flush=True)

fit_s = {}
t0 = time.time()
hyb = F.ProphetLSTM(trace).fit(train)
fit_s["prophet_lstm_total"] = time.time() - t0
hyb.save(f"{ROOT}/models/{trace}-prophet_lstm")
ar = F.AutoRegressive(10).fit(train)
pickle.dump(ar, open(f"{ROOT}/models/{trace}-ar10.pkl", "wb"))
prophet = F.ProphetOnly(trace)
prophet.m = hyb.m

# evaluate on the whole series, then keep test rows (lags may reach into train: that is causal)
n_tr = len(train)
series = {
    "persistence": F.Persistence().offline_series(df),
    "moving_avg5": F.MovingAverage(5).offline_series(df),
    "seasonal_naive_1d": F.SeasonalNaive().offline_series(df),
    "ar10": ar.offline_series(df),
    "prophet": prophet.offline_series(df),
    "prophet_lstm": hyb.offline_series(df, leaky=False),
    "prophet_lstm_LEAKY": hyb.offline_series(df, leaky=True),
}
y = df.y.to_numpy(float)
res = {k: accuracy(y[n_tr:], v[n_tr:]) for k, v in series.items()}
# correlation diagnostic of the leak (C2 finding F1)
r = y - prophet.offline_series(df)
leaky_resid = series["prophet_lstm_LEAKY"] - prophet.offline_series(df)
res["_diag"] = dict(corr_leaky_residual_vs_same_row_residual=float(np.corrcoef(r[n_tr:], leaky_resid[n_tr:])[0, 1]),
                    lstm_epochs=len(hyb.history["loss"]), fit_seconds=fit_s,
                    residual_scaler=dict(lo=hyb.lo, hi=hyb.hi, fitted_on="train only"),
                    train_rows=n_tr, test_rows=len(test))
json.dump(res, open(f"{ROOT}/results/e1_accuracy_{trace}.json", "w"), indent=2)
out = pd.DataFrame({"ds": df.ds, "y": y, **series}).iloc[n_tr:]
out.to_csv(f"{ROOT}/results/e1_forecasts_{trace}.csv.gz", index=False)
for k, v in res.items():
    if not k.startswith("_"):
        print(f"{k:22s} RMSE={v['RMSE']:12.3f} MAE={v['MAE']:12.3f} R2={v['R2']:.4f}")
print(json.dumps(res["_diag"], indent=1))
