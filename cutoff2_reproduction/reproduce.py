"""Re-runs the authors' Prophet-LSTM notebook logic (NASA / FIFA) and adds two audit checks.

Usage: python reproduce.py nasa|fifa <repo_dir> <out_dir>
Faithful part = cell-by-cell copy of 'Hybrid Model - <DS> - Prophet LSTM.ipynb' (plots removed).
Audit part    = (a) same predictions shifted to the timestamp they actually forecast, (b) persistence baseline.
"""
import json, sys, time, platform
import numpy as np, pandas as pd, tensorflow as tf
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

ds_name, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
fifa = ds_name == "fifa"
np.random.seed(42); tf.random.set_seed(42)

def metrics(y, p, t_ms=None):
    m = mean_squared_error(y, p)
    return dict(MSE=m, RMSE=float(np.sqrt(m)), MAE=mean_absolute_error(y, p), R2=r2_score(y, p), TPT_ms=t_ms)

# --- data (notebook cells) ---
df = pd.read_csv(f"{repo}/datasets/{ds_name}.csv")
df.columns = ["ds", "y_us" if fifa else "y"]
df["ds"] = pd.to_datetime(df["ds"])
if fifa:
    scaler_y = MinMaxScaler(feature_range=(0, 1))  # as released: fit on train+test
    df["y"] = scaler_y.fit_transform(df[["y_us"]])
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=False)
train_df = train_df.copy()
train_df["y"] = train_df["y"].ffill()
train_df = train_df.drop_duplicates(subset="ds")

# --- Prophet (notebook hyperparameters, not the paper's Table 2) ---
m1 = Prophet(growth="flat" if fifa else "linear", changepoint_prior_scale=5.1, yearly_seasonality=False,
             weekly_seasonality=20, daily_seasonality=50, seasonality_prior_scale=30)
t0 = time.time(); m1.fit(train_df[["ds", "y"]]); prophet_fit_s = time.time() - t0
fc_train = m1.predict(train_df[["ds"]].copy())
t0 = time.time(); fc_test = m1.predict(test_df[["ds"]].copy()); t_prophet = (time.time() - t0) * 1000
pa = test_df.merge(fc_test[["ds", "yhat"]], on="ds", how="left")
prophet_only = metrics(pa["y"], pa["yhat"], t_prophet)

# --- residuals + LSTM ---
fc = pd.concat([fc_train, fc_test])
res = df.merge(fc[["ds", "yhat"]], on="ds", how="left")
res["residual"] = res["y"] - res["yhat"]
if fifa:
    arr = np.array(res[["residual"]])
else:
    scaler = MinMaxScaler(feature_range=(0, 1))  # as released: fit on train+test residuals
    arr = scaler.fit_transform(res[["residual"]])
look_back = 1
X = np.array([arr[i:i + look_back, 0] for i in range(len(arr) - look_back)])
y = np.array([arr[i + look_back, 0] for i in range(len(arr) - look_back)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
Xtr, Xte = Xtr.reshape(-1, look_back, 1), Xte.reshape(-1, look_back, 1)

m2 = Sequential([Input((look_back, 1)), LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
m2.compile(optimizer="adam", loss="mean_squared_error")
t0 = time.time()
hist = m2.fit(Xtr, ytr, epochs=20, batch_size=16, validation_split=0.2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)], verbose=2)
lstm_fit_s = time.time() - t0
p_tr = m2.predict(Xtr, verbose=0)
t0 = time.time(); p_te = m2.predict(Xte, verbose=0); t_lstm = (time.time() - t0) * 1000
if not fifa:
    p_tr, p_te = scaler.inverse_transform(p_tr), scaler.inverse_transform(p_te)
p_all = np.concatenate((p_tr, p_te))[:, 0]

# --- combine exactly as released: prediction for t+1 is written into row t ---
final = res.copy()
rp = np.full(len(final), np.nan); rp[:len(p_all)] = p_all
final["residual_predict"] = pd.Series(rp, index=final.index).fillna(final["residual"])
final["y_predicted"] = final["yhat"] + final["residual_predict"]
ev = final.iloc[train_df.shape[0]:]
as_released = metrics(ev["y"], ev["y_predicted"], t_prophet + t_lstm)

# --- audit (a): same predictions placed at the timestamp they forecast (row t+1) ---
rp2 = np.full(len(final), np.nan); rp2[look_back:look_back + len(p_all)] = p_all
final["y_pred_aligned"] = final["yhat"] + pd.Series(rp2, index=final.index).fillna(final["residual"])
ev = final.iloc[train_df.shape[0]:]
aligned = metrics(ev["y"], ev["y_pred_aligned"], t_prophet + t_lstm)

# --- audit (b): persistence y_hat(t) = y(t-1) on the same test rows ---
persist = metrics(ev["y"].iloc[1:], final["y"].shift(1).loc[ev.index].iloc[1:], 0.0)

# --- audit (c): does the as-released residual track the *same-row* residual? ---
corr_same = float(np.corrcoef(ev["residual"], ev["residual_predict"])[0, 1])

result = dict(dataset=ds_name, rows=len(df), train_rows=len(train_df), test_rows=len(df) - len(train_df),
              prophet_only=prophet_only, hybrid_as_released=as_released, hybrid_aligned=aligned,
              persistence=persist, corr_residual_vs_released_prediction_same_row=corr_same,
              lstm_epochs_run=len(hist.history["loss"]), prophet_fit_s=prophet_fit_s, lstm_fit_s=lstm_fit_s,
              env=dict(python=platform.python_version(), tf=tf.__version__, numpy=np.__version__,
                       pandas=pd.__version__, prophet=__import__("prophet").__version__,
                       sklearn=__import__("sklearn").__version__, cpu=platform.processor() or platform.machine()))
assert len(p_all) == len(df) - look_back  # sanity: one residual prediction per row except the first
json.dump(result, open(f"{out}/{ds_name}_results.json", "w"), indent=2, default=float)
print(json.dumps(result, indent=2, default=float))
