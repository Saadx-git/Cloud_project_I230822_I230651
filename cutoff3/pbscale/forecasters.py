"""One-step-ahead request-rate forecasters with a single causal interface.

Every forecaster exposes

    predict_next(ds_hist, y_hist) -> float

which returns the forecast for minute t+1 using ONLY observations up to and
including minute t (the last element of y_hist).  This is the call that the
online scaler makes once per control period, and it is what the overhead
benchmark times.

`offline_series(df)` returns the same one-step forecasts for every row of a
DataFrame in one vectorised pass (used only for accuracy tables); row i of the
output is the forecast *for* row i, built from rows < i.  For the hybrid it can
additionally return the base paper's mis-aligned ("leaky") series.

Heavy libraries (prophet, tensorflow) are imported lazily so that the
lightweight forecasters can run in a small container.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

# Prophet configuration of the released notebooks (base paper, Table 1 of our C2 report)
PROPHET_KW = dict(changepoint_prior_scale=5.1, yearly_seasonality=False, weekly_seasonality=20,
                  daily_seasonality=50, seasonality_prior_scale=30)
PROPHET_GROWTH = {"nasa": "linear", "fifa": "flat"}


class Forecaster:
    name = "base"
    min_history = 1

    def fit(self, train: pd.DataFrame) -> "Forecaster":
        return self

    def predict_next(self, ds_hist, y_hist) -> float:
        raise NotImplementedError

    def offline_series(self, df: pd.DataFrame) -> np.ndarray:
        """Causal one-step forecasts for every row of df (NaN where history is too short)."""
        y = df.y.to_numpy(float)
        out = np.full(len(y), np.nan)
        for i in range(self.min_history, len(y)):
            out[i] = self.predict_next(df.ds.iloc[:i], y[:i])
        return out


# --------------------------------------------------------------------------- lightweight
class Persistence(Forecaster):
    name = "persistence"

    def predict_next(self, ds_hist, y_hist):
        return float(y_hist[-1])

    def offline_series(self, df):
        return df.y.shift(1).to_numpy(float)


class MovingAverage(Forecaster):
    name = "moving_avg5"

    def __init__(self, k=5):
        self.k = k
        self.min_history = k

    def predict_next(self, ds_hist, y_hist):
        return float(np.mean(y_hist[-self.k:]))

    def offline_series(self, df):
        return df.y.rolling(self.k).mean().shift(1).to_numpy(float)


class SeasonalNaive(Forecaster):
    """y(t+1) = y(t+1-period); period=1440 min (one day)."""
    name = "seasonal_naive_1d"

    def __init__(self, period=1440):
        self.period = period
        self.min_history = period

    def predict_next(self, ds_hist, y_hist):
        return float(y_hist[-self.period])

    def offline_series(self, df):
        return df.y.shift(self.period).to_numpy(float)


class AutoRegressive(Forecaster):
    """Ordinary-least-squares AR(p) on the last p minutes (with intercept)."""
    name = "ar10"

    def __init__(self, p=10):
        self.p = p
        self.min_history = p
        self.coef = None

    def _lags(self, y):
        return np.column_stack([y[self.p - k - 1: len(y) - k - 1] for k in range(self.p)])

    def fit(self, train):
        y = train.y.to_numpy(float)
        X = np.column_stack([np.ones(len(y) - self.p), self._lags(y)])
        self.coef, *_ = np.linalg.lstsq(X, y[self.p:], rcond=None)
        return self

    def predict_next(self, ds_hist, y_hist):
        x = np.asarray(y_hist[-1: -self.p - 1: -1], float)  # most recent first
        return float(self.coef[0] + x @ self.coef[1:])

    def offline_series(self, df):
        y = df.y.to_numpy(float)
        out = np.full(len(y), np.nan)
        X = np.column_stack([np.ones(len(y) - self.p), self._lags(y)])
        out[self.p:] = X @ self.coef
        return out


# --------------------------------------------------------------------------- Prophet family
def _prophet_quiet():
    import logging
    for n in ("prophet", "cmdstanpy"):
        logging.getLogger(n).setLevel(logging.ERROR)


class ProphetOnly(Forecaster):
    """Prophet's yhat at t+1 (Baseline 2).

    fast=False reproduces the released scaler call (Prophet.predict on a
    DataFrame, default 1000 uncertainty samples); fast=True disables
    uncertainty sampling, which the point forecast does not need.
    """
    name = "prophet"

    def __init__(self, trace="nasa", fast=False):
        self.trace, self.fast, self.m = trace, fast, None

    def fit(self, train):
        from prophet import Prophet
        _prophet_quiet()
        self.m = Prophet(growth=PROPHET_GROWTH[self.trace], **PROPHET_KW)
        self.m.fit(train[["ds", "y"]])
        return self

    def _yhat(self, ds_list):
        m = self.m
        if self.fast:
            m.uncertainty_samples = 0
        return m.predict(pd.DataFrame({"ds": pd.to_datetime(ds_list)}))["yhat"].to_numpy(float)

    def predict_next(self, ds_hist, y_hist):
        t = pd.Timestamp(ds_hist.iloc[-1] if hasattr(ds_hist, "iloc") else ds_hist[-1])
        return float(self._yhat([t + pd.Timedelta(minutes=1)])[0])

    def offline_series(self, df):
        # Prophet's forecast for a row's own timestamp depends on no observation
        # after training, so this is causal for every test row.
        return self._yhat(df.ds)

    # persistence -------------------------------------------------------------
    def save(self, path):
        from prophet.serialize import model_to_json
        with open(path, "w") as f:
            f.write(model_to_json(self.m))

    def load(self, path):
        from prophet.serialize import model_from_json
        _prophet_quiet()
        with open(path) as f:
            self.m = model_from_json(f.read())
        return self


class ProphetLSTM(ProphetOnly):
    """Prophet-LSTM residual hybrid (proposed forecaster; base-paper architecture).

    y_hat(t+1) = prophet(t+1) + LSTM( r(t) ),  r(t) = y(t) - prophet(t)

    Deviations from the released code, both to make the evaluation causal:
      * the residual MinMax scaler is fitted on training residuals only (fix F2);
      * offline forecasts are written at the row they forecast (fix F1).
    """
    name = "prophet_lstm"

    def __init__(self, trace="nasa", fast=False, epochs=20, seed=42):
        super().__init__(trace, fast)
        self.epochs, self.seed = epochs, seed
        self.lstm, self.lo, self.hi = None, None, None
        self.history = None

    # residual scaling (MinMax to [0,1], parameters from training residuals)
    def _s(self, r):
        return (np.asarray(r, float) - self.lo) / (self.hi - self.lo)

    def _inv(self, s):
        return np.asarray(s, float) * (self.hi - self.lo) + self.lo

    def fit(self, train):
        super().fit(train)
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Input
        from tensorflow.keras.models import Sequential
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        r = train.y.to_numpy(float) - self._yhat(train.ds)
        self.lo, self.hi = float(r.min()), float(r.max())
        s = self._s(r)
        X, Y = s[:-1].reshape(-1, 1, 1), s[1:]
        self.lstm = Sequential([Input((1, 1)), LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
        self.lstm.compile(optimizer="adam", loss="mean_squared_error")
        h = self.lstm.fit(X, Y, epochs=self.epochs, batch_size=16, validation_split=0.2, verbose=2,
                          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
        self.history = {k: [float(v) for v in vs] for k, vs in h.history.items()}
        return self

    def _lstm_one(self, s_t: float) -> float:
        x = np.array([[[s_t]]], dtype=np.float32)
        if self.fast:
            return float(self.lstm(x, training=False).numpy()[0, 0])
        return float(self.lstm.predict(x, verbose=0)[0, 0])  # as in the released scaler

    def predict_next(self, ds_hist, y_hist):
        t = pd.Timestamp(ds_hist.iloc[-1] if hasattr(ds_hist, "iloc") else ds_hist[-1])
        yh_t, yh_t1 = self._yhat([t, t + pd.Timedelta(minutes=1)])
        r_t = float(y_hist[-1]) - yh_t
        return float(yh_t1 + self._inv(self._lstm_one(float(self._s(r_t))))[()])

    def offline_series(self, df, leaky=False):
        yh = self._yhat(df.ds)
        r = df.y.to_numpy(float) - yh
        p = self._inv(self.lstm.predict(self._s(r).reshape(-1, 1, 1), verbose=0, batch_size=4096)[:, 0])
        # p[i] = forecast of r[i+1] made from r[i]
        out = np.full(len(r), np.nan)
        if leaky:  # base-paper evaluation: forecast for i+1 written into row i
            out[:] = yh + p
        else:      # causal: forecast for i+1 written into row i+1
            out[1:] = yh[1:] + p[:-1]
        return out

    def save(self, prefix):
        super().save(prefix + "-prophet.json")
        self.lstm.save(prefix + "-lstm.keras")
        with open(prefix + "-meta.json", "w") as f:
            json.dump(dict(lo=self.lo, hi=self.hi, trace=self.trace, history=self.history), f)

    def load(self, prefix):
        super().load(prefix + "-prophet.json")
        import keras
        self.lstm = keras.models.load_model(prefix + "-lstm.keras")
        meta = json.load(open(prefix + "-meta.json"))
        self.lo, self.hi = meta["lo"], meta["hi"]
        return self


# --------------------------------------------------------------------------- factory
MODEL_DIR = os.environ.get("PBSCALE_MODELS",
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))


def build(name: str, trace: str, fast: bool = False) -> Forecaster:
    """Load a trained forecaster by name ('persistence', 'ar10', 'prophet', 'prophet_lstm', ...)."""
    if name == "persistence":
        return Persistence()
    if name == "moving_avg5":
        return MovingAverage(5)
    if name == "seasonal_naive_1d":
        return SeasonalNaive()
    if name == "ar10":
        with open(os.path.join(MODEL_DIR, f"{trace}-ar10.pkl"), "rb") as f:
            return pickle.load(f)
    if name == "prophet":
        return ProphetOnly(trace, fast).load(os.path.join(MODEL_DIR, f"{trace}-prophet_lstm-prophet.json"))
    if name == "prophet_lstm":
        return ProphetLSTM(trace, fast).load(os.path.join(MODEL_DIR, f"{trace}-prophet_lstm"))
    raise ValueError(name)
