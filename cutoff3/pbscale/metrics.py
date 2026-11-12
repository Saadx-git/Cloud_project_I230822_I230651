"""Forecast-accuracy and provisioning metrics."""
from __future__ import annotations

import numpy as np


def accuracy(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    e = y - p
    mse = float(np.mean(e ** 2))
    return dict(n=int(ok.sum()), MSE=mse, RMSE=float(np.sqrt(mse)), MAE=float(np.mean(np.abs(e))),
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
                sMAPE=float(np.mean(2 * np.abs(e) / np.maximum(np.abs(y) + np.abs(p), 1e-9)) * 100))


def provisioning(r, p) -> dict:
    """Under/over-provisioning metrics of the base paper's autoscaler notebook (SPEC RG style).

    r: demanded load per step; p: provisioned capacity per step (same units).
      Theta_U = 100/T * sum(max(r-p,0)/r)    average relative under-provisioning [%]
      Theta_O = 100/T * sum(max(p-r,0)/r)    average relative over-provisioning  [%]
      T_U     = 100/T * #(r>p)              share of time under-provisioned     [%]
      T_O     = 100/T * #(p>r)              share of time over-provisioned      [%]
    Steps with r == 0 use r = 1 in the denominator.
    """
    r, p = np.asarray(r, float), np.asarray(p, float)
    T = len(r)
    d = np.maximum(r, 1.0)
    return dict(Theta_U=float(100 / T * np.sum(np.maximum(r - p, 0) / d)),
                Theta_O=float(100 / T * np.sum(np.maximum(p - r, 0) / d)),
                T_U=float(100 / T * np.sum(r > p)),
                T_O=float(100 / T * np.sum(p > r)))
