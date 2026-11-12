"""Trace loading, chronological split and workload scenarios.

All scenarios are cut from the *test* part (last 30 %) of each trace, so no
model has seen them during training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.environ.get("PBSCALE_DATA", os.path.join(ROOT, "upstream", "repo", "datasets"))
TRAIN_FRACTION = 0.7  # same chronological 70/30 split as the base paper


def load_trace(name: str) -> pd.DataFrame:
    """Return a DataFrame with columns ds (datetime) and y (requests/minute), raw units."""
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    return df.drop_duplicates("ds").reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # identical to sklearn train_test_split(test_size=0.3, shuffle=False) used by the paper
    n = len(df) - int(np.ceil(len(df) * (1 - TRAIN_FRACTION)))
    return df.iloc[:n].copy(), df.iloc[n:].copy()


@dataclass(frozen=True)
class Scenario:
    key: str
    trace: str
    start: str
    end: str  # exclusive
    kind: str
    description: str


SCENARIOS = {
    "steady": Scenario("steady", "nasa", "1995-08-26 00:00", "1995-08-27 00:00", "steady",
                       "NASA-HTTP, Saturday 26 Aug 1995: low, smooth weekend load"),
    "ramp": Scenario("ramp", "fifa", "1998-07-07 09:51", "1998-07-07 16:32", "gradual ramp",
                     "WorldCup98, 7 Jul 1998 09:51-16:31: the 400-min slice of the base paper's "
                     "autoscaler notebook; load roughly doubles over 6.5 h"),
    "bursty": Scenario("bursty", "fifa", "1998-07-07 18:00", "1998-07-07 23:00", "bursty",
                       "WorldCup98, 7 Jul 1998 18:00-23:00: semi-final flash crowd (22k -> 226k req/min) "
                       "and collapse"),
    "diurnal": Scenario("diurnal", "nasa", "1995-08-22 00:00", "1995-08-23 00:00", "diurnal",
                        "NASA-HTTP, Tuesday 22 Aug 1995: weekday diurnal rise and fall"),
}


def scenario_frame(key: str) -> pd.DataFrame:
    sc = SCENARIOS[key]
    df = load_trace(sc.trace)
    _, test = split(df)
    assert pd.Timestamp(sc.start) >= test.ds.min(), f"scenario {key} overlaps training data"
    # complete minute grid: a minute absent from the access log had zero requests
    grid = pd.DataFrame({"ds": pd.date_range(sc.start, sc.end, freq="min", inclusive="left")})
    out = grid.merge(df, on="ds", how="left").fillna({"y": 0.0})
    return out


def history_before(key: str, minutes: int) -> pd.DataFrame:
    """Observed trace immediately preceding a scenario (warm-up history for forecasters)."""
    sc = SCENARIOS[key]
    df = load_trace(sc.trace)
    t0 = pd.Timestamp(sc.start)
    return df[(df.ds < t0) & (df.ds >= t0 - pd.Timedelta(minutes=minutes))].reset_index(drop=True)


def describe(y: np.ndarray) -> dict:
    y = np.asarray(y, float)
    d = np.diff(y)
    return dict(minutes=int(len(y)), mean=float(y.mean()), max=float(y.max()), min=float(y.min()),
                cv=float(y.std() / y.mean()), peak_to_mean=float(y.max() / y.mean()),
                p99_abs_step_pct=float(np.percentile(np.abs(d) / np.maximum(y[:-1], 1), 99) * 100),
                lag1_autocorr=float(np.corrcoef(y[:-1], y[1:])[0, 1]))
