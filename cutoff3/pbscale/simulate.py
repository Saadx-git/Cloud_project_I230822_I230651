"""Trace-driven discrete-time autoscaling simulator (one step = one control period).

Both the reactive and the predictive autoscalers share the *same* replica
policy, so that differences come only from the input signal:

  desired_raw(t+1) = ceil( signal / (C * target_util) )
  scale-up         : applied immediately (pods become ready after `startup_steps`)
  scale-down       : Kubernetes stabilisation window - the highest recommendation
                     of the last `stabilization_steps` decisions is used
  bounds           : [min_pods, max_pods]

Signals
  hpa        : Kubernetes HPA rule on observed CPU utilisation of step t
               u = min(load_t / (ready_t * C), cpu_limit_ratio)   (CPU saturates at the limit)
               desired = ceil(ready_t * u / target_util); no change if |u/target - 1| <= tolerance
  predictive : forecast of the load at t+1 (from any forecaster); if the forecast was
               late (latency > deadline) the persistence fallback y_t is used instead.

Provisioned capacity at step t = ready_t * C * target_util (the load the pods can
serve while staying at the utilisation target that the SLO is tuned for).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .metrics import provisioning


@dataclass
class Policy:
    capacity: float               # load units one pod serves at 100 % of its CPU request
    target_util: float = 0.7
    tolerance: float = 0.1        # HPA default
    stabilization_steps: int = 5  # HPA default 300 s at a 60 s step
    startup_steps: int = 0        # pods requested at t serve step t+1+startup_steps
    min_pods: int = 1
    max_pods: int = 1000
    cpu_limit_ratio: float = 2.0  # pod CPU limit / CPU request


def _ceil_div(a, b):
    return int(np.ceil(max(a, 0.0) / b - 1e-9))


def simulate(y, pol: Policy, kind: str, forecast=None, late=None, init_pods=None) -> dict:
    """y[t]: load in step t.  forecast[t]: forecast of y[t] made at the end of step t-1.
    late[t]: True if that forecast missed its deadline (fallback y[t-1] is used)."""
    y = np.asarray(y, float)
    T = len(y)
    cap_t = pol.capacity * pol.target_util
    n0 = init_pods if init_pods is not None else max(pol.min_pods, _ceil_div(y[0], cap_t))
    pods = [0] * n0          # ready time of each pod (sorted ascending)
    target = n0              # replica count in the Deployment spec
    ready = np.zeros(T, int)
    ready[0] = n0
    recs, fallback = [], 0
    for t in range(T - 1):
        # ---- decision at the end of step t, for step t+1
        if kind == "hpa":
            u = min(y[t] / (ready[t] * pol.capacity), pol.cpu_limit_ratio)
            ratio = u / pol.target_util
            rec = ready[t] if abs(ratio - 1) <= pol.tolerance else int(np.ceil(ready[t] * ratio - 1e-9))
        else:
            f = forecast[t + 1]
            if (late is not None and late[t + 1]) or not np.isfinite(f):
                f = y[t]
                fallback += 1
            rec = _ceil_div(f, cap_t)
        rec = int(np.clip(rec, pol.min_pods, pol.max_pods))
        recs.append(rec)
        if rec >= target:
            new = rec
        else:  # scale-down stabilisation: highest recommendation in the window
            new = min(target, max(recs[-max(pol.stabilization_steps, 1):]))
        # ---- actuate
        if new > len(pods):
            pods += [t + 1 + pol.startup_steps] * (new - len(pods))
        elif new < len(pods):
            pods = sorted(pods)[:new]  # not-yet-ready (newest) pods are removed first
        target = new
        ready[t + 1] = sum(1 for r in pods if r <= t + 1)
    prov = ready * cap_t
    m = provisioning(y, prov)
    m.update(pod_steps=int(ready.sum()), mean_pods=float(ready.mean()), max_pods=int(ready.max()),
             saturated_pct=float(100 * np.mean(y > ready * pol.capacity)),
             fallback_decisions=int(fallback))
    return dict(metrics=m, ready=ready, policy=asdict(pol))
