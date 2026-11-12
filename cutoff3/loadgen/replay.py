"""Open-loop trace replayer (standard library only).

Each trace minute is replayed for `--step` wall-clock seconds at a constant
Poisson arrival rate  lambda = y_trace * scale  (requests/s).  Every request
opens a new TCP connection (HTTP/1.0) so that kube-proxy spreads load across
pods.  Per-second aggregates are written to a CSV file:

  t_s, trace_min, offered_rps, sent, ok, err, timeout, slow(>SLO), lat_p50_ms, lat_p95_ms, lat_max_ms

Usage:
  python replay.py --trace-csv slice.csv --scale 0.0004 --step 15 --url http://127.0.0.1:30080/work --out load.csv
"""
import argparse
import asyncio
import csv
import random
import time
from urllib.parse import urlparse


async def one(host, port, path, timeout):
    t0 = time.perf_counter()
    try:
        r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        w.write(f"GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
        await w.drain()
        line = await asyncio.wait_for(r.readline(), timeout)
        await asyncio.wait_for(r.read(), timeout)
        w.close()
        ok = line.split()[1:2] == [b"200"]
        return ("ok" if ok else "err"), time.perf_counter() - t0
    except asyncio.TimeoutError:
        return "timeout", time.perf_counter() - t0
    except OSError:
        return "err", time.perf_counter() - t0


def pct(xs, q):
    if not xs:
        return ""
    xs = sorted(xs)
    return round(1000 * xs[min(len(xs) - 1, int(q * len(xs)))], 2)


async def main(a):
    rows = list(csv.DictReader(open(a.trace_csv)))
    u = urlparse(a.url)
    host, port, path = u.hostname, u.port or 80, u.path or "/"
    rng = random.Random(a.seed)
    buckets = {}  # second -> dict
    pending = set()
    t_start = time.perf_counter()

    def rec(sec, key, lat=None):
        b = buckets.setdefault(sec, dict(sent=0, ok=0, err=0, timeout=0, slow=0, lat=[]))
        b[key] += 1
        if lat is not None:
            b["lat"].append(lat)
            if lat > a.slo:
                b["slow"] += 1

    async def fire(sec):
        res, lat = await one(host, port, path, a.timeout)
        rec(sec, res, lat if res == "ok" else None)

    t_next = 0.0
    for i, row in enumerate(rows):
        lam = float(row["y"]) * a.scale
        seg_end = (i + 1) * a.step
        while True:
            t_next += rng.expovariate(lam) if lam > 0 else a.step
            if t_next >= seg_end:
                t_next = seg_end
                break
            delay = t_start + t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            sec = int(t_next)
            rec(sec, "sent")
            buckets[sec]["trace_min"] = i
            buckets[sec]["offered"] = lam
            task = asyncio.create_task(fire(sec))
            pending.add(task)
            task.add_done_callback(pending.discard)
    if pending:
        await asyncio.wait(pending, timeout=a.timeout + 1)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "trace_min", "offered_rps", "sent", "ok", "err", "timeout", "slow",
                    "lat_p50_ms", "lat_p95_ms", "lat_max_ms"])
        for s in sorted(buckets):
            b = buckets[s]
            w.writerow([s, b.get("trace_min", ""), b.get("offered", ""), b["sent"], b["ok"], b["err"],
                        b["timeout"], b["slow"], pct(b["lat"], .5), pct(b["lat"], .95), pct(b["lat"], 1.0)])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace-csv", required=True, help="CSV with columns ds,y (requests per trace minute)")
    p.add_argument("--scale", type=float, required=True, help="live req/s per trace req/min")
    p.add_argument("--step", type=float, default=15.0, help="wall seconds per trace minute")
    p.add_argument("--url", default="http://127.0.0.1:30080/work")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--slo", type=float, default=0.5, help="latency SLO in seconds (slow = ok but slower)")
    p.add_argument("--out", required=True)
    asyncio.run(main(p.parse_args()))
