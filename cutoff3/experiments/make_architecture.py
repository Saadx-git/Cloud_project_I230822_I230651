"""Draw the system/experiment architecture figure (figures/fig_architecture.pdf/.png)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, GREEN, GREY = "#dbe8f8", "#fbe3d6", "#d9f1e7", "#efeeea"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

fig, ax = plt.subplots(figsize=(10.2, 4.6))
ax.set_xlim(0, 102)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, title, body="", fc=GREY, ec=LINE, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=8.3, color=INK,
            fontweight="bold" if bold else "normal")
    if body:
        ax.text(x + w / 2, y + h - 4.4, body, ha="center", va="top", fontsize=7, color=MUTED, linespacing=1.35)


def arrow(x1, y1, x2, y2, text="", dx=0, dy=0, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=INK, lw=0.9, ls=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", va="center", fontsize=6.6, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=0.6))


# cluster boundary
ax.add_patch(FancyBboxPatch((17, 1), 84, 44, boxstyle="round,pad=0.2,rounding_size=1.2", fc="none", ec=LINE,
                            lw=0.8, ls=(0, (4, 3))))
ax.text(18.5, 43.8, "kind cluster (single node, Kubernetes 1.32)", ha="left", va="top", fontsize=7, color=MUTED)

box(0.5, 18, 14, 13, "Load generator", "replay.py\nopen-loop Poisson\ntrace minute = 15 s\nNASA / WorldCup98")
box(21, 18, 17, 13, "Workload service", "Deployment 'work'\nPBKDF2 per request\n100m req / 200m lim\ncapacity C = 4 req/s", fc=GREEN)
box(45, 31, 17, 11, "Prometheus", "scrape every 5 s\napp_requests_total")
box(45, 8, 17, 9, "metrics-server", "CPU usage, 15 s")
box(69, 27, 30, 16, "Budgeted predictive scaler", "KEDA external scaler (gRPC), CPU limit 100m-500m\n"
    "every period P: observe y_t -> worker.forecast(t+1)\nwait <= deadline D (= P)\n"
    "late -> fallback y_t, kill + respawn worker\nlogs CPU/decision, latency, cgroup throttling", fc=BLUE)
box(69, 7, 13.5, 11, "HPA (B1)", "CPU target 70 %\ntolerance 10 %\nscale-down 75 s", fc=ORANGE)
box(85.5, 7, 13.5, 11, "KEDA -> HPA", "AverageValue\n= 0.7 C per pod\n(milli-req/s)", fc=BLUE)

arrow(14.8, 24.5, 20.8, 24.5, "HTTP")
arrow(38.2, 28, 44.8, 36, "requests\ncounter", dx=-1.5, dy=1.5)
arrow(38.2, 21, 44.8, 13, "CPU", dx=-1.5)
arrow(62.2, 36.5, 68.8, 36.5, "rate()")
arrow(62.2, 12.5, 68.8, 12.5, "")
arrow(92, 26.8, 92, 18.2, "forecast\nreq/s", dx=4.2)
# actuation: both HPAs set the replica count of 'work' (orthogonal routing below the boxes)
for x in (75.75, 92.25):
    ax.plot([x, x], [6.8, 4.0], color=INK, lw=0.9)
ax.plot([92.25, 29.5], [4.0, 4.0], color=INK, lw=0.9)
arrow(29.5, 4.0, 29.5, 17.8, "")
ax.text(52, 4.0, "set replicas = ceil(signal / 0.7 C): one replica policy shared by every autoscaler", ha="center",
        va="center", fontsize=7, color=INK, bbox=dict(fc="white", ec="none", pad=0.8))

fig.tight_layout()
os.makedirs(f"{ROOT}/figures", exist_ok=True)
fig.savefig(f"{ROOT}/figures/fig_architecture.pdf", bbox_inches="tight")
fig.savefig(f"{ROOT}/figures/fig_architecture.png", dpi=200, bbox_inches="tight")
print("figure fig_architecture")
