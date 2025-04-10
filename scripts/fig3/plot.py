#!/usr/bin/env python3
"""
plot_p99_slowdown.py

Read two summary files with columns
    Load  AverageLatency  99thPercentileLatency  CompletionRate
and draw p99‑slowdown curves (log‑scale) like the reference
figure.  Slowdown is defined as

    slowdown = 99thPercentileLatency / min(99thPercentileLatency)

within each data set, so the lowest‑load point is 1·×.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

worker_file = "logs/worker_notified_forward_summary.txt"
dispatcher_file = "logs/worker_vs_dispatcher_summary.txt"

def load_and_normalise(path: Path):
    """Return (loads, slowdown) for one data set."""
    df = pd.read_csv(path, delim_whitespace=True)
    baseline = df["AverageLatency"].min()
    slowdown = df["99thPercentileLatency"] / baseline
    return df["Load"], slowdown

w_load, w_slow = load_and_normalise(worker_file)
d_load, d_slow = load_and_normalise(dispatcher_file)

# ── plotting ──────────────────────────────────────────────────────────────
plt.figure(figsize=(4.5, 5))           # roughly matches the original aspect
plt.plot(w_load, w_slow,
         marker="x",  markersize=8,  linewidth=1.5,
         label="Worker‑Centric")
plt.plot(d_load, d_slow,
         marker="s",  markersize=6,  linewidth=1.5,
         label="Dispatcher‑Centric",
         color="tab:orange")

plt.yscale("log")
plt.yticks([1, 2, 4, 16, 64, 256], [1, 2, 4, 16, 64, 256])
plt.ylabel("p99 Slowdown (log scale)")
plt.xlabel("Load (MRPS)")
plt.xlim(left=0, right=3.1)
plt.ylim(bottom=1, top=256)
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)

out = "worker_vs_dispatcher.png"
plt.tight_layout()
plt.savefig(out, dpi=300)
print(f"Wrote {out}")