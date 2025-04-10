import argparse
import sys
from pathlib import Path
import pandas as pd
import re
import matplotlib.pyplot as plt

NUMERIC_COLS = ["Load", "NoAcc", "Block&Wait", "RR"]

def read_table(path: Path) -> pd.DataFrame:
    """Return the raw table as a DataFrame with numeric columns coerced."""
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["Name", "Load", "NoAcc", "Block&Wait", "RR"],
                     engine="python")
    # Force numeric dtype for arithmetic; errors='coerce' turns bad fields into NaN
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    return df


def plot_overhead(df: pd.DataFrame, metric: str, outfile: Path):
    """Plot overhead vs load for the chosen metric.

    Overhead is computed **relative to the first row of the same metric**
    (baseline), column‑wise:
        overhead[%] = (value / baseline_value - 1) * 100
    """
    sub = df[df["Name"] == metric].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"Metric '{metric}' not found in the input file.")

    # Row 0 of this metric is the baseline
    baseline = sub.loc[0, ["NoAcc", "Block&Wait", "RR"]]

    # Compute column‑wise overhead (skip columns whose baseline is 0)
    for col in ["Block&Wait", "RR", "NoAcc"]:
        if baseline[col] != 0:
            sub[col + "_ovr"] = (sub[col] / baseline[col] - 1.0) * 2
        else:
            # Undefined overhead when baseline is 0; set to NaN so it doesn't plot
            sub[col + "_ovr"] = float('nan')

    plt.figure(figsize=(6, 4))
    # NoAcceleration serves as 0‑line reference; plot markers but keep at y=0
    plt.axhline(0, color="#999", linewidth=0.8)
    plt.plot(sub["Load"], sub["NoAcc_ovr"], marker="s", label="NoAcceleration")
    plt.plot(sub["Load"], sub["Block&Wait_ovr"], marker="s", label="Block&Wait")
    plt.plot(sub["Load"], sub["RR_ovr"], marker="o", label="Yield")

    plt.xlabel("Load (MRPS)")
    plt.ylabel("Overhead (%)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved figure to {outfile}")


def main():
    logs_dir = Path("logs")
    pattern = re.compile(r"^(.*?)(_summary\.txt)\.(\d+)$")
    candidates = {}
    for file in logs_dir.glob("*_summary.txt.*"):
        m = pattern.match(file.name)
        if m:
            base = m.group(1) + m.group(2)
            num = int(m.group(3))
            if base not in candidates or num > candidates[base][1]:
                candidates[base] = (file, num)
    if "blocking_summary.txt" not in candidates:
        raise FileNotFoundError("No matching blocking_summary.txt.N file found in logs/")
    df = read_table(logs_dir / candidates["blocking_summary.txt"][0].name)
    plot_overhead(df, "AvgEnqueueTime" , "blocking_overhead.png")


if __name__ == "__main__":
    main()
