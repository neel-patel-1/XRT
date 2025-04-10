import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(f"usage: {Path(__file__).name} <data_file> <output_file>")

# Read the input file path and output file path from command-line arguments
input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])

if not input_file.exists():
    sys.exit(f"Error: Input file {input_file} does not exist.")

# Load into DataFrame
df = pd.read_csv(input_file, delim_whitespace=True, header=None)

load = df.iloc[:, 0]
metric_cols = df.columns[1:]

baseline_value = df.iat[0, 1]          # ← first row, second column
slowdown = df.loc[:, 1:] / baseline_value
# Compute per‑metric slowdown vs. the first row
# slowdown = df[metric_cols].div(df.loc[0, metric_cols])

markers = ['x', 's', '^', 'D', 'v', 'o']

labels = [
    "No Acceleration",
    "Block & Wait",
    "RR-Worker",
    "XRT"
]
fig, ax = plt.subplots()
for idx, col in enumerate(metric_cols):
    ax.plot(load, slowdown[col], marker=markers[idx % len(markers)], label=labels[idx])

ax.set_yscale('log')
ax.set_xlabel('Load (MRPS)')
ax.set_ylabel('p99.9 Slowdown (log scale)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
fig.tight_layout()

# Save to PNG
output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.suffix != '.png':
    output_path = output_path.with_suffix('.png')
# Save the figure
if output_path.exists():
    print(f"Warning: {output_path} already exists. Overwriting.")
else:
    print(f"Saving figure to {output_path}")
# Save the figure
fig.savefig(output_path, dpi=300, bbox_inches='tight')