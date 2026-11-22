"""Generate scientific plots from the statistical benchmark."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "results" / "benchmarks"
    / "classical_vs_quantum"
    / "benchmark_classical_vs_quantum_statistics.json"
)

OUTPUT_DIR = (
    ROOT / "results" / "benchmarks"
    / "classical_vs_quantum" / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with INPUT.open(encoding="utf-8") as f:
    data = json.load(f)

classical = data["classical"]
quantum = data["quantum"]

# ------------------------------------------------------------
# 1. Execution time comparison
# ------------------------------------------------------------

names = ["A*", "Dijkstra", "Exhaustive", "Grover simulation"]

means = [
    classical["astar"]["timing_s"]["mean"],
    classical["dijkstra"]["timing_s"]["mean"],
    classical["exhaustive"]["timing_s"]["mean"],
    quantum["simulation_time_s"]["mean"],
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(names, means)

ax.set_yscale("log")
ax.set_ylabel("Execution time (seconds, logarithmic scale)")
ax.set_title("Classical vs Quantum Simulation Runtime")
ax.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, means):
    ax.annotate(
        f"{value:.3g} s",
        (bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_runtime_comparison.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# 2. Classical timing distributions
# ------------------------------------------------------------

labels = ["A*", "Dijkstra", "Exhaustive"]

timing_data = [
    classical["astar"]["individual_times_s"],
    classical["dijkstra"]["individual_times_s"],
    classical["exhaustive"]["individual_times_s"],
]

fig, ax = plt.subplots(figsize=(10, 6))

positions = np.arange(1, len(labels) + 1)

ax.boxplot(
    timing_data,
    positions=positions,
    widths=0.5,
    showmeans=True,
    showfliers=True,
)

# Display every individual measurement.
rng = np.random.default_rng(42)

for position, values in zip(positions, timing_data):
    jitter = rng.uniform(-0.10, 0.10, size=len(values))
    ax.scatter(
        position + jitter,
        values,
        alpha=0.65,
        s=24,
        label="Individual measurements" if position == 1 else None,
    )

ax.set_yscale("log")
ax.set_xticks(positions, labels)
ax.set_ylabel("Execution time (seconds, logarithmic scale)")
ax.set_title("Classical Planner Timing Distributions (30 Runs)")
ax.grid(axis="y", alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_classical_timing_distribution.png", dpi=300)
plt.close(fig)

# 3. Grover success probability per simulation
# ------------------------------------------------------------

runs = quantum["runs"]

run_numbers = [r["run"] for r in runs]
success = [100 * r["success_probability"] for r in runs]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(run_numbers, success, marker="o")
ax.axhline(
    quantum["aggregate_success_probability"] * 100,
    linestyle="--",
    label="Aggregate success",
)

ax.set_ylim(95, 101)
ax.set_xticks(run_numbers)
ax.set_xlabel("Simulation")
ax.set_ylabel("Valid shots (%)")
ax.set_title("Grover Success Probability")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_grover_success_rate.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# 4. Quantum pipeline timing
# ------------------------------------------------------------

pipeline_names = [
    "Construction",
    "Transpilation",
    "Simulation",
]

pipeline_times = [
    quantum["construction_time_s"],
    quantum["transpilation_time_s"],
    quantum["simulation_time_s"]["mean"],
]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(pipeline_names, pipeline_times)

ax.set_yscale("log")
ax.set_ylabel("Time (seconds, logarithmic scale)")
ax.set_title("Quantum Execution Pipeline")
ax.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, pipeline_times):
    ax.annotate(
        f"{value:.4g} s",
        (bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
    )

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_quantum_pipeline.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# 5. Summary CSV
# ------------------------------------------------------------

rows = [
    ["A*", classical["astar"]["timing_s"]],
    ["Dijkstra", classical["dijkstra"]["timing_s"]],
    ["Classical exhaustive", classical["exhaustive"]["timing_s"]],
    ["Grover simulation", quantum["simulation_time_s"]],
]

with (OUTPUT_DIR / "statistics_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "Algorithm", "Samples", "Mean_s", "Median_s",
        "StdDev_s", "Min_s", "Max_s"
    ])

    for name, stats in rows:
        writer.writerow([
            name,
            stats["samples"],
            stats["mean"],
            stats["median"],
            stats["stdev"],
            stats["min"],
            stats["max"],
        ])

print("===== SCIENTIFIC FIGURES GENERATED =====")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name, f"({path.stat().st_size} bytes)")

print("\nOutput directory:", OUTPUT_DIR)
