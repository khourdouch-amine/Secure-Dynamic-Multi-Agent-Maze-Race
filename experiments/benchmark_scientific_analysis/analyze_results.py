#!/usr/bin/env python3
"""Consolidated scientific analysis of classical and Grover benchmarks.

Original benchmark JSON files are read-only.
Outputs:
- six high-resolution figures,
- three CSV tables,
- one Markdown scientific report.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

MULTI_ROOT = ROOT / "results/benchmarks/multi_scale_statistics"
BASELINE_FILE = (
    ROOT / "results/benchmarks/classical_vs_quantum"
    / "benchmark_classical_vs_quantum_statistics.json"
)

OUTPUT = ROOT / "results/benchmarks/scientific_analysis"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

SCENARIO_FILES = {
    "G1_3x3_L4": "g1_3x3_l4_statistics.json",
    "G2_4x4_L4": "g2_4x4_l4_statistics.json",
    "G3_4x4_L6": "g3_4x4_l6_statistics.json",
}

COLORS = {
    "astar": "#2878B5",
    "dijkstra": "#55A868",
    "exhaustive": "#C58A32",
    "grover": "#8172B2",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file missing: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def seconds_to_ms(value):
    return None if value is None else value * 1000.0


def format_time(value):
    if value is None:
        return "Not measured"
    if value < 1e-3:
        return f"{value * 1e6:.2f} µs"
    if value < 1:
        return f"{value * 1e3:.3f} ms"
    return f"{value:.3f} s"


def get_ci(stats):
    ci = stats.get("mean_confidence_interval")
    if ci is None:
        return None
    if len(ci) != 2 or not all(math.isfinite(float(x)) for x in ci):
        return None
    return [float(ci[0]), float(ci[1])]


def get_std(stats):
    return stats.get("stdev_sample", stats.get("stdev"))


def load_datasets():
    datasets = {}

    for name, filename in SCENARIO_FILES.items():
        data = load_json(MULTI_ROOT / filename)
        datasets[name] = data

    datasets["BASELINE"] = load_json(BASELINE_FILE)
    return datasets


def get_scenario_info(name, data):
    scenario = data.get("scenario", {})
    configuration = data.get("configuration", {})

    if scenario:
        return {
            "name": name,
            "grid": scenario.get("grid"),
            "route_length": scenario.get("route_length"),
            "search_space": scenario.get("search_space"),
            "valid_routes": scenario.get("valid_routes"),
        }

    return {
        "name": name,
        "grid": None,
        "route_length": configuration.get("route_length"),
        "search_space": configuration.get("search_space"),
        "valid_routes": configuration.get("valid_routes"),
    }


def classical_rows(datasets):
    rows = []

    for scenario_name in SCENARIO_FILES:
        data = datasets[scenario_name]
        info = get_scenario_info(scenario_name, data)

        for method in ("astar", "dijkstra", "exhaustive"):
            result = data["classical"][method]
            stats = result["timing_s"]

            rows.append({
                "scenario": scenario_name,
                "grid": "x".join(map(str, info["grid"])),
                "route_length": info["route_length"],
                "search_space": info["search_space"],
                "valid_routes": info["valid_routes"],
                "method": method,
                "objective": result.get("objective", ""),
                "samples": stats["samples"],
                "mean_s": stats["mean"],
                "median_s": stats["median"],
                "std_sample_s": get_std(stats),
                "min_s": stats["min"],
                "max_s": stats["max"],
                "ci95_low_s": (
                    get_ci(stats)[0] if get_ci(stats) else None
                ),
                "ci95_high_s": (
                    get_ci(stats)[1] if get_ci(stats) else None
                ),
                "valid_runs": result.get("valid_runs"),
                "total_runs": result.get("total_runs"),
            })

    return rows


def quantum_rows(datasets):
    rows = []

    for scenario_name in SCENARIO_FILES:
        data = datasets[scenario_name]
        quantum = data.get("quantum")
        structure = data.get("grover_structure", {})

        if quantum is None:
            rows.append({
                "scenario": scenario_name,
                "simulated": False,
                "shots": 0,
                "success_probability": None,
                "wilson_low": None,
                "wilson_high": None,
                "simulation_mean_s": None,
                "simulation_median_s": None,
                "construction_s": structure.get("construction_time_s"),
                "transpilation_s": None,
                "qubits": structure.get("qubits"),
                "initial_depth": structure.get("initial_depth"),
                "transpiled_depth": None,
                "transpiled_operations": None,
                "iterations": structure.get("iterations"),
                "approximation_warning": False,
                "bond_dimension_cap": None,
            })
            continue

        runs = quantum["runs"]
        ci = quantum["aggregate_success_probability_wilson_95_ci"]

        rows.append({
            "scenario": scenario_name,
            "simulated": True,
            "shots": quantum["total_shots"],
            "success_probability": quantum["aggregate_success_probability"],
            "wilson_low": ci[0],
            "wilson_high": ci[1],
            "simulation_mean_s": quantum["simulation_time_s"]["mean"],
            "simulation_median_s": quantum["simulation_time_s"]["median"],
            "construction_s": structure.get("construction_time_s"),
            "transpilation_s": quantum.get("transpilation_time_s"),
            "qubits": structure.get("qubits"),
            "initial_depth": structure.get("initial_depth"),
            "transpiled_depth": quantum.get("transpiled_depth"),
            "transpiled_operations": quantum.get("transpiled_operations"),
            "iterations": structure.get("iterations"),
            "approximation_warning": quantum["approximation_warning"],
            "bond_dimension_cap": quantum.get("bond_dimension_cap"),
        })

    return rows


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classical_planners(datasets):
    scenarios = list(SCENARIO_FILES)
    methods = ("astar", "dijkstra")
    labels = {"astar": "A*", "dijkstra": "Dijkstra"}

    x = np.arange(len(scenarios))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, method in enumerate(methods):
        means = [
            datasets[s]["classical"][method]["timing_s"]["mean"] * 1e6
            for s in scenarios
        ]

        errors_low = []
        errors_high = []

        for s in scenarios:
            stats = datasets[s]["classical"][method]["timing_s"]
            ci = get_ci(stats)
            mean = stats["mean"]

            if ci:
                errors_low.append(max(0.0, (mean - ci[0]) * 1e6))
                errors_high.append(max(0.0, (ci[1] - mean) * 1e6))
            else:
                errors_low.append(0.0)
                errors_high.append(0.0)

        ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            label=labels[method],
            color=COLORS[method],
            yerr=np.array([errors_low, errors_high]),
            capsize=4,
        )

    ax.set_xticks(x, scenarios)
    ax.set_ylabel("Mean execution time (µs)")
    ax.set_title("Shortest-path planners across benchmark scenarios")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, "01_classical_planners.png")


def plot_exhaustive(datasets):
    scenarios = list(SCENARIO_FILES)
    means = []
    medians = []
    low = []
    high = []

    for scenario in scenarios:
        stats = datasets[scenario]["classical"]["exhaustive"]["timing_s"]
        means.append(stats["mean"] * 1000)
        medians.append(stats["median"] * 1000)

        ci = get_ci(stats)
        if ci:
            # Preserve the complete Student-t confidence interval.
            low.append((stats["mean"] - ci[0]) * 1000)
            high.append((ci[1] - stats["mean"]) * 1000)
        else:
            low.append(0.0)
            high.append(0.0)

    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        x, means, color=COLORS["exhaustive"],
        yerr=np.array([low, high]), capsize=5,
        label="Mean ± 95% CI",
    )

    ax.scatter(
        x, medians, color="black", marker="D",
        s=42, label="Median", zorder=4,
    )

    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.3f} ms",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 7), textcoords="offset points",
            ha="center", fontsize=9,
        )

    ax.set_xticks(x, scenarios)
    ax.set_ylabel("Execution time (ms)")
    ax.set_title("Classical exhaustive fixed-length route enumeration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, "02_exhaustive_search.png")


def plot_quantum_pipeline(datasets):
    scenarios = ["G1_3x3_L4", "G2_4x4_L4"]
    stages = ["Construction", "Transpilation", "Simulation"]

    values = []

    for scenario in scenarios:
        data = datasets[scenario]
        q = data["quantum"]
        structure = data.get("grover_structure", {})

        values.append([
            structure.get("construction_time_s", 0),
            q.get("transpilation_time_s", 0),
            q["simulation_time_s"]["mean"],
        ])

    x = np.arange(len(scenarios))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))

    stage_colors = ["#4C9F70", "#D49A36", COLORS["grover"]]

    for i, stage in enumerate(stages):
        vals = [values[j][i] for j in range(len(scenarios))]
        ax.bar(
            x + (i - 1) * width, vals, width,
            label=stage, color=stage_colors[i],
        )

    ax.set_yscale("log")
    ax.set_xticks(x, scenarios)
    ax.set_ylabel("Time (s, logarithmic scale)")
    ax.set_title("Geometric Grover pipeline: CPU simulation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    save_figure(fig, "03_grover_pipeline.png")


def plot_grover_success(datasets):
    scenarios = ["G1_3x3_L4", "G2_4x4_L4"]
    fig, ax = plt.subplots(figsize=(9, 6))

    for index, scenario in enumerate(scenarios):
        q = datasets[scenario]["quantum"]
        p = q["aggregate_success_probability"] * 100
        ci = q["aggregate_success_probability_wilson_95_ci"]
        lower = (p - ci[0] * 100)
        upper = (ci[1] * 100 - p)

        ax.errorbar(
            index, p,
            yerr=[[lower], [upper]],
            fmt="o", markersize=9, capsize=7,
            label=f"{scenario} ({q['total_shots']} shots)",
        )

    ax.set_xticks(range(len(scenarios)), scenarios)
    ax.set_ylim(90, 101)
    ax.set_ylabel("Valid measured routes (%)")
    ax.set_title("Grover aggregate success with Wilson 95% intervals")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax.text(
        0.02, 0.02,
        "G2: approximate MPS simulation, bond dimension ≤ 128.",
        transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom",
    )

    save_figure(fig, "04_grover_success.png")


def plot_circuit_structure(datasets):
    scenarios = list(SCENARIO_FILES)
    qubits = []
    depths = []
    iterations = []

    for scenario in scenarios:
        structure = datasets[scenario].get("grover_structure", {})
        qubits.append(structure.get("qubits", 0))
        depths.append(structure.get("initial_depth", 0))
        iterations.append(structure.get("iterations", 0))

    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, qubits, color=COLORS["grover"])
    axes[0].set_xticks(x, scenarios, rotation=15)
    axes[0].set_ylabel("Qubits")
    axes[0].set_title("Logical circuit width")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, depths, color="#C58A32")
    axes[1].set_xticks(x, scenarios, rotation=15)
    axes[1].set_ylabel("Untranspiled circuit depth")
    axes[1].set_title("Initial circuit depth")
    axes[1].grid(axis="y", alpha=0.25)

    save_figure(fig, "05_circuit_structure.png")


def plot_classical_distributions(datasets):
    scenarios = list(SCENARIO_FILES)
    methods = ("astar", "dijkstra", "exhaustive")
    labels = ("A*", "Dijkstra", "Exhaustive")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rng = np.random.default_rng(2026)

    for ax, scenario in zip(axes, scenarios):
        samples = [
            datasets[scenario]["classical"][method]["individual_times_s"]
            for method in methods
        ]

        samples_us = [[v * 1e6 for v in values] for values in samples]

        ax.boxplot(
            samples_us,
            tick_labels=labels,
            showmeans=True,
            showfliers=True,
        )

        for i, values in enumerate(samples_us, start=1):
            jitter = rng.uniform(-0.08, 0.08, len(values))
            ax.scatter(
                i + jitter, values,
                s=14, alpha=0.55,
            )

        ax.set_yscale("log")
        ax.set_title(scenario)
        ax.set_ylabel("Execution time (µs)")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Classical timing distributions (30 measurements per method)")
    save_figure(fig, "06_classical_distributions.png")


def build_report(datasets, classical, quantum):
    lines = [
        "# Consolidated Scientific Analysis",
        "",
        "## 1. Scope",
        "",
        "This report consolidates the classical and geometric Grover "
        "benchmark campaigns. It uses the recorded JSON measurements "
        "without altering the source datasets.",
        "",
        "The analysis covers three multi-scale scenarios:",
        "",
        "| Scenario | Grid | Route length | Search space | Valid routes |",
        "|---|---:|---:|---:|---:|",
    ]

    for name in SCENARIO_FILES:
        info = get_scenario_info(name, datasets[name])
        lines.append(
            f"| {name} | {info['grid'][0]}×{info['grid'][1]} "
            f"| {info['route_length']} | {info['search_space']} "
            f"| {info['valid_routes']} |"
        )

    lines += [
        "",
        "## 2. Experimental methodology",
        "",
        "- A* and Dijkstra: 30 measured repetitions per scenario.",
        "- Classical exhaustive enumeration: 30 repetitions per scenario.",
        "- Classical warm-up runs are excluded from timing.",
        "- Grover: five independent Aer CPU simulations for G1 and G2.",
        "- G1: 1,024 shots per simulation, without an explicit MPS bond cap.",
        "- G2: 32 shots per simulation, MPS bond dimension capped at 128.",
        "- G3: no Grover transpilation or simulation.",
        "- Classical timing intervals use the recorded Student-t confidence intervals.",
        "- Grover success intervals use Wilson score confidence intervals.",
        "",
        "## 3. Objective compatibility",
        "",
        "A* and Dijkstra solve a shortest-path problem. Grover searches "
        "a fixed-length encoded route space. These are different objectives "
        "and their runtimes must not be interpreted as a direct speed ranking.",
        "",
        "Classical exhaustive enumeration is the direct algorithmic "
        "comparator for Grover because both evaluate the same fixed-length "
        "route search space. The exhaustive baseline enumerates all encoded "
        "candidate routes and returns the valid indices.",
        "",
        "This shared search space establishes objective compatibility, "
        "not direct execution-time equivalence. The classical baseline runs "
        "as CPU code, whereas Aer simulates a quantum circuit using a "
        "classical matrix-product-state method. Their measured wall-clock "
        "times therefore include fundamentally different computational "
        "workloads and must not be interpreted as a quantum speedup ratio.",
        "",
        "The exhaustive timing covers the call to "
        "`GlobalClassicalOracle.find_valid_indices()` only. One warm-up "
        "call is excluded. Thirty measured calls are timed individually; "
        "result-consistency and route-validity checks occur after each "
        "timed call and are not included in the recorded enumeration time.",
        "",
        "## 4. Classical results",
        "",
        "| Scenario | Method | Mean | Median | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]

    for row in classical:
        ci = (
            f"[{format_time(row['ci95_low_s'])}, "
            f"{format_time(row['ci95_high_s'])}]"
            if row["ci95_low_s"] is not None else "Unavailable"
        )

        lines.append(
            f"| {row['scenario']} | {row['method']} "
            f"| {format_time(row['mean_s'])} "
            f"| {format_time(row['median_s'])} | {ci} |"
        )

    lines += [
        "",
        "### Interpretation",
        "",
        "A* and Dijkstra remain extremely fast on these small grids. "
        "Their measured runtimes increase with scenario size, but the "
        "sample sizes and limited grid scales do not support extrapolation "
        "to large mazes.",
        "",
        "G1 exhaustive enumeration has a pronounced high timing outlier. "
        "Its mean is approximately 1.741 ms, while its median is "
        "approximately 0.504 ms. The raw observation is retained; "
        "the median provides a more robust description of its central tendency. "
        "The conventional Student-t 95% confidence interval for the G1 mean "
        "extends below zero (approximately -0.697 to 4.179 ms). Negative "
        "execution time is physically impossible; this lower bound is a "
        "statistical artifact of applying a symmetric t interval to a highly "
        "skewed sample with an extreme observation. It is shown untruncated "
        "and should not be interpreted as a physically plausible time bound.",
        "",
        "G2 exhaustive enumeration has a mean near 0.487 ms. G3, with "
        "4,096 candidate sequences, has a mean near 8.515 ms. This "
        "illustrates the growth in exhaustive evaluation cost over the "
        "tested scenarios, not a universal complexity estimate.",
        "",
        "## 5. Grover simulation results",
        "",
        "| Scenario | Shots | Valid shots | Success | Wilson 95% CI | Mean simulation |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in ("G1_3x3_L4", "G2_4x4_L4"):
        q = datasets[name]["quantum"]
        ci = q["aggregate_success_probability_wilson_95_ci"]
        lines.append(
            f"| {name} | {q['total_shots']} "
            f"| {q['aggregate_valid_shots']} "
            f"| {100*q['aggregate_success_probability']:.3f}% "
            f"| [{100*ci[0]:.3f}%, {100*ci[1]:.3f}%] "
            f"| {q['simulation_time_s']['mean']:.3f} s |"
        )

    lines += [
        "",
        "G1 achieved 99.727% aggregate valid measurements over 5,120 shots. "
        "G2 achieved 100% observed valid measurements over 160 shots. "
        "This is an empirical result for the tested sample, not a guarantee "
        "of perfect success probability. Its Wilson interval is wider than "
        "G1's because the sample contains substantially fewer shots. "
        "Furthermore, G2 uses an MPS bond-dimension cap of 128, so its "
        "simulation is approximate and should be interpreted diagnostically.",
        "",
        "The five simulator runs use 32 shots each. The aggregate Wilson "
        "interval describes the pooled shot outcomes; it does not quantify "
        "all sources of simulator approximation or variability between "
        "independent runs.",
        "",
        "These observations describe the tested simulator configurations. "
        "They do not establish performance on quantum hardware."
        "",
        "## 6. Circuit structure",
        "",
        "| Scenario | Qubits | Initial depth | Transpiled depth | Transpiled operations |",
        "|---|---:|---:|---:|---:|",
    ]

    for name in SCENARIO_FILES:
        structure = datasets[name].get("grover_structure", {})
        q = datasets[name].get("quantum")
        lines.append(
            f"| {name} | {structure.get('qubits')} "
            f"| {structure.get('initial_depth')} "
            f"| {q.get('transpiled_depth') if q else 'Not performed'} "
            f"| {q.get('transpiled_operations') if q else 'Not performed'} |"
        )

    lines += [
        "",
        "The circuit grows from 18 qubits and initial depth 955 in G1 "
        "to 26 qubits and initial depth 3,243 in G3. G3 was not "
        "transpiled or simulated because of the computational resource "
        "constraints of the experimental environment.",
        "",
        "## 7. Important implementation limitation",
        "",
        "The Grover circuit construction process performs classical "
        "solution counting to determine the number of Grover iterations. "
        "Consequently, the measured pipeline is not a fully end-to-end "
        "quantum search procedure. This classical preprocessing must be "
        "considered when interpreting algorithmic performance.",
        "",
        "## 8. Threats to validity and limitations",
        "",
        "1. Small grids and search spaces limit generalization.",
        "2. A* and Dijkstra do not solve the same objective as fixed-length Grover.",
        "3. Aer MPS simulation is a classical simulation of a quantum circuit.",
        "4. G2 uses an approximate MPS configuration and only 160 total shots.",
        "5. G3 has no transpilation or simulation measurements.",
        "6. G1 exhaustive timing contains a strong outlier.",
        "7. CPU timing depends on the host machine, Python runtime and system load.",
        "8. Grover iteration selection uses classically counted solutions.",
        "9. No hardware execution or quantum advantage has been demonstrated.",
        "",
        "## 9. Conclusion",
        "",
        "The experiments demonstrate a working geometric Grover circuit "
        "for the tested small fixed-length route-search instances, with "
        "high observed valid-measurement proportions in G1 and G2. They "
        "also show that classical planners are extremely efficient on "
        "these small mazes and that classical exhaustive enumeration "
        "remains inexpensive at the tested search-space sizes.",
        "",
        "The present results support a methodological and implementation "
        "comparison, not a claim of quantum computational advantage. "
        "Larger instances, fair end-to-end baselines, hardware-aware "
        "resource accounting and additional independent experiments "
        "would be required for stronger conclusions.",
        "",
        "## 10. Experimental computing environment",
        "",
        "- Host processor: Intel Core i5-1135G7, 2.40 GHz, 4 cores and 8 threads.",
        "- Execution environment: Ubuntu under WSL2, x86_64.",
        "- Python: 3.12.3.",
        "- Qiskit: 2.6.2; Qiskit Aer: 0.18.4.",
        "- NumPy: 2.5.5; SciPy: 1.18.1; Matplotlib: 3.11.2.",
        "- Aer simulation method: `matrix_product_state`.",
        "- Aer MPS OpenMP threads: 1; configured memory limit: 7,000 MB.",
        "",
        "Reported execution times are environment-dependent. They should "
        "be reproduced on the same or a documented equivalent environment "
        "before making timing comparisons across machines.",
        "",
        "## 11. Reproducibility",
        "",
        "Source datasets:",
        "",
        "- `results/benchmarks/classical_vs_quantum/benchmark_classical_vs_quantum_statistics.json`",
        "- `results/benchmarks/multi_scale_statistics/g1_3x3_l4_statistics.json`",
        "- `results/benchmarks/multi_scale_statistics/g2_4x4_l4_statistics.json`",
        "- `results/benchmarks/multi_scale_statistics/g3_4x4_l6_statistics.json`",
        "",
        "Figures and CSV tables are generated by "
        "`experiments/benchmark_scientific_analysis/analyze_results.py`.",
        "",
    ]

    report_path = OUTPUT / "scientific_analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    print("Loading experimental datasets...")
    datasets = load_datasets()

    print("Extracting classical measurements...")
    classical = classical_rows(datasets)

    print("Extracting Grover measurements...")
    quantum = quantum_rows(datasets)

    print("Writing CSV tables...")
    write_csv(TABLES / "classical_statistics.csv", classical)
    write_csv(TABLES / "grover_statistics.csv", quantum)

    individual_rows = []
    for scenario in SCENARIO_FILES:
        for method in ("astar", "dijkstra", "exhaustive"):
            values = datasets[scenario]["classical"][method]["individual_times_s"]
            for index, value in enumerate(values, start=1):
                individual_rows.append({
                    "scenario": scenario,
                    "method": method,
                    "run": index,
                    "time_s": value,
                })

    write_csv(TABLES / "classical_individual_measurements.csv", individual_rows)

    print("Generating six scientific figures...")
    plot_classical_planners(datasets)
    plot_exhaustive(datasets)
    plot_quantum_pipeline(datasets)
    plot_grover_success(datasets)
    plot_circuit_structure(datasets)
    plot_classical_distributions(datasets)

    print("Writing scientific report...")
    build_report(datasets, classical, quantum)

    print("\n===== SCIENTIFIC ANALYSIS GENERATED =====")
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    print("\nOriginal benchmark datasets were not modified.")


if __name__ == "__main__":
    main()
