"""Reproducible multi-scale statistical benchmark: classical search vs Grover."""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

from scipy.stats import norm, t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qiskit import transpile
from qiskit_aer import AerSimulator

from planners.classical.astar import AStarPlanner
from planners.classical.dijkstra import DijkstraPlanner

from planners.quantum.grover_global.maze_model import GroverMazeModel
from planners.quantum.grover_global.classical_oracle import GlobalClassicalOracle
from planners.quantum.grover_global.geometric_grover import build_geometric_grover
from planners.quantum.route_encoding import RouteEncoding

from experiments.benchmark_multi_scale.scenarios import SCENARIOS
from experiments.benchmark_multi_scale.benchmark_classical import (
    GridMazeAdapter,
    validate_path,
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASSICAL_REPETITIONS = 30
GROVER_REPETITIONS = 5
BASE_SEED = 4200
TRANSPILER_SEED = 42
CONFIDENCE_LEVEL = 0.95

SIMULATION_CASES = {
    "G1_3x3_L4": {
        "shots": 1024,
        "max_bond_dimension": None,
    },
    "G2_4x4_L4": {
        "shots": 32,
        "max_bond_dimension": 128,
    },
}

SIMULATOR_BASE_OPTIONS = {
    "max_memory_mb": 7000,
    "mps_omp_threads": 1,
    "matrix_product_state_truncation_threshold": 1e-16,
}

OUTPUT_DIR = ROOT / "results" / "benchmarks" / "multi_scale_statistics"


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def describe(values):
    """Descriptive statistics and two-sided Student 95% CI."""
    values = [float(value) for value in values]

    if not values:
        raise ValueError("Cannot describe an empty sample.")

    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0

    if len(values) > 1:
        standard_error = stdev / math.sqrt(len(values))
        critical = float(
            student_t.ppf(
                (1.0 + CONFIDENCE_LEVEL) / 2.0,
                df=len(values) - 1,
            )
        )
        margin = critical * standard_error
        ci = [mean - margin, mean + margin]
    else:
        standard_error = 0.0
        ci = None

    return {
        "samples": len(values),
        "mean": mean,
        "median": median,
        "stdev_sample": stdev,
        "standard_error": standard_error,
        "min": min(values),
        "max": max(values),
        "confidence_level": CONFIDENCE_LEVEL,
        "mean_confidence_interval": ci,
        "confidence_interval_method": "two-sided Student t",
    }


def wilson_interval(successes, total, confidence=CONFIDENCE_LEVEL):
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("Total observations must be positive.")

    if not 0 <= successes <= total:
        raise ValueError("Success count must be between zero and total.")

    z = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / total
    denominator = 1.0 + z * z / total

    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )

    return [
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    ]


# ============================================================
# SCENARIO MODEL AND VALIDATION
# ============================================================

def create_model(scenario):
    return GroverMazeModel(
        width=scenario.width,
        height=scenario.height,
        start=scenario.start,
        goal=scenario.goal,
        obstacles=scenario.obstacles,
    )


def validate_route_indices(indices, oracle):
    """Independently validate every route returned by exhaustive search."""
    for index in indices:
        if not oracle.evaluate(index):
            return False
    return True


def validate_classical_path(path, maze, scenario):
    return validate_path(
        path,
        maze,
        scenario.start,
        scenario.goal,
    )


# ============================================================
# CLASSICAL STATISTICAL BENCHMARK
# ============================================================

def benchmark_classical(scenario, model):
    print("\n===== CLASSICAL BENCHMARK =====", flush=True)

    maze = GridMazeAdapter(
        scenario.width,
        scenario.height,
        scenario.obstacles,
    )

    planners = {
        "astar": AStarPlanner(),
        "dijkstra": DijkstraPlanner(),
    }

    classical_results = {}

    for key, planner in planners.items():
        print(
            f"\n--- {planner.name}: warm-up + "
            f"{CLASSICAL_REPETITIONS} measurements ---",
            flush=True,
        )

        # Warm-up is deliberately excluded from the measurements.
        warmup_path = planner.plan(maze, scenario.start, scenario.goal)

        if not validate_classical_path(warmup_path, maze, scenario):
            raise RuntimeError(
                f"{planner.name} warm-up returned an invalid path "
                f"for {scenario.name}."
            )

        times = []
        paths = []

        for run in range(CLASSICAL_REPETITIONS):
            start_time = time.perf_counter()
            path = planner.plan(maze, scenario.start, scenario.goal)
            elapsed = time.perf_counter() - start_time

            valid = validate_classical_path(path, maze, scenario)

            if not valid:
                raise RuntimeError(
                    f"{planner.name} returned an invalid path "
                    f"in run {run + 1} for {scenario.name}."
                )

            times.append(elapsed)
            paths.append(path)

            print(
                f"{planner.name} {run + 1:02d}/{CLASSICAL_REPETITIONS}: "
                f"{elapsed:.8f} s | valid=True",
                flush=True,
            )

        if any(path != paths[0] for path in paths):
            raise RuntimeError(
                f"{planner.name} returned inconsistent paths "
                f"for {scenario.name}."
            )

        classical_results[key] = {
            "algorithm": planner.name,
            "objective": "shortest path",
            "timing_s": describe(times),
            "individual_times_s": times,
            "valid_runs": len(paths),
            "total_runs": CLASSICAL_REPETITIONS,
            "path": [list(position) for position in paths[0]],
            "path_length_moves": len(paths[0]) - 1,
            "all_paths_valid": True,
        }

    # Direct classical counterpart to Grover's fixed-length route search.
    print(
        f"\n--- Exhaustive enumeration: warm-up + "
        f"{CLASSICAL_REPETITIONS} measurements ---",
        flush=True,
    )

    oracle = GlobalClassicalOracle(model, scenario.route_length)

    warmup_indices = oracle.find_valid_indices()

    if not validate_route_indices(warmup_indices, oracle):
        raise RuntimeError("Invalid route found during exhaustive warm-up.")

    exhaustive_times = []
    reference_indices = list(warmup_indices)

    for run in range(CLASSICAL_REPETITIONS):
        start_time = time.perf_counter()
        indices = oracle.find_valid_indices()
        elapsed = time.perf_counter() - start_time

        if indices != reference_indices:
            raise RuntimeError(
                f"Inconsistent exhaustive results in {scenario.name}."
            )

        if not validate_route_indices(indices, oracle):
            raise RuntimeError(
                f"Invalid exhaustive route in {scenario.name}."
            )

        exhaustive_times.append(elapsed)

        print(
            f"Exhaustive {run + 1:02d}/{CLASSICAL_REPETITIONS}: "
            f"{elapsed:.8f} s | solutions={len(indices)}",
            flush=True,
        )

    classical_results["exhaustive"] = {
        "algorithm": "Classical exhaustive route enumeration",
        "objective": "enumerate all valid fixed-length routes",
        "timing_s": describe(exhaustive_times),
        "individual_times_s": exhaustive_times,
        "valid_routes": len(reference_indices),
        "valid_indices": reference_indices,
        "search_space": oracle.search_space,
        "valid_runs": CLASSICAL_REPETITIONS,
        "total_runs": CLASSICAL_REPETITIONS,
        "all_results_consistent": True,
    }

    return classical_results, reference_indices


# ============================================================
# GROVER CIRCUIT CONSTRUCTION AND STRUCTURAL DATA
# ============================================================

def build_grover_circuit(scenario, expected_indices):
    print("\n===== GROVER CIRCUIT CONSTRUCTION =====", flush=True)

    start_time = time.perf_counter()

    circuit, iterations, solution_count = build_geometric_grover(
        width=scenario.width,
        height=scenario.height,
        start=scenario.start,
        goal=scenario.goal,
        obstacles=scenario.obstacles,
        route_length=scenario.route_length,
    )

    construction_time = time.perf_counter() - start_time

    if solution_count != len(expected_indices):
        raise RuntimeError(
            f"Grover/classical solution mismatch in {scenario.name}: "
            f"{solution_count} != {len(expected_indices)}"
        )

    structure = {
        "iterations": iterations,
        "oracle_calls_per_circuit": iterations,
        "qubits": circuit.num_qubits,
        "initial_depth": circuit.depth(),
        "initial_operations": sum(circuit.count_ops().values()),
        "construction_time_s": construction_time,
        "classically_counted_solutions": solution_count,
    }

    print(f"Iterations: {iterations}", flush=True)
    print(f"Qubits: {circuit.num_qubits}", flush=True)
    print(f"Initial depth: {circuit.depth()}", flush=True)
    print(f"Construction time: {construction_time:.6f} s", flush=True)

    return circuit, structure


# ============================================================
# GROVER SIMULATION
# ============================================================


def save_checkpoint(path, data):
    """Atomically save a JSON checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def benchmark_grover_simulations(scenario, model, circuit, expected_indices):
    case = SIMULATION_CASES[scenario.name]
    shots = case["shots"]
    max_bond = case["max_bond_dimension"]

    simulator_options = dict(SIMULATOR_BASE_OPTIONS)

    if max_bond is not None:
        simulator_options[
            "matrix_product_state_max_bond_dimension"
        ] = max_bond

    simulator = AerSimulator(
        method="matrix_product_state",
        **simulator_options,
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "checkpoints"
        / f"{scenario.name.lower()}_grover_checkpoint.json"
    )

    checkpoint_metadata = {
        "scenario": scenario.name,
        "shots_per_run": shots,
        "max_bond_dimension": max_bond,
        "grover_repetitions": GROVER_REPETITIONS,
        "base_seed": BASE_SEED,
        "transpiler_seed": TRANSPILER_SEED,
        "expected_indices": sorted(expected_indices),
        "simulator_options": simulator_options,
    }

    if checkpoint_path.exists():
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if checkpoint.get("metadata") != checkpoint_metadata:
            raise RuntimeError(
                "Checkpoint configuration mismatch. "
                f"Inspect or archive: {checkpoint_path}"
            )
        print(
            f"\nCheckpoint loaded: {checkpoint_path}",
            flush=True,
        )
    else:
        checkpoint = {
            "metadata": checkpoint_metadata,
            "transpilation": None,
            "runs": [],
        }

    existing_runs = {
        item["seed"]: item for item in checkpoint.get("runs", [])
    }

    print("\n===== GROVER TRANSPILATION =====", flush=True)
    print(f"Shots per run: {shots}", flush=True)
    print(f"Maximum bond dimension: {max_bond}", flush=True)
    print(
        f"Memory limit: {simulator_options['max_memory_mb']} MiB",
        flush=True,
    )

    start_time = time.perf_counter()

    compiled = transpile(
        circuit,
        simulator,
        optimization_level=2,
        seed_transpiler=TRANSPILER_SEED,
    )

    transpilation_time = time.perf_counter() - start_time

    transpiled_depth = compiled.depth()
    transpiled_operations = sum(compiled.count_ops().values())

    print(
        f"Transpiled depth: {transpiled_depth} | "
        f"operations: {transpiled_operations} | "
        f"time: {transpilation_time:.4f} s",
        flush=True,
    )

    checkpoint["transpilation"] = {
        "transpilation_time_s": transpilation_time,
        "transpiled_depth": transpiled_depth,
        "transpiled_operations": transpiled_operations,
    }
    save_checkpoint(checkpoint_path, checkpoint)

    print("\n===== REPEATED GROVER SIMULATIONS =====", flush=True)

    expected_set = set(expected_indices)

    for run in range(GROVER_REPETITIONS):
        seed = BASE_SEED + run

        if seed in existing_runs:
            print(
                f"Run {run + 1}/{GROVER_REPETITIONS}: "
                f"seed={seed} already checkpointed; skipping.",
                flush=True,
            )
            continue

        print(
            f"\nStarting Grover run {run + 1}/{GROVER_REPETITIONS} "
            f"(seed={seed})...",
            flush=True,
        )

        start_time = time.perf_counter()

        result = simulator.run(
            compiled,
            shots=shots,
            seed_simulator=seed,
        ).result()

        simulation_time = time.perf_counter() - start_time
        counts = result.get_counts()

        valid_shots = 0
        detected_indices = set()
        decoded_counts = []

        for bitstring, count in counts.items():
            normalized = bitstring.replace(" ", "")
            index = int(normalized, 2)

            route = RouteEncoding.decode(
                index,
                scenario.route_length,
            )

            valid, route_path, collision_step = model.evaluate_route(route)
            is_valid = valid and collision_step is None

            if is_valid:
                valid_shots += count
                detected_indices.add(index)

            decoded_counts.append({
                "bitstring": normalized,
                "index": index,
                "route": list(route),
                "count": count,
                "valid": is_valid,
                "path": [list(position) for position in route_path],
                "collision_step": collision_step,
            })

        recovered_all = detected_indices == expected_set
        probability = valid_shots / shots

        run_data = {
            "run": run + 1,
            "seed": seed,
            "shots": shots,
            "valid_shots": valid_shots,
            "invalid_shots": shots - valid_shots,
            "success_probability": probability,
            "success_probability_wilson_95_ci": wilson_interval(
                valid_shots, shots
            ),
            "simulation_time_s": simulation_time,
            "all_solutions_recovered": recovered_all,
            "detected_solutions": sorted(detected_indices),
            "decoded_counts": decoded_counts,
        }

        checkpoint["runs"].append(run_data)
        checkpoint["runs"].sort(key=lambda item: item["seed"])
        save_checkpoint(checkpoint_path, checkpoint)
        existing_runs[seed] = run_data

        print(
            f"Run {run + 1}/{GROVER_REPETITIONS}: "
            f"{valid_shots}/{shots} valid "
            f"({100 * probability:.2f}%) | "
            f"{simulation_time:.3f} s | "
            f"all solutions recovered={recovered_all}",
            flush=True,
        )
        print(
            f"Checkpoint saved: {checkpoint_path}",
            flush=True,
        )

    runs = sorted(
        checkpoint["runs"],
        key=lambda item: item["seed"],
    )

    expected_seeds = [
        BASE_SEED + run for run in range(GROVER_REPETITIONS)
    ]
    actual_seeds = [run["seed"] for run in runs]

    if actual_seeds != expected_seeds:
        raise RuntimeError(
            f"Incomplete simulation campaign. "
            f"Expected seeds {expected_seeds}, got {actual_seeds}."
        )

    total_valid_shots = sum(run["valid_shots"] for run in runs)
    total_shots = sum(run["shots"] for run in runs)
    recovered_all_runs = sum(
        int(run["all_solutions_recovered"]) for run in runs
    )

    simulation_times = [
        run["simulation_time_s"] for run in runs
    ]
    per_run_probabilities = [
        run["success_probability"] for run in runs
    ]

    quantum_results = {
        "algorithm": "Geometric Grover",
        "simulator": "Qiskit Aer matrix_product_state",
        "simulation_repetitions": GROVER_REPETITIONS,
        "shots_per_run": shots,
        "total_shots": total_shots,
        "base_seed": BASE_SEED,
        "seeds": actual_seeds,
        "transpiler_seed": TRANSPILER_SEED,
        "transpilation_time_s": transpilation_time,
        "transpiled_depth": transpiled_depth,
        "transpiled_operations": transpiled_operations,
        "simulation_time_s": describe(simulation_times),
        "success_probability_per_run": describe(per_run_probabilities),
        "aggregate_valid_shots": total_valid_shots,
        "aggregate_invalid_shots": total_shots - total_valid_shots,
        "aggregate_success_probability": total_valid_shots / total_shots,
        "aggregate_success_probability_wilson_95_ci": wilson_interval(
            total_valid_shots,
            total_shots,
        ),
        "runs_recovering_all_solutions": recovered_all_runs,
        "all_solutions_recovered_in_every_run": (
            recovered_all_runs == GROVER_REPETITIONS
        ),
        "bond_dimension_cap": max_bond,
        "approximation_warning": max_bond is not None,
        "simulator_options": simulator_options,
        "checkpoint_file": str(checkpoint_path.relative_to(ROOT)),
        "runs": runs,
    }

    return quantum_results


# ============================================================
# SCENARIO EXECUTION
# ============================================================

def run_scenario(scenario):
    print("\n" + "=" * 76, flush=True)
    print(f"MULTI-SCALE STATISTICAL CAMPAIGN: {scenario.name}", flush=True)
    print("=" * 76, flush=True)

    model = create_model(scenario)

    classical_results, valid_indices = benchmark_classical(
        scenario,
        model,
    )

    circuit, grover_structure = build_grover_circuit(
        scenario,
        valid_indices,
    )

    quantum_results = None
    simulation_policy = {
        "performed": False,
        "reason": "Scenario excluded from quantum transpilation and simulation.",
    }

    if scenario.name in SIMULATION_CASES:
        quantum_results = benchmark_grover_simulations(
            scenario,
            model,
            circuit,
            valid_indices,
        )

        simulation_policy = {
            "performed": True,
            "reason": "Scenario included in the controlled MPS simulation campaign.",
        }

    report = {
        "experiment": "Multi-scale statistical classical vs geometric Grover",
        "scenario": {
            "name": scenario.name,
            "grid": [scenario.width, scenario.height],
            "start": list(scenario.start),
            "goal": list(scenario.goal),
            "obstacles": [
                list(position) for position in sorted(scenario.obstacles)
            ],
            "route_length": scenario.route_length,
            "search_space": 4 ** scenario.route_length,
            "valid_routes": len(valid_indices),
            "valid_indices": valid_indices,
            "description": scenario.description,
        },
        "configuration": {
            "classical_repetitions": CLASSICAL_REPETITIONS,
            "grover_repetitions": (
                GROVER_REPETITIONS
                if scenario.name in SIMULATION_CASES
                else 0
            ),
            "confidence_level": CONFIDENCE_LEVEL,
            "base_seed": BASE_SEED,
        },
        "classical": classical_results,
        "grover_structure": grover_structure,
        "quantum_simulation_policy": simulation_policy,
        "quantum": quantum_results,
        "methodological_notes": [
            "A* and Dijkstra solve the shortest-path objective.",
            "Classical exhaustive enumeration and Grover use the same fixed-length route search space.",
            "Classical timing measurements exclude one warm-up per algorithm.",
            "Classical timing confidence intervals use the two-sided Student t distribution.",
            "Quantum success-proportion confidence intervals use the Wilson score method.",
            "Grover circuit construction internally performs classical solution counting to determine the iteration count.",
            "Circuit construction, transpilation and repeated simulation times are reported separately.",
            "Grover simulations use Qiskit Aer CPU matrix-product-state simulation, not quantum hardware.",
            "G2 uses a maximum MPS bond dimension of 128 and only 32 shots per run; results are diagnostic and approximate.",
            "G3 is limited to classical timing and Grover circuit structural/construction analysis; no transpilation or quantum simulation is performed.",
            "The experiment does not establish quantum advantage.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    destination = OUTPUT_DIR / f"{scenario.name.lower()}_statistics.json"
    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n===== SCENARIO REPORT SAVED =====", flush=True)
    print(destination, flush=True)

    return report


# ============================================================
# CONSOLIDATED REPORT
# ============================================================

def build_consolidated_report(reports):
    consolidated = {
        "experiment": "Multi-scale statistical classical vs geometric Grover",
        "scenario_count": len(reports),
        "scenarios": {},
        "methodological_notes": [
            "A* and Dijkstra are shortest-path baselines, not identical-objective comparators to Grover.",
            "Classical exhaustive enumeration is the direct fixed-length route-search comparator.",
            "G1 and G2 include repeated Aer MPS simulations; G3 is not simulated.",
            "No quantum advantage is claimed.",
        ],
    }

    for report in reports:
        scenario_name = report["scenario"]["name"]
        classical = report["classical"]
        quantum = report["quantum"]

        entry = {
            "grid": report["scenario"]["grid"],
            "route_length": report["scenario"]["route_length"],
            "search_space": report["scenario"]["search_space"],
            "valid_routes": report["scenario"]["valid_routes"],
            "astar_mean_s": classical["astar"]["timing_s"]["mean"],
            "astar_ci_95_s": classical["astar"]["timing_s"][
                "mean_confidence_interval"
            ],
            "dijkstra_mean_s": classical["dijkstra"]["timing_s"]["mean"],
            "dijkstra_ci_95_s": classical["dijkstra"]["timing_s"][
                "mean_confidence_interval"
            ],
            "exhaustive_mean_s": classical["exhaustive"]["timing_s"]["mean"],
            "exhaustive_ci_95_s": classical["exhaustive"]["timing_s"][
                "mean_confidence_interval"
            ],
            "grover_iterations": report["grover_structure"]["iterations"],
            "grover_qubits": report["grover_structure"]["qubits"],
            "grover_initial_depth": report["grover_structure"]["initial_depth"],
            "grover_construction_time_s": report["grover_structure"][
                "construction_time_s"
            ],
            "quantum_simulation_performed": quantum is not None,
        }

        if quantum is not None:
            entry.update({
                "grover_transpilation_time_s": quantum["transpilation_time_s"],
                "grover_transpiled_depth": quantum["transpiled_depth"],
                "grover_simulation_mean_s": quantum["simulation_time_s"]["mean"],
                "grover_simulation_ci_95_s": quantum["simulation_time_s"][
                    "mean_confidence_interval"
                ],
                "grover_aggregate_success_probability": quantum[
                    "aggregate_success_probability"
                ],
                "grover_aggregate_success_wilson_95_ci": quantum[
                    "aggregate_success_probability_wilson_95_ci"
                ],
                "grover_total_shots": quantum["total_shots"],
                "grover_runs_recovering_all_solutions": quantum[
                    "runs_recovering_all_solutions"
                ],
                "bond_dimension_cap": quantum["bond_dimension_cap"],
                "approximation_warning": quantum["approximation_warning"],
            })

        consolidated["scenarios"][scenario_name] = entry

    destination = OUTPUT_DIR / "consolidated_multi_scale_statistics.json"
    destination.write_text(
        json.dumps(consolidated, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76, flush=True)
    print("CONSOLIDATED MULTI-SCALE STATISTICAL SUMMARY", flush=True)
    print("=" * 76, flush=True)

    for name, item in consolidated["scenarios"].items():
        print(f"\n{name}", flush=True)
        print(
            f"A*: {item['astar_mean_s']:.8f} s | "
            f"Dijkstra: {item['dijkstra_mean_s']:.8f} s",
            flush=True,
        )
        print(
            f"Exhaustive: {item['exhaustive_mean_s']:.8f} s",
            flush=True,
        )

        if item["quantum_simulation_performed"]:
            print(
                f"Grover success: "
                f"{100 * item['grover_aggregate_success_probability']:.2f}% "
                f"({item['grover_total_shots']} shots)",
                flush=True,
            )
            print(
                f"Grover simulation mean: "
                f"{item['grover_simulation_mean_s']:.4f} s",
                flush=True,
            )
        else:
            print("Grover simulation: NOT PERFORMED", flush=True)

    print(f"\nConsolidated report: {destination}", flush=True)
    return destination


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scale statistical Grover benchmark."
    )
    parser.add_argument(
        "--scenario",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Run only the selected scenario.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("MULTI-SCALE STATISTICAL CLASSICAL VS QUANTUM BENCHMARK")
    print("=" * 76)
    print(f"Classical repetitions per algorithm: {CLASSICAL_REPETITIONS}")
    print(f"Grover simulations per eligible scenario: {GROVER_REPETITIONS}")
    print(f"Base simulator seed: {BASE_SEED}")
    print(f"Confidence level: {100 * CONFIDENCE_LEVEL:.1f}%")
    print(f"Output directory: {OUTPUT_DIR}")

    selected_scenarios = [
        scenario
        for scenario in SCENARIOS
        if args.scenario is None or scenario.name == args.scenario
    ]

    print(
        "Selected scenarios:",
        ", ".join(scenario.name for scenario in selected_scenarios),
        flush=True,
    )

    for scenario in selected_scenarios:
        run_scenario(scenario)

    # Consolidate only when every scenario report is present.
    report_paths = [
        OUTPUT_DIR / f"{scenario.name.lower()}_statistics.json"
        for scenario in SCENARIOS
    ]

    if all(report_path.exists() for report_path in report_paths):
        reports = [
            json.loads(report_path.read_text(encoding="utf-8"))
            for report_path in report_paths
        ]
        build_consolidated_report(reports)
    else:
        missing = [
            path.name for path in report_paths if not path.exists()
        ]
        print(
            "\nConsolidated report postponed; missing scenario reports: "
            + ", ".join(missing),
            flush=True,
        )

    print("\nSELECTED CAMPAIGN COMPLETED.", flush=True)


if __name__ == "__main__":
    main()
