"""Build a consolidated report from the multi-scale benchmark."""

import json
import time
from pathlib import Path

from qiskit import transpile
from qiskit_aer import AerSimulator

from planners.classical.astar import AStarPlanner
from planners.classical.dijkstra import DijkstraPlanner
from planners.quantum.grover_global.geometric_grover import (
    build_geometric_grover,
)
from planners.quantum.grover_global.maze_model import GroverMazeModel
from planners.quantum.grover_global.classical_oracle import (
    GlobalClassicalOracle,
)

from .scenarios import SCENARIOS, validate_scenario
from .benchmark_classical import GridMazeAdapter, validate_path


OUTPUT = Path("results/benchmarks/multi_scale")
OUTPUT.mkdir(parents=True, exist_ok=True)

SIMULATOR = AerSimulator(method="matrix_product_state")
TRANSPILATION_ENABLED = {"G1_3x3_L4", "G2_4x4_L4"}


def collect_scenario(scenario):
    print(f"\n===== {scenario.name} =====", flush=True)

    feasibility = validate_scenario(scenario)

    maze = GridMazeAdapter(
        scenario.width,
        scenario.height,
        scenario.obstacles,
    )

    classical_results = {}

    for planner in (AStarPlanner(), DijkstraPlanner()):
        start = time.perf_counter()
        path = planner.plan(maze, scenario.start, scenario.goal)
        elapsed = time.perf_counter() - start

        valid = validate_path(
            path, maze, scenario.start, scenario.goal
        )

        if not valid:
            raise RuntimeError(
                f"Invalid path from {planner.name} in {scenario.name}"
            )

        classical_results[planner.name] = {
            "path": [list(p) for p in path],
            "path_length_moves": len(path) - 1,
            "valid": valid,
            "single_run_time_s": elapsed,
            "timing_note": "Single measurement; not a statistical benchmark.",
        }

        print(
            f"{planner.name}: {len(path) - 1} moves, "
            f"{elapsed:.8f} s, valid={valid}",
            flush=True,
        )

    model = GroverMazeModel(
        scenario.width,
        scenario.height,
        scenario.start,
        scenario.goal,
        scenario.obstacles,
    )
    oracle = GlobalClassicalOracle(model, scenario.route_length)

    start = time.perf_counter()
    valid_indices = oracle.find_valid_indices()
    counting_time = time.perf_counter() - start

    start = time.perf_counter()
    circuit, iterations, solutions = build_geometric_grover(
        scenario.width,
        scenario.height,
        scenario.start,
        scenario.goal,
        scenario.obstacles,
        scenario.route_length,
    )
    construction_time = time.perf_counter() - start

    if solutions != len(valid_indices):
        raise RuntimeError("Inconsistent classical solution counts.")

    grover = {
        "search_space": 4 ** scenario.route_length,
        "valid_routes": solutions,
        "valid_route_indices": valid_indices,
        "grover_iterations": iterations,
        "classical_solution_counting_time_s": counting_time,
        "circuit_construction_time_s": construction_time,
        "qubits": circuit.num_qubits,
        "initial_depth": circuit.depth(),
        "initial_operations": sum(circuit.count_ops().values()),
        "transpilation": {
            "performed": False,
            "backend": "AerSimulator MPS",
        },
        "simulation": {
            "performed": False,
        },
    }

    if scenario.name in TRANSPILATION_ENABLED:
        start = time.perf_counter()
        compiled = transpile(
            circuit,
            SIMULATOR,
            optimization_level=2,
            seed_transpiler=42,
        )
        transpilation_time = time.perf_counter() - start

        grover["transpilation"] = {
            "performed": True,
            "backend": "AerSimulator MPS",
            "optimization_level": 2,
            "seed_transpiler": 42,
            "time_s": transpilation_time,
            "depth": compiled.depth(),
            "operations": sum(compiled.count_ops().values()),
        }

    simulation_file = (
        OUTPUT / f"{scenario.name.lower()}_simulation.json"
    )

    if simulation_file.exists():
        simulation = json.loads(
            simulation_file.read_text(encoding="utf-8")
        )
    else:
        simulation = {
            "performed": False,
            "reason": (
                "Not executed in this campaign; "
                "excluded by resource preflight policy."
            ),
        }

    print(
        f"Grover: {grover['qubits']} qubits, "
        f"{grover['initial_depth']} initial depth, "
        f"{iterations} iterations",
        flush=True,
    )

    return {
        "scenario": {
            **feasibility,
            "grid": [scenario.width, scenario.height],
            "start": list(scenario.start),
            "goal": list(scenario.goal),
            "obstacles": [
                list(p) for p in sorted(scenario.obstacles)
            ],
            "description": scenario.description,
        },
        "classical_shortest_path": classical_results,
        "grover": grover,
        "quantum_simulation": simulation,
    }


def main():
    report = {
        "title": "Secure Dynamic Multi-Agent Maze Race - Multi-Scale Benchmark",
        "date": "2026-11-19",
        "methodology": {
            "classical_planners": ["A*", "Dijkstra"],
            "quantum_algorithm": "Geometric Grover search",
            "simulator": "Qiskit Aer, matrix_product_state, CPU",
            "transpilation": (
                "Optimization level 2, Aer MPS target, "
                "transpiler seed 42"
            ),
            "important_caveats": [
                "A* and Dijkstra solve shortest-path planning.",
                "Grover searches fixed-length encoded route sequences.",
                "These objectives are not identical.",
                "Grover iteration counts use classical solution counting.",
                "No quantum advantage is claimed.",
                "G2 uses an MPS bond-dimension cap of 128.",
                "G2 has only 32 shots and is diagnostic.",
                "G3 was not transpiled or simulated.",
                "Classical timings in this report are single measurements.",
            ],
        },
        "scenarios": [],
    }

    for scenario in SCENARIOS:
        report["scenarios"].append(collect_scenario(scenario))

    destination = OUTPUT / "consolidated_multi_scale_report.json"
    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n===== CONSOLIDATED REPORT =====")
    print(f"Scenarios: {len(report['scenarios'])}")
    print(f"Report: {destination}")
    print("Existing simulation JSON files were not modified.")


if __name__ == "__main__":
    main()
