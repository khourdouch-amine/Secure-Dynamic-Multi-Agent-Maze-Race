"""Controlled Aer MPS simulations for multi-scale Grover scenarios."""

import json
import time
from pathlib import Path

from qiskit import transpile
from qiskit_aer import AerSimulator

from planners.quantum.grover_global.geometric_grover import (
    build_geometric_grover,
)
from planners.quantum.grover_global.maze_model import GroverMazeModel
from planners.quantum.grover_global.classical_oracle import GlobalClassicalOracle
from planners.quantum.route_encoding import RouteEncoding

from .scenarios import SCENARIOS


CONFIG = {
    "max_memory_mb": 7000,
    "seed_simulator": 42,
    "mps_omp_threads": 1,
    "matrix_product_state_truncation_threshold": 1e-16,
}

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


def run_scenario(scenario):
    case = SIMULATION_CASES[scenario.name]
    shots = case["shots"]
    max_bond = case["max_bond_dimension"]

    print(f"\n===== SIMULATION {scenario.name} =====", flush=True)
    print(f"Shots: {shots}", flush=True)
    print(f"Maximum bond dimension: {max_bond}", flush=True)
    print(f"Memory limit: {CONFIG['max_memory_mb']} MiB", flush=True)

    model = GroverMazeModel(
        scenario.width,
        scenario.height,
        scenario.start,
        scenario.goal,
        scenario.obstacles,
    )

    classical = GlobalClassicalOracle(model, scenario.route_length)
    valid_indices = set(classical.find_valid_indices())

    circuit, iterations, solutions = build_geometric_grover(
        scenario.width,
        scenario.height,
        scenario.start,
        scenario.goal,
        scenario.obstacles,
        scenario.route_length,
    )

    if solutions != len(valid_indices):
        raise RuntimeError("Classical and circuit solution counts differ.")

    simulator_options = dict(CONFIG)
    if max_bond is not None:
        simulator_options["matrix_product_state_max_bond_dimension"] = max_bond

    simulator = AerSimulator(
        method="matrix_product_state",
        **simulator_options,
    )

    print("Transpiling...", flush=True)
    t0 = time.perf_counter()
    compiled = transpile(
        circuit,
        simulator,
        optimization_level=2,
        seed_transpiler=42,
    )
    transpilation_time = time.perf_counter() - t0

    print(
        f"Transpilation complete: depth={compiled.depth()}, "
        f"operations={sum(compiled.count_ops().values())}, "
        f"time={transpilation_time:.3f}s",
        flush=True,
    )

    print("Starting Aer simulation...", flush=True)
    t0 = time.perf_counter()
    result = simulator.run(compiled, shots=shots).result()
    simulation_time = time.perf_counter() - t0
    counts = result.get_counts()

    valid_shots = 0
    detected = set()
    decoded = []

    for bitstring, count in counts.items():
        index = int(bitstring.replace(" ", ""), 2)
        route = RouteEncoding.decode(index, scenario.route_length)
        valid, path, collision_step = model.evaluate_route(route)

        if valid:
            valid_shots += count
            detected.add(index)

        decoded.append({
            "bitstring": bitstring,
            "index": index,
            "route": list(route),
            "valid": valid,
            "count": count,
            "path": [list(position) for position in path],
            "collision_step": collision_step,
        })

    all_solutions_recovered = detected == valid_indices

    data = {
        "scenario": scenario.name,
        "grid": [scenario.width, scenario.height],
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "obstacles": [list(p) for p in sorted(scenario.obstacles)],
        "route_length": scenario.route_length,
        "search_space": 4 ** scenario.route_length,
        "valid_routes": solutions,
        "grover_iterations": iterations,
        "shots": shots,
        "valid_shots": valid_shots,
        "success_probability": valid_shots / shots,
        "all_solutions_recovered": all_solutions_recovered,
        "detected_solutions": sorted(detected),
        "classical_valid_indices": sorted(valid_indices),
        "transpilation_time_s": transpilation_time,
        "simulation_time_s": simulation_time,
        "qubits": circuit.num_qubits,
        "initial_depth": circuit.depth(),
        "transpiled_depth": compiled.depth(),
        "transpiled_operations": sum(compiled.count_ops().values()),
        "simulator": "matrix_product_state",
        "simulator_options": simulator_options,
        "bond_dimension_cap": max_bond,
        "approximation_warning": max_bond is not None,
        "results": decoded,
        "note": (
            "CPU simulation only. Grover iteration count uses classical "
            "solution counting. No quantum advantage claimed. "
            "MPS numerical truncation may affect results."
        ),
    }

    output = Path("results/benchmarks/multi_scale")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{scenario.name.lower()}_simulation.json"
    destination.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    print("\n===== SIMULATION RESULT =====")
    print(f"Valid shots: {valid_shots}/{shots}")
    print(f"Observed success rate: {100 * valid_shots / shots:.2f}%")
    print(f"All valid routes recovered: {all_solutions_recovered}")
    print(f"Simulation time: {simulation_time:.3f}s")
    print(f"Report: {destination}", flush=True)

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=tuple(SIMULATION_CASES),
        required=True,
    )
    args = parser.parse_args()

    scenario = next(s for s in SCENARIOS if s.name == args.scenario)
    run_scenario(scenario)
