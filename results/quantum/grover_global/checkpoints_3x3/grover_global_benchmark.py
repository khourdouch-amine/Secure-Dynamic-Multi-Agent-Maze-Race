"""Reproducible benchmark: geometric Grover vs classical exhaustive search."""

import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit import transpile
from qiskit_aer import AerSimulator

from planners.quantum.grover_global.geometric_grover import (
    build_geometric_grover,
)
from planners.quantum.grover_global.maze_model import GroverMazeModel
from planners.quantum.grover_global.classical_oracle import GlobalClassicalOracle
from planners.quantum.route_encoding import RouteEncoding


WIDTH = 3
HEIGHT = 3
START = (0, 0)
GOAL = (2, 2)
OBSTACLES = {(1, 0)}
ROUTE_LENGTH = 4
SHOTS = 1024


def main():
    model = GroverMazeModel(WIDTH, HEIGHT, START, GOAL, OBSTACLES)

    # Classical exhaustive reference.
    classical = GlobalClassicalOracle(model, ROUTE_LENGTH)
    t0 = time.perf_counter()
    valid_indices = classical.find_valid_indices()
    classical_time = time.perf_counter() - t0

    # Geometric Grover circuit.
    t0 = time.perf_counter()
    circuit, iterations, solution_count = build_geometric_grover(
        WIDTH, HEIGHT, START, GOAL, OBSTACLES, ROUTE_LENGTH
    )
    construction_time = time.perf_counter() - t0

    simulator = AerSimulator(method="matrix_product_state")

    t0 = time.perf_counter()
    compiled = transpile(circuit, simulator, optimization_level=2)
    transpilation_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = simulator.run(compiled, shots=SHOTS).result()
    simulation_time = time.perf_counter() - t0
    counts = result.get_counts()

    valid_shots = 0
    detected = set()
    decoded = []

    for bitstring, count in counts.items():
        index = int(bitstring.replace(" ", ""), 2)
        route = RouteEncoding.decode(index, ROUTE_LENGTH)
        valid, path, _ = model.evaluate_route(route)

        if valid:
            valid_shots += count
            detected.add(index)

        decoded.append({
            "index": index,
            "route": list(route),
            "valid": valid,
            "count": count,
            "path": [list(p) for p in path],
        })

    data = {
        "experiment": "Geometric Global Grover",
        "grid": [WIDTH, HEIGHT],
        "start": list(START),
        "goal": list(GOAL),
        "obstacles": [list(p) for p in sorted(OBSTACLES)],
        "route_length": ROUTE_LENGTH,
        "search_space": 4 ** ROUTE_LENGTH,
        "valid_routes": solution_count,
        "grover_iterations": iterations,
        "shots": SHOTS,
        "valid_shots": valid_shots,
        "success_probability": valid_shots / SHOTS,
        "all_solutions_recovered": detected == set(valid_indices),
        "classical_exhaustive_time_s": classical_time,
        "circuit_construction_time_s": construction_time,
        "transpilation_time_s": transpilation_time,
        "quantum_simulation_time_s": simulation_time,
        "qubits": circuit.num_qubits,
        "circuit_depth": circuit.depth(),
        "results": decoded,
        "note": (
            "Simulation benchmark only. Grover iteration count currently "
            "uses classical solution counting. No quantum advantage claimed."
        ),
    }

    output = Path("results/quantum/grover_global")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "benchmark_3x3.json"
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("===== GEOMETRIC GROVER BENCHMARK =====")
    print("Valid routes:", solution_count)
    print("Grover iterations:", iterations)
    print(f"Success probability: {100 * valid_shots / SHOTS:.2f}%")
    print("All solutions recovered:", detected == set(valid_indices))
    print(f"Classical search: {classical_time:.6f} s")
    print(f"Circuit construction: {construction_time:.6f} s")
    print(f"Transpilation: {transpilation_time:.6f} s")
    print(f"Quantum simulation: {simulation_time:.6f} s")
    print("Qubits:", circuit.num_qubits)
    print("Depth:", circuit.depth())
    print("Saved:", destination)


if __name__ == "__main__":
    main()
