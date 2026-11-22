"""Structural and construction benchmark for geometric Grover circuits."""

import time

from qiskit import transpile
from qiskit_aer import AerSimulator

from planners.quantum.grover_global.geometric_grover import (
    build_geometric_grover,
)
from planners.quantum.grover_global.maze_model import GroverMazeModel
from planners.quantum.grover_global.classical_oracle import GlobalClassicalOracle

from .scenarios import SCENARIOS


# G3 is intentionally excluded from transpilation.
TRANSPILATION_ENABLED = {"G1_3x3_L4", "G2_4x4_L4"}


def count_classical_solutions(scenario):
    model = GroverMazeModel(
        width=scenario.width,
        height=scenario.height,
        start=scenario.start,
        goal=scenario.goal,
        obstacles=scenario.obstacles,
    )
    oracle = GlobalClassicalOracle(model, scenario.route_length)

    start = time.perf_counter()
    valid_indices = oracle.find_valid_indices()
    elapsed = time.perf_counter() - start

    return len(valid_indices), elapsed


def run_grover_benchmark():
    print("=" * 72)
    print("MULTI-SCALE GEOMETRIC GROVER: CIRCUIT RESOURCE ANALYSIS")
    print("=" * 72)
    simulator = AerSimulator(method="matrix_product_state")

    for scenario in SCENARIOS:
        print(f"\n===== {scenario.name} =====")

        solutions, counting_time = count_classical_solutions(scenario)

        print(f"Search space: {4 ** scenario.route_length}")
        print(f"Classically counted solutions: {solutions}")
        print(f"Classical counting time: {counting_time:.6f} s")

        start = time.perf_counter()
        circuit, iterations, circuit_solutions = build_geometric_grover(
            width=scenario.width,
            height=scenario.height,
            start=scenario.start,
            goal=scenario.goal,
            obstacles=scenario.obstacles,
            route_length=scenario.route_length,
        )
        build_time = time.perf_counter() - start

        if circuit_solutions != solutions:
            raise RuntimeError(
                f"Solution count mismatch in {scenario.name}: "
                f"{circuit_solutions} != {solutions}"
            )

        print(f"Grover iterations: {iterations}")
        print(f"Circuit construction time: {build_time:.6f} s")
        print(f"Qubits: {circuit.num_qubits}")
        print(f"Initial depth: {circuit.depth()}")
        print(f"Initial operation count: {sum(circuit.count_ops().values())}")

        if scenario.name not in TRANSPILATION_ENABLED:
            print("Transpilation: SKIPPED (resource preflight policy)")
            continue

        start = time.perf_counter()
        compiled = transpile(
            circuit,
            simulator,
            optimization_level=2,
            seed_transpiler=42,
        )
        transpilation_time = time.perf_counter() - start

        print(f"Transpiled depth: {compiled.depth()}")
        print(f"Transpiled operation count: {sum(compiled.count_ops().values())}")
        print(f"Transpilation time: {transpilation_time:.6f} s")
        print("Quantum simulation: NOT RUN")


if __name__ == "__main__":
    run_grover_benchmark()
