"""Reproducible classical vs geometric Grover benchmark."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qiskit import transpile
from qiskit_aer import AerSimulator

from environment.maze import Maze, STATIC_WALL
from planners.classical.astar import AStarPlanner
from planners.classical.dijkstra import DijkstraPlanner

from planners.quantum.grover_global.maze_model import GroverMazeModel
from planners.quantum.grover_global.classical_oracle import GlobalClassicalOracle
from planners.quantum.grover_global.geometric_grover import (
    build_geometric_grover,
)
from planners.quantum.route_encoding import RouteEncoding


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

WIDTH = 5
HEIGHT = 5

START = (1, 1)
GOAL = (3, 3)

OBSTACLES = {(2, 1)}

ROUTE_LENGTH = 4
SHOTS = 1024
SEED = 42

# Translation from the classical 5x5 maze to the Grover 3x3
OFFSET = (1, 1)

GROVER_WIDTH = 3
GROVER_HEIGHT = 3

GROVER_START = (0, 0)
GROVER_GOAL = (2, 2)

GROVER_OBSTACLES = {(1, 0)}

OUTPUT = (
    ROOT
    / "results"
    / "benchmarks"
    / "classical_vs_quantum"
    / "benchmark_classical_vs_quantum.json"
)


# ============================================================
# COMMON VALIDATION
# ============================================================

DIRECTION_FROM_DELTA = {
    (1, 0): 0,
    (0, -1): 1,
    (-1, 0): 2,
    (0, 1): 3,
}


def translate_path_to_grover(path):
    """Translate classical maze coordinates to Grover coordinates."""
    ox, oy = OFFSET
    return [(x - ox, y - oy) for x, y in path]


def path_to_sequence(path):
    """Convert a four-connected path into movement directions."""
    sequence = []

    for a, b in zip(path, path[1:]):
        delta = (b[0] - a[0], b[1] - a[1])

        if delta not in DIRECTION_FROM_DELTA:
            raise ValueError(f"Non-cardinal movement detected: {delta}")

        sequence.append(DIRECTION_FROM_DELTA[delta])

    return tuple(sequence)


def validate_classical_path(path, quantum_model):
    """Validate a classical path against the Grover maze model."""
    if path is None:
        return {
            "valid": False,
            "translated_path": None,
            "sequence": None,
        }

    translated = translate_path_to_grover(path)
    sequence = path_to_sequence(translated)

    valid, evaluated_path, collision_step = (
        quantum_model.evaluate_route(sequence)
    )

    valid = (
        valid
        and collision_step is None
        and evaluated_path == translated
        and translated[0] == GROVER_START
        and translated[-1] == GROVER_GOAL
    )

    return {
        "valid": valid,
        "translated_path": translated,
        "sequence": list(sequence),
        "steps": len(sequence),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("===== CLASSICAL VS QUANTUM BENCHMARK =====")

    # --------------------------------------------------------
    # 1. Build the classical maze
    # --------------------------------------------------------

    maze = Maze(
        width=WIDTH,
        height=HEIGHT,
        seed=SEED,
    )

    maze.set_cell(2, 1, STATIC_WALL)

    print("\n===== COMMON MAZE =====")
    print(maze.display_ascii())

    # --------------------------------------------------------
    # 2. Build the translated Grover model
    # --------------------------------------------------------

    model = GroverMazeModel(
        width=GROVER_WIDTH,
        height=GROVER_HEIGHT,
        start=GROVER_START,
        goal=GROVER_GOAL,
        obstacles=GROVER_OBSTACLES,
    )

    # --------------------------------------------------------
    # 3. A* benchmark
    # --------------------------------------------------------

    print("\n===== A* =====")

    astar = AStarPlanner()

    t0 = time.perf_counter()
    astar_path = astar.plan(maze, START, GOAL)
    astar_time = time.perf_counter() - t0

    astar_validation = validate_classical_path(astar_path, model)

    print("Path:", astar_path)
    print("Steps:", len(astar_path) - 1 if astar_path else None)
    print("Valid:", astar_validation["valid"])
    print(f"Time: {astar_time:.8f} s")

    # --------------------------------------------------------
    # 4. Dijkstra benchmark
    # --------------------------------------------------------

    print("\n===== DIJKSTRA =====")

    dijkstra = DijkstraPlanner()

    t0 = time.perf_counter()
    dijkstra_path = dijkstra.plan(maze, START, GOAL)
    dijkstra_time = time.perf_counter() - t0

    dijkstra_validation = validate_classical_path(
        dijkstra_path, model
    )

    print("Path:", dijkstra_path)
    print("Steps:", len(dijkstra_path) - 1 if dijkstra_path else None)
    print("Valid:", dijkstra_validation["valid"])
    print(f"Time: {dijkstra_time:.8f} s")

    # --------------------------------------------------------
    # 5. Classical exhaustive reference
    # --------------------------------------------------------

    print("\n===== CLASSICAL EXHAUSTIVE SEARCH =====")

    classical_oracle = GlobalClassicalOracle(
        model,
        ROUTE_LENGTH,
    )

    t0 = time.perf_counter()
    valid_indices = classical_oracle.find_valid_indices()
    exhaustive_time = time.perf_counter() - t0

    print("Search space:", classical_oracle.search_space)
    print("Valid routes:", len(valid_indices))
    print(f"Enumeration time: {exhaustive_time:.8f} s")

    # --------------------------------------------------------
    # 6. Geometric Grover circuit construction
    # --------------------------------------------------------

    print("\n===== GEOMETRIC GROVER =====")

    t0 = time.perf_counter()

    circuit, iterations, solution_count = build_geometric_grover(
        GROVER_WIDTH,
        GROVER_HEIGHT,
        GROVER_START,
        GROVER_GOAL,
        GROVER_OBSTACLES,
        ROUTE_LENGTH,
    )

    construction_time = time.perf_counter() - t0

    if solution_count != len(valid_indices):
        raise RuntimeError(
            "Classical and Grover solution counts do not match."
        )

    print("Valid routes:", solution_count)
    print("Grover iterations:", iterations)
    print("Qubits:", circuit.num_qubits)
    print("Initial circuit depth:", circuit.depth())
    print(f"Construction time: {construction_time:.6f} s")

    # --------------------------------------------------------
    # 7. Transpilation
    # --------------------------------------------------------

    simulator = AerSimulator(
        method="matrix_product_state",
    )

    print("\nTranspilation...")

    t0 = time.perf_counter()

    compiled = transpile(
        circuit,
        simulator,
        optimization_level=2,
    )

    transpilation_time = time.perf_counter() - t0

    print("Transpiled depth:", compiled.depth())
    print("Transpiled operations:", compiled.size())
    print(f"Transpilation time: {transpilation_time:.6f} s")

    # --------------------------------------------------------
    # 8. Quantum simulation
    # --------------------------------------------------------

    print("\nSimulation:", SHOTS, "shots")

    t0 = time.perf_counter()

    result = simulator.run(
        compiled,
        shots=SHOTS,
        seed_simulator=SEED,
    ).result()

    simulation_time = time.perf_counter() - t0

    counts = result.get_counts()

    # --------------------------------------------------------
    # 9. Independent measurement validation
    # --------------------------------------------------------

    valid_shots = 0
    detected_indices = set()
    decoded_routes = []

    for bitstring, count in counts.items():

        sequence = RouteEncoding.decode_bitstring(
            bitstring,
            ROUTE_LENGTH,
        )

        valid, path, collision_step = model.evaluate_route(sequence)

        index = RouteEncoding.encode(sequence)

        if valid and collision_step is None:
            valid_shots += count
            detected_indices.add(index)

        decoded_routes.append({
            "bitstring": bitstring,
            "index": index,
            "sequence": list(sequence),
            "valid": valid,
            "path": [list(p) for p in path],
            "collision_step": collision_step,
            "count": count,
        })

    success_probability = valid_shots / SHOTS

    all_solutions_recovered = (
        detected_indices == set(valid_indices)
    )

    print("\n===== QUANTUM RESULTS =====")
    print("Counts:", counts)
    print("Valid shots:", valid_shots, "/", SHOTS)
    print(f"Success probability: {100 * success_probability:.2f}%")
    print("All solutions recovered:", all_solutions_recovered)
    print(f"Simulation time: {simulation_time:.6f} s")

    # --------------------------------------------------------
    # 10. Consolidated report
    # --------------------------------------------------------

    data = {
        "experiment": "Classical vs Geometric Grover",
        "seed": SEED,
        "shots": SHOTS,

        "classical_maze": {
            "width": WIDTH,
            "height": HEIGHT,
            "start": list(START),
            "goal": list(GOAL),
            "obstacles": [list(p) for p in sorted(OBSTACLES)],
        },

        "grover_model": {
            "width": GROVER_WIDTH,
            "height": GROVER_HEIGHT,
            "start": list(GROVER_START),
            "goal": list(GROVER_GOAL),
            "obstacles": [
                list(p) for p in sorted(GROVER_OBSTACLES)
            ],
            "coordinate_translation": list(OFFSET),
            "route_length": ROUTE_LENGTH,
            "search_space": classical_oracle.search_space,
            "valid_routes": solution_count,
        },

        "classical": {
            "astar": {
                "time_s": astar_time,
                "path": astar_path,
                "validation": astar_validation,
            },
            "dijkstra": {
                "time_s": dijkstra_time,
                "path": dijkstra_path,
                "validation": dijkstra_validation,
            },
            "exhaustive": {
                "time_s": exhaustive_time,
                "evaluated_routes": classical_oracle.search_space,
                "valid_routes": len(valid_indices),
                "valid_indices": valid_indices,
            },
        },

        "quantum": {
            "algorithm": "Geometric Grover",
            "iterations": iterations,
            "oracle_calls": iterations,
            "qubits": circuit.num_qubits,
            "initial_depth": circuit.depth(),
            "transpiled_depth": compiled.depth(),
            "transpiled_operations": compiled.size(),
            "construction_time_s": construction_time,
            "transpilation_time_s": transpilation_time,
            "simulation_time_s": simulation_time,
            "valid_shots": valid_shots,
            "success_probability": success_probability,
            "all_solutions_recovered": all_solutions_recovered,
            "counts": counts,
            "decoded_routes": decoded_routes,
        },

        "methodological_notes": [
            "The classical maze and Grover model represent the same topology through a coordinate translation.",
            "A* and Dijkstra return shortest paths; Grover searches fixed-length movement sequences.",
            "The classical exhaustive oracle evaluates every encoded route.",
            "Grover construction internally performs classical solution counting to determine its iteration count.",
            "Construction time therefore includes classical preprocessing.",
            "Quantum results are obtained from Qiskit Aer simulation, not quantum hardware.",
            "No quantum advantage is claimed.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    print("\n===== FINAL REPORT =====")
    print("A* valid:", astar_validation["valid"])
    print("Dijkstra valid:", dijkstra_validation["valid"])
    print("Classical valid routes:", len(valid_indices))
    print("Grover valid shots:", valid_shots, "/", SHOTS)
    print("All solutions recovered:", all_solutions_recovered)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
