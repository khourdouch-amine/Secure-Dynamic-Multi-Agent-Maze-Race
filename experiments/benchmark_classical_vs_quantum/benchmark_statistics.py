"""Statistical benchmark: classical planners vs geometric Grover."""

import json
import statistics
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
from planners.quantum.grover_global.geometric_grover import build_geometric_grover
from planners.quantum.route_encoding import RouteEncoding


# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 5
HEIGHT = 5

START = (1, 1)
GOAL = (3, 3)
OBSTACLES = {(2, 1)}

ROUTE_LENGTH = 4
OFFSET = (1, 1)

GROVER_WIDTH = 3
GROVER_HEIGHT = 3
GROVER_START = (0, 0)
GROVER_GOAL = (2, 2)
GROVER_OBSTACLES = {(1, 0)}

CLASSICAL_REPETITIONS = 30
GROVER_REPETITIONS = 5
SHOTS = 1024
BASE_SEED = 42

OUTPUT = (
    ROOT / "results" / "benchmarks"
    / "classical_vs_quantum"
    / "benchmark_classical_vs_quantum_statistics.json"
)

DIRECTION_FROM_DELTA = {
    (1, 0): 0,
    (0, -1): 1,
    (-1, 0): 2,
    (0, 1): 3,
}


# ============================================================
# STATISTICS
# ============================================================

def describe(values):
    """Return descriptive statistics for a list of measurements."""
    values = [float(v) for v in values]

    return {
        "samples": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


# ============================================================
# VALIDATION
# ============================================================

def translate_path(path):
    ox, oy = OFFSET
    return [(x - ox, y - oy) for x, y in path]


def validate_path(path, model):
    if path is None:
        return False

    translated = translate_path(path)

    if not translated:
        return False

    sequence = []

    for a, b in zip(translated, translated[1:]):
        delta = (b[0] - a[0], b[1] - a[1])

        if delta not in DIRECTION_FROM_DELTA:
            return False

        sequence.append(DIRECTION_FROM_DELTA[delta])

    valid, evaluated_path, collision_step = model.evaluate_route(
        tuple(sequence)
    )

    return (
        valid
        and collision_step is None
        and evaluated_path == translated
        and translated[0] == GROVER_START
        and translated[-1] == GROVER_GOAL
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STATISTICAL CLASSICAL VS QUANTUM BENCHMARK")
    print("=" * 60)

    print("\nConfiguration:")
    print("Classical repetitions:", CLASSICAL_REPETITIONS)
    print("Grover repetitions:", GROVER_REPETITIONS)
    print("Shots per Grover simulation:", SHOTS)

    # --------------------------------------------------------
    # 1. Common environment
    # --------------------------------------------------------

    maze = Maze(width=WIDTH, height=HEIGHT, seed=BASE_SEED)
    maze.set_cell(2, 1, STATIC_WALL)

    model = GroverMazeModel(
        width=GROVER_WIDTH,
        height=GROVER_HEIGHT,
        start=GROVER_START,
        goal=GROVER_GOAL,
        obstacles=GROVER_OBSTACLES,
    )

    print("\n===== COMMON MAZE =====")
    print(maze.display_ascii())

    astar = AStarPlanner()
    dijkstra = DijkstraPlanner()

    # --------------------------------------------------------
    # 2. Warm-up (excluded from measurements)
    # --------------------------------------------------------

    print("\n===== WARM-UP =====")

    astar.plan(maze, START, GOAL)
    dijkstra.plan(maze, START, GOAL)

    classical_oracle = GlobalClassicalOracle(model, ROUTE_LENGTH)
    classical_oracle.find_valid_indices()

    print("Warm-up completed.")

    # --------------------------------------------------------
    # 3. A* repeated measurements
    # --------------------------------------------------------

    print("\n===== A* : 30 REPETITIONS =====")

    astar_times = []
    astar_valid = []

    for i in range(CLASSICAL_REPETITIONS):
        t0 = time.perf_counter()
        path = astar.plan(maze, START, GOAL)
        elapsed = time.perf_counter() - t0

        valid = validate_path(path, model)

        astar_times.append(elapsed)
        astar_valid.append(valid)

        print(
            f"Run {i + 1:02d}/{CLASSICAL_REPETITIONS}: "
            f"{elapsed:.8f} s | valid={valid}"
        )

    if not all(astar_valid):
        raise RuntimeError("A* produced at least one invalid path.")

    # --------------------------------------------------------
    # 4. Dijkstra repeated measurements
    # --------------------------------------------------------

    print("\n===== DIJKSTRA : 30 REPETITIONS =====")

    dijkstra_times = []
    dijkstra_valid = []

    for i in range(CLASSICAL_REPETITIONS):
        t0 = time.perf_counter()
        path = dijkstra.plan(maze, START, GOAL)
        elapsed = time.perf_counter() - t0

        valid = validate_path(path, model)

        dijkstra_times.append(elapsed)
        dijkstra_valid.append(valid)

        print(
            f"Run {i + 1:02d}/{CLASSICAL_REPETITIONS}: "
            f"{elapsed:.8f} s | valid={valid}"
        )

    if not all(dijkstra_valid):
        raise RuntimeError("Dijkstra produced at least one invalid path.")

    # --------------------------------------------------------
    # 5. Classical exhaustive enumeration
    # --------------------------------------------------------

    print("\n===== CLASSICAL EXHAUSTIVE : 30 REPETITIONS =====")

    exhaustive_times = []
    valid_indices = None

    for i in range(CLASSICAL_REPETITIONS):
        t0 = time.perf_counter()
        indices = classical_oracle.find_valid_indices()
        elapsed = time.perf_counter() - t0

        if valid_indices is None:
            valid_indices = indices
        elif indices != valid_indices:
            raise RuntimeError("Exhaustive results are inconsistent.")

        exhaustive_times.append(elapsed)

        print(
            f"Run {i + 1:02d}/{CLASSICAL_REPETITIONS}: "
            f"{elapsed:.8f} s | solutions={len(indices)}"
        )

    # --------------------------------------------------------
    # 6. Grover circuit construction
    # --------------------------------------------------------

    print("\n===== GROVER CIRCUIT CONSTRUCTION =====")

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
        raise RuntimeError("Classical and Grover solution counts differ.")

    print("Solutions:", solution_count)
    print("Iterations:", iterations)
    print("Qubits:", circuit.num_qubits)
    print("Initial depth:", circuit.depth())
    print(f"Construction time: {construction_time:.6f} s")

    # --------------------------------------------------------
    # 7. Transpilation
    # --------------------------------------------------------

    print("\n===== TRANSPILATION =====")

    simulator = AerSimulator(method="matrix_product_state")

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
    # 8. Repeated Grover simulations
    # --------------------------------------------------------

    print("\n===== GROVER : 5 SIMULATIONS =====")

    grover_runs = []
    total_valid_shots = 0
    total_shots = 0
    all_recovered = []

    expected_indices = set(valid_indices)

    for i in range(GROVER_REPETITIONS):

        seed = BASE_SEED + i

        t0 = time.perf_counter()

        result = simulator.run(
            compiled,
            shots=SHOTS,
            seed_simulator=seed,
        ).result()

        elapsed = time.perf_counter() - t0
        counts = result.get_counts()

        valid_shots = 0
        detected_indices = set()

        for bitstring, count in counts.items():

            sequence = RouteEncoding.decode_bitstring(
                bitstring,
                ROUTE_LENGTH,
            )

            valid, path, collision_step = model.evaluate_route(sequence)

            if valid and collision_step is None:
                valid_shots += count
                detected_indices.add(RouteEncoding.encode(sequence))

        probability = valid_shots / SHOTS
        recovered = detected_indices == expected_indices

        total_valid_shots += valid_shots
        total_shots += SHOTS
        all_recovered.append(recovered)

        grover_runs.append({
            "run": i + 1,
            "seed": seed,
            "shots": SHOTS,
            "valid_shots": valid_shots,
            "success_probability": probability,
            "simulation_time_s": elapsed,
            "all_solutions_recovered": recovered,
            "counts": counts,
        })

        print(
            f"Run {i + 1}/{GROVER_REPETITIONS}: "
            f"{valid_shots}/{SHOTS} valid "
            f"({100 * probability:.2f}%) | "
            f"{elapsed:.4f} s | "
            f"all solutions={recovered}"
        )

    # --------------------------------------------------------
    # 9. Consolidated statistics
    # --------------------------------------------------------

    grover_times = [
        run["simulation_time_s"] for run in grover_runs
    ]

    grover_probabilities = [
        run["success_probability"] for run in grover_runs
    ]

    aggregate_probability = total_valid_shots / total_shots

    data = {
        "experiment": "Statistical Classical vs Geometric Grover",
        "configuration": {
            "classical_repetitions": CLASSICAL_REPETITIONS,
            "grover_repetitions": GROVER_REPETITIONS,
            "shots_per_grover_run": SHOTS,
            "base_seed": BASE_SEED,
            "route_length": ROUTE_LENGTH,
            "search_space": classical_oracle.search_space,
            "valid_routes": len(valid_indices),
        },
        "classical": {
            "astar": {
                "timing_s": describe(astar_times),
                "individual_times_s": astar_times,
                "valid_runs": sum(astar_valid),
                "total_runs": len(astar_valid),
            },
            "dijkstra": {
                "timing_s": describe(dijkstra_times),
                "individual_times_s": dijkstra_times,
                "valid_runs": sum(dijkstra_valid),
                "total_runs": len(dijkstra_valid),
            },
            "exhaustive": {
                "timing_s": describe(exhaustive_times),
                "individual_times_s": exhaustive_times,
                "valid_routes": len(valid_indices),
                "valid_indices": valid_indices,
            },
        },
        "quantum": {
            "algorithm": "Geometric Grover",
            "iterations": iterations,
            "oracle_calls_per_circuit": iterations,
            "qubits": circuit.num_qubits,
            "initial_depth": circuit.depth(),
            "transpiled_depth": compiled.depth(),
            "transpiled_operations": compiled.size(),
            "construction_time_s": construction_time,
            "transpilation_time_s": transpilation_time,
            "simulation_time_s": describe(grover_times),
            "success_probability_per_run": describe(
                grover_probabilities
            ),
            "aggregate_success_probability": aggregate_probability,
            "aggregate_valid_shots": total_valid_shots,
            "aggregate_total_shots": total_shots,
            "runs_recovering_all_solutions": sum(all_recovered),
            "runs": grover_runs,
        },
        "methodological_notes": [
            "Classical algorithms are measured repeatedly after one excluded warm-up.",
            "Grover uses one constructed and transpiled circuit for all simulation runs.",
            "Each Grover simulation uses a distinct simulator seed.",
            "Grover construction uses classical solution counting to estimate iterations.",
            "Construction and transpilation are reported separately from simulation.",
            "A* and Dijkstra solve shortest-path search; Grover searches fixed-length encoded routes.",
            "All quantum results are obtained with Aer simulation, not quantum hardware.",
            "This experiment does not establish quantum advantage.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STATISTICAL BENCHMARK SUMMARY")
    print("=" * 60)

    for name, values in (
        ("A*", astar_times),
        ("Dijkstra", dijkstra_times),
        ("Classical exhaustive", exhaustive_times),
        ("Grover simulation", grover_times),
    ):
        stats = describe(values)
        print(
            f"{name:22s} "
            f"mean={stats['mean']:.8f} s | "
            f"median={stats['median']:.8f} s | "
            f"std={stats['stdev']:.8f} s"
        )

    print(
        f"\nGrover aggregate success: "
        f"{100 * aggregate_probability:.2f}%"
    )

    print(
        "Runs recovering all solutions:",
        sum(all_recovered),
        "/",
        GROVER_REPETITIONS,
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
