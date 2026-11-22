import numpy as np
from collections import deque
from qiskit.quantum_info import Statevector

from environment.maze import Maze, STATIC_WALL
from planners.quantum.quantum_walk import QuantumWalkPlanner
from planners.quantum.quantum_walk_oracle import QuantumWalkOracle
from planners.quantum.quantum_walk_coin import QuantumWalkCoin


MISSIONS = [
    ((1, 1), (5, 5)),
    ((1, 5), (5, 1)),
    ((5, 1), (1, 5)),
    ((5, 5), (1, 1)),
]

VARIANTS = [
    ("Hadamard", "hadamard", False),
    ("Hadamard+Oracle", "hadamard", True),
    ("Grover", "grover", False),
    ("Grover+Oracle", "grover", True),
]

CONFIGURATIONS = {
    "VERTICAL": [
        (3, y) for y in range(1, 6) if y != 3
    ],
    "HORIZONTAL": [
        (x, 3) for x in range(1, 6) if x != 3
    ],
    "CROSS": (
        [(3, y) for y in range(1, 6) if y not in (2, 3, 4)]
        + [(x, 3) for x in range(1, 6) if x not in (2, 3, 4)]
    ),
}


def is_connected(maze, start, goal):
    queue = deque([start])
    visited = {start}

    while queue:
        cell = queue.popleft()

        if cell == goal:
            return True

        for neighbor in maze.get_neighbors(*cell):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False


def build_experiment(planner, oracle, start, goal, steps,
                     coin_type, use_oracle):

    circuit = planner.build_circuit(
        start,
        steps=0,
        measure=False,
    )

    coin = (
        QuantumWalkCoin.hadamard_coin()
        if coin_type == "hadamard"
        else QuantumWalkCoin.grover_coin()
    )

    all_qubits = list(range(planner.total_qubits))

    for _ in range(steps):
        circuit.compose(
            coin,
            qubits=[0, 1],
            inplace=True,
        )

        if use_oracle:
            circuit.append(
                oracle.to_gate(goal),
                all_qubits,
            )

        planner.shift.apply_to(circuit, all_qubits)

    return circuit


aggregate = {
    name: {"peaks": [], "late_means": []}
    for name, _, _ in VARIANTS
}

print("===== QUANTUM WALK OBSTACLE BENCHMARK =====")
print("Maze dimensions: 7x7")
print("Missions per configuration:", len(MISSIONS))
print("Evolution steps: 1..16")

for config_name, walls in CONFIGURATIONS.items():
    maze = Maze(width=7, height=7, seed=42)

    for x, y in walls:
        maze.set_cell(x, y, STATIC_WALL)

    planner = QuantumWalkPlanner(maze)
    oracle = QuantumWalkOracle(
        planner.encoder,
        planner.total_qubits,
    )

    print(f"\n========== {config_name} ==========")
    print("Static walls:", len(walls))
    print("Walkable positions:", planner.number_of_positions)
    print("Total qubits:", planner.total_qubits)

    for mission_index, (start, goal) in enumerate(MISSIONS, 1):
        if not is_connected(maze, start, goal):
            raise RuntimeError(
                f"Disconnected mission: {config_name}, "
                f"{start} -> {goal}"
            )

        print(f"\nMission {mission_index}: {start} -> {goal}")

        for name, coin_type, use_oracle in VARIANTS:
            target_probabilities = []

            for steps in range(1, 17):
                circuit = build_experiment(
                    planner,
                    oracle,
                    start,
                    goal,
                    steps,
                    coin_type,
                    use_oracle,
                )

                statevector = Statevector.from_instruction(circuit)

                probabilities = planner.get_position_probabilities(
                    statevector.data
                )

                assert np.isclose(
                    sum(probabilities.values()),
                    1.0,
                    atol=1e-9,
                )

                target_probabilities.append(probabilities[goal])

            peak = max(target_probabilities)
            peak_step = target_probabilities.index(peak) + 1
            late_mean = float(np.mean(target_probabilities[8:16]))

            aggregate[name]["peaks"].append(peak)
            aggregate[name]["late_means"].append(late_mean)

            print(
                f"{name:17s} | "
                f"peak={peak:.6f} (step {peak_step:2d}) | "
                f"mean 9-16={late_mean:.6f}"
            )

print("\n========== GLOBAL AGGREGATE ==========")

for name, _, _ in VARIANTS:
    peaks = aggregate[name]["peaks"]
    late = aggregate[name]["late_means"]

    print(
        f"{name:17s} | "
        f"mean peak={np.mean(peaks):.6f} | "
        f"mean late={np.mean(late):.6f}"
    )

print("\n========== GROVER VS HADAMARD ==========")

for metric in ("peaks", "late_means"):
    h = np.mean(aggregate["Hadamard"][metric])
    g = np.mean(aggregate["Grover"][metric])

    print(
        f"{metric:12s} | "
        f"Hadamard={h:.6f} | "
        f"Grover={g:.6f} | "
        f"Grover/Hadamard={g / h if h > 0 else float('nan'):.3f}"
    )

print("\nOBSTACLE BENCHMARK COMPLETED")
