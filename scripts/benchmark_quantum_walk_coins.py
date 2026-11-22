import numpy as np
from qiskit.quantum_info import Statevector

from environment.maze import Maze
from planners.quantum.quantum_walk import QuantumWalkPlanner
from planners.quantum.quantum_walk_oracle import QuantumWalkOracle
from planners.quantum.quantum_walk_coin import QuantumWalkCoin


def build_experiment(planner, oracle, start, goal, steps,
                     coin_type, use_oracle):

    circuit = planner.build_circuit(start, steps=0, measure=False)

    coin = (
        QuantumWalkCoin.hadamard_coin()
        if coin_type == "hadamard"
        else QuantumWalkCoin.grover_coin()
    )

    all_qubits = list(range(planner.total_qubits))

    for _ in range(steps):
        circuit.compose(coin, qubits=[0, 1], inplace=True)

        if use_oracle:
            circuit.append(oracle.to_gate(goal), all_qubits)

        planner.shift.apply_to(circuit, all_qubits)

    return circuit


maze = Maze(width=7, height=7, seed=42)
planner = QuantumWalkPlanner(maze)
oracle = QuantumWalkOracle(planner.encoder, planner.total_qubits)

missions = [
    ((1, 1), (5, 5)),
    ((1, 5), (5, 1)),
    ((5, 1), (1, 5)),
    ((5, 5), (1, 1)),
]

variants = [
    ("Hadamard", "hadamard", False),
    ("Hadamard+Oracle", "hadamard", True),
    ("Grover", "grover", False),
    ("Grover+Oracle", "grover", True),
]

all_results = {name: [] for name, _, _ in variants}

print("===== MULTI-MISSION QUANTUM WALK BENCHMARK =====")
print("Maze: 7x7")
print("Qubits:", planner.total_qubits)
print("Steps per experiment: 1..16")
print("Missions:", len(missions))

for mission_index, (start, goal) in enumerate(missions, 1):
    print(f"\n===== MISSION {mission_index} =====")
    print("Start:", start, "| Goal:", goal)

    for name, coin_type, use_oracle in variants:
        target_probabilities = []

        for steps in range(1, 17):
            circuit = build_experiment(
                planner, oracle, start, goal,
                steps, coin_type, use_oracle
            )

            statevector = Statevector.from_instruction(circuit)
            probabilities = planner.get_position_probabilities(
                statevector.data
            )

            total = sum(probabilities.values())
            assert np.isclose(total, 1.0, atol=1e-9)

            target_probabilities.append(probabilities[goal])

        peak_probability = max(target_probabilities)
        peak_step = target_probabilities.index(peak_probability) + 1
        late_mean = float(np.mean(target_probabilities[8:16]))

        all_results[name].append({
            "peak": peak_probability,
            "peak_step": peak_step,
            "late_mean": late_mean,
        })

        print(
            f"{name:17s} | "
            f"peak={peak_probability:.6f} "
            f"(step {peak_step:2d}) | "
            f"mean steps 9-16={late_mean:.6f}"
        )

print("\n===== AGGREGATE RESULTS =====")

for name, _, _ in variants:
    records = all_results[name]

    mean_peak = np.mean([r["peak"] for r in records])
    mean_late = np.mean([r["late_mean"] for r in records])

    print(
        f"{name:17s} | "
        f"mean peak={mean_peak:.6f} | "
        f"mean late probability={mean_late:.6f}"
    )

print("\n===== ORACLE EFFECT =====")

for coin_name in ("Hadamard", "Grover"):
    baseline = all_results[coin_name]
    marked = all_results[coin_name + "+Oracle"]

    peak_deltas = [
        marked[i]["peak"] - baseline[i]["peak"]
        for i in range(len(missions))
    ]

    late_deltas = [
        marked[i]["late_mean"] - baseline[i]["late_mean"]
        for i in range(len(missions))
    ]

    print(
        f"{coin_name:10s} | "
        f"mean peak delta={np.mean(peak_deltas):+.6f} | "
        f"mean late delta={np.mean(late_deltas):+.6f} | "
        f"missions improved={sum(d > 1e-9 for d in late_deltas)}/4"
    )

print("\nMULTI-MISSION BENCHMARK COMPLETED")
