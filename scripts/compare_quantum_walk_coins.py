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


maze = Maze(width=7, height=7, seed=42)

start = (1, 1)
goal = (5, 5)

planner = QuantumWalkPlanner(maze)
oracle = QuantumWalkOracle(
    planner.encoder,
    planner.total_qubits,
)

experiments = [
    ("HADAMARD", "hadamard", False),
    ("HADAMARD + ORACLE", "hadamard", True),
    ("GROVER", "grover", False),
    ("GROVER + ORACLE", "grover", True),
]

print("===== QUANTUM WALK COIN COMPARISON =====")
print("Start:", start)
print("Goal:", goal)
print("Qubits:", planner.total_qubits)

for steps in range(1, 17):
    print(f"\n--- Steps: {steps} ---")

    for label, coin_type, use_oracle in experiments:
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

        total_probability = sum(probabilities.values())
        goal_probability = probabilities[goal]

        neighbor_probability = sum(
            probabilities[cell]
            for cell in maze.get_neighbors(*goal)
        )

        assert np.isclose(total_probability, 1.0, atol=1e-9)

        print(
            f"{label:20s} | "
            f"goal={goal_probability:.6f} | "
            f"neighbors={neighbor_probability:.6f}"
        )

print("\nQUANTUM WALK COIN COMPARISON COMPLETED")
