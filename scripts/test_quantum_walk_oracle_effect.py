import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from environment.maze import Maze
from planners.quantum.quantum_walk import QuantumWalkPlanner
from planners.quantum.quantum_walk_oracle import QuantumWalkOracle


def build_experiment(planner, oracle, start, goal, steps, use_oracle):
    circuit = planner.build_circuit(start, steps=0, measure=False)

    for _ in range(steps):
        circuit.compose(
            planner.coin.hadamard_coin(),
            qubits=[0, 1],
            inplace=True,
        )

        if use_oracle:
            circuit.append(
                oracle.to_gate(goal),
                list(range(planner.total_qubits)),
            )

        planner.shift.apply_to(
            circuit,
            list(range(planner.total_qubits)),
        )

    circuit.save_statevector()
    return circuit


def summarize(planner, statevector, goal):
    probabilities = planner.get_position_probabilities(statevector)

    goal_probability = probabilities[goal]

    neighbors = planner.maze.get_neighbors(*goal)
    neighbor_probability = sum(probabilities[cell] for cell in neighbors)

    assert np.isclose(
        sum(probabilities.values()),
        1.0,
        atol=1e-9,
    ), "Position probabilities are not normalized."

    return goal_probability, neighbor_probability, probabilities


maze = Maze(width=7, height=7, seed=42)

start = (1, 1)
goal = (5, 5)

planner = QuantumWalkPlanner(maze)
oracle = QuantumWalkOracle(planner.encoder, planner.total_qubits)

backend = AerSimulator(method="statevector")

print("===== QUANTUM WALK ORACLE COMPARISON =====")
print("Start:", start)
print("Goal:", goal)
print("Total qubits:", planner.total_qubits)

for steps in range(1, 17):
    results = {}

    for label, use_oracle in [
        ("STANDARD", False),
        ("WITH_ORACLE", True),
    ]:
        circuit = build_experiment(
            planner,
            oracle,
            start,
            goal,
            steps,
            use_oracle,
        )

        compiled = transpile(
            circuit,
            backend,
            seed_transpiler=42,
        )

        result = backend.run(
            compiled,
            shots=1,
            seed_simulator=42,
        ).result()

        statevector = result.get_statevector()

        goal_p, neighbor_p, probabilities = summarize(
            planner,
            statevector,
            goal,
        )

        results[label] = (goal_p, neighbor_p)

    standard = results["STANDARD"]
    marked = results["WITH_ORACLE"]

    print(f"\nSteps: {steps}")
    print(
        f"Standard: goal={standard[0]:.6f}, "
        f"neighbors={standard[1]:.6f}"
    )
    print(
        f"Oracle:   goal={marked[0]:.6f}, "
        f"neighbors={marked[1]:.6f}"
    )


print("\nORACLE EFFECT EXPERIMENT COMPLETED")
