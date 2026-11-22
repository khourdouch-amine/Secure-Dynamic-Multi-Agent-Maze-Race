"""
Grover-assisted local navigation.

Grover amplifies neighboring cells that minimize Manhattan distance
to the destination. The circuit is simulated using Qiskit Aer.
This is a quantum-assisted local decision policy, not a guaranteed
globally optimal path planner.
"""

from __future__ import annotations

from collections import deque
from math import asin, ceil, log2, pi, sqrt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit_aer import AerSimulator

from planners.quantum.quantum_base import QuantumPlannerBase
from planners.quantum.grid_encoder import GridGraphEncoder


class GroverNavigationPlanner(QuantumPlannerBase):
    """Select local navigation actions using Grover amplification."""

    def __init__(
        self,
        shots: int = 256,
        seed: int = 42,
        max_steps: int = 1000,
    ):
        super().__init__("Grover-assisted navigation")

        if shots < 1:
            raise ValueError("shots must be positive.")

        if max_steps < 1:
            raise ValueError("max_steps must be positive.")

        self.shots = shots
        self.seed = seed
        self.max_steps = max_steps
        self.backend = AerSimulator()
        self.last_grover_iterations = 0
        self.last_circuit_executed = False
        self.last_decision_used_grover = False
        self.last_selection_fallback = False

    @staticmethod
    def _compute_goal_distances(maze, goal):
        """Compute shortest grid distances from every reachable cell to goal."""
        distances = {tuple(goal): 0}
        queue = deque([tuple(goal)])

        while queue:
            current = queue.popleft()

            for neighbor in maze.get_neighbors(*current):
                neighbor = tuple(neighbor)

                if neighbor in distances:
                    continue

                if not maze.is_walkable(*neighbor):
                    continue

                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

        return distances

    def _grover_select(self, candidates, goal, step_index, distance_map=None):
        """
        Amplify candidate indices having minimum Manhattan distance.

        Returns one candidate index sampled from the measured circuit.
        """
        self.last_grover_iterations = 0
        self.last_circuit_executed = False
        self.last_decision_used_grover = False
        self.last_selection_fallback = False

        count = len(candidates)

        if count == 0:
            return None

        if count == 1:
            self.last_selection_fallback = True
            return 0

        number_of_qubits = max(1, ceil(log2(count)))
        dimension = 2 ** number_of_qubits

        if distance_map is None:
            distances = [
                abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
                for cell in candidates
            ]
        else:
            distances = [
                distance_map.get(cell, float("inf"))
                for cell in candidates
            ]

        best_distance = min(distances)

        marked = [
            index
            for index, distance in enumerate(distances)
            if distance == best_distance
        ]

        # Grover's search domain is padded to a power of two.
        # Invalid padded states are deliberately not marked.
        oracle = QuantumCircuit(number_of_qubits, name="navigation_oracle")

        for index in marked:
            binary = format(index, f"0{number_of_qubits}b")

            for qubit, bit in enumerate(reversed(binary)):
                if bit == "0":
                    oracle.x(qubit)

            oracle.h(number_of_qubits - 1)
            oracle.mcx(
                list(range(number_of_qubits - 1)),
                number_of_qubits - 1,
            )
            oracle.h(number_of_qubits - 1)

            for qubit, bit in enumerate(reversed(binary)):
                if bit == "0":
                    oracle.x(qubit)

        oracle_gate = oracle.to_gate()

        circuit = QuantumCircuit(number_of_qubits, number_of_qubits)
        circuit.h(range(number_of_qubits))

        # Select the iteration count from the Grover success probability.
        # N includes padded basis states; M is the number of marked states.
        theta = asin(sqrt(len(marked) / dimension))
        optimal_iterations = max(
            0,
            round(pi / (4 * theta) - 0.5),
        )

        self.last_grover_iterations = optimal_iterations

        # When no Grover iteration is optimal, avoid a random
        # measurement over marked and unmarked states.
        # Use a deterministic classical fallback for this small search.
        if optimal_iterations == 0:
            self.last_selection_fallback = True
            return marked[0]

        grover = grover_operator(oracle_gate)

        for _ in range(optimal_iterations):
            circuit.compose(grover, inplace=True)

        circuit.measure(range(number_of_qubits), range(number_of_qubits))

        compiled = transpile(
            circuit,
            self.backend,
            seed_transpiler=self.seed + step_index,
        )

        result = self.backend.run(
            compiled,
            shots=self.shots,
            seed_simulator=self.seed + step_index,
        ).result()

        counts = result.get_counts()
        self.last_circuit_executed = True

        # Only accept measured states that are both valid and marked.
        marked_set = set(marked)
        marked_counts = {
            bitstring: frequency
            for bitstring, frequency in counts.items()
            if int(bitstring, 2) < count
            and int(bitstring, 2) in marked_set
        }

        if not marked_counts:
            self.last_selection_fallback = True
            return marked[0]

        selected = int(
            max(marked_counts, key=marked_counts.get),
            2,
        )

        self.last_decision_used_grover = True
        return selected

    def plan(self, maze, start, goal):
        start, goal = self.validate_problem(maze, start, goal)

        decisions = 0
        quantum_decisions = 0
        classical_fallback_decisions = 0
        grover_circuit_executions = 0
        total_grover_iterations = 0

        if start == goal:
            self.last_metadata = {
                "algorithm": self.name,
                "quantum_backend": "Qiskit Aer",
                "decisions": 0,
                "shots_per_decision": self.shots,
            }
            return [start]

        encoder = GridGraphEncoder(maze)
        goal_distances = self._compute_goal_distances(maze, goal)

        if start not in goal_distances:
            self.last_metadata = {
                "algorithm": self.name,
                "failure": "goal is unreachable from start",
                "quantum_backend": "Qiskit Aer",
            }
            return None

        current = start
        path = [current]
        visited = {current}

        for step_index in range(self.max_steps):
            if current == goal:
                break

            state = encoder.encode_state(current)

            candidates = [
                encoder.decode_state(neighbor_state)
                for neighbor_state in encoder.neighbors(state)
                if (
                    encoder.decode_state(neighbor_state) not in visited
                    and encoder.decode_state(neighbor_state) in goal_distances
                )
            ]

            if not candidates:
                return None

            selected_index = self._grover_select(
                candidates,
                goal,
                step_index,
                distance_map=goal_distances,
            )

            if selected_index is None:
                return None

            next_cell = candidates[selected_index]

            # Classical safety validation of the measured action.
            if not maze.is_walkable(*next_cell):
                return None

            if (
                abs(next_cell[0] - current[0])
                + abs(next_cell[1] - current[1])
            ) != 1:
                return None

            current = next_cell
            path.append(current)
            visited.add(current)

            decisions += 1

            if self.last_circuit_executed:
                grover_circuit_executions += 1
                total_grover_iterations += self.last_grover_iterations

            if self.last_decision_used_grover:
                quantum_decisions += 1
            else:
                classical_fallback_decisions += 1

        if current != goal:
            return None

        self.last_metadata = {
            "algorithm": self.name,
            "quantum_backend": "Qiskit Aer",
            "shots_per_circuit": self.shots,
            "total_decisions": decisions,
            "quantum_decisions": quantum_decisions,
            "classical_fallback_decisions": classical_fallback_decisions,
            "grover_circuit_executions": grover_circuit_executions,
            "total_grover_iterations": total_grover_iterations,
            "total_simulated_shots": grover_circuit_executions * self.shots,
            "qubits_per_decision_max": 2,
            "oracle_criterion": "minimum BFS distance to goal",
            "distance_map": "classical BFS",
            "navigation_policy": "BFS-guided Grover action selection",
            "shortest_path_guaranteed_on_static_maze": True,
        }

        return path
