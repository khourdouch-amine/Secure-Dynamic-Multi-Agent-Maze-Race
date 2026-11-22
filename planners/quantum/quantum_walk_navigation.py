"""
Hybrid quantum-classical navigation using a discrete-time coined
quantum walk and Qiskit Aer measurements.

Quantum component:
    Each admissible next cell is evaluated through a quantum walk.
    The measured probability of reaching the goal determines its score.

Classical component:
    Maze validity, candidate filtering, loop penalty, and fallback
    selection are handled classically.

This is a simulated hybrid navigation method, not a claim of
quantum speedup.
"""

from collections import deque
from time import perf_counter

from qiskit import transpile
from qiskit_aer import AerSimulator

from planners.quantum.quantum_base import QuantumPlannerBase
from planners.quantum.quantum_walk import QuantumWalkPlanner


class QuantumWalkNavigationPlanner(QuantumPlannerBase):
    """Navigate a grid using quantum-walk candidate evaluation."""

    def __init__(
        self,
        maze,
        coin_type="grover",
        shots=256,
        walk_steps=8,
        max_steps=None,
        seed=42,
        max_total_qubits=None,
    ):
        super().__init__(f"Quantum Walk Navigation ({coin_type})")

        if coin_type not in {"hadamard", "grover"}:
            raise ValueError("coin_type must be 'hadamard' or 'grover'.")

        if not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be a positive integer.")

        if not isinstance(walk_steps, int) or walk_steps < 0:
            raise ValueError("walk_steps must be a non-negative integer.")

        if max_steps is not None and (
            not isinstance(max_steps, int) or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer.")

        self.name = f"Quantum Walk Navigation ({coin_type})"
        self.maze = maze
        self.coin_type = coin_type
        self.shots = shots
        self.walk_steps = walk_steps
        self.seed = seed

        self.max_total_qubits = max_total_qubits

        self.walk = QuantumWalkPlanner(
            maze,
            max_total_qubits=max_total_qubits,
        )
        self.encoder = self.walk.encoder

        self._automatic_max_steps = max_steps is None

        self.max_steps = (
            max_steps
            if max_steps is not None
            else max(100, 4 * self.walk.number_of_positions)
        )

        self.simulator = AerSimulator()
        self._circuit_counter = 0

        self.last_metadata = {}

    def _goal_distances(self, goal):
        """Compute classical graph distances for fallback decisions only."""
        distances = {goal: 0}
        queue = deque([goal])

        while queue:
            current = queue.popleft()

            for neighbor in self.maze.get_neighbors(*current):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return distances

    def _measure_goal_probability(self, candidate, goal):
        """Run one measured quantum walk initialized at candidate."""
        circuit = self.walk.build_circuit(
            candidate,
            steps=self.walk_steps,
            measure=True,
            coin_type=self.coin_type,
        )

        compiled = transpile(
            circuit,
            self.simulator,
            seed_transpiler=self.seed,
        )

        result = self.simulator.run(
            compiled,
            shots=self.shots,
            seed_simulator=self.seed + self._circuit_counter,
        ).result()

        self._circuit_counter += 1

        counts = result.get_counts()
        goal_index = self.encoder.encode_state(goal)

        goal_count = 0

        for bitstring, frequency in counts.items():
            position_index = int(bitstring.replace(" ", ""), 2)

            if position_index == goal_index:
                goal_count += frequency

        return goal_count / self.shots

    def _record_metadata(
        self,
        decisions,
        quantum_decisions,
        fallback_decisions,
        circuit_executions,
        path,
        failure=None,
    ):
        self.last_metadata = {
            "algorithm": self.name,
            "quantum_backend": self.simulator.name,
            "coin_type": self.coin_type,
            "shots_per_circuit": self.shots,
            "walk_steps_per_evaluation": self.walk_steps,
            "total_decisions": decisions,
            "quantum_decisions": quantum_decisions,
            "classical_fallback_decisions": fallback_decisions,
            "quantum_circuit_executions": circuit_executions,
            "path_length": max(0, len(path) - 1),
            "path_nodes": len(path),
            "failure": failure,
        }

    def plan(self, maze, start, goal):
        """Return a cardinal grid path or None if navigation fails."""
        if maze is not self.maze:
            raise ValueError(
                "The planner must be used with the maze supplied at initialization."
            )

        # Refresh the quantum representation before validating the problem.
        environment_changed = self.walk.refresh_environment()

        if environment_changed:
            self.encoder = self.walk.encoder

            if self._automatic_max_steps:
                self.max_steps = max(
                    100,
                    4 * self.walk.number_of_positions,
                )

        start, goal = self.validate_problem(maze, start, goal)

        self._circuit_counter = 0

        if start == goal:
            self._record_metadata(0, 0, 0, 0, [start])
            return [start]

        goal_distances = self._goal_distances(goal)

        if start not in goal_distances:
            self._record_metadata(
                0, 0, 0, 0, [],
                failure="goal is unreachable",
            )
            return None

        current = start
        path = [current]
        visit_counts = {current: 1}

        decisions = 0
        quantum_decisions = 0
        fallback_decisions = 0

        for _ in range(self.max_steps):
            if current == goal:
                break

            current_distance = goal_distances[current]

            candidates = [
                neighbor
                for neighbor in maze.get_neighbors(*current)
                if goal_distances.get(neighbor) == current_distance - 1
            ]

            if not candidates:
                self._record_metadata(
                    decisions,
                    quantum_decisions,
                    fallback_decisions,
                    self._circuit_counter,
                    path,
                    failure="no admissible neighboring cell",
                )
                return None

            scores = {}

            for candidate in candidates:
                probability = self._measure_goal_probability(
                    candidate,
                    goal,
                )

                visits = visit_counts.get(candidate, 0)

                # Penalize repeated visits without forbidding them.
                scores[candidate] = probability / (1 + visits)

            best_score = max(scores.values())

            if best_score > 0.0:
                selected = max(
                    candidates,
                    key=lambda cell: (
                        scores[cell],
                        -goal_distances[cell],
                        -visit_counts.get(cell, 0),
                        -self.encoder.encode_state(cell),
                    ),
                )
                quantum_decisions += 1

            else:
                # Explicit classical fallback when measurements provide
                # no observed probability for the goal.
                selected = min(
                    candidates,
                    key=lambda cell: (
                        goal_distances[cell],
                        visit_counts.get(cell, 0),
                        self.encoder.encode_state(cell),
                    ),
                )
                fallback_decisions += 1

            if not maze.is_walkable(*selected):
                raise RuntimeError("Selected navigation cell is not walkable.")

            if (
                abs(selected[0] - current[0])
                + abs(selected[1] - current[1])
            ) != 1:
                raise RuntimeError("Selected move is not cardinally adjacent.")

            current = selected
            path.append(current)
            visit_counts[current] = visit_counts.get(current, 0) + 1
            decisions += 1

        if current != goal:
            self._record_metadata(
                decisions,
                quantum_decisions,
                fallback_decisions,
                self._circuit_counter,
                path,
                failure="maximum navigation steps reached",
            )
            return None

        self.validate_path(maze, path, start, goal)

        self._record_metadata(
            decisions,
            quantum_decisions,
            fallback_decisions,
            self._circuit_counter,
            path,
        )

        return path
