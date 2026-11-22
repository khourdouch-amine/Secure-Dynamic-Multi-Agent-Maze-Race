"""
Variational Quantum Classifier (VQC) navigation planner.

Hybrid architecture:
- Classical BFS generates supervised action labels.
- A PennyLane variational circuit scores four cardinal actions.
- Classical COBYLA optimizes the circuit parameters.
- Classical path validation and BFS fallback preserve correctness.

No quantum speedup or pure-quantum navigation is claimed.
"""

from collections import deque

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from planners.quantum.quantum_base import QuantumPlannerBase


class VQCNavigationPlanner(QuantumPlannerBase):
    """Navigate with a supervised variational quantum policy."""

    DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
    ACTIONS = ("right", "left", "down", "up")
    N_QUBITS = 6
    N_ACTIONS = 4

    def __init__(
        self,
        layers=2,
        shots=128,
        max_iterations=40,
        max_training_samples=64,
        seed=42,
    ):
        super().__init__("VQC Navigation (hybrid)")

        if not isinstance(layers, int) or layers < 1:
            raise ValueError("layers must be positive.")
        if not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be positive.")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if not isinstance(max_training_samples, int) or max_training_samples < 1:
            raise ValueError("max_training_samples must be positive.")

        self.layers = layers
        self.shots = shots
        self.max_iterations = max_iterations
        self.max_training_samples = max_training_samples
        self.seed = seed

    @staticmethod
    def _distances_to_goal(maze, goal):
        distances = {goal: 0}
        queue = deque([goal])

        while queue:
            cell = queue.popleft()
            for neighbor in maze.get_neighbors(*cell):
                neighbor = tuple(neighbor)
                if neighbor not in distances and maze.is_walkable(*neighbor):
                    distances[neighbor] = distances[cell] + 1
                    queue.append(neighbor)

        return distances

    def _features(self, maze, cell, goal):
        x, y = cell
        dx = (goal[0] - x) / max(1, maze.width - 1)
        dy = (goal[1] - y) / max(1, maze.height - 1)

        free = [
            float(maze.is_walkable(x + ox, y + oy))
            for ox, oy in self.DIRECTIONS
        ]

        return np.asarray(
            [np.pi * value for value in free] + [np.pi * dx, np.pi * dy],
            dtype=float,
        )

    def _training_data(self, maze, goal, distances, start):
        samples = []

        for cell in sorted(distances):
            if cell == goal:
                continue

            legal = [
                (index, (cell[0] + dx, cell[1] + dy))
                for index, (dx, dy) in enumerate(self.DIRECTIONS)
                if maze.is_walkable(cell[0] + dx, cell[1] + dy)
                and (cell[0] + dx, cell[1] + dy) in distances
            ]

            if not legal:
                continue

            action, _ = min(
                legal,
                key=lambda item: (distances[item[1]], item[0]),
            )
            target = -np.ones(self.N_ACTIONS, dtype=float)
            target[action] = 1.0
            samples.append((self._features(maze, cell, goal), target))

        if len(samples) > self.max_training_samples:
            indices = np.linspace(
                0,
                len(samples) - 1,
                self.max_training_samples,
                dtype=int,
            )
            selected = [samples[index] for index in indices]

            start_features = self._features(maze, start, goal)
            start_sample = next(
                (
                    sample for sample in samples
                    if np.array_equal(sample[0], start_features)
                ),
                None,
            )

            if start_sample is not None and not any(
                np.array_equal(sample[0], start_features)
                for sample in selected
            ):
                selected[-1] = start_sample

            samples = selected

        return samples

    def _build_circuits(self):
        device = qml.device(
            "default.qubit",
            wires=self.N_QUBITS,
            shots=None,
            seed=self.seed,
        )

        def circuit(features, weights):
            qml.AngleEmbedding(
                features,
                wires=range(self.N_QUBITS),
                rotation="Y",
            )

            for layer in range(self.layers):
                for wire in range(self.N_QUBITS):
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)

                for wire in range(self.N_QUBITS):
                    qml.CNOT(
                        wires=[wire, (wire + 1) % self.N_QUBITS]
                    )

            return tuple(
                qml.expval(qml.PauliZ(action))
                for action in range(self.N_ACTIONS)
            )

        @qml.qnode(device)
        def analytic_policy(features, weights):
            return circuit(features, weights)

        @qml.qnode(device)
        def sampled_policy(features, weights):
            return circuit(features, weights)

        sampled_policy = qml.set_shots(
            sampled_policy,
            shots=self.shots,
        )

        return analytic_policy, sampled_policy

    @staticmethod
    def _classical_fallback(maze, start, goal):
        queue = deque([start])
        parents = {start: None}

        while queue:
            cell = queue.popleft()

            if cell == goal:
                path = []
                while cell is not None:
                    path.append(cell)
                    cell = parents[cell]
                return list(reversed(path))

            for neighbor in maze.get_neighbors(*cell):
                neighbor = tuple(neighbor)
                if neighbor not in parents and maze.is_walkable(*neighbor):
                    parents[neighbor] = cell
                    queue.append(neighbor)

        return None

    def plan(self, maze, start, goal):
        start, goal = self.validate_problem(maze, start, goal)

        if start == goal:
            self.last_metadata = {
                "algorithm": self.name,
                "algorithm_type": "hybrid_quantum_classical",
                "path_source": "trivial",
                "quantum_circuit_executions": 0,
                "failure": None,
            }
            return [start]

        distances = self._distances_to_goal(maze, goal)

        if start not in distances:
            self.last_metadata = {
                "algorithm": self.name,
                "algorithm_type": "hybrid_quantum_classical",
                "path_source": "no_path",
                "quantum_circuit_executions": 0,
                "failure": "goal is unreachable",
            }
            return None

        training_data = self._training_data(
            maze, goal, distances, start
        )

        if not training_data:
            return None

        analytic_policy, sampled_policy = self._build_circuits()

        rng = np.random.default_rng(self.seed)
        initial_weights = rng.uniform(
            -0.1,
            0.1,
            size=(self.layers, self.N_QUBITS, 2),
        )

        evaluations = 0

        def objective(flat_weights):
            nonlocal evaluations
            evaluations += 1

            weights = flat_weights.reshape(
                self.layers, self.N_QUBITS, 2
            )

            losses = []
            for features, target in training_data:
                predictions = np.asarray(
                    analytic_policy(features, weights),
                    dtype=float,
                )
                losses.append(
                    float(np.mean((predictions - target) ** 2))
                )

            return float(np.mean(losses))

        parameter_count = initial_weights.size
        effective_max_iterations = max(
            self.max_iterations,
            parameter_count + 2,
        )

        result = minimize(
            objective,
            initial_weights.reshape(-1),
            method="COBYLA",
            options={"maxiter": effective_max_iterations},
        )

        trained_weights = result.x.reshape(
            self.layers, self.N_QUBITS, 2
        )

        path = [start]
        visited = {start}
        current = start
        decisions = 0
        fallback_reason = None

        for _ in range(maze.width * maze.height):
            if current == goal:
                break

            features = self._features(maze, current, goal)
            scores = np.asarray(
                sampled_policy(features, trained_weights),
                dtype=float,
            )
            decisions += 1

            legal = []
            for action, (dx, dy) in enumerate(self.DIRECTIONS):
                neighbor = (current[0] + dx, current[1] + dy)

                if maze.is_walkable(*neighbor) and neighbor not in visited:
                    legal.append((action, neighbor))

            if not legal:
                fallback_reason = "policy reached a dead end or cycle"
                break

            _, neighbor = max(
                legal,
                key=lambda item: (scores[item[0]], -item[0]),
            )

            path.append(neighbor)
            visited.add(neighbor)
            current = neighbor

        if current != goal:
            fallback_reason = (
                fallback_reason or "policy step budget exhausted"
            )
            path = self._classical_fallback(maze, start, goal)
            path_source = "classical_bfs_fallback"
        else:
            path_source = "vqc_policy"

        circuit_executions = evaluations * len(training_data) + decisions

        if path is None:
            self.last_metadata = {
                "algorithm": self.name,
                "algorithm_type": "hybrid_quantum_classical",
                "path_source": "no_path",
                "quantum_circuit_executions": circuit_executions,
                "failure": "classical fallback could not find a path",
            }
            return None

        path = self.validate_path(maze, path, start, goal)

        self.last_metadata = {
            "algorithm": self.name,
            "algorithm_type": "hybrid_quantum_classical",
            "quantum_backend": "PennyLane default.qubit",
            "qubits": self.N_QUBITS,
            "layers": self.layers,
            "shots": self.shots,
            "training_samples": len(training_data),
            "classical_teacher": "BFS shortest-path labels",
            "classical_optimizer": "COBYLA",
            "optimizer_evaluations": evaluations,
            "optimizer_max_iterations": effective_max_iterations,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "path_source": path_source,
            "fallback_used": path_source == "classical_bfs_fallback",
            "fallback_reason": fallback_reason,
            "quantum_policy_decisions": decisions,
            "quantum_circuit_executions": circuit_executions,
            "path_length": len(path) - 1,
            "failure": None,
        }

        return path
