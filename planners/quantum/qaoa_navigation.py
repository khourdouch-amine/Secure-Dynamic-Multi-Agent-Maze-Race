"""
QAOA-based path selection.

Classical component:
    Builds a bounded set of candidate paths using NetworkX.

Quantum component:
    A QAOA circuit selects one candidate through a one-hot
    constrained binary optimization objective.

Classical component:
    COBYLA optimizes QAOA parameters and validates the selected path.

This is a hybrid quantum-classical algorithm. It does not claim
quantum speedup or pure quantum navigation.
"""

from collections import Counter

import networkx as nx
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

from planners.quantum.quantum_base import QuantumPlannerBase


class QAOANavigationPlanner(QuantumPlannerBase):
    """Select a maze path using a QAOA circuit."""

    def __init__(
        self,
        candidates=6,
        layers=2,
        shots=256,
        max_iterations=80,
        penalty=2.0,
        seed=42,
    ):
        super().__init__("QAOA Path Selection (hybrid)")

        if not isinstance(candidates, int) or not 1 <= candidates <= 10:
            raise ValueError("candidates must be between 1 and 10.")
        if not isinstance(layers, int) or layers < 1:
            raise ValueError("layers must be positive.")
        if not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be positive.")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if penalty <= 0:
            raise ValueError("penalty must be positive.")

        self.max_candidates = candidates
        self.layers = layers
        self.shots = shots
        self.max_iterations = max_iterations
        self.penalty = float(penalty)
        self.seed = seed

    @staticmethod
    def _candidate_paths(maze, start, goal, limit):
        """Generate up to limit shortest simple paths classically."""
        graph = nx.Graph()

        for y in range(maze.height):
            for x in range(maze.width):
                cell = (x, y)
                if maze.is_walkable(x, y):
                    graph.add_node(cell)

        for cell in list(graph.nodes):
            for neighbor in maze.get_neighbors(*cell):
                neighbor = tuple(neighbor)
                if neighbor in graph:
                    graph.add_edge(cell, neighbor)

        try:
            generator = nx.shortest_simple_paths(graph, start, goal)
            paths = []

            for path in generator:
                paths.append([tuple(cell) for cell in path])
                if len(paths) >= limit:
                    break

            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    @staticmethod
    def _cost_hamiltonian(costs, penalty):
        """
        Encode:
            sum_i cost[i] * x[i] + penalty * (sum_i x[i] - 1)^2
        with x[i] = (1 - Z[i]) / 2.
        """
        n = len(costs)
        coefficients = [0.0]
        observables = [qml.Identity(0)]

        linear = [float(cost) - penalty for cost in costs]

        # Constant contribution.
        constant = penalty + 0.5 * sum(linear)
        constant += (penalty / 2.0) * (n * (n - 1) / 2)
        coefficients[0] = constant

        # Single-qubit Z terms.
        z_coefficients = [
            -linear[i] / 2.0 - penalty * (n - 1) / 2.0
            for i in range(n)
        ]

        for i, coefficient in enumerate(z_coefficients):
            coefficients.append(coefficient)
            observables.append(qml.PauliZ(i))

        # Pairwise Z_i Z_j terms.
        for i in range(n):
            for j in range(i + 1, n):
                coefficients.append(penalty / 2.0)
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

        return qml.Hamiltonian(coefficients, observables)

    def _build_qaoa(self, number_of_candidates):
        """Build analytic objective and finite-shot sampling circuits."""
        cost_h = self._cost_hamiltonian(
            self._normalized_costs,
            self.penalty,
        )
        mixer_h = qml.sum(
            *[qml.PauliX(i) for i in range(number_of_candidates)]
        )

        analytic_device = qml.device(
            "default.qubit",
            wires=number_of_candidates,
            shots=None,
        )
        sampling_device = qml.device(
            "default.qubit",
            wires=number_of_candidates,
            shots=None,
            seed=self.seed,
        )

        def circuit(parameters):
            for wire in range(number_of_candidates):
                qml.Hadamard(wires=wire)

            for layer in range(self.layers):
                qml.qaoa.cost_layer(parameters[0, layer], cost_h)
                qml.qaoa.mixer_layer(parameters[1, layer], mixer_h)

        @qml.qnode(analytic_device)
        def objective(parameters):
            circuit(parameters)
            return qml.expval(cost_h)

        @qml.qnode(sampling_device)
        def sample(parameters):
            circuit(parameters)
            return qml.sample(wires=range(number_of_candidates))

        sample = qml.set_shots(sample, shots=self.shots)

        return objective, sample

    def plan(self, maze, start, goal):
        """Return a validated path selected by the QAOA circuit."""
        start, goal = self.validate_problem(maze, start, goal)

        if start == goal:
            self.last_metadata = {
                "algorithm": self.name,
                "candidate_paths": 1,
                "quantum_circuit_executions": 0,
                "classical_optimizer": "COBYLA",
                "failure": None,
            }
            return [start]

        paths = self._candidate_paths(
            maze,
            start,
            goal,
            self.max_candidates,
        )

        if not paths:
            self.last_metadata = {
                "algorithm": self.name,
                "candidate_paths": 0,
                "quantum_circuit_executions": 0,
                "failure": "no path found",
            }
            return None

        if len(paths) == 1:
            path = self.validate_path(maze, paths[0], start, goal)
            self.last_metadata = {
                "algorithm": self.name,
                "candidate_paths": 1,
                "selected_candidate": 0,
                "quantum_circuit_executions": 0,
                "classical_optimizer": "not required",
                "failure": None,
            }
            return path

        raw_costs = [float(len(path) - 1) for path in paths]
        minimum = min(raw_costs)
        maximum = max(raw_costs)
        span = max(1.0, maximum - minimum)

        self._normalized_costs = [
            (cost - minimum) / span for cost in raw_costs
        ]

        objective, sample = self._build_qaoa(len(paths))

        initial_parameters = pnp.array(
            [
                [0.1] * self.layers,
                [0.1] * self.layers,
            ],
            requires_grad=False,
        )

        evaluations = 0

        def classical_objective(flat_parameters):
            nonlocal evaluations
            evaluations += 1
            parameters = pnp.array(
                flat_parameters.reshape(2, self.layers),
                requires_grad=False,
            )
            return float(objective(parameters))

        result = minimize(
            classical_objective,
            initial_parameters.reshape(-1),
            method="COBYLA",
            options={"maxiter": self.max_iterations},
        )

        optimized_parameters = pnp.array(
            result.x.reshape(2, self.layers),
            requires_grad=False,
        )

        measured = sample(optimized_parameters)
        bitstrings = [
            tuple(int(bit) for bit in row)
            for row in measured
        ]

        # Only one-hot bitstrings represent valid candidate selections.
        valid_samples = [
            bits for bits in bitstrings if sum(bits) == 1
        ]

        if valid_samples:
            frequencies = Counter(valid_samples)
            selected_bits = max(
                frequencies,
                key=lambda bits: (frequencies[bits], -bits.index(1)),
            )
            selected_index = selected_bits.index(1)
            selection_source = "qaoa_measurement"
        else:
            # Explicit fallback: select the lowest-cost classical candidate.
            selected_index = min(
                range(len(paths)),
                key=lambda index: (raw_costs[index], index),
            )
            selection_source = "classical_fallback"

        selected_path = self.validate_path(
            maze,
            paths[selected_index],
            start,
            goal,
        )

        self.last_metadata = {
            "algorithm": self.name,
            "algorithm_type": "hybrid_quantum_classical",
            "quantum_backend": "PennyLane default.qubit",
            "qaoa_layers": self.layers,
            "qubits": len(paths),
            "shots": self.shots,
            "candidate_paths": len(paths),
            "candidate_path_lengths": [int(cost) for cost in raw_costs],
            "selected_candidate": selected_index,
            "selected_path_length": len(selected_path) - 1,
            "selection_source": selection_source,
            "valid_one_hot_measurements": len(valid_samples),
            "quantum_circuit_executions": self.shots,
            "classical_optimizer": "COBYLA",
            "classical_optimizer_evaluations": evaluations,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "failure": None,
        }

        return selected_path
