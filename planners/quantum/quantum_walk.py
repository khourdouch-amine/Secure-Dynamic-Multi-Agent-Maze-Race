"""
Discrete-time coined quantum walk circuit for a static grid maze.

One evolution step:
    U = S (C tensor I)

The coin register occupies the two least-significant qubits.
The position register occupies the remaining qubits.
"""

from math import ceil, log2

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister

from planners.quantum.grid_encoder import GridGraphEncoder
from planners.quantum.quantum_walk_coin import QuantumWalkCoin
from planners.quantum.quantum_walk_shift import QuantumWalkShift


class QuantumWalkPlanner:
    """Build discrete-time coined quantum walk circuits."""

    MAX_TOTAL_QUBITS = 8

    def __init__(self, maze, max_total_qubits=None):
        self.maze = maze
        self.encoder = GridGraphEncoder(maze)
        self.coin = QuantumWalkCoin()

        self.number_of_positions = len(self.encoder.cell_to_index)

        if self.number_of_positions == 0:
            raise ValueError("The maze contains no walkable positions.")

        self.position_qubits = max(
            1,
            ceil(log2(self.number_of_positions)),
        )

        self.total_qubits = self.position_qubits + 2

        limit = (
            self.MAX_TOTAL_QUBITS
            if max_total_qubits is None
            else max_total_qubits
        )

        self.max_total_qubits = limit

        if self.total_qubits > limit:
            raise ValueError(
                f"This simulator configuration requires "
                f"{self.total_qubits} qubits; the configured limit is "
                f"{limit}. Use a smaller maze."
            )

        self.shift = QuantumWalkShift(maze)

        if self.shift.total_qubits != self.total_qubits:
            raise RuntimeError("Inconsistent quantum register dimensions.")

        self.last_circuit = None
        self._maze_snapshot = maze.grid.copy()

    def refresh_environment(self):
        """
        Rebuild the quantum graph and shift operator if the maze changed.

        Returns:
            True if the quantum representation was rebuilt, False otherwise.
        """
        if np.array_equal(self._maze_snapshot, self.maze.grid):
            return False

        refreshed = QuantumWalkPlanner(
            self.maze,
            max_total_qubits=self.max_total_qubits,
        )

        self.__dict__.update(refreshed.__dict__)

        return True

    def build_circuit(
        self,
        start,
        steps=1,
        measure=False,
        coin_type="hadamard",
    ):
        """
        Build a coined quantum walk circuit.

        Args:
            start: Initial grid coordinate.
            steps: Number of discrete-time walk steps.
            measure: Whether to measure the position register.

        Returns:
            A Qiskit QuantumCircuit.
        """
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer.")

        if coin_type not in {"hadamard", "grover"}:
            raise ValueError(
                "coin_type must be either 'hadamard' or 'grover'."
            )

        if start not in self.encoder.cell_to_index:
            raise ValueError("The starting cell is not walkable.")

        circuit = QuantumCircuit(self.total_qubits)

        position_index = self.encoder.encode_state(start)

        # Position register starts in the requested computational basis state.
        for bit in range(self.position_qubits):
            if (position_index >> bit) & 1:
                circuit.x(2 + bit)

        coin_qubits = [0, 1]
        all_qubits = list(range(self.total_qubits))

        for _ in range(steps):
            # Coin operation C.
            coin_circuit = (
                self.coin.hadamard_coin()
                if coin_type == "hadamard"
                else self.coin.grover_coin()
            )

            circuit.compose(
                coin_circuit,
                qubits=coin_qubits,
                inplace=True,
            )

            # Shift operation S.
            self.shift.apply_to(circuit, all_qubits)

        if measure:
            position_register = ClassicalRegister(
                self.position_qubits,
                name="position",
            )
            circuit.add_register(position_register)

            for bit in range(self.position_qubits):
                circuit.measure(
                    2 + bit,
                    position_register[bit],
                )

        self.last_circuit = circuit
        return circuit

    def get_position_probabilities(self, statevector):
        """Aggregate statevector probabilities over coin states."""
        probabilities = {}

        for position_index in range(self.number_of_positions):
            probability = 0.0

            for coin_state in range(4):
                basis_index = position_index * 4 + coin_state
                probability += abs(statevector[basis_index]) ** 2

            cell = self.encoder.decode_state(position_index)
            probabilities[cell] = float(probability)

        return probabilities
