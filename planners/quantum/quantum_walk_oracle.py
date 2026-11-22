"""
Goal phase oracle for a coined quantum walk.

Every coin state associated with the goal position
receives a phase of -1.
"""

import numpy as np
from qiskit.circuit.library import UnitaryGate


class QuantumWalkOracle:
    """Construct a phase oracle for a target grid position."""

    def __init__(self, encoder, total_qubits):
        self.encoder = encoder
        self.total_qubits = total_qubits
        self.dimension = 2 ** total_qubits

        self.number_of_positions = len(encoder.cell_to_index)

        expected_qubits = max(
            1,
            (self.number_of_positions - 1).bit_length(),
        ) + 2

        if total_qubits != expected_qubits:
            raise ValueError("Inconsistent quantum register dimensions.")

    def build_matrix(self, goal):
        """Build a diagonal phase oracle for the specified goal."""
        if goal not in self.encoder.cell_to_index:
            raise ValueError("Goal cell is not walkable.")

        goal_index = self.encoder.encode_state(goal)

        phases = np.ones(self.dimension, dtype=np.complex128)

        for coin_state in range(4):
            basis_index = goal_index * 4 + coin_state
            phases[basis_index] = -1.0

        return np.diag(phases)

    def to_gate(self, goal):
        """Return the goal phase oracle as a Qiskit gate."""
        return UnitaryGate(
            self.build_matrix(goal),
            label="goal_oracle",
        )
