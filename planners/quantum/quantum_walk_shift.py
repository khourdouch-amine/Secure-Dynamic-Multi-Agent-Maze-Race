"""
Reversible shift operator for a four-direction discrete-time
coined quantum walk on a grid.

Coin encoding:
    0: right
    1: left
    2: down
    3: up

The shift is an involutive permutation:
    S|u, d> = |v, opposite(d)>  if the neighbor v is walkable
    S|u, d> = |u, d>            otherwise
"""

from math import ceil, log2

import numpy as np
from qiskit.circuit.library import UnitaryGate

from planners.quantum.grid_encoder import GridGraphEncoder


class QuantumWalkShift:
    """Build a reversible shift operator for a grid maze."""

    NUMBER_OF_COIN_QUBITS = 2
    NUMBER_OF_DIRECTIONS = 4

    DIRECTIONS = {
        0: (1, 0),    # Right
        1: (0, -1),   # Up
        2: (-1, 0),   # Left
        3: (0, 1),    # Down
    }

    OPPOSITE_DIRECTIONS = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }

    def __init__(self, maze):
        self.maze = maze
        self.encoder = GridGraphEncoder(maze)

        self.number_of_positions = len(self.encoder.cell_to_index)

        if self.number_of_positions == 0:
            raise ValueError("The maze contains no walkable positions.")

        self.position_qubits = max(
            1,
            ceil(log2(self.number_of_positions)),
        )

        self.total_qubits = (
            self.position_qubits + self.NUMBER_OF_COIN_QUBITS
        )

        self.dimension = 2 ** self.total_qubits

        self.matrix = self._build_matrix()
        self._gate = UnitaryGate(self.matrix, label="maze_shift")

    def _build_matrix(self):
        """Construct the shift as a permutation matrix."""

        shift = np.zeros(
            (self.dimension, self.dimension),
            dtype=np.float64,
        )

        # Initialize all basis states as identity.
        # This also preserves unused padded position states.
        np.fill_diagonal(shift, 1.0)

        for position_index in range(self.number_of_positions):
            x, y = self.encoder.decode_state(position_index)

            for direction, (dx, dy) in self.DIRECTIONS.items():
                source_index = (
                    position_index * self.NUMBER_OF_DIRECTIONS
                    + direction
                )

                neighbor = (x + dx, y + dy)

                if (
                    self.maze.is_walkable(*neighbor)
                    and neighbor in self.encoder.cell_to_index
                ):
                    target_position = self.encoder.encode_state(neighbor)

                    target_direction = self.OPPOSITE_DIRECTIONS[
                        direction
                    ]

                    target_index = (
                        target_position * self.NUMBER_OF_DIRECTIONS
                        + target_direction
                    )

                    # Replace the identity transition.
                    shift[source_index, source_index] = 0.0
                    shift[target_index, source_index] = 1.0

        return shift

    def to_gate(self):
        """Return the cached Qiskit unitary gate."""
        return self._gate

    def apply_to(self, circuit, qubits):
        """Append the shift gate to a quantum circuit."""
        if len(qubits) != self.total_qubits:
            raise ValueError(
                f"Expected {self.total_qubits} qubits, "
                f"received {len(qubits)}."
            )

        circuit.append(self.to_gate(), qubits)

        return circuit
