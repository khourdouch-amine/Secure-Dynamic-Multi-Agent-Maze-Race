"""
Coin operator for a discrete-time coined quantum walk.

The two-qubit coin register encodes four cardinal directions:
00: right
01: up
10: left
11: down
"""

from functools import lru_cache

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


class QuantumWalkCoin:
    """Build quantum coin operators for a four-direction walk."""

    NUMBER_OF_COIN_QUBITS = 2
    NUMBER_OF_DIRECTIONS = 4

    DIRECTIONS = {
        0: (1, 0),    # Right
        1: (0, -1),   # Up
        2: (-1, 0),   # Left
        3: (0, 1),    # Down
    }

    @staticmethod
    @lru_cache(maxsize=1)
    def hadamard_coin():
        """
        Apply H tensor H to the two-qubit coin register.

        The resulting operation creates an equal-amplitude
        superposition of the four direction states.
        """
        circuit = QuantumCircuit(2, name="hadamard_coin")

        circuit.h(0)
        circuit.h(1)

        return circuit


    @staticmethod
    @lru_cache(maxsize=1)
    def grover_coin():
        """
        Build the four-dimensional Grover diffusion coin.

        C = 2|s><s| - I, where |s> is the uniform
        superposition of the four cardinal directions.
        """
        uniform = np.full((4, 4), 0.5, dtype=np.complex128)
        matrix = uniform - np.eye(4, dtype=np.complex128)

        circuit = QuantumCircuit(2, name="grover_coin")
        circuit.append(
            UnitaryGate(matrix, label="grover_coin"),
            [0, 1],
        )

        return circuit

    @staticmethod
    def prepare_uniform_coin_state():
        """Prepare an equal superposition of four directions."""
        circuit = QuantumCircuit(2, name="uniform_coin")

        circuit.h(0)
        circuit.h(1)

        return circuit

    @staticmethod
    def direction_from_bits(bitstring):
        """Decode a two-bit measurement into a direction index."""
        normalized = bitstring.replace(" ", "")

        if len(normalized) != 2 or any(bit not in "01" for bit in normalized):
            raise ValueError("Coin measurement must contain exactly two bits.")

        return int(normalized, 2)
