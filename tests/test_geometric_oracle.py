import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from planners.quantum.grover_global.geometric_oracle import (
    _compute_invalid_flag,
    _uncompute_invalid_flag,
)


def _run_invalid_flag(x_value, y_value, width, height, obstacles):
    nx = max(1, width.bit_length())
    ny = max(1, height.bit_length())

    x = QuantumRegister(nx, "x")
    y = QuantumRegister(ny, "y")
    flag = QuantumRegister(1, "flag")

    qc = QuantumCircuit(x, y, flag)

    for bit in range(nx):
        if (x_value >> bit) & 1:
            qc.x(x[bit])

    for bit in range(ny):
        if (y_value >> bit) & 1:
            qc.x(y[bit])

    _compute_invalid_flag(
        qc, list(x), list(y), flag[0],
        width, height, obstacles,
    )

    state = Statevector.from_instruction(qc)
    probabilities = np.abs(state.data) ** 2
    index = int(np.argmax(probabilities))

    return index, nx, ny


def test_invalid_flag_exhaustive_on_power_of_two_grid():
    width = 4
    height = 4
    obstacles = {(1, 0), (2, 1)}

    for x_value in range(8):
        for y_value in range(8):
            index, nx, ny = _run_invalid_flag(
                x_value, y_value, width, height, obstacles
            )

            actual_x = index & ((1 << nx) - 1)
            actual_y = (index >> nx) & ((1 << ny) - 1)
            actual_flag = (index >> (nx + ny)) & 1

            expected_flag = int(
                x_value >= width
                or y_value >= height
                or (x_value, y_value) in obstacles
            )

            assert actual_x == x_value
            assert actual_y == y_value
            assert actual_flag == expected_flag


def test_invalid_flag_uncomputation_restores_all_registers():
    x = QuantumRegister(3, "x")
    y = QuantumRegister(3, "y")
    flag = QuantumRegister(1, "flag")

    qc = QuantumCircuit(x, y, flag)

    qc.x(x[0])
    qc.x(y[1])

    _compute_invalid_flag(
        qc, list(x), list(y), flag[0],
        4, 4, {(1, 0), (2, 1)},
    )

    _uncompute_invalid_flag(
        qc, list(x), list(y), flag[0],
        4, 4, {(1, 0), (2, 1)},
    )

    state = Statevector.from_instruction(qc)

    assert np.isclose(np.abs(state.data[17]) ** 2, 1.0)


def test_invalid_flag_exhaustive_on_non_power_of_two_grid():
    width = 3
    height = 5
    obstacles = {(1, 2), (2, 3)}

    for x_value in range(1 << width.bit_length()):
        for y_value in range(1 << height.bit_length()):
            index, nx, ny = _run_invalid_flag(
                x_value, y_value, width, height, obstacles
            )

            actual_flag = (index >> (nx + ny)) & 1

            expected_flag = int(
                x_value >= width
                or y_value >= height
                or (x_value, y_value) in obstacles
            )

            assert actual_flag == expected_flag
