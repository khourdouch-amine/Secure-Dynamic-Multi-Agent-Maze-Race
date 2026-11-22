import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from environment.maze import Maze, STATIC_WALL
from planners.quantum.quantum_walk_shift import QuantumWalkShift
from planners.quantum.quantum_walk_coin import QuantumWalkCoin


def make_open_maze():
    return Maze(width=5, height=5, seed=42)


def make_obstacle_maze():
    maze = make_open_maze()
    maze.set_cell(2, 2, STATIC_WALL)
    return maze


def make_corridor_maze():
    maze = make_open_maze()

    allowed = {
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (3, 3),
    }

    for y in range(1, 4):
        for x in range(1, 4):
            if (x, y) not in allowed:
                maze.set_cell(x, y, STATIC_WALL)

    return maze


@pytest.mark.parametrize(
    "maze_factory",
    [
        make_open_maze,
        make_obstacle_maze,
        make_corridor_maze,
    ],
)
def test_shift_is_a_permutation_unitary_and_involution(maze_factory):
    shift = QuantumWalkShift(maze_factory())
    matrix = shift.matrix
    identity = np.eye(shift.dimension)

    assert np.all((matrix == 0) | (matrix == 1))
    assert np.all(matrix.sum(axis=0) == 1)
    assert np.all(matrix.sum(axis=1) == 1)

    assert np.allclose(matrix.T @ matrix, identity)
    assert np.allclose(matrix @ matrix, identity)


def test_shift_gate_matches_its_matrix():
    shift = QuantumWalkShift(make_open_maze())

    gate_matrix = Operator(shift.to_gate()).data

    assert np.allclose(gate_matrix, shift.matrix)


def test_shift_apply_to_rejects_wrong_qubit_count():
    shift = QuantumWalkShift(make_open_maze())
    circuit = QuantumCircuit(shift.total_qubits)

    with pytest.raises(ValueError):
        shift.apply_to(circuit, list(range(shift.total_qubits - 1)))


def test_hadamard_coin_is_unitary():
    matrix = Operator(QuantumWalkCoin.hadamard_coin()).data
    identity = np.eye(4)

    assert np.allclose(matrix.conj().T @ matrix, identity)


def test_grover_coin_is_unitary_hermitian_and_involutive():
    matrix = Operator(QuantumWalkCoin.grover_coin()).data
    identity = np.eye(4)

    assert np.allclose(matrix.conj().T @ matrix, identity)
    assert np.allclose(matrix, matrix.conj().T)
    assert np.allclose(matrix @ matrix, identity)


def test_uniform_coin_preparation_creates_equal_probabilities():
    circuit = QuantumWalkCoin.prepare_uniform_coin_state()
    state = Operator(circuit).data[:, 0]
    probabilities = np.abs(state) ** 2

    assert np.allclose(probabilities, np.full(4, 0.25))


def test_direction_bitstring_decoding():
    expected = {
        "00": 0,
        "01": 1,
        "10": 2,
        "11": 3,
    }

    for bitstring, direction in expected.items():
        assert QuantumWalkCoin.direction_from_bits(bitstring) == direction


@pytest.mark.parametrize(
    "bitstring",
    ["0", "000", "2a", "", "0 1 0"],
)
def test_invalid_direction_bitstrings_are_rejected(bitstring):
    with pytest.raises(ValueError):
        QuantumWalkCoin.direction_from_bits(bitstring)
