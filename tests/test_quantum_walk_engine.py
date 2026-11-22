import numpy as np
import pytest

from qiskit.quantum_info import Operator, Statevector

from environment.maze import Maze, STATIC_WALL
from planners.quantum.quantum_walk import QuantumWalkPlanner


@pytest.fixture
def maze():
    return Maze(width=5, height=5, seed=42)


@pytest.fixture
def planner(maze):
    return QuantumWalkPlanner(maze, max_total_qubits=8)


def test_initialization_encodes_all_walkable_cells(planner, maze):
    assert planner.number_of_positions == 9
    assert planner.number_of_positions == len(
        planner.encoder.cell_to_index
    )
    assert planner.total_qubits == planner.position_qubits + 2


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_circuit_has_expected_qubit_count(planner, maze, steps):
    circuit = planner.build_circuit(
        maze.start,
        steps=steps,
        measure=False,
        coin_type="hadamard",
    )

    assert circuit.num_qubits == planner.total_qubits
    assert circuit.num_clbits == 0


@pytest.mark.parametrize(
    "coin_type",
    ["hadamard", "grover"],
)
def test_complete_walk_circuit_is_unitary(planner, maze, coin_type):
    circuit = planner.build_circuit(
        maze.start,
        steps=2,
        measure=False,
        coin_type=coin_type,
    )

    matrix = Operator(circuit).data
    identity = np.eye(2 ** planner.total_qubits)

    assert np.allclose(matrix.conj().T @ matrix, identity)


def test_zero_step_circuit_preserves_initial_position(planner, maze):
    circuit = planner.build_circuit(
        maze.start,
        steps=0,
        measure=False,
    )

    state = Statevector.from_instruction(circuit)
    probabilities = planner.get_position_probabilities(state.data)

    assert probabilities[maze.start] == pytest.approx(1.0)
    assert sum(probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "coin_type",
    ["hadamard", "grover"],
)
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_walk_probabilities_are_normalized(
    planner,
    maze,
    coin_type,
    steps,
):
    circuit = planner.build_circuit(
        maze.start,
        steps=steps,
        measure=False,
        coin_type=coin_type,
    )

    state = Statevector.from_instruction(circuit)
    probabilities = planner.get_position_probabilities(state.data)

    assert set(probabilities) == set(planner.encoder.cells)
    assert all(value >= -1e-12 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-10)


def test_measured_circuit_has_position_classical_register(planner, maze):
    circuit = planner.build_circuit(
        maze.start,
        steps=2,
        measure=True,
    )

    assert circuit.num_clbits == planner.position_qubits
    assert len(circuit.cregs) == 1
    assert circuit.cregs[0].name == "position"
    assert circuit.cregs[0].size == planner.position_qubits


@pytest.mark.parametrize(
    "steps",
    [-1, 1.5, "2"],
)
def test_invalid_step_count_is_rejected(planner, maze, steps):
    with pytest.raises(ValueError):
        planner.build_circuit(maze.start, steps=steps)


def test_invalid_coin_type_is_rejected(planner, maze):
    with pytest.raises(ValueError):
        planner.build_circuit(
            maze.start,
            steps=1,
            coin_type="invalid",
        )


def test_blocked_start_is_rejected(planner):
    with pytest.raises(ValueError):
        planner.build_circuit(
            (0, 0),
            steps=1,
        )


def test_environment_refresh_rebuilds_quantum_graph(planner, maze):
    original_count = planner.number_of_positions

    maze.set_cell(2, 2, STATIC_WALL)

    assert planner.refresh_environment() is True
    assert planner.number_of_positions == original_count - 1
    assert (2, 2) not in planner.encoder.cell_to_index

    assert planner.refresh_environment() is False


def test_environment_refresh_updates_walkable_neighbors(planner, maze):
    maze.set_cell(2, 1, STATIC_WALL)

    assert planner.refresh_environment() is True

    neighbors = planner.encoder.adjacency[
        planner.encoder.encode_state(maze.start)
    ]

    neighbor_cells = {
        planner.encoder.decode_state(index)
        for index in neighbors
    }

    assert (2, 1) not in neighbor_cells
