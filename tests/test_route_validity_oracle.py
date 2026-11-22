import pytest

from environment.maze import Maze
from planners.quantum.route_encoding import RouteEncoding
from planners.quantum.route_validity_oracle import RouteValidityOracle


@pytest.fixture
def oracle():
    maze = Maze(width=5, height=5, seed=42)

    return RouteValidityOracle(
        maze=maze,
        start=(1, 1),
        goal=(3, 1),
        route_length=4,
    )


def test_reference_oracle_finds_eight_valid_routes(oracle):
    valid_indices = oracle.get_valid_indices()

    assert len(valid_indices) == 8


def test_known_valid_route(oracle):
    sequence = (0, 3, 1, 0)

    assert oracle.is_valid(sequence)


def test_route_reaching_wrong_destination_is_invalid(oracle):
    sequence = (0, 0, 0, 0)

    assert not oracle.is_valid(sequence)


def test_route_leaving_maze_is_invalid(oracle):
    sequence = (2, 2, 3, 0)

    assert not oracle.is_valid(sequence)


def test_wrong_route_length_is_invalid(oracle):
    assert not oracle.is_valid((0, 0, 2))


def test_index_evaluation_matches_sequence_evaluation(oracle):
    sequence = (0, 3, 1, 0)
    index = RouteEncoding.encode(sequence)

    assert oracle.is_valid_index(index)
    assert oracle.is_valid(sequence)


def test_blocked_cell_invalidates_route():
    maze = Maze(width=5, height=5, seed=42)
    maze.set_cell(2, 1, 1)

    reference = RouteValidityOracle(
        maze=maze,
        start=(1, 1),
        goal=(3, 1),
        route_length=4,
    )

    assert not reference.is_valid((0, 3, 1, 0))


def test_invalid_start_is_rejected():
    maze = Maze(width=5, height=5, seed=42)
    maze.set_cell(2, 1, 1)

    with pytest.raises(ValueError):
        RouteValidityOracle(
            maze=maze,
            start=(2, 1),
            goal=(3, 1),
            route_length=4,
        )
