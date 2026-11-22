import pytest

from environment.maze import Maze, STATIC_WALL
from planners.quantum.quantum_walk_local_navigation import (
    QuantumWalkLocalNavigationPlanner,
)


@pytest.fixture
def small_maze():
    return Maze(width=5, height=5, seed=42)


@pytest.fixture
def large_maze():
    return Maze(width=9, height=9, seed=42)


@pytest.fixture
def planner():
    return QuantumWalkLocalNavigationPlanner(
        window_size=5,
        coin_type="grover",
        shots=16,
        walk_steps=0,
        max_total_qubits=8,
        seed=42,
        max_segments=20,
    )


def test_local_navigation_returns_valid_global_path(
    planner, small_maze
):
    path = planner.plan(
        small_maze,
        small_maze.start,
        small_maze.goal,
    )

    assert path is not None
    planner.validate_path(
        small_maze,
        path,
        small_maze.start,
        small_maze.goal,
    )

    assert planner.last_metadata["failure"] is None
    assert planner.last_metadata["segments"] >= 1
    assert planner.last_metadata["path_length"] == len(path) - 1


def test_large_maze_requires_multiple_local_segments(
    planner, large_maze
):
    path = planner.plan(
        large_maze,
        large_maze.start,
        large_maze.goal,
    )

    assert path is not None
    planner.validate_path(
        large_maze,
        path,
        large_maze.start,
        large_maze.goal,
    )

    metadata = planner.last_metadata

    assert metadata["failure"] is None
    assert metadata["segments"] > 1
    assert metadata["global_guidance_calls"] == metadata["segments"]


def test_zero_walk_steps_activates_quantum_navigation_fallback(
    planner, small_maze
):
    path = planner.plan(
        small_maze,
        small_maze.start,
        small_maze.goal,
    )

    assert path is not None

    metadata = planner.last_metadata

    assert metadata["failure"] is None
    assert metadata["classical_fallback_decisions"] > 0
    assert metadata["quantum_circuit_executions"] > 0


def test_unreachable_goal_returns_none(planner, small_maze):
    for y in range(1, small_maze.height - 1):
        small_maze.set_cell(2, y, STATIC_WALL)

    path = planner.plan(
        small_maze,
        small_maze.start,
        small_maze.goal,
    )

    assert path is None
    assert planner.last_metadata["failure"] == "global goal unreachable"


def test_start_equal_goal_returns_single_cell_path(
    planner, small_maze
):
    path = planner.plan(
        small_maze,
        small_maze.start,
        small_maze.start,
    )

    assert path == [small_maze.start]
    assert planner.last_metadata["segments"] == 0
    assert planner.last_metadata["path_length"] == 0


def test_maximum_segments_limit_is_respected(
    planner, large_maze
):
    planner.max_segments = 1

    path = planner.plan(
        large_maze,
        large_maze.start,
        large_maze.goal,
    )

    assert path is None
    assert planner.last_metadata["failure"] == "maximum segments reached"


def test_global_path_remains_cardinal_and_walkable(
    planner, large_maze
):
    path = planner.plan(
        large_maze,
        large_maze.start,
        large_maze.goal,
    )

    assert path is not None

    for cell in path:
        assert large_maze.is_walkable(*cell)

    for first, second in zip(path, path[1:]):
        assert abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1


@pytest.mark.parametrize(
    "parameters",
    [
        {"window_size": 2},
        {"window_size": 4},
        {"window_size": 0},
        {"shots": 0},
        {"walk_steps": -1},
        {"max_total_qubits": 2},
        {"max_segments": 0},
        {"coin_type": "invalid"},
    ],
)
def test_invalid_constructor_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        QuantumWalkLocalNavigationPlanner(**parameters)
