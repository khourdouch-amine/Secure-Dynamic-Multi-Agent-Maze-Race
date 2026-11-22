import pytest

from environment.maze import Maze, STATIC_WALL
from planners.quantum.quantum_walk_navigation import (
    QuantumWalkNavigationPlanner,
)


@pytest.fixture
def maze():
    return Maze(width=5, height=5, seed=42)


@pytest.fixture
def planner(maze):
    return QuantumWalkNavigationPlanner(
        maze,
        coin_type="grover",
        shots=32,
        walk_steps=2,
        seed=42,
        max_total_qubits=8,
    )


def test_quantum_scored_navigation_returns_valid_path(
    planner, maze, monkeypatch
):
    monkeypatch.setattr(
        planner,
        "_measure_goal_probability",
        lambda candidate, goal: 0.5,
    )

    path = planner.plan(maze, maze.start, maze.goal)

    assert path is not None
    planner.validate_path(maze, path, maze.start, maze.goal)

    metadata = planner.last_metadata

    assert metadata["failure"] is None
    assert metadata["quantum_decisions"] == len(path) - 1
    assert metadata["classical_fallback_decisions"] == 0
    assert metadata["total_decisions"] == len(path) - 1


def test_quantum_probability_influences_candidate_selection(
    planner, maze, monkeypatch
):
    measured_candidates = []

    def controlled_probability(candidate, goal):
        measured_candidates.append(candidate)
        return 0.9 if candidate == (2, 1) else 0.1

    monkeypatch.setattr(
        planner,
        "_measure_goal_probability",
        controlled_probability,
    )

    path = planner.plan(maze, maze.start, maze.goal)

    assert path is not None
    assert path[1] == (2, 1)
    assert measured_candidates
    planner.validate_path(maze, path, maze.start, maze.goal)


def test_zero_quantum_probability_activates_classical_fallback(
    planner, maze, monkeypatch
):
    monkeypatch.setattr(
        planner,
        "_measure_goal_probability",
        lambda candidate, goal: 0.0,
    )

    path = planner.plan(maze, maze.start, maze.goal)

    assert path is not None
    planner.validate_path(maze, path, maze.start, maze.goal)

    metadata = planner.last_metadata

    assert metadata["quantum_decisions"] == 0
    assert metadata["classical_fallback_decisions"] == len(path) - 1
    assert metadata["failure"] is None


def test_unreachable_goal_returns_none(planner, maze, monkeypatch):
    for y in range(1, maze.height - 1):
        maze.set_cell(2, y, STATIC_WALL)

    monkeypatch.setattr(
        planner,
        "_measure_goal_probability",
        lambda candidate, goal: 1.0,
    )

    path = planner.plan(maze, maze.start, maze.goal)

    assert path is None
    assert planner.last_metadata["failure"] == "goal is unreachable"
    assert planner.last_metadata["total_decisions"] == 0
    assert planner.last_metadata["quantum_circuit_executions"] == 0


def test_maximum_navigation_steps_is_respected(
    planner, maze, monkeypatch
):
    planner.max_steps = 1

    monkeypatch.setattr(
        planner,
        "_measure_goal_probability",
        lambda candidate, goal: 1.0,
    )

    path = planner.plan(maze, maze.start, maze.goal)

    assert path is None
    assert planner.last_metadata["failure"] == (
        "maximum navigation steps reached"
    )
    assert planner.last_metadata["total_decisions"] == 1


def test_start_equal_goal_returns_single_cell_path(planner, maze, monkeypatch):
    monkeypatch.setattr(
        planner,
        "_measure_goal_probability",
        lambda candidate, goal: pytest.fail(
            "No quantum evaluation should be executed."
        ),
    )

    path = planner.plan(maze, maze.start, maze.start)

    assert path == [maze.start]
    assert planner.last_metadata["path_length"] == 0
    assert planner.last_metadata["total_decisions"] == 0
    assert planner.last_metadata["failure"] is None


def test_planner_rejects_a_different_maze_instance(planner):
    another_maze = Maze(width=5, height=5, seed=42)

    with pytest.raises(ValueError, match="maze supplied at initialization"):
        planner.plan(
            another_maze,
            another_maze.start,
            another_maze.goal,
        )


def test_environment_change_refreshes_quantum_encoder(
    planner, maze, monkeypatch
):
    original_count = planner.walk.number_of_positions

    maze.set_cell(2, 2, STATIC_WALL)

    monkeypatch.setattr(
        planner,
        "_measure_goal_probability",
        lambda candidate, goal: 0.0,
    )

    path = planner.plan(maze, maze.start, maze.goal)

    assert path is not None
    assert planner.walk.number_of_positions == original_count - 1
    assert (2, 2) not in planner.encoder.cell_to_index
    assert planner.last_metadata["failure"] is None


@pytest.mark.parametrize(
    "parameters",
    [
        {"shots": 0},
        {"shots": -1},
        {"walk_steps": -1},
        {"max_steps": 0},
        {"coin_type": "invalid"},
    ],
)
def test_invalid_constructor_parameters_are_rejected(maze, parameters):
    with pytest.raises(ValueError):
        QuantumWalkNavigationPlanner(
            maze,
            max_total_qubits=8,
            **parameters,
        )
