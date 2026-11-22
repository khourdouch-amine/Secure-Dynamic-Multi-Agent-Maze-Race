import pytest

from planners.quantum.qaoa_navigation import QAOANavigationPlanner


class MazeFixture:
    def __init__(self, width=3, height=3, blocked=None):
        self.width = width
        self.height = height
        self.blocked = set(blocked or [])

    def is_inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x, y):
        return self.is_inside(x, y) and (x, y) not in self.blocked

    def get_neighbors(self, x, y):
        candidates = (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        )
        return [
            cell for cell in candidates
            if self.is_walkable(*cell)
        ]


def test_qaoa_selects_candidate_with_quantum_measurements():
    maze = MazeFixture()
    planner = QAOANavigationPlanner(
        candidates=3,
        layers=1,
        shots=64,
        max_iterations=10,
        seed=42,
    )

    path = planner.plan(maze, (0, 0), (2, 2))

    assert path is not None
    assert planner.validate_path(maze, path, (0, 0), (2, 2))
    assert planner.last_metadata["candidate_paths"] == 3
    assert planner.last_metadata["selection_source"] == "qaoa_measurement"
    assert planner.last_metadata["quantum_circuit_executions"] == 64
    assert planner.last_metadata["algorithm_type"] == "hybrid_quantum_classical"


def test_qaoa_returns_none_when_no_path_exists():
    maze = MazeFixture(blocked={
        (1, 0), (0, 1), (1, 1), (1, 2), (2, 1)
    })
    planner = QAOANavigationPlanner()

    assert planner.plan(maze, (0, 0), (2, 2)) is None
    assert planner.last_metadata["candidate_paths"] == 0


def test_qaoa_handles_identical_start_and_goal():
    maze = MazeFixture()
    planner = QAOANavigationPlanner()

    path = planner.plan(maze, (1, 1), (1, 1))

    assert path == [(1, 1)]
    assert planner.last_metadata["quantum_circuit_executions"] == 0


def test_qaoa_rejects_invalid_start():
    maze = MazeFixture()
    planner = QAOANavigationPlanner()

    with pytest.raises(ValueError):
        planner.plan(maze, (-1, 0), (2, 2))
