import pytest

from planners.quantum.vqc_navigation import VQCNavigationPlanner


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
        return [cell for cell in candidates if self.is_walkable(*cell)]


def test_vqc_navigates_without_classical_fallback():
    maze = MazeFixture()
    planner = VQCNavigationPlanner(
        layers=1,
        shots=32,
        max_iterations=60,
        max_training_samples=8,
        seed=42,
    )

    path = planner.plan(maze, (0, 0), (2, 2))

    assert path is not None
    assert planner.validate_path(maze, path, (0, 0), (2, 2))
    assert planner.last_metadata["path_source"] == "vqc_policy"
    assert planner.last_metadata["fallback_used"] is False
    assert planner.last_metadata["algorithm_type"] == "hybrid_quantum_classical"
    assert planner.last_metadata["training_samples"] == 8
    assert planner.last_metadata["optimizer_evaluations"] == 60


def test_vqc_returns_none_when_no_path_exists():
    maze = MazeFixture(blocked={
        (1, 0), (0, 1), (1, 1), (1, 2), (2, 1)
    })
    planner = VQCNavigationPlanner()

    assert planner.plan(maze, (0, 0), (2, 2)) is None
    assert planner.last_metadata["path_source"] == "no_path"


def test_vqc_handles_identical_start_and_goal():
    maze = MazeFixture()
    planner = VQCNavigationPlanner()

    path = planner.plan(maze, (1, 1), (1, 1))

    assert path == [(1, 1)]
    assert planner.last_metadata["path_source"] == "trivial"


def test_vqc_rejects_invalid_start():
    maze = MazeFixture()
    planner = VQCNavigationPlanner()

    with pytest.raises(ValueError):
        planner.plan(maze, (-1, 0), (2, 2))
