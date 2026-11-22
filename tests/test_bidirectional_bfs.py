import pytest

from environment.maze import Maze, STATIC_WALL
from planners.classical.bidirectional_bfs import BidirectionalBFSPlanner
from planners.classical.dijkstra import DijkstraPlanner


def make_open_maze():
    return Maze(width=9, height=9, seed=42)


def assert_valid_path(maze, path, start, goal):
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal

    for x, y in path:
        assert maze.is_walkable(x, y)

    for current, following in zip(path, path[1:]):
        distance = (
            abs(current[0] - following[0])
            + abs(current[1] - following[1])
        )
        assert distance == 1


def test_finds_optimal_path_on_open_grid():
    maze = make_open_maze()
    start = (1, 1)
    goal = (7, 7)

    planner = BidirectionalBFSPlanner()
    path = planner.plan(maze, start, goal)
    reference = DijkstraPlanner().plan(maze, start, goal)

    assert_valid_path(maze, path, start, goal)
    assert len(path) == len(reference)
    assert planner.last_metadata["meeting_point"] is not None


def test_finds_optimal_detour_around_obstacles():
    maze = make_open_maze()
    start = (1, 1)
    goal = (7, 1)

    for y in range(1, 7):
        maze.set_cell(4, y, STATIC_WALL)

    planner = BidirectionalBFSPlanner()
    path = planner.plan(maze, start, goal)
    reference = DijkstraPlanner().plan(maze, start, goal)

    assert_valid_path(maze, path, start, goal)
    assert len(path) == len(reference)


def test_returns_none_when_goal_is_unreachable():
    maze = make_open_maze()
    start = (1, 1)
    goal = (7, 1)

    for y in range(1, 8):
        maze.set_cell(4, y, STATIC_WALL)

    planner = BidirectionalBFSPlanner()

    assert planner.plan(maze, start, goal) is None
    assert planner.last_metadata["meeting_point"] is None


def test_start_equal_goal_returns_single_cell():
    maze = make_open_maze()
    planner = BidirectionalBFSPlanner()

    assert planner.plan(maze, (2, 2), (2, 2)) == [(2, 2)]


@pytest.mark.parametrize(
    ("start", "goal"),
    [
        ((0, 1), (7, 7)),
        ((1, 1), (9, 7)),
        ((3, 3), (7, 7)),
    ],
)
def test_rejects_invalid_or_blocked_endpoints(start, goal):
    maze = make_open_maze()
    maze.set_cell(3, 3, STATIC_WALL)

    planner = BidirectionalBFSPlanner()

    assert planner.plan(maze, start, goal) is None
