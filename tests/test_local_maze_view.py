import pytest

from environment.maze import Maze, STATIC_WALL
from planners.quantum.local_maze_view import LocalMazeView


@pytest.fixture
def maze():
    return Maze(width=9, height=9, seed=42)


def test_local_view_has_expected_dimensions(maze):
    view = LocalMazeView(maze, center=(4, 4), size=5)

    assert view.width == 5
    assert view.height == 5
    assert view.grid.shape == (5, 5)


def test_local_global_coordinate_conversion_is_reversible(maze):
    view = LocalMazeView(maze, center=(4, 4), size=5)

    for y in range(view.height):
        for x in range(view.width):
            local_cell = (x, y)
            global_cell = view.to_global(local_cell)

            assert view.to_local(global_cell) == local_cell


def test_window_origin_is_correct(maze):
    view = LocalMazeView(maze, center=(4, 4), size=5)

    assert view.origin == (2, 2)
    assert view.to_global((0, 0)) == (2, 2)
    assert view.to_global((4, 4)) == (6, 6)


def test_cells_outside_global_maze_are_blocked(maze):
    view = LocalMazeView(maze, center=(1, 1), size=5)

    for y in range(view.height):
        for x in range(view.width):
            global_x, global_y = view.to_global((x, y))

            if not maze.is_inside(global_x, global_y):
                assert not view.is_walkable(x, y)


def test_global_obstacles_are_preserved_in_local_view(maze):
    maze.set_cell(4, 4, STATIC_WALL)

    view = LocalMazeView(maze, center=(4, 4), size=5)

    local_obstacle = view.to_local((4, 4))

    assert not view.is_walkable(*local_obstacle)


def test_local_view_does_not_modify_global_maze(maze):
    original_grid = maze.grid.copy()

    LocalMazeView(maze, center=(4, 4), size=5)

    assert (maze.grid == original_grid).all()


def test_neighbors_are_walkable_and_cardinal(maze):
    view = LocalMazeView(maze, center=(4, 4), size=5)

    neighbors = view.get_neighbors(2, 2)

    assert neighbors

    for nx, ny in neighbors:
        assert view.is_walkable(nx, ny)
        assert abs(nx - 2) + abs(ny - 2) == 1


@pytest.mark.parametrize("size", [2, 4, 0, -3])
def test_invalid_window_size_is_rejected(maze, size):
    with pytest.raises(ValueError):
        LocalMazeView(maze, center=(4, 4), size=size)


def test_outside_local_coordinates_are_not_walkable(maze):
    view = LocalMazeView(maze, center=(4, 4), size=5)

    assert not view.is_walkable(-1, 2)
    assert not view.is_walkable(5, 2)
    assert not view.is_walkable(2, -1)
    assert not view.is_walkable(2, 5)


def test_invalid_center_length_is_rejected(maze):
    with pytest.raises(ValueError):
        LocalMazeView(maze, center=(4,), size=5)
