import numpy as np
import pytest

from environment.maze import (
    Maze,
    FREE,
    STATIC_WALL,
    DYNAMIC_WALL,
    MOVING_OBSTACLE,
)


def reachable_cells(maze, start):
    """Return all cells reachable from start using four-connectivity."""
    visited = {start}
    queue = [start]

    while queue:
        x, y = queue.pop(0)

        for neighbor in maze.get_neighbors(x, y):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited


def test_default_maze_preserves_legacy_behavior():
    maze = Maze(width=25, height=25, seed=42)

    assert maze.grid.shape == (25, 25)
    assert maze.start == (1, 1)
    assert maze.goal == (23, 23)
    assert np.all(maze.grid[1:-1, 1:-1] == FREE)


def test_maze_boundaries_are_static_walls():
    maze = Maze(width=25, height=25, seed=42)

    assert np.all(maze.grid[0, :] == STATIC_WALL)
    assert np.all(maze.grid[-1, :] == STATIC_WALL)
    assert np.all(maze.grid[:, 0] == STATIC_WALL)
    assert np.all(maze.grid[:, -1] == STATIC_WALL)


def test_start_and_goal_are_walkable():
    maze = Maze(width=25, height=25, seed=42)

    assert maze.is_walkable(*maze.start)
    assert maze.is_walkable(*maze.goal)


def test_custom_dimensions_are_preserved():
    maze = Maze(width=9, height=9, seed=42)

    assert maze.grid.shape == (9, 9)
    assert maze.width == 9
    assert maze.height == 9
    assert maze.start == (1, 1)
    assert maze.goal == (7, 7)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (4, 25),
        (25, 4),
        (0, 25),
        (25, 0),
        (-5, 25),
        (25, -5),
    ],
)
def test_invalid_dimensions_are_rejected(width, height):
    with pytest.raises(ValueError):
        Maze(width=width, height=height)


def test_outside_cells_are_not_walkable():
    maze = Maze(width=9, height=9, seed=42)

    assert not maze.is_walkable(-1, 1)
    assert not maze.is_walkable(1, -1)
    assert not maze.is_walkable(9, 1)
    assert not maze.is_walkable(1, 9)


def test_set_cell_protects_outer_boundary():
    maze = Maze(width=9, height=9, seed=42)

    with pytest.raises(ValueError):
        maze.set_cell(0, 1, FREE)

    with pytest.raises(ValueError):
        maze.set_cell(8, 1, FREE)


def test_set_cell_protects_start_and_goal():
    maze = Maze(width=9, height=9, seed=42)

    with pytest.raises(ValueError):
        maze.set_cell(*maze.start, STATIC_WALL)

    with pytest.raises(ValueError):
        maze.set_cell(*maze.goal, STATIC_WALL)


def test_set_cell_rejects_unknown_cell_type():
    maze = Maze(width=9, height=9, seed=42)

    with pytest.raises(ValueError):
        maze.set_cell(2, 2, 99)


def test_set_cell_rejects_outside_coordinates():
    maze = Maze(width=9, height=9, seed=42)

    with pytest.raises(IndexError):
        maze.set_cell(-1, 2, STATIC_WALL)


def test_get_neighbors_returns_only_four_connected_free_cells():
    maze = Maze(width=9, height=9, seed=42)

    maze.set_cell(3, 2, STATIC_WALL)

    neighbors = set(maze.get_neighbors(2, 2))

    assert neighbors == {
        (1, 2),
        (2, 1),
        (2, 3),
    }


def test_dynamic_elements_can_be_reset():
    maze = Maze(width=9, height=9, seed=42)

    maze.set_cell(2, 2, DYNAMIC_WALL)
    maze.set_cell(3, 3, MOVING_OBSTACLE)
    maze.set_cell(4, 4, STATIC_WALL)

    maze.reset_dynamic_elements()

    assert maze.grid[2, 2] == FREE
    assert maze.grid[3, 3] == FREE
    assert maze.grid[4, 4] == STATIC_WALL


def test_ascii_display_contains_start_goal_and_boundaries():
    maze = Maze(width=9, height=9, seed=42)

    display = maze.display_ascii()
    rows = display.splitlines()

    assert len(rows) == 9
    assert all(len(row) == 9 for row in rows)
    assert "S" in display
    assert "G" in display
    assert "#" in display


@pytest.mark.parametrize("size", [5, 7, 9, 25])
def test_static_generation_supports_odd_square_dimensions(size):
    maze = Maze(width=size, height=size, seed=42)

    maze.generate_static_maze()

    assert maze.grid.shape == (size, size)
    assert maze.is_walkable(*maze.start)
    assert maze.is_walkable(*maze.goal)


def test_static_generation_creates_interior_walls():
    maze = Maze(width=25, height=25, seed=42)

    maze.generate_static_maze()

    interior = maze.grid[1:-1, 1:-1]

    assert np.any(interior == STATIC_WALL)
    assert np.any(interior == FREE)


def test_static_generation_preserves_all_boundaries():
    maze = Maze(width=25, height=25, seed=42)

    maze.generate_static_maze()

    assert np.all(maze.grid[0, :] == STATIC_WALL)
    assert np.all(maze.grid[-1, :] == STATIC_WALL)
    assert np.all(maze.grid[:, 0] == STATIC_WALL)
    assert np.all(maze.grid[:, -1] == STATIC_WALL)


def test_static_generation_is_reproducible_with_same_seed():
    first = Maze(width=25, height=25, seed=42)
    second = Maze(width=25, height=25, seed=42)

    first.generate_static_maze()
    second.generate_static_maze()

    assert np.array_equal(first.grid, second.grid)


def test_static_generation_changes_with_different_seeds():
    first = Maze(width=25, height=25, seed=42)
    second = Maze(width=25, height=25, seed=43)

    first.generate_static_maze()
    second.generate_static_maze()

    assert not np.array_equal(first.grid, second.grid)


def test_static_generation_connects_all_walkable_cells():
    maze = Maze(width=25, height=25, seed=42)

    maze.generate_static_maze()

    reachable = reachable_cells(maze, maze.start)

    walkable = {
        (x, y)
        for y in range(maze.height)
        for x in range(maze.width)
        if maze.is_walkable(x, y)
    }

    assert reachable == walkable
    assert maze.goal in reachable


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (6, 7),
        (7, 6),
        (8, 8),
        (25, 24),
    ],
)
def test_static_generation_rejects_even_dimensions(width, height):
    maze = Maze(width=width, height=height, seed=42)

    with pytest.raises(ValueError):
        maze.generate_static_maze()
