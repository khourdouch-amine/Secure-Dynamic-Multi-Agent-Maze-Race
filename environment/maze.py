"""
Grid-based maze environment for Secure Dynamic Multi-Agent Maze Race.
"""

from __future__ import annotations

import numpy as np

from config import settings


FREE = 0
STATIC_WALL = 1
DYNAMIC_WALL = 2
MOVING_OBSTACLE = 3


class Maze:
    """Represent a rectangular grid maze."""

    def __init__(
        self,
        width: int = settings.MAZE_WIDTH,
        height: int = settings.MAZE_HEIGHT,
        seed: int = settings.RANDOM_SEED,
    ) -> None:
        if width < 5 or height < 5:
            raise ValueError("Maze dimensions must be at least 5x5.")

        self.width = width
        self.height = height
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((height, width), dtype=np.uint8)

        self._create_boundaries()

        self.start = (1, 1)
        self.goal = (width - 2, height - 2)

    def _create_boundaries(self) -> None:
        """Create an impassable outer border."""
        self.grid[0, :] = STATIC_WALL
        self.grid[-1, :] = STATIC_WALL
        self.grid[:, 0] = STATIC_WALL
        self.grid[:, -1] = STATIC_WALL

    def generate_static_maze(self) -> None:
        """Generate a connected perfect maze using randomized DFS.

        The generator requires odd dimensions. The constructor itself
        continues to support even dimensions for backward compatibility.
        """
        if self.width % 2 == 0 or self.height % 2 == 0:
            raise ValueError(
                "Static maze generation requires odd width and height."
            )

        # Reset the random generator to guarantee reproducibility.
        self.rng = np.random.default_rng(self.seed)

        # Start with static walls throughout the interior.
        self.grid[1:-1, 1:-1] = STATIC_WALL

        # The start cell is the first node of the DFS traversal.
        start_x, start_y = self.start
        self.grid[start_y, start_x] = FREE

        visited = {(start_x, start_y)}
        stack = [(start_x, start_y)]

        directions = [
            (2, 0),
            (-2, 0),
            (0, 2),
            (0, -2),
        ]

        while stack:
            x, y = stack[-1]

            candidates = [
                (x + dx, y + dy)
                for dx, dy in directions
                if (
                    1 <= x + dx < self.width - 1
                    and 1 <= y + dy < self.height - 1
                    and (x + dx, y + dy) not in visited
                )
            ]

            if not candidates:
                stack.pop()
                continue

            index = int(self.rng.integers(len(candidates)))
            next_x, next_y = candidates[index]

            # Open the intermediate wall and the destination node.
            wall_x = (x + next_x) // 2
            wall_y = (y + next_y) // 2

            self.grid[wall_y, wall_x] = FREE
            self.grid[next_y, next_x] = FREE

            visited.add((next_x, next_y))
            stack.append((next_x, next_y))

    def is_inside(self, x: int, y: int) -> bool:
        """Return True if a cell is inside the grid."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        """Return True if a cell can be traversed."""
        if not self.is_inside(x, y):
            return False

        return self.grid[y, x] == FREE

    def set_cell(self, x: int, y: int, cell_type: int) -> None:
        """Set a cell while protecting the outer boundary."""
        if not self.is_inside(x, y):
            raise IndexError(f"Cell ({x}, {y}) is outside the maze.")

        if cell_type not in (
            FREE,
            STATIC_WALL,
            DYNAMIC_WALL,
            MOVING_OBSTACLE,
        ):
            raise ValueError(f"Unknown cell type: {cell_type}")

        if x in (0, self.width - 1) or y in (0, self.height - 1):
            raise ValueError("The outer boundary cannot be modified.")

        if (x, y) in (self.start, self.goal) and cell_type != FREE:
            raise ValueError("Start and goal cells must remain free.")

        self.grid[y, x] = cell_type

    def reset_dynamic_elements(self) -> None:
        """Remove dynamic walls and moving obstacles."""
        mask = np.isin(self.grid, [DYNAMIC_WALL, MOVING_OBSTACLE])
        self.grid[mask] = FREE

    def get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return traversable four-connected neighboring cells."""
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        return [
            (nx, ny)
            for nx, ny in candidates
            if self.is_walkable(nx, ny)
        ]

    def display_ascii(self) -> str:
        """Return an ASCII representation of the maze."""
        symbols = {
            FREE: ".",
            STATIC_WALL: "#",
            DYNAMIC_WALL: "D",
            MOVING_OBSTACLE: "O",
        }

        rows = []

        for y in range(self.height):
            row = "".join(symbols[int(value)] for value in self.grid[y])

            row = list(row)
            if y == self.start[1]:
                row[self.start[0]] = "S"
            if y == self.goal[1]:
                row[self.goal[0]] = "G"

            rows.append("".join(row))

        return "\n".join(rows)
