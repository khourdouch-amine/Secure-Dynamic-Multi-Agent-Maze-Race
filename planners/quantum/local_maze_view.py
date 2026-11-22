"""
Local window adapter for quantum navigation.

The adapter exposes a bounded local view of the global maze
without modifying the original environment.
"""

import numpy as np


class LocalMazeView:
    """Represent a square local window around a global position."""

    def __init__(self, global_maze, center, size=7):
        if not isinstance(size, int) or size < 3 or size % 2 == 0:
            raise ValueError("Window size must be an odd integer >= 3.")

        self.global_maze = global_maze
        self.center = tuple(center)
        self.size = size

        if len(self.center) != 2:
            raise ValueError("Center must contain two coordinates.")

        self.width = size
        self.height = size

        half = size // 2
        self.origin = (
            self.center[0] - half,
            self.center[1] - half,
        )

        self.grid = np.ones((size, size), dtype=np.uint8)

        for local_y in range(size):
            for local_x in range(size):
                global_x, global_y = self.to_global(
                    (local_x, local_y)
                )

                if global_maze.is_inside(global_x, global_y):
                    self.grid[local_y, local_x] = (
                        global_maze.grid[global_y, global_x]
                    )

    def is_inside(self, x, y):
        """Check whether a coordinate belongs to the local window."""
        return 0 <= x < self.width and 0 <= y < self.height

    def to_global(self, cell):
        """Convert local coordinates into global coordinates."""
        x, y = cell
        return (
            self.origin[0] + x,
            self.origin[1] + y,
        )

    def to_local(self, cell):
        """Convert global coordinates into local coordinates."""
        x, y = cell
        return (
            x - self.origin[0],
            y - self.origin[1],
        )

    def is_walkable(self, x, y):
        """Check traversability inside the local window."""
        if not self.is_inside(x, y):
            return False

        return self.grid[y, x] == 0

    def get_neighbors(self, x, y):
        """Return walkable four-connected local neighbors."""
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
