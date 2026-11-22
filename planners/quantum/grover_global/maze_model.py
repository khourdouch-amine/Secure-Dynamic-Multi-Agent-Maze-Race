"""Parameterized grid model for global Grover route search."""

from dataclasses import dataclass
from typing import Iterable

from planners.quantum.route_encoding import RouteEncoding


@dataclass(frozen=True)
class GroverMazeModel:
    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]

    def __init__(self, width, height, start, goal, obstacles=()):
        if width < 2 or height < 2:
            raise ValueError("Maze dimensions must be at least 2.")

        start = tuple(start)
        goal = tuple(goal)
        obstacles = frozenset(tuple(p) for p in obstacles)

        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "obstacles", obstacles)

        for name, position in (("start", start), ("goal", goal)):
            if len(position) != 2:
                raise ValueError(f"{name} must contain two coordinates.")
            if not self.is_inside(*position):
                raise ValueError(f"{name} is outside the maze.")
            if position in obstacles:
                raise ValueError(f"{name} cannot be an obstacle.")

        for position in obstacles:
            if not self.is_inside(*position):
                raise ValueError(f"Obstacle {position} is outside the maze.")

    def is_inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x, y):
        return self.is_inside(x, y) and (x, y) not in self.obstacles

    def evaluate_route(self, sequence):
        """Return (valid, path, collision_step)."""
        if not sequence:
            return False, [self.start], None

        x, y = self.start
        path = [(x, y)]

        for step, direction in enumerate(sequence, start=1):
            if direction not in RouteEncoding.DIRECTIONS:
                return False, path, step

            dx, dy = RouteEncoding.DIRECTIONS[direction]
            x += dx
            y += dy
            path.append((x, y))

            if not self.is_walkable(x, y):
                return False, path, step

        return (x, y) == self.goal, path, None

    def valid_directions_from(self, position):
        """Return legal movement codes from a position."""
        x, y = position
        return tuple(
            direction
            for direction, (dx, dy) in RouteEncoding.DIRECTIONS.items()
            if self.is_walkable(x + dx, y + dy)
        )
