"""
Common interface and validation utilities for quantum navigation planners.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter


class QuantumPlannerBase(ABC):
    """Base class for quantum and hybrid quantum-classical planners."""

    def __init__(self, name: str):
        self.name = name
        self.last_execution_time = 0.0
        self.last_success = False
        self.last_metadata = {}

    @abstractmethod
    def plan(self, maze, start, goal):
        """Compute a navigation path using the implemented algorithm."""
        raise NotImplementedError

    @staticmethod
    def validate_problem(maze, start, goal):
        """Validate start and goal cells."""
        start = tuple(start)
        goal = tuple(goal)

        if len(start) != 2 or len(goal) != 2:
            raise ValueError("Start and goal must be 2D coordinates.")

        if not maze.is_inside(*start):
            raise ValueError(f"Start is outside the maze: {start}")

        if not maze.is_inside(*goal):
            raise ValueError(f"Goal is outside the maze: {goal}")

        if not maze.is_walkable(*start):
            raise ValueError(f"Start is not walkable: {start}")

        if not maze.is_walkable(*goal):
            raise ValueError(f"Goal is not walkable: {goal}")

        return start, goal

    @staticmethod
    def validate_path(maze, path, start, goal):
        """Validate a cardinal, collision-free path."""
        if path is None:
            return None

        path = [tuple(point) for point in path]

        if not path:
            return None

        if path[0] != tuple(start):
            raise ValueError("Path does not start at the requested start.")

        if path[-1] != tuple(goal):
            raise ValueError("Path does not reach the requested goal.")

        for point in path:
            if not maze.is_inside(*point):
                raise ValueError(f"Path leaves maze boundaries: {point}")

            if not maze.is_walkable(*point):
                raise ValueError(f"Path crosses a blocked cell: {point}")

        for current, following in zip(path, path[1:]):
            distance = (
                abs(current[0] - following[0])
                + abs(current[1] - following[1])
            )

            if distance != 1:
                raise ValueError(
                    f"Non-cardinal or discontinuous movement: "
                    f"{current} -> {following}"
                )

        return path

    def execute(self, maze, start, goal):
        """Execute and time the planner with common validation."""
        start_time = perf_counter()
        self.last_success = False
        self.last_metadata = {}

        try:
            start, goal = self.validate_problem(maze, start, goal)
            path = self.plan(maze, start, goal)
            path = self.validate_path(maze, path, start, goal)

            self.last_success = path is not None
            return path

        finally:
            self.last_execution_time = perf_counter() - start_time
