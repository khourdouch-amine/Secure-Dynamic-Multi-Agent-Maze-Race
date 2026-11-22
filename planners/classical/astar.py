"""
A* path planning algorithm.
"""

from __future__ import annotations

import heapq
from itertools import count


class AStarPlanner:
    """Grid-based A* planner using Manhattan distance."""

    def __init__(self):
        self.name = "A*"

    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, maze, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        if not maze.is_walkable(*start):
            return None

        if not maze.is_walkable(*goal):
            return None

        sequence = count()
        frontier = [(self.heuristic(start, goal), next(sequence), start)]

        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, _, current = heapq.heappop(frontier)

            if current == goal:
                path = []
                node = current

                while node is not None:
                    path.append(node)
                    node = came_from[node]

                return list(reversed(path))

            for neighbor in maze.get_neighbors(*current):
                new_cost = cost_so_far[current] + 1

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current

                    priority = new_cost + self.heuristic(neighbor, goal)

                    heapq.heappush(
                        frontier,
                        (priority, next(sequence), neighbor),
                    )

        return None
