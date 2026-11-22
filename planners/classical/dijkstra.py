"""
Dijkstra shortest-path planning algorithm.
"""

from __future__ import annotations

import heapq
from itertools import count


class DijkstraPlanner:
    """Grid-based Dijkstra planner with uniform movement costs."""

    def __init__(self):
        self.name = "Dijkstra"

    def plan(self, maze, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        if not maze.is_walkable(*start):
            return None

        if not maze.is_walkable(*goal):
            return None

        sequence = count()
        frontier = [(0, next(sequence), start)]

        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current_cost, _, current = heapq.heappop(frontier)

            if current_cost != cost_so_far[current]:
                continue

            if current == goal:
                path = []
                node = current

                while node is not None:
                    path.append(node)
                    node = came_from[node]

                return list(reversed(path))

            for neighbor in maze.get_neighbors(*current):
                new_cost = current_cost + 1

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current

                    heapq.heappush(
                        frontier,
                        (new_cost, next(sequence), neighbor),
                    )

        return None
