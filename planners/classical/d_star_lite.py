"""
D* Lite incremental path planner for grid environments.
"""

from __future__ import annotations

import heapq
import math
from itertools import count


class DStarLitePlanner:
    """D* Lite planner with incremental vertex updates."""

    def __init__(self, maze, start, goal):
        self.name = "D* Lite"
        self.maze = maze
        self.start = tuple(start)
        self.last_start = tuple(start)
        self.goal = tuple(goal)

        self.km = 0.0
        self.g = {}
        self.rhs = {self.goal: 0.0}

        self.queue = []
        self.queue_keys = {}
        self.sequence = count()

        self._push(self.goal, self._calculate_key(self.goal))
        self._compute_shortest_path()

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_g(self, node):
        return self.g.get(node, math.inf)

    def _get_rhs(self, node):
        return self.rhs.get(node, math.inf)

    def _calculate_key(self, node):
        value = min(self._get_g(node), self._get_rhs(node))

        return (
            value + self._heuristic(self.start, node) + self.km,
            value,
        )

    def _push(self, node, key):
        self.queue_keys[node] = key

        heapq.heappush(
            self.queue,
            (key[0], key[1], next(self.sequence), node),
        )

    def _remove(self, node):
        self.queue_keys.pop(node, None)

    def _top_key(self):
        while self.queue:
            k1, k2, _, node = self.queue[0]

            if node not in self.queue_keys:
                heapq.heappop(self.queue)
                continue

            if (k1, k2) != self.queue_keys[node]:
                heapq.heappop(self.queue)
                continue

            return (k1, k2)

        return (math.inf, math.inf)

    def _pop(self):
        while self.queue:
            k1, k2, _, node = heapq.heappop(self.queue)
            key = (k1, k2)

            if node not in self.queue_keys:
                continue

            if key != self.queue_keys[node]:
                continue

            self.queue_keys.pop(node, None)
            return node, key

        return None, (math.inf, math.inf)

    def _neighbors(self, node):
        if not self.maze.is_walkable(*node):
            return []

        return self.maze.get_neighbors(*node)

    def _cost(self, a, b):
        if (
            self.maze.is_walkable(*a)
            and self.maze.is_walkable(*b)
            and self._heuristic(a, b) == 1
        ):
            return 1.0

        return math.inf

    def _update_vertex(self, node):
        if node != self.goal:
            successors = self._neighbors(node)

            self.rhs[node] = min(
                (
                    self._cost(node, successor) + self._get_g(successor)
                    for successor in successors
                ),
                default=math.inf,
            )

        self._remove(node)

        if self._get_g(node) != self._get_rhs(node):
            self._push(node, self._calculate_key(node))

    def _compute_shortest_path(self):
        while (
            self._top_key() < self._calculate_key(self.start)
            or self._get_rhs(self.start) != self._get_g(self.start)
        ):
            old_key = self._top_key()
            node, new_key = self._pop()

            if node is None:
                break

            if old_key < self._calculate_key(node):
                self._push(node, self._calculate_key(node))

            elif self._get_g(node) > self._get_rhs(node):
                self.g[node] = self._get_rhs(node)

                for predecessor in self._neighbors(node):
                    self._update_vertex(predecessor)

            else:
                self.g[node] = math.inf
                self._update_vertex(node)

                for predecessor in self._neighbors(node):
                    self._update_vertex(predecessor)

    def move_start(self, new_start):
        new_start = tuple(new_start)

        self.km += self._heuristic(self.last_start, new_start)
        self.start = new_start
        self.last_start = new_start

        self._compute_shortest_path()

    def notify_changed_cells(self, changed_cells):
        """Update changed cells and their four adjacent vertices."""
        affected = set()

        for cell in map(tuple, changed_cells):
            x, y = cell
            affected.add(cell)

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y + dy)

                if self.maze.is_inside(*neighbor):
                    affected.add(neighbor)

        for cell in affected:
            self._update_vertex(cell)

        self._compute_shortest_path()

    def plan(self, maze=None, start=None, goal=None):
        if maze is not None and maze is not self.maze:
            raise ValueError("D* Lite is bound to its original maze instance.")

        if goal is not None and tuple(goal) != self.goal:
            raise ValueError("Changing the goal requires reinitialization.")

        if start is not None and tuple(start) != self.start:
            self.move_start(start)

        if not self.maze.is_walkable(*self.start):
            return None

        self._compute_shortest_path()

        if math.isinf(self._get_g(self.start)):
            return None

        path = [self.start]
        current = self.start
        visited = {current}

        while current != self.goal:
            candidates = self._neighbors(current)

            if not candidates:
                return None

            next_node = min(
                candidates,
                key=lambda node: (
                    self._cost(current, node) + self._get_g(node),
                    node,
                ),
            )

            if (
                math.isinf(self._get_g(next_node))
                or next_node in visited
            ):
                return None

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path
