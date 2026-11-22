"""Discrete graph adaptation of RRT* for four-connected grid mazes."""

from __future__ import annotations

from collections import deque
import math
import numpy as np


class RRTStarPlanner:
    """Sampling-based RRT* using cardinal grid transitions."""

    def __init__(
        self,
        max_iterations=5000,
        step_size=1.0,
        goal_radius=2.0,
        neighbor_radius=4.0,
        seed=42,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if step_size <= 0 or goal_radius < 0 or neighbor_radius < 1:
            raise ValueError("Invalid RRT* distance parameter")

        self.name = "RRT* (discrete grid)"
        self.max_iterations = int(max_iterations)
        self.step_size = float(step_size)
        self.requested_step_size = float(step_size)
        self.goal_radius = float(goal_radius)
        self.neighbor_radius = float(neighbor_radius)
        self.rng = np.random.default_rng(seed)
        self.last_metadata = {}

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def line_is_free(maze, a, b):
        """Legacy geometric diagnostic; not used for graph expansion."""
        distance = RRTStarPlanner.distance(a, b)
        samples = max(2, int(math.ceil(distance * 4)))

        for i in range(samples + 1):
            t = i / samples
            x = round(a[0] + t * (b[0] - a[0]))
            y = round(a[1] + t * (b[1] - a[1]))
            if not maze.is_walkable(x, y):
                return False

        return True

    @staticmethod
    def _distances_to_sample(maze, sample):
        """BFS whose parent links point from each cell toward sample."""
        queue = deque([sample])
        distances = {sample: 0}
        toward_sample = {sample: None}

        while queue:
            current = queue.popleft()

            for neighbor in maze.get_neighbors(*current):
                if neighbor in distances:
                    continue

                distances[neighbor] = distances[current] + 1
                toward_sample[neighbor] = current
                queue.append(neighbor)

        return distances, toward_sample

    @staticmethod
    def _is_ancestor(parents, ancestor, node):
        current = node

        while current is not None:
            if current == ancestor:
                return True
            current = parents[current]

        return False

    @staticmethod
    def _propagate_costs(node, children, costs):
        """Recompute costs using direct parent-child relationships."""
        queue = deque([node])

        while queue:
            parent = queue.popleft()

            for child in children[parent]:
                costs[child] = costs[parent] + 1.0
                queue.append(child)

    def plan(self, maze, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        self.last_metadata = {
            "algorithm": "RRT* discrete grid",
            "iterations": 0,
            "accepted_nodes": 0,
            "rewires": 0,
            "goal_reached": False,
        }

        if not maze.is_walkable(*start) or not maze.is_walkable(*goal):
            return None

        if start == goal:
            self.last_metadata["goal_reached"] = True
            return [start]

        free_cells = [
            (x, y)
            for y in range(maze.height)
            for x in range(maze.width)
            if maze.is_walkable(x, y)
        ]

        nodes = [start]
        parents = {start: None}
        children = {start: set()}
        costs = {start: 0.0}
        rewires = 0

        for iteration in range(self.max_iterations):
            self.last_metadata["iterations"] = iteration + 1

            # Goal-biased random sampling.
            if self.rng.random() < 0.15:
                sample = goal
            else:
                sample = free_cells[
                    int(self.rng.integers(len(free_cells)))
                ]

            if sample in parents:
                continue

            distances, toward_sample = self._distances_to_sample(
                maze, sample
            )

            reachable = [node for node in nodes if node in distances]
            if not reachable:
                continue

            nearest = min(reachable, key=lambda node: distances[node])
            new_node = toward_sample[nearest]

            if new_node is None or new_node in parents:
                continue

            # Every tree edge must be one valid cardinal transition.
            if new_node not in maze.get_neighbors(*nearest):
                raise RuntimeError("Non-cardinal RRT* extension rejected")

            nearby = [
                node
                for node in maze.get_neighbors(*new_node)
                if node in parents
            ]

            if not nearby:
                continue

            # Choose the lowest-cost feasible adjacent parent.
            best_parent = min(
                nearby,
                key=lambda node: costs[node] + 1.0,
            )

            parents[new_node] = best_parent
            children[new_node] = set()
            children[best_parent].add(new_node)
            costs[new_node] = costs[best_parent] + 1.0
            nodes.append(new_node)
            self.last_metadata["accepted_nodes"] += 1

            # RRT* local rewiring, without introducing parent cycles.
            for neighbor in nearby:
                candidate_cost = costs[new_node] + 1.0

                if (
                    candidate_cost + 1e-12 < costs[neighbor]
                    and not self._is_ancestor(
                        parents, neighbor, new_node
                    )
                ):
                    old_parent = parents[neighbor]
                    children[old_parent].remove(neighbor)
                    parents[neighbor] = new_node
                    children[new_node].add(neighbor)
                    costs[neighbor] = candidate_cost
                    self._propagate_costs(
                        neighbor, children, costs
                    )
                    rewires += 1

        self.last_metadata["rewires"] = rewires
        self.last_metadata["goal_reached"] = goal in parents

        if goal not in parents:
            return None

        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = parents[current]

        path.reverse()

        if any(
            b not in maze.get_neighbors(*a)
            for a, b in zip(path, path[1:])
        ):
            raise RuntimeError(
                "RRT* produced a non-cardinal or blocked path"
            )

        return path
