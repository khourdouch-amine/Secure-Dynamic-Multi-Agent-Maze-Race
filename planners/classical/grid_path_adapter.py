"""
Convert RRT* waypoint paths into valid four-connected grid paths.
"""

from collections import deque


class GridPathAdapter:
    """Discretize continuous planner waypoints into cardinal grid moves."""

    def __init__(self, margin=2):
        if margin < 0:
            raise ValueError("margin must be non-negative")

        self.margin = int(margin)

    @staticmethod
    def _is_cardinal_path(path):
        return all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            for a, b in zip(path, path[1:])
        )

    def _connect_segment(self, maze, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        if not maze.is_walkable(*start):
            return None

        if not maze.is_walkable(*goal):
            return None

        if start == goal:
            return [start]

        min_x = min(start[0], goal[0]) - self.margin
        max_x = max(start[0], goal[0]) + self.margin
        min_y = min(start[1], goal[1]) - self.margin
        max_y = max(start[1], goal[1]) + self.margin

        queue = deque([start])
        parents = {start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for neighbor in maze.get_neighbors(*current):
                x, y = neighbor

                if not (
                    min_x <= x <= max_x
                    and min_y <= y <= max_y
                ):
                    continue

                if neighbor in parents:
                    continue

                parents[neighbor] = current
                queue.append(neighbor)

        if goal not in parents:
            return None

        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = parents[current]

        path.reverse()
        return path

    def adapt(self, maze, waypoint_path):
        """
        Convert a waypoint path into a four-connected grid path.

        Returns None if any segment cannot be connected locally.
        """
        if not waypoint_path:
            return None

        waypoints = [tuple(point) for point in waypoint_path]

        if any(
            len(point) != 2
            or any(not isinstance(v, int) for v in point)
            for point in waypoints
        ):
            raise ValueError("Waypoints must contain integer grid coordinates.")

        if any(not maze.is_walkable(*point) for point in waypoints):
            return None

        if len(waypoints) == 1:
            return waypoints.copy()

        result = [waypoints[0]]

        for start, goal in zip(waypoints, waypoints[1:]):
            segment = self._connect_segment(maze, start, goal)

            if segment is None:
                return None

            result.extend(segment[1:])

        if not self._is_cardinal_path(result):
            raise RuntimeError("Adapter produced a non-cardinal path.")

        if any(not maze.is_walkable(*cell) for cell in result):
            raise RuntimeError("Adapter produced a path through an obstacle.")

        return result
