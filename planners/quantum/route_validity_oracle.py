"""Classical reference oracle for fixed-length maze routes."""

from planners.quantum.route_encoding import RouteEncoding


class RouteValidityOracle:
    """
    Evaluate fixed-length routes against a maze.

    A route is valid if:
      1. It starts at the configured start position.
      2. Every movement remains inside walkable cells.
      3. It reaches the configured goal after exactly route_length moves.

    Repeated positions are currently allowed.
    """

    def __init__(self, maze, start, goal, route_length):
        if not isinstance(route_length, int) or isinstance(route_length, bool):
            raise TypeError("route_length must be an integer.")

        if route_length < 1:
            raise ValueError("route_length must be positive.")

        if not maze.is_inside(*start) or not maze.is_inside(*goal):
            raise ValueError("Start and goal must be inside the maze.")

        if not maze.is_walkable(*start):
            raise ValueError("Start position must be walkable.")

        if not maze.is_walkable(*goal):
            raise ValueError("Goal position must be walkable.")

        self.maze = maze
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.route_length = route_length

    def is_valid(self, sequence):
        """Return True if the movement sequence describes a valid route."""
        if len(sequence) != self.route_length:
            return False

        try:
            path = RouteEncoding.sequence_to_path(self.start, sequence)
        except (TypeError, ValueError):
            return False

        for x, y in path[1:]:
            if not self.maze.is_inside(x, y):
                return False

            if not self.maze.is_walkable(x, y):
                return False

        return path[-1] == self.goal

    def is_valid_index(self, index):
        """Evaluate a computational-basis index."""
        try:
            sequence = RouteEncoding.decode(index, self.route_length)
        except (TypeError, ValueError):
            return False

        return self.is_valid(sequence)

    def get_valid_indices(self):
        """
        Enumerate all valid basis-state indices.

        This is a classical exhaustive reference, not an efficient
        quantum oracle implementation.
        """
        return [
            index
            for index in range(4 ** self.route_length)
            if self.is_valid_index(index)
        ]
