"""Classical reference oracle for global Grover route search."""

from planners.quantum.route_encoding import RouteEncoding


class GlobalClassicalOracle:
    def __init__(self, maze_model, route_length):
        if isinstance(route_length, bool) or not isinstance(route_length, int):
            raise TypeError("route_length must be an integer.")
        if route_length < 1:
            raise ValueError("route_length must be positive.")

        self.maze = maze_model
        self.route_length = route_length
        self.search_space = 4 ** route_length

    def evaluate(self, index):
        try:
            route = RouteEncoding.decode(index, self.route_length)
        except (TypeError, ValueError):
            return False

        valid, _, collision_step = self.maze.evaluate_route(route)
        return valid and collision_step is None

    def find_valid_indices(self):
        return [
            index
            for index in range(self.search_space)
            if self.evaluate(index)
        ]

    def count_valid_routes(self):
        return len(self.find_valid_indices())

    def decode_valid_routes(self):
        return [
            RouteEncoding.decode(index, self.route_length)
            for index in self.find_valid_indices()
        ]
