"""Classical shortest-path baselines for multi-scale scenarios."""

from planners.classical.astar import AStarPlanner
from planners.classical.dijkstra import DijkstraPlanner

from .scenarios import SCENARIOS


class GridMazeAdapter:
    """Minimal grid adapter shared by the classical planners."""

    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)

    def is_walkable(self, x, y):
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and (x, y) not in self.obstacles
        )

    def get_neighbors(self, x, y):
        candidates = (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        )

        return [
            (nx, ny)
            for nx, ny in candidates
            if self.is_walkable(nx, ny)
        ]


def validate_path(path, maze, start, goal):
    """Check path endpoints, adjacency and obstacle avoidance."""
    if path is None:
        return False

    if not path or path[0] != start or path[-1] != goal:
        return False

    for x, y in path:
        if not maze.is_walkable(x, y):
            return False

    for first, second in zip(path, path[1:]):
        if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
            return False

    return True


def run_classical_baselines():
    astar = AStarPlanner()
    dijkstra = DijkstraPlanner()

    print("=" * 65)
    print("CLASSICAL SHORTEST-PATH BASELINES")
    print("=" * 65)

    for scenario in SCENARIOS:
        maze = GridMazeAdapter(
            scenario.width,
            scenario.height,
            scenario.obstacles,
        )

        print(f"\n===== {scenario.name} =====")
        print(f"Grid: {scenario.width}x{scenario.height}")
        print(f"Start: {scenario.start} | Goal: {scenario.goal}")
        print(f"Obstacles: {sorted(scenario.obstacles)}")

        for planner in (astar, dijkstra):
            path = planner.plan(maze, scenario.start, scenario.goal)
            valid = validate_path(
                path,
                maze,
                scenario.start,
                scenario.goal,
            )

            print(f"\n{planner.name}")
            print(f"Path: {path}")
            print(f"Path length: {len(path) - 1 if path else None} moves")
            print(f"Valid path: {valid}")

            if not valid:
                raise RuntimeError(
                    f"Invalid path returned by {planner.name} "
                    f"for {scenario.name}"
                )


if __name__ == "__main__":
    run_classical_baselines()
