"""
RRT* planner adapted to the four-connected grid agent interface.
"""

from planners.classical.rrt_star import RRTStarPlanner
from planners.classical.grid_path_adapter import GridPathAdapter


class RRTStarGridPlanner:
    """RRT* with automatic conversion to cardinal grid paths."""

    def __init__(
        self,
        max_iterations=5000,
        step_size=2.0,
        goal_radius=2.0,
        neighbor_radius=4.0,
        seed=42,
        adapter_margin=2,
    ):
        self.name = "RRT*"
        
        self.rrt_star = RRTStarPlanner(
            max_iterations=max_iterations,
            step_size=step_size,
            goal_radius=goal_radius,
            neighbor_radius=neighbor_radius,
            seed=seed,
        )

        self.adapter = GridPathAdapter(
            margin=adapter_margin
        )

    def plan(self, maze, start, goal):
        """Return a valid four-connected path or None."""

        waypoints = self.rrt_star.plan(
            maze,
            start,
            goal,
        )

        if waypoints is None:
            return None

        return self.adapter.adapt(
            maze,
            waypoints,
        )
