"""
Common agent interface for Classical AI and Quantum AI robots.
"""

from __future__ import annotations

from time import perf_counter


class BaseAgent:
    """Common navigation interface for all autonomous agents."""

    VALID_FAMILIES = {"classical", "quantum"}

    def __init__(
        self,
        agent_id: str,
        start: tuple[int, int],
        goal: tuple[int, int],
        planner,
        family: str,
    ) -> None:
        if family not in self.VALID_FAMILIES:
            raise ValueError(f"Unsupported agent family: {family}")

        self.agent_id = str(agent_id)
        self.position = tuple(start)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.planner = planner
        self.family = family

        # Permanent mission destination and temporary navigation target.
        self.navigation_target = self.goal
        self.recovery_state = "idle"

        self.path: list[tuple[int, int]] = []
        self.status = "idle"

        self.metrics = {
            "planning_time": 0.0,
            "path_length": 0.0,
            "moves": 0,
            "collisions": 0,
            "replans": 0,
        }

    def begin_recovery(self, target: tuple[int, int]) -> None:
        """Set a temporary waypoint without changing the mission goal."""
        target = tuple(target)

        if len(target) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in target
        ):
            raise ValueError("Recovery target must contain two integers.")

        if target == self.position:
            raise ValueError("Recovery target must differ from current position.")

        if target == self.goal:
            raise ValueError("Recovery target must differ from mission goal.")

        self.navigation_target = target
        self.recovery_state = "active"
        self.path = []
        self.status = "needs_replanning"

    def complete_recovery(self) -> None:
        """Restore the permanent mission target after reaching the waypoint."""
        if self.recovery_state != "active":
            raise RuntimeError("No active recovery to complete.")

        if self.position != self.navigation_target:
            raise RuntimeError("Recovery waypoint has not been reached.")

        self.navigation_target = self.goal
        self.recovery_state = "idle"
        self.path = []
        self.status = "needs_replanning"

    def fail_recovery(self) -> None:
        """Cancel the temporary target while preserving the mission goal."""
        if self.recovery_state != "active":
            raise RuntimeError("No active recovery to fail.")

        self.navigation_target = self.goal
        self.recovery_state = "failed"
        self.path = []
        self.status = "needs_replanning"

    def mark_recovery_failed(self) -> None:
        """Mark recovery as exhausted without changing the mission goal."""
        if self.recovery_state == "active":
            raise RuntimeError(
                "Use fail_recovery() for an active recovery."
            )

        self.navigation_target = self.goal
        self.recovery_state = "failed"
        self.path = []
        self.status = "needs_replanning"

    def reset_recovery(self) -> None:
        """Reset a failed recovery after the agent makes progress."""
        if self.recovery_state != "failed":
            raise RuntimeError("Recovery is not in a failed state.")

        self.navigation_target = self.goal
        self.recovery_state = "idle"
        self.path = []
        self.status = "needs_replanning"

    @property
    def reached_navigation_target(self) -> bool:
        """Return whether the current navigation target has been reached."""
        return self.position == self.navigation_target

    def compute_plan(
        self,
        maze,
        target: tuple[int, int] | None = None,
        planner=None,
    ) -> list[tuple[int, int]]:
        """Plan toward the current or an explicitly supplied target.

        A temporary planner can be supplied for recovery operations,
        for example an independent D* Lite instance with its own goal.
        """
        destination = (
            self.navigation_target if target is None else tuple(target)
        )
        selected_planner = self.planner if planner is None else planner

        if len(destination) != 2:
            raise ValueError("Planning target must contain two coordinates.")

        start_time = perf_counter()

        try:
            result = selected_planner.plan(
                maze,
                self.position,
                destination,
            )
        except Exception:
            self.status = "planning_error"
            raise

        elapsed = perf_counter() - start_time
        self.metrics["planning_time"] += elapsed
        self.metrics["replans"] += 1

        if result is None:
            self.path = []
            self.status = "no_path"
            return []

        path = [tuple(point) for point in result]

        if not path or path[0] != self.position:
            raise ValueError(
                "Planner path must start at the agent's current position."
            )

        if path[-1] != destination:
            self.path = []
            self.status = "no_path"
            return []

        self.path = path
        self.metrics["path_length"] = sum(
            (
                (path[i][0] - path[i - 1][0]) ** 2
                + (path[i][1] - path[i - 1][1]) ** 2
            ) ** 0.5
            for i in range(1, len(path))
        )

        self.status = "planned"
        return self.path.copy()

    def move_one_step(self, maze, occupied_positions=()) -> bool:
        """Move one cell along the planned path if it is safe."""
        if self.status not in {"planned", "moving", "waiting"}:
            return False

        if self.reached_navigation_target:
            self.status = (
                "recovery_waypoint_reached"
                if self.recovery_state == "active"
                else "reached_goal"
            )
            return False

        try:
            current_index = self.path.index(self.position)
        except ValueError:
            self.status = "needs_replanning"
            return False

        if current_index + 1 >= len(self.path):
            self.status = (
                "recovery_waypoint_reached"
                if self.recovery_state == "active"
                else "reached_goal"
            )
            return False

        next_position = self.path[current_index + 1]

        if not maze.is_walkable(*next_position):
            self.status = "needs_replanning"
            return False

        if next_position in set(occupied_positions):
            self.metrics["collisions"] += 1
            self.status = "waiting"
            return False

        self.position = next_position
        self.metrics["moves"] += 1
        self.status = (
            "recovery_waypoint_reached"
            if self.reached_navigation_target and self.recovery_state == "active"
            else "reached_goal"
            if self.reached_goal
            else "moving"
        )

        return True

    @property
    def reached_goal(self) -> bool:
        """Return whether the agent has reached its destination."""
        return self.position == self.goal

    def get_metrics(self) -> dict:
        """Return a copy of the agent performance metrics."""
        return self.metrics.copy()
