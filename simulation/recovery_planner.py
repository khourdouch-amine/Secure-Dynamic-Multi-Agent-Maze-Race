"""
Safety-aware waypoint planning for deadlock recovery.

This module proposes temporary navigation targets. It never moves agents
and never modifies their permanent planners or mission goals.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from math import hypot, isfinite

from planners.classical.astar import AStarPlanner
from planners.classical.bidirectional_bfs import BidirectionalBFSPlanner
from planners.classical.dijkstra import DijkstraPlanner
from planners.classical.d_star_lite import DStarLitePlanner
from planners.classical.rrt_star_grid import RRTStarGridPlanner

from planners.quantum.grover_navigation import GroverNavigationPlanner
from planners.quantum.quantum_walk_navigation import (
    QuantumWalkNavigationPlanner,
)
from planners.quantum.quantum_walk_local_navigation import (
    QuantumWalkLocalNavigationPlanner,
)
from planners.quantum.qaoa_navigation import QAOANavigationPlanner
from planners.quantum.vqc_navigation import VQCNavigationPlanner

from environment.moving_obstacle import MovingObstacle
from security.physical.geometry import (
    DEFAULT_CELL_SIZE,
    DEFAULT_ROBOT_OBSTACLE_MARGIN,
    DEFAULT_ROBOT_RADIUS,
    DEFAULT_ROBOT_ROBOT_DISTANCE,
    minimum_distance_during_motion,
    minimum_distance_during_motion_to_square,
    robot_is_safe_from_square,
    robots_are_safe,
)
from security.physical.collision import (
    is_safe_during_motion_to_obstacle,
)


@dataclass(frozen=True)
class RecoveryPlan:
    """A validated temporary path and its independent planner."""

    waypoint: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    planner: object
    arrival_time: float
    planner_name: str
    family: str
    candidates_examined: int


class RecoveryPlanner:
    """
    Select a safe temporary waypoint using the agent's original algorithm.

    Unknown planner types are rejected. No classical fallback is silently
    substituted for a quantum or hybrid-quantum planner.
    """

    def __init__(
        self,
        time_step: float = 1.0 / 30.0,
        max_candidates: int = 64,
        robot_radius: float = DEFAULT_ROBOT_RADIUS,
        robot_robot_distance: float = DEFAULT_ROBOT_ROBOT_DISTANCE,
        obstacle_margin: float = DEFAULT_ROBOT_OBSTACLE_MARGIN,
    ) -> None:
        if (
            isinstance(time_step, bool)
            or not isinstance(time_step, (int, float))
            or not isfinite(float(time_step))
            or time_step <= 0
        ):
            raise ValueError("time_step must be finite and positive.")

        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            raise ValueError("max_candidates must be a positive integer.")

        for value, name in (
            (robot_radius, "robot_radius"),
            (robot_robot_distance, "robot_robot_distance"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")

        if (
            isinstance(obstacle_margin, bool)
            or not isinstance(obstacle_margin, (int, float))
            or not isfinite(float(obstacle_margin))
            or obstacle_margin < 0
        ):
            raise ValueError("obstacle_margin must be finite and non-negative.")

        self.time_step = float(time_step)
        self.max_candidates = max_candidates
        self.robot_radius = float(robot_radius)
        self.robot_robot_distance = float(robot_robot_distance)
        self.obstacle_margin = float(obstacle_margin)


    @staticmethod
    def _distances_to_goal(maze, goal):
        """Return shortest four-connected distances from every reachable cell."""
        goal = tuple(goal)

        if len(goal) != 2 or not maze.is_walkable(*goal):
            return {}

        distances = {goal: 0}
        frontier = deque([goal])

        while frontier:
            current = frontier.popleft()

            for neighbor in maze.get_neighbors(*current):
                neighbor = tuple(neighbor)

                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    frontier.append(neighbor)

        return distances

    @staticmethod
    def _clone_planner(planner, maze, start, goal):
        """Create an independent planner preserving its algorithm/config."""

        if isinstance(planner, BidirectionalBFSPlanner):
            return BidirectionalBFSPlanner()

        if isinstance(planner, AStarPlanner):
            return AStarPlanner()

        if isinstance(planner, DijkstraPlanner):
            return DijkstraPlanner()

        if isinstance(planner, DStarLitePlanner):
            return DStarLitePlanner(maze, start, goal)

        if isinstance(planner, RRTStarGridPlanner):
            source = planner.rrt_star

            clone = RRTStarGridPlanner(
                max_iterations=source.max_iterations,
                step_size=source.step_size,
                goal_radius=source.goal_radius,
                neighbor_radius=source.neighbor_radius,
                seed=42,
                adapter_margin=planner.adapter.margin,
            )
            clone.rrt_star.rng.bit_generator.state = deepcopy(
                source.rng.bit_generator.state
            )
            return clone

        if isinstance(planner, GroverNavigationPlanner):
            return GroverNavigationPlanner(
                shots=planner.shots,
                seed=planner.seed,
                max_steps=planner.max_steps,
            )

        if isinstance(planner, QuantumWalkNavigationPlanner):
            return QuantumWalkNavigationPlanner(
                maze=maze,
                coin_type=planner.coin_type,
                shots=planner.shots,
                walk_steps=planner.walk_steps,
                max_steps=(
                    None
                    if planner._automatic_max_steps
                    else planner.max_steps
                ),
                seed=planner.seed,
                max_total_qubits=planner.max_total_qubits,
            )

        if isinstance(planner, QuantumWalkLocalNavigationPlanner):
            return QuantumWalkLocalNavigationPlanner(
                window_size=planner.window_size,
                coin_type=planner.coin_type,
                shots=planner.shots,
                walk_steps=planner.walk_steps,
                max_total_qubits=planner.max_total_qubits,
                seed=planner.seed,
                max_segments=planner.max_segments,
            )

        if isinstance(planner, QAOANavigationPlanner):
            return QAOANavigationPlanner(
                candidates=planner.max_candidates,
                layers=planner.layers,
                shots=planner.shots,
                max_iterations=planner.max_iterations,
                penalty=planner.penalty,
                seed=planner.seed,
            )

        if isinstance(planner, VQCNavigationPlanner):
            return VQCNavigationPlanner(
                layers=planner.layers,
                shots=planner.shots,
                max_iterations=planner.max_iterations,
                max_training_samples=planner.max_training_samples,
                seed=planner.seed,
            )

        raise NotImplementedError(
            "Recovery is not implemented for planner type "
            f"{type(planner).__name__}; no fallback will be substituted."
        )

    @staticmethod
    def _validate_path(maze, path, start, goal):
        """Validate endpoints, walkability and four-connected movements."""

        if path is None:
            return None

        try:
            normalized = [tuple(point) for point in path]
        except (TypeError, ValueError):
            return None

        if not normalized or normalized[0] != start:
            return None

        if normalized[-1] != goal:
            return None

        for point in normalized:
            if (
                len(point) != 2
                or not maze.is_inside(*point)
                or not maze.is_walkable(*point)
            ):
                return None

        for first, second in zip(normalized, normalized[1:]):
            if (
                abs(first[0] - second[0])
                + abs(first[1] - second[1])
                != 1
            ):
                return None

        return normalized

    def _candidate_is_safe(
        self,
        candidate,
        other_positions,
        blocked_squares,
    ):
        """Check static square clearance and robot-to-robot spacing."""

        if candidate in other_positions:
            return False

        for position in other_positions:
            if not robots_are_safe(
                candidate,
                position,
                min_center_distance=self.robot_robot_distance,
            ):
                return False

        for square in blocked_squares:
            if not robot_is_safe_from_square(
                candidate,
                square,
                margin=self.obstacle_margin,
                robot_radius=self.robot_radius,
            ):
                return False

        return True

    def _path_is_safe(
        self,
        path,
        other_positions,
        trajectories,
        blocked_squares,
        simulation_time,
    ):
        """Check robot spacing and all static/dynamic clearances."""

        # A one-point path still occupies its starting/arrival cell.
        if len(path) == 1:
            for square in blocked_squares:
                if not robot_is_safe_from_square(
                    path[0],
                    square,
                    margin=self.obstacle_margin,
                    robot_radius=self.robot_radius,
                ):
                    return False

        for index, (start, end) in enumerate(zip(path, path[1:])):
            interval_start = simulation_time + index * self.time_step
            interval_end = interval_start + self.time_step

            # Check the complete segment against every static square.
            for square in blocked_squares:
                clearance = minimum_distance_during_motion_to_square(
                    robot_start=start,
                    robot_end=end,
                    obstacle_start=square,
                    obstacle_end=square,
                    square_side=DEFAULT_CELL_SIZE,
                ) - self.robot_radius - self.obstacle_margin

                if clearance < 0.0:
                    return False

            for other_position in other_positions:
                minimum_distance = minimum_distance_during_motion(
                    start,
                    end,
                    other_position,
                    other_position,
                )
                if minimum_distance < self.robot_robot_distance:
                    return False

            for trajectory in trajectories:
                if not isinstance(trajectory, MovingObstacle):
                    continue

                if not is_safe_during_motion_to_obstacle(
                    robot_start=start,
                    robot_end=end,
                    obstacle=trajectory,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    robot_radius=self.robot_radius,
                    safety_margin=self.obstacle_margin,
                ):
                    return False

        # Also check the waypoint while the robot is waiting there.
        arrival_time = simulation_time + max(0, len(path) - 1) * self.time_step
        hold_end = arrival_time + self.time_step

        for trajectory in trajectories:
            if not isinstance(trajectory, MovingObstacle):
                continue

            if not is_safe_during_motion_to_obstacle(
                robot_start=path[-1],
                robot_end=path[-1],
                obstacle=trajectory,
                interval_start=arrival_time,
                interval_end=hold_end,
                robot_radius=self.robot_radius,
                safety_margin=self.obstacle_margin,
            ):
                return False

        return True

    def plan(
        self,
        agent,
        maze,
        dynamic_environment,
        agents,
        simulation_time: float,
        obstacle_trajectories=(),
        excluded_waypoints=(),
    ) -> RecoveryPlan | None:
        """
        Find a safe waypoint and path for an agent.

        Returns None if no candidate passes all checks. The agent, its
        permanent goal and its permanent planner are never modified.
        """

        try:
            simulation_time = float(simulation_time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "simulation_time must be finite and non-negative."
            ) from exc

        if not isfinite(simulation_time) or simulation_time < 0:
            raise ValueError(
                "simulation_time must be finite and non-negative."
            )

        start = tuple(agent.position)
        permanent_goal = tuple(agent.goal)
        original_planner = agent.planner

        if start == permanent_goal:
            return None

        if not maze.is_walkable(*start):
            return None

        other_agents = [
            other
            for other in agents
            if other is not agent
        ]

        other_positions = {
            tuple(other.position)
            for other in other_agents
        }

        other_goals = {
            tuple(other.goal)
            for other in other_agents
            if hasattr(other, "goal")
        }

        # Combine currently active trajectories with the exact interval
        # history returned by DynamicEnvironment.update().
        trajectory_list = []

        if dynamic_environment is not None:
            active = getattr(
                dynamic_environment,
                "obstacle_trajectories",
                {},
            )
            trajectory_list.extend(active.values())

        trajectory_list.extend(obstacle_trajectories or ())

        # Preserve order while avoiding duplicate object references.
        trajectories = tuple(
            item
            for index, item in enumerate(trajectory_list)
            if not any(item is previous for previous in trajectory_list[:index])
        )

        # Endpoints belonging to a continuous trajectory must be checked
        # against that trajectory, not treated as permanently occupied.
        trajectory_endpoints = set()

        for trajectory in trajectories:
            if isinstance(trajectory, MovingObstacle):
                trajectory_endpoints.add(
                    tuple(int(round(value)) for value in trajectory.start_position)
                )
                trajectory_endpoints.add(
                    tuple(int(round(value)) for value in trajectory.target_position)
                )

        blocked_squares = set()

        if dynamic_environment is not None:
            blocked_squares.update(
                getattr(dynamic_environment, "wall_positions", ())
            )

            stationary_obstacles = set(
                getattr(dynamic_environment, "obstacle_positions", ())
            )
            stationary_obstacles.difference_update(trajectory_endpoints)
            blocked_squares.update(stationary_obstacles)

            reserved_cells = set(
                getattr(dynamic_environment, "obstacle_reserved_cells", ())
            )
            reserved_cells.difference_update(trajectory_endpoints)
            blocked_squares.update(reserved_cells)

        excluded_waypoints = {
            tuple(waypoint) for waypoint in (excluded_waypoints or ())
        }

        candidates = []
        goal_distances = self._distances_to_goal(
            maze,
            permanent_goal,
        )

        if start not in goal_distances:
            return None

        for y in range(1, maze.height - 1):
            for x in range(1, maze.width - 1):
                candidate = (x, y)

                if candidate in (start, permanent_goal):
                    continue

                if candidate in excluded_waypoints:
                    continue

                if candidate in other_goals:
                    continue

                if not maze.is_walkable(x, y):
                    continue

                if not self._candidate_is_safe(
                    candidate,
                    other_positions,
                    blocked_squares,
                ):
                    continue

                remaining_distance = goal_distances.get(candidate)

                if remaining_distance is None:
                    continue

                separation = min(
                    (
                        hypot(x - px, y - py)
                        for px, py in other_positions
                    ),
                    default=float("inf"),
                )

                # Prefer topological progress toward the permanent goal,
                # then nearby waypoints with more room around other robots.
                candidates.append(
                    (
                        remaining_distance,
                        abs(x - start[0]) + abs(y - start[1]),
                        -separation,
                        candidate,
                    )
                )

        candidates.sort()

        examined = 0

        family = getattr(agent, "family", "classical")
        candidate_budget = self.max_candidates

        if family == "quantum":
            candidate_budget = min(candidate_budget, 6)

        for _, _, _, waypoint in candidates[:candidate_budget]:
            examined += 1

            temporary_planner = self._clone_planner(
                original_planner,
                maze,
                start,
                waypoint,
            )

            # Do not hide planner errors: they indicate an implementation
            # or runtime problem that must remain visible during validation.
            raw_path = temporary_planner.plan(
                maze,
                start,
                waypoint,
            )

            path = self._validate_path(
                maze,
                raw_path,
                start,
                waypoint,
            )

            if path is None:
                continue

            if not self._path_is_safe(
                path,
                other_positions,
                trajectories,
                blocked_squares,
                simulation_time,
            ):
                continue

            arrival_time = (
                float(simulation_time)
                + max(0, len(path) - 1) * self.time_step
            )

            return RecoveryPlan(
                waypoint=waypoint,
                path=tuple(path),
                planner=temporary_planner,
                arrival_time=arrival_time,
                planner_name=type(original_planner).__name__,
                family=getattr(agent, "family", "unknown"),
                candidates_examined=examined,
            )

        return None
