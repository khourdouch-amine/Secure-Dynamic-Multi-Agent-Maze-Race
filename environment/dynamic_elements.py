"""
Dynamic walls and moving obstacles manager.

Supports protected robot positions and connectivity checks
for multiple start-goal missions.
"""

from __future__ import annotations

from collections import deque
from math import isfinite

import numpy as np

from config import settings
from environment.maze import (
    FREE,
    DYNAMIC_WALL,
    MOVING_OBSTACLE,
)

from environment.moving_obstacle import MovingObstacle


class DynamicEnvironment:
    """Manage dynamic elements while preserving mission connectivity."""

    def __init__(
        self,
        maze,
        number_of_walls=settings.NUMBER_OF_DYNAMIC_WALLS,
        number_of_obstacles=settings.NUMBER_OF_MOVING_OBSTACLES,
        seed=settings.RANDOM_SEED + 1,
        reserved_cells=(),
        connectivity_pairs=None,
    ):
        self.maze = maze
        self.number_of_walls = number_of_walls
        self.number_of_obstacles = number_of_obstacles

        self.wall_interval = settings.DYNAMIC_WALL_INTERVAL
        self.obstacle_interval = settings.OBSTACLE_UPDATE_INTERVAL

        self.rng = np.random.default_rng(seed)

        self.wall_positions = set()
        self.obstacle_positions = set()
        self.obstacle_registry = {}
        self.obstacle_trajectories = {}
        self.obstacle_reserved_cells = set()

        # Keep the original global mission protected.
        pairs = [(maze.start, maze.goal)]

        if connectivity_pairs is not None:
            pairs.extend(
                (tuple(start), tuple(goal))
                for start, goal in connectivity_pairs
            )

        # Remove duplicate mission pairs while preserving order.
        self.connectivity_pairs = list(dict.fromkeys(pairs))

        # Every mission endpoint is automatically protected.
        self.protected_cells = {
            tuple(cell) for cell in reserved_cells
        }

        for start, goal in self.connectivity_pairs:
            self.protected_cells.add(start)
            self.protected_cells.add(goal)

        self.protected_cells.update((maze.start, maze.goal))

        self._validate_configuration()

        self.last_wall_update = 0.0
        self.last_obstacle_update = 0.0
        self._last_update_time = 0.0

        # Preserve theoretical update schedules for both dynamic systems.
        self.next_wall_update = self.wall_interval
        self.next_obstacle_update = self.obstacle_interval

        self._clear_all()
        self._generate_walls(set())
        self._spawn_missing_obstacles(set())

    @classmethod
    def from_agents(
        cls,
        maze,
        agents,
        number_of_walls=settings.NUMBER_OF_DYNAMIC_WALLS,
        number_of_obstacles=settings.NUMBER_OF_MOVING_OBSTACLES,
        seed=settings.RANDOM_SEED + 1,
    ):
        """Build a dynamic environment protecting all agent missions."""
        agents = list(agents)

        if not agents:
            raise ValueError(
                "At least one agent is required to initialize the environment."
            )

        reserved_cells = set()
        connectivity_pairs = []
        agent_ids = set()

        for agent in agents:
            if not hasattr(agent, "start") or not hasattr(agent, "goal"):
                raise ValueError(
                    "Every agent must define start and goal positions."
                )

            start = tuple(agent.start)
            position = tuple(getattr(agent, "position", start))
            goal = tuple(agent.goal)

            if any(
                len(cell) != 2
                for cell in (start, position, goal)
            ):
                raise ValueError(
                    "Agent start, position and goal must be 2D coordinates."
                )

            agent_id = getattr(agent, "agent_id", None)

            if agent_id is not None:
                if agent_id in agent_ids:
                    raise ValueError("Agent IDs must be unique.")
                agent_ids.add(agent_id)

            reserved_cells.update((start, position, goal))
            connectivity_pairs.extend(
                ((start, goal), (position, goal))
            )

        return cls(
            maze=maze,
            number_of_walls=number_of_walls,
            number_of_obstacles=number_of_obstacles,
            seed=seed,
            reserved_cells=reserved_cells,
            connectivity_pairs=connectivity_pairs,
        )

    def _validate_configuration(self):
        """Validate mission endpoints and protected cells."""
        if self.number_of_walls < 0 or self.number_of_obstacles < 0:
            raise ValueError("Dynamic element counts cannot be negative.")

        for position in self.protected_cells:
            x, y = position

            if not self.maze.is_inside(x, y):
                raise ValueError(
                    f"Protected cell is outside the maze: {position}"
                )

            if not self.maze.is_walkable(x, y):
                raise ValueError(
                    f"Protected cell is not initially walkable: {position}"
                )

        if not self.connectivity_pairs:
            raise ValueError("At least one connectivity pair is required.")

        for start, goal in self.connectivity_pairs:
            if not self.maze.is_inside(*start):
                raise ValueError(f"Mission start outside maze: {start}")

            if not self.maze.is_inside(*goal):
                raise ValueError(f"Mission goal outside maze: {goal}")

            if not self.maze.is_walkable(*start):
                raise ValueError(f"Mission start is blocked: {start}")

            if not self.maze.is_walkable(*goal):
                raise ValueError(f"Mission goal is blocked: {goal}")

    def _clear_all(self):
        """Remove dynamic elements without modifying static walls."""
        mask = np.isin(
            self.maze.grid,
            [DYNAMIC_WALL, MOVING_OBSTACLE],
        )
        self.maze.grid[mask] = FREE

        self.wall_positions.clear()
        self.obstacle_positions.clear()
        self.obstacle_registry.clear()
        self.obstacle_trajectories.clear()
        self.obstacle_reserved_cells.clear()

    def _clear_walls_only(self):
        """Remove dynamic walls without disturbing moving obstacles."""
        for x, y in self.wall_positions:
            if self.maze.grid[y, x] == DYNAMIC_WALL:
                self.maze.grid[y, x] = FREE

        self.wall_positions.clear()

    def _has_path(self, start, goal):
        """Check one mission using breadth-first search."""
        if not self.maze.is_walkable(*start):
            return False

        if not self.maze.is_walkable(*goal):
            return False

        queue = deque([start])
        visited = {start}

        while queue:
            x, y = queue.popleft()

            if (x, y) == goal:
                return True

            for nx, ny in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ):
                position = (nx, ny)

                if position in visited:
                    continue

                if not self.maze.is_inside(nx, ny):
                    continue

                if not self.maze.is_walkable(nx, ny):
                    continue

                visited.add(position)
                queue.append(position)

        return False

    def _all_missions_connected(self):
        """Check connectivity for every configured mission."""
        return all(
            self._has_path(start, goal)
            for start, goal in self.connectivity_pairs
        )

    def _try_place(self, position, cell_type):
        """Place an element only if all missions remain connected."""
        x, y = position

        if not self.maze.is_inside(x, y):
            return False

        if position in self.protected_cells:
            return False

        if self.maze.grid[y, x] != FREE:
            return False

        self.maze.grid[y, x] = cell_type

        if self._all_missions_connected():
            return True

        self.maze.grid[y, x] = FREE
        return False

    def _candidate_cells(self, forbidden):
        """Return randomly ordered available interior cells."""
        forbidden = set(forbidden) | self.protected_cells

        candidates = [
            (x, y)
            for y in range(1, self.maze.height - 1)
            for x in range(1, self.maze.width - 1)
            if (x, y) not in forbidden
            and self.maze.grid[y, x] == FREE
        ]

        if candidates:
            order = self.rng.permutation(len(candidates))
            candidates = [candidates[i] for i in order]

        return candidates

    def _generate_walls(self, occupied_cells):
        """Generate dynamic walls while preserving all missions."""
        forbidden = set(occupied_cells) | self.protected_cells

        for position in self._candidate_cells(forbidden):
            if len(self.wall_positions) >= self.number_of_walls:
                break

            if self._try_place(position, DYNAMIC_WALL):
                self.wall_positions.add(position)

        if len(self.wall_positions) != self.number_of_walls:
            raise RuntimeError(
                "Could not generate the requested number of "
                "dynamic walls while preserving all missions."
            )

    def _spawn_missing_obstacles(self, occupied_cells):
        """Place missing obstacles without blocking protected missions."""
        forbidden = (
            set(occupied_cells)
            | self.protected_cells
            | self.wall_positions
            | self.obstacle_positions
            | self.obstacle_reserved_cells
        )

        available_ids = [
            f"O{index:02d}"
            for index in range(self.number_of_obstacles)
            if f"O{index:02d}" not in self.obstacle_registry
        ]

        for position in self._candidate_cells(forbidden):
            if len(self.obstacle_positions) >= self.number_of_obstacles:
                break

            if self._try_place(position, MOVING_OBSTACLE):
                self.obstacle_positions.add(position)
                obstacle_id = available_ids.pop(0)
                self.obstacle_registry[obstacle_id] = position
                self.obstacle_reserved_cells.add(position)

        if len(self.obstacle_positions) != self.number_of_obstacles:
            raise RuntimeError(
                "Could not generate the requested number of "
                "moving obstacles while preserving all missions."
            )

    def update_obstacles(self, occupied_cells=()):
        """Move obstacles independently by one grid cell."""
        occupied = set(occupied_cells)

        if self.obstacle_trajectories:
            raise RuntimeError(
                "Cannot use discrete obstacle updates while "
                "continuous trajectories are active."
            )

        previous_obstacles = list(self.obstacle_registry.items())

        # Release the previous stationary obstacle reservations.
        self.obstacle_reserved_cells.difference_update(
            position for _, position in previous_obstacles
        )

        for _, (x, y) in previous_obstacles:
            self.maze.grid[y, x] = FREE

        self.obstacle_positions.clear()
        self.obstacle_registry.clear()

        order = self.rng.permutation(len(previous_obstacles))

        for index in order:
            obstacle_id, previous_position = previous_obstacles[index]
            x, y = previous_position

            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]

            neighbors = [
                position
                for position in neighbors
                if position not in occupied
                and position not in self.protected_cells
                and position not in self.obstacle_positions
                and self.maze.is_inside(*position)
                and self.maze.is_walkable(*position)
            ]

            if neighbors:
                order_neighbors = self.rng.permutation(len(neighbors))
                neighbors = [neighbors[i] for i in order_neighbors]

            placed = False

            for position in neighbors:
                if self._try_place(position, MOVING_OBSTACLE):
                    self.obstacle_positions.add(position)
                    self.obstacle_registry[obstacle_id] = position
                    placed = True
                    break

            if not placed and (x, y) not in occupied:
                if self._try_place((x, y), MOVING_OBSTACLE):
                    self.obstacle_positions.add((x, y))
                    self.obstacle_registry[obstacle_id] = (x, y)

        # Every newly placed obstacle must reserve its current cell.
        self.obstacle_reserved_cells.update(self.obstacle_positions)

        self._spawn_missing_obstacles(occupied)

    def _complete_continuous_trajectories(self, event_time):
        """Finalize all obstacle trajectories ending at event_time."""
        completed = [
            obstacle_id
            for obstacle_id, trajectory in self.obstacle_trajectories.items()
            if trajectory.end_time <= event_time + 1e-12
        ]

        for obstacle_id in completed:
            trajectory = self.obstacle_trajectories.pop(obstacle_id)

            source = tuple(
                int(round(value))
                for value in trajectory.start_position
            )

            target = tuple(
                int(round(value))
                for value in trajectory.target_position
            )

            # Release the source only when no other obstacle uses it.
            source_is_still_used = (
                source in self.obstacle_positions
                or any(
                    source in {
                        tuple(int(round(value)) for value in item.start_position),
                        tuple(int(round(value)) for value in item.target_position),
                    }
                    for item in self.obstacle_trajectories.values()
                )
            )

            if not source_is_still_used:
                self.obstacle_reserved_cells.discard(source)

                x, y = source
                if self.maze.grid[y, x] == MOVING_OBSTACLE:
                    self.maze.grid[y, x] = FREE

            # The destination remains the obstacle's stationary cell.
            self.obstacle_reserved_cells.add(target)
            self.obstacle_positions.add(target)
            self.obstacle_registry[obstacle_id] = target

    def _start_continuous_obstacle_trajectories(
        self,
        event_time,
        occupied_cells=(),
    ):
        """Start one-cell continuous trajectories for eligible obstacles."""
        occupied = set(occupied_cells)

        obstacle_ids = list(self.obstacle_registry)
        if obstacle_ids:
            order = self.rng.permutation(len(obstacle_ids))
            obstacle_ids = [obstacle_ids[i] for i in order]

        for obstacle_id in obstacle_ids:
            if obstacle_id in self.obstacle_trajectories:
                continue

            source = self.obstacle_registry[obstacle_id]
            x, y = source

            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]

            forbidden = (
                occupied
                | self.protected_cells
                | self.obstacle_reserved_cells
                | self.wall_positions
            )

            neighbors = [
                cell
                for cell in neighbors
                if cell not in forbidden
                and self.maze.is_inside(*cell)
                and self.maze.grid[cell[1], cell[0]] == FREE
            ]

            if not neighbors:
                continue

            order = self.rng.permutation(len(neighbors))
            neighbors = [neighbors[i] for i in order]

            target = None

            for candidate in neighbors:
                if self._try_place(candidate, MOVING_OBSTACLE):
                    target = candidate
                    break

            if target is None:
                continue

            trajectory = MovingObstacle(
                start_position=(float(source[0]), float(source[1])),
                target_position=(float(target[0]), float(target[1])),
                start_time=float(event_time),
                duration=float(self.obstacle_interval),
            )

            self.obstacle_trajectories[obstacle_id] = trajectory

            # The registry and obstacle_positions represent the destination.
            self.obstacle_registry[obstacle_id] = target
            self.obstacle_positions.discard(source)
            self.obstacle_positions.add(target)

            # Both endpoints remain blocked for the whole transition.
            self.obstacle_reserved_cells.add(source)
            self.obstacle_reserved_cells.add(target)

    def update_walls(self, occupied_cells=()):
        """Regenerate walls without relocating obstacles in active motion."""
        occupied = set(occupied_cells)

        self._clear_walls_only()

        # Preserve both endpoints of every active trajectory in the grid.
        for trajectory in self.obstacle_trajectories.values():
            for position in (
                trajectory.start_position,
                trajectory.target_position,
            ):
                x, y = (int(round(value)) for value in position)

                if self.maze.grid[y, x] == FREE:
                    self.maze.grid[y, x] = MOVING_OBSTACLE

                self.obstacle_reserved_cells.add((x, y))

        # Only stationary obstacles may be relocated.
        for obstacle_id, old_position in list(
            self.obstacle_registry.items()
        ):
            if obstacle_id in self.obstacle_trajectories:
                continue

            if old_position not in occupied:
                continue

            x, y = old_position

            self.maze.grid[y, x] = FREE
            self.obstacle_positions.discard(old_position)
            self.obstacle_reserved_cells.discard(old_position)
            del self.obstacle_registry[obstacle_id]

            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]

            neighbors = [
                candidate
                for candidate in neighbors
                if candidate not in occupied
                and candidate not in self.protected_cells
                and candidate not in self.obstacle_positions
                and candidate not in self.obstacle_reserved_cells
                and self.maze.is_inside(*candidate)
                and self.maze.grid[candidate[1], candidate[0]] == FREE
            ]

            if neighbors:
                order = self.rng.permutation(len(neighbors))
                neighbors = [neighbors[i] for i in order]

            placed = False

            for candidate in neighbors:
                if self._try_place(candidate, MOVING_OBSTACLE):
                    self.obstacle_positions.add(candidate)
                    self.obstacle_registry[obstacle_id] = candidate
                    self.obstacle_reserved_cells.add(candidate)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"Could not relocate obstacle {obstacle_id} "
                    f"from occupied cell {old_position}."
                )

        forbidden = (
            occupied
            | self.obstacle_positions
            | self.obstacle_reserved_cells
            | self.protected_cells
        )

        self._generate_walls(forbidden)
        self._spawn_missing_obstacles(occupied)

    def update(self, simulation_time, occupied_cells=()):
        """Update dynamic elements using continuous obstacle trajectories."""
        try:
            simulation_time = float(simulation_time)
        except (TypeError, OverflowError) as exc:
            raise ValueError(
                "Simulation time must be a finite number."
            ) from exc

        if not isfinite(simulation_time):
            raise ValueError("Simulation time must be finite.")

        if simulation_time < 0.0:
            raise ValueError("Simulation time cannot be negative.")

        occupied = set(occupied_cells)

        previous_blocked = (
            self.wall_positions | self.obstacle_reserved_cells
        )

        walls_updated = False
        obstacles_updated = False

        # Preserve every trajectory overlapping the current update interval.
        interval_start = self._last_update_time
        trajectories_during_interval = []

        # Process every elapsed obstacle deadline chronologically.
        while (
            self.next_obstacle_update
            <= simulation_time + 1e-12
        ):
            event_time = round(self.next_obstacle_update, 12)

            # Capture trajectories before completed ones leave the registry.
            for obstacle_id, trajectory in self.obstacle_trajectories.items():
                if trajectory.end_time >= interval_start - 1e-12:
                    if not any(
                        item is trajectory
                        for item in trajectories_during_interval
                    ):
                        trajectories_during_interval.append(trajectory)

            self._complete_continuous_trajectories(event_time)

            self._start_continuous_obstacle_trajectories(
                event_time,
                occupied_cells=occupied,
            )

            # Also capture trajectories created during this update.
            for obstacle_id, trajectory in self.obstacle_trajectories.items():
                if trajectory.start_time <= simulation_time + 1e-12:
                    if not any(
                        item is trajectory
                        for item in trajectories_during_interval
                    ):
                        trajectories_during_interval.append(trajectory)

            self.last_obstacle_update = event_time
            self.next_obstacle_update = round(
                event_time + self.obstacle_interval,
                12,
            )

            obstacles_updated = True

        # Regenerate walls when their next theoretical deadline is reached.
        # A delayed simulation call triggers one regeneration, then advances
        # the schedule past all elapsed deadlines without cumulative drift.
        if self.next_wall_update <= simulation_time + 1e-12:
            deadline = self.next_wall_update

            elapsed_intervals = int(
                (simulation_time - deadline + 1e-12)
                / self.wall_interval
            )

            last_deadline = round(
                deadline + elapsed_intervals * self.wall_interval,
                12,
            )

            self.update_walls(occupied_cells=occupied)

            self.last_wall_update = last_deadline
            self.next_wall_update = round(
                last_deadline + self.wall_interval,
                12,
            )

            walls_updated = True

        current_blocked = (
            self.wall_positions | self.obstacle_reserved_cells
        )

        changed_cells = sorted(
            previous_blocked.symmetric_difference(current_blocked)
        )

        # Keep the interval history even when trajectories have completed.
        for obstacle_id, trajectory in self.obstacle_trajectories.items():
            if trajectory.end_time >= interval_start - 1e-12:
                if not any(
                    item is trajectory
                    for item in trajectories_during_interval
                ):
                    trajectories_during_interval.append(trajectory)

        self._last_update_time = max(
            self._last_update_time,
            simulation_time,
        )

        return {
            "walls_updated": walls_updated,
            "obstacles_updated": obstacles_updated,
            "changed_cells": changed_cells,
            "obstacle_trajectories": tuple(
                trajectories_during_interval
            ),
        }
