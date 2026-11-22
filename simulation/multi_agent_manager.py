"""
Centralized multi-agent movement and conflict resolution.

This module handles simultaneous movement decisions independently
from the classical and quantum planning algorithms.
"""

from __future__ import annotations

import math

from security.physical.collision import is_safe_during_motion_to_obstacle
from security.physical.geometry import robots_are_safe_during_motion
from simulation.recovery_coordinator import RecoveryCoordinator
from simulation.recovery_planner import RecoveryPlanner


class MultiAgentManager:
    """Coordinate agents and resolve simultaneous movement conflicts."""

    def __init__(
        self,
        maze,
        agents,
        dynamic_environment=None,
        time_step=1.0 / 30.0,
        blocked_threshold=5,
        recovery_planner=None,
        recovery_retry_interval=1.0,
        max_recovery_attempts=3,
        traffic_coordinator=None,
    ):
        if time_step <= 0:
            raise ValueError("time_step must be strictly positive.")

        if (
            isinstance(recovery_retry_interval, bool)
            or not isinstance(recovery_retry_interval, (int, float))
            or not math.isfinite(recovery_retry_interval)
            or recovery_retry_interval <= 0
        ):
            raise ValueError(
                "recovery_retry_interval must be a finite, "
                "strictly positive number."
            )

        if (
            not isinstance(max_recovery_attempts, int)
            or isinstance(max_recovery_attempts, bool)
            or max_recovery_attempts <= 0
        ):
            raise ValueError(
                "max_recovery_attempts must be a positive integer."
            )

        self.maze = maze
        self.agents = list(agents)
        self.dynamic_environment = dynamic_environment
        self.traffic_coordinator = traffic_coordinator
        self.time_step = float(time_step)
        self.recovery_retry_interval = float(recovery_retry_interval)

        # Recovery planners remain separate from permanent agent planners.
        self.recovery_planner = (
            recovery_planner
            if recovery_planner is not None
            else RecoveryPlanner(time_step=self.time_step)
        )
        self.max_recovery_attempts = max_recovery_attempts
        self._recovery_planners = {}
        self._recovery_retry_after = {}
        self._recovery_attempts = {}
        self._recovery_memory = {}
        self._max_remembered_recovery_waypoints = 32
        self._step_recovery_events = []

        self.simulation_time = 0.0
        self._step_count = 0
        self._clock_origin = 0.0

        self.agent_ids = [agent.agent_id for agent in self.agents]

        if len(self.agent_ids) != len(set(self.agent_ids)):
            raise ValueError("Agent IDs must be unique.")

        positions = [agent.position for agent in self.agents]

        if len(positions) != len(set(positions)):
            raise ValueError("Agents cannot share an initial position.")

        for agent in self.agents:
            if not self.maze.is_inside(*agent.position):
                raise ValueError(
                    f"Agent {agent.agent_id} starts outside the maze."
                )

            if not self.maze.is_walkable(*agent.position):
                raise ValueError(
                    f"Agent {agent.agent_id} starts on a cell that is not "
                    f"walkable: {agent.position}"
                )

            if not self.maze.is_inside(*agent.goal):
                raise ValueError(
                    f"Agent {agent.agent_id} has a goal outside the maze."
                )

            if not self.maze.is_walkable(*agent.goal):
                raise ValueError(
                    f"Agent {agent.agent_id} has a goal on a cell that is "
                    f"not walkable: {agent.goal}"
                )

        self.priority_offset = 0
        self.total_conflicts = 0
        self.total_blocked_moves = 0
        self.total_successful_moves = 0

        self.total_invalid_proposals = 0
        self.total_destination_conflicts = 0
        self.total_cycle_rejections = 0
        self.total_dependency_blocked = 0
        self.total_obstacle_safety_blocks = 0
        self.total_robot_safety_blocks = 0

        self.last_resolution_metrics = {}
        self.last_physical_metrics = {}

        # Observe initial positions so the first blocked movement cycle
        # is counted without requiring an extra baseline cycle.
        self.recovery_coordinator = RecoveryCoordinator(
            blocked_threshold=blocked_threshold
        )

        for agent in self.agents:
            self.recovery_coordinator.observe(
                agent.agent_id,
                agent.position,
                self.simulation_time,
            )

    def _priority_order(self):
        """Return a rotating deterministic priority order."""
        count = len(self.agents)

        if count == 0:
            return []

        offset = self.priority_offset % count

        return (
            self.agents[offset:]
            + self.agents[:offset]
        )

    def resolve_moves(self, proposals):
        """
        Resolve proposed one-cell movements simultaneously.

        Parameters
        ----------
        proposals:
            Dictionary mapping agent IDs to target grid positions.

        Returns
        -------
        dict:
            Accepted movements: agent ID -> target position.
        """
        agent_by_id = {
            agent.agent_id: agent for agent in self.agents
        }

        positions = {
            agent.agent_id: tuple(agent.position)
            for agent in self.agents
        }

        if len(set(positions.values())) != len(positions):
            raise RuntimeError("Current agent positions overlap.")

        unknown_ids = set(proposals) - set(agent_by_id)

        if unknown_ids:
            raise ValueError(f"Unknown agent IDs: {unknown_ids}")

        # Validate proposals and discard unsafe destinations.
        valid = {}
        invalid_proposals = 0

        for agent_id, target in proposals.items():
            target = tuple(target)
            current = positions[agent_id]

            if not self.maze.is_inside(*target):
                invalid_proposals += 1
                continue

            if not self.maze.is_walkable(*target):
                invalid_proposals += 1
                continue

            if abs(target[0] - current[0]) + abs(target[1] - current[1]) != 1:
                invalid_proposals += 1
                continue

            valid[agent_id] = target

        # Resolve multiple robots targeting the same cell.
        priority = [
            agent.agent_id for agent in self._priority_order()
        ]

        priority_rank = {
            agent_id: index
            for index, agent_id in enumerate(priority)
        }

        target_groups = {}

        for agent_id, target in valid.items():
            target_groups.setdefault(target, []).append(agent_id)

        candidates = set(valid)
        destination_conflicts = 0

        for contenders in target_groups.values():
            if len(contenders) > 1:
                winner = min(
                    contenders,
                    key=lambda agent_id: priority_rank[agent_id],
                )

                for agent_id in contenders:
                    if agent_id != winner:
                        candidates.discard(agent_id)
                        self.total_conflicts += 1
                        destination_conflicts += 1

        # Determine which agent currently occupies each cell.
        occupant_by_position = {
            position: agent_id
            for agent_id, position in positions.items()
        }

        # Reject swaps and longer movement cycles.
        dependency = {}

        for agent_id in candidates:
            occupant = occupant_by_position.get(valid[agent_id])

            if occupant is not None and occupant != agent_id:
                dependency[agent_id] = occupant

        rejected_cycles = set()
        globally_seen = set()

        for start_id in dependency:
            if start_id in globally_seen:
                continue

            chain = []
            local_index = {}
            current = start_id

            while current in dependency:
                if current in local_index:
                    cycle_start = local_index[current]
                    rejected_cycles.update(chain[cycle_start:])
                    break

                if current in globally_seen:
                    break

                local_index[current] = len(chain)
                chain.append(current)
                current = dependency[current]

            globally_seen.update(chain)

        candidates.difference_update(rejected_cycles)
        cycle_rejections = len(rejected_cycles)
        self.last_rejected_cycles = sorted(rejected_cycles)

        # Remove movements whose target occupant cannot leave.
        dependency_blocked = 0
        changed = True

        while changed:
            changed = False

            for agent_id in list(candidates):
                occupant = occupant_by_position.get(valid[agent_id])

                if occupant is not None and occupant != agent_id:
                    if occupant not in candidates:
                        candidates.remove(agent_id)
                        dependency_blocked += 1
                        changed = True

        accepted = {
            agent_id: valid[agent_id]
            for agent_id in candidates
        }

        blocked = len(proposals) - len(accepted)

        resolution_metrics = {
            "invalid_proposals": invalid_proposals,
            "destination_conflicts": destination_conflicts,
            "cycle_rejections": cycle_rejections,
            "dependency_blocked": dependency_blocked,
        }

        classified_blocked = sum(resolution_metrics.values())
        if classified_blocked != blocked:
            raise RuntimeError(
                "Grid rejection accounting mismatch: "
                f"{classified_blocked} classified, {blocked} blocked."
            )

        self.last_resolution_metrics = resolution_metrics
        self.total_invalid_proposals += invalid_proposals
        self.total_destination_conflicts += destination_conflicts
        self.total_cycle_rejections += cycle_rejections
        self.total_dependency_blocked += dependency_blocked

        self.total_blocked_moves += blocked
        self.total_successful_moves += len(accepted)

        # Rotate priority for the next simultaneous movement cycle.
        if self.agents:
            self.priority_offset = (
                self.priority_offset + 1
            ) % len(self.agents)

        return accepted

    def _filter_physical_moves(
        self,
        accepted,
        interval_start,
        interval_end,
        obstacle_trajectories=(),
        priority_order=None,
    ):
        """Reject simultaneous moves that violate continuous safety."""
        filtered = dict(accepted)
        obstacle_safety_blocks = 0
        robot_safety_blocks = 0

        if priority_order is None:
            priority_order = self._priority_order()

        priority_rank = {
            agent.agent_id: index
            for index, agent in enumerate(priority_order)
        }

        # Check every accepted robot trajectory against moving obstacles.
        for agent in priority_order:
            agent_id = agent.agent_id

            if agent_id not in filtered:
                continue

            start = tuple(float(v) for v in agent.position)
            end = tuple(float(v) for v in filtered[agent_id])

            safe = all(
                is_safe_during_motion_to_obstacle(
                    robot_start=start,
                    robot_end=end,
                    obstacle=trajectory,
                    interval_start=interval_start,
                    interval_end=interval_end,
                )
                for trajectory in obstacle_trajectories
            )

            if not safe:
                del filtered[agent_id]
                obstacle_safety_blocks += 1

        # Resolve continuous robot-robot conflicts. Recompute after each
        # rejection because a stopped robot has a stationary trajectory.
        while True:
            rejected_id = None

            for index, agent_a in enumerate(self.agents):
                start_a = tuple(float(v) for v in agent_a.position)
                end_a = tuple(
                    float(v)
                    for v in filtered.get(agent_a.agent_id, agent_a.position)
                )

                for agent_b in self.agents[index + 1:]:
                    start_b = tuple(float(v) for v in agent_b.position)
                    end_b = tuple(
                        float(v)
                        for v in filtered.get(
                            agent_b.agent_id,
                            agent_b.position,
                        )
                    )

                    if robots_are_safe_during_motion(
                        start_a,
                        end_a,
                        start_b,
                        end_b,
                    ):
                        continue

                    moving = [
                        agent_id
                        for agent_id in (
                            agent_a.agent_id,
                            agent_b.agent_id,
                        )
                        if agent_id in filtered
                    ]

                    if not moving:
                        continue

                    # If both move, retain the higher-priority movement.
                    # If only one moves, that movement must stop.
                    rejected_id = max(
                        moving,
                        key=lambda agent_id: priority_rank[agent_id],
                    )
                    break

                if rejected_id is not None:
                    break

            if rejected_id is None:
                break

            del filtered[rejected_id]
            robot_safety_blocks += 1
            rejected_agent = next(
                agent
                for agent in self.agents
                if agent.agent_id == rejected_id
            )
            # A prevented collision is not an actual collision.

        self.last_physical_metrics = {
            "obstacle_safety_blocks": obstacle_safety_blocks,
            "robot_safety_blocks": robot_safety_blocks,
        }
        self.total_obstacle_safety_blocks += obstacle_safety_blocks
        self.total_robot_safety_blocks += robot_safety_blocks

        if sum(self.last_physical_metrics.values()) != (
            len(accepted) - len(filtered)
        ):
            raise RuntimeError("Physical safety rejection accounting mismatch.")

        return filtered

    def _refresh_recovery_memory(self, agent):
        """Clear excluded waypoints only after genuine topological progress."""
        agent_id = agent.agent_id
        goal = tuple(agent.goal)
        distances = RecoveryPlanner._distances_to_goal(self.maze, goal)
        current_distance = distances.get(tuple(agent.position))

        memory = self._recovery_memory.get(agent_id)

        if memory is None or memory["goal"] != goal:
            self._recovery_memory[agent_id] = {
                "goal": goal,
                "best_distance": current_distance,
                "excluded_waypoints": [],
                "pending_recovery_position": None,
            }
            return

        pending_position = memory.get("pending_recovery_position")

        if pending_position is not None:
            memory["pending_recovery_position"] = None
            if tuple(agent.position) == pending_position:
                return

        best_distance = memory["best_distance"]

        if (
            current_distance is not None
            and (best_distance is None or current_distance < best_distance)
        ):
            memory["best_distance"] = current_distance
            memory["excluded_waypoints"].clear()

    def _remember_recovery_waypoint(self, agent, waypoint):
        """Remember a temporary waypoint after the agent reaches it."""
        memory = self._recovery_memory.get(agent.agent_id)

        if memory is None or memory["goal"] != tuple(agent.goal):
            self._refresh_recovery_memory(agent)
            memory = self._recovery_memory[agent.agent_id]
        waypoint = tuple(waypoint)
        memory["pending_recovery_position"] = tuple(agent.position)
        excluded = memory["excluded_waypoints"]

        if waypoint in excluded:
            excluded.remove(waypoint)

        excluded.append(waypoint)
        del excluded[:-self._max_remembered_recovery_waypoints]

    def _excluded_recovery_waypoints(self, agent):
        """Return the bounded set of waypoints excluded for this agent."""
        self._refresh_recovery_memory(agent)
        memory = self._recovery_memory[agent.agent_id]
        return set(memory["excluded_waypoints"])

    def _planner_for(self, agent):
        """Return the temporary planner during recovery, otherwise the original."""
        return self._recovery_planners.get(
            agent.agent_id,
            getattr(agent, "planner", None),
        )

    def _is_incremental_planner(self, agent):
        """Return True if the active planner supports incremental map updates."""
        planner = self._planner_for(agent)

        return (
            callable(getattr(planner, "notify_changed_cells", None))
            and callable(getattr(planner, "move_start", None))
        )

    def _defer_recovery_retry(self, agent_id):
        """Schedule the next recovery attempt using simulation time."""
        self._recovery_retry_after[agent_id] = (
            self.simulation_time + self.recovery_retry_interval
        )

    def _record_recovery_failure(self, agent, reason):
        """Record a failed recovery attempt and defer or exhaust retries."""
        agent_id = agent.agent_id
        attempt = self._recovery_attempts.get(agent_id, 0)

        if attempt >= self.max_recovery_attempts:
            agent.mark_recovery_failed()
            self._recovery_retry_after.pop(agent_id, None)
            event = {
                "event": "exhausted",
                "agent_id": agent_id,
                "attempt": attempt,
                "reason": reason,
                "recovery_state": agent.recovery_state,
            }
        else:
            self._defer_recovery_retry(agent_id)
            event = {
                "event": "attempt_failed",
                "agent_id": agent_id,
                "attempt": attempt,
                "reason": reason,
                "retry_after": self._recovery_retry_after[agent_id],
                "recovery_state": agent.recovery_state,
            }

        self._step_recovery_events.append(event)

    def _try_start_recovery(self, agent, obstacle_trajectories=()):
        """Start a validated temporary recovery path for a persistently blocked agent."""
        agent_id = agent.agent_id

        if getattr(agent, "recovery_state", None) in {"active", "failed"}:
            return False

        required_recovery_methods = (
            "begin_recovery",
            "mark_recovery_failed",
            "reset_recovery",
        )
        if not all(
            callable(getattr(agent, name, None))
            for name in required_recovery_methods
        ):
            return False

        if not self.recovery_coordinator.is_blocked(agent_id):
            self._recovery_retry_after.pop(agent_id, None)
            return False

        retry_after = self._recovery_retry_after.get(agent_id)
        if retry_after is not None and self.simulation_time < retry_after:
            return False

        self._recovery_attempts[agent_id] = (
            self._recovery_attempts.get(agent_id, 0) + 1
        )
        attempt_number = self._recovery_attempts[agent_id]

        excluded_waypoints = self._excluded_recovery_waypoints(agent)

        plan = self.recovery_planner.plan(
            agent=agent,
            maze=self.maze,
            dynamic_environment=self.dynamic_environment,
            agents=self.agents,
            simulation_time=self.simulation_time,
            obstacle_trajectories=obstacle_trajectories,
            excluded_waypoints=excluded_waypoints,
        )

        if plan is None:
            self._record_recovery_failure(agent, "no_valid_plan")
            return False

        path = [tuple(point) for point in plan.path]
        waypoint = tuple(plan.waypoint)

        if (
            len(path) < 2
            or path[0] != tuple(agent.position)
            or path[-1] != waypoint
            or waypoint == tuple(agent.goal)
        ):
            self._record_recovery_failure(agent, "invalid_path")
            return False

        for index, point in enumerate(path):
            if not self.maze.is_walkable(*point):
                self._record_recovery_failure(agent, "non_walkable_cell")
                return False

            if index:
                previous = path[index - 1]
                if (
                    abs(point[0] - previous[0])
                    + abs(point[1] - previous[1])
                    != 1
                ):
                    self._record_recovery_failure(agent, "non_adjacent_path_step")
                    return False

        agent.begin_recovery(waypoint)
        agent.path = path
        agent.metrics["path_length"] = sum(
            (
                (path[i][0] - path[i - 1][0]) ** 2
                + (path[i][1] - path[i - 1][1]) ** 2
            ) ** 0.5
            for i in range(1, len(path))
        )
        agent.metrics["replans"] += 1

        self._recovery_planners[agent_id] = plan.planner
        self._recovery_retry_after.pop(agent_id, None)
        self._recovery_attempts.pop(agent_id, None)
        agent.status = "planned"
        self.recovery_coordinator.reset_agent(agent_id)
        self._step_recovery_events.append({
            "event": "started",
            "agent_id": agent_id,
            "attempt": attempt_number,
            "start": tuple(agent.position),
            "waypoint": waypoint,
            "path": path,
            "planner_name": plan.planner_name,
            "family": plan.family,
            "candidates_examined": plan.candidates_examined,
        })

        return True

    def step(self, simulation_time=None):
        """
        Execute one simultaneous movement cycle.

        Dynamic map changes are forwarded to incremental planners.
        D* Lite agents preserve planner state across simulation cycles.
        """
        if simulation_time is None:
            self._step_count += 1
            next_time = round(
                self._clock_origin
                + self._step_count * self.time_step,
                12,
            )
        else:
            next_time = float(simulation_time)

            if next_time < self.simulation_time:
                raise ValueError("Simulation time cannot move backwards.")

            if next_time < 0:
                raise ValueError("Simulation time cannot be negative.")

            self._clock_origin = next_time
            self._step_count = 0

        if next_time < self.simulation_time:
            raise ValueError("Simulation time cannot move backwards.")

        if next_time < 0:
            raise ValueError("Simulation time cannot be negative.")

        previous_time = self.simulation_time
        self.simulation_time = next_time
        self._step_recovery_events = []

        # Phase 0: update the dynamic environment.
        if self.dynamic_environment is not None:
            occupied_cells = {
                tuple(agent.position) for agent in self.agents
            }

            environment_updates = self.dynamic_environment.update(
                self.simulation_time,
                occupied_cells=occupied_cells,
            )
        else:
            environment_updates = {
                "walls_updated": False,
                "obstacles_updated": False,
                "changed_cells": [],
            }

        changed_cells = environment_updates.get("changed_cells", [])
        replanned_count = 0

        # Phase 1: synchronize planners with dynamic map changes.
        changed_cell_set = {
            tuple(cell) for cell in changed_cells
        }

        for agent in self.agents:
            if agent.reached_goal:
                continue

            self._try_start_recovery(
                agent,
                obstacle_trajectories=environment_updates.get(
                    "obstacle_trajectories",
                    (),
                ),
            )

            planner = self._planner_for(agent)
            incremental = self._is_incremental_planner(agent)

            # Synchronize only the planner currently controlling navigation.
            if changed_cells and incremental:
                planner.notify_changed_cells(changed_cells)

            # Replan only if a changed cell intersects the remaining path.
            remaining_path = set()

            if agent.path:
                try:
                    current_index = agent.path.index(agent.position)
                    remaining_path = {
                        tuple(cell)
                        for cell in agent.path[current_index + 1:]
                    }
                except ValueError:
                    remaining_path = {
                        tuple(cell) for cell in agent.path
                    }

            path_affected = bool(
                changed_cell_set.intersection(remaining_path)
            )

            must_replan = (
                path_affected
                or agent.status == "needs_replanning"
            )

            if must_replan:
                if getattr(agent, "recovery_state", None) == "active":
                    agent.compute_plan(self.maze, planner=planner)
                else:
                    agent.compute_plan(self.maze)
                replanned_count += 1

        positions_before = {
            agent.agent_id: tuple(agent.position)
            for agent in self.agents
        }

        proposals = {}

        # Phase 2: collect movement intentions.
        for agent in self.agents:
            if agent.reached_goal:
                agent.status = "reached_goal"
                continue

            if agent.status not in {"planned", "moving", "waiting"}:
                continue

            try:
                current_index = agent.path.index(agent.position)
            except ValueError:
                agent.status = "needs_replanning"
                continue

            if current_index + 1 >= len(agent.path):
                agent.status = "needs_replanning"
                continue

            target = tuple(agent.path[current_index + 1])

            if not self.maze.is_walkable(*target):
                agent.status = "needs_replanning"

                # Preserve the temporary target and planner during recovery.
                if getattr(agent, "recovery_state", None) == "active":
                    agent.compute_plan(
                        self.maze,
                        planner=self._planner_for(agent),
                    )
                else:
                    agent.compute_plan(self.maze)
                replanned_count += 1

                if len(agent.path) < 2:
                    continue

                try:
                    current_index = agent.path.index(agent.position)
                except ValueError:
                    agent.status = "needs_replanning"
                    continue

                if current_index + 1 >= len(agent.path):
                    agent.status = "needs_replanning"
                    continue

                target = tuple(agent.path[current_index + 1])

                if not self.maze.is_walkable(*target):
                    agent.status = "needs_replanning"
                    continue

            proposals[agent.agent_id] = target

        # Preserve planner intentions separately from coordinated intentions.
        original_proposals = dict(proposals)
        coordinated_proposals = dict(original_proposals)
        traffic_yields = frozenset()

        if self.traffic_coordinator is not None:
            coordination = (
                self.traffic_coordinator.coordinate_with_yields(
                    self.agents,
                    original_proposals,
                )
            )
            coordinated_proposals = dict(coordination.proposals)
            traffic_yields = frozenset(coordination.yielded_agents)

            unknown_yields = traffic_yields - set(original_proposals)
            unknown_proposals = (
                set(coordinated_proposals) - set(original_proposals)
            )
            overlapping_yields = (
                traffic_yields & set(coordinated_proposals)
            )

            if unknown_yields or unknown_proposals or overlapping_yields:
                raise ValueError(
                    "Invalid traffic coordination result: "
                    f"unknown yields={unknown_yields}, "
                    f"unknown proposals={unknown_proposals}, "
                    f"yielded agents still proposing={overlapping_yields}."
                )

            unaccounted_proposals = (
                set(original_proposals)
                - set(coordinated_proposals)
                - traffic_yields
            )

            if unaccounted_proposals:
                raise ValueError(
                    "Invalid traffic coordination result: "
                    f"proposals not accounted for={unaccounted_proposals}."
                )

        # Phase 3: resolve grid conflicts, then enforce physical safety.
        priority_order = self._priority_order()
        accepted = self.resolve_moves(coordinated_proposals)

        accepted_before_safety = len(accepted)
        accepted = self._filter_physical_moves(
            accepted,
            interval_start=previous_time,
            interval_end=self.simulation_time,
            obstacle_trajectories=environment_updates.get(
                "obstacle_trajectories",
                (),
            ),
            priority_order=priority_order,
        )

        safety_blocked = accepted_before_safety - len(accepted)
        self.total_successful_moves -= safety_blocked
        self.total_blocked_moves += safety_blocked

        # Phase 4: apply movements and synchronize D* Lite start states.
        for agent in self.agents:
            agent_id = agent.agent_id

            if agent_id in accepted:
                agent.position = accepted[agent_id]
                agent.metrics["moves"] += 1

                planner = self._planner_for(agent)
                if self._is_incremental_planner(agent):
                    planner.move_start(agent.position)

                if getattr(agent, "recovery_state", None) == "failed":
                    agent.reset_recovery()
                    self._recovery_attempts.pop(agent_id, None)
                    self._recovery_retry_after.pop(agent_id, None)

                if agent.reached_goal:
                    agent.status = "reached_goal"

                elif (
                    getattr(agent, "recovery_state", None) == "active"
                    and getattr(agent, "reached_navigation_target", False)
                ):
                    self._step_recovery_events.append({
                        "event": "completed",
                        "agent_id": agent_id,
                        "position": tuple(agent.position),
                        "waypoint": tuple(agent.navigation_target),
                    })
                    self._remember_recovery_waypoint(
                        agent,
                        agent.navigation_target,
                    )
                    agent.complete_recovery()
                    self._recovery_planners.pop(agent_id, None)
                    self._recovery_attempts.pop(agent_id, None)
                    self._recovery_retry_after.pop(agent_id, None)
                    self.recovery_coordinator.reset_agent(agent_id)

                else:
                    agent.status = "moving"
                    if getattr(agent, "recovery_state", None) != "active":
                        self._refresh_recovery_memory(agent)

            elif (
                agent_id in coordinated_proposals
                or agent_id in traffic_yields
            ):
                agent.status = "waiting"

        # Phase 5: voluntary yields preserve recovery history.
        for agent in self.agents:
            agent_id = agent.agent_id

            if agent_id in traffic_yields:
                self.recovery_coordinator.pause_agent(agent_id)
            elif agent_id in original_proposals:
                self.recovery_coordinator.observe(
                    agent_id,
                    agent.position,
                    self.simulation_time,
                )
            else:
                self.recovery_coordinator.reset_agent(agent_id)

        return {
            "proposed": len(original_proposals),
            "coordinated_proposed": len(coordinated_proposals),
            "traffic_yielded": len(traffic_yields),
            "traffic_yields": tuple(sorted(traffic_yields)),
            "moved": len(accepted),
            "blocked": len(coordinated_proposals) - len(accepted),
            "safety_blocked": safety_blocked,
            **self.last_resolution_metrics,
            **self.last_physical_metrics,
            "replanned": replanned_count,
            "simulation_time": self.simulation_time,
            "recovery_events": [dict(event) for event in self._step_recovery_events],
            "environment_updates": environment_updates,
            "positions_before": positions_before,
            "proposals": {
                agent_id: tuple(target)
                for agent_id, target in original_proposals.items()
            },
            "coordinated_proposals": {
                agent_id: tuple(target)
                for agent_id, target in coordinated_proposals.items()
            },
            "rejected_cycles": list(self.last_rejected_cycles),
            "positions": {
                agent.agent_id: tuple(agent.position)
                for agent in self.agents
            },
        }

    def get_metrics(self):
        """Return aggregate movement-resolution metrics."""
        return {
            "agents": len(self.agents),
            "successful_moves": self.total_successful_moves,
            "blocked_moves": self.total_blocked_moves,
            "conflicts": self.total_conflicts,
            "invalid_proposals": self.total_invalid_proposals,
            "destination_conflicts": self.total_destination_conflicts,
            "cycle_rejections": self.total_cycle_rejections,
            "dependency_blocked": self.total_dependency_blocked,
            "obstacle_safety_blocks": self.total_obstacle_safety_blocks,
            "robot_safety_blocks": self.total_robot_safety_blocks,
        }
