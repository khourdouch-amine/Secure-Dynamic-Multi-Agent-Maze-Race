"""
Hierarchical local quantum-walk navigation.

A classical global BFS provides regional guidance.
A simulated coined quantum walk selects local movements
inside a bounded window.

This is a hybrid quantum-classical navigation planner.
It does not claim quantum speedup.
"""

from collections import deque

from planners.quantum.quantum_base import QuantumPlannerBase
from planners.quantum.local_maze_view import LocalMazeView
from planners.quantum.quantum_walk_navigation import (
    QuantumWalkNavigationPlanner,
)


class QuantumWalkLocalNavigationPlanner(QuantumPlannerBase):
    """Navigate a large maze using successive local quantum walks."""

    def __init__(
        self,
        window_size=7,
        coin_type="grover",
        shots=64,
        walk_steps=4,
        max_total_qubits=8,
        seed=42,
        max_segments=1000,
    ):
        super().__init__("Hierarchical Local Quantum Walk")

        if not isinstance(window_size, int) or window_size < 3:
            raise ValueError("window_size must be an integer >= 3.")

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")

        if coin_type not in {"hadamard", "grover"}:
            raise ValueError("Unsupported coin type.")

        if not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be positive.")

        if not isinstance(walk_steps, int) or walk_steps < 0:
            raise ValueError("walk_steps must be non-negative.")

        if (
            not isinstance(max_total_qubits, int)
            or max_total_qubits < 3
        ):
            raise ValueError("max_total_qubits must be >= 3.")

        if not isinstance(max_segments, int) or max_segments < 1:
            raise ValueError("max_segments must be positive.")

        self.window_size = window_size
        self.coin_type = coin_type
        self.shots = shots
        self.walk_steps = walk_steps
        self.max_total_qubits = max_total_qubits
        self.seed = seed
        self.max_segments = max_segments

        self.last_metadata = {}

    @staticmethod
    def _shortest_path(maze, start, goal):
        """Compute a global BFS reference path."""
        start = tuple(start)
        goal = tuple(goal)

        parents = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for neighbor in maze.get_neighbors(*current):
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)

        if goal not in parents:
            return None

        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = parents[current]

        return list(reversed(path))

    @staticmethod
    def _local_bfs(view, start, goal):
        """Classical fallback inside the local window."""
        start = tuple(start)
        goal = tuple(goal)

        parents = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for neighbor in view.get_neighbors(*current):
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)

        if goal not in parents:
            return None

        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = parents[current]

        return list(reversed(path))

    def _select_local_subgoal(self, view, global_path):
        """
        Select the furthest consecutive global-path cell
        contained in the current local window.
        """
        selected = None

        for global_cell in global_path[1:]:
            local_cell = view.to_local(global_cell)

            if not view.is_inside(*local_cell):
                break

            if not view.is_walkable(*local_cell):
                break

            selected = global_cell

        return selected

    def _new_metadata(self):
        return {
            "algorithm": self.name,
            "window_size": self.window_size,
            "coin_type": self.coin_type,
            "shots_per_circuit": self.shots,
            "walk_steps_per_evaluation": self.walk_steps,
            "max_total_qubits": self.max_total_qubits,
            "segments": 0,
            "global_guidance_calls": 0,
            "quantum_decisions": 0,
            "classical_fallback_decisions": 0,
            "quantum_circuit_executions": 0,
            "classical_fallback_segments": 0,
            "path_length": 0,
            "path_nodes": 0,
            "failure": None,
        }

    def plan(self, maze, start, goal):
        """Return a complete global path assembled from local segments."""
        start, goal = self.validate_problem(maze, start, goal)

        metadata = self._new_metadata()

        if start == goal:
            metadata["path_nodes"] = 1
            self.last_metadata = metadata
            return [start]

        current = start
        complete_path = [start]

        for _ in range(self.max_segments):

            # Phase 1: global classical guidance.
            global_path = self._shortest_path(
                maze,
                current,
                goal,
            )

            metadata["global_guidance_calls"] += 1

            if global_path is None:
                metadata["failure"] = "global goal unreachable"
                self.last_metadata = metadata
                return None

            if len(global_path) == 1:
                break

            # Phase 2: extract the local environment.
            view = LocalMazeView(
                maze,
                center=current,
                size=self.window_size,
            )

            local_start = view.to_local(current)

            # Phase 3: select a local subgoal.
            local_subgoal_global = self._select_local_subgoal(
                view,
                global_path,
            )

            if local_subgoal_global is None:
                metadata["failure"] = (
                    "no global-path cell inside local window"
                )
                self.last_metadata = metadata
                return None

            local_goal = view.to_local(local_subgoal_global)

            # Phase 4: local quantum-walk navigation.
            local_planner = QuantumWalkNavigationPlanner(
                view,
                coin_type=self.coin_type,
                shots=self.shots,
                walk_steps=self.walk_steps,
                seed=self.seed + metadata["segments"],
                max_total_qubits=self.max_total_qubits,
            )

            local_path = local_planner.plan(
                view,
                local_start,
                local_goal,
            )

            local_metadata = local_planner.last_metadata

            metadata["quantum_decisions"] += (
                local_metadata.get("quantum_decisions", 0)
            )

            metadata["classical_fallback_decisions"] += (
                local_metadata.get("classical_fallback_decisions", 0)
            )

            metadata["quantum_circuit_executions"] += (
                local_metadata.get("quantum_circuit_executions", 0)
            )

            # Phase 5: explicit classical fallback if necessary.
            if local_path is None:
                local_path = self._local_bfs(
                    view,
                    local_start,
                    local_goal,
                )

                metadata["classical_fallback_segments"] += 1

            if local_path is None or len(local_path) < 2:
                metadata["failure"] = "local navigation failed"
                self.last_metadata = metadata
                return None

            # Phase 6: convert the local path into global coordinates.
            global_segment = [
                view.to_global(cell)
                for cell in local_path
            ]

            if global_segment[0] != current:
                metadata["failure"] = "local path origin mismatch"
                self.last_metadata = metadata
                return None

            for cell in global_segment[1:]:
                if not maze.is_walkable(*cell):
                    metadata["failure"] = "local path crosses obstacle"
                    self.last_metadata = metadata
                    return None

                previous = complete_path[-1]

                distance = (
                    abs(cell[0] - previous[0])
                    + abs(cell[1] - previous[1])
                )

                if distance != 1:
                    metadata["failure"] = "discontinuous global path"
                    self.last_metadata = metadata
                    return None

                complete_path.append(cell)

            current = complete_path[-1]
            metadata["segments"] += 1

            if current == goal:
                break

        if current != goal:
            metadata["failure"] = "maximum segments reached"
            self.last_metadata = metadata
            return None

        self.validate_path(
            maze,
            complete_path,
            start,
            goal,
        )

        metadata["path_length"] = len(complete_path) - 1
        metadata["path_nodes"] = len(complete_path)

        self.last_metadata = metadata

        return complete_path
