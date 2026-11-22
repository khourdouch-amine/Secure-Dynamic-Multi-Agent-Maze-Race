"""Independent coordination of conflicting multi-agent traffic intentions."""

from collections import deque
from dataclasses import dataclass

from security.physical.geometry import minimum_distance_during_motion


@dataclass(frozen=True)
class TrafficCoordinationResult:
    """Adjusted movement intentions and explicitly identified voluntary yields."""

    proposals: dict
    yielded_agents: frozenset


class TrafficCoordinator:
    """Resolve a two-agent reciprocal swap using a safe joint-space search."""

    MIN_CENTER_DISTANCE = 1.0

    def __init__(self, maze):
        self.maze = maze

    def detect_destination_conflicts(self, agents, proposals):
        """Return deterministic groups targeting the same destination."""
        known_agents = {agent.agent_id for agent in agents}
        unknown_agents = set(proposals) - known_agents
        if unknown_agents:
            raise ValueError(
                f"Proposals contain unknown agents: {sorted(unknown_agents)}"
            )

        agents_by_destination = {}
        for agent_id, destination in proposals.items():
            destination = tuple(destination)
            agents_by_destination.setdefault(destination, []).append(agent_id)

        conflicts = [
            (destination, tuple(sorted(agent_ids)))
            for destination, agent_ids in agents_by_destination.items()
            if len(agent_ids) >= 2
        ]
        return tuple(sorted(conflicts, key=lambda item: item[0]))

    def detect_trajectory_conflicts(self, agents, proposals):
        """Return deterministic pairs whose synchronized motions violate the safety distance."""
        known_agents = {agent.agent_id for agent in agents}
        unknown_agents = set(proposals) - known_agents
        if unknown_agents:
            raise ValueError(
                f"Proposals contain unknown agents: {sorted(unknown_agents)}"
            )

        positions = {
            agent.agent_id: tuple(agent.position)
            for agent in agents
        }
        destinations = {
            agent_id: tuple(proposals.get(agent_id, positions[agent_id]))
            for agent_id in positions
        }

        agent_ids = sorted(positions)
        conflicts = []

        for index, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[index + 1:]:
                minimum_distance = minimum_distance_during_motion(
                    positions[agent_a],
                    destinations[agent_a],
                    positions[agent_b],
                    destinations[agent_b],
                )
                if minimum_distance < self.MIN_CENTER_DISTANCE - 1e-9:
                    conflicts.append((agent_a, agent_b))

        return tuple(conflicts)

    def detect_swaps(self, agents, proposals):
        """Return deterministic pairs performing reciprocal position exchanges."""
        positions = {
            agent.agent_id: tuple(agent.position)
            for agent in agents
        }

        unknown_agents = set(proposals) - set(positions)
        if unknown_agents:
            raise ValueError(
                f"Proposals contain unknown agents: {sorted(unknown_agents)}"
            )

        destinations = {
            agent_id: tuple(target)
            for agent_id, target in proposals.items()
        }

        occupant_by_position = {
            position: agent_id
            for agent_id, position in positions.items()
        }

        swaps = set()

        for agent_id, destination in destinations.items():
            other_id = occupant_by_position.get(destination)

            if other_id is None or other_id == agent_id:
                continue

            if destinations.get(other_id) == positions[agent_id]:
                swaps.add(tuple(sorted((agent_id, other_id))))

        return tuple(sorted(swaps))

    def detect_dependencies(self, agents, proposals):
        """Return deterministic (requesting_agent, occupying_agent) pairs."""
        positions = {
            agent.agent_id: tuple(agent.position)
            for agent in agents
        }

        unknown_agents = set(proposals) - set(positions)
        if unknown_agents:
            raise ValueError(
                f"Proposals contain unknown agents: {sorted(unknown_agents)}"
            )

        occupant_by_position = {
            position: agent_id
            for agent_id, position in positions.items()
        }

        dependencies = set()

        for agent_id, destination in proposals.items():
            destination = tuple(destination)
            occupant_id = occupant_by_position.get(destination)

            if occupant_id is not None and occupant_id != agent_id:
                dependencies.add((agent_id, occupant_id))

        return tuple(sorted(dependencies))

    def detect_cycles(self, agents, proposals):
        """Return deterministic directed dependency cycles of length >= 3.

        Reciprocal two-agent exchanges are handled separately by detect_swaps().
        """
        positions = {
            agent.agent_id: tuple(agent.position)
            for agent in agents
        }

        unknown_agents = set(proposals) - set(positions)
        if unknown_agents:
            raise ValueError(
                f"Proposals contain unknown agents: {sorted(unknown_agents)}"
            )

        occupant_by_position = {
            position: agent_id
            for agent_id, position in positions.items()
        }

        dependency = {}
        for agent_id, destination in proposals.items():
            occupant_id = occupant_by_position.get(tuple(destination))
            if occupant_id is not None and occupant_id != agent_id:
                dependency[agent_id] = occupant_id

        cycles = set()
        globally_seen = set()

        for start in sorted(dependency):
            if start in globally_seen:
                continue

            path = []
            local_index = {}
            current = start

            while current in dependency and current not in globally_seen:
                if current in local_index:
                    cycle = path[local_index[current]:]

                    if len(cycle) >= 3:
                        smallest_index = min(
                            range(len(cycle)),
                            key=lambda index: cycle[index],
                        )
                        canonical_cycle = (
                            tuple(cycle[smallest_index:])
                            + tuple(cycle[:smallest_index])
                        )
                        cycles.add(canonical_cycle)
                    break

                local_index[current] = len(path)
                path.append(current)
                current = dependency[current]

            globally_seen.update(path)

        return tuple(sorted(cycles))

    MAX_GROUP_SIZE = 3
    MAX_SEARCH_STATES = 5000
    MAX_SEARCH_DEPTH = 64

    def coordinate(self, agents, proposals):
        """Coordinate bounded conflict groups of two or three proposing agents."""
        original = dict(proposals)
        agents_by_id = {agent.agent_id: agent for agent in agents}

        if len(agents_by_id) != len(agents):
            raise ValueError("Agent identifiers must be unique.")

        # Preserve proposals belonging to agents outside this coordination
        # call; the manager remains responsible for validating its own agents.

        # Only agents with an actual movement intention can be coordinated.
        proposing_agents = [
            agent for agent in agents
            if agent.agent_id in original
        ]
        known_proposals = {
            agent.agent_id: original[agent.agent_id]
            for agent in proposing_agents
        }

        # Preserve the established two-agent reciprocal-swap behavior.
        if len(agents) == 2:
            agent_a, agent_b = agents
            position_a = tuple(agent_a.position)
            position_b = tuple(agent_b.position)
            proposal_a = original.get(agent_a.agent_id)
            proposal_b = original.get(agent_b.agent_id)

            reciprocal_swap = (
                proposal_a is not None
                and proposal_b is not None
                and tuple(proposal_a) == position_b
                and tuple(proposal_b) == position_a
            )

            if reciprocal_swap:
                joint_path = self._find_safe_joint_path(
                    position_a,
                    tuple(getattr(agent_a, "navigation_target", agent_a.goal)),
                    position_b,
                    tuple(getattr(agent_b, "navigation_target", agent_b.goal)),
                )
                if joint_path is not None and len(joint_path) >= 2:
                    next_a, next_b = joint_path[1]
                    adjusted = dict(original)
                    adjusted.pop(agent_a.agent_id, None)
                    adjusted.pop(agent_b.agent_id, None)
                    if next_a != position_a:
                        adjusted[agent_a.agent_id] = next_a
                    if next_b != position_b:
                        adjusted[agent_b.agent_id] = next_b
                    return adjusted

        adjacency = {
            agent.agent_id: set()
            for agent in proposing_agents
        }

        for agent_a, agent_b in self.detect_trajectory_conflicts(
            proposing_agents, known_proposals
        ):
            adjacency[agent_a].add(agent_b)
            adjacency[agent_b].add(agent_a)

        components = []
        visited = set()

        for start_id in sorted(adjacency):
            if start_id in visited or not adjacency[start_id]:
                continue

            stack = [start_id]
            visited.add(start_id)
            component = []

            while stack:
                current = stack.pop()
                component.append(current)

                for neighbor in sorted(
                    adjacency[current], reverse=True
                ):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            components.append(tuple(sorted(component)))

        adjusted = dict(original)

        for component in components:
            if not 2 <= len(component) <= self.MAX_GROUP_SIZE:
                continue

            group = [
                agents_by_id[agent_id]
                for agent_id in component
            ]

            starts = tuple(
                tuple(agent.position) for agent in group
            )
            goals = tuple(
                tuple(getattr(agent, "navigation_target", agent.goal))
                for agent in group
            )

            next_state = self._select_safe_joint_step(starts, goals)

            if next_state is None:
                continue

            for agent, destination in zip(group, next_state):
                adjusted.pop(agent.agent_id, None)

                if destination != tuple(agent.position):
                    adjusted[agent.agent_id] = destination

        return adjusted

    def coordinate_with_yields(self, agents, proposals):
        """Return adjusted proposals and explicitly identified voluntary yields."""
        original = dict(proposals)
        adjusted = self.coordinate(agents, original)

        return TrafficCoordinationResult(
            proposals=adjusted,
            yielded_agents=frozenset(
                set(original) - set(adjusted)
            ),
        )

    def _select_safe_joint_step(self, starts, goals):
        """Select one safe joint movement using a bounded local search."""
        from itertools import product

        starts = tuple(tuple(position) for position in starts)
        goals = tuple(tuple(position) for position in goals)

        if not 2 <= len(starts) <= self.MAX_GROUP_SIZE:
            return None
        if len(starts) != len(goals):
            return None

        def distance(position, goal):
            return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

        current_score = sum(
            distance(position, goal)
            for position, goal in zip(starts, goals)
        )

        action_sets = []
        for position in starts:
            candidates = [position]
            candidates.extend(
                tuple(neighbor)
                for neighbor in self.maze.get_neighbors(*position)
            )
            action_sets.append(tuple(dict.fromkeys(candidates)))

        best_state = None
        best_score = None

        for actions in product(*action_sets):
            state = tuple(tuple(position) for position in actions)

            if state == starts:
                continue

            safe = all(
                minimum_distance_during_motion(
                    starts[i], state[i],
                    starts[j], state[j],
                ) >= self.MIN_CENTER_DISTANCE - 1e-9
                for i in range(len(starts))
                for j in range(i + 1, len(starts))
            )
            if not safe:
                continue

            progress_score = sum(
                distance(position, goal)
                for position, goal in zip(state, goals)
            )
            if progress_score >= current_score:
                continue

            moved_count = sum(
                destination != start
                for start, destination in zip(starts, state)
            )
            score = (progress_score, -moved_count, state)

            if best_score is None or score < best_score:
                best_score = score
                best_state = state

        return best_state

    def _find_safe_joint_path_for_group(self, starts, goals):
        """Find a bounded synchronized path for two or three agents."""
        from itertools import product

        start = tuple(tuple(position) for position in starts)
        goal = tuple(tuple(position) for position in goals)

        if (
            not 2 <= len(start) <= self.MAX_GROUP_SIZE
            or len(start) != len(goal)
        ):
            return None

        if start == goal:
            return [start]

        frontier = deque([start])
        parent = {start: None}
        depth = {start: 0}

        while frontier and len(parent) < self.MAX_SEARCH_STATES:
            state = frontier.popleft()
            current_depth = depth[state]

            if state == goal:
                path = []
                current = state

                while current is not None:
                    path.append(current)
                    current = parent[current]

                return list(reversed(path))

            if current_depth >= self.MAX_SEARCH_DEPTH:
                continue

            action_sets = [
                (position, *self.maze.get_neighbors(*position))
                for position in state
            ]

            for actions in product(*action_sets):
                next_state = tuple(
                    tuple(position) for position in actions
                )

                if next_state in parent:
                    continue

                safe = all(
                    minimum_distance_during_motion(
                        state[i], next_state[i],
                        state[j], next_state[j],
                    ) >= self.MIN_CENTER_DISTANCE - 1e-9
                    for i in range(len(state))
                    for j in range(i + 1, len(state))
                )

                if not safe:
                    continue

                parent[next_state] = state
                depth[next_state] = current_depth + 1

                if next_state == goal:
                    path = []
                    current = next_state

                    while current is not None:
                        path.append(current)
                        current = parent[current]

                    return list(reversed(path))

                frontier.append(next_state)

                if len(parent) >= self.MAX_SEARCH_STATES:
                    break

        return None

    def _find_safe_joint_path(self, start_a, goal_a, start_b, goal_b):
        start = (tuple(start_a), tuple(start_b))
        goal = (tuple(goal_a), tuple(goal_b))

        frontier = deque([start])
        parent = {start: None}

        while frontier:
            state = frontier.popleft()

            if state == goal:
                path = []
                current = state
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return list(reversed(path))

            position_a, position_b = state
            actions_a = [position_a, *self.maze.get_neighbors(*position_a)]
            actions_b = [position_b, *self.maze.get_neighbors(*position_b)]

            for next_a in actions_a:
                for next_b in actions_b:
                    next_state = (tuple(next_a), tuple(next_b))

                    if next_state in parent:
                        continue

                    minimum_distance = minimum_distance_during_motion(
                        position_a,
                        next_a,
                        position_b,
                        next_b,
                    )

                    if minimum_distance < self.MIN_CENTER_DISTANCE - 1e-9:
                        continue

                    parent[next_state] = state
                    frontier.append(next_state)

        return None
