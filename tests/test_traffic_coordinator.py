from collections import deque

from environment.maze import Maze
from security.physical.geometry import minimum_distance_during_motion


MIN_CENTER_DISTANCE = 1.0


def find_safe_joint_path(maze, start_a, goal_a, start_b, goal_b):
    """Find a shortest synchronized path for two robots with safe motions."""
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

        actions_a = [position_a, *maze.get_neighbors(*position_a)]
        actions_b = [position_b, *maze.get_neighbors(*position_b)]

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

                if minimum_distance < MIN_CENTER_DISTANCE - 1e-9:
                    continue

                parent[next_state] = state
                frontier.append(next_state)

    return None


def test_r05_r09_have_a_safe_coordinated_solution():
    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()

    r05_start = (7, 11)
    r05_goal = (5, 7)
    r09_start = (6, 11)
    r09_goal = (7, 1)

    joint_path = find_safe_joint_path(
        maze,
        r05_start,
        r05_goal,
        r09_start,
        r09_goal,
    )

    assert joint_path is not None
    assert joint_path[0] == (r05_start, r09_start)
    assert joint_path[-1] == (r05_goal, r09_goal)

    for previous, current in zip(joint_path, joint_path[1:]):
        previous_a, previous_b = previous
        current_a, current_b = current

        assert (
            minimum_distance_during_motion(
                previous_a,
                current_a,
                previous_b,
                current_b,
            )
            >= MIN_CENTER_DISTANCE - 1e-9
        )


def test_traffic_coordinator_resolves_reciprocal_swap():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()

    r05 = SimpleNamespace(
        agent_id="R05",
        position=(7, 11),
        goal=(5, 7),
    )
    r09 = SimpleNamespace(
        agent_id="R09",
        position=(6, 11),
        goal=(7, 1),
    )

    coordinator = TrafficCoordinator(maze)

    coordinated = coordinator.coordinate(
        [r05, r09],
        {
            "R05": (6, 11),
            "R09": (7, 11),
        },
    )

    # R05 waits: waiting is represented by the absence of a proposal.
    assert coordinated == {
        "R09": (5, 11),
    }


def test_traffic_coordinator_preserves_non_conflicting_proposals():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(3, 1)),
        SimpleNamespace(agent_id="R2", position=(1, 3), goal=(3, 3)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (2, 3),
    }

    result = TrafficCoordinator(maze).coordinate(agents, proposals)

    assert result == proposals


def test_traffic_coordinator_preserves_proposals_when_no_safe_path_exists():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    class DisconnectedMaze:
        def get_neighbors(self, x, y):
            return []

    agents = [
        SimpleNamespace(agent_id="R1", position=(0, 0), goal=(1, 0)),
        SimpleNamespace(agent_id="R2", position=(1, 0), goal=(0, 0)),
    ]
    proposals = {
        "R1": (1, 0),
        "R2": (0, 0),
    }

    result = TrafficCoordinator(DisconnectedMaze()).coordinate(
        agents, proposals
    )

    assert result == proposals


def test_traffic_coordinator_preserves_unrelated_agent_proposals():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()

    agents = [
        SimpleNamespace(agent_id="R05", position=(7, 11), goal=(5, 7)),
        SimpleNamespace(agent_id="R09", position=(6, 11), goal=(7, 1)),
    ]
    proposals = {
        "R05": (6, 11),
        "R09": (7, 11),
        "R03": (10, 10),
    }

    result = TrafficCoordinator(maze).coordinate(agents, proposals)

    assert result == {
        "R09": (5, 11),
        "R03": (10, 10),
    }


def test_waiting_robot_remains_a_stationary_physical_obstacle():
    from security.physical.geometry import robots_are_safe_during_motion

    waiting_robot_start = (7, 11)
    waiting_robot_end = (7, 11)

    moving_robot_start = (6, 11)
    moving_robot_end = (7, 11)

    assert not robots_are_safe_during_motion(
        waiting_robot_start,
        waiting_robot_end,
        moving_robot_start,
        moving_robot_end,
    )


def test_coordinate_with_yields_reports_voluntary_waiting_agent():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()

    agents = [
        SimpleNamespace(agent_id="R05", position=(7, 11), goal=(5, 7)),
        SimpleNamespace(agent_id="R09", position=(6, 11), goal=(7, 1)),
    ]
    proposals = {
        "R05": (6, 11),
        "R09": (7, 11),
    }

    result = TrafficCoordinator(maze).coordinate_with_yields(
        agents, proposals
    )

    assert result.proposals == {"R09": (5, 11)}
    assert result.yielded_agents == frozenset({"R05"})


def test_coordinate_with_yields_reports_no_yield_without_conflict():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(3, 1)),
        SimpleNamespace(agent_id="R2", position=(1, 3), goal=(3, 3)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (2, 3),
    }

    result = TrafficCoordinator(maze).coordinate_with_yields(
        agents, proposals
    )

    assert result.proposals == proposals
    assert result.yielded_agents == frozenset()


def test_coordinate_with_yields_preserves_proposals_without_safe_solution():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    class DisconnectedMaze:
        def get_neighbors(self, x, y):
            return []

    agents = [
        SimpleNamespace(agent_id="R1", position=(0, 0), goal=(1, 0)),
        SimpleNamespace(agent_id="R2", position=(1, 0), goal=(0, 0)),
    ]
    proposals = {
        "R1": (1, 0),
        "R2": (0, 0),
    }

    result = TrafficCoordinator(DisconnectedMaze()).coordinate_with_yields(
        agents, proposals
    )

    assert result.proposals == proposals
    assert result.yielded_agents == frozenset()


def test_detect_destination_conflicts_groups_multiple_agents():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(3, 1)),
        SimpleNamespace(agent_id="R2", position=(3, 1), goal=(1, 1)),
        SimpleNamespace(agent_id="R3", position=(2, 2), goal=(2, 4)),
        SimpleNamespace(agent_id="R4", position=(4, 1), goal=(4, 3)),
        SimpleNamespace(agent_id="R5", position=(4, 3), goal=(4, 1)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (2, 1),
        "R3": (2, 1),
        "R4": (4, 2),
        "R5": (4, 2),
    }
    original = dict(proposals)

    conflicts = coordinator.detect_destination_conflicts(
        agents, proposals
    )

    assert conflicts == (
        ((2, 1), ("R1", "R2", "R3")),
        ((4, 2), ("R4", "R5")),
    )
    assert proposals == original


def test_detect_destination_conflicts_rejects_unknown_agent_ids():
    from types import SimpleNamespace

    import pytest

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(3, 1)),
    ]
    proposals = {
        "R1": (2, 1),
        "UNKNOWN": (2, 1),
    }

    with pytest.raises(ValueError, match="unknown agents"):
        coordinator.detect_destination_conflicts(agents, proposals)


def test_detect_destination_conflicts_is_order_independent():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(3, 1)),
        SimpleNamespace(agent_id="R2", position=(3, 1), goal=(1, 1)),
        SimpleNamespace(agent_id="R3", position=(2, 2), goal=(2, 4)),
        SimpleNamespace(agent_id="R4", position=(4, 1), goal=(4, 3)),
    ]

    first = {
        "R1": (2, 1),
        "R2": (2, 1),
        "R3": (4, 2),
    }
    second = {
        "R3": (4, 2),
        "R2": (2, 1),
        "R1": (2, 1),
    }

    expected = (((2, 1), ("R1", "R2")),)

    assert coordinator.detect_destination_conflicts(agents, first) == expected
    assert coordinator.detect_destination_conflicts(agents, second) == expected


def test_detect_trajectory_conflicts_finds_crossing_paths():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(2, 1), goal=(1, 1)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (1, 1),
    }
    original = dict(proposals)

    conflicts = coordinator.detect_trajectory_conflicts(
        agents, proposals
    )

    assert conflicts == (("R1", "R2"),)
    assert proposals == original


def test_detect_trajectory_conflicts_includes_stationary_agents():
    from types import SimpleNamespace

    from simulation.multi_agent_manager import MultiAgentManager
    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(2, 1), goal=(2, 2)),
    ]
    proposals = {"R1": (2, 1)}

    conflicts = coordinator.detect_trajectory_conflicts(
        agents, proposals
    )

    assert conflicts == (("R1", "R2"),)
    assert proposals == {"R1": (2, 1)}


def test_detect_trajectory_conflicts_returns_empty_when_safe():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(1, 3), goal=(2, 3)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (2, 3),
    }

    conflicts = coordinator.detect_trajectory_conflicts(
        agents, proposals
    )

    assert conflicts == ()


def test_detect_trajectory_conflicts_rejects_unknown_agent_ids():
    import pytest
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
    ]
    proposals = {
        "UNKNOWN": (2, 1),
    }

    with pytest.raises(ValueError, match="unknown agents"):
        coordinator.detect_trajectory_conflicts(agents, proposals)


def test_detect_swaps_finds_reciprocal_movements():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(2, 1), goal=(1, 1)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (1, 1),
    }
    original = dict(proposals)

    swaps = coordinator.detect_swaps(agents, proposals)

    assert swaps == (("R1", "R2"),)
    assert proposals == original


def test_detect_swaps_ignores_non_reciprocal_movements():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(2, 1), goal=(2, 2)),
        SimpleNamespace(agent_id="R3", position=(3, 1), goal=(3, 2)),
    ]
    proposals = {
        "R1": (2, 1),
        "R3": (3, 2),
    }

    swaps = coordinator.detect_swaps(agents, proposals)

    assert swaps == ()


def test_detect_swaps_rejects_unknown_agent_ids():
    import pytest
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
    ]
    proposals = {"UNKNOWN": (2, 1)}

    with pytest.raises(ValueError, match="unknown agents"):
        coordinator.detect_swaps(agents, proposals)


def test_detect_dependencies_finds_occupied_destinations():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(2, 1), goal=(3, 1)),
        SimpleNamespace(agent_id="R3", position=(4, 1), goal=(5, 1)),
    ]
    proposals = {
        "R1": (2, 1),
        "R3": (5, 1),
    }

    dependencies = coordinator.detect_dependencies(agents, proposals)

    assert dependencies == (("R1", "R2"),)


def test_detect_dependencies_ignores_unoccupied_destinations():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(3, 1), goal=(4, 1)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (4, 1),
    }

    dependencies = coordinator.detect_dependencies(agents, proposals)

    assert dependencies == ()


def test_detect_dependencies_rejects_unknown_agent_ids():
    import pytest
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
    ]
    proposals = {"UNKNOWN": (2, 1)}

    with pytest.raises(ValueError, match="unknown agents"):
        coordinator.detect_dependencies(agents, proposals)


def test_detect_cycles_finds_three_agent_dependency_cycle():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(2, 1), goal=(2, 2)),
        SimpleNamespace(agent_id="R3", position=(2, 2), goal=(1, 1)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (2, 2),
        "R3": (1, 1),
    }

    cycles = coordinator.detect_cycles(agents, proposals)

    assert cycles == (("R1", "R2", "R3"),)


def test_detect_cycles_ignores_acyclic_dependencies():
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
        SimpleNamespace(agent_id="R2", position=(2, 1), goal=(3, 1)),
        SimpleNamespace(agent_id="R3", position=(3, 1), goal=(4, 1)),
        SimpleNamespace(agent_id="R4", position=(4, 1), goal=(5, 1)),
    ]
    proposals = {
        "R1": (2, 1),
        "R2": (3, 1),
        "R3": (4, 1),
    }

    cycles = coordinator.detect_cycles(agents, proposals)

    assert cycles == ()


def test_detect_cycles_rejects_unknown_agent_ids():
    import pytest
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()
    coordinator = TrafficCoordinator(maze)

    agents = [
        SimpleNamespace(agent_id="R1", position=(1, 1), goal=(2, 1)),
    ]
    proposals = {"UNKNOWN": (2, 1)}

    with pytest.raises(ValueError, match="unknown agents"):
        coordinator.detect_cycles(agents, proposals)


def test_traffic_coordinator_resolves_three_agent_destination_conflict():
    from itertools import combinations
    from types import SimpleNamespace

    from simulation.traffic_coordinator import TrafficCoordinator

    maze = Maze(width=25, height=25, seed=42)
    maze.generate_static_maze()

    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    scenario = None

    for x in range(2, 23):
        for y in range(2, 23):
            center = (x, y)
            if not maze.is_walkable(*center):
                continue

            for selected in combinations(directions, 3):
                starts = [
                    (x + dx, y + dy) for dx, dy in selected
                ]
                goals = [
                    (x + 2 * dx, y + 2 * dy)
                    for dx, dy in selected
                ]

                if all(
                    maze.is_walkable(*position)
                    for position in starts + goals
                ):
                    scenario = (center, starts, goals)
                    break

            if scenario is not None:
                break
        if scenario is not None:
            break

    assert scenario is not None, "No suitable three-robot test geometry found."

    center, starts, goals = scenario
    agents = [
        SimpleNamespace(
            agent_id=f"R{i + 1}",
            position=start,
            goal=goal,
            navigation_target=goal,
        )
        for i, (start, goal) in enumerate(zip(starts, goals))
    ]
    proposals = {
        agent.agent_id: center for agent in agents
    }

    coordinator = TrafficCoordinator(maze)
    result = coordinator.coordinate(agents, proposals)

    assert result == {
        agent.agent_id: goal
        for agent, goal in zip(agents, goals)
    }
    assert len(set(result.values())) == 3
    assert coordinator.detect_trajectory_conflicts(
        agents, result
    ) == ()
