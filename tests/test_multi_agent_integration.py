"""Integration tests for classical and quantum agents."""

import pytest

from agents.classical_agent import ClassicalAgent
from agents.quantum_agent import QuantumAgent
from environment.dynamic_elements import DynamicEnvironment
from environment.maze import Maze
from simulation.multi_agent_manager import MultiAgentManager


class DeterministicPlanner:
    """Lightweight planner used for architecture integration tests."""

    def plan(self, maze, start, goal):
        x, y = start
        gx, gy = goal

        path = [(x, y)]

        while x != gx:
            x += 1 if gx > x else -1
            path.append((x, y))

        while y != gy:
            y += 1 if gy > y else -1
            path.append((x, y))

        if all(maze.is_walkable(*cell) for cell in path):
            return path

        return None


def test_classical_and_quantum_agents_share_dynamic_environment():
    maze = Maze(width=25, height=25, seed=42)

    classical = ClassicalAgent(
        agent_id="C1",
        start=(2, 2),
        goal=(5, 2),
        planner=DeterministicPlanner(),
        algorithm_name="Deterministic Classical Test",
    )

    quantum = QuantumAgent(
        agent_id="Q1",
        start=(2, 5),
        goal=(5, 5),
        planner=DeterministicPlanner(),
        algorithm_name="Deterministic Quantum Test",
    )

    agents = [classical, quantum]

    dynamic_environment = DynamicEnvironment.from_agents(
        maze,
        agents,
        seed=42,
    )

    assert classical.family == "classical"
    assert quantum.family == "quantum"

    for agent in agents:
        assert agent.start in dynamic_environment.protected_cells
        assert agent.position in dynamic_environment.protected_cells
        assert agent.goal in dynamic_environment.protected_cells
        assert (agent.start, agent.goal) in (
            dynamic_environment.connectivity_pairs
        )

    manager = MultiAgentManager(
        maze,
        agents,
        dynamic_environment=dynamic_environment,
    )

    for agent in agents:
        agent.compute_plan(maze)

    result = manager.step()

    assert result["moved"] == 2
    assert classical.position == (3, 2)
    assert quantum.position == (3, 5)

    assert result["positions"] == {
        "C1": (3, 2),
        "Q1": (3, 5),
    }

    assert len(dynamic_environment.wall_positions) == 15
    assert len(dynamic_environment.obstacle_positions) == 12


def test_manager_triggers_safe_recovery_after_persistent_blockage():
    from environment.moving_obstacle import MovingObstacle
    from simulation.recovery_planner import RecoveryPlan

    class OneShotObstacleEnvironment:
        def __init__(self, obstacle):
            self.obstacle = obstacle
            self.calls = 0

        def update(self, simulation_time, occupied_cells=()):
            self.calls += 1
            trajectories = (self.obstacle,) if self.calls == 1 else ()

            return {
                "walls_updated": False,
                "obstacles_updated": False,
                "changed_cells": [],
                "obstacle_trajectories": trajectories,
            }

    class FakeRecoveryPlanner:
        def __init__(self):
            self.calls = []

        def plan(
            self,
            agent,
            maze,
            dynamic_environment,
            agents,
            simulation_time,
            obstacle_trajectories=(),
            excluded_waypoints=(),
        ):
            self.calls.append(agent.agent_id)

            if agent.agent_id != "C1":
                return None

            waypoint = (3, 3)
            temporary_planner = DeterministicPlanner()

            return RecoveryPlan(
                waypoint=waypoint,
                path=((2, 2), (2, 3), waypoint),
                planner=temporary_planner,
                arrival_time=simulation_time + 1.0,
                planner_name="DeterministicPlanner",
                family="classical",
                candidates_examined=1,
            )

    maze = Maze(width=9, height=9, seed=42)

    agent = ClassicalAgent(
        agent_id="C1",
        start=(2, 2),
        goal=(5, 2),
        planner=DeterministicPlanner(),
        algorithm_name="Deterministic Classical Test",
    )
    agent.compute_plan(maze)

    original_planner = agent.planner

    obstacle = MovingObstacle(
        start_position=(2.5, 1.0),
        target_position=(2.5, 3.0),
        start_time=0.0,
        duration=1.0,
    )

    environment = OneShotObstacleEnvironment(obstacle)
    recovery_planner = FakeRecoveryPlanner()

    manager = MultiAgentManager(
        maze,
        [agent],
        dynamic_environment=environment,
        time_step=1.0,
        blocked_threshold=1,
        recovery_planner=recovery_planner,
    )

    first_result = manager.step()

    assert first_result["moved"] == 0
    assert first_result["safety_blocked"] == 1
    assert agent.position == (2, 2)
    assert manager.recovery_coordinator.is_blocked("C1")

    second_result = manager.step()

    assert recovery_planner.calls == ["C1"]
    assert second_result["moved"] == 1
    assert agent.position == (2, 3)

    started_events = second_result["recovery_events"]
    assert len(started_events) == 1
    assert started_events[0]["event"] == "started"
    assert started_events[0]["agent_id"] == "C1"
    assert started_events[0]["attempt"] == 1
    assert started_events[0]["start"] == (2, 2)
    assert started_events[0]["waypoint"] == (3, 3)
    assert started_events[0]["family"] == "classical"

    assert agent.goal == (5, 2)
    assert agent.navigation_target == (3, 3)
    assert agent.recovery_state == "active"
    assert agent.planner is original_planner

    temporary_planner = manager._recovery_planners["C1"]
    assert temporary_planner is not original_planner

    # Reach the temporary waypoint through the real movement pipeline.
    third_result = manager.step()

    assert third_result["moved"] == 1
    assert agent.position == (3, 3)

    completed_events = third_result["recovery_events"]
    assert len(completed_events) == 1
    assert completed_events[0]["event"] == "completed"
    assert completed_events[0]["agent_id"] == "C1"
    assert completed_events[0]["position"] == (3, 3)
    assert completed_events[0]["waypoint"] == (3, 3)

    assert agent.navigation_target == agent.goal
    assert agent.recovery_state == "idle"
    assert agent.status == "needs_replanning"
    assert "C1" not in manager._recovery_planners
    assert agent.planner is original_planner

    # The next cycle must resume navigation with the permanent planner.
    fourth_result = manager.step()

    assert fourth_result["moved"] == 1
    assert agent.position == (4, 3)
    assert agent.goal == (5, 2)
    assert agent.navigation_target == agent.goal
    assert agent.recovery_state == "idle"
    assert agent.planner is original_planner

    # Continue the mission until the permanent goal is reached.
    for _ in range(10):
        if agent.reached_goal:
            break

        mission_result = manager.step()
        assert mission_result["moved"] == 1

    assert agent.position == agent.goal
    assert agent.reached_goal
    assert agent.status == "reached_goal"
    assert agent.recovery_state == "idle"
    assert agent.navigation_target == agent.goal
    assert agent.planner is original_planner
    assert "C1" not in manager._recovery_planners


def test_recovery_planning_retries_after_simulation_time_delay():
    maze = Maze(width=9, height=9, seed=42)

    agent = ClassicalAgent(
        agent_id="C1",
        start=(2, 2),
        goal=(5, 2),
        planner=DeterministicPlanner(),
        algorithm_name="Deterministic Classical Test",
    )

    class NoPlanRecoveryPlanner:
        def __init__(self):
            self.calls = 0

        def plan(self, **kwargs):
            self.calls += 1
            return None

    recovery_planner = NoPlanRecoveryPlanner()

    manager = MultiAgentManager(
        maze,
        [agent],
        time_step=0.1,
        blocked_threshold=1,
        recovery_planner=recovery_planner,
    )

    manager.simulation_time = 1.0
    manager.recovery_coordinator.observe(
        agent.agent_id,
        agent.position,
        manager.simulation_time,
    )

    assert manager.recovery_coordinator.is_blocked(agent.agent_id)

    assert manager._try_start_recovery(agent) is False
    assert recovery_planner.calls == 1

    # No second attempt at the same simulation time.
    assert manager._try_start_recovery(agent) is False
    assert recovery_planner.calls == 1

    # Still inside the one-second retry delay.
    manager.simulation_time = 1.9

    assert manager._try_start_recovery(agent) is False
    assert recovery_planner.calls == 1

    # A new attempt is allowed at the retry deadline.
    manager.simulation_time = 2.0

    assert manager._try_start_recovery(agent) is False
    assert recovery_planner.calls == 2



@pytest.mark.parametrize(
    "invalid_interval",
    [
        0,
        -1,
        float("inf"),
        float("-inf"),
        float("nan"),
        None,
        True,
        False,
        "1.0",
    ],
)
def test_manager_rejects_invalid_recovery_retry_interval(invalid_interval):
    maze = Maze(width=9, height=9, seed=42)

    with pytest.raises(ValueError, match="recovery_retry_interval"):
        MultiAgentManager(
            maze,
            [],
            recovery_retry_interval=invalid_interval,
        )


def test_recovery_stops_after_maximum_failed_attempts_and_resumes_navigation():
    maze = Maze(width=9, height=9, seed=42)

    agent = ClassicalAgent(
        agent_id="C1",
        start=(2, 2),
        goal=(5, 2),
        planner=DeterministicPlanner(),
        algorithm_name="Deterministic Classical Test",
    )
    original_planner = agent.planner

    class NoPlanRecoveryPlanner:
        def __init__(self):
            self.calls = 0

        def plan(self, **kwargs):
            self.calls += 1
            return None

    recovery_planner = NoPlanRecoveryPlanner()

    manager = MultiAgentManager(
        maze,
        [agent],
        time_step=1.0,
        blocked_threshold=1,
        recovery_planner=recovery_planner,
        recovery_retry_interval=1.0,
        max_recovery_attempts=3,
    )

    manager.simulation_time = 1.0
    manager.recovery_coordinator.observe(
        agent.agent_id,
        agent.position,
        manager.simulation_time,
    )

    assert manager._try_start_recovery(agent) is False
    assert recovery_planner.calls == 1

    manager.simulation_time = 2.0
    assert manager._try_start_recovery(agent) is False
    assert recovery_planner.calls == 2

    manager.simulation_time = 3.0
    assert manager._try_start_recovery(agent) is False
    assert recovery_planner.calls == 3

    recovery_events = manager._step_recovery_events
    assert [event["event"] for event in recovery_events] == [
        "attempt_failed",
        "attempt_failed",
        "exhausted",
    ]
    assert [event["attempt"] for event in recovery_events] == [1, 2, 3]
    assert all(event["reason"] == "no_valid_plan" for event in recovery_events)
    assert [event.get("retry_after") for event in recovery_events] == [
        2.0,
        3.0,
        None,
    ]
    assert all(event["agent_id"] == "C1" for event in recovery_events)

    assert agent.recovery_state == "failed"
    assert agent.navigation_target == agent.goal
    assert agent.goal == (5, 2)
    assert agent.planner is original_planner

    # Exhaustion prevents another recovery-planner invocation.
    manager.simulation_time = 4.0
    assert manager._try_start_recovery(agent) is False
    assert recovery_planner.calls == 3

    # The permanent planner still allows a physically accepted move.
    result = manager.step(simulation_time=4.1)

    assert result["moved"] == 1
    assert agent.position == (3, 2)
    assert agent.recovery_state == "idle"
    assert agent.navigation_target == agent.goal
    assert agent.planner is original_planner


@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, 1.5, None, True, False, "3"],
)
def test_manager_rejects_invalid_max_recovery_attempts(invalid_limit):
    maze = Maze(width=9, height=9, seed=42)

    with pytest.raises(ValueError, match="max_recovery_attempts"):
        MultiAgentManager(
            maze,
            [],
            max_recovery_attempts=invalid_limit,
        )


def test_recovery_memory_persists_until_permanent_goal_progress():
    maze = Maze(width=25, height=25, seed=42)

    agent = ClassicalAgent(
        agent_id="R1",
        start=(7, 11),
        goal=(5, 7),
        planner=DeterministicPlanner(),
        algorithm_name="Deterministic Classical Test",
    )

    manager = MultiAgentManager(maze, [agent])

    manager._refresh_recovery_memory(agent)
    manager._remember_recovery_waypoint(agent, (7, 12))

    # Returning to the original recovery position is not goal progress.
    agent.position = (7, 11)
    manager._refresh_recovery_memory(agent)

    assert manager._excluded_recovery_waypoints(agent) == {(7, 12)}

    # A genuine reduction in topological distance clears the old exclusions.
    agent.position = (6, 11)
    manager._refresh_recovery_memory(agent)

    assert manager._excluded_recovery_waypoints(agent) == set()


def test_temporary_recovery_progress_does_not_clear_waypoint_memory():
    maze = Maze(width=25, height=25, seed=42)

    agent = ClassicalAgent(
        agent_id="R1",
        start=(7, 11),
        goal=(5, 7),
        planner=DeterministicPlanner(),
        algorithm_name="Deterministic Classical Test",
    )

    manager = MultiAgentManager(maze, [agent])
    manager._refresh_recovery_memory(agent)

    # The temporary waypoint is closer to the permanent goal.
    agent.position = (6, 11)
    manager._remember_recovery_waypoint(agent, (6, 11))

    assert manager._excluded_recovery_waypoints(agent) == {(6, 11)}

    # Returning to the original position must not erase the exclusion.
    agent.position = (7, 11)
    manager._refresh_recovery_memory(agent)

    assert manager._excluded_recovery_waypoints(agent) == {(6, 11)}

    # Only subsequent genuine navigation progress clears the exclusion.
    agent.position = (6, 11)
    manager._refresh_recovery_memory(agent)

    assert manager._excluded_recovery_waypoints(agent) == set()


def test_temporary_recovery_progress_does_not_clear_waypoint_memory():
    maze = Maze(width=25, height=25, seed=42)

    agent = ClassicalAgent(
        agent_id="R1",
        start=(7, 11),
        goal=(5, 7),
        planner=DeterministicPlanner(),
        algorithm_name="Deterministic Classical Test",
    )

    manager = MultiAgentManager(maze, [agent])
    manager._refresh_recovery_memory(agent)

    # The temporary waypoint is closer to the permanent goal.
    agent.position = (6, 11)
    manager._remember_recovery_waypoint(agent, (6, 11))

    assert manager._excluded_recovery_waypoints(agent) == {(6, 11)}

    # Returning to the original position must not erase the exclusion.
    agent.position = (7, 11)
    manager._refresh_recovery_memory(agent)

    assert manager._excluded_recovery_waypoints(agent) == {(6, 11)}

    # Only subsequent genuine navigation progress clears the exclusion.
    agent.position = (6, 11)
    manager._refresh_recovery_memory(agent)

    assert manager._excluded_recovery_waypoints(agent) == set()
