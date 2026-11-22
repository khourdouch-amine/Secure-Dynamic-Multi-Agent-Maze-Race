"""Unit tests for permanent goals and temporary recovery navigation."""

import pytest

from agents.base_agent import BaseAgent


class OpenMaze:
    """Minimal walkable maze for BaseAgent tests."""

    def is_walkable(self, x, y):
        return 0 <= x < 20 and 0 <= y < 20


class RecordingPlanner:
    """Planner stub recording every planning request."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def plan(self, maze, start, goal):
        self.calls.append((maze, start, goal))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def maze():
    return OpenMaze()


@pytest.fixture
def agent():
    planner = RecordingPlanner([(1, 1), (2, 1), (3, 1)])
    return BaseAgent(
        agent_id="R1",
        start=(1, 1),
        goal=(3, 1),
        planner=planner,
        family="classical",
    )


def test_initial_navigation_target_is_permanent_goal(agent):
    assert agent.goal == (3, 1)
    assert agent.navigation_target == agent.goal
    assert agent.recovery_state == "idle"
    assert agent.reached_navigation_target is False
    assert agent.reached_goal is False


def test_begin_recovery_preserves_permanent_goal(agent):
    agent.compute_plan(OpenMaze())
    agent.begin_recovery((1, 3))

    assert agent.goal == (3, 1)
    assert agent.navigation_target == (1, 3)
    assert agent.recovery_state == "active"
    assert agent.path == []
    assert agent.status == "needs_replanning"


@pytest.mark.parametrize(
    "target",
    [
        (1, 1),       # Current position
        (3, 1),       # Permanent goal
        (2.5, 3),     # Float coordinate
        (True, 3),    # Boolean coordinate
        (2,),         # Incorrect dimension
        ("2", 3),     # Non-integer coordinate
    ],
)
def test_begin_recovery_rejects_invalid_targets(agent, target):
    with pytest.raises(ValueError):
        agent.begin_recovery(target)


def test_compute_plan_uses_temporary_navigation_target(agent, maze):
    agent.begin_recovery((1, 3))
    recovery_planner = RecordingPlanner(
        [(1, 1), (1, 2), (1, 3)]
    )

    path = agent.compute_plan(maze, planner=recovery_planner)

    assert path == [(1, 1), (1, 2), (1, 3)]
    assert recovery_planner.calls == [(maze, (1, 1), (1, 3))]
    assert agent.planner is not recovery_planner
    assert agent.goal == (3, 1)
    assert agent.status == "planned"


def test_compute_plan_uses_original_planner_by_default(agent, maze):
    agent.compute_plan(maze)

    assert agent.planner.calls == [(maze, (1, 1), (3, 1))]


def test_compute_plan_rejects_path_with_wrong_start(agent, maze):
    agent.planner.result = [(2, 1), (3, 1)]

    with pytest.raises(ValueError, match="start"):
        agent.compute_plan(maze)


def test_compute_plan_returns_no_path_when_planner_returns_none(agent, maze):
    agent.planner.result = None

    assert agent.compute_plan(maze) == []
    assert agent.path == []
    assert agent.status == "no_path"


def test_compute_plan_returns_no_path_for_wrong_destination(agent, maze):
    agent.planner.result = [(1, 1), (2, 1)]

    assert agent.compute_plan(maze) == []
    assert agent.path == []
    assert agent.status == "no_path"


def test_compute_plan_marks_planning_error(agent, maze):
    agent.planner.error = RuntimeError("planner failure")

    with pytest.raises(RuntimeError, match="planner failure"):
        agent.compute_plan(maze)

    assert agent.status == "planning_error"


def test_recovery_waypoint_is_reached_without_finishing_mission(agent, maze):
    agent.begin_recovery((1, 3))
    agent.planner.result = [(1, 1), (1, 2), (1, 3)]
    agent.compute_plan(maze)

    assert agent.move_one_step(maze) is True
    assert agent.position == (1, 2)
    assert agent.status == "moving"

    assert agent.move_one_step(maze) is True
    assert agent.position == (1, 3)
    assert agent.status == "recovery_waypoint_reached"

    assert agent.reached_navigation_target is True
    assert agent.reached_goal is False
    assert agent.goal == (3, 1)


def test_complete_recovery_restores_permanent_goal(agent):
    agent.begin_recovery((1, 3))
    agent.position = (1, 3)

    agent.complete_recovery()

    assert agent.navigation_target == agent.goal
    assert agent.recovery_state == "idle"
    assert agent.path == []
    assert agent.status == "needs_replanning"


def test_complete_recovery_requires_reaching_waypoint(agent):
    agent.begin_recovery((1, 3))

    with pytest.raises(RuntimeError, match="not been reached"):
        agent.complete_recovery()


def test_complete_recovery_requires_active_recovery(agent):
    with pytest.raises(RuntimeError, match="No active recovery"):
        agent.complete_recovery()


def test_fail_recovery_restores_goal_without_changing_it(agent):
    agent.begin_recovery((1, 3))
    agent.fail_recovery()

    assert agent.goal == (3, 1)
    assert agent.navigation_target == (3, 1)
    assert agent.recovery_state == "failed"
    assert agent.path == []
    assert agent.status == "needs_replanning"


def test_fail_recovery_requires_active_recovery(agent):
    with pytest.raises(RuntimeError, match="No active recovery"):
        agent.fail_recovery()


def test_normal_navigation_still_reaches_permanent_goal(agent, maze):
    agent.compute_plan(maze)

    assert agent.move_one_step(maze) is True
    assert agent.move_one_step(maze) is True

    assert agent.position == agent.goal
    assert agent.reached_goal is True
    assert agent.status == "reached_goal"


def test_mark_recovery_failed_without_active_recovery(agent):
    agent.mark_recovery_failed()

    assert agent.goal == (3, 1)
    assert agent.navigation_target == agent.goal
    assert agent.recovery_state == "failed"
    assert agent.path == []
    assert agent.status == "needs_replanning"


def test_mark_recovery_failed_rejects_active_recovery(agent):
    agent.begin_recovery((1, 3))

    with pytest.raises(RuntimeError, match="Use fail_recovery"):
        agent.mark_recovery_failed()


def test_reset_recovery_restores_idle_state(agent):
    agent.mark_recovery_failed()
    agent.reset_recovery()

    assert agent.goal == (3, 1)
    assert agent.navigation_target == agent.goal
    assert agent.recovery_state == "idle"
    assert agent.path == []
    assert agent.status == "needs_replanning"


def test_reset_recovery_requires_failed_state(agent):
    with pytest.raises(RuntimeError, match="not in a failed state"):
        agent.reset_recovery()
