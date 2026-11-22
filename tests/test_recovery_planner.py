"""Unit tests for safety-aware recovery waypoint planning."""

from agents.base_agent import BaseAgent
from environment.maze import Maze
from planners.classical.astar import AStarPlanner
from simulation.recovery_planner import RecoveryPlanner


def test_recovery_plan_preserves_agent_goal_and_original_planner():
    maze = Maze(width=7, height=7, seed=42)
    original_planner = AStarPlanner()

    agent = BaseAgent(
        agent_id="R1",
        start=(1, 1),
        goal=(5, 5),
        planner=original_planner,
        family="classical",
    )

    recovery = RecoveryPlanner(max_candidates=8)

    result = recovery.plan(
        agent=agent,
        maze=maze,
        dynamic_environment=None,
        agents=[agent],
        simulation_time=0.0,
    )

    assert result is not None
    assert result.waypoint != agent.position
    assert result.waypoint != agent.goal
    assert result.path[0] == agent.position
    assert result.path[-1] == result.waypoint

    assert result.planner is not original_planner
    assert isinstance(result.planner, AStarPlanner)

    assert agent.goal == (5, 5)
    assert agent.navigation_target == (5, 5)
    assert agent.planner is original_planner
    assert agent.recovery_state == "idle"
    assert agent.position == (1, 1)


def test_recovery_returns_none_when_agent_is_already_at_goal():
    maze = Maze(width=7, height=7, seed=42)

    agent = BaseAgent(
        agent_id="R1",
        start=(1, 1),
        goal=(1, 1),
        planner=AStarPlanner(),
        family="classical",
    )

    recovery = RecoveryPlanner()

    result = recovery.plan(
        agent=agent,
        maze=maze,
        dynamic_environment=None,
        agents=[agent],
        simulation_time=0.0,
    )

    assert result is None
    assert agent.position == agent.goal
    assert agent.navigation_target == agent.goal
    assert agent.recovery_state == "idle"


def test_rrt_star_clone_preserves_configuration_and_rng_independence():
    from planners.classical.rrt_star_grid import RRTStarGridPlanner

    original = RRTStarGridPlanner(
        max_iterations=1200,
        step_size=1.5,
        goal_radius=1.25,
        neighbor_radius=3.5,
        seed=17,
        adapter_margin=3,
    )

    clone = RecoveryPlanner._clone_planner(
        original,
        Maze(width=7, height=7, seed=42),
        (1, 1),
        (4, 4),
    )

    assert clone is not original
    assert clone.rrt_star.max_iterations == 1200
    assert clone.rrt_star.step_size == 1.5
    assert clone.rrt_star.goal_radius == 1.25
    assert clone.rrt_star.neighbor_radius == 3.5
    assert clone.adapter.margin == 3

    assert clone.rrt_star.rng is not original.rrt_star.rng
    assert (
        clone.rrt_star.rng.bit_generator.state
        == original.rrt_star.rng.bit_generator.state
    )

    clone.rrt_star.rng.random()

    assert (
        clone.rrt_star.rng.bit_generator.state
        != original.rrt_star.rng.bit_generator.state
    )


def test_unknown_planner_is_rejected_without_classical_fallback():
    class UnknownPlanner:
        def plan(self, maze, start, goal):
            return [start, goal]

    maze = Maze(width=7, height=7, seed=42)
    agent = BaseAgent(
        agent_id="R1",
        start=(1, 1),
        goal=(5, 5),
        planner=UnknownPlanner(),
        family="quantum",
    )

    recovery = RecoveryPlanner(max_candidates=1)

    import pytest

    with pytest.raises(NotImplementedError, match="no fallback"):
        recovery.plan(
            agent=agent,
            maze=maze,
            dynamic_environment=None,
            agents=[agent],
            simulation_time=0.0,
        )

    assert agent.planner.__class__ is UnknownPlanner
    assert agent.goal == (5, 5)
    assert agent.navigation_target == (5, 5)
    assert agent.position == (1, 1)


def test_path_is_rejected_when_continuous_clearance_to_wall_is_insufficient():
    recovery = RecoveryPlanner(
        robot_radius=0.25,
        obstacle_margin=0.10,
    )

    safe = recovery._path_is_safe(
        path=[(1, 1), (2, 1)],
        other_positions=set(),
        trajectories=(),
        blocked_squares={(1, 3)},
        simulation_time=0.0,
    )

    unsafe = recovery._path_is_safe(
        path=[(1, 1), (2, 1)],
        other_positions=set(),
        trajectories=(),
        blocked_squares={(2, 1)},
        simulation_time=0.0,
    )

    assert safe is True
    assert unsafe is False


def test_path_rejects_collision_with_obstacle_crossing_robot_trajectory():
    from environment.moving_obstacle import MovingObstacle

    recovery = RecoveryPlanner(time_step=1.0 / 30.0)

    crossing_obstacle = MovingObstacle(
        start_position=(1.5, 0.0),
        target_position=(1.5, 2.0),
        start_time=0.0,
        duration=1.0 / 30.0,
        size=1.0,
    )

    safe = recovery._path_is_safe(
        path=[(1, 1), (2, 1)],
        other_positions=set(),
        trajectories=(crossing_obstacle,),
        blocked_squares=set(),
        simulation_time=0.0,
    )

    assert safe is False


def test_path_rejects_waypoint_occupied_by_obstacle_during_wait():
    from environment.moving_obstacle import MovingObstacle

    recovery = RecoveryPlanner(time_step=1.0 / 30.0)

    crossing_obstacle = MovingObstacle(
        start_position=(0.0, 2.0),
        target_position=(4.0, 2.0),
        start_time=0.0,
        duration=1.0 / 30.0,
        size=1.0,
    )

    safe = recovery._path_is_safe(
        path=[(2, 2)],
        other_positions=set(),
        trajectories=(crossing_obstacle,),
        blocked_squares=set(),
        simulation_time=0.0,
    )

    assert safe is False


def test_candidate_respects_exact_robot_robot_distance_boundary():
    recovery = RecoveryPlanner(robot_robot_distance=1.0)

    # Exactly 1 meter between centers: safe.
    safe = recovery._candidate_is_safe(
        candidate=(1.0, 1.0),
        other_positions={(2.0, 1.0)},
        blocked_squares=set(),
    )

    # Less than 1 meter between centers: unsafe.
    unsafe = recovery._candidate_is_safe(
        candidate=(1.1, 1.0),
        other_positions={(2.0, 1.0)},
        blocked_squares=set(),
    )

    assert safe is True
    assert unsafe is False


def test_path_rejects_robot_conflict_between_segment_endpoints():
    recovery = RecoveryPlanner(robot_robot_distance=1.0)

    # Both endpoints are more than 1 m from the stationary robot,
    # but the continuous segment passes within 0.9 m of it.
    safe = recovery._path_is_safe(
        path=[(1, 1), (2, 1)],
        other_positions={(1.5, 1.9)},
        trajectories=(),
        blocked_squares=set(),
        simulation_time=0.0,
    )

    assert safe is False


def test_bfs_distances_follow_maze_topology():
    maze = Maze(width=7, height=7, seed=42)

    # Create a wall that forces a detour around its lower endpoint.
    for y in range(1, 5):
        maze.set_cell(3, y, 1)

    distances = RecoveryPlanner._distances_to_goal(
        maze,
        goal=(5, 1),
    )

    assert distances[(5, 1)] == 0
    assert distances[(4, 1)] == 1
    assert distances[(2, 1)] == 11
    assert (3, 1) not in distances


def test_recovery_planner_excludes_previously_unsuccessful_waypoints():
    maze = Maze(width=9, height=9, seed=42)

    agent = BaseAgent(
        agent_id="R1",
        start=(2, 2),
        goal=(7, 7),
        planner=AStarPlanner(),
        family="classical",
    )

    recovery = RecoveryPlanner(max_candidates=32)

    first = recovery.plan(
        agent=agent,
        maze=maze,
        dynamic_environment=None,
        agents=[agent],
        simulation_time=0.0,
    )

    assert first is not None

    second = recovery.plan(
        agent=agent,
        maze=maze,
        dynamic_environment=None,
        agents=[agent],
        simulation_time=1.0,
        excluded_waypoints={first.waypoint},
    )

    assert second is not None
    assert second.waypoint != first.waypoint


def test_qaoa_clone_preserves_configuration_and_is_independent():
    from planners.quantum.qaoa_navigation import QAOANavigationPlanner

    original = QAOANavigationPlanner(
        candidates=5,
        layers=3,
        shots=64,
        max_iterations=45,
        penalty=3.5,
        seed=17,
    )

    clone = RecoveryPlanner._clone_planner(
        original,
        Maze(width=7, height=7, seed=42),
        (1, 1),
        (4, 4),
    )

    assert isinstance(clone, QAOANavigationPlanner)
    assert clone is not original
    assert clone.max_candidates == 5
    assert clone.layers == 3
    assert clone.shots == 64
    assert clone.max_iterations == 45
    assert clone.penalty == 3.5
    assert clone.seed == 17

    clone.max_candidates = 2
    assert original.max_candidates == 5


def test_vqc_clone_preserves_configuration_and_is_independent():
    from planners.quantum.vqc_navigation import VQCNavigationPlanner

    original = VQCNavigationPlanner(
        layers=3,
        shots=64,
        max_iterations=55,
        max_training_samples=12,
        seed=23,
    )

    clone = RecoveryPlanner._clone_planner(
        original,
        Maze(width=7, height=7, seed=42),
        (1, 1),
        (4, 4),
    )

    assert isinstance(clone, VQCNavigationPlanner)
    assert clone is not original
    assert clone.layers == 3
    assert clone.shots == 64
    assert clone.max_iterations == 55
    assert clone.max_training_samples == 12
    assert clone.seed == 23

    clone.max_training_samples = 4
    assert original.max_training_samples == 12


def test_qaoa_recovery_uses_quantum_measurement_without_mutating_agent():
    from planners.quantum.qaoa_navigation import QAOANavigationPlanner

    maze = Maze(width=7, height=7, seed=42)
    planner = QAOANavigationPlanner(
        candidates=3,
        layers=1,
        shots=64,
        max_iterations=14,
        penalty=2.0,
        seed=42,
    )
    agent = BaseAgent(
        agent_id="TEST_QAOA",
        start=(1, 1),
        goal=(5, 5),
        planner=planner,
        family="quantum",
    )

    result = RecoveryPlanner(max_candidates=1).plan(
        agent=agent,
        maze=maze,
        dynamic_environment=None,
        agents=[agent],
        simulation_time=0.0,
    )

    assert result is not None
    assert result.path[0] == (1, 1)
    assert result.path[-1] == result.waypoint
    assert isinstance(result.planner, QAOANavigationPlanner)
    assert result.planner is not planner
    assert result.planner.last_metadata["selection_source"] == "qaoa_measurement"
    assert result.planner.last_metadata["quantum_circuit_executions"] > 0
    assert agent.planner is planner
    assert agent.position == (1, 1)
    assert agent.goal == (5, 5)


def test_vqc_recovery_uses_quantum_policy_without_fallback_or_mutation():
    from planners.quantum.vqc_navigation import VQCNavigationPlanner

    maze = Maze(width=7, height=7, seed=42)
    planner = VQCNavigationPlanner(
        layers=1,
        shots=32,
        max_iterations=60,
        max_training_samples=8,
        seed=42,
    )
    agent = BaseAgent(
        agent_id="TEST_VQC",
        start=(1, 1),
        goal=(5, 5),
        planner=planner,
        family="quantum",
    )

    result = RecoveryPlanner(max_candidates=1).plan(
        agent=agent,
        maze=maze,
        dynamic_environment=None,
        agents=[agent],
        simulation_time=0.0,
    )

    assert result is not None
    assert result.path[0] == (1, 1)
    assert result.path[-1] == result.waypoint
    assert isinstance(result.planner, VQCNavigationPlanner)
    assert result.planner is not planner
    assert result.planner.last_metadata["path_source"] == "vqc_policy"
    assert result.planner.last_metadata["fallback_used"] is False
    assert result.planner.last_metadata["quantum_circuit_executions"] > 0
    assert agent.planner is planner
    assert agent.position == (1, 1)
    assert agent.goal == (5, 5)
