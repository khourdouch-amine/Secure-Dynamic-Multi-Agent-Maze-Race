from dataclasses import dataclass, field

import pytest

from environment.maze import Maze
from environment.moving_obstacle import MovingObstacle
from simulation.multi_agent_manager import MultiAgentManager


@dataclass
class AgentStub:
    agent_id: str
    position: tuple
    goal: tuple
    path: list
    status: str = "planned"
    metrics: dict = field(default_factory=lambda: {
        "moves": 0,
        "collisions": 0,
        "replans": 0,
    })

    @property
    def reached_goal(self):
        return self.position == self.goal


@pytest.fixture
def maze():
    return Maze(width=25, height=25, seed=42)


def test_independent_agents_move_simultaneously(maze):
    a1 = AgentStub("R1", (1, 1), (3, 1), [(1, 1), (2, 1), (3, 1)])
    a2 = AgentStub("R2", (1, 3), (3, 3), [(1, 3), (2, 3), (3, 3)])

    manager = MultiAgentManager(maze, [a1, a2])
    result = manager.step()

    assert result["moved"] == 2
    assert a1.position == (2, 1)
    assert a2.position == (2, 3)
    assert manager.get_metrics()["successful_moves"] == 2


def test_two_agents_cannot_enter_the_same_cell(maze):
    a1 = AgentStub("R1", (1, 2), (3, 2), [(1, 2), (2, 2), (3, 2)])
    a2 = AgentStub("R2", (3, 2), (1, 2), [(3, 2), (2, 2), (1, 2)])

    manager = MultiAgentManager(maze, [a1, a2])
    result = manager.step()

    assert result["moved"] == 1
    assert result["blocked"] == 1
    assert len({a1.position, a2.position}) == 2
    assert manager.get_metrics()["conflicts"] == 1


def test_agents_cannot_swap_positions_in_one_step(maze):
    a1 = AgentStub("R1", (1, 1), (3, 1), [(1, 1), (2, 1), (3, 1)])
    a2 = AgentStub("R2", (2, 1), (1, 1), [(2, 1), (1, 1)])

    manager = MultiAgentManager(maze, [a1, a2])
    result = manager.step()

    assert result["moved"] == 0
    assert a1.position == (1, 1)
    assert a2.position == (2, 1)
    assert result["blocked"] == 2


def test_duplicate_agent_ids_are_rejected(maze):
    a1 = AgentStub("R1", (1, 1), (3, 1), [(1, 1), (2, 1), (3, 1)])
    a2 = AgentStub("R1", (1, 3), (3, 3), [(1, 3), (2, 3), (3, 3)])

    with pytest.raises(ValueError, match="unique"):
        MultiAgentManager(maze, [a1, a2])


def test_duplicate_initial_positions_are_rejected(maze):
    a1 = AgentStub("R1", (1, 1), (3, 1), [(1, 1), (2, 1), (3, 1)])
    a2 = AgentStub("R2", (1, 1), (3, 3), [(1, 1), (1, 2), (3, 3)])

    with pytest.raises(ValueError, match="share an initial position"):
        MultiAgentManager(maze, [a1, a2])


class DynamicEnvironmentStub:
    def __init__(self, obstacle_trajectories=()):
        self.calls = []
        self.obstacle_trajectories = tuple(obstacle_trajectories)

    def update(self, simulation_time, occupied_cells=()):
        self.calls.append({
            "time": simulation_time,
            "occupied": set(occupied_cells),
        })
        return {
            "walls_updated": False,
            "obstacles_updated": False,
            "obstacle_trajectories": self.obstacle_trajectories,
        }


def test_simulation_clock_advances_by_fixed_time_step(maze):
    agent = AgentStub(
        "R1", (1, 1), (3, 1),
        [(1, 1), (2, 1), (3, 1)]
    )
    environment = DynamicEnvironmentStub()
    manager = MultiAgentManager(
        maze, [agent], dynamic_environment=environment
    )

    manager.step()
    manager.step()

    assert manager.simulation_time == pytest.approx(2.0 / 30.0)
    assert len(environment.calls) == 2
    assert environment.calls[0]["time"] == pytest.approx(1.0 / 30.0)
    assert environment.calls[1]["time"] == pytest.approx(2.0 / 30.0)


def test_dynamic_environment_receives_occupied_positions(maze):
    agent = AgentStub(
        "R1", (1, 1), (3, 1),
        [(1, 1), (2, 1), (3, 1)]
    )
    environment = DynamicEnvironmentStub()
    manager = MultiAgentManager(
        maze, [agent], dynamic_environment=environment
    )

    manager.step()

    assert environment.calls[0]["occupied"] == {(1, 1)}


def test_simulation_clock_cannot_move_backwards(maze):
    agent = AgentStub(
        "R1", (1, 1), (3, 1),
        [(1, 1), (2, 1), (3, 1)]
    )
    manager = MultiAgentManager(maze, [agent])

    manager.step(simulation_time=1.0)

    with pytest.raises(ValueError, match="backwards"):
        manager.step(simulation_time=0.5)


def test_agents_cannot_start_on_blocked_cells(maze):
    from environment.maze import DYNAMIC_WALL

    maze.set_cell(2, 2, DYNAMIC_WALL)

    agent = AgentStub(
        "R1",
        (2, 2),
        (3, 3),
        [(2, 2), (2, 3), (3, 3)],
    )

    with pytest.raises(ValueError, match="not walkable"):
        MultiAgentManager(maze, [agent])


def test_crossing_robot_trajectories_are_physically_filtered(maze):
    a1 = AgentStub("R1", (1, 1), (2, 1), [(1, 1), (2, 1)])
    a2 = AgentStub("R2", (1, 2), (1, 1), [(1, 2), (1, 1)])

    manager = MultiAgentManager(
        maze,
        [a1, a2],
        time_step=1.0,
    )

    result = manager.step()

    assert result["moved"] == 1
    assert result["safety_blocked"] == 1
    assert result["blocked"] == 1

    assert a1.position == (2, 1)
    assert a2.position == (1, 2)

    # A prevented collision is not an actual collision.
    assert a1.metrics["collisions"] == 0
    assert a2.metrics["collisions"] == 0

    assert manager.get_metrics()["successful_moves"] == 1
    assert manager.get_metrics()["blocked_moves"] == 1


def test_robot_move_is_blocked_by_crossing_moving_obstacle(maze):
    agent = AgentStub(
        "R1",
        (1, 1),
        (2, 1),
        [(1, 1), (2, 1)],
    )

    obstacle = MovingObstacle(
        start_position=(1.5, 0.0),
        target_position=(1.5, 2.0),
        start_time=0.0,
        duration=1.0,
    )

    environment = DynamicEnvironmentStub(
        obstacle_trajectories=(obstacle,),
    )

    manager = MultiAgentManager(
        maze,
        [agent],
        dynamic_environment=environment,
        time_step=1.0,
    )

    result = manager.step()

    assert result["moved"] == 0
    assert result["safety_blocked"] == 1
    assert result["blocked"] == 1
    assert agent.position == (1, 1)
    assert agent.metrics["collisions"] == 0


def test_voluntary_traffic_yield_preserves_recovery_history(maze):
    from types import SimpleNamespace

    class YieldingTrafficCoordinator:
        def coordinate_with_yields(self, agents, proposals):
            return SimpleNamespace(
                proposals={},
                yielded_agents=frozenset({"R1"}),
            )

    agent = AgentStub(
        "R1",
        (1, 1),
        (3, 1),
        [(1, 1), (2, 1), (3, 1)],
    )

    manager = MultiAgentManager(
        maze,
        [agent],
        traffic_coordinator=YieldingTrafficCoordinator(),
    )

    result = manager.step()

    assert agent.position == (1, 1)
    assert agent.status == "waiting"
    assert result["traffic_yielded"] == 1
    assert result["blocked"] == 0
    assert result["moved"] == 0

    # A subsequent unchanged observation must continue the preserved history.
    manager.recovery_coordinator.observe(
        "R1",
        agent.position,
        manager.simulation_time + manager.time_step,
    )
    assert manager.recovery_coordinator.get_blocked_cycles("R1") == 1


def test_traffic_coordinator_cannot_silently_drop_an_agent_proposal(maze):
    from types import SimpleNamespace

    class DroppingTrafficCoordinator:
        def coordinate_with_yields(self, agents, proposals):
            return SimpleNamespace(
                proposals={},
                yielded_agents=frozenset(),
            )

    agent = AgentStub(
        "R1",
        (1, 1),
        (3, 1),
        [(1, 1), (2, 1), (3, 1)],
    )

    manager = MultiAgentManager(
        maze,
        [agent],
        traffic_coordinator=DroppingTrafficCoordinator(),
    )

    with pytest.raises(ValueError, match="account"):
        manager.step()


def test_priority_order_rotates_deterministically_and_wraps(maze):
    agents = [
        AgentStub("R1", (1, 1), (3, 1), [(1, 1), (2, 1), (3, 1)]),
        AgentStub("R2", (1, 3), (3, 3), [(1, 3), (2, 3), (3, 3)]),
        AgentStub("R3", (1, 5), (3, 5), [(1, 5), (2, 5), (3, 5)]),
    ]
    manager = MultiAgentManager(maze, agents)

    expected_orders = [
        ["R1", "R2", "R3"],
        ["R2", "R3", "R1"],
        ["R3", "R1", "R2"],
        ["R1", "R2", "R3"],
    ]

    for expected in expected_orders:
        assert [
            agent.agent_id for agent in manager._priority_order()
        ] == expected
        manager.resolve_moves({})


def test_repeated_destination_conflict_rotates_winner(maze):
    target = next(
        cell
        for y in range(1, maze.height - 1)
        for x in range(1, maze.width - 1)
        if maze.is_walkable(x, y)
        for cell in [(x, y)]
        if len(maze.get_neighbors(*cell)) >= 2
    )

    neighbors = maze.get_neighbors(*target)
    position_1, position_2 = neighbors[:2]

    agents = [
        AgentStub("R1", position_1, target, [position_1, target]),
        AgentStub("R2", position_2, target, [position_2, target]),
    ]
    manager = MultiAgentManager(maze, agents)

    winners = []

    for _ in range(4):
        accepted = manager.resolve_moves({
            "R1": target,
            "R2": target,
        })
        winners.append(next(iter(accepted)))

    assert winners == ["R1", "R2", "R1", "R2"]


def test_physical_filter_receives_priority_order_from_current_cycle(maze):
    agents = [
        AgentStub("R1", (1, 1), (3, 1), [(1, 1), (2, 1), (3, 1)]),
        AgentStub("R2", (1, 3), (3, 3), [(1, 3), (2, 3), (3, 3)]),
        AgentStub("R3", (1, 5), (3, 5), [(1, 5), (2, 5), (3, 5)]),
    ]
    manager = MultiAgentManager(maze, agents)

    captured = {}
    original_filter = manager._filter_physical_moves

    def recording_filter(*args, **kwargs):
        captured["priority_order"] = [
            agent.agent_id
            for agent in kwargs["priority_order"]
        ]
        return original_filter(*args, **kwargs)

    manager._filter_physical_moves = recording_filter

    manager.step()

    assert captured["priority_order"] == ["R1", "R2", "R3"]
    assert manager.priority_offset == 1



def test_moving_robot_respects_exact_clearance_from_stationary_robot(maze):
    stationary = AgentStub(
        "R1",
        (2, 1),
        (2, 1),
        [(2, 1)],
        status="reached_goal",
    )
    moving = AgentStub(
        "R2",
        (1, 1),
        (1, 2),
        [(1, 1), (1, 2)],
    )

    manager = MultiAgentManager(maze, [stationary, moving], time_step=1.0)
    result = manager.step()

    assert result["moved"] == 1
    assert result["safety_blocked"] == 0
    assert result["robot_safety_blocks"] == 0
    assert stationary.position == (2, 1)
    assert moving.position == (1, 2)


def test_physical_safety_rechecks_conflicts_after_each_rejection(maze):
    agent_a = AgentStub(
        "A",
        (1, 1),
        (2, 1),
        [(1, 1), (2, 1)],
    )
    agent_b = AgentStub(
        "B",
        (1, 2),
        (1, 1),
        [(1, 2), (1, 1)],
    )
    agent_c = AgentStub(
        "C",
        (1, 3),
        (1, 2),
        [(1, 3), (1, 2)],
    )

    manager = MultiAgentManager(
        maze,
        [agent_a, agent_b, agent_c],
        time_step=1.0,
    )
    result = manager.step()

    assert result["moved"] == 1
    assert result["safety_blocked"] == 2
    assert result["robot_safety_blocks"] == 2

    assert agent_a.position == (2, 1)
    assert agent_b.position == (1, 2)
    assert agent_c.position == (1, 3)
