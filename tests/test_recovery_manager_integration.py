from dataclasses import dataclass, field

from environment.maze import Maze
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


def test_persistent_swap_deadlock_is_detected():
    maze = Maze(width=25, height=25, seed=42)

    r1 = AgentStub(
        "R1", (1, 1), (3, 1),
        [(1, 1), (2, 1), (3, 1)],
    )
    r2 = AgentStub(
        "R2", (2, 1), (1, 1),
        [(2, 1), (1, 1)],
    )

    manager = MultiAgentManager(
        maze,
        [r1, r2],
        blocked_threshold=5,
    )

    for _ in range(4):
        result = manager.step()
        assert result["moved"] == 0

    assert manager.recovery_coordinator.get_blocked_agents() == []

    result = manager.step()

    assert result["moved"] == 0
    assert manager.recovery_coordinator.get_blocked_agents() == [
        "R1", "R2"
    ]
    assert manager.recovery_coordinator.get_blocked_cycles("R1") == 5
    assert manager.recovery_coordinator.get_blocked_cycles("R2") == 5


def test_agent_without_movement_proposal_is_not_marked_blocked():
    maze = Maze(width=25, height=25, seed=42)

    agent = AgentStub(
        "R1",
        (1, 1),
        (3, 1),
        [],
        status="idle",
    )

    manager = MultiAgentManager(
        maze,
        [agent],
        blocked_threshold=2,
    )

    for _ in range(5):
        result = manager.step()
        assert result["proposed"] == 0

    assert manager.recovery_coordinator.get_blocked_cycles("R1") == 0
    assert not manager.recovery_coordinator.is_blocked("R1")
