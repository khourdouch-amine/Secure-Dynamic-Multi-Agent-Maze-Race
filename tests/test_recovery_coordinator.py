import math

import pytest

from simulation.recovery_coordinator import RecoveryCoordinator


def test_first_observation_does_not_count_as_blockage():
    coordinator = RecoveryCoordinator(blocked_threshold=5)

    assert coordinator.observe("R1", (1, 1), 0.0) is False
    assert coordinator.get_blocked_cycles("R1") == 0


def test_detects_blockage_after_five_unchanged_cycles():
    coordinator = RecoveryCoordinator(blocked_threshold=5)

    coordinator.observe("R1", (1, 1), 0.0)

    for cycle in range(1, 5):
        assert coordinator.observe(
            "R1", (1, 1), cycle / 30
        ) is False

    assert coordinator.observe("R1", (1, 1), 5 / 30) is True
    assert coordinator.get_blocked_cycles("R1") == 5


def test_actual_movement_resets_blockage_counter():
    coordinator = RecoveryCoordinator(blocked_threshold=3)

    coordinator.observe("R1", (1, 1), 0.0)
    coordinator.observe("R1", (1, 1), 1.0)
    coordinator.observe("R1", (1, 1), 2.0)

    assert coordinator.get_blocked_cycles("R1") == 2

    assert coordinator.observe("R1", (2, 1), 3.0) is False
    assert coordinator.get_blocked_cycles("R1") == 0


def test_agents_are_tracked_independently():
    coordinator = RecoveryCoordinator(blocked_threshold=2)

    coordinator.observe("R1", (1, 1), 0.0)
    coordinator.observe("R2", (5, 5), 0.0)

    coordinator.observe("R1", (1, 1), 1.0)
    coordinator.observe("R2", (5, 5), 1.0)
    coordinator.observe("R1", (1, 1), 2.0)

    assert coordinator.is_blocked("R1")
    assert not coordinator.is_blocked("R2")
    assert coordinator.get_blocked_agents() == ["R1"]


def test_reset_agent_clears_only_selected_agent():
    coordinator = RecoveryCoordinator(blocked_threshold=1)

    coordinator.observe("R1", (1, 1), 0.0)
    coordinator.observe("R1", (1, 1), 1.0)
    coordinator.observe("R2", (2, 2), 0.0)
    coordinator.observe("R2", (2, 2), 1.0)

    coordinator.reset_agent("R1")

    assert not coordinator.is_blocked("R1")
    assert coordinator.is_blocked("R2")
    assert coordinator.get_blocked_cycles("R1") == 0


def test_reset_clears_all_agents():
    coordinator = RecoveryCoordinator(blocked_threshold=1)

    coordinator.observe("R1", (1, 1), 0.0)
    coordinator.observe("R1", (1, 1), 1.0)

    coordinator.reset()

    assert coordinator.get_blocked_agents() == []
    assert coordinator.get_blocked_cycles("R1") == 0


@pytest.mark.parametrize("threshold", [0, -1, 1.5, True, "5"])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        RecoveryCoordinator(blocked_threshold=threshold)


@pytest.mark.parametrize(
    "position",
    [
        (1,),
        (1, 2, 3),
        (1.5, 2),
        (True, 2),
        "12",
        None,
    ],
)
def test_invalid_positions_are_rejected(position):
    coordinator = RecoveryCoordinator()

    with pytest.raises(ValueError):
        coordinator.observe("R1", position, 0.0)


@pytest.mark.parametrize("timestamp", [-1, math.inf, -math.inf, math.nan])
def test_invalid_simulation_time_is_rejected(timestamp):
    coordinator = RecoveryCoordinator()

    with pytest.raises(ValueError):
        coordinator.observe("R1", (1, 1), timestamp)


def test_simulation_time_cannot_move_backwards():
    coordinator = RecoveryCoordinator()

    coordinator.observe("R1", (1, 1), 2.0)

    with pytest.raises(ValueError):
        coordinator.observe("R1", (1, 1), 1.0)


def test_agent_id_cannot_be_none():
    coordinator = RecoveryCoordinator()

    with pytest.raises(ValueError):
        coordinator.observe(None, (1, 1), 0.0)


def test_voluntary_traffic_yield_preserves_blockage_history():
    coordinator = RecoveryCoordinator(blocked_threshold=3)

    coordinator.observe("R1", (1, 1), 0.0)
    coordinator.observe("R1", (1, 1), 1.0)

    assert coordinator.get_blocked_cycles("R1") == 1

    coordinator.pause_agent("R1")

    assert coordinator.get_blocked_cycles("R1") == 1
    assert coordinator.observe("R1", (1, 1), 3.0) is False
    assert coordinator.get_blocked_cycles("R1") == 2
