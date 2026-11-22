import pytest

from environment.moving_obstacle import MovingObstacle
from security.physical.collision import (
    is_safe_during_motion_to_obstacle,
    minimum_clearance_during_motion_to_obstacle,
)


def test_stationary_obstacle_collision():
    obstacle = MovingObstacle(
        (0.0, 0.0), (0.0, 0.0), 2.0, 1.0
    )

    clearance = minimum_clearance_during_motion_to_obstacle(
        (0.0, 2.0), (0.0, 2.0),
        obstacle, 0.0, 4.0,
    )

    assert clearance == pytest.approx(1.15)


def test_obstacle_crosses_stationary_robot():
    obstacle = MovingObstacle(
        (-2.0, 0.0), (2.0, 0.0), 1.0, 2.0
    )

    clearance = minimum_clearance_during_motion_to_obstacle(
        (0.0, 0.0), (0.0, 0.0),
        obstacle, 0.0, 4.0,
    )

    assert clearance == pytest.approx(-0.35)
    assert not is_safe_during_motion_to_obstacle(
        (0.0, 0.0), (0.0, 0.0),
        obstacle, 0.0, 4.0,
    )


def test_robot_crosses_stationary_obstacle():
    obstacle = MovingObstacle(
        (0.0, 0.0), (0.0, 0.0), 0.0, 1.0
    )

    clearance = minimum_clearance_during_motion_to_obstacle(
        (-2.0, 0.0), (2.0, 0.0),
        obstacle, 0.0, 2.0,
    )

    assert clearance == pytest.approx(-0.35)


def test_collision_during_motion_despite_safe_endpoints():
    obstacle = MovingObstacle(
        (0.0, 0.0), (0.0, 0.0), 0.0, 1.0
    )

    clearance = minimum_clearance_during_motion_to_obstacle(
        (-2.0, 0.0), (2.0, 0.0),
        obstacle, 0.0, 2.0,
    )

    assert clearance < 0.0


def test_obstacle_stationary_before_and_after_motion():
    obstacle = MovingObstacle(
        (0.0, 0.0), (3.0, 0.0), 2.0, 1.0
    )

    before = minimum_clearance_during_motion_to_obstacle(
        (0.0, 2.0), (0.0, 2.0),
        obstacle, 0.0, 1.0,
    )

    after = minimum_clearance_during_motion_to_obstacle(
        (3.0, 2.0), (3.0, 2.0),
        obstacle, 4.0, 5.0,
    )

    assert before == pytest.approx(1.15)
    assert after == pytest.approx(1.15)


def test_collision_only_during_obstacle_movement():
    obstacle = MovingObstacle(
        (-2.0, 0.0), (2.0, 0.0), 2.0, 2.0
    )

    clearance = minimum_clearance_during_motion_to_obstacle(
        (0.0, 0.0), (0.0, 0.0),
        obstacle, 0.0, 5.0,
    )

    assert clearance == pytest.approx(-0.35)


def test_zero_duration_interval_checks_static_clearance():
    obstacle = MovingObstacle(
        (0.0, 0.0), (2.0, 0.0), 1.0, 2.0
    )

    clearance = minimum_clearance_during_motion_to_obstacle(
        (0.0, 2.0), (0.0, 2.0),
        obstacle, 1.5, 1.5,
    )

    assert clearance == pytest.approx(1.15)


def test_invalid_interval_order_is_rejected():
    obstacle = MovingObstacle(
        (0.0, 0.0), (1.0, 0.0), 0.0, 1.0
    )

    with pytest.raises(ValueError, match="interval_end"):
        minimum_clearance_during_motion_to_obstacle(
            (0.0, 0.0), (1.0, 0.0),
            obstacle, 2.0, 1.0,
        )


def test_negative_safety_margin_is_rejected():
    obstacle = MovingObstacle(
        (0.0, 0.0), (1.0, 0.0), 0.0, 1.0
    )

    with pytest.raises(ValueError, match="safety_margin"):
        minimum_clearance_during_motion_to_obstacle(
            (0.0, 2.0), (1.0, 2.0),
            obstacle, 0.0, 1.0,
            safety_margin=-0.1,
        )


def test_non_moving_obstacle_type_is_rejected():
    with pytest.raises(TypeError, match="MovingObstacle"):
        minimum_clearance_during_motion_to_obstacle(
            (0.0, 0.0), (1.0, 0.0),
            object(), 0.0, 1.0,
        )
