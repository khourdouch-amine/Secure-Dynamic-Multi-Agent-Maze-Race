import math

import pytest

from security.physical.cbf import CBFController


def test_no_obstacle_preserves_nominal_control():
    controller = CBFController()

    result = controller.filter(
        position=(0.0, 0.0),
        orientation=0.0,
        nominal_control=(1.0, 0.5),
    )

    assert result.success
    assert result.status == "solved"
    assert result.control == pytest.approx((1.0, 0.5), abs=1e-5)
    assert result.minimum_clearance is None


def test_control_respects_physical_speed_limits():
    controller = CBFController()

    result = controller.filter(
        position=(0.0, 0.0),
        orientation=0.0,
        nominal_control=(5.0, 4.0),
    )

    assert result.success
    assert abs(result.control[0]) <= 2.0 + 1e-6
    assert abs(result.control[1]) <= 2.0 + 1e-6
    assert result.control == pytest.approx((2.0, 2.0), abs=1e-5)


def test_far_obstacle_does_not_significantly_change_control():
    controller = CBFController()

    result = controller.filter(
        position=(0.0, 0.0),
        orientation=0.0,
        nominal_control=(1.5, 0.4),
        obstacles=[(5.0, 0.0)],
    )

    assert result.success
    assert result.control == pytest.approx((1.5, 0.4), abs=1e-5)
    assert result.minimum_clearance == pytest.approx(4.15)


def test_obstacle_ahead_reduces_forward_velocity():
    controller = CBFController()

    result = controller.filter(
        position=(0.0, 0.0),
        orientation=0.0,
        nominal_control=(1.0, 0.0),
        obstacles=[(1.0, 0.0)],
    )

    assert result.success
    assert result.control[0] <= 0.3 + 1e-5
    assert result.control[0] >= -1e-5
    assert result.minimum_clearance == pytest.approx(0.15)


def test_obstacle_behind_does_not_block_forward_motion():
    controller = CBFController()

    result = controller.filter(
        position=(0.0, 0.0),
        orientation=0.0,
        nominal_control=(1.0, 0.0),
        obstacles=[(-1.0, 0.0)],
    )

    assert result.success
    assert result.control == pytest.approx((1.0, 0.0), abs=1e-5)


def test_two_opposite_obstacles_are_handled_simultaneously():
    controller = CBFController()

    result = controller.filter(
        position=(0.0, 0.0),
        orientation=0.0,
        nominal_control=(1.0, 0.0),
        obstacles=[
            (1.0, 0.0),
            (-1.0, 0.0),
        ],
    )

    assert result.success
    assert result.control[0] <= 0.3 + 1e-5
    assert result.control[0] >= -0.3 - 1e-5
    assert result.minimum_clearance == pytest.approx(0.15)


def test_initial_position_inside_required_clearance_is_rejected():
    controller = CBFController()

    result = controller.filter(
        position=(0.0, 0.0),
        orientation=0.0,
        nominal_control=(1.0, 0.0),
        obstacles=[(0.8, 0.0)],
    )

    assert not result.success
    assert result.status == "initial_state_unsafe"
    assert result.control == (0.0, 0.0)
    assert result.minimum_clearance < 0.0


def test_orientation_changes_obstacle_constraint():
    controller = CBFController()

    result = controller.filter(
        position=(0.0, 0.0),
        orientation=math.pi / 2.0,
        nominal_control=(1.0, 0.0),
        obstacles=[(1.0, 0.0)],
    )

    assert result.success
    assert result.control == pytest.approx((1.0, 0.0), abs=1e-5)


def test_invalid_controller_parameters_are_rejected():
    with pytest.raises(ValueError):
        CBFController(robot_radius=0.0)

    with pytest.raises(ValueError):
        CBFController(obstacle_margin=-0.1)
