import numpy as np
import pytest

from agents.robot_model import DifferentialDriveRobot


def test_default_initial_state():
    robot = DifferentialDriveRobot()

    assert robot.position == pytest.approx((1.0, 1.0))
    assert robot.orientation == pytest.approx(0.0)
    assert robot.radius == pytest.approx(0.25)


def test_straight_line_motion():
    robot = DifferentialDriveRobot(x=1.0, y=2.0, theta=0.0)

    state = robot.step((1.0, 0.0), dt=0.5)

    assert state == pytest.approx((1.5, 2.0, 0.0))


def test_motion_respects_robot_orientation():
    robot = DifferentialDriveRobot(
        x=1.0,
        y=2.0,
        theta=np.pi / 2,
    )

    state = robot.step((1.0, 0.0), dt=0.5)

    assert state[0] == pytest.approx(1.0)
    assert state[1] == pytest.approx(2.5)


def test_pure_rotation():
    robot = DifferentialDriveRobot()

    state = robot.step((0.0, 1.0), dt=0.5)

    assert state[0] == pytest.approx(1.0)
    assert state[1] == pytest.approx(1.0)
    assert state[2] == pytest.approx(0.5)


def test_linear_and_angular_commands_are_clipped():
    robot = DifferentialDriveRobot(
        max_linear_speed=2.0,
        max_angular_speed=1.0,
    )

    state = robot.step((5.0, 3.0), dt=1.0)

    assert state[0] == pytest.approx(3.0)
    assert state[1] == pytest.approx(1.0)
    assert state[2] == pytest.approx(1.0)


def test_orientation_is_normalized():
    robot = DifferentialDriveRobot(theta=np.pi - 0.1)

    state = robot.step((0.0, 1.0), dt=0.2)

    assert -np.pi <= state[2] < np.pi
    assert state[2] == pytest.approx(-np.pi + 0.1)


def test_zero_time_step_does_not_change_state():
    robot = DifferentialDriveRobot(x=2.0, y=3.0, theta=0.5)

    initial_state = robot.state.copy()
    state = robot.step((1.0, 1.0), dt=0.0)

    assert state == pytest.approx(initial_state)


@pytest.mark.parametrize("dt", [-0.1, np.inf, -np.inf, np.nan])
def test_invalid_time_step_is_rejected(dt):
    robot = DifferentialDriveRobot()

    with pytest.raises(ValueError):
        robot.step((1.0, 0.0), dt=dt)


@pytest.mark.parametrize(
    "control",
    [
        (1.0,),
        (1.0, 2.0, 3.0),
        (np.nan, 0.0),
        (0.0, np.inf),
        (0.0, -np.inf),
    ],
)
def test_invalid_control_is_rejected(control):
    robot = DifferentialDriveRobot()

    with pytest.raises(ValueError):
        robot.step(control)


@pytest.mark.parametrize(
    "parameters",
    [
        {"x": np.nan},
        {"y": np.inf},
        {"theta": -np.inf},
        {"max_linear_speed": 0.0},
        {"max_linear_speed": -1.0},
        {"max_angular_speed": 0.0},
        {"max_angular_speed": np.inf},
    ],
)
def test_invalid_constructor_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        DifferentialDriveRobot(**parameters)
