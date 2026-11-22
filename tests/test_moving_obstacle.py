import pytest

from environment.moving_obstacle import MovingObstacle


def make_obstacle():
    return MovingObstacle(
        start_position=(2.0, 3.0),
        target_position=(4.0, 7.0),
        start_time=2.0,
        duration=4.0,
        size=1.0,
    )


def test_position_before_movement():
    obstacle = make_obstacle()

    assert obstacle.position_at(1.0) == (2.0, 3.0)


def test_position_at_movement_start():
    obstacle = make_obstacle()

    assert obstacle.position_at(2.0) == (2.0, 3.0)


def test_position_at_midpoint():
    obstacle = make_obstacle()

    assert obstacle.position_at(4.0) == pytest.approx((3.0, 5.0))


def test_position_at_movement_end():
    obstacle = make_obstacle()

    assert obstacle.position_at(6.0) == (4.0, 7.0)


def test_position_after_movement():
    obstacle = make_obstacle()

    assert obstacle.position_at(10.0) == (4.0, 7.0)


def test_constant_velocity_during_movement():
    obstacle = make_obstacle()

    assert obstacle.velocity == pytest.approx((0.5, 1.0))
    assert obstacle.velocity_at(2.0) == pytest.approx((0.5, 1.0))
    assert obstacle.velocity_at(4.0) == pytest.approx((0.5, 1.0))
    assert obstacle.velocity_at(5.999) == pytest.approx((0.5, 1.0))


def test_velocity_is_zero_outside_movement_interval():
    obstacle = make_obstacle()

    assert obstacle.velocity_at(1.0) == (0.0, 0.0)
    assert obstacle.velocity_at(6.0) == (0.0, 0.0)
    assert obstacle.velocity_at(8.0) == (0.0, 0.0)


def test_movement_state():
    obstacle = make_obstacle()

    assert obstacle.is_moving(1.0) is False
    assert obstacle.is_moving(2.0) is True
    assert obstacle.is_moving(4.0) is True
    assert obstacle.is_moving(6.0) is False


def test_end_time():
    obstacle = make_obstacle()

    assert obstacle.end_time == pytest.approx(6.0)


def test_obstacle_size_is_preserved():
    obstacle = make_obstacle()

    assert obstacle.size == pytest.approx(1.0)


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_non_positive_duration_is_rejected(duration):
    with pytest.raises(ValueError):
        MovingObstacle((0, 0), (1, 0), 0.0, duration)


@pytest.mark.parametrize("size", [0.0, -1.0])
def test_non_positive_size_is_rejected(size):
    with pytest.raises(ValueError):
        MovingObstacle((0, 0), (1, 0), 0.0, 1.0, size)


@pytest.mark.parametrize(
    "position",
    [
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (0.0,),
        (0.0, 1.0, 2.0),
    ],
)
def test_invalid_start_position_is_rejected(position):
    with pytest.raises(ValueError):
        MovingObstacle(position, (1.0, 1.0), 0.0, 1.0)


@pytest.mark.parametrize(
    "start_time",
    [float("nan"), float("inf"), -float("inf")],
)
def test_non_finite_start_time_is_rejected(start_time):
    with pytest.raises(ValueError):
        MovingObstacle((0, 0), (1, 0), start_time, 1.0)


@pytest.mark.parametrize(
    "simulation_time",
    [float("nan"), float("inf"), -float("inf")],
)
def test_non_finite_simulation_time_is_rejected(simulation_time):
    obstacle = make_obstacle()

    with pytest.raises(ValueError):
        obstacle.position_at(simulation_time)

    with pytest.raises(ValueError):
        obstacle.velocity_at(simulation_time)

    with pytest.raises(ValueError):
        obstacle.is_moving(simulation_time)


def test_stationary_obstacle_has_zero_velocity():
    obstacle = MovingObstacle(
        start_position=(3.0, 4.0),
        target_position=(3.0, 4.0),
        start_time=1.0,
        duration=2.0,
    )

    assert obstacle.position_at(2.0) == (3.0, 4.0)
    assert obstacle.velocity == (0.0, 0.0)
    assert obstacle.velocity_at(1.5) == (0.0, 0.0)


def test_overflowing_end_time_is_rejected():
    with pytest.raises(ValueError, match="end_time"):
        MovingObstacle(
            (0.0, 0.0),
            (1.0, 0.0),
            1e308,
            1e308,
        )


def test_overflowing_displacement_is_rejected():
    with pytest.raises(ValueError, match="displacement"):
        MovingObstacle(
            (-1e308, 0.0),
            (1e308, 0.0),
            0.0,
            1.0,
        )


def test_non_finite_velocity_is_rejected():
    with pytest.raises(ValueError, match="velocity"):
        MovingObstacle(
            (0.0, 0.0),
            (1.0, 0.0),
            0.0,
            1e-309,
        )
