import math

import pytest

from security.physical.geometry import (
    distance_between_points,
    grid_to_world,
    point_to_square_distance,
    robot_is_safe_from_square,
    robot_robot_clearance,
    robot_square_clearance,
    robots_are_safe,
    world_to_grid,
)


def test_grid_to_world_uses_cell_centers():
    assert grid_to_world((3, 4)) == (3.0, 4.0)
    assert grid_to_world((3, 4), cell_size=2.0) == (6.0, 8.0)


def test_world_to_grid_uses_nearest_cell_center():
    assert world_to_grid((3.2, 4.4)) == (3, 4)
    assert world_to_grid((3.5, 4.5)) == (4, 5)
    assert world_to_grid((3.0, 4.0), cell_size=2.0) == (2, 2)


def test_grid_world_conversion_round_trip():
    for cell in [(0, 0), (1, 1), (12, 8), (24, 24)]:
        assert world_to_grid(grid_to_world(cell)) == cell


def test_euclidean_distance():
    assert distance_between_points((0, 0), (3, 4)) == pytest.approx(5.0)


def test_robot_robot_surface_clearance():
    clearance = robot_robot_clearance((0, 0), (1, 0))

    assert clearance == pytest.approx(0.5)


def test_robot_robot_clearance_detects_touching_and_overlap():
    assert robot_robot_clearance((0, 0), (0.5, 0)) == pytest.approx(0.0)
    assert robot_robot_clearance((0, 0), (0.4, 0)) == pytest.approx(-0.1)


def test_robot_robot_minimum_center_distance():
    assert robots_are_safe((0, 0), (1, 0))
    assert not robots_are_safe((0, 0), (0.99, 0))


def test_point_to_square_distance_outside():
    assert point_to_square_distance((1.5, 0), (0, 0)) == pytest.approx(1.0)


def test_point_to_square_distance_diagonal():
    assert point_to_square_distance((1, 1), (0, 0)) == pytest.approx(
        math.sqrt(0.5)
    )


def test_point_inside_square_has_zero_distance():
    assert point_to_square_distance((0.2, 0.1), (0, 0)) == 0.0


def test_robot_square_surface_clearance():
    clearance = robot_square_clearance((1, 0), (0, 0))

    assert clearance == pytest.approx(0.25)


def test_robot_square_safety_margin():
    assert robot_is_safe_from_square((1.0, 0), (0, 0))
    assert not robot_is_safe_from_square((0.84, 0), (0, 0))


def test_invalid_points_are_rejected():
    with pytest.raises(ValueError):
        distance_between_points((0, 0), (1, float("inf")))


def test_invalid_cell_size_is_rejected():
    with pytest.raises(ValueError):
        grid_to_world((1, 1), cell_size=0)


def test_negative_safety_margin_is_rejected():
    with pytest.raises(ValueError):
        robot_is_safe_from_square((1, 0), (0, 0), margin=-0.1)


def test_minimum_distance_during_motion_detects_crossing():
    from security.physical.geometry import minimum_distance_during_motion

    distance = minimum_distance_during_motion(
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
    )

    assert distance == pytest.approx(0.0)


def test_minimum_distance_during_motion_detects_close_approach():
    from security.physical.geometry import minimum_distance_during_motion

    distance = minimum_distance_during_motion(
        (-1, 0), (1, 0),
        (0, 0.8), (0, 0.8),
    )

    assert distance == pytest.approx(0.8)


def test_minimum_distance_during_motion_respects_synchronization():
    from security.physical.geometry import minimum_distance_during_motion

    # Les trajectoires géométriques se croisent, mais pas au même instant.
    distance = minimum_distance_during_motion(
        (-1, 0), (1, 0),
        (0, 1), (0, 2),
    )

    assert distance > 1.0


def test_robots_are_safe_during_motion():
    from security.physical.geometry import robots_are_safe_during_motion

    assert not robots_are_safe_during_motion(
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
    )

    assert robots_are_safe_during_motion(
        (-1, 0), (1, 0),
        (-1, 2), (1, 2),
    )


def test_stationary_robots_keep_their_distance():
    from security.physical.geometry import minimum_distance_during_motion

    distance = minimum_distance_during_motion(
        (0, 0), (0, 0),
        (1.5, 0), (1.5, 0),
    )

    assert distance == pytest.approx(1.5)


def test_signed_square_distance_outside():
    from security.physical.geometry import point_to_square_signed_distance

    assert point_to_square_signed_distance((1.0, 0), (0, 0)) == pytest.approx(0.5)


def test_signed_square_distance_on_boundary():
    from security.physical.geometry import point_to_square_signed_distance

    assert point_to_square_signed_distance((0.5, 0), (0, 0)) == pytest.approx(0.0)


def test_signed_square_distance_inside():
    from security.physical.geometry import point_to_square_signed_distance

    assert point_to_square_signed_distance((0, 0), (0, 0)) == pytest.approx(-0.5)
    assert point_to_square_signed_distance((0.4, 0), (0, 0)) == pytest.approx(-0.1)


def test_robot_square_clearance_inside_square():
    assert robot_square_clearance((0, 0), (0, 0)) == pytest.approx(-0.75)
    assert robot_square_clearance((0.4, 0), (0, 0)) == pytest.approx(-0.35)


def test_robot_square_clearance_outside_square():
    assert robot_square_clearance((1.0, 0), (0, 0)) == pytest.approx(0.25)


def test_square_distance_gradient_in_front_of_face():
    from security.physical.geometry import point_to_square_distance_gradient

    distance, gradient = point_to_square_distance_gradient((1.0, 0.0), (0, 0))

    assert distance == pytest.approx(0.5)
    assert gradient == pytest.approx((1.0, 0.0))


def test_square_distance_gradient_diagonal_from_corner():
    from security.physical.geometry import point_to_square_distance_gradient

    distance, gradient = point_to_square_distance_gradient((1.0, 1.0), (0, 0))

    assert distance == pytest.approx(math.sqrt(0.5))
    assert gradient[0] == pytest.approx(1.0 / math.sqrt(2.0))
    assert gradient[1] == pytest.approx(1.0 / math.sqrt(2.0))


def test_square_distance_gradient_rejects_inside_point():
    from security.physical.geometry import point_to_square_distance_gradient

    with pytest.raises(ValueError):
        point_to_square_distance_gradient((0.0, 0.0), (0, 0))


def test_square_distance_gradient_rejects_face_boundary():
    from security.physical.geometry import point_to_square_distance_gradient

    with pytest.raises(ValueError):
        point_to_square_distance_gradient((0.5, 0.0), (0, 0))


def test_square_distance_gradient_has_unit_norm():
    from security.physical.geometry import point_to_square_distance_gradient

    _, gradient = point_to_square_distance_gradient((1.0, 1.0), (0, 0))

    assert math.hypot(*gradient) == pytest.approx(1.0)


def test_square_distance_gradient_on_extension_of_vertical_face():
    from security.physical.geometry import point_to_square_distance_gradient

    distance, gradient = point_to_square_distance_gradient((0.5, 1.0), (0, 0))

    assert distance == pytest.approx(0.5)
    assert gradient == pytest.approx((0.0, 1.0))


def test_square_distance_gradient_on_extension_of_horizontal_face():
    from security.physical.geometry import point_to_square_distance_gradient

    distance, gradient = point_to_square_distance_gradient((1.0, 0.5), (0, 0))

    assert distance == pytest.approx(0.5)
    assert gradient == pytest.approx((1.0, 0.0))


def test_moving_square_crosses_stationary_robot():
    from security.physical.geometry import (
        minimum_distance_during_motion_to_square,
    )

    distance = minimum_distance_during_motion_to_square(
        robot_start=(0.0, 0.0),
        robot_end=(0.0, 0.0),
        obstacle_start=(-2.0, 0.0),
        obstacle_end=(2.0, 0.0),
    )

    assert distance == pytest.approx(0.0)


def test_robot_crosses_stationary_square():
    from security.physical.geometry import (
        minimum_distance_during_motion_to_square,
    )

    distance = minimum_distance_during_motion_to_square(
        robot_start=(-2.0, 0.0),
        robot_end=(2.0, 0.0),
        obstacle_start=(0.0, 0.0),
        obstacle_end=(0.0, 0.0),
    )

    assert distance == pytest.approx(0.0)


def test_robot_and_square_move_in_same_direction():
    from security.physical.geometry import (
        minimum_distance_during_motion_to_square,
    )

    distance = minimum_distance_during_motion_to_square(
        robot_start=(0.0, 2.0),
        robot_end=(4.0, 2.0),
        obstacle_start=(0.0, 0.0),
        obstacle_end=(4.0, 0.0),
    )

    assert distance == pytest.approx(1.5)


def test_diagonal_clearance_from_square_corner():
    from security.physical.geometry import (
        minimum_distance_during_motion_to_square,
    )

    distance = minimum_distance_during_motion_to_square(
        robot_start=(1.0, 1.0),
        robot_end=(1.0, 1.0),
        obstacle_start=(0.0, 0.0),
        obstacle_end=(0.0, 0.0),
    )

    assert distance == pytest.approx(2 ** 0.5 / 2.0)


def test_endpoints_safe_but_mid_motion_collision():
    from security.physical.geometry import (
        minimum_distance_during_motion_to_square,
    )

    distance = minimum_distance_during_motion_to_square(
        robot_start=(-2.0, 0.0),
        robot_end=(2.0, 0.0),
        obstacle_start=(0.0, 0.0),
        obstacle_end=(0.0, 0.0),
    )

    assert distance == pytest.approx(0.0)


def test_parallel_motion_with_positive_clearance():
    from security.physical.geometry import (
        minimum_distance_during_motion_to_square,
    )

    distance = minimum_distance_during_motion_to_square(
        robot_start=(0.0, 2.0),
        robot_end=(4.0, 2.0),
        obstacle_start=(0.0, 0.0),
        obstacle_end=(4.0, 0.0),
    )

    assert distance > 0.0


def test_invalid_square_side_is_rejected():
    from security.physical.geometry import (
        minimum_distance_during_motion_to_square,
    )

    with pytest.raises(ValueError):
        minimum_distance_during_motion_to_square(
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            square_side=0.0,
        )


def test_non_finite_motion_position_is_rejected():
    from security.physical.geometry import (
        minimum_distance_during_motion_to_square,
    )

    with pytest.raises(ValueError):
        minimum_distance_during_motion_to_square(
            (0.0, 0.0),
            (float("inf"), 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        )
