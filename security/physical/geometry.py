"""
Physical geometry utilities for the multi-agent maze simulation.

Coordinate convention
---------------------
The center of grid cell (x, y) is located at world position (x, y),
expressed in meters.

Grid cells are axis-aligned squares.
Robots are represented by circular footprints.
"""

from __future__ import annotations

from math import floor, hypot, isfinite
from typing import TypeAlias


Point2D: TypeAlias = tuple[float, float]
GridCell: TypeAlias = tuple[int, int]

DEFAULT_CELL_SIZE = 1.0
DEFAULT_ROBOT_RADIUS = 0.25
DEFAULT_ROBOT_ROBOT_DISTANCE = 1.0
DEFAULT_ROBOT_OBSTACLE_MARGIN = 0.10


def _validate_point(point: Point2D, name: str = "point") -> None:
    if len(point) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates.")

    if not all(isfinite(float(value)) for value in point):
        raise ValueError(f"{name} coordinates must be finite.")


def _validate_positive(value: float, name: str) -> None:
    if not isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")


def grid_to_world(
    cell: GridCell,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> Point2D:
    """Convert a grid-cell center into world coordinates."""
    _validate_positive(cell_size, "cell_size")

    if len(cell) != 2:
        raise ValueError("cell must contain exactly two coordinates.")

    x, y = cell

    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("Grid coordinates must be integers.")

    return float(x * cell_size), float(y * cell_size)


def world_to_grid(
    position: Point2D,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> GridCell:
    """
    Map a world position to its nearest grid-cell center.

    Exact half-cell ties are assigned to the cell with the
    greater coordinate.
    """
    _validate_point(position, "position")
    _validate_positive(cell_size, "cell_size")

    x, y = position

    return (
        floor(x / cell_size + 0.5),
        floor(y / cell_size + 0.5),
    )


def distance_between_points(
    point_a: Point2D,
    point_b: Point2D,
) -> float:
    """Return the Euclidean distance between two points."""
    _validate_point(point_a, "point_a")
    _validate_point(point_b, "point_b")

    return hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )


def robot_robot_clearance(
    position_a: Point2D,
    position_b: Point2D,
    radius_a: float = DEFAULT_ROBOT_RADIUS,
    radius_b: float = DEFAULT_ROBOT_RADIUS,
) -> float:
    """
    Return the signed surface clearance between two circular robots.

    Positive: separated.
    Zero: exactly touching.
    Negative: geometric overlap.
    """
    _validate_positive(radius_a, "radius_a")
    _validate_positive(radius_b, "radius_b")

    return (
        distance_between_points(position_a, position_b)
        - radius_a
        - radius_b
    )


def robots_are_safe(
    position_a: Point2D,
    position_b: Point2D,
    min_center_distance: float = DEFAULT_ROBOT_ROBOT_DISTANCE,
) -> bool:
    """Check the required minimum center-to-center distance."""
    _validate_positive(min_center_distance, "min_center_distance")

    return (
        distance_between_points(position_a, position_b)
        >= min_center_distance
    )


def point_to_square_distance(
    point: Point2D,
    square_center: Point2D,
    side_length: float = DEFAULT_CELL_SIZE,
) -> float:
    """Return the shortest distance from a point to an axis-aligned square."""
    _validate_point(point, "point")
    _validate_point(square_center, "square_center")
    _validate_positive(side_length, "side_length")

    half = side_length / 2.0

    min_x = square_center[0] - half
    max_x = square_center[0] + half
    min_y = square_center[1] - half
    max_y = square_center[1] + half

    dx = max(min_x - point[0], 0.0, point[0] - max_x)
    dy = max(min_y - point[1], 0.0, point[1] - max_y)

    return hypot(dx, dy)


def point_to_square_signed_distance(
    point: Point2D,
    square_center: Point2D,
    side_length: float = DEFAULT_CELL_SIZE,
) -> float:
    """
    Return the signed distance from a point to an axis-aligned square.

    Positive: point outside the square.
    Zero: point on the square boundary.
    Negative: point inside the square.
    """
    _validate_point(point, "point")
    _validate_point(square_center, "square_center")
    _validate_positive(side_length, "side_length")

    half = side_length / 2.0

    qx = abs(point[0] - square_center[0]) - half
    qy = abs(point[1] - square_center[1]) - half

    outside_distance = hypot(max(qx, 0.0), max(qy, 0.0))
    inside_distance = min(max(qx, qy), 0.0)

    return outside_distance + inside_distance


def robot_square_clearance(
    robot_position: Point2D,
    square_center: Point2D,
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    square_side: float = DEFAULT_CELL_SIZE,
) -> float:
    """
    Return signed clearance between a circular robot and a square.

    Positive: separated.
    Zero: exactly touching.
    Negative: overlap.
    """
    _validate_positive(robot_radius, "robot_radius")

    return (
        point_to_square_signed_distance(
            robot_position,
            square_center,
            square_side,
        )
        - robot_radius
    )


def robot_is_safe_from_square(
    robot_position: Point2D,
    square_center: Point2D,
    margin: float = DEFAULT_ROBOT_OBSTACLE_MARGIN,
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    square_side: float = DEFAULT_CELL_SIZE,
) -> bool:
    """Check whether a robot respects the required square clearance."""
    if not isfinite(float(margin)) or margin < 0:
        raise ValueError("margin must be finite and non-negative.")

    clearance = robot_square_clearance(
        robot_position,
        square_center,
        robot_radius,
        square_side,
    )

    return clearance >= margin


def minimum_distance_during_motion(
    start_a: Point2D,
    end_a: Point2D,
    start_b: Point2D,
    end_b: Point2D,
) -> float:
    """
    Return the minimum distance between two synchronized linear motions.

    Both robots move linearly from their start to end positions during
    the same normalized time interval t in [0, 1], at constant velocity.

    This checks the complete interval, not only its endpoints.
    """
    _validate_point(start_a, "start_a")
    _validate_point(end_a, "end_a")
    _validate_point(start_b, "start_b")
    _validate_point(end_b, "end_b")

    relative_x = start_a[0] - start_b[0]
    relative_y = start_a[1] - start_b[1]

    relative_velocity_x = (
        (end_a[0] - start_a[0])
        - (end_b[0] - start_b[0])
    )
    relative_velocity_y = (
        (end_a[1] - start_a[1])
        - (end_b[1] - start_b[1])
    )

    velocity_squared = (
        relative_velocity_x ** 2
        + relative_velocity_y ** 2
    )

    if velocity_squared == 0.0:
        return hypot(relative_x, relative_y)

    closest_time = -(
        relative_x * relative_velocity_x
        + relative_y * relative_velocity_y
    ) / velocity_squared

    closest_time = max(0.0, min(1.0, closest_time))

    closest_x = relative_x + closest_time * relative_velocity_x
    closest_y = relative_y + closest_time * relative_velocity_y

    return hypot(closest_x, closest_y)


def robots_are_safe_during_motion(
    start_a: Point2D,
    end_a: Point2D,
    start_b: Point2D,
    end_b: Point2D,
    min_center_distance: float = DEFAULT_ROBOT_ROBOT_DISTANCE,
) -> bool:
    """Check the minimum center distance throughout both motions."""
    _validate_positive(min_center_distance, "min_center_distance")

    return (
        minimum_distance_during_motion(
            start_a,
            end_a,
            start_b,
            end_b,
        )
        >= min_center_distance
    )


def point_to_square_distance_gradient(
    point: Point2D,
    square_center: Point2D,
    side_length: float = DEFAULT_CELL_SIZE,
    tolerance: float = 1e-10,
) -> tuple[float, Point2D]:
    """
    Return distance and gradient outside an axis-aligned square.

    The gradient is defined only in the exterior region. Points on
    non-differentiable boundaries are rejected explicitly.
    """
    _validate_point(point, "point")
    _validate_point(square_center, "square_center")
    _validate_positive(side_length, "side_length")

    if not isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError("tolerance must be finite and strictly positive.")

    dx = point[0] - square_center[0]
    dy = point[1] - square_center[1]
    half = side_length / 2.0

    qx = abs(dx) - half
    qy = abs(dy) - half

    if qx <= 0.0 and qy <= 0.0:
        raise ValueError("Distance gradient is undefined inside the square.")

    distance = hypot(max(qx, 0.0), max(qy, 0.0))

    if distance <= tolerance:
        raise ValueError("Distance gradient is numerically undefined.")

    sign_x = 1.0 if dx > 0.0 else -1.0
    sign_y = 1.0 if dy > 0.0 else -1.0

    gradient_x = max(qx, 0.0) * sign_x / distance
    gradient_y = max(qy, 0.0) * sign_y / distance

    return distance, (gradient_x, gradient_y)

def minimum_distance_during_motion_to_square(
    robot_start: Point2D,
    robot_end: Point2D,
    obstacle_start: Point2D,
    obstacle_end: Point2D,
    square_side: float = DEFAULT_CELL_SIZE,
) -> float:
    """
    Return the exact minimum distance from a moving point to a
    translating axis-aligned square over a synchronized interval.

    The robot center and square center both follow linear trajectories
    over the same normalized time interval t in [0, 1].

    The square does not rotate. The calculation is performed in the
    obstacle-relative frame and minimizes the squared point-to-square
    distance analytically on each piecewise-quadratic interval.

    The returned distance is measured from the robot center to the
    square boundary. Robot radius and safety margin are not included.
    """
    _validate_point(robot_start, "robot_start")
    _validate_point(robot_end, "robot_end")
    _validate_point(obstacle_start, "obstacle_start")
    _validate_point(obstacle_end, "obstacle_end")
    _validate_positive(square_side, "square_side")

    half = square_side / 2.0

    relative_start = (
        robot_start[0] - obstacle_start[0],
        robot_start[1] - obstacle_start[1],
    )
    relative_end = (
        robot_end[0] - obstacle_end[0],
        robot_end[1] - obstacle_end[1],
    )

    velocity = (
        relative_end[0] - relative_start[0],
        relative_end[1] - relative_start[1],
    )

    # The distance formula changes its quadratic expression only when
    # a relative coordinate crosses one of the square's four faces.
    breakpoints = [0.0, 1.0]

    for axis in range(2):
        speed = velocity[axis]

        if speed == 0.0:
            continue

        for boundary in (-half, half):
            crossing_time = (
                boundary - relative_start[axis]
            ) / speed

            if 0.0 < crossing_time < 1.0:
                breakpoints.append(crossing_time)

    breakpoints = sorted(set(breakpoints))
    candidates = list(breakpoints)

    # On each interval, the active outside-square coordinates are fixed.
    # Their squared distance is a convex quadratic function of time.
    for left, right in zip(breakpoints, breakpoints[1:]):
        midpoint = 0.5 * (left + right)

        active_axes = []

        for axis in range(2):
            coordinate = (
                relative_start[axis] + midpoint * velocity[axis]
            )

            if coordinate > half:
                offset = -half
            elif coordinate < -half:
                offset = half
            else:
                continue

            residual_start = relative_start[axis] + offset
            active_axes.append((residual_start, velocity[axis]))

        denominator = sum(
            speed * speed
            for _, speed in active_axes
        )

        if denominator == 0.0:
            continue

        numerator = sum(
            residual * speed
            for residual, speed in active_axes
        )

        stationary_time = -numerator / denominator

        if left < stationary_time < right:
            candidates.append(stationary_time)

    minimum_squared_distance = float("inf")

    for time in candidates:
        x = relative_start[0] + time * velocity[0]
        y = relative_start[1] + time * velocity[1]

        dx = max(abs(x) - half, 0.0)
        dy = max(abs(y) - half, 0.0)

        squared_distance = dx * dx + dy * dy

        minimum_squared_distance = min(
            minimum_squared_distance,
            squared_distance,
        )

    return float(minimum_squared_distance ** 0.5)

