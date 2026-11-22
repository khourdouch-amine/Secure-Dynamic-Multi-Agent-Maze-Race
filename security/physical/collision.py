"""
Continuous collision checking between a robot and a moving square obstacle.

The robot center follows a linear trajectory during the checked interval.
The obstacle follows the piecewise trajectory defined by MovingObstacle.
"""

from __future__ import annotations

from math import isfinite

from environment.moving_obstacle import MovingObstacle
from security.physical.geometry import (
    DEFAULT_ROBOT_OBSTACLE_MARGIN,
    DEFAULT_ROBOT_RADIUS,
    minimum_distance_during_motion_to_square,
)


def minimum_clearance_during_motion_to_obstacle(
    robot_start: tuple[float, float],
    robot_end: tuple[float, float],
    obstacle: MovingObstacle,
    interval_start: float,
    interval_end: float,
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    safety_margin: float = DEFAULT_ROBOT_OBSTACLE_MARGIN,
) -> float:
    """
    Return the minimum robot-obstacle safety clearance over an interval.

    The robot follows a linear trajectory between robot_start and
    robot_end over [interval_start, interval_end].

    The obstacle can be stationary before and after its movement.
    The interval is automatically split at obstacle start_time and
    end_time so that each segment has linear trajectories.

    Positive clearance means the required safety margin is respected.
    Zero means the safety boundary is reached.
    Negative clearance means a safety violation.

    The robot radius and safety margin are subtracted from the minimum
    center-to-square distance.
    """
    if not isinstance(obstacle, MovingObstacle):
        raise TypeError("obstacle must be a MovingObstacle instance.")

    def validate_point(point, name):
        try:
            if len(point) != 2:
                raise ValueError
            normalized = (float(point[0]), float(point[1]))
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError(
                f"{name} must contain exactly two finite coordinates."
            ) from exc

        if not all(isfinite(value) for value in normalized):
            raise ValueError(
                f"{name} coordinates must be finite."
            )

        return normalized

    robot_start = validate_point(robot_start, "robot_start")
    robot_end = validate_point(robot_end, "robot_end")

    try:
        start = float(interval_start)
        end = float(interval_end)
        radius = float(robot_radius)
        margin = float(safety_margin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Time and clearance parameters must be numeric.") from exc

    if not isfinite(start) or not isfinite(end):
        raise ValueError("Interval times must be finite.")

    if end < start:
        raise ValueError("interval_end must be greater than or equal to interval_start.")

    if not isfinite(radius) or radius <= 0.0:
        raise ValueError("robot_radius must be finite and strictly positive.")

    if not isfinite(margin) or margin < 0.0:
        raise ValueError("safety_margin must be finite and non-negative.")

    interval_duration = end - start

    if not isfinite(interval_duration):
        raise ValueError("Interval duration must be finite.")

    # Split the interval at every obstacle trajectory transition.
    breakpoints = [start, end]

    for transition in (obstacle.start_time, obstacle.end_time):
        if start < transition < end:
            breakpoints.append(transition)

    breakpoints = sorted(set(breakpoints))

    def robot_position_at(time):
        if interval_duration == 0.0:
            return robot_start

        progress = (time - start) / interval_duration

        return (
            robot_start[0] + progress * (robot_end[0] - robot_start[0]),
            robot_start[1] + progress * (robot_end[1] - robot_start[1]),
        )

    minimum_distance = float("inf")

    # A zero-duration interval is a static collision check.
    if len(breakpoints) == 1:
        obstacle_position = obstacle.position_at(start)

        minimum_distance = minimum_distance_during_motion_to_square(
            robot_start,
            robot_start,
            obstacle_position,
            obstacle_position,
            obstacle.size,
        )

    else:
        for segment_start, segment_end in zip(
            breakpoints,
            breakpoints[1:],
        ):
            robot_segment_start = robot_position_at(segment_start)
            robot_segment_end = robot_position_at(segment_end)

            obstacle_segment_start = obstacle.position_at(segment_start)
            obstacle_segment_end = obstacle.position_at(segment_end)

            segment_distance = minimum_distance_during_motion_to_square(
                robot_segment_start,
                robot_segment_end,
                obstacle_segment_start,
                obstacle_segment_end,
                obstacle.size,
            )

            minimum_distance = min(
                minimum_distance,
                segment_distance,
            )

    return minimum_distance - radius - margin


def is_safe_during_motion_to_obstacle(
    robot_start: tuple[float, float],
    robot_end: tuple[float, float],
    obstacle: MovingObstacle,
    interval_start: float,
    interval_end: float,
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    safety_margin: float = DEFAULT_ROBOT_OBSTACLE_MARGIN,
) -> bool:
    """Return whether the required clearance is respected throughout."""
    return minimum_clearance_during_motion_to_obstacle(
        robot_start=robot_start,
        robot_end=robot_end,
        obstacle=obstacle,
        interval_start=interval_start,
        interval_end=interval_end,
        robot_radius=robot_radius,
        safety_margin=safety_margin,
    ) >= 0.0
