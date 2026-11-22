"""
Continuous-time model of a square moving obstacle.

The obstacle follows a linear trajectory between two world positions.
Its position is expressed in meters and time in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MovingObstacle:
    """Square obstacle moving linearly between two positions."""

    start_position: tuple[float, float]
    target_position: tuple[float, float]
    start_time: float
    duration: float
    size: float = 1.0

    def __post_init__(self):
        """Validate and normalize the trajectory parameters."""
        for name in ("start_position", "target_position"):
            point = getattr(self, name)

            try:
                if len(point) != 2:
                    raise ValueError(
                        f"{name} must contain exactly two coordinates."
                    )

                normalized = tuple(float(value) for value in point)

            except (TypeError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must contain two finite coordinates."
                ) from exc

            if not all(isfinite(value) for value in normalized):
                raise ValueError(
                    f"{name} coordinates must be finite."
                )

            object.__setattr__(self, name, normalized)

        for name in ("start_time", "duration", "size"):
            try:
                value = float(getattr(self, name))
            except (TypeError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite number.") from exc

            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")

            object.__setattr__(self, name, value)

        if self.duration <= 0.0:
            raise ValueError("duration must be strictly positive.")

        if self.size <= 0.0:
            raise ValueError("size must be strictly positive.")

        if not isfinite(self.start_time + self.duration):
            raise ValueError("end_time must be finite.")

        dx = self.target_position[0] - self.start_position[0]
        dy = self.target_position[1] - self.start_position[1]

        if not isfinite(dx) or not isfinite(dy):
            raise ValueError("Trajectory displacement must be finite.")

        if not isfinite(dx / self.duration) or not isfinite(dy / self.duration):
            raise ValueError("Trajectory velocity must be finite.")

    @property
    def end_time(self) -> float:
        """Return the time at which the movement ends."""
        return self.start_time + self.duration

    @property
    def velocity(self) -> tuple[float, float]:
        """Return the constant velocity during the movement."""
        dx = self.target_position[0] - self.start_position[0]
        dy = self.target_position[1] - self.start_position[1]

        return (
            dx / self.duration,
            dy / self.duration,
        )

    def _validate_time(self, simulation_time: float) -> float:
        """Validate a simulation timestamp."""
        try:
            value = float(simulation_time)
        except (TypeError, OverflowError) as exc:
            raise ValueError(
                "simulation_time must be a finite number."
            ) from exc

        if not isfinite(value):
            raise ValueError("simulation_time must be finite.")

        return value

    def position_at(
        self,
        simulation_time: float,
    ) -> tuple[float, float]:
        """
        Return the obstacle center at a given simulation time.

        Before the movement, the obstacle remains at its start position.
        After the movement, it remains at its target position.
        """
        time = self._validate_time(simulation_time)

        if time <= self.start_time:
            return self.start_position

        if time >= self.end_time:
            return self.target_position

        progress = (time - self.start_time) / self.duration

        x0, y0 = self.start_position
        x1, y1 = self.target_position

        return (
            x0 + progress * (x1 - x0),
            y0 + progress * (y1 - y0),
        )

    def velocity_at(
        self,
        simulation_time: float,
    ) -> tuple[float, float]:
        """
        Return the obstacle velocity at a given simulation time.

        Velocity is zero outside the movement interval.
        The movement interval is [start_time, end_time).
        """
        time = self._validate_time(simulation_time)

        if time < self.start_time or time >= self.end_time:
            return (0.0, 0.0)

        return self.velocity

    def is_moving(self, simulation_time: float) -> bool:
        """Return whether the obstacle is currently moving."""
        time = self._validate_time(simulation_time)

        return self.start_time <= time < self.end_time
