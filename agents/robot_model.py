"""
Kinematic model of a differential-drive mobile robot.
"""

from __future__ import annotations

from math import isfinite

import numpy as np


class DifferentialDriveRobot:
    """Planar robot with linear and angular velocity controls."""

    def __init__(
        self,
        x=1.0,
        y=1.0,
        theta=0.0,
        max_linear_speed=2.0,
        max_angular_speed=2.0,
    ):
        try:
            x = float(x)
            y = float(y)
            theta = float(theta)
            max_linear_speed = float(max_linear_speed)
            max_angular_speed = float(max_angular_speed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Robot parameters must be valid numbers."
            ) from exc

        if not all(isfinite(value) for value in (x, y, theta)):
            raise ValueError("Initial robot state must be finite.")

        if not isfinite(max_linear_speed) or max_linear_speed <= 0:
            raise ValueError(
                "max_linear_speed must be finite and positive."
            )

        if not isfinite(max_angular_speed) or max_angular_speed <= 0:
            raise ValueError(
                "max_angular_speed must be finite and positive."
            )

        self.state = np.array([x, y, theta], dtype=float)

        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

        self.radius = 0.25

    def dynamics(self, state, control):
        """Return the differential-drive state derivative."""
        x, y, theta = state
        v, omega = control

        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega,
        ])

    def step(self, control, dt=0.1):
        """
        Advance the robot state using explicit Euler integration.

        Invalid inputs are rejected before modifying the robot state.
        """
        try:
            if len(control) != 2:
                raise ValueError(
                    "control must contain exactly (v, omega)."
                )

            v, omega = (float(value) for value in control)
            dt = float(dt)

        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Control and time step must be valid numeric values."
            ) from exc

        if not all(isfinite(value) for value in (v, omega)):
            raise ValueError("Control values must be finite.")

        if not isfinite(dt) or dt < 0:
            raise ValueError(
                "dt must be finite and non-negative."
            )

        v = float(np.clip(
            v,
            -self.max_linear_speed,
            self.max_linear_speed,
        ))

        omega = float(np.clip(
            omega,
            -self.max_angular_speed,
            self.max_angular_speed,
        ))

        if dt == 0.0:
            return self.state.copy()

        derivative = self.dynamics(self.state, (v, omega))

        next_state = self.state + derivative * dt

        next_state[2] = (
            next_state[2] + np.pi
        ) % (2.0 * np.pi) - np.pi

        if not np.all(np.isfinite(next_state)):
            raise ValueError(
                "The integration produced a non-finite robot state."
            )

        self.state = next_state

        return self.state.copy()

    @property
    def position(self):
        """Return the robot position in world coordinates."""
        return self.state[:2].copy()

    @property
    def orientation(self):
        """Return the robot orientation in radians."""
        return float(self.state[2])
