"""
Control Barrier Function safety filter for a differential-drive robot.

Scope:
- Static axis-aligned square obstacles.
- Circular robot footprint.
- Linear and angular velocity control.
- One-step CBF-QP solved with OSQP.

This module does not handle robot-robot interactions or moving obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Iterable

import numpy as np
import osqp
from scipy import sparse

from security.physical.geometry import (
    DEFAULT_CELL_SIZE,
    DEFAULT_ROBOT_OBSTACLE_MARGIN,
    DEFAULT_ROBOT_RADIUS,
    Point2D,
    point_to_square_distance_gradient,
    robot_square_clearance,
)


@dataclass(frozen=True)
class CBFResult:
    """Diagnostic returned by the CBF safety filter."""

    control: tuple[float, float]
    status: str
    success: bool
    nominal_control: tuple[float, float]
    minimum_clearance: float | None
    active_constraints: int
    message: str


class CBFController:
    """
    One-step CBF-QP safety filter for a differential-drive robot.

    The optimization variables are:
        u = [v, omega]

    The objective minimizes the weighted deviation from the nominal
    controller command while enforcing static-obstacle CBF constraints.
    """

    def __init__(
        self,
        robot_radius: float = DEFAULT_ROBOT_RADIUS,
        obstacle_margin: float = DEFAULT_ROBOT_OBSTACLE_MARGIN,
        obstacle_size: float = DEFAULT_CELL_SIZE,
        max_linear_speed: float = 2.0,
        max_angular_speed: float = 2.0,
        alpha: float = 2.0,
        linear_weight: float = 1.0,
        angular_weight: float = 1.0,
    ) -> None:
        positive_parameters = {
            "robot_radius": robot_radius,
            "obstacle_size": obstacle_size,
            "max_linear_speed": max_linear_speed,
            "max_angular_speed": max_angular_speed,
            "alpha": alpha,
            "linear_weight": linear_weight,
            "angular_weight": angular_weight,
        }

        for name, value in positive_parameters.items():
            if not isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        if not isfinite(float(obstacle_margin)) or obstacle_margin < 0:
            raise ValueError("obstacle_margin must be finite and non-negative.")

        self.robot_radius = float(robot_radius)
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_size = float(obstacle_size)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.alpha = float(alpha)

        self.weights = np.array(
            [linear_weight, angular_weight],
            dtype=float,
        )

    @staticmethod
    def _validate_control(control: tuple[float, float]) -> tuple[float, float]:
        if len(control) != 2:
            raise ValueError("control must contain (v, omega).")

        values = tuple(float(value) for value in control)

        if not all(isfinite(value) for value in values):
            raise ValueError("Control values must be finite.")

        return values

    @staticmethod
    def _validate_pose(
        position: Point2D,
        orientation: float,
    ) -> tuple[float, float, float]:
        if len(position) != 2:
            raise ValueError("position must contain exactly two coordinates.")

        x, y = float(position[0]), float(position[1])
        theta = float(orientation)

        if not all(isfinite(value) for value in (x, y, theta)):
            raise ValueError("Robot pose must contain finite values.")

        return x, y, theta

    def _failure(
        self,
        status: str,
        nominal_control: tuple[float, float],
        minimum_clearance: float | None,
        message: str,
    ) -> CBFResult:
        """
        Return a zero-command fallback with an explicit failure status.

        A zero command is not, by itself, a physical safety guarantee.
        """
        return CBFResult(
            control=(0.0, 0.0),
            status=status,
            success=False,
            nominal_control=nominal_control,
            minimum_clearance=minimum_clearance,
            active_constraints=0,
            message=message,
        )

    def filter(
        self,
        position: Point2D,
        orientation: float,
        nominal_control: tuple[float, float],
        obstacles: Iterable[Point2D] = (),
    ) -> CBFResult:
        """
        Filter a nominal (v, omega) command against static square obstacles.

        Each obstacle is represented by its center in world coordinates.
        """
        nominal = self._validate_control(nominal_control)
        x, y, theta = self._validate_pose(position, orientation)

        obstacle_list = list(obstacles)
        robot_position = (x, y)

        clearances: list[float] = []
        cbf_rows: list[list[float]] = []
        lower_bounds: list[float] = []

        heading = np.array([cos(theta), sin(theta)], dtype=float)

        for obstacle in obstacle_list:
            if len(obstacle) != 2:
                raise ValueError("Each obstacle must contain two coordinates.")

            obstacle_center = (
                float(obstacle[0]),
                float(obstacle[1]),
            )

            if not all(isfinite(value) for value in obstacle_center):
                raise ValueError("Obstacle coordinates must be finite.")

            clearance = robot_square_clearance(
                robot_position,
                obstacle_center,
                robot_radius=self.robot_radius,
                square_side=self.obstacle_size,
            ) - self.obstacle_margin

            clearances.append(clearance)

            if clearance < -1e-9:
                return self._failure(
                    status="initial_state_unsafe",
                    nominal_control=nominal,
                    minimum_clearance=min(clearances),
                    message=(
                        "The initial robot position violates the required "
                        "obstacle clearance. Forward safety is not guaranteed."
                    ),
                )

            try:
                distance, gradient = point_to_square_distance_gradient(
                    robot_position,
                    obstacle_center,
                    side_length=self.obstacle_size,
                )
            except ValueError:
                return self._failure(
                    status="invalid_geometry",
                    nominal_control=nominal,
                    minimum_clearance=min(clearances),
                    message=(
                        "The robot center is inside or on an obstacle square; "
                        "the exterior distance gradient is undefined."
                    ),
                )

            h = distance - self.robot_radius - self.obstacle_margin
            gradient_vector = np.asarray(gradient, dtype=float)

            coefficient_v = float(gradient_vector @ heading)

            # CBF condition:
            # coefficient_v * v + alpha * h >= 0
            cbf_rows.append([coefficient_v, 0.0])
            lower_bounds.append(-self.alpha * h)

        minimum_clearance = min(clearances) if clearances else None

        # OSQP objective: 1/2 * u.T @ P @ u + q.T @ u
        # Equivalent, up to a constant, to:
        # 1/2 * (u - u_ref).T @ W @ (u - u_ref)
        P = sparse.diags(self.weights, format="csc")
        q = -self.weights * np.asarray(nominal, dtype=float)

        identity = sparse.eye(2, format="csc")
        if cbf_rows:
            safety_matrix = sparse.csc_matrix(np.asarray(cbf_rows))
            A = sparse.vstack(
                [safety_matrix, identity],
                format="csc",
            )
            lower = np.concatenate([
                np.asarray(lower_bounds, dtype=float),
                np.array([
                    -self.max_linear_speed,
                    -self.max_angular_speed,
                ]),
            ])
            upper = np.concatenate([
                np.full(len(cbf_rows), np.inf),
                np.array([
                    self.max_linear_speed,
                    self.max_angular_speed,
                ]),
            ])
        else:
            A = identity
            lower = np.array([
                -self.max_linear_speed,
                -self.max_angular_speed,
            ])
            upper = np.array([
                self.max_linear_speed,
                self.max_angular_speed,
            ])

        solver = osqp.OSQP()

        try:
            solver.setup(
                P=P,
                q=q,
                A=A,
                l=lower,
                u=upper,
                verbose=False,
                polishing=True,
                eps_abs=1e-8,
                eps_rel=1e-8,
                max_iter=20000,
            )
            solution = solver.solve(raise_error=False)
        except Exception as exc:
            return self._failure(
                status="solver_error",
                nominal_control=nominal,
                minimum_clearance=minimum_clearance,
                message=str(exc),
            )

        status = str(solution.info.status).lower()

        if status not in {"solved", "solved inaccurate"} or solution.x is None:
            return self._failure(
                status=status,
                nominal_control=nominal,
                minimum_clearance=minimum_clearance,
                message="OSQP did not find a feasible solution.",
            )

        control_array = np.asarray(solution.x, dtype=float)

        if not np.all(np.isfinite(control_array)):
            return self._failure(
                status="invalid_solution",
                nominal_control=nominal,
                minimum_clearance=minimum_clearance,
                message="OSQP returned non-finite control values.",
            )

        constraint_values = np.asarray(A @ control_array).reshape(-1)

        lower_violation = np.maximum(lower - constraint_values, 0.0)
        upper_violation = np.maximum(constraint_values - upper, 0.0)
        max_violation = float(
            max(
                np.max(lower_violation, initial=0.0),
                np.max(upper_violation, initial=0.0),
            )
        )

        if not isfinite(max_violation) or max_violation > 1e-5:
            return self._failure(
                status="constraint_violation",
                nominal_control=nominal,
                minimum_clearance=minimum_clearance,
                message=(
                    "The returned solution does not satisfy all QP "
                    f"constraints within tolerance: {max_violation:.3e}."
                ),
            )

        control = (
            float(control_array[0]),
            float(control_array[1]),
        )

        active_constraints = 0
        for row, lower_bound in zip(cbf_rows, lower_bounds):
            residual = (
                row[0] * control[0]
                + row[1] * control[1]
                - lower_bound
            )
            if abs(residual) <= 1e-5:
                active_constraints += 1

        return CBFResult(
            control=control,
            status=status,
            success=True,
            nominal_control=nominal,
            minimum_clearance=minimum_clearance,
            active_constraints=active_constraints,
            message="A feasible CBF-QP solution was found.",
        )
