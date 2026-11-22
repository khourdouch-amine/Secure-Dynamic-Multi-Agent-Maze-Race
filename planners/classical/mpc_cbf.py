"""
Local safety controller using a CBF-constrained quadratic program.

The controller selects linear and angular velocities that follow
a nominal navigation command while respecting local safety constraints.
"""

from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse


class MPCCBFController:
    """CBF-QP safety filter for a differential-drive robot."""

    def __init__(
        self,
        safety_distance=0.8,
        alpha=2.0,
        max_linear_speed=2.0,
        max_angular_speed=2.0,
    ):
        self.name = "MPC + CBF-QP"

        self.safety_distance = float(safety_distance)
        self.alpha = float(alpha)

        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)

    def compute_control(
        self,
        position,
        orientation,
        nominal_control,
        obstacles,
    ):
        """
        Compute a safe control.

        position: (x, y)
        orientation: robot heading in radians
        nominal_control: (v, omega)
        obstacles: iterable of (x, y) obstacle positions
        """

        x, y = map(float, position)
        theta = float(orientation)

        v_ref, omega_ref = map(float, nominal_control)

        # Objective: remain close to the nominal command.
        P = sparse.diags([2.0, 2.0], format="csc")
        q = np.array([-2.0 * v_ref, -2.0 * omega_ref])

        # Control limits.
        A_rows = [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]

        lower = [
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ]

        upper = [
            self.max_linear_speed,
            self.max_linear_speed,
            self.max_angular_speed,
            self.max_angular_speed,
        ]

        # CBF constraints:
        # 2*(p-po).dot(v*[cos(theta), sin(theta)])
        # + alpha*(||p-po||^2 - d_safe^2) >= 0
        for obstacle in obstacles:
            ox, oy = map(float, obstacle)

            dx = x - ox
            dy = y - oy

            h = dx**2 + dy**2 - self.safety_distance**2

            a_v = 2.0 * (
                dx * np.cos(theta)
                + dy * np.sin(theta)
            )

            A_rows.append([a_v, 0.0])
            lower.append(-self.alpha * h)
            upper.append(np.inf)

        A = sparse.csc_matrix(np.asarray(A_rows, dtype=float))

        solver = osqp.OSQP()

        solver.setup(
            P=P,
            q=q,
            A=A,
            l=np.asarray(lower),
            u=np.asarray(upper),
            verbose=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=10000,
            polishing=True,
        )

        result = solver.solve()

        if result.info.status.lower() not in ("solved", "solved inaccurate"):
            return np.zeros(2), result.info.status

        return np.asarray(result.x), result.info.status
