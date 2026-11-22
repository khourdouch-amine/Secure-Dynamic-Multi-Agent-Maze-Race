"""Reversible geometric phase oracle for global Grover maze search."""

from math import log2

from qiskit import QuantumCircuit, QuantumRegister

from .reversible_coordinates import (
    controlled_increment,
    controlled_decrement,
    apply_reversible_move,
)


def _register_width(size):
    return max(1, size.bit_length())


def _compute_position_flag(qc, x, y, flag, target_x, target_y):
    """Toggle flag iff coordinates equal the specified position."""
    controls = list(x) + list(y)
    state = target_x | (target_y << len(x))
    qc.mcx(controls, flag, ctrl_state=state)


def _compute_invalid_flag(qc, x, y, flag, width, height, obstacles):
    """
    Toggle flag iff the current position is outside or blocked.

    Power-of-two dimensions use their most significant coordinate bits
    to detect out-of-bounds positions directly.
    """
    blocked = set(obstacles)

    power_of_two = (
        width > 0
        and height > 0
        and (width & (width - 1)) == 0
        and (height & (height - 1)) == 0
    )

    if power_of_two:
        # Disjoint reversible predicates:
        # x MSB = 1, or (x MSB = 0 and y MSB = 1).
        qc.cx(x[-1], flag)

        qc.x(x[-1])
        qc.ccx(y[-1], x[-1], flag)
        qc.x(x[-1])

        # Exact obstacle predicates, disjoint from out-of-bounds states.
        for px, py in sorted(blocked):
            _compute_position_flag(qc, x, y, flag, px, py)

        return

    # General fallback for non-power-of-two dimensions.
    for px in range(1 << len(x)):
        for py in range(1 << len(y)):
            if (
                px >= width
                or py >= height
                or (px, py) in blocked
            ):
                _compute_position_flag(qc, x, y, flag, px, py)


def _uncompute_invalid_flag(qc, x, y, flag, width, height, obstacles):
    """Reverse invalid-position computation."""
    _compute_invalid_flag(qc, x, y, flag, width, height, obstacles)


def _apply_inverse_move(qc, direction_bits, x, y):
    """Reverse one controlled cardinal movement."""
    controlled_decrement(qc, x, direction_bits, 0)
    controlled_increment(qc, y, direction_bits, 1)
    controlled_increment(qc, x, direction_bits, 2)
    controlled_decrement(qc, y, direction_bits, 3)


def build_geometric_phase_oracle(
    width,
    height,
    start,
    goal,
    obstacles,
    route_length,
):
    """
    Build a reversible phase oracle for fixed-length maze routes.

    Route encoding:
        00: right
        01: up
        10: left
        11: down

    The phase qubit must be initialized to |-> by the caller.
    All other auxiliary registers are restored to zero.
    """
    if width < 2 or height < 2:
        raise ValueError("Maze dimensions must be at least 2.")

    if route_length < 1:
        raise ValueError("route_length must be positive.")

    start = tuple(start)
    goal = tuple(goal)
    obstacles = frozenset(tuple(p) for p in obstacles)

    for name, position in (("start", start), ("goal", goal)):
        if len(position) != 2:
            raise ValueError(f"{name} must contain two coordinates.")
        x0, y0 = position
        if not (0 <= x0 < width and 0 <= y0 < height):
            raise ValueError(f"{name} is outside the maze.")
        if position in obstacles:
            raise ValueError(f"{name} cannot be an obstacle.")

    if any(
        not (0 <= x < width and 0 <= y < height)
        for x, y in obstacles
    ):
        raise ValueError("An obstacle is outside the maze.")

    nx = _register_width(width)
    ny = _register_width(height)

    route = QuantumRegister(2 * route_length, "route")
    x = QuantumRegister(nx, "x")
    y = QuantumRegister(ny, "y")
    bad = QuantumRegister(route_length, "bad")
    at_goal = QuantumRegister(1, "goal")
    phase = QuantumRegister(1, "phase")

    qc = QuantumCircuit(route, x, y, bad, at_goal, phase,
                        name="GeometricPhaseOracle")

    # Initialize coordinates.
    for bit in range(nx):
        if (start[0] >> bit) & 1:
            qc.x(x[bit])

    for bit in range(ny):
        if (start[1] >> bit) & 1:
            qc.x(y[bit])

    # Forward route evolution and collision detection.
    for step in range(route_length):
        direction = list(route[2 * step: 2 * step + 2])
        apply_reversible_move(qc, direction, list(x), list(y))
        _compute_invalid_flag(
            qc, list(x), list(y), bad[step],
            width, height, obstacles,
        )

    # Compute destination condition.
    _compute_position_flag(
        qc, list(x), list(y), at_goal[0], goal[0], goal[1]
    )

    # Phase flip iff at goal and every collision flag is zero.
    controls = list(bad) + [at_goal[0]]
    goal_state = 1 << route_length
    qc.mcx(controls, phase[0], ctrl_state=goal_state)

    # Uncompute destination and collision flags.
    _compute_position_flag(
        qc, list(x), list(y), at_goal[0], goal[0], goal[1]
    )

    for step in reversed(range(route_length)):
        _uncompute_invalid_flag(
            qc, list(x), list(y), bad[step],
            width, height, obstacles,
        )
        direction = list(route[2 * step: 2 * step + 2])
        _apply_inverse_move(qc, direction, list(x), list(y))

    # Restore coordinate registers to |0...0>.
    for bit in range(nx):
        if (start[0] >> bit) & 1:
            qc.x(x[bit])

    for bit in range(ny):
        if (start[1] >> bit) & 1:
            qc.x(y[bit])

    return qc
