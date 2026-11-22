"""Reversible coordinate arithmetic for Grover route search."""


def controlled_increment(qc, register, controls, ctrl_state):
    """Conditionally increment a little-endian quantum register."""
    for bit in reversed(range(len(register))):
        lower = register[:bit]
        combined = list(controls) + list(lower)

        lower_state = (1 << len(lower)) - 1
        state = ctrl_state | (lower_state << len(controls))

        qc.mcx(combined, register[bit], ctrl_state=state)


def controlled_decrement(qc, register, controls, ctrl_state):
    """Conditionally decrement a little-endian quantum register."""
    for bit in reversed(range(len(register))):
        lower = register[:bit]
        combined = list(controls) + list(lower)

        state = ctrl_state
        qc.mcx(combined, register[bit], ctrl_state=state)


def apply_reversible_move(qc, direction_bits, x_register, y_register):
    """
    Apply one reversible cardinal movement.

    Direction encoding:
        00: right
        01: up
        10: left
        11: down
    """
    controlled_increment(qc, x_register, direction_bits, 0)
    controlled_decrement(qc, y_register, direction_bits, 1)
    controlled_decrement(qc, x_register, direction_bits, 2)
    controlled_increment(qc, y_register, direction_bits, 3)
