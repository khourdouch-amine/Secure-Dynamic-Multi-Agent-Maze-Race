"""Encoding and decoding of fixed-length cardinal movement sequences."""

class RouteEncoding:
    """
    Encode each movement with two bits:

        00 -> right
        01 -> up
        10 -> left
        11 -> down

    The first movement occupies the two least significant bits.
    """

    DIRECTIONS = {
        0: (1, 0),
        1: (0, -1),
        2: (-1, 0),
        3: (0, 1),
    }

    @staticmethod
    def encode(sequence):
        """Convert a movement sequence into an integer basis-state index."""
        if not sequence:
            raise ValueError("The route sequence cannot be empty.")

        index = 0

        for step, direction in enumerate(sequence):
            if isinstance(direction, bool) or not isinstance(direction, int):
                raise TypeError("Each direction must be an integer from 0 to 3.")

            if direction not in RouteEncoding.DIRECTIONS:
                raise ValueError("Each direction must be between 0 and 3.")

            index |= direction << (2 * step)

        return index

    @staticmethod
    def decode(index, route_length):
        """Convert an integer basis-state index into a movement sequence."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Index must be an integer.")

        if isinstance(route_length, bool) or not isinstance(route_length, int):
            raise TypeError("Route length must be an integer.")

        if route_length < 1:
            raise ValueError("Route length must be positive.")

        if index < 0 or index >= 4 ** route_length:
            raise ValueError("Index is outside the encoded search space.")

        return tuple(
            (index >> (2 * step)) & 0b11
            for step in range(route_length)
        )

    @staticmethod
    def decode_bitstring(bitstring, route_length):
        """Decode a Qiskit measurement bitstring into movement directions."""
        if not isinstance(bitstring, str):
            raise TypeError("Bitstring must be a string.")

        normalized = bitstring.replace(" ", "")

        if len(normalized) != 2 * route_length:
            raise ValueError("Bitstring length does not match route length.")

        if any(bit not in "01" for bit in normalized):
            raise ValueError("Bitstring must contain only 0 and 1.")

        return RouteEncoding.decode(int(normalized, 2), route_length)

    @staticmethod
    def sequence_to_path(start, sequence):
        """Convert a movement sequence into its corresponding coordinate path."""
        if (
            not isinstance(start, (tuple, list))
            or len(start) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in start)
        ):
            raise ValueError("Start must contain two integer coordinates.")

        x, y = start
        path = [(x, y)]

        for direction in sequence:
            if direction not in RouteEncoding.DIRECTIONS:
                raise ValueError("Invalid movement direction.")

            dx, dy = RouteEncoding.DIRECTIONS[direction]
            x += dx
            y += dy
            path.append((x, y))

        return path
