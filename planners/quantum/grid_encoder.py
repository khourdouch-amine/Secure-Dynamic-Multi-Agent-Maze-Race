"""
Graph encoding of a 2D maze for quantum navigation algorithms.
"""

from __future__ import annotations


class GridGraphEncoder:
    """Map walkable maze cells to graph-state indices."""

    def __init__(self, maze):
        self.width = maze.width
        self.height = maze.height

        self.cells = [
            (x, y)
            for y in range(maze.height)
            for x in range(maze.width)
            if maze.is_walkable(x, y)
        ]

        self.cell_to_index = {
            cell: index
            for index, cell in enumerate(self.cells)
        }

        self.adjacency = {
            index: [
                self.cell_to_index[neighbor]
                for neighbor in maze.get_neighbors(*cell)
                if neighbor in self.cell_to_index
            ]
            for index, cell in enumerate(self.cells)
        }

        self.number_of_states = len(self.cells)

    def encode_state(self, cell):
        """Convert a grid coordinate into a graph-state index."""
        cell = tuple(cell)

        if cell not in self.cell_to_index:
            raise ValueError(f"Cell is not encoded or not walkable: {cell}")

        return self.cell_to_index[cell]

    def decode_state(self, state):
        """Convert a graph-state index into a grid coordinate."""
        if not isinstance(state, int):
            raise TypeError("State index must be an integer.")

        if not 0 <= state < self.number_of_states:
            raise ValueError(f"Invalid state index: {state}")

        return self.cells[state]

    def neighbors(self, state):
        """Return neighboring graph-state indices."""
        if not 0 <= state < self.number_of_states:
            raise ValueError(f"Invalid state index: {state}")

        return self.adjacency[state]

    def get_edge_count(self):
        """Return the number of undirected graph edges."""
        return sum(
            len(neighbors)
            for neighbors in self.adjacency.values()
        ) // 2

    def get_qubit_requirement(self):
        """Return the number of qubits needed to index all states."""
        if self.number_of_states <= 1:
            return 1

        return (self.number_of_states - 1).bit_length()
