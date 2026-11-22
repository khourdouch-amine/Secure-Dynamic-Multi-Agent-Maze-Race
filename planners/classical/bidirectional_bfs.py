"""
Bidirectional Breadth-First Search for four-connected grid navigation.
"""

from __future__ import annotations


class BidirectionalBFSPlanner:
    """Optimal bidirectional BFS planner for uniform-cost grid mazes."""

    def __init__(self):
        self.name = "Bidirectional BFS"
        self.last_metadata = {}

    @staticmethod
    def _reconstruct_path(meet, forward_parent, backward_parent):
        """Reconstruct the path from start through meet to goal."""
        first_half = []
        node = meet

        while node is not None:
            first_half.append(node)
            node = forward_parent[node]

        first_half.reverse()

        second_half = []
        node = backward_parent[meet]

        while node is not None:
            second_half.append(node)
            node = backward_parent[node]

        return first_half + second_half

    def plan(self, maze, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        self.last_metadata = {
            "algorithm": self.name,
            "expanded_forward": 0,
            "expanded_backward": 0,
            "meeting_point": None,
        }

        if not maze.is_walkable(*start):
            return None

        if not maze.is_walkable(*goal):
            return None

        if start == goal:
            return [start]

        forward_parent = {start: None}
        backward_parent = {goal: None}

        forward_frontier = [start]
        backward_frontier = [goal]

        while forward_frontier and backward_frontier:
            # Expand the smaller frontier, one complete BFS layer at a time.
            expand_forward = (
                len(forward_frontier) <= len(backward_frontier)
            )

            if expand_forward:
                current_frontier = forward_frontier
                next_frontier = []

                for current in current_frontier:
                    self.last_metadata["expanded_forward"] += 1

                    for neighbor in maze.get_neighbors(*current):
                        if neighbor in forward_parent:
                            continue

                        forward_parent[neighbor] = current
                        next_frontier.append(neighbor)

                        if neighbor in backward_parent:
                            self.last_metadata["meeting_point"] = neighbor
                            return self._reconstruct_path(
                                neighbor,
                                forward_parent,
                                backward_parent,
                            )

                forward_frontier = next_frontier

            else:
                current_frontier = backward_frontier
                next_frontier = []

                for current in current_frontier:
                    self.last_metadata["expanded_backward"] += 1

                    for neighbor in maze.get_neighbors(*current):
                        if neighbor in backward_parent:
                            continue

                        backward_parent[neighbor] = current
                        next_frontier.append(neighbor)

                        if neighbor in forward_parent:
                            self.last_metadata["meeting_point"] = neighbor
                            return self._reconstruct_path(
                                neighbor,
                                forward_parent,
                                backward_parent,
                            )

                backward_frontier = next_frontier

        return None
