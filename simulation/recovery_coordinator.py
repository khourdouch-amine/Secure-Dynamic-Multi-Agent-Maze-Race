"""
Persistent blockage detection for multi-agent recovery.

This module only observes agent progress. It never moves agents,
changes their goals, or bypasses the physical safety manager.
"""

from __future__ import annotations

from math import isfinite


class RecoveryCoordinator:
    """Detect agents that remain blocked for consecutive simulation steps."""

    def __init__(self, blocked_threshold: int = 5) -> None:
        if (
            isinstance(blocked_threshold, bool)
            or not isinstance(blocked_threshold, int)
            or blocked_threshold < 1
        ):
            raise ValueError(
                "blocked_threshold must be a positive integer."
            )

        self.blocked_threshold = blocked_threshold
        self._blocked_cycles: dict[object, int] = {}
        self._last_positions: dict[object, tuple[int, int]] = {}
        self._last_times: dict[object, float] = {}

    @staticmethod
    def _validate_position(position) -> tuple[int, int]:
        """Validate a discrete grid position."""
        try:
            if len(position) != 2:
                raise ValueError

            x, y = position

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "position must contain exactly two integer coordinates."
            ) from exc

        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int)
            or not isinstance(y, int)
        ):
            raise ValueError(
                "position coordinates must be integers."
            )

        return (x, y)

    def observe(
        self,
        agent_id,
        position,
        simulation_time: float,
    ) -> bool:
        """
        Record one simulation step for an agent.

        The first observation initializes its history without counting
        a blockage. Subsequent unchanged positions increment the counter.
        Actual displacement resets it.
        """
        if agent_id is None:
            raise ValueError("agent_id cannot be None.")

        position = self._validate_position(position)

        try:
            timestamp = float(simulation_time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "simulation_time must be a finite number."
            ) from exc

        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "simulation_time must be finite and non-negative."
            )

        previous_time = self._last_times.get(agent_id)

        if previous_time is not None and timestamp < previous_time:
            raise ValueError(
                "simulation_time cannot move backwards for an agent."
            )

        previous_position = self._last_positions.get(agent_id)

        if previous_position is None:
            # The first observation is only a baseline, not a blocked cycle.
            self._blocked_cycles[agent_id] = 0
        elif position != previous_position:
            self._blocked_cycles[agent_id] = 0
        else:
            self._blocked_cycles[agent_id] += 1

        self._last_positions[agent_id] = position
        self._last_times[agent_id] = timestamp

        return self.is_blocked(agent_id)

    def is_blocked(self, agent_id) -> bool:
        """Return whether an agent has reached the blockage threshold."""
        return (
            self._blocked_cycles.get(agent_id, 0)
            >= self.blocked_threshold
        )

    def get_blocked_cycles(self, agent_id) -> int:
        """Return the current consecutive blockage count."""
        return self._blocked_cycles.get(agent_id, 0)

    def get_blocked_agents(self) -> list:
        """Return all agents currently classified as persistently blocked."""
        return [
            agent_id
            for agent_id, cycles in self._blocked_cycles.items()
            if cycles >= self.blocked_threshold
        ]

    def reset_agent(self, agent_id) -> None:
        """Forget the observation history of one agent."""
        self._blocked_cycles.pop(agent_id, None)
        self._last_positions.pop(agent_id, None)
        self._last_times.pop(agent_id, None)

    def pause_agent(self, agent_id) -> None:
        """Preserve blockage history during a voluntary traffic yield."""
        if agent_id is None:
            raise ValueError("agent_id cannot be None.")

    def reset(self) -> None:
        """Clear all blockage observations."""
        self._blocked_cycles.clear()
        self._last_positions.clear()
        self._last_times.clear()
