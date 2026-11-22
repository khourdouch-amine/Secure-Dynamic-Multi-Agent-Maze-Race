"""
Classical AI autonomous robot.
"""

from agents.base_agent import BaseAgent


class ClassicalAgent(BaseAgent):
    """Robot controlled by a classical AI planning algorithm."""

    def __init__(
        self,
        agent_id,
        start,
        goal,
        planner,
        algorithm_name,
    ):
        super().__init__(
            agent_id=agent_id,
            start=start,
            goal=goal,
            planner=planner,
            family="classical",
        )

        self.algorithm_name = algorithm_name
        self.ai_type = "classical"
