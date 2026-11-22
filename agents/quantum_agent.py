"""
Quantum and hybrid quantum-classical AI autonomous robot.
"""

from agents.base_agent import BaseAgent


class QuantumAgent(BaseAgent):
    """Robot controlled by a quantum or hybrid quantum-classical planner."""

    def __init__(
        self,
        agent_id,
        start,
        goal,
        planner,
        algorithm_name,
        ai_type="hybrid_quantum_classical",
    ):
        super().__init__(
            agent_id=agent_id,
            start=start,
            goal=goal,
            planner=planner,
            family="quantum",
        )

        self.algorithm_name = algorithm_name
        self.ai_type = ai_type
