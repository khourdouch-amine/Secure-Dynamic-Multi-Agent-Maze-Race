"""First shared dynamic multi-agent experiment."""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.classical_agent import ClassicalAgent
from agents.quantum_agent import QuantumAgent
from environment.dynamic_elements import DynamicEnvironment
from environment.maze import Maze
from planners.classical.astar import AStarPlanner
from planners.quantum.quantum_walk_navigation import (
    QuantumWalkNavigationPlanner,
)
from simulation.multi_agent_manager import MultiAgentManager


SEED = 42
WIDTH = 7
HEIGHT = 7
MAX_CYCLES = 600
TIME_STEP = 1.0 / 30.0

SHOTS = 32
WALK_STEPS = 2
MAX_QUBITS = 8

RESULTS = (
    Path(__file__).resolve().parent
    / "results"
    / "dynamic_multi_agent_seed42.json"
)


def main():
    maze = Maze(width=WIDTH, height=HEIGHT, seed=SEED)

    # Opposing missions: both agents share the same environment.
    classical = ClassicalAgent(
        agent_id="CLASSICAL_ASTAR",
        start=(1, 3),
        goal=(5, 3),
        planner=AStarPlanner(),
        algorithm_name="A*",
    )

    quantum_planner = QuantumWalkNavigationPlanner(
        maze=maze,
        coin_type="grover",
        shots=SHOTS,
        walk_steps=WALK_STEPS,
        seed=SEED,
        max_total_qubits=MAX_QUBITS,
    )

    quantum = QuantumAgent(
        agent_id="HYBRID_QUANTUM_WALK",
        start=(5, 3),
        goal=(1, 3),
        planner=quantum_planner,
        algorithm_name="Quantum Walk Navigation",
        ai_type="hybrid_quantum_classical",
    )

    agents = [classical, quantum]

    environment = DynamicEnvironment.from_agents(
        maze,
        agents,
        number_of_walls=0,
        number_of_obstacles=1,
        seed=SEED + 1,
    )

    # Accelerated deterministic schedule for this short pilot.
    environment.obstacle_interval = 0.10
    environment.next_obstacle_update = 0.10

    manager = MultiAgentManager(
        maze,
        agents,
        dynamic_environment=environment,
        time_step=TIME_STEP,
        blocked_threshold=5,
        max_recovery_attempts=3,
    )

    print("===== INITIAL SCENARIO =====")
    print("Seed:", SEED)
    print("Moving obstacles:", environment.obstacle_registry)
    print("Obstacle interval:", environment.obstacle_interval)

    for agent in agents:
        agent.compute_plan(maze)
        print(
            agent.agent_id,
            "| initial path:", agent.path,
            "| length:", agent.metrics["path_length"],
        )

    history = []
    cycles = 0

    print("\n===== SIMULATION =====")

    while not all(agent.reached_goal for agent in agents):
        if cycles >= MAX_CYCLES:
            break

        result = manager.step()
        cycles += 1

        history.append({
            "cycle": cycles,
            "simulation_time": result["simulation_time"],
            "positions_before": result["positions_before"],
            "proposals": result["proposals"],
            "positions": result["positions"],
            "rejected_cycles": result["rejected_cycles"],
            "recovery_events": result["recovery_events"],
            "proposed": result["proposed"],
            "moved": result["moved"],
            "blocked": result["blocked"],
            "safety_blocked": result["safety_blocked"],
            "invalid_proposals": result["invalid_proposals"],
            "destination_conflicts": result["destination_conflicts"],
            "cycle_rejections": result["cycle_rejections"],
            "dependency_blocked": result["dependency_blocked"],
            "obstacle_safety_blocks": result["obstacle_safety_blocks"],
            "robot_safety_blocks": result["robot_safety_blocks"],
            "replanned": result["replanned"],
            "obstacles_updated": result["environment_updates"][
                "obstacles_updated"
            ],
            "changed_cells": result["environment_updates"]["changed_cells"],
        })

        if cycles % 30 == 0 or all(
            agent.reached_goal for agent in agents
        ):
            print(
                f"cycle={cycles:03d} | "
                f"time={result['simulation_time']:.2f}s | "
                f"positions={result['positions']} | "
                f"blocked={manager.total_blocked_moves} | "
                f"conflicts={manager.total_conflicts}"
            )

    summary = {
        "seed": SEED,
        "cycles": cycles,
        "simulation_time_s": manager.simulation_time,
        "completed": {
            agent.agent_id: agent.reached_goal
            for agent in agents
        },
        "agents": {
            agent.agent_id: {
                "ai_type": agent.ai_type,
                "start": agent.start,
                "goal": agent.goal,
                "final_position": agent.position,
                "status": agent.status,
                "recovery_state": getattr(
                    agent, "recovery_state", None
                ),
                "metrics": agent.metrics,
            }
            for agent in agents
        },
        "manager_metrics": manager.get_metrics(),
        "initial_obstacles": {
            key: value
            for key, value in environment.obstacle_registry.items()
        },
        "final_obstacles": {
            key: value
            for key, value in environment.obstacle_registry.items()
        },
        "history": history,
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== FINAL RESULTS =====")
    print(json.dumps({
        "cycles": cycles,
        "completed": summary["completed"],
        "manager_metrics": summary["manager_metrics"],
        "final_positions": {
            agent.agent_id: agent.position for agent in agents
        },
        "recovery_states": {
            agent.agent_id: getattr(agent, "recovery_state", None)
            for agent in agents
        },
        "result_file": str(RESULTS),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
