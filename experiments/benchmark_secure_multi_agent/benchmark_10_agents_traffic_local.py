"""Controlled ten-agent integration benchmark: 5 classical + 5 hybrid quantum."""

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
from planners.quantum.quantum_walk_local_navigation import (
    QuantumWalkLocalNavigationPlanner,
)
from simulation.multi_agent_manager import MultiAgentManager
from simulation.traffic_coordinator import TrafficCoordinator


SEED = 42
WIDTH = 25
HEIGHT = 25
TIME_STEP = 1.0 / 30.0
MAX_CYCLES = 2000

STARTS = [
    (1, 1), (23, 23), (1, 23), (23, 1), (11, 12),
    (12, 23), (23, 12), (13, 2), (1, 13), (17, 17),
]

GOALS = [
    (11, 7), (23, 7), (1, 18), (23, 18), (7, 23),
    (17, 23), (7, 1), (7, 18), (5, 7), (17, 8),
]

GOAL_ASSIGNMENT = [0, 7, 3, 9, 8, 2, 4, 1, 6, 5]

RESULTS = (
    Path(__file__).resolve().parent
    / "results"
    / "ten_agents_static_seed42_traffic_local.json"
)
CHECKPOINTS = RESULTS.with_name("ten_agents_static_seed42_traffic_local_checkpoint.json")

def agent_diagnostics(agents, manager):
    blocked = getattr(manager.recovery_coordinator, "_blocked_cycles", {})
    return {
        agent.agent_id: {
            "family": agent.family,
            "position": tuple(agent.position),
            "goal": tuple(agent.goal),
            "navigation_target": tuple(agent.navigation_target),
            "reached_goal": agent.reached_goal,
            "status": agent.status,
            "recovery_state": agent.recovery_state,
            "blocked_cycles": blocked.get(agent.agent_id, 0),
            "recovery_attempts": manager._recovery_attempts.get(agent.agent_id, 0),
            "path_length": len(agent.path) if agent.path else 0,
            "replans": agent.metrics.get("replans", 0),
        } for agent in agents
    }

def write_checkpoint(cycles, history, agents, manager, status):
    data = {
        "status": status,
        "last_completed_cycle": cycles,
        "manager_metrics": manager.get_metrics(),
        "agents": agent_diagnostics(agents, manager),
        "history": history,
    }
    CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINTS.with_suffix(CHECKPOINTS.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(CHECKPOINTS)


def build_agents(maze):
    agents = []

    for index, goal_index in enumerate(GOAL_ASSIGNMENT):
        agent_id = f"R{index + 1:02d}"
        start = STARTS[index]
        goal = GOALS[goal_index]

        if index < 5:
            agent = ClassicalAgent(
                agent_id=agent_id,
                start=start,
                goal=goal,
                planner=AStarPlanner(),
                algorithm_name="A*",
            )
        else:
            planner = QuantumWalkLocalNavigationPlanner(
                window_size=7,
                coin_type="grover",
                shots=16,
                walk_steps=2,
                max_total_qubits=8,
                seed=SEED + index + 6,
                max_segments=200,
            )

            agent = QuantumAgent(
                agent_id=agent_id,
                start=start,
                goal=goal,
                planner=planner,
                algorithm_name="Hierarchical Local Quantum Walk",
                ai_type="hybrid_quantum_classical",
            )

        agents.append(agent)

    return agents


def main():
    maze = Maze(width=WIDTH, height=HEIGHT, seed=SEED)
    maze.generate_static_maze()

    agents = build_agents(maze)

    environment = DynamicEnvironment.from_agents(
        maze,
        agents,
        number_of_walls=0,
        number_of_obstacles=0,
        seed=SEED + 1,
    )

    manager = MultiAgentManager(
        maze,
        agents,
        dynamic_environment=environment,
        time_step=TIME_STEP,
        blocked_threshold=5,
        max_recovery_attempts=3,
        traffic_coordinator=TrafficCoordinator(maze),
    )

    print("===== INITIALISATION DES 10 ROBOTS =====")
    print("Maze:", f"{WIDTH}x{HEIGHT}", "| Seed:", SEED)
    print("Robots: 5 classical + 5 hybrid quantum-classical")
    print("Dynamic walls: 0 | Moving obstacles: 0")

    for agent in agents:
        path = agent.compute_plan(maze)

        if not path:
            raise RuntimeError(
                f"Initial planning failed for {agent.agent_id}"
            )

        print(
            f"{agent.agent_id} | {agent.family:8s} | "
            f"{agent.position} -> {agent.goal} | "
            f"distance={len(path) - 1} | "
            f"planning={agent.metrics['planning_time']:.3f}s"
        )

    history = []
    cycles = 0

    print("\n===== SIMULATION MULTI-AGENT =====")

    while not all(agent.reached_goal for agent in agents):
        if cycles >= MAX_CYCLES:
            break

        try:
            result = manager.step()
        except KeyboardInterrupt:
            print("\nInterruption pendant manager.step().")
            write_checkpoint(cycles, history, agents, manager, "interrupted_during_step")
            print("Checkpoint sauvegardé :", CHECKPOINTS)
            return
        cycles += 1

        history.append({
            "cycle": cycles,
            "simulation_time": result["simulation_time"],
            "positions_before": result["positions_before"],
            "proposals": result["proposals"],
            "positions": result["positions"],
            "proposed": result["proposed"],
            "moved": result["moved"],
            "blocked": result["blocked"],
            "safety_blocked": result["safety_blocked"],
            "replanned": result["replanned"],
            "invalid_proposals": result["invalid_proposals"],
            "destination_conflicts": result["destination_conflicts"],
            "cycle_rejections": result["cycle_rejections"],
            "dependency_blocked": result["dependency_blocked"],
            "obstacle_safety_blocks": result["obstacle_safety_blocks"],
            "robot_safety_blocks": result["robot_safety_blocks"],
            "rejected_cycles": result["rejected_cycles"],
            "recovery_events": result["recovery_events"],
            "agent_states": agent_diagnostics(agents, manager),
        })

        if cycles % 10 == 0:
            write_checkpoint(cycles, history, agents, manager, "running")

        if cycles % 50 == 0:
            completed = sum(agent.reached_goal for agent in agents)
            print(
                f"Cycle {cycles:4d} | "
                f"reached={completed}/10 | "
                f"moved={result['moved']} | "
                f"blocked={result['blocked']} | "
                f"replanned={result['replanned']}"
            )

    completed_agents = [
        agent.agent_id for agent in agents if agent.reached_goal
    ]

    summary = {
        "scenario": {
            "seed": SEED,
            "width": WIDTH,
            "height": HEIGHT,
            "time_step": TIME_STEP,
            "dynamic_walls": 0,
            "moving_obstacles": 0,
            "classical_agents": 5,
            "hybrid_quantum_classical_agents": 5,
            "goal_assignment": GOAL_ASSIGNMENT,
        },
        "cycles": cycles,
        "simulation_time": cycles * TIME_STEP,
        "completed": len(completed_agents),
        "completed_agents": completed_agents,
        "all_completed": len(completed_agents) == len(agents),
        "manager_metrics": manager.get_metrics(),
        "agents": {
            agent.agent_id: {
                "family": agent.family,
                "ai_type": agent.ai_type,
                "start": agent.start,
                "goal": agent.goal,
                "final_position": agent.position,
                "reached_goal": agent.reached_goal,
                "status": agent.status,
                "recovery_state": agent.recovery_state,
                "metrics": agent.metrics,
                "quantum_metadata": getattr(
                    agent.planner, "last_metadata", {}
                ),
            }
            for agent in agents
        },
        "history": history,
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_checkpoint(cycles, history, agents, manager, "completed" if summary["all_completed"] else "max_cycles_reached")

    print("\n===== RÉSULTATS FINAUX =====")
    print(json.dumps({
        "cycles": cycles,
        "completed": len(completed_agents),
        "all_completed": summary["all_completed"],
        "manager_metrics": summary["manager_metrics"],
        "result_file": str(RESULTS),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
