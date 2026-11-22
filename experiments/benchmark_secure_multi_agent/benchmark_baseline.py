"""Reproducible baseline: A* vs hybrid quantum-walk navigation."""

import csv
import json
import sys
from pathlib import Path
from time import perf_counter

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


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SEEDS = (42, 43, 44)
WIDTH = 7
HEIGHT = 7
START = (1, 1)
GOAL = (5, 5)
MAX_CYCLES = 1000

QUANTUM_SHOTS = 32
QUANTUM_WALK_STEPS = 2
QUANTUM_MAX_QUBITS = 8


def run_experiment(seed, algorithm):
    maze = Maze(width=WIDTH, height=HEIGHT, seed=seed)

    if algorithm == "A*":
        planner = AStarPlanner()
        agent = ClassicalAgent(
            agent_id=f"A_STAR_{seed}",
            start=START,
            goal=GOAL,
            planner=planner,
            algorithm_name="A*",
        )
    else:
        planner = QuantumWalkNavigationPlanner(
            maze=maze,
            coin_type="grover",
            shots=QUANTUM_SHOTS,
            walk_steps=QUANTUM_WALK_STEPS,
            seed=seed,
            max_total_qubits=QUANTUM_MAX_QUBITS,
        )
        agent = QuantumAgent(
            agent_id=f"QUANTUM_WALK_{seed}",
            start=START,
            goal=GOAL,
            planner=planner,
            algorithm_name="Quantum Walk Navigation",
            ai_type="hybrid_quantum_classical",
        )

    environment = DynamicEnvironment.from_agents(
        maze,
        [agent],
        number_of_walls=0,
        number_of_obstacles=0,
        seed=seed,
    )

    manager = MultiAgentManager(
        maze,
        [agent],
        dynamic_environment=environment,
        time_step=1.0 / 30.0,
    )

    planning_start = perf_counter()
    agent.compute_plan(maze)
    planning_latency = perf_counter() - planning_start

    simulation_start = perf_counter()
    cycles = 0

    while not agent.reached_goal and cycles < MAX_CYCLES:
        manager.step()
        cycles += 1

    execution_time = perf_counter() - simulation_start
    completed = agent.reached_goal

    row = {
        "seed": seed,
        "algorithm": algorithm,
        "ai_type": agent.ai_type,
        "completed": completed,
        "path_length": agent.metrics["path_length"],
        "planning_latency_s": planning_latency,
        "agent_planning_time_s": agent.metrics["planning_time"],
        "execution_time_s": execution_time,
        "simulation_time_s": manager.simulation_time,
        "simulation_cycles": cycles,
        "moves": agent.metrics["moves"],
        "replans": agent.metrics["replans"],
        "collisions": agent.metrics["collisions"],
        "successful_moves": manager.total_successful_moves,
        "blocked_moves": manager.total_blocked_moves,
        "conflicts": manager.total_conflicts,
        "quantum_decisions": 0,
        "classical_fallback_decisions": 0,
        "quantum_circuit_executions": 0,
        "quantum_shots": 0,
        "quantum_walk_steps": 0,
    }

    if algorithm != "A*":
        metadata = planner.last_metadata
        row.update({
            "quantum_decisions": metadata["quantum_decisions"],
            "classical_fallback_decisions":
                metadata["classical_fallback_decisions"],
            "quantum_circuit_executions":
                metadata["quantum_circuit_executions"],
            "quantum_shots": QUANTUM_SHOTS,
            "quantum_walk_steps": QUANTUM_WALK_STEPS,
        })

    if not completed:
        raise RuntimeError(
            f"{algorithm}, seed={seed}: mission not completed "
            f"after {MAX_CYCLES} cycles."
        )

    return row


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in SEEDS:
        for algorithm in ("A*", "Quantum Walk hybride"):
            print(f"\nRunning {algorithm} | seed={seed}", flush=True)
            row = run_experiment(seed, algorithm)
            rows.append(row)

            print(
                f"  completed={row['completed']} | "
                f"path={row['path_length']} | "
                f"planning={row['planning_latency_s']:.6f}s | "
                f"moves={row['moves']} | "
                f"cycles={row['simulation_cycles']}",
                flush=True,
            )

    csv_path = RESULTS / "baseline_results.csv"
    json_path = RESULTS / "baseline_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, ensure_ascii=False)

    print("\n===== BASELINE COMPLETE =====")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Experiments: {len(rows)}")
    print(
        "Successful missions:",
        sum(row["completed"] for row in rows),
        "/",
        len(rows),
    )


if __name__ == "__main__":
    main()
