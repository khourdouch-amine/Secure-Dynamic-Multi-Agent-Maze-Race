import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment.maze import Maze

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/benchmark_secure_multi_agent/results"
DATA_FILE = RESULTS / "ten_agents_static_seed42.json"
OUTPUT_FILE = RESULTS / "maze_overview_algorithms.png"

ALGORITHMS = {
    "R01": "A*",
    "R02": "Dijkstra",
    "R03": "RRT*",
    "R04": "D* Lite",
    "R05": "Bidirectional BFS",
    "R06": "Grover-assisted Navigation",
    "R07": "Global Quantum Walk",
    "R08": "Hierarchical Local Quantum Walk",
    "R09": "QAOA Navigation",
    "R10": "Variational Quantum Classifier (VQC)",
}

with DATA_FILE.open("r", encoding="utf-8") as file:
    data = json.load(file)

scenario = data["scenario"]
width, height, seed = scenario["width"], scenario["height"], scenario["seed"]

maze = Maze(width=width, height=height, seed=seed)
maze.generate_static_maze()

agents = data["agents"]
history = data["history"]
robot_ids = list(agents)
colors = plt.get_cmap("tab10").colors

fig, ax = plt.subplots(figsize=(16, 11))
fig.subplots_adjust(left=0.07, right=0.72, top=0.91, bottom=0.10)

ax.imshow(
    maze.grid,
    cmap=matplotlib.colors.ListedColormap(["#f7f7f7", "#252a34"]),
    origin="lower",
    interpolation="nearest",
    vmin=0,
    vmax=1,
    extent=(-0.5, width - 0.5, -0.5, height - 0.5),
)

for i, robot_id in enumerate(robot_ids):
    color = colors[i % len(colors)]
    agent = agents[robot_id]

    trajectory = np.asarray(
        [cycle["positions"][robot_id] for cycle in history],
        dtype=float,
    )

    ax.plot(
        trajectory[:, 0], trajectory[:, 1],
        color=color, linewidth=1.4, alpha=0.8, zorder=3,
    )

    start = agent["start"]
    goal = agent["goal"]
    final = agent["final_position"]

    ax.scatter(
        *start, marker="o", s=75, color=color,
        edgecolors="black", linewidths=0.8, zorder=5,
    )
    ax.scatter(
        *goal, marker="*", s=190, color=color,
        edgecolors="black", linewidths=0.7, zorder=6,
    )
    ax.scatter(
        *final, marker="X", s=95, color=color,
        edgecolors="black", linewidths=0.8, zorder=7,
    )

    ax.annotate(
        robot_id, final, xytext=(5, 5),
        textcoords="offset points", fontsize=8,
        fontweight="bold", color=color, zorder=8,
    )

ax.set_xlim(-0.5, width - 0.5)
ax.set_ylim(-0.5, height - 0.5)
ax.set_xticks(range(width))
ax.set_yticks(range(height))
ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
ax.grid(which="minor", color="#b8b8b8", linewidth=0.35, alpha=0.6)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_aspect("equal")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")

ax.set_title(
    f"Secure Dynamic Multi-Agent Maze Race — Robot Trajectories\n"
    f"{width}x{height} maze | Seed {seed} | "
    f"{len(agents)} robots | {len(history)} cycles",
    fontsize=13, fontweight="bold", pad=14,
)

fig.text(
    0.75, 0.88, "ROBOT / ALGORITHM",
    fontsize=12, fontweight="bold",
)

for i, robot_id in enumerate(robot_ids):
    y = 0.83 - i * 0.065
    color = colors[i % len(colors)]
    family = agents[robot_id].get("family", "unknown")
    reached = agents[robot_id].get("reached_goal", False)
    status = "Goal reached" if reached else "Not reached"

    fig.text(
        0.75, y,
        f"{robot_id} — {ALGORITHMS[robot_id]}",
        fontsize=9, color=color, fontweight="bold",
    )
    fig.text(
        0.77, y - 0.022,
        f"{family} | {status}",
        fontsize=7.5, color="#555555",
    )

fig.text(
    0.75, 0.12,
    "MARKERS\n"
    "Circle: start\n"
    "Star: goal\n"
    "X: final position\n"
    "Line: executed trajectory",
    fontsize=9, linespacing=1.7,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f2f2",
              edgecolor="#cccccc"),
)

fig.savefig(OUTPUT_FILE, dpi=220, bbox_inches="tight")
plt.close(fig)

print("Updated visualization generated.")
print("Output:", OUTPUT_FILE)
print("Robots with algorithm labels:", len(ALGORITHMS))
print("Reached goals:", data["completed"])
