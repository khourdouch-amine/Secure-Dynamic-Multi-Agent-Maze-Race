import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from environment.maze import Maze

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/benchmark_secure_multi_agent/results"
DATA_FILE = RESULTS / "ten_agents_static_seed42.json"
OUTPUT_FILE = RESULTS / "maze_animation.gif"

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

FPS = 10
TARGET_DURATION = 40
FRAME_COUNT = FPS * TARGET_DURATION

with DATA_FILE.open("r", encoding="utf-8") as file:
    data = json.load(file)

scenario = data["scenario"]
width, height = scenario["width"], scenario["height"]

maze = Maze(width=width, height=height, seed=scenario["seed"])
maze.generate_static_maze()

agents = data["agents"]
history = data["history"]
robot_ids = list(agents)
colors = plt.get_cmap("tab10").colors

# Exactly 400 frames, including the first and last recorded cycles.
frame_indices = np.linspace(
    0, len(history) - 1, FRAME_COUNT, dtype=int
)

paths = {
    robot_id: np.asarray(
        [cycle["positions"][robot_id] for cycle in history],
        dtype=float,
    )
    for robot_id in robot_ids
}

fig, ax = plt.subplots(figsize=(15, 10))
fig.subplots_adjust(left=0.06, right=0.72, top=0.91, bottom=0.09)

ax.imshow(
    maze.grid,
    cmap=matplotlib.colors.ListedColormap(["#f7f7f7", "#252a34"]),
    origin="lower",
    interpolation="nearest",
    vmin=0,
    vmax=1,
    extent=(-0.5, width - 0.5, -0.5, height - 0.5),
)

ax.set_xlim(-0.5, width - 0.5)
ax.set_ylim(-0.5, height - 0.5)
ax.set_aspect("equal")
ax.set_xticks(range(width))
ax.set_yticks(range(height))
ax.grid(color="#b8b8b8", linewidth=0.3, alpha=0.5)
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")

lines = {}
robots = {}
labels = {}
algorithm_texts = {}

for i, robot_id in enumerate(robot_ids):
    color = colors[i % len(colors)]
    agent = agents[robot_id]

    ax.scatter(
        *agent["start"], marker="o", s=45,
        color=color, edgecolors="black", linewidths=0.6,
        zorder=4,
    )
    ax.scatter(
        *agent["goal"], marker="*", s=130,
        color=color, edgecolors="black", linewidths=0.5,
        zorder=4,
    )

    lines[robot_id], = ax.plot(
        [], [], color=color, linewidth=1.2, alpha=0.7
    )
    robots[robot_id] = ax.scatter(
        [], [], s=65, marker="o", color=color,
        edgecolors="black", linewidths=0.8, zorder=6,
    )
    labels[robot_id] = ax.text(
        0, 0, robot_id, fontsize=7, fontweight="bold",
        color=color, zorder=7,
    )

fig.text(
    0.75, 0.88, "ROBOT / ALGORITHM",
    fontsize=12, fontweight="bold",
)

for i, robot_id in enumerate(robot_ids):
    y = 0.82 - i * 0.065
    color = colors[i % len(colors)]
    family = agents[robot_id].get("family", "unknown")

    fig.text(
        0.75, y,
        f"{robot_id} — {ALGORITHMS[robot_id]}",
        fontsize=8.5, color=color, fontweight="bold",
    )
    algorithm_texts[robot_id] = fig.text(
        0.77, y - 0.023,
        f"{family} | Initialisation",
        fontsize=7.5, color="#555555",
    )

fig.text(
    0.75, 0.12,
    "MARKERS\nCircle: start\nStar: goal\n"
    "Circle (moving): robot\nLine: executed trajectory",
    fontsize=9, linespacing=1.6,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#f2f2f2",
        edgecolor="#cccccc",
    ),
)

title = ax.set_title("", fontsize=12, fontweight="bold")

def update(frame_number):
    cycle_index = int(frame_indices[frame_number])
    cycle = history[cycle_index]
    states = cycle.get("agent_states", {})

    for robot_id in robot_ids:
        trajectory = paths[robot_id][:cycle_index + 1]
        x, y = trajectory[-1]

        lines[robot_id].set_data(
            trajectory[:, 0], trajectory[:, 1]
        )
        robots[robot_id].set_offsets([[x, y]])
        labels[robot_id].set_position((x + 0.18, y + 0.18))

        state = states.get(robot_id, {})
        reached = state.get("reached_goal", False)
        status = "Goal reached" if reached else state.get("status", "moving")

        algorithm_texts[robot_id].set_text(
            f"{agents[robot_id].get('family', 'unknown')} | {status}"
        )

    title.set_text(
        f"Secure Dynamic Multi-Agent Maze Race | "
        f"Cycle {cycle['cycle']}/{len(history)} | "
        f"Simulation time: {cycle['simulation_time']:.2f} s"
    )

    return [
        title,
        *lines.values(),
        *robots.values(),
        *labels.values(),
        *algorithm_texts.values(),
    ]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=FRAME_COUNT,
    interval=1000 / FPS,
    blit=False,
)

print(
    f"Generating {FRAME_COUNT} frames at {FPS} FPS "
    f"({TARGET_DURATION} seconds)...",
    flush=True,
)

ani.save(
    OUTPUT_FILE,
    writer=animation.PillowWriter(fps=FPS),
    dpi=85,
)

plt.close(fig)

print("Animation generated successfully.")
print("Output:", OUTPUT_FILE)
print("Frames:", FRAME_COUNT)
print("FPS:", FPS)
print("Duration:", f"{FRAME_COUNT / FPS:.1f} seconds")
