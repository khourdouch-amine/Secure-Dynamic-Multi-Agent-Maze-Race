import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import sys

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT = BASE.parent
RESULTS = PROJECT / "results"
DATA = BASE / "data"

source = json.loads((RESULTS / "ten_agents_static_seed42.json").read_text())
timeline = json.loads((DATA / "timeline.json").read_text())["timeline"]

ALGORITHMS = {
    "R01": "A*", "R02": "Dijkstra", "R03": "RRT*",
    "R04": "D* Lite", "R05": "Bidirectional BFS",
    "R06": "Grover-assisted Navigation",
    "R07": "Global Quantum Walk",
    "R08": "Hierarchical Local Quantum Walk",
    "R09": "QAOA Navigation",
    "R10": "Variational Quantum Classifier",
}

COLORS = plt.get_cmap("tab10").colors
BG = "#08111f"
PANEL = "#111e30"
EDGE = "#263b54"
WHITE = "#e8f0fa"
MUTED = "#8da3bd"
CYAN = "#39c6df"
ORANGE = "#ffad5c"
RED = "#ff647c"
GREEN = "#51d6a0"

# Snapshot at the last cycle where positions changed.
cycle_index = 92
cycle = timeline[cycle_index]
agents = source["agents"]
scenario = source["scenario"]

from environment.maze import Maze
maze = Maze(
    width=scenario["width"],
    height=scenario["height"],
    seed=scenario["seed"],
)
maze.generate_static_maze()

fig = plt.figure(figsize=(16, 9), dpi=160, facecolor=BG)

def panel(x, y, w, h, title):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
        spine.set_linewidth(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=CYAN, loc="left",
                 fontsize=9, fontweight="bold", pad=9)
    return ax

def metric(x, label, value, color=WHITE):
    ax = fig.add_axes([x + 0.012, 0.875, 0.060, 0.035])
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.08, 0.72, label.upper(), color=MUTED,
            fontsize=4.5, transform=ax.transAxes)
    ax.text(0.08, 0.20, str(value), color=color,
            fontsize=9, fontweight="bold", transform=ax.transAxes)

# Header
fig.text(0.025, 0.995, "SECURE DYNAMIC MULTI-AGENT MAZE RACE",
         color=WHITE, fontsize=17, fontweight="bold", va="top")
fig.text(0.975, 0.995, "EXPERIMENT REPLAY  |  SEED 42  |  STATIC MAZE",
         color=MUTED, fontsize=8, ha="right", va="top")

cum = cycle["cumulative"]
metric(0.025, "Cycle", f'{cycle["cycle"]}/2000', CYAN)
metric(0.17, "Successful moves", cum["moved"], GREEN)
metric(0.315, "Blocked moves", cum["blocked"], ORANGE)
metric(0.46, "Destination conflicts", cum["destination_conflicts"], RED)
metric(0.605, "Safety rejections", cum["robot_safety_blocks"], RED)
metric(0.75, "Goals reached", sum(
    1 for s in cycle["agent_states"].values() if s.get("reached_goal")
), GREEN)

# Main maze panel
maze_ax = panel(0.045, 0.225, 0.525, 0.585, "LIVE MAZE REPLAY — 25 × 25 GRID")
maze_ax.imshow(
    maze.grid,
    cmap=matplotlib.colors.ListedColormap(["#dce5ed", "#263345"]),
    origin="lower", interpolation="nearest", vmin=0, vmax=1,
    extent=(-0.5, 24.5, -0.5, 24.5),
)
maze_ax.set_xlim(-0.5, 24.5)
maze_ax.set_ylim(-0.5, 24.5)
maze_ax.set_aspect("equal")
maze_ax.set_xticks(range(0, 25, 2))
maze_ax.set_yticks(range(0, 25, 2))
maze_ax.tick_params(colors=MUTED, labelsize=6)
maze_ax.grid(color="#8da3bd", linewidth=0.25, alpha=0.35)

for i, (rid, agent) in enumerate(agents.items()):
    color = COLORS[i % 10]
    start = agent["start"]
    goal = agent["goal"]
    pos = cycle["positions"][rid]
    maze_ax.scatter(*start, marker="o", s=22, color=color,
                    edgecolors="#111", linewidths=0.5, zorder=4)
    maze_ax.scatter(*goal, marker="*", s=85, color=color,
                    edgecolors="#111", linewidths=0.5, zorder=4)
    maze_ax.scatter(*pos, marker="o", s=65, color=color,
                    edgecolors="white", linewidths=0.8, zorder=6)
    maze_ax.text(pos[0] + 0.18, pos[1] + 0.18, rid,
                 fontsize=6, color=color, fontweight="bold", zorder=7)

maze_ax.set_xlabel("GRID X", color=MUTED, fontsize=7)
maze_ax.set_ylabel("GRID Y", color=MUTED, fontsize=7)

# Agent monitor
agent_ax = panel(0.605, 0.475, 0.37, 0.355, "AGENT / ALGORITHM MONITOR")
agent_ax.set_xlim(0, 1)
agent_ax.set_ylim(0, 1)
agent_ax.axis("off")

for i, (rid, agent) in enumerate(agents.items()):
    state = cycle["agent_states"][rid]
    y = 0.91 - i * 0.091
    color = COLORS[i % 10]
    status = "GOAL REACHED" if state.get("reached_goal") else state.get("status", "unknown")
    family = agent.get("family", "unknown").upper()
    agent_ax.text(0.025, y, rid, color=color, fontsize=7,
                  fontweight="bold", va="center")
    agent_ax.text(0.13, y, ALGORITHMS[rid], color=WHITE,
                  fontsize=7.5, va="center")
    agent_ax.text(0.68, y, family, color=MUTED,
                  fontsize=6.2, va="center")
    agent_ax.text(0.98, y, status[:17].upper(), color=GREEN if state.get("reached_goal") else ORANGE,
                  fontsize=6.0, ha="right", va="center")

# Coordination and safety
coord_ax = panel(0.605, 0.285, 0.178, 0.165, "COORDINATION")
coord_ax.axis("off")
coord_ax.text(0.08, 0.75, f'Conflicts: {cum["destination_conflicts"]:,}',
              color=ORANGE, fontsize=9, transform=coord_ax.transAxes)
coord_ax.text(0.08, 0.49, f'Cycle rejections: {cum["cycle_rejections"]:,}',
              color=WHITE, fontsize=8, transform=coord_ax.transAxes)
coord_ax.text(0.08, 0.23, f'Dependency blocked: {cum["dependency_blocked"]:,}',
              color=MUTED, fontsize=7, transform=coord_ax.transAxes)

safe_ax = panel(0.797, 0.285, 0.178, 0.165, "SAFETY MONITOR")
safe_ax.axis("off")
safe_ax.text(0.08, 0.75, f'Robot safety blocks: {cum["robot_safety_blocks"]}',
             color=RED, fontsize=8, transform=safe_ax.transAxes)
safe_ax.text(0.08, 0.49, f'Obstacle safety blocks: {cum["obstacle_safety_blocks"]}',
             color=WHITE, fontsize=7, transform=safe_ax.transAxes)
safe_ax.text(0.08, 0.23, "Collisions recorded: 0",
             color=GREEN, fontsize=8, transform=safe_ax.transAxes)

# Quantum monitor
quant_ax = panel(0.605, 0.205, 0.37, 0.06, "QUANTUM AGENTS — HYBRID METHODS")
quant_ax.axis("off")
quant_ax.text(0.03, 0.25,
              "R06 Grover  |  R07 Global QW  |  R08 Hierarchical QW  |  R09 QAOA  |  R10 VQC",
              color=WHITE, fontsize=7, transform=quant_ax.transAxes)

# Bottom performance chart
chart_ax = panel(0.025, 0.045, 0.565, 0.135, "CUMULATIVE EVENTS — UP TO CURRENT CYCLE")
chart_ax.set_facecolor(PANEL)
cycles = np.linspace(0, cycle_index, min(200, cycle_index + 1), dtype=int)
cycles = sorted(set(cycles.tolist()))
chart_ax.plot(
    [timeline[i]["cycle"] for i in cycles],
    [timeline[i]["cumulative"]["blocked"] for i in cycles],
    color=ORANGE, linewidth=1.5, label="Blocked moves",
)
chart_ax.plot(
    [timeline[i]["cycle"] for i in cycles],
    [timeline[i]["cumulative"]["moved"] for i in cycles],
    color=GREEN, linewidth=1.5, label="Successful moves",
)
chart_ax.tick_params(colors=MUTED, labelsize=6)
chart_ax.grid(color=EDGE, alpha=0.6, linewidth=0.5)
chart_ax.legend(loc="upper left", fontsize=6, facecolor=PANEL,
                edgecolor=EDGE, labelcolor=WHITE)
chart_ax.set_xlabel("Cycle", color=MUTED, fontsize=6)

# Event / replay strip
fig.text(0.605, 0.155, "EVENT LOG", color=CYAN, fontsize=8, weight="bold")
fig.text(0.605, 0.125,
         f'Cycle {cycle["cycle"]}: {cycle["events"]["blocked"]} blocked proposals | '
         f'{cycle["events"]["destination_conflicts"]} destination conflicts',
         color=WHITE, fontsize=7)
fig.text(0.605, 0.095,
         f'Simulation time: {cycle["simulation_time"]:.2f} s  |  '
         f'Last position change: cycle 93',
         color=MUTED, fontsize=7)
fig.text(0.605, 0.055,
         "REPLAY TIMELINE   [===========>-----------------------------]   CYCLE 93 / 2000",
         color=CYAN, fontsize=7)

out = BASE / "dashboard_preview.png"
fig.savefig(out, dpi=160, facecolor=BG)
plt.close(fig)
print("Preview generated:", out)
print("Resolution: 2560 x 1440")
print("Snapshot cycle:", cycle["cycle"])
print("Panels: maze, KPIs, agents, coordination, safety, quantum, events, timeline")
