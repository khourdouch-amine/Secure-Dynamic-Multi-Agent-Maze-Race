import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))

from environment.maze import Maze

RESULTS = ROOT / "experiments/benchmark_secure_multi_agent/results"
DATA = BASE / "data"
OUTPUT = BASE / "secure_dynamic_multi_agent_maze_race.mp4"

FPS = 10
DURATION = 40
FRAMES = FPS * DURATION
DPI = 160

with open(RESULTS / "ten_agents_static_seed42.json") as f:
    source = json.load(f)

with open(DATA / "timeline.json") as f:
    timeline = json.load(f)["timeline"]

agents = source["agents"]
scenario = source["scenario"]
robot_ids = list(agents)

ALGORITHMS = {
    "R01": "A*", "R02": "Dijkstra", "R03": "RRT*",
    "R04": "D* Lite", "R05": "Bidirectional BFS",
    "R06": "Grover-assisted Navigation",
    "R07": "Global Quantum Walk",
    "R08": "Hierarchical Local Quantum Walk",
    "R09": "QAOA Navigation",
    "R10": "Variational Quantum Classifier",
}

BG = "#08111f"
PANEL = "#111e30"
EDGE = "#263b54"
WHITE = "#e8f0fa"
MUTED = "#8da3bd"
CYAN = "#39c6df"
ORANGE = "#ffad5c"
RED = "#ff647c"
GREEN = "#51d6a0"
COLORS = plt.get_cmap("tab10").colors

maze = Maze(
    width=scenario["width"],
    height=scenario["height"],
    seed=scenario["seed"],
)
maze.generate_static_maze()

fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor=BG)

def panel(x, y, w, h, title):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(EDGE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=CYAN, loc="left",
                 fontsize=9, fontweight="bold", pad=8)
    return ax

# Header
fig.text(0.025, 0.995, "SECURE DYNAMIC MULTI-AGENT MAZE RACE",
         color=WHITE, fontsize=17, fontweight="bold", va="top")
fig.text(0.975, 0.995, "EXPERIMENT REPLAY | SEED 42 | STATIC MAZE",
         color=MUTED, fontsize=8, ha="right", va="top")

# Compact KPI cards
kpi_specs = [
    (0.025, "CYCLE", CYAN),
    (0.17, "SUCCESSFUL MOVES", GREEN),
    (0.315, "BLOCKED MOVES", ORANGE),
    (0.46, "DESTINATION CONFLICTS", RED),
    (0.605, "SAFETY REJECTIONS", RED),
    (0.75, "GOALS REACHED", GREEN),
]
kpi_values = []

for x, label, color in kpi_specs:
    ax = fig.add_axes([x + 0.012, 0.875, 0.060, 0.035])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(EDGE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.08, 0.72, label, color=MUTED,
            fontsize=5.5, transform=ax.transAxes)
    value = ax.text(0.08, 0.16, "0", color=color,
                    fontsize=9, fontweight="bold",
                    transform=ax.transAxes)
    kpi_values.append(value)

# Central maze
maze_ax = panel(0.045, 0.225, 0.525, 0.585,
                "LIVE MAZE REPLAY — 25 × 25 GRID")
maze_ax.imshow(
    maze.grid,
    cmap=ListedColormap(["#dce5ed", "#263345"]),
    origin="lower", interpolation="nearest",
    vmin=0, vmax=1, extent=(-0.5, 24.5, -0.5, 24.5),
)
maze_ax.set_xlim(-0.5, 24.5)
maze_ax.set_ylim(-0.5, 24.5)
maze_ax.set_aspect("equal")
maze_ax.set_xticks(range(0, 25, 2))
maze_ax.set_yticks(range(0, 25, 2))
maze_ax.tick_params(colors=MUTED, labelsize=6)
maze_ax.grid(color="#8da3bd", linewidth=0.25, alpha=0.35)
maze_ax.set_xlabel("GRID X", color=MUTED, fontsize=7)
maze_ax.set_ylabel("GRID Y", color=MUTED, fontsize=7)

robot_artists = {}
trail_artists = {}
robot_labels = {}

paths = {
    rid: np.asarray(
        [h["positions"][rid] for h in timeline], dtype=float
    )
    for rid in robot_ids
}

for i, rid in enumerate(robot_ids):
    color = COLORS[i % len(COLORS)]
    agent = agents[rid]

    maze_ax.scatter(*agent["start"], marker="o", s=22,
                    color=color, edgecolors="#111", linewidths=.5,
                    zorder=4)
    maze_ax.scatter(*agent["goal"], marker="*", s=85,
                    color=color, edgecolors="#111", linewidths=.5,
                    zorder=4)

    trail_artists[rid], = maze_ax.plot(
        [], [], color=color, linewidth=1.1, alpha=.65, zorder=5
    )
    robot_artists[rid] = maze_ax.scatter(
        [], [], s=58, color=color, edgecolors="white",
        linewidths=.8, zorder=7
    )
    robot_labels[rid] = maze_ax.text(
        0, 0, rid, color=color, fontsize=6,
        fontweight="bold", zorder=8
    )

# Agent monitor
agent_ax = panel(0.605, 0.475, 0.37, 0.355,
                 "AGENT / ALGORITHM MONITOR")
agent_ax.set_xlim(0, 1)
agent_ax.set_ylim(0, 1)
agent_ax.axis("off")

agent_status_text = {}
for i, rid in enumerate(robot_ids):
    y = 0.91 - i * 0.091
    color = COLORS[i % 10]
    agent_ax.text(.025, y, rid, color=color,
                  fontsize=7, fontweight="bold", va="center")
    agent_ax.text(.13, y, ALGORITHMS[rid],
                  color=WHITE, fontsize=6.5, va="center")
    agent_ax.text(.68, y, agents[rid].get("family", "unknown").upper(),
                  color=MUTED, fontsize=5.5, va="center")
    agent_status_text[rid] = agent_ax.text(
        .98, y, "WAITING", color=ORANGE,
        fontsize=5.5, ha="right", va="center"
    )

# Coordination panel
coord_ax = panel(.605, .285, .178, .165, "COORDINATION")
coord_ax.axis("off")
coord_texts = [
    coord_ax.text(.08, .75, "", color=ORANGE, fontsize=8,
                  transform=coord_ax.transAxes),
    coord_ax.text(.08, .49, "", color=WHITE, fontsize=7,
                  transform=coord_ax.transAxes),
    coord_ax.text(.08, .23, "", color=MUTED, fontsize=7,
                  transform=coord_ax.transAxes),
]

# Safety panel
safe_ax = panel(.797, .285, .178, .165, "SAFETY MONITOR")
safe_ax.axis("off")
safe_texts = [
    safe_ax.text(.08, .75, "", color=RED, fontsize=7.5,
                 transform=safe_ax.transAxes),
    safe_ax.text(.08, .49, "", color=WHITE, fontsize=7,
                 transform=safe_ax.transAxes),
    safe_ax.text(.08, .23, "Recorded collisions: 0",
                 color=GREEN, fontsize=7.5,
                 transform=safe_ax.transAxes),
]

# Quantum methods panel
quant_ax = panel(.605, .205, .37, .06,
                 "QUANTUM AGENTS — HYBRID METHODS")
quant_ax.axis("off")
quant_ax.text(
    .03, .25,
    "R06 Grover | R07 Global QW | R08 Hierarchical QW | R09 QAOA | R10 VQC",
    color=WHITE, fontsize=7, transform=quant_ax.transAxes
)

# Cumulative chart
chart_ax = panel(.025, .045, .565, .135,
                 "CUMULATIVE EVENTS — UP TO CURRENT CYCLE")
chart_ax.set_xlim(1, len(timeline))
chart_ax.set_ylim(0, max(100, sum(h["events"]["blocked"] for h in timeline) * 1.05))
chart_ax.tick_params(colors=MUTED, labelsize=6)
chart_ax.grid(color=EDGE, alpha=.6, linewidth=.5)
chart_ax.set_xlabel("Cycle", color=MUTED, fontsize=6)

blocked_line, = chart_ax.plot([], [], color=ORANGE,
                              linewidth=1.5, label="Blocked moves")
moved_line, = chart_ax.plot([], [], color=GREEN,
                            linewidth=1.5, label="Successful moves")
chart_ax.legend(loc="upper left", fontsize=6, facecolor=PANEL,
                edgecolor=EDGE, labelcolor=WHITE)

# Event log and replay timeline
fig.text(.605, .155, "EVENT LOG", color=CYAN,
         fontsize=8, fontweight="bold")
event_text = fig.text(.605, .125, "", color=WHITE, fontsize=7)
time_text = fig.text(.605, .095, "", color=MUTED, fontsize=7)
progress_text = fig.text(.605, .055, "", color=CYAN, fontsize=7)
phase_text = fig.text(.605, .025, "", color=ORANGE, fontsize=7)

def update(frame):
    idx = round(frame * (len(timeline) - 1) / (FRAMES - 1))
    h = timeline[idx]
    cumulative = h["cumulative"]
    states = h["agent_states"]

    # KPI values
    goals = sum(1 for state in states.values()
                if state.get("reached_goal", False))
    values = [
        f'{h["cycle"]}/2000',
        f'{cumulative["moved"]:,}',
        f'{cumulative["blocked"]:,}',
        f'{cumulative["destination_conflicts"]:,}',
        f'{cumulative["robot_safety_blocks"]:,}',
        str(goals),
    ]
    for artist, value in zip(kpi_values, values):
        artist.set_text(value)

    # Robot positions and trails
    for rid in robot_ids:
        path = paths[rid][:idx + 1]
        stride = max(1, len(path) // 250)
        sampled = path[::stride]
        if not np.array_equal(sampled[-1], path[-1]):
            sampled = np.vstack([sampled, path[-1]])

        trail_artists[rid].set_data(sampled[:, 0], sampled[:, 1])
        pos = path[-1]
        robot_artists[rid].set_offsets([pos])
        robot_labels[rid].set_position((pos[0] + .16, pos[1] + .16))

        state = states[rid]
        status = ("GOAL REACHED" if state.get("reached_goal")
                  else str(state.get("status", "WAITING")).upper())
        agent_status_text[rid].set_text(status[:18])
        agent_status_text[rid].set_color(
            GREEN if state.get("reached_goal") else ORANGE
        )

    # Coordination and safety
    coord_texts[0].set_text(
        f'Conflicts: {cumulative["destination_conflicts"]:,}'
    )
    coord_texts[1].set_text(
        f'Cycle rejections: {cumulative["cycle_rejections"]:,}'
    )
    coord_texts[2].set_text(
        f'Dependency blocked: {cumulative["dependency_blocked"]:,}'
    )
    safe_texts[0].set_text(
        f'Robot safety blocks: {cumulative["robot_safety_blocks"]}'
    )
    safe_texts[1].set_text(
        f'Obstacle safety blocks: {cumulative["obstacle_safety_blocks"]}'
    )

    # Chart synchronized to current cycle
    chart_indices = np.linspace(0, idx, min(200, idx + 1), dtype=int)
    chart_indices = np.unique(chart_indices)
    xs = [timeline[j]["cycle"] for j in chart_indices]
    blocked = [timeline[j]["cumulative"]["blocked"] for j in chart_indices]
    moved = [timeline[j]["cumulative"]["moved"] for j in chart_indices]
    blocked_line.set_data(xs, blocked)
    moved_line.set_data(xs, moved)

    # Event log
    ev = h["events"]
    event_text.set_text(
        f'Cycle {h["cycle"]}: {ev["blocked"]} blocked proposals | '
        f'{ev["destination_conflicts"]} destination conflicts | '
        f'{ev["robot_safety_blocks"]} safety rejections'
    )
    time_text.set_text(
        f'Simulation time: {h["simulation_time"]:.2f} s | '
        f'Goals reached: {goals}/10'
    )

    percent = h["cycle"] / len(timeline) * 100
    filled = round(percent / 5)
    progress_text.set_text(
        f'REPLAY  [{"=" * filled}{"-" * (20-filled)}]  '
        f'{h["cycle"]}/2000  ({percent:.1f}%)'
    )

    if h["cycle"] <= 93:
        phase_text.set_text("PHASE: INITIAL MOVEMENT")
    else:
        phase_text.set_text(
            "PHASE: POSITION STAGNATION — no position changes after cycle 93"
        )

    return []

ani = animation.FuncAnimation(
    fig, update, frames=FRAMES, interval=1000/FPS,
    blit=False, cache_frame_data=False
)

print(f"Rendering {FRAMES} frames | {FPS} FPS | {DURATION}s", flush=True)
print("Output:", OUTPUT, flush=True)

writer = animation.FFMpegWriter(
    fps=FPS,
    codec="h264",
    bitrate=6500,
    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
)

ani.save(OUTPUT, writer=writer, dpi=DPI)
plt.close(fig)

print("\nVIDEO GENERATED SUCCESSFULLY")
print("File:", OUTPUT)
print("Resolution: 2560x1440")
print("Duration: 6 min 30 sec")
print("Format: MP4 H.264")
print("Size:", round(OUTPUT.stat().st_size / (1024*1024), 2), "MB")
