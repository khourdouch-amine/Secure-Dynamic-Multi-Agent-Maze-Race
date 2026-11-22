import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "dashboard/data/dashboard_data.json"
OUT = ROOT / "dashboard/scenes"
OUT.mkdir(parents=True, exist_ok=True)

data = json.loads(DATA_FILE.read_text())
static = data["static_10_agent_benchmark"]
dynamic = data["dynamic_2_agent_benchmark"]
agents = static["agents"]
manager = static["manager_metrics"]

BG = "#07111F"
PANEL = "#101F32"
CYAN = "#35D6E8"
ORANGE = "#FF9D45"
WHITE = "#F1F5F9"
MUTED = "#A8B7C8"
GREEN = "#43D19E"
RED = "#FF647C"
PURPLE = "#A88BFA"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#263A50",
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": WHITE,
    "font.family": "DejaVu Sans",
})

def canvas(title, subtitle, number):
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.text(0.025, 0.93, title, fontsize=40, weight="bold", color=WHITE)
    fig.text(0.027, 0.885, subtitle, fontsize=16, color=MUTED)
    fig.text(0.975, 0.94, f"{number:02d} / 07", ha="right",
             fontsize=17, color=CYAN)
    fig.text(0.025, 0.035, "SECURE DYNAMIC MULTI-AGENT MAZE RACE",
             fontsize=15, color=CYAN, weight="bold")
    fig.text(0.975, 0.035, "SCIENTIFIC TECHNICAL DASHBOARD",
             fontsize=14, color=MUTED, ha="right")
    return fig

def panel(fig, x, y, w, h, title):
    scale = 0.95 / 0.88
    x = 0.025 + (x - 0.06) * scale
    w = w * scale
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(PANEL)
    ax.set_title(title, loc="left", color=CYAN, fontsize=18, pad=12)
    for s in ax.spines.values():
        s.set_color("#263A50")
    return ax

def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    print("Created:", path)

# 01 — Project introduction
fig = canvas("SECURE DYNAMIC MULTI-AGENT MAZE RACE",
             "A benchmark of navigation, coordination and operational safety", 1)
fig.text(0.07, 0.70, "10", fontsize=95, weight="bold", color=CYAN)
fig.text(0.075, 0.63, "NAVIGATION AGENTS", fontsize=20, color=WHITE)
fig.text(0.38, 0.70, "02", fontsize=95, weight="bold", color=ORANGE)
fig.text(0.385, 0.63, "ALGORITHM FAMILIES", fontsize=20, color=WHITE)
fig.text(0.69, 0.70, "01", fontsize=95, weight="bold", color=GREEN)
fig.text(0.695, 0.63, "SHARED ENVIRONMENT", fontsize=20, color=WHITE)
ax = panel(fig, 0.07, 0.17, 0.86, 0.32, "RESEARCH OBJECTIVE")
ax.axis("off")
ax.text(0.04, 0.72, "Evaluate mission completion under multi-agent constraints.",
        fontsize=26, weight="bold", transform=ax.transAxes)
ax.text(0.04, 0.43, "Measure operational conditions, coordination conflicts and safety filtering.",
        fontsize=20, color=MUTED, transform=ax.transAxes)
ax.text(0.04, 0.16, "Compare classical planners with hybrid quantum-classical approaches.",
        fontsize=20, color=MUTED, transform=ax.transAxes)
save(fig, "01_introduction.png")

# 02 — System architecture
fig = canvas("SYSTEM ARCHITECTURE",
             "Environment → agents → coordination manager → safety filtering → execution", 2)
ax = panel(fig, 0.06, 0.18, 0.88, 0.62, "MULTI-AGENT CONTROL PIPELINE")
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
blocks = [
    (3, 20, 18, 25, "MAZE ENVIRONMENT", "25 × 25 grid\nSeed 42"),
    (27, 20, 18, 25, "10 NAVIGATION AGENTS", "5 classical\n5 hybrid quantum-classical"),
    (51, 20, 18, 25, "COORDINATION", "Proposals\nConflict resolution"),
    (75, 20, 21, 25, "SAFETY FILTER", "Physical checks\nAllowed movements"),
]
for x, y, w, h, title, body in blocks:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                         facecolor="#142A40", edgecolor=CYAN, linewidth=1.6)
    ax.add_patch(box)
    ax.text(x+w/2, y+h*0.68, title, ha="center", fontsize=17,
            weight="bold", color=CYAN)
    ax.text(x+w/2, y+h*0.35, body, ha="center", va="center",
            fontsize=15, color=WHITE)
for x in (22, 46, 70):
    ax.annotate("", xy=(x+4, 32), xytext=(x, 32),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
ax.text(50, 7, "Movement execution and state update", ha="center",
        fontsize=18, color=GREEN)
save(fig, "02_architecture.png")

# 03 — Ten configured algorithms
fig = canvas("THE TEN NAVIGATION ALGORITHMS",
             "Configured planner mapping from the benchmark source code", 3)
ax = panel(fig, 0.06, 0.12, 0.88, 0.70, "ROBOT / PLANNER / FAMILY")
ax.axis("off")
names = list(data["algorithm_mapping"].items())
for i, (rid, algorithm) in enumerate(names):
    y = 0.91 - i * 0.092
    family = "CLASSICAL" if i < 5 else "HYBRID QUANTUM–CLASSICAL"
    color = CYAN if i < 5 else ORANGE
    ax.text(0.04, y, rid, fontsize=18, weight="bold", color=color,
            transform=ax.transAxes, va="center")
    ax.text(0.18, y, algorithm, fontsize=18, color=WHITE,
            transform=ax.transAxes, va="center")
    ax.text(0.97, y, family, fontsize=14, color=color,
            transform=ax.transAxes, va="center", ha="right")
    ax.plot([0.03, 0.97], [y-0.043, y-0.043], color="#263A50",
            lw=0.7, transform=ax.transAxes)
save(fig, "03_algorithms.png")

# 04 — Static benchmark outcome
fig = canvas("STATIC BENCHMARK — MISSION OUTCOME",
             "10 agents · 25 × 25 maze · seed 42 · 2,000 cycles · no moving obstacles", 4)
ax = panel(fig, 0.06, 0.20, 0.40, 0.57, "MISSION COMPLETION")
ax.bar(["Completed", "Not completed"],
       [static["completed"], len(agents)-static["completed"]],
       color=[GREEN, RED], width=0.55)
ax.set_ylabel("Number of agents")
ax.set_ylim(0, 10)
for i, value in enumerate([static["completed"], len(agents)-static["completed"]]):
    ax.text(i, value+0.2, str(value), ha="center", fontsize=27, weight="bold")
ax2 = panel(fig, 0.52, 0.20, 0.42, 0.57, "INDIVIDUAL MISSION STATUS")
ax2.axis("off")
for i, (rid, record) in enumerate(agents.items()):
    y = 0.91 - i * 0.09
    ok = record["reached_goal"]
    ax2.text(0.05, y, rid, color=CYAN if record["family"] == "classical" else ORANGE,
             weight="bold", transform=ax2.transAxes)
    ax2.text(0.28, y, "GOAL REACHED" if ok else "NOT REACHED",
             color=GREEN if ok else RED, transform=ax2.transAxes)
    ax2.text(0.96, y, f'{record["metrics"].get("moves", 0)} moves',
             color=WHITE, ha="right", transform=ax2.transAxes)
fig.text(0.08, 0.12, "Observed result: 3 / 10 missions completed (30%).",
         fontsize=22, color=ORANGE, weight="bold")
save(fig, "04_mission_results.png")

# 05 — Coordination and manager metrics
fig = canvas("MULTI-AGENT COORDINATION",
             "Manager-level counts recorded during the 10-agent static benchmark", 5)
labels = ["Successful\nmoves", "Blocked\nmoves", "Destination\nconflicts",
          "Cycle\nrejections", "Dependency\nblocked", "Robot safety\nblocks"]
keys = ["successful_moves", "blocked_moves", "destination_conflicts",
        "cycle_rejections", "dependency_blocked", "robot_safety_blocks"]
values = [manager.get(k, 0) for k in keys]
ax = panel(fig, 0.07, 0.18, 0.86, 0.60, "RECORDED MANAGER EVENTS")
bars = ax.bar(labels, values, color=[GREEN, ORANGE, RED, PURPLE, CYAN, "#E5B85C"])
ax.set_ylabel("Event count")
ax.tick_params(axis="x", labelsize=9)
for bar, value in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values)*0.015,
            f"{value:,}", ha="center", fontsize=17, color=WHITE)
save(fig, "05_coordination.png")

# 06 — Safety metrics
fig = canvas("OPERATIONAL SAFETY MONITORING",
             "Distinguishing proposed conflicts, manager filtering and observed collisions", 6)
ax = panel(fig, 0.07, 0.20, 0.40, 0.55, "SAFETY FILTER COUNTS")
labels = ["Robot–robot\nsafety blocks", "Obstacle\nsafety blocks", "Invalid\nproposals"]
values = [manager.get("robot_safety_blocks", 0),
          manager.get("obstacle_safety_blocks", 0),
          manager.get("invalid_proposals", 0)]
bars = ax.bar(labels, values, color=[ORANGE, CYAN, PURPLE])
ax.set_ylabel("Recorded events")
for bar, value in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            str(value), ha="center", color=WHITE, weight="bold")
ax2 = panel(fig, 0.53, 0.20, 0.40, 0.55, "INTERPRETATION")
ax2.axis("off")
notes = [
    "23 robot–robot safety blocks recorded.",
    "0 obstacle safety blocks recorded.",
    "0 invalid proposals recorded.",
    "These are manager event counts, not a universal safety guarantee.",
]
for i, note in enumerate(notes):
    ax2.text(0.05, 0.86-i*0.20, "• " + note, fontsize=17,
             color=WHITE, transform=ax2.transAxes, wrap=True)
save(fig, "06_safety.png")

# 07 — Quantum / hybrid execution evidence and limitations
fig = canvas("QUANTUM-INSPIRED NAVIGATION — EVIDENCE & LIMITATIONS",
             "Simulation-based hybrid methods: quantum circuit activity is not quantum hardware execution", 7)
ax = panel(fig, 0.06, 0.18, 0.42, 0.58, "BASELINE RECORDS — SIX MEASUREMENTS")
ax.axis("off")
baseline = data["baseline_results"]["records"]
for i, row in enumerate(baseline):
    y = 0.90 - i*0.145
    label = row.get("algorithm", "Unknown")
    family = row.get("ai_type", "")
    latency = row.get("planning_latency_s", 0)
    ax.text(0.03, y, f"{i+1}. {label}", fontsize=15, color=WHITE,
            transform=ax.transAxes)
    ax.text(0.97, y, f"{latency:.4f} s", fontsize=15, color=CYAN,
            ha="right", transform=ax.transAxes)
    ax.text(0.05, y-0.055, family, fontsize=14, color=MUTED,
            transform=ax.transAxes)
ax2 = panel(fig, 0.53, 0.18, 0.41, 0.58, "SCIENTIFIC QUALIFICATIONS")
ax2.axis("off")
notes = [
    "R06–R10 are configured as hybrid quantum-classical agents.",
    "Circuits are simulated; no quantum hardware execution is established.",
    "Baseline contains repeated A* and Quantum Walk records.",
    "Ten-agent quantum metadata repeats one planner label; configured labels come from source mapping.",
    "No quantum speedup claim is supported by these results.",
]
for i, note in enumerate(notes):
    ax2.text(0.04, 0.92-i*0.18, "• " + note, fontsize=15,
             color=WHITE, transform=ax2.transAxes, wrap=True)
save(fig, "07_quantum_methodology.png")

print("\nAll seven dashboard scenes generated.")
print("Output directory:", OUT)
