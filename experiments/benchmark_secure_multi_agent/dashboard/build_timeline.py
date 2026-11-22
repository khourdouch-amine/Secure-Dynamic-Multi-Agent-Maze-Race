import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
source = ROOT / "results/ten_agents_static_seed42.json"
output = ROOT / "dashboard/data/timeline.json"

data = json.loads(source.read_text())
history = data["history"]

event_keys = [
    "moved", "blocked", "safety_blocked", "replanned",
    "invalid_proposals", "destination_conflicts",
    "cycle_rejections", "dependency_blocked",
    "obstacle_safety_blocks", "robot_safety_blocks"
]

cumulative = {key: 0 for key in event_keys}
timeline = []

for h in history:
    events = {}
    for key in event_keys:
        value = int(h.get(key, 0) or 0)
        events[key] = value
        cumulative[key] += value

    timeline.append({
        "cycle": h["cycle"],
        "simulation_time": h["simulation_time"],
        "positions": h["positions"],
        "positions_before": h["positions_before"],
        "proposals": h["proposals"],
        "agent_states": h["agent_states"],
        "events": events,
        "cumulative": dict(cumulative),
    })

result = {
    "experiment": "ten_agents_static_seed42",
    "total_cycles": len(timeline),
    "simulation_duration_s": history[-1]["simulation_time"],
    "robots": list(history[0]["positions"].keys()),
    "timeline": timeline,
    "final_totals": cumulative,
}

output.write_text(json.dumps(result, separators=(",", ":")))

print("Timeline generated:", output)
print("Cycles:", len(timeline))
print("Robots:", len(result["robots"]))
print("Duration:", round(result["simulation_duration_s"], 3), "s")
print("Final cumulative counters:")
for key, value in cumulative.items():
    print(f"  {key}: {value}")
print("File size:", round(output.stat().st_size / 1024, 1), "KB")
