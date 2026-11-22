"""Reproducible scenarios for multi-scale classical/quantum benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]
    route_length: int
    description: str


SCENARIOS = (
    Scenario(
        name="G1_3x3_L4",
        width=3,
        height=3,
        start=(0, 0),
        goal=(2, 2),
        obstacles=frozenset({(1, 0)}),
        route_length=4,
        description="Reference scenario: compact grid, four-step routes.",
    ),
    Scenario(
        name="G2_4x4_L4",
        width=4,
        height=4,
        start=(0, 0),
        goal=(2, 2),
        obstacles=frozenset({(1, 0), (2, 1)}),
        route_length=4,
        description="Larger grid with the same route search-space size as G1.",
    ),
    Scenario(
        name="G3_4x4_L6",
        width=4,
        height=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=frozenset({(1, 0), (2, 1)}),
        route_length=6,
        description="Longer routes and a larger Grover search space.",
    ),
)


def validate_scenario(scenario):
    """Validate geometry and exact-length route feasibility."""
    from planners.quantum.grover_global.maze_model import GroverMazeModel
    from planners.quantum.grover_global.classical_oracle import GlobalClassicalOracle

    model = GroverMazeModel(
        width=scenario.width,
        height=scenario.height,
        start=scenario.start,
        goal=scenario.goal,
        obstacles=scenario.obstacles,
    )

    oracle = GlobalClassicalOracle(model, scenario.route_length)
    valid_indices = oracle.find_valid_indices()

    return {
        "name": scenario.name,
        "search_space": 4 ** scenario.route_length,
        "route_length": scenario.route_length,
        "valid_routes": len(valid_indices),
        "feasible": len(valid_indices) > 0,
    }


if __name__ == "__main__":
    for scenario in SCENARIOS:
        result = validate_scenario(scenario)
        print(result)
