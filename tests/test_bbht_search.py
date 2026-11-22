import pytest

from planners.quantum.grover_global.bbht_search import (
    build_geometric_grover_trial,
    search_geometric_route_bbht,
)


def test_build_geometric_grover_trial_dimensions():
    circuit = build_geometric_grover_trial(
        width=3,
        height=3,
        start=(0, 0),
        goal=(2, 0),
        obstacles=(),
        route_length=2,
        iterations=1,
    )

    assert circuit.num_qubits == 12
    assert circuit.num_clbits == 4
    assert circuit.name == "GeometricGroverTrial"


def test_bbht_finds_valid_route_with_amplification():
    result = search_geometric_route_bbht(
        width=3,
        height=3,
        start=(0, 0),
        goal=(2, 0),
        obstacles=(),
        route_length=2,
        max_rounds=20,
        seed=0,
    )

    assert result.found is True
    assert result.status == "found"
    assert result.route == (0, 0)
    assert result.path == ((0, 0), (1, 0), (2, 0))
    assert result.total_grover_iterations > 0
    assert result.route_index == 0


def test_bbht_reports_budget_exhaustion_without_solution():
    result = search_geometric_route_bbht(
        width=3,
        height=3,
        start=(0, 0),
        goal=(2, 2),
        obstacles=(),
        route_length=2,
        max_rounds=5,
        seed=42,
    )

    assert result.found is False
    assert result.status == "not_found_within_budget"
    assert result.route is None
    assert result.path is None
    assert result.rounds == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"route_length": 0},
        {"route_length": True},
        {"max_rounds": 0},
        {"max_rounds": True},
        {"growth_factor": 1.0},
        {"seed": True},
    ],
)
def test_bbht_rejects_invalid_parameters(kwargs):
    parameters = {
        "width": 3,
        "height": 3,
        "start": (0, 0),
        "goal": (2, 0),
        "obstacles": (),
        "route_length": 2,
        "max_rounds": 10,
        "growth_factor": 1.2,
        "seed": 42,
    }
    parameters.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        search_geometric_route_bbht(**parameters)


def test_bbht_is_reproducible_with_same_seed():
    parameters = {
        "width": 3,
        "height": 3,
        "start": (0, 0),
        "goal": (2, 0),
        "obstacles": (),
        "route_length": 2,
        "max_rounds": 20,
        "seed": 0,
    }

    first = search_geometric_route_bbht(**parameters)
    second = search_geometric_route_bbht(**parameters)

    assert first.route == second.route
    assert first.path == second.path
    assert first.iteration_history == second.iteration_history
    assert first.total_grover_iterations == second.total_grover_iterations
