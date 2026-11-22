"""Grover search using the reversible geometric maze oracle."""

from math import asin, pi, sqrt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .geometric_oracle import build_geometric_phase_oracle


def optimal_iterations(search_space, solutions):
    if not 0 < solutions <= search_space:
        raise ValueError("Invalid number of solutions.")
    theta = asin(sqrt(solutions / search_space))
    return max(0, round(pi / (4 * theta) - 0.5))


def build_geometric_grover(
    width, height, start, goal, obstacles, route_length,
    iterations=None,
):
    n = 2 * route_length
    search_space = 2**n

    # Small classical enumeration is used only to estimate
    # the Grover iteration count, never to construct the oracle.
    from .maze_model import GroverMazeModel
    from .classical_oracle import GlobalClassicalOracle

    model = GroverMazeModel(width, height, start, goal, obstacles)
    reference = GlobalClassicalOracle(model, route_length)
    solutions = reference.count_valid_routes()

    if solutions == 0:
        raise ValueError("No valid routes exist.")

    if iterations is None:
        iterations = optimal_iterations(search_space, solutions)

    route = QuantumRegister(n, "route")
    x = QuantumRegister(max(1, width.bit_length()), "x")
    y = QuantumRegister(max(1, height.bit_length()), "y")
    bad = QuantumRegister(route_length, "bad")
    goal_flag = QuantumRegister(1, "goal")
    phase = QuantumRegister(1, "phase")
    classical = ClassicalRegister(n, "result")

    qc = QuantumCircuit(
        route, x, y, bad, goal_flag, phase, classical,
        name="GeometricGrover",
    )

    qc.x(phase[0])
    qc.h(phase[0])
    qc.h(route)

    oracle = build_geometric_phase_oracle(
        width, height, start, goal, obstacles, route_length
    )

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)

        qc.h(route)
        qc.x(route)
        qc.h(route[-1])
        qc.mcx(list(route[:-1]), route[-1])
        qc.h(route[-1])
        qc.x(route)
        qc.h(route)

    qc.measure(route, classical)

    return qc, iterations, solutions
