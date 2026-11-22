"""BBHT quantum search for fixed-length geometric maze routes.

The quantum oracle marks valid routes reversibly. Classical processing is
restricted to BBHT iteration scheduling and verification of measured candidates.
No classical enumeration or solution counting is performed.
"""

from dataclasses import dataclass
from math import ceil, sqrt
import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

from planners.quantum.route_encoding import RouteEncoding
from .geometric_oracle import build_geometric_phase_oracle
from .maze_model import GroverMazeModel


@dataclass(frozen=True)
class BBHTResult:
    found: bool
    route: tuple | None
    path: tuple | None
    route_index: int | None
    rounds: int
    total_grover_iterations: int
    iteration_history: tuple
    status: str


def build_geometric_grover_trial(
    width,
    height,
    start,
    goal,
    obstacles,
    route_length,
    iterations,
):
    """Build one Grover trial without knowing the number of valid routes."""
    if isinstance(route_length, bool) or not isinstance(route_length, int):
        raise TypeError("route_length must be an integer.")
    if route_length < 1:
        raise ValueError("route_length must be positive.")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError("iterations must be an integer.")
    if iterations < 0:
        raise ValueError("iterations cannot be negative.")

    n = 2 * route_length

    route = QuantumRegister(n, "route")
    x = QuantumRegister(max(1, width.bit_length()), "x")
    y = QuantumRegister(max(1, height.bit_length()), "y")
    bad = QuantumRegister(route_length, "bad")
    goal_flag = QuantumRegister(1, "goal")
    phase = QuantumRegister(1, "phase")
    classical = ClassicalRegister(n, "result")

    qc = QuantumCircuit(
        route, x, y, bad, goal_flag, phase, classical,
        name="GeometricGroverTrial",
    )

    # Prepare phase kickback and a uniform superposition of routes.
    qc.x(phase[0])
    qc.h(phase[0])
    qc.h(route)

    oracle = build_geometric_phase_oracle(
        width, height, start, goal, obstacles, route_length
    )
    oracle_qubits = [
        *route, *x, *y, *bad, *goal_flag, *phase
    ]

    for _ in range(iterations):
        qc.compose(oracle, qubits=oracle_qubits, inplace=True)

        # Grover diffusion over the route register.
        qc.h(route)
        qc.x(route)
        qc.h(route[-1])
        qc.mcx(list(route[:-1]), route[-1])
        qc.h(route[-1])
        qc.x(route)
        qc.h(route)

    qc.measure(route, classical)
    return qc


def search_geometric_route_bbht(
    width,
    height,
    start,
    goal,
    obstacles,
    route_length,
    *,
    max_rounds=64,
    growth_factor=1.2,
    seed=42,
    simulator=None,
    optimization_level=1,
):
    """Search for one valid route using BBHT with an unknown solution count.

    Each round performs one quantum measurement. A measured route is checked
    classically, but the complete search space is never enumerated.

    Failure to find a route within max_rounds is not proof that no route exists.
    """
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int):
        raise TypeError("max_rounds must be an integer.")
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive.")
    if growth_factor <= 1:
        raise ValueError("growth_factor must be greater than 1.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    model = GroverMazeModel(width, height, start, goal, obstacles)

    if isinstance(route_length, bool) or not isinstance(route_length, int):
        raise TypeError("route_length must be an integer.")
    if route_length < 1:
        raise ValueError("route_length must be positive.")

    search_space = 4 ** route_length
    max_m = int(sqrt(search_space))
    m = 1

    rng = random.Random(seed)
    backend = simulator or AerSimulator(method="matrix_product_state")

    history = []
    total_iterations = 0

    for round_index in range(max_rounds):
        # BBHT: choose a random number of Grover iterations in [0, m).
        iteration_count = rng.randrange(m)
        history.append(iteration_count)
        total_iterations += iteration_count

        circuit = build_geometric_grover_trial(
            width,
            height,
            start,
            goal,
            obstacles,
            route_length,
            iteration_count,
        )

        compiled = transpile(
            circuit,
            backend,
            optimization_level=optimization_level,
            seed_transpiler=seed + round_index,
        )

        result = backend.run(
            compiled,
            shots=1,
            seed_simulator=seed + round_index,
        ).result()

        counts = result.get_counts(compiled)
        if not counts:
            raise RuntimeError("The quantum simulator returned no measurement.")

        bitstring = max(counts, key=counts.get)
        route = RouteEncoding.decode_bitstring(bitstring, route_length)
        valid, path, collision_step = model.evaluate_route(route)

        if valid and collision_step is None:
            return BBHTResult(
                found=True,
                route=route,
                path=tuple(path),
                route_index=RouteEncoding.encode(route),
                rounds=round_index + 1,
                total_grover_iterations=total_iterations,
                iteration_history=tuple(history),
                status="found",
            )

        # Increase the BBHT search range, capped at sqrt(N).
        m = min(ceil(growth_factor * m), max_m)

    return BBHTResult(
        found=False,
        route=None,
        path=None,
        route_index=None,
        rounds=max_rounds,
        total_grover_iterations=total_iterations,
        iteration_history=tuple(history),
        status="not_found_within_budget",
    )
