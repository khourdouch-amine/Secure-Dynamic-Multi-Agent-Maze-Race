"""Grover search engine for fixed-length maze routes."""

from math import asin, pi, sqrt

from qiskit import QuantumCircuit


def optimal_grover_iterations(search_space, marked_count):
    if not 0 < marked_count <= search_space:
        raise ValueError("Invalid number of marked states.")

    theta = asin(sqrt(marked_count / search_space))
    return max(0, round(pi / (4 * theta) - 0.5))


def build_grover_circuit(route_length, marked_indices, iterations=None):
    n = 2 * route_length
    search_space = 2**n
    marked_indices = sorted(set(marked_indices))

    if not marked_indices:
        raise ValueError("At least one marked state is required.")

    if any(i < 0 or i >= search_space for i in marked_indices):
        raise ValueError("Marked index outside search space.")

    if iterations is None:
        iterations = optimal_grover_iterations(
            search_space, len(marked_indices)
        )

    if iterations < 0:
        raise ValueError("Iterations cannot be negative.")

    oracle = QuantumCircuit(n + 1)
    phase = n

    for index in marked_indices:
        for bit in range(n):
            if ((index >> bit) & 1) == 0:
                oracle.x(bit)

        oracle.mcx(list(range(n)), phase)

        for bit in range(n):
            if ((index >> bit) & 1) == 0:
                oracle.x(bit)

    qc = QuantumCircuit(n + 1, n, name="GlobalGroverSearch")

    # Uniform superposition and phase-kickback ancilla.
    qc.x(phase)
    qc.h(phase)
    qc.h(range(n))

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)

        # Grover diffusion operator on route qubits.
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))

    qc.measure(range(n), range(n))
    return qc, iterations
