"""Lookup-table phase oracle for Grover validation."""

from qiskit import QuantumCircuit


def build_lookup_phase_oracle(route_length, marked_indices):
    if route_length < 1:
        raise ValueError("route_length must be positive.")

    n = 2 * route_length
    marked_indices = sorted(set(marked_indices))

    if not marked_indices:
        raise ValueError("At least one marked state is required.")

    if any(i < 0 or i >= 2**n for i in marked_indices):
        raise ValueError("Marked index outside the search space.")

    qc = QuantumCircuit(n + 1, name="LookupPhaseOracle")
    phase = n

    for index in marked_indices:
        for bit in range(n):
            if ((index >> bit) & 1) == 0:
                qc.x(bit)

        qc.mcx(list(range(n)), phase)

        for bit in range(n):
            if ((index >> bit) & 1) == 0:
                qc.x(bit)

    return qc
