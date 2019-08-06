"""Main file for executing pytest-benchmark"""

import pytest
import numpy as np
import cirq

np.random.seed(31415926)


TRIAL_RUNS = [
    (4, 10),
    (5, 10),
    (6, 10),
]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TYPE-ZERO TESTS: ALL CIRCUIT OPERATIONS HAVE A SPECIALIZED UNITARY OPERATION
# THAT ALLOWS EFFICIENT UNITARY ACTION DURING CIRCUIT SIMULATION

OPS_LIST_0 = [
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.I,
    cirq.H,
    cirq.CZ,
    cirq.CNOT,
    cirq.SWAP,
    cirq.ISWAP,
]

def _generator_type_zero(n_qubits, depth):
    """Construct a (possibly dense) circuit from OPS_LIST_0."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []
    for layer in range(depth):

        m = int(np.random.randint(2, n_qubits+1))
        # an exclusive list of m qubits with index in (0, n_qubits)
        qubits_this_layer = np.random.choice(np.arange(n_qubits), size=m, replace=False)
        qubits_this_layer = [qubits[i] for i in qubits_this_layer]

        # a set of initialized gates to apply this layer
        gates_this_layer = np.random.randint(0, high=len(OPS_LIST_0), size=m)
        initialized = []

        for i, j in zip(gates_this_layer, range(m)):
            q0 = qubits_this_layer[j]
            q1 = qubits_this_layer[(j+1)%m]
            # two-qubit on `nearest neighbor`
            try:
                initialized.append(OPS_LIST_0[i](q0, q1 ))
            except:
                initialized.append(OPS_LIST_0[i](q0))

        ops += initialized
    return cirq.Circuit.from_ops(ops)


@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_specialized_unitary_fallback(benchmark, n_qubits, depth):
    """
    Perform a circuit simulation that uses purely `specialized` operations
    for computing the output state. This is the first type of operation that
    is tried by cirq.protocols.apply_unitary. It typically consists of applying
    a permutation and eigenvalue-based negation on a specific qubit subspace.
    Relevant ops:
        cirq.X (cirq.XPowGate(1))
        cirq.Y (cirq.YPowGate(1))
        cirq.Z (cirq.ZPowGate(1))
        cirq.I
        cirq.H
        cirq.CZ (cirq.CZPowGate(1))
        cirq.CNOT
        cirq.SWAP
        cirq.ISWAP
    (For this gateset minus H, it is obvious that matrix permutation is more
    efficient than einsum)
    """
    result = benchmark(_cirq_to_cirq_type_zero, n_qubits, depth)
