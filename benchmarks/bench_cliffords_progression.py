"""Benchmark on progressively more difficult unitary structures."""

import pytest
import numpy as np

import cirq
import sys
sys.path.insert(0, ".")
from helpers import SimCirqTPU, SimTFCirq, SimTFQEigen, SimCirq

np.random.seed(31415926)

# pairs like (number of qubits, depth)
TRIAL_RUNS = [
    (4, 10),
    (5, 10),
    (6, 10),
]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CLIFFORD CIRCUIT TESTS: ALL CIRCUIT OPERATIONS HAVE A SPECIALIZED UNITARY
# OPERATION THAT ALLOWS EFFICIENT UNITARY ACTION DURING CIRCUIT SIMULATION
OPS_LIST_0 = [
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.I,
    cirq.H,
    cirq.CZ,
    cirq.CNOT,
    cirq.SWAP,
]

def _clifford_circuit_generator(n_qubits, depth):
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


@pytest.mark.parametrize('helper', [SimCirqTPU, SimTFCirq, SimTFQEigen, SimCirq])
@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_clifford_circuit_simulation(benchmark, helper, n_qubits, depth):
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
    (For this gateset minus H, it is obvious that matrix permutation is more
    efficient than einsum)
    """
    target = _clifford_circuit_generator(n_qubits, depth)
    setup = helper.prepare(target)
    result = benchmark(helper.execute(setup))



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TYPE-ONE TESTS: ALL CIRCUIT OPERATIONS HAVE A SUBSPACE MULTIPLICATION METHOD


OPS_LIST_1 = [
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.H,
]


def _generator_no_cliffords_1(n_qubits, depth):
    """Construct a (possibly dense) circuit from OPS_LIST_1."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []
    for layer in range(depth):

        m = int(np.random.randint(2, n_qubits+1))
        # an exclusive list of m qubits with index in (0, n_qubits)
        qubits_this_layer = np.random.choice(np.arange(n_qubits), size=m, replace=False)
        qubits_this_layer = [qubits[i] for i in qubits_this_layer]

        # a set of exponents not equal to one; doesn't matter what they are
        # for efficient computation
        exponents_this_layer = np.abs(np.random.randn(m))
        for k, v in enumerate(exponents_this_layer):
            if np.isclose(v, 1):
                exponents_this_layer[k] = v + 0.01
        # a set of initialized gates to apply this layer
        gates_this_layer = np.random.randint(0, high=len(OPS_LIST_1), size=m)
        gates_this_layer = [
            OPS_LIST_1[i](j)**k for i, j, k in zip(
                gates_this_layer, qubits_this_layer, exponents_this_layer)
        ]

        ops += gates_this_layer
    return cirq.Circuit.from_ops(ops)


@pytest.mark.parametrize('helper', [SimCirqTPU, SimTFCirq, SimTFQEigen, SimCirq])
@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_no_clifford_type_1_simulation(benchmark, helper, n_qubits, depth):
    """
    Perform a circuit simulation that uses purely single-qubit gates with
    efficient multiplication via cirq.linalg.apply_matrix_to_slices but do not
    define `_apply_unitary_`.
    Relevant ops:
        cirq.XPowGate(!1)
        cirq.YPowGate(!1)
        cirq.ZPowGate(!1)
        cirq.H ** (!1)

    """
    target = _generator_no_cliffords_1(n_qubits, depth)
    setup = helper.prepare(target)
    result = benchmark(helper.execute(setup))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TYPE-TWO TESTS: CIRCUIT IS SIMULATED VIA EINSUM.

# reminder: these gates will be exponentiated
OPS_LIST_2 = [
    cirq.CZ,
    cirq.CNOT,
    cirq.SWAP,
    cirq.ISWAP,
]


def _generator_no_cliffords_2(n_qubits, depth):
    """Construct a (possibly dense) circuit from OPS_LIST_2."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []
    for layer in range(depth):

        m = int(np.random.randint(2, n_qubits+1))
        # an exclusive list of m qubits with index in (0, n_qubits)
        # two-qubit gate may reach beyond this list.
        qubits_this_layer = np.random.choice(
            np.arange(n_qubits), size=m, replace=False)
        qubits_this_layer = [qubits[i] for i in qubits_this_layer]

        # a set of exponents not equal to one; doesn't matter what they are
        # for efficient computation
        exponents_this_layer = np.abs(np.random.randn(m))
        for k, v in enumerate(exponents_this_layer):
            if np.isclose(v, 1):
                exponents_this_layer[k] = v + 0.01
        # a set of initialized gates to apply this layer
        gates_this_layer = np.random.randint(0, high=len(OPS_LIST_2), size=m)
        gates_this_layer = [
            OPS_LIST_2[i](qubits_this_layer[j], qubits_this_layer[(j+1) % m])**k
            for i, j, k in zip(gates_this_layer, range(m), exponents_this_layer)
        ]

        ops += gates_this_layer
    return cirq.Circuit.from_ops(ops)


@pytest.mark.parametrize('helper', [SimCirqTPU, SimTFCirq, SimTFQEigen, SimCirq])
@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_no_clifford_type_2_simulation(benchmark, helper, n_qubits, depth):
    """
    Perform a circuit siulation that uses purely two-qubit gates subject to
    cirq.linalg.targeted_left_multiply (wrapper for np.einsum) but do NOT
    define `_apply_unitary_`
    relevant ops:
        cirq.CZPowGate(!1)
        cirq.CNOT ** (!1)
        cirq.SWAP ** (!1)
        cirq.ISWAP ** (!1)
    """
    target = _generator_no_cliffords_2(n_qubits, depth)
    setup = helper.prepare(target)
    result = benchmark(helper.execute(setup))
