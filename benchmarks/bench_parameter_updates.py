import pytest
import sympy
import numpy as np
import cirq

import sys
sys.path.insert(0, ".")
from helpers import _CirqTPU, _TFCirq, _TFQEigen, _Cirq
np.random.seed(31415926)


# pairs like (number of qubits, depth)
@pytest.fixture(scope="module")
def meta():
    """Shared circuit and parameter data across tests."""
    depth = 50
    qubits = cirq.LineQubit.range(10)
    n_loops = 10
    return {"depth": depth,
            "qubits": qubits,
            "n_loops": n_loops,
            "params": np.random.randn(n_loops, len(qubits) * depth)}


def get_parametrized_two_qubit_gates():
    return [
        cirq.SwapPowGate,
        cirq.CNotPowGate,
        cirq.ISwapPowGate,
        cirq.ZZPowGate,
    ]


def get_parametrized_single_qubit_gates():
    return [
        cirq.Rx,
        cirq.Ry,
        cirq.Rz,
    ]


ALL_GATES = get_parametrized_single_qubit_gates() + \
            get_parametrized_two_qubit_gates()


def random_single_qubit_gates_layer(qubits, params):
    """Compose a list of random one-qubit parametrized gates."""
    rand_oneq = np.random.choice(
        get_parametrized_single_qubit_gates(), size=len(qubits), replace=True)
    return [g(e)(q) for g,q,e in zip(rand_oneq, qubits, params)]


# def random_two_qubit_gates_layer(qubits, params):
#     """Compose a list of random two-qubit parametrized gates."""
#     n = len(qubits)
#     qubit_pairs = ((qubits[i], qubits[(i+1)%n]) for i in range(n))
#     rand_twoq = np.random.choice(
#         get_parametrized_two_qubit_gates(), size=n, replace=True)
#     return [g(exponent=e)(qi,qj) for g,(qi,qj),e in zip(rand_twoq, qubit_pairs, params)]


def _generator_type_zero(depth, qubits, params):
    """Initialize a circuit according to a set of parameters.

        Uses only parametrized single-qubit gates.

        Args:
            `params` should be shape (n_qubits * depth, ) - it is a flattened
            array of parameters on the qubits x depth grid.

        Returns:
            cirq.Circuit
        returns a nested list of ops (OP_TREE)
    """
    out = []
    n_qubits = len(qubits)
    for d in range(depth):
        # slice params into layer-sized chunks
        param_subset = params[n_qubits*d:n_qubits*(d+1)]
        out.append(random_single_qubit_gates_layer(qubits, param_subset))
        # FIXME: _with_exponent doesn't work as expected with 2-qubit gates..
        # if not (d % 2):
        #
        # out.append(random_single_qubit_gates_layer(qubits, param_subset))
        # else:
        #     out.append(random_two_qubit_gates_layer(qubits, param_subset))
    return cirq.Circuit.from_ops(out)


@pytest.mark.parametrize('helper', [_TFCirq, _Cirq])
def test_single_qubit_parametrized_n_loops(benchmark, meta, helper):
    """
    Benchmark description:

    For number of parameters N_PARAMS, construct a circuit of mixed one- and
    two-qubit parametrized gates of total depth DEPTH. Compare the following
    param resolution methods:
        (a) Sympy variables with native cirq ParamResolver feed dict
        (b) Circuit reconstructed directly from new parameters
    """
    inst = helper(meta)
    c = _generator_type_zero(meta["depth"], meta["qubits"], meta["params"][0])

    params = inst.prepare_parameters(meta["params"][0])
    cprime = helper.prepare(c)

    def n_loops(meta, inst, cprime, params):
        """Global scope outer-loop."""
        for k, params in zip(range(meta["n_loops"]), meta["params"]):
            # skip validation, come back to it later
            # results[k] = inst.updated_execute(cprime, params)
            inst.updated_execute(cprime, params)

    #  TODO: implement initial_state
    # x = np.ones(2**n_qubits, dtype=np.complex64) / np.sqrt(2**n_qubits)
    # check = np.random.randint(2 ** len(meta["qubits"]))
    # results = np.zeros(meta["n_loops"])

    benchmark(n_loops, meta, inst, cprime, params)

if __name__ == "__main__":
    depth = 50
    qubits = cirq.LineQubit.range(10)
    n_loops = 10
    meta = {"depth": depth,
            "qubits": qubits,
            "n_loops": n_loops,
            "params": np.random.randn(n_loops, len(qubits) * depth)}
