import random

from timeit import default_timer as timer
from scipy import sparse
import numpy as np

import cirq

from tfq.gates import gates
from tfq.backends.google import GoogleBackend







TFQ_GATES = gates.get_single_qubit_gates() + \
            gates.get_parametrized_single_qubit_gates() + \
            gates.get_two_qubit_gates() + \
            gates.get_parametrized_two_qubit_gates()


def random_single_qubit_gates_layer(n_qubits, parametrized=False):
    """Compose dense layer of single-qubit gates."""

    all_gates = gates.get_single_qubit_gates()
    if parametrized:
        all_gates = gates.get_parametrized_single_qubit_gates()

    gates_out = []
    for q1 in range(n_qubits):
        gate_func = random.choice(all_gates)
        rand_param = random.uniform(-np.pi, np.pi)
        gate = gate_func(q1, noise=None)
        if parametrized:
            gate = gate_func(q1, value=rand_param, noise=None)
        gates_out.append(gate)

    return gates_out


def random_two_qubit_gates_layer(n_qubits, parametrized=False):
    """Compose dense layer of single-qubit gates."""

    all_gates = gates.get_two_qubit_gates()
    if parametrized:
        all_gates = gates.get_parametrized_two_qubit_gates()

    gates_out = []
    for q2 in range(1,n_qubits,2):
        gate_func = random.choice(all_gates)
        q1 = q2 - 1
        rand_param = random.uniform(-np.pi, np.pi)
        gate = gate_func(q1, q2, noise=None)
        if parametrized:
            gate = gate_func(q1, q2, value=rand_param, noise=None)
        gates_out.append(gate)

    return gates_out


def gates_to_sparse(gatelist):
    """turn a single-layer list of gates into a sparse matrix."""
    base = gatelist[0].matrix()

    # two-qubit gate layer: this will miss dimensionality of last qubit;
    # additional tensor-up covered by sparse_unitary_matmul
    for g in gatelist[1:]:
        base = sparse.kron(base, g.matrix(), format="coo")
    return base


def sparse_unitary_matmul(op, x):
    """Apply unitary to vector x using sparse matrix multiplication.
    Args:
        op - scipy sparse matrix
        x (np.ndarray): 1D vector
    Returns:
        (np.ndarray): 1D vector, the result of op*x
    """

    while op.shape[0] < x.shape[0]:
        op = sparse.kron(
            op,
            sparse.coo_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])),
            format="coo",
        )
    return op.dot(x)

TRIAL_TYPES = []
# set up two trials:
# 1. interspersed single and two-qubit unparametrized gates
# 2. interspersed single and two-qubit parametrized gates

def trial(depth, n_qubits, parametrized):
    out = []
    for d in range(depth):
        if not (d % 2):
            out.append(random_single_qubit_gates_layer(n_qubits, parametrized))
        else:
            out.append(random_two_qubit_gates_layer(n_qubits, False))
    return out


def timeit_n_rounds(n_qubits, layers, sim_trials):
    cirqtimes = []
    tfqtimes = []

    for _ in range(sim_trials):
        # initial state prep
        x = np.zeros((2**n_qubits, ), dtype=np.complex64)
        x[np.random.randint(2**n_qubits)] = 1

        # prepare cirq apply unitary routine on flattened gatelist
        cirqgates = GoogleBackend()._gates_to_native([g for l in layers for g in l])
        start = timer()
        out = cirq.Simulator().simulate(cirqgates, initial_state=np.copy(x))
        end = timer()
        trial = end - start
        cirqtimes.append(trial)
        cirqfinal = out.final_state
        # print("cirq", trial)

        # 'basic' simulator runtime includes time taken composing sparse matrices
        tfqfinal = np.copy(x)
        start = timer()
        for l in layers:
            tfqfinal = sparse_unitary_matmul(gates_to_sparse(l), tfqfinal)
        end = timer()
        trial = end - start
        # print("tfq", trial)
        tfqtimes.append(trial)
        try:
            np.testing.assert_array_almost_equal(tfqfinal, cirqfinal)
        except AssertionError:
            print("ERROR: results are not equivalent. Discard this trial.")
            print(layers)
    return np.asarray(cirqtimes), np.asarray(tfqtimes)


SIM_TRIALS = 10  # average runtime over this many runs
DEPTH = 10  # depth of dense matrix layers
MAX_N = 16  # run trials for n_qubits = 2...MAX_N
QUBITS = range(2,MAX_N)
avg_cirq = []
avg_tfq = []
avg_cirq_param = []
avg_tfq_param = []
for n_qubits in QUBITS:
    print("simulating {} qubits".format(n_qubits))
    no_param = trial(DEPTH, n_qubits, False)
    c1, t1 = timeit_n_rounds(n_qubits, no_param, SIM_TRIALS)
    avg_cirq.append(np.mean(c1))
    avg_tfq.append(np.mean(t1))

    param = trial(DEPTH, n_qubits, True)
    c2, t2 = timeit_n_rounds(n_qubits, param, SIM_TRIALS)
    avg_cirq_param.append(np.mean(c2))
    avg_tfq_param.append(np.mean(t2))


import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].semilogy(QUBITS, avg_cirq, QUBITS, avg_tfq,  )
axes[0].set_yscale('log')
axes[0].set_xlabel("number of qubits")
axes[0].set_ylabel("avg runtime over {} trials".format(SIM_TRIALS))
axes[0].set_title("depth={}, parametrized=False".format(DEPTH))
axes[0].legend(["cirq simulator", "sparse matrix simulator"])

axes[1].semilogy(QUBITS, avg_cirq_param, QUBITS, avg_tfq_param,  )
axes[1].set_yscale('log')
axes[1].set_xlabel("number of qubits")
axes[1].set_ylabel("avg runtime over {} trials".format(SIM_TRIALS))
axes[1].set_title("depth={}, parametrized=Partial".format(DEPTH))
axes[1].legend(["cirq simulator", "sparse matrix simulator"])
plt.show()
