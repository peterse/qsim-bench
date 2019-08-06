import pytest
import sympy
import numpy as np
import cirq

from helpers import _CirqTPU, _TFCirq, _TFQEigen, _Cirq
np.random.seed(31415926)


# pairs like (number of qubits, depth)
@pytest.fixture(scope="module")
def meta():
    """Shared circuit and parameter data across tests."""
    depth = 50
    qubits = cirq.LineQubit.range(10)
    n_loops = 100000
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
            `params` should be shape (n_qubits * depth, )

        Returns:
            cirq.Circuit
        returns a nested list of ops (OP_TREE)
    """
    out = []
    n_qubits = len(qubits)
    for d in range(depth):
        # slice params into layer-sized chunks
        param_subset = params[n_qubits*d:n_qubits**(d+1)]
        out.append(random_single_qubit_gates_layer(qubits, param_subset))
        # FIXME: _with_exponent doesn't work as expected with 2-qubit gates..
        # if not (d % 2):
        #
        # out.append(random_single_qubit_gates_layer(qubits, param_subset))
        # else:
        #     out.append(random_two_qubit_gates_layer(qubits, param_subset))
    return cirq.Circuit.from_ops(out)


@pytest.mark.parametrize('helper', [_CirqTPU, _TFCirq, _TFQEigen, _Cirq])
def test_single_qubit_parametrized_n_loops(meta, benchmark, helper):
    c = _generator_type_zero(meta["depth"], meta["qubits"], meta["params"])
    for k, param_row in zip(range(meta["n_loops"]), meta["params"]):
        pass




        """
        TODO: this can be removed once validation run is set up.
        VERIFICATION:
        Results between two resolved circuits will be compared
        according to the amplitude of the wavefunction at a random index
        `v_ind`, for every trial, for every parameter in the param updates.
        """
        sympy_outcomes = np.zeros(n_param_updates).astype(np.complex64)
        float_outcomes = np.zeros(n_param_updates).astype(np.complex64)
        v_ind = np.random.randint(2**n_qubits - 1)

    """
    TIMING:
    Sympy parameter resolution: Each trial consists of the time to run all of:
      3. call to cirq.Simulator() with param resolver kwarg
      4. access `final_state` field of TrialResult
      5. put a random wavefunction element to outcomes[v_ind]

    note that the parameter resolvers are constructed ahead of time, which
    is actually generous to the timing.

    circuit reconstructions: Each trial consists of the time to run all of:
      1. Iteratively copy the existing circuit op-wise, inserting new
           angles according to randomly generated params
      2. call to cirq.Simulator() using this new state
      3. put a random wavefunction element to outcomes[v_ind]
    """
    sympy_times = []
    float_times = []
    n_qubits = len(qubits)
    for k in range(sim_trials):

        float_circuit = update_params(float_circuit, all_params[j][:])
        float_outcomes[j] =  cirq.Simulator().simulate(float_circuit, initial_state=np.copy(x)).final_state[v_ind]

        # precompute all parameter updates to apply to both circuits
        all_params = np.random.rand(n_param_updates, n_qubits*depth)
        all_params = np.ones((n_param_updates, n_qubits*depth))
        # initialize a persistent sympy-parametrized circuit
        symbol_strings = []
        for i in range(n_qubits*depth):
            symbol_strings.append("{}".format(i) )
        layer_symbols = [sympy.Symbol(s) for s in symbol_strings]
        global trial
        layers = trial(depth, qubits, layer_symbols)

        # consistent initial state prep
        x = np.ones(2**n_qubits, dtype=np.complex64) / np.sqrt(2**n_qubits)
        sympy_circuit = cirq.Circuit.from_ops([g for l in layers for g in l])
        # cirq param resolver: Time includes only update
        start = timer()
        resolvers = [dict(zip(symbol_strings, all_params[j][:])) for j in range(n_param_updates)]
        for j in range(n_param_updates):

            sympy_outcomes[j] = cirq.Simulator().simulate(sympy_circuit,
                                      initial_state=np.copy(x),
                                      param_resolver=resolvers[j]).final_state[v_ind]

        sympy_trial_time = timer() - start
        sympy_times.append(sympy_trial_time)

        # initialize a copy of the circuit, this time using hard-coded angles
        # the symbols will be overwritten with the first update.
        float_circuit = sympy_circuit.copy()
        start = timer()
        for j in range(n_param_updates):
            # Regenerate _entire_ circuit with updates to float values
            # each time includes the circuit construction time
            float_circuit = update_params(float_circuit, all_params[j][:])
            float_outcomes[j] =  cirq.Simulator().simulate(float_circuit, initial_state=np.copy(x)).final_state[v_ind]

        float_trial_time = timer() - start
        float_times.append(float_trial_time)

        np.testing.assert_array_almost_equal(float_outcomes, sympy_outcomes)
        print("trial {}:")
        print("  sympy: ", sympy_trial_time)
        print("  float: ", float_trial_time)

    return np.asarray(sympy_times), np.asarray(float_times)





def timeit_n_rounds_k_updates(qubits, depth, sim_trials, n_param_updates):
    """
        Args:
            qubits: QubitId representation of circuit qubits
            depth (int): Circuit depth.
            sim_trials (int): Number of trials to run each simulation for the
                purpose of averaging
            n_param_updates (int): Number of updates to invoke per trial.

        Returns:
            Array of shape (sim_trials, ) containing times for each trial
    """

    """


"""
Benchmark description:

    For number of parameters N_PARAMS, construct a circuit of mixed one- and
    two-qubit parametrized gates of total depth DEPTH. Compare the following
    param resolution methods:
        (a) Sympy variables with native cirq ParamResolver feed dict
        (b) Circuit reconstructed directly from new parameters
"""
SIM_TRIALS = 20  # average runtime over this many runs
N_PARAMS = 20
N_PARAM_UPDATES = 100 # how many times to replace the parameters
# DEPTH = 10  # depth of dense matrix layers
MAX_N = 5  # run trials for n_qubits = 2...MAX_N
# N_QUBITS = list(range(2, MAX_N))
N_QUBITS = np.asarray([5])
DEPTHS = np.asarray(range(1,10))
all_sympy_runs = np.zeros((len(N_QUBITS), len(DEPTHS), SIM_TRIALS))
all_float_runs = np.zeros((len(N_QUBITS), len(DEPTHS), SIM_TRIALS))

for i, n_qubits in enumerate(N_QUBITS):
    for j, depth in enumerate(DEPTHS):
        qubits = cirq.LineQubit.range(n_qubits)
        print("simulating {} qubits".format(n_qubits))
        s1, f1 = timeit_n_rounds_k_updates(qubits, depth, SIM_TRIALS, N_PARAM_UPDATES)
        all_sympy_runs[i][j][:] = s1
        all_float_runs[i][j][:] = f1

np.save('cirq_sympy_bench.npy', np.asarray(all_sympy_runs))
np.save('cirq_float_bench.npy', np.asarray(all_float_runs))
np.save('cirq_sympy_meta.npy', np.array([N_QUBITS, SIM_TRIALS, DEPTHS, N_PARAM_UPDATES]))
