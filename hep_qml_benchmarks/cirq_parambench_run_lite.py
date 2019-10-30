
from timeit import default_timer as timer
import numpy as np
import sympy
import cirq


def update_params(circuit, params):
    """Competitor method to cirq.ParamResolver."""
    new_op_tree = []
    for op, param in zip(circuit.all_operations(), params):
        new_op_tree.append(op.gate._with_exponent(param/np.pi)(*op.qubits))
    return cirq.Circuit.from_ops(new_op_tree)


trials = 100
for depth in [10, 15, 20]:
    sympy_circuit = cirq.Circuit.from_ops([cirq.Rx(sympy.Symbol(str(k)))(cirq.LineQubit(0)) for k in range(depth)])

    random_params = np.random.randn(trials, depth)
    truth = np.zeros(trials, dtype=np.complex64)
    # time twenty runs
    start = timer()
    for j in range(trials):
        resolver = dict(zip([str(k) for k in range(depth)], random_params[j]))
        wf1 = cirq.Simulator().simulate(sympy_circuit, param_resolver=resolver).final_state
        truth[j] = wf1[1]
    end = timer() - start
    print(f"{depth} parameters, {trials} trials using Sympy+ParamResolver: {end} seconds")

    start = timer()
    verify = np.zeros(trials, dtype=np.complex64)
    for j in range(trials):
        float_circuit = update_params(sympy_circuit, random_params[j])
        wf2 = cirq.Simulator().simulate(float_circuit).final_state
        verify[j] = wf2[1]
    end = timer() - start
    print(f"{depth} parameters, {trials} trials using reconstructed circuit: {end} seconds")
    np.testing.assert_array_almost_equal(verify, truth)
