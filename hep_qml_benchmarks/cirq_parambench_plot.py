import numpy as np
import matplotlib.pyplot as plt

N_QUBITS, SIM_TRIALS, DEPTHS, N_PARAM_UPDATES = np.load('cirq_sympy_meta.npy')
# Outputs are in the shape (len(N_QUBITS), len(DEPTHS), SIM_TRIALS)
all_sympy_runs = np.load('cirq_sympy_bench.npy', )
all_float_runs = np.load('cirq_float_bench.npy', )



# CAREFUL: this will generate a lot of plots if you change N_QUBITS...
fig, axes = plt.subplots(ncols=len(N_QUBITS), nrows=1)
# FIXME: remove if you generate more than one qubits entry...
axes = [axes]
for i, n_qubit in enumerate(N_QUBITS):
    n_parameters = n_qubit * DEPTHS
    #
    mean_sympy = np.mean(all_sympy_runs[i], axis=1)
    min_sympy = np.min(all_sympy_runs[i], axis=1)
    max_sympy = np.max(all_sympy_runs[i], axis=1)
    print(min_sympy, max_sympy)

    mean_float = np.mean(all_float_runs[i], axis=1)
    min_float = np.min(all_float_runs[i], axis=1)
    max_float = np.max(all_float_runs[i], axis=1)
    print(min_float, max_float)

    axes[i].errorbar(n_parameters, mean_sympy, yerr=[min_sympy, max_sympy], solid_capstyle='projecting', capsize=5, fmt='o')
    axes[i].errorbar(n_parameters, mean_float, yerr=[min_float, max_float], solid_capstyle='projecting', capsize=5, fmt='o')
    axes[i].set_ylim(1E-3,1E2)
    axes[i].set_yscale('log')
    # axes[0][0].set_yscale('log')
    axes[i].set_ylabel("avg runtime over {} trials".format(SIM_TRIALS))
    axes[i].set_xlabel("number of parameters")
    axes[i].legend(["ParamResolver+Sympy", "reconstructed circuit"])
plt.show()
