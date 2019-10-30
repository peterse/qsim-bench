import numpy as np
import matplotlib.pyplot as plt

# check for fair simulation comparison
SIM_TRIALS, DEPTH = np.load('cirq_v0.4.0_meta.npy')
temp1, temp2 = np.load('cirq_v0.5.0_meta.npy')
assert temp1 == SIM_TRIALS
assert temp2 == DEPTH

v04 = np.load('cirq_v0.4.0_bench.npy')
v05 = np.load('cirq_v0.5.0_bench.npy')

fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0][0].semilogy(v04[0], v04[1], v05[0], v05[1])
axes[0][0].set_yscale('log')
axes[0][0].set_ylabel("avg runtime over {} trials".format(SIM_TRIALS))
axes[0][0].set_title("depth={}, parametrized=False".format(DEPTH))

axes[0][1].semilogy(v04[0], v04[2], v05[0], v05[2])
axes[0][1].set_yscale('log')
axes[0][1].set_ylabel("avg runtime over {} trials".format(SIM_TRIALS))
axes[0][1].set_title("depth={}, parametrized=Partial".format(DEPTH))

axes[1][0].semilogy(v04[0], v04[3], v05[0], v05[3])
axes[1][0].set_yscale('log')
axes[1][0].set_xlabel("number of qubits")
axes[1][0].set_ylabel("avg runtime over {} trials".format(SIM_TRIALS))
axes[1][0].set_title("kernel circuit".format(DEPTH))
axes[1][0].legend(["v0.4.0", "v0.5.0"])

axes[1][1].plot(v04[0], v05[1]/v04[1], v05[0], v05[2]/v04[2], v05[0], v05[3]/v04[3])
plt.axhline(1, ls="--", c="k")
axes[1][1].set_xlabel("number of qubits")
axes[1][1].set_ylabel("relative runtime, v4/v5")
axes[1][1].set_title("relative runtime, v5/v4")
axes[1][1].legend(["plot 1", "plot 2", "plot 3"])
plt.show()
