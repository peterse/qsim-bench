import cirq
import numpy as np
import sympy
# cirq tpu
from cirq.contrib.tpu import (
    circuit_to_tensorflow_runnable
)
import tensorflow as tf

import tfc


class Helper:
    def __init__(self, meta):
        """
        Args:
        """


class _CirqTPU(Helper):
    @staticmethod
    def prepare(c):
        """Preprocess circuit c before timing starts."""
        return circuit_to_tensorflow_runnable(c)

    @staticmethod
    def execute(cprime):
        """Compile ops from cirq into XLA-compatibile ops via cirq.contrib.tpu.

        Note: Timing on the compilation step is being benchmarked because the
        cirq tpu add-on relies heavily on linalg operations.
        """

        # todo: simpler operation than returning full wf? like:
        #expectation = lambda: tf.norm(r.compute()[:128], 2)

        with tf.Session() as session:
            output = session.run(cprime.compute(), feed_dict=cprime.feed_dict)
        return output


class _TFCirq(Helper):

    # def __init__(self, meta=None):
    #     self.labels = "placeholder{}:0".format(i) for i in range(meta["params"])

    @staticmethod
    def prepare(c, **kwargs):
        """Preprocess circuit c before timing starts."""
        simulator = tfc.TFWaveFunctionSimulator(dtype=tf.complex64)
        return simulator.simulate(c, **kwargs)

    @staticmethod
    def execute(cprime):
        """Compile ops from cirq into XLA-compatibile ops via cirq.contrib.tpu.
        """
        # todo: simpler operation than returning full wf? like:
        #expectation = lambda: tf.norm(r.compute()[:128], 2)

        with tf.Session() as session:
            output = session.run(cprime)
        return output

    def prepare_parameters(self, params):
        """Prepare a class-specific invocation of the input parameters."""
        self.labels = [f"v{i}" for i in range(len(params))]
        self.placeholders = [tf.placeholder(tf.complex64, shape=(), name=s) for s in self.labels]
        return self.placeholders

    def updated_execute(self, cprime, params):
        """Perform parameter updates for all parameters in circuit cprime.

        This expects a feed dict of the form {placeholder-name: new_value}

        Timing disclosure:
          1. Construct feeder from `params` arg
          2. Run compiled TFWaveFunctionSimulator graph
        """
        # new_labels = [s+":0" for s in self.labels]

        feed_dict = dict(zip(self.placeholders, params))
        with tf.Session() as session:
            output = session.run(cprime, feed_dict=feed_dict)
        return output


class _TFQEigen(Helper):
    @staticmethod
    def prepare(c):
        """Preprocess circuit c before timing starts."""

        # TODO
        return

    @staticmethod
    def execute(cprime):
        """TODO."""
        return



class _Cirq(Helper):

    @staticmethod
    def prepare(c):
        """Preprocess circuit c before timing starts."""
        return c

    @staticmethod
    def execute(cprime, **kwargs):
        """Execute ops in cirq; output will be baseline benchmark truth."""
        return cirq.Simulator().simulate(cprime, **kwargs)

    def prepare_parameters(self, params):
        """Prepare a class-specific invocation of the input parameters.

        Override cirq's Sympy resolver for circuit rewrites.
        """
        self.symbols = [sympy.Symbol("v{i}") for i in range(len(params))]
        return

    def updated_execute(self, cprime, params):
        """Perform parameter updates for all parameters in circuit cprime.

        This expects a feed dict of the form {placeholder-name: new_value}

        Update all parameters and run in a single, timable action. This
        expects a feed dict of the form {symbol: new_value}.

        WARNING:This abbreviated method is UNSAFE: and will ignore symbol
        resolutions for a 10x speedup.

        Timing disclosure:
          1. Iteratively copy the existing circuit op-wise, inserting new
               angles according to randomly generated params
          2. call to cirq.Simulator() using this new state
        """
        # TODO: fix cirq's paramresolver
        # param_resolver = dict(zip(self.symbols, params))
        # return cirq.Simulator().simulate(c2, param_resolver=param_resolver).final_state

        new_op_tree = []
        for op, param in zip(cprime.all_operations(), params):
            new_op_tree.append(op.gate._with_exponent(param/np.pi)(*op.qubits))
        c2 = cirq.Circuit.from_ops(new_op_tree)
        return cirq.Simulator().simulate(c2).final_state
