
import cirq
import numpy as np

# cirq tpu
from cirq.contrib.tpu import (
    circuit_to_tensorflow_runnable
)
import tensorflow as tf

# tf-cirq
from cirq.contrib.tf_backend.tf_simulator import (
    TFWaveFunctionSimulator
)

class Helper:
    def __init__(self, labels):
        """
        Args:
            labels: Ordered list of strings corresponding to targets for
                updates during call to `updated_execute`.
        """
        self.labels = labels


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
    @staticmethod
    def prepare(c, **kwargs):
        """Preprocess circuit c before timing starts."""
        simulator = TFWaveFunctionSimulator(dtype=tf.complex64)
        return simulator.simulate(c, **kwargs)

    @staticmethod
    def execute(cprime):
        """Compile ops from cirq into XLA-compatibile ops via cirq.contrib.tpu.

        Note: Timing on the compilation step is being benchmarked because the
        cirq tpu add-on relies heavily on linalg operations.
        """

        # todo: simpler operation than returning full wf? like:
        #expectation = lambda: tf.norm(r.compute()[:128], 2)

        with tf.Session() as session:
            output = session.run(cprime)
        return output

    def updated_execute(self, cprime, params):
        """Perform parameter updates for all parameters in circuit cprime.

        This expects a feed dict of the form {placeholder-name: new_value}
        """
        feed_dict = dict(zip(self.labels, params))
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

    @staticmethod
    def updated_execute(cprime, params):
        """Perform parameter updates for all parameters in circuit cprime.

        Update all parameters and run in a single, timable action. This
        expects a feed dict of the form {symbol: new_value}. This abbreviated
        method is UNSAFE: and will ignore symbol resolutions for a 10x speedup.
        """
        new_op_tree = []
        for op, param in zip(cprime.all_operations(), params):
            new_op_tree.append(op.gate._with_exponent(param/np.pi)(*op.qubits))
        c2 = cirq.Circuit.from_ops(new_op_tree)
        cirq.Simulator().simulate(c2)
