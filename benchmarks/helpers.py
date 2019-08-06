
import cirq


# cirq tpu
from cirq.contrib.tpu import (
    circuit_to_tensorflow_runnable
)
import tensorflow as tf

# tf-cirq
from cirq.contrib.tf_backend.tf_simulator import (
    TFWaveFunctionSimulator
)

class _CirqTPU:
    @classmethod
    def prepare(c):
        """Preprocess circuit c before timing starts."""
        return circuit_to_tensorflow_runnable(c)

    @classmethod
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


class _TFCirq:
    @classmethod
    def prepare(c):
        """Preprocess circuit c before timing starts."""
        simulator = TFWaveFunctionSimulator(dtype=tf.complex64)
        return simulator.simulate(c)

    @classmethod
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


class _TFQEigen:
    @classmethod
    def prepare(c):
        """Preprocess circuit c before timing starts."""

        # TODO
        return

    @classmethod
    def execute(cprime):
        """TODO."""
        return


class _Cirq:

    @classmethod
    def prepare(c):
        """Preprocess circuit c before timing starts."""
        return c

    def execute(cprime):
        """Execute ops in cirq; output will be baseline benchmark truth."""
        return cirq.Simulator().simulate(cprime)
