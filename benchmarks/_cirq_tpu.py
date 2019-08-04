from cirq.contrib.tpu import (
    circuit_to_tensorflow_runnable
)
import tensorflow as tf


def prepare_for_bench(c):
    """Preprocess circuit c before timing starts."""
    return circuit_to_tensorflow_runnable(c)


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
