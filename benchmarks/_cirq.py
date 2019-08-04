import cirq


def prepare_for_bench(c):
    """Preprocess circuit c before timing starts."""
    return c


def execute(cprime):
    """Execute ops in cirq; output will be baseline benchmark truth."""
    return cirq.Simulator().simulate(cprime)
