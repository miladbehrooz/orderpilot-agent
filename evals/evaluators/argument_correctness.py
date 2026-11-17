"""Argument correctness: were the expected tools called with the expected key arguments?"""

from evals.evaluators.trajectory import TrajectoryMatchEvaluator


class ArgumentCorrectnessEvaluator(TrajectoryMatchEvaluator):
    """Evaluator for the ``argument_correctness`` metric.

    Each expected call needs a call of that tool whose arguments include every expected key
    with an equal value (``2670`` and ``"2670"`` differ). Extra arguments are fine, and an
    empty expected ``args`` matches any call of that tool.
    """

    key = "argument_correctness"
    tool_args_match_mode = "superset"


__all__ = ["ArgumentCorrectnessEvaluator"]
