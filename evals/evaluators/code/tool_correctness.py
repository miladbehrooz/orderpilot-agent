"""Tool correctness: were all expected tools called, whatever their arguments?"""

from evals.evaluators.code.trajectory_match import TrajectoryMatchEvaluator


class ToolCorrectnessEvaluator(TrajectoryMatchEvaluator):
    """Evaluator for the ``tool_correctness`` metric.

    Every expected tool must be called (one call per expected call); arguments are ignored,
    so "wrong tool" is reported apart from "right tool, wrong order".
    """

    key = "tool_correctness"
    tool_args_match_mode = "ignore"


__all__ = ["ToolCorrectnessEvaluator"]
