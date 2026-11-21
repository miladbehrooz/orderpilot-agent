"""Argument quality: are the arguments supported by the conversation? A reference-free judge."""

from evals.evaluators.trajectory_judge import TrajectoryJudgeEvaluator


class ArgumentQualityEvaluator(TrajectoryJudgeEvaluator):
    """Evaluator for the ``argument_quality`` metric.

    The judge checks that every argument comes from the conversation or an earlier tool
    result: the order and items the customer meant, no invented ids. Tool choice and extra
    calls are other metrics. Its reference-based counterpart is ``argument_correctness``.
    """

    key = "argument_quality"
    prompt_version = "v1"


__all__ = ["ArgumentQualityEvaluator"]
