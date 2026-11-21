"""Trajectory efficiency: were needless calls avoided? A reference-free judge."""

from evals.evaluators.judges.trajectory_judge import TrajectoryJudgeEvaluator


class TrajectoryEfficiencyEvaluator(TrajectoryJudgeEvaluator):
    """Evaluator for the ``trajectory_efficiency`` metric.

    The judge checks for calls the answer didn't need and the contract didn't require, and
    for repeated calls. It says whether extra steps were needed, which ``step_count`` (the
    exact number, reported next to it) can't.
    """

    key = "trajectory_efficiency"
    prompt_version = "v1"


__all__ = ["TrajectoryEfficiencyEvaluator"]
