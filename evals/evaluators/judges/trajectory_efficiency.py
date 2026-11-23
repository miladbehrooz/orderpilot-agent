"""Trajectory efficiency: were needless calls avoided? A reference-free judge."""

from evals.evaluators.judges.trajectory_judge import TrajectoryJudgeEvaluator


class TrajectoryEfficiencyEvaluator(TrajectoryJudgeEvaluator):
    """Evaluator for the ``trajectory_efficiency`` metric.

    The judge checks for calls the request didn't need and the contract didn't require, and
    for repeated calls. It says whether extra steps were needed, which ``step_count`` (the
    exact number, reported next to it) can't. v2 no longer counts a call as needed just
    because the answer used its result (see ``docs/eval-journal.md``).
    """

    key = "trajectory_efficiency"
    prompt_version = "v2"


__all__ = ["TrajectoryEfficiencyEvaluator"]
