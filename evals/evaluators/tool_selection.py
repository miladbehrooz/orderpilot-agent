"""Tool selection: were suitable tools called, in a sensible order? A reference-free judge."""

from evals.evaluators.trajectory_judge import TrajectoryJudgeEvaluator


class ToolSelectionEvaluator(TrajectoryJudgeEvaluator):
    """Evaluator for the ``tool_selection`` metric.

    The judge checks that the tools the agent's contract requires were called, used for
    what they are for, and in an order that works. Arguments and extra calls are other
    metrics. Its reference-based counterpart is ``tool_correctness``.
    """

    key = "tool_selection"
    prompt_version = "v1"


__all__ = ["ToolSelectionEvaluator"]
