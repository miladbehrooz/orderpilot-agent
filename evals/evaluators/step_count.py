"""Step count: how many tool calls did the turn take?"""

from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

from evals.evaluators.base import Evaluator
from evals.evaluators.messages import model_calls, tool_calls


class StepCountEvaluator(Evaluator):
    """Evaluator for the ``step_count`` measure.

    The score is the number of tool calls in the turn; the comment also gives the number of
    model calls. It is a ``measure``: the report gives its mean and percentiles next to
    latency, never a pass/fail.
    """

    key = "step_count"
    kind = "measure"

    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]:
        """Count the tool calls and model calls of the turn.

        Args:
            run: The target run; ``outputs["messages"]`` is this turn, in OpenAI format.
            example: The example (unused: the measure needs no reference).

        Returns:
            One result whose score is the number of tool calls.
        """
        messages = (run.outputs or {}).get("messages", [])
        calls = len(tool_calls(messages))
        comment = f"{calls} tool calls in {model_calls(messages)} model calls."
        return [EvaluationResult(key=self.key, score=calls, comment=comment)]


__all__ = ["StepCountEvaluator"]
