"""Reason code: did the eligibility tool return the expected ``reason_code``?"""

import json

from langchain_core.messages import ToolMessage
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

from evals.evaluators.base import Evaluator
from evals.evaluators.messages import tool_results


class ReasonCodeEvaluator(Evaluator):
    """Evaluator for the ``reason_code`` metric.

    The case names a tool and the code it must return. The tool's last result in the turn
    counts, so a retry after a wrong order id is graded on the check the agent finally
    explains. The expected code is hand-written, never computed by ``policy.py``, so this
    checks the decision that reached the agent on the real database end to end.
    """

    key = "reason_code"
    required_reference = ("expected_reason_code",)

    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]:
        """Compare the reason code of the tool's last result with the expected one.

        Args:
            run: The target run; ``outputs["messages"]`` is this turn, in OpenAI format.
            example: The example; ``outputs["expected_reason_code"]`` is the reference.

        Returns:
            One pass/fail result whose comment gives the actual and, on a failure, the
            expected code.
        """
        messages = (run.outputs or {}).get("messages", [])
        expected = (example.outputs or {})["expected_reason_code"]
        tool, code = expected["tool"], expected["code"]
        results = tool_results(messages, tool)
        if not results:
            return [EvaluationResult(key=self.key, score=False, comment=f"No {tool} result.")]
        actual = _reason_code(results[-1])
        passed = actual == code
        comment = f"{tool} returned {actual or 'no readable reason_code'}"
        if not passed:
            comment += f", expected {code}"
        if len(results) > 1:
            comment += f" (last of {len(results)} results)"
        return [EvaluationResult(key=self.key, score=passed, comment=f"{comment}.")]


def _reason_code(message: ToolMessage) -> str | None:
    """Read the reason code of a tool result.

    Args:
        message: The tool result; its content is the tool's JSON result.

    Returns:
        The ``reason_code``, or ``None`` if the content isn't a JSON object with one.
    """
    try:
        content = json.loads(message.text)
    except json.JSONDecodeError:
        return None
    code = content.get("reason_code") if isinstance(content, dict) else None
    return code if isinstance(code, str) else None


__all__ = ["ReasonCodeEvaluator"]
