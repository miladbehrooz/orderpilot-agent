"""Trajectory-match base: the turn's tool calls against the case's expected tool calls."""

import json
from collections.abc import Sequence
from typing import Any, ClassVar

from langchain_core.messages import AIMessage, ToolCall, convert_to_messages
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run
from openevals.trajectory.match import create_trajectory_match_evaluator
from openevals.types import ToolArgsMatchMode

from evals.evaluators.base import Evaluator


class TrajectoryMatchEvaluator(Evaluator):
    """Base for metrics that check the expected tool calls with openevals' trajectory match.

    The trajectory match is a superset: every expected call needs its own matching call in
    the turn; extra calls and their order don't matter. Subclasses choose how a call's
    arguments are compared.

    The score comes from openevals, which gives no comment. On a failure, each expected call
    is matched again on its own, with the same matcher, to name the missing ones.

    Attributes:
        tool_args_match_mode: openevals mode for comparing a call's arguments with the
            expected ones (``"ignore"`` or ``"superset"``).
    """

    required_reference = ("expected_tool_calls",)
    tool_args_match_mode: ClassVar[ToolArgsMatchMode]

    def __init__(self) -> None:
        super().__init__()
        self._match = create_trajectory_match_evaluator(
            trajectory_match_mode="superset", tool_args_match_mode=self.tool_args_match_mode
        )

    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]:
        """Check that the turn contains every expected tool call.

        Args:
            run: The target run; ``outputs["messages"]`` is this turn, in OpenAI format.
            example: The example; ``outputs["expected_tool_calls"]`` is the reference.

        Returns:
            One pass/fail result whose comment names the missing calls and the calls made.
        """
        messages: list[dict[str, Any]] = (run.outputs or {}).get("messages", [])
        expected: list[dict[str, Any]] = (example.outputs or {})["expected_tool_calls"]
        passed = self._matches(messages, expected)
        made = ", ".join(_describe(c["name"], c["args"]) for c in _tool_calls(messages))
        comment = f"Called: {made or 'no tools'}."
        if not passed:
            missing = [call for call in expected if not self._matches(messages, [call])]
            if missing:
                described = ", ".join(self._describe_expected(call) for call in missing)
                comment = f"Missing: {described}. {comment}"
            else:
                comment = f"Each expected call matches, but not one call per expectation. {comment}"
        return [EvaluationResult(key=self.key, score=passed, comment=comment)]

    def _matches(self, messages: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
        """Run the openevals superset match of the turn against some expected calls.

        Args:
            messages: The turn, in OpenAI format.
            expected: Expected calls as ``{"name", "args"}`` dicts.

        Returns:
            Whether every expected call has its own matching call in the turn.

        Raises:
            TypeError: If openevals returns several results instead of one.
        """
        reference = [{"role": "assistant", "content": "", "tool_calls": expected}]
        result = self._match(outputs=messages, reference_outputs=reference)
        if isinstance(result, list):
            raise TypeError(f"Expected one trajectory match result, got {len(result)}")
        return bool(result["score"])

    def _describe_expected(self, call: dict[str, Any]) -> str:
        """Describe an expected call, with its arguments only when they are compared.

        Args:
            call: Expected call as a ``{"name", "args"}`` dict.

        Returns:
            E.g. ``check_return_eligibility(order_id=2670)``, or only the tool name when
            arguments are ignored.
        """
        if self.tool_args_match_mode == "ignore":
            name: str = call["name"]
            return name
        return _describe(call["name"], call["args"])


def _tool_calls(messages: Sequence[dict[str, Any]]) -> list[ToolCall]:
    """Return the tool calls made in a turn, with parsed arguments.

    Args:
        messages: The turn, in OpenAI format.

    Returns:
        The calls of every AI message, in order.
    """
    return [
        call
        for message in convert_to_messages(messages)
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def _describe(name: str, args: dict[str, Any]) -> str:
    """Format a tool call for a comment.

    Args:
        name: Tool name.
        args: Call arguments.

    Returns:
        E.g. ``check_return_eligibility(order_id=2670, order_item_ids=[3862])``.
    """
    formatted = ", ".join(f"{key}={json.dumps(value)}" for key, value in args.items())
    return f"{name}({formatted})"


__all__ = ["TrajectoryMatchEvaluator"]
