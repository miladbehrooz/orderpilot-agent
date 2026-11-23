"""Builders of LangSmith runs and examples for the eval tests (served as fixtures in conftest)."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langsmith.schemas import Example, Run

RunFactory = Callable[..., Run]
ExampleFactory = Callable[..., Example]


def build_run(
    answer: str = "Yes, until 2026-12-04.",
    error: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> Run:
    """Build a target run.

    Args:
        answer: The agent's final answer.
        error: The run's error, if it failed.
        messages: The turn in OpenAI format; empty if omitted.

    Returns:
        A real LangSmith ``Run``.
    """
    run_id = uuid4()
    outputs = {"answer": answer, "messages": messages or []}
    return Run.model_validate(
        {
            "id": run_id,
            "trace_id": run_id,
            "name": "invoke",
            "run_type": "chain",
            "start_time": datetime(2026, 11, 16, tzinfo=UTC),
            "inputs": {},
            "outputs": None if error else outputs,
            "error": error,
        }
    )


def build_example(
    metrics: list[str] | None = None,
    expected_tool_calls: list[dict[str, Any]] | None = None,
    expected_reason_code: str = "ELIGIBLE",
) -> Example:
    """Build an example for a P3-style case.

    Args:
        metrics: Metrics listed in the example metadata; ``answer_correctness`` if omitted.
        expected_tool_calls: Expected calls as ``{"name", "args"}`` dicts; by default one
            ``check_return_eligibility`` call for order 7.
        expected_reason_code: Code ``check_return_eligibility`` must return.

    Returns:
        A real LangSmith ``Example``.
    """
    if expected_tool_calls is None:
        expected_tool_calls = [{"name": "check_return_eligibility", "args": {"order_id": 7}}]
    return Example.model_validate(
        {
            "id": uuid4(),
            "inputs": {"customer_email": "a@example.com", "question": "Can I return order 7?"},
            "outputs": {
                "expected_outcome": "Yes, returnable until 2026-12-04.",
                "key_facts": {"order_id": 7, "return_deadline": "2026-12-04"},
                "expected_tool_calls": expected_tool_calls,
                "expected_reason_code": {
                    "tool": "check_return_eligibility",
                    "code": expected_reason_code,
                },
            },
            "metadata": {
                "case_id": "P3-01",
                "scenario_id": "P3",
                "metrics": ["answer_correctness"] if metrics is None else metrics,
            },
        }
    )


__all__ = ["ExampleFactory", "RunFactory", "build_example", "build_run"]
