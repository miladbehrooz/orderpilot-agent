"""Evaluators: the shared error policy, answer correctness, the factory, and the LangSmith contract.

The judge model is never called: ``create_llm_as_judge`` is mocked where the base class
looks it up.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.language_models import BaseChatModel
from langsmith.evaluation.evaluator import run_evaluator
from langsmith.schemas import Example, Run
from pytest_mock import MockerFixture, MockType

from evals.errors import EvalSetupError
from evals.evaluators import (
    AnswerCorrectnessEvaluator,
    EvaluatorFactory,
    JudgeVerdict,
)
from evals.evaluators import base as base_module
from evals.evaluators import factory as factory_module

RunFactory = Callable[..., Run]
ExampleFactory = Callable[..., Example]


@pytest.fixture
def make_run() -> RunFactory:
    """Return a builder of target runs.

    Returns:
        ``make_run(answer=..., error=...)`` building a real LangSmith ``Run``.
    """

    def _make(answer: str = "Yes, until 2026-12-04.", error: str | None = None) -> Run:
        """Build a target run.

        Args:
            answer: The agent's final answer.
            error: The run's error, if it failed.

        Returns:
            The run.
        """
        run_id = uuid4()
        return Run.model_validate(
            {
                "id": run_id,
                "trace_id": run_id,
                "name": "invoke",
                "run_type": "chain",
                "start_time": datetime(2026, 11, 16, tzinfo=UTC),
                "inputs": {},
                "outputs": None if error else {"answer": answer, "messages": []},
                "error": error,
            }
        )

    return _make


@pytest.fixture
def make_example() -> ExampleFactory:
    """Return a builder of dataset examples.

    Returns:
        ``make_example(metrics=...)`` building a real LangSmith ``Example``.
    """

    def _make(metrics: list[str] | None = None) -> Example:
        """Build an example for a P3-style case.

        Args:
            metrics: Metrics listed in the example metadata.

        Returns:
            The example.
        """
        return Example.model_validate(
            {
                "id": uuid4(),
                "inputs": {"customer_email": "a@example.com", "question": "Can I return order 7?"},
                "outputs": {
                    "expected_outcome": "Yes, returnable until 2026-12-04.",
                    "key_facts": {"order_id": 7, "return_deadline": "2026-12-04"},
                },
                "metadata": {
                    "case_id": "P3-01",
                    "scenario_id": "P3",
                    "metrics": ["answer_correctness"] if metrics is None else metrics,
                },
            }
        )

    return _make


@pytest.fixture
def judge_fn(mocker: MockerFixture) -> MockType:
    """Mock ``create_llm_as_judge`` and return the judge function it builds.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked judge callable; it returns a passing verdict unless changed.
    """
    create: MockType = mocker.patch.object(base_module, "create_llm_as_judge", autospec=True)
    judge: MockType = create.return_value
    judge.return_value = JudgeVerdict(reasoning="Decision and deadline match.", verdict=True)
    return judge


@pytest.fixture
def evaluator(mocker: MockerFixture, judge_fn: MockType) -> AnswerCorrectnessEvaluator:
    """Return the answer-correctness evaluator with a mocked judge.

    Args:
        mocker: Pytest-mock fixture.
        judge_fn: The mocked judge callable.

    Returns:
        The evaluator.
    """
    return AnswerCorrectnessEvaluator(mocker.create_autospec(BaseChatModel, instance=True))


def _results(output: Any) -> list[Any]:
    """Return the result list of an evaluator output.

    Args:
        output: What the evaluator returned.

    Returns:
        ``output["results"]``.
    """
    results: list[Any] = output["results"]
    return results


def test_evaluator_skips_case_without_metric(
    evaluator: AnswerCorrectnessEvaluator,
    judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """A case that doesn't list the metric gets no result, and the judge isn't called.

    Args:
        evaluator: Evaluator under test.
        judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
    """
    output = evaluator(make_run(), make_example(metrics=["trajectory_match"]))
    assert _results(output) == []
    judge_fn.assert_not_called()


def test_evaluator_scores_agent_error_as_failure(
    evaluator: AnswerCorrectnessEvaluator,
    judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """A run that crashed fails the metric; it never drops out of the denominator.

    Args:
        evaluator: Evaluator under test.
        judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
    """
    (result,) = _results(evaluator(make_run(error="GraphRecursionError"), make_example()))
    assert (result.key, result.score) == ("answer_correctness", False)
    assert "GraphRecursionError" in result.comment
    judge_fn.assert_not_called()


def test_evaluator_reports_judge_failure_separately(
    evaluator: AnswerCorrectnessEvaluator,
    judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """A broken judge emits evaluator_error and no score, so it can't look like a failing agent.

    Args:
        evaluator: Evaluator under test.
        judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
    """
    judge_fn.side_effect = TimeoutError("judge timed out")
    (result,) = _results(evaluator(make_run(), make_example()))
    assert result.key == "evaluator_error"
    assert result.metadata == {"metric": "answer_correctness"}
    assert "judge timed out" in result.comment


@pytest.mark.parametrize("verdict", [True, False])
def test_answer_correctness_maps_verdict_and_reasoning(
    evaluator: AnswerCorrectnessEvaluator,
    judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
    verdict: bool,
) -> None:
    """The judge's verdict becomes the score, its reasoning the comment, with the prompt version.

    Args:
        evaluator: Evaluator under test.
        judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
        verdict: The judge's verdict.
    """
    judge_fn.return_value = JudgeVerdict(reasoning="Checked the deadline.", verdict=verdict)
    (result,) = _results(evaluator(make_run(), make_example()))
    assert (result.key, result.score, result.comment) == (
        "answer_correctness",
        verdict,
        "Checked the deadline.",
    )
    assert result.metadata == {"prompt_version": "v1"}


def test_answer_correctness_shows_judge_only_question_answer_and_reference(
    evaluator: AnswerCorrectnessEvaluator,
    judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """The judge gets the question, answer and reference; never tool outputs or metadata.

    Args:
        evaluator: Evaluator under test.
        judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
    """
    evaluator(make_run(answer="Yes."), make_example())
    kwargs = judge_fn.call_args.kwargs
    assert set(kwargs) == {"question", "answer", "expected_outcome", "key_facts"}
    assert kwargs["question"] == "Can I return order 7?"
    assert kwargs["answer"] == "Yes."
    assert '"return_deadline": "2026-12-04"' in kwargs["key_facts"]


def test_answer_correctness_prompt_has_every_placeholder() -> None:
    """The prompt template fills with exactly the variables the evaluator passes."""
    filled = AnswerCorrectnessEvaluator.prompt.format(
        question="Q", answer="A", expected_outcome="E", key_facts="F"
    )
    assert all(f"\n{value}\n" in filled for value in "QAEF")


def test_langsmith_accepts_evaluator_object(
    evaluator: AnswerCorrectnessEvaluator, make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Contract: LangSmith wraps our object and keeps our feedback key.

    Args:
        evaluator: Evaluator under test.
        make_run: Run builder.
        make_example: Example builder.
    """
    wrapped = run_evaluator(evaluator)
    output = wrapped.evaluate_run(make_run(), make_example())
    (result,) = _results(output)
    assert (result.key, result.score) == ("answer_correctness", True)


def test_langsmith_accepts_not_applicable_result(
    evaluator: AnswerCorrectnessEvaluator, make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Contract: an empty result list is accepted by LangSmith and emits no feedback.

    Args:
        evaluator: Evaluator under test.
        make_run: Run builder.
        make_example: Example builder.
    """
    output = run_evaluator(evaluator).evaluate_run(make_run(), make_example(metrics=[]))
    assert _results(output) == []


def test_factory_builds_judge_only_when_needed(mocker: MockerFixture, judge_fn: MockType) -> None:
    """The judge is built once for judge metrics, and never for an empty selection.

    Args:
        mocker: Pytest-mock fixture.
        judge_fn: The mocked judge callable (keeps the real judge out of the evaluator).
    """
    build_judge = mocker.patch.object(factory_module, "build_judge", autospec=True)
    build_judge.return_value = mocker.create_autospec(BaseChatModel, instance=True)

    assert EvaluatorFactory.build([]) == []
    build_judge.assert_not_called()

    (built,) = EvaluatorFactory.build(["answer_correctness"])
    assert isinstance(built, AnswerCorrectnessEvaluator)
    assert built.__name__ == "answer_correctness"
    build_judge.assert_called_once_with()


def test_factory_rejects_unknown_metric() -> None:
    """An unknown metric name fails with the list of valid metric names."""
    with pytest.raises(EvalSetupError, match="answer_correctness"):
        EvaluatorFactory.build(["answer_corectness"])


def test_factory_reports_judge_prompt_versions() -> None:
    """Prompt versions of the selected judge metrics go into the experiment metadata."""
    assert EvaluatorFactory.prompt_versions(["answer_correctness"]) == {"answer_correctness": "v1"}
    assert EvaluatorFactory.metric_names() == ["answer_correctness"]
