"""Evaluators: error policy, answer and trajectory metrics, the factory, the LangSmith contract.

The judge model is never called: ``create_llm_as_judge`` is mocked where the base class
looks it up. The trajectory metrics run the real openevals matcher on hand-built turns.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langsmith.evaluation.evaluator import run_evaluator
from langsmith.schemas import Example, Run
from pydantic import ValidationError
from pytest_mock import MockerFixture, MockType

from evals.cases import Reference
from evals.errors import EvalSetupError
from evals.evaluators import (
    AnswerCorrectnessEvaluator,
    ArgumentCorrectnessEvaluator,
    EvaluatorFactory,
    JudgeVerdict,
    ToolCorrectnessEvaluator,
    TrajectoryMatchEvaluator,
)
from evals.evaluators import base as base_module
from evals.evaluators import factory as factory_module
from evals.prompts import JUDGE_PROMPTS, JudgePrompt, prompt_for

RunFactory = Callable[..., Run]
ExampleFactory = Callable[..., Example]


@pytest.fixture
def make_run() -> RunFactory:
    """Return a builder of target runs.

    Returns:
        ``make_run(answer=..., error=...)`` building a real LangSmith ``Run``.
    """

    def _make(
        answer: str = "Yes, until 2026-12-04.",
        error: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> Run:
        """Build a target run.

        Args:
            answer: The agent's final answer.
            error: The run's error, if it failed.
            messages: The turn in OpenAI format (see ``turn``); empty if omitted.

        Returns:
            The run.
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

    return _make


@pytest.fixture
def make_example() -> ExampleFactory:
    """Return a builder of dataset examples.

    Returns:
        ``make_example(metrics=..., expected_tool_calls=...)`` building a real LangSmith
        ``Example``.
    """

    def _make(
        metrics: list[str] | None = None, expected_tool_calls: list[dict[str, Any]] | None = None
    ) -> Example:
        """Build an example for a P3-style case.

        Args:
            metrics: Metrics listed in the example metadata.
            expected_tool_calls: Expected calls as ``{"name", "args"}`` dicts; by default
                one ``check_return_eligibility`` call for order 7.

        Returns:
            The example.
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
def evaluator(
    mocker: MockerFixture, judge_fn: MockType, today: datetime
) -> AnswerCorrectnessEvaluator:
    """Return the answer-correctness evaluator with a mocked judge.

    Args:
        mocker: Pytest-mock fixture.
        judge_fn: The mocked judge callable.
        today: TODAY of the test shop, given to the judge.

    Returns:
        The evaluator.
    """
    return AnswerCorrectnessEvaluator(mocker.create_autospec(BaseChatModel, instance=True), today)


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
    assert result.metadata == {"prompt_version": "v2"}


def test_answer_correctness_shows_judge_only_question_answer_and_reference(
    evaluator: AnswerCorrectnessEvaluator,
    judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """The judge gets today's date, the question, answer and reference; never tool calls.

    Args:
        evaluator: Evaluator under test.
        judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
    """
    evaluator(make_run(answer="Yes."), make_example())
    kwargs = judge_fn.call_args.kwargs
    assert set(kwargs) == {"today", "question", "answer", "expected_outcome", "key_facts"}
    assert kwargs["today"] == "2026-11-12"
    assert kwargs["question"] == "Can I return order 7?"
    assert kwargs["answer"] == "Yes."
    assert '"return_deadline": "2026-12-04"' in kwargs["key_facts"]
    assert all("check_return_eligibility" not in str(value) for value in kwargs.values())


def test_answer_correctness_prompt_has_every_placeholder() -> None:
    """The v2 prompt fills with exactly the variables the evaluator passes."""
    filled = prompt_for("answer_correctness", "v2").format(
        question="Q", answer="A", expected_outcome="E", key_facts="F", today="2026-11-12"
    )
    assert all(f"\n{value}\n" in filled for value in "QAEF")
    assert "Today is 2026-11-12." in filled


def test_every_prompt_version_is_registered_once() -> None:
    """Each metric and version pair appears once, so a lookup is never ambiguous."""
    pairs = [(p.metric, p.version) for p in JUDGE_PROMPTS]
    assert len(pairs) == len(set(pairs))


def test_unknown_prompt_version_names_the_known_ones() -> None:
    """Asking for a version that doesn't exist fails with the versions that do."""
    with pytest.raises(EvalSetupError, match="answer_correctness v1"):
        prompt_for("answer_correctness", "v9")


def test_prompt_version_must_look_like_a_version() -> None:
    """A version such as ``V2`` is rejected when the prompt is defined."""
    with pytest.raises(ValidationError):
        JudgePrompt(metric="answer_correctness", version="V2", text="x")


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


def test_factory_builds_judge_only_when_needed(
    mocker: MockerFixture, judge_fn: MockType, today: datetime
) -> None:
    """The judge is built once for judge metrics, and never for an empty selection.

    Args:
        mocker: Pytest-mock fixture.
        judge_fn: The mocked judge callable (keeps the real judge out of the evaluator).
        today: TODAY of the test shop.
    """
    build_judge = mocker.patch.object(factory_module, "build_judge", autospec=True)
    build_judge.return_value = mocker.create_autospec(BaseChatModel, instance=True)

    assert EvaluatorFactory.build([], today) == []
    build_judge.assert_not_called()

    (built,) = EvaluatorFactory.build(["answer_correctness"], today)
    assert isinstance(built, AnswerCorrectnessEvaluator)
    assert built.__name__ == "answer_correctness"
    assert built.today == "2026-11-12"
    build_judge.assert_called_once_with()


def test_factory_rejects_unknown_metric(today: datetime) -> None:
    """An unknown metric name fails with the list of valid metric names.

    Args:
        today: TODAY of the test shop.
    """
    with pytest.raises(EvalSetupError, match="answer_correctness"):
        EvaluatorFactory.build(["answer_corectness"], today)


def test_factory_reports_judge_prompt_versions() -> None:
    """Prompt versions of the selected judge metrics go into the experiment metadata."""
    assert EvaluatorFactory.prompt_versions(["answer_correctness"]) == {"answer_correctness": "v2"}
    assert EvaluatorFactory.prompt_versions(["tool_correctness", "argument_correctness"]) == {}
    assert EvaluatorFactory.metric_names() == [
        "answer_correctness",
        "tool_correctness",
        "argument_correctness",
    ]


def turn(*calls: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one agent turn in OpenAI format, as the target returns it.

    Args:
        *calls: ``(tool name, args)`` of each tool call, one AI message per call.

    Returns:
        The question, each call with its tool result, and a final answer.
    """
    messages: list[AnyMessage] = [HumanMessage("Can I return order 2670?")]
    for index, (name, args) in enumerate(calls):
        call_id = f"call-{index}"
        messages.append(AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}]))
        messages.append(ToolMessage('{"reason_code": "OK"}', tool_call_id=call_id, name=name))
    messages.append(AIMessage("Yes, you can."))
    return convert_to_openai_messages(messages)


def p8_expected() -> list[dict[str, Any]]:
    """Return P8-style expected calls: look the item up, then check only that item.

    Returns:
        Expected calls as ``{"name", "args"}`` dicts.
    """
    return [
        {"name": "get_order_details", "args": {"order_id": 2670}},
        {"name": "check_return_eligibility", "args": {"order_id": 2670, "order_item_ids": [3862]}},
    ]


def _grade(
    evaluator: TrajectoryMatchEvaluator,
    make_run: RunFactory,
    make_example: ExampleFactory,
    messages: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> Any:
    """Grade one turn and return its single result.

    Args:
        evaluator: Trajectory evaluator under test.
        make_run: Run builder.
        make_example: Example builder.
        messages: The turn in OpenAI format.
        expected: Expected calls as ``{"name", "args"}`` dicts.

    Returns:
        The one ``EvaluationResult``.
    """
    example = make_example(metrics=[evaluator.key], expected_tool_calls=expected)
    (result,) = _results(evaluator(make_run(messages=messages), example))
    return result


def test_tool_correctness_passes_when_every_expected_tool_is_called(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Every expected tool called passes, and the comment lists the calls made.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(
        ("get_order_details", {"order_id": 2670}),
        ("check_return_eligibility", {"order_id": 2670, "order_item_ids": [3862]}),
    )
    result = _grade(ToolCorrectnessEvaluator(), make_run, make_example, messages, p8_expected())
    assert (result.key, result.score) == ("tool_correctness", True)
    assert result.comment == (
        "Called: get_order_details(order_id=2670), "
        "check_return_eligibility(order_id=2670, order_item_ids=[3862])."
    )


def test_tool_correctness_ignores_extra_calls_and_order(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Extra calls and a different order still pass.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(
        ("list_my_orders", {}),
        ("check_return_eligibility", {"order_id": 2670}),
        ("get_order_details", {"order_id": 2670}),
    )
    result = _grade(ToolCorrectnessEvaluator(), make_run, make_example, messages, p8_expected())
    assert result.score is True


def test_tool_correctness_names_the_missing_tool(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """A missing tool fails, and the comment names it without arguments.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(("check_return_eligibility", {"order_id": 2670}))
    result = _grade(ToolCorrectnessEvaluator(), make_run, make_example, messages, p8_expected())
    assert result.score is False
    assert result.comment == (
        "Missing: get_order_details. Called: check_return_eligibility(order_id=2670)."
    )


def test_tool_correctness_ignores_arguments(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """The right tools with the wrong order id still pass this metric.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(
        ("get_order_details", {"order_id": 1}),
        ("check_return_eligibility", {"order_id": 1}),
    )
    result = _grade(ToolCorrectnessEvaluator(), make_run, make_example, messages, p8_expected())
    assert result.score is True


def test_tool_correctness_needs_one_call_per_expected_call(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Two expected calls of one tool need two calls; the comment says why it failed.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    expected = [
        {"name": "get_order_details", "args": {"order_id": 1}},
        {"name": "get_order_details", "args": {"order_id": 2}},
    ]
    messages = turn(("get_order_details", {"order_id": 1}))
    result = _grade(ToolCorrectnessEvaluator(), make_run, make_example, messages, expected)
    assert result.score is False
    assert result.comment.startswith("Each expected call matches, but not one call per")


def test_tool_correctness_reports_no_tools(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """A turn without tool calls fails and says that no tool was called.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    result = _grade(ToolCorrectnessEvaluator(), make_run, make_example, turn(), p8_expected())
    assert result.score is False
    assert result.comment.endswith("Called: no tools.")


def test_argument_correctness_fails_on_wrong_order_id(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """The right tools with a wrong order id fail, naming the expected call with its arguments.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(
        ("get_order_details", {"order_id": 2670}),
        ("check_return_eligibility", {"order_id": 2607, "order_item_ids": [3862]}),
    )
    result = _grade(ArgumentCorrectnessEvaluator(), make_run, make_example, messages, p8_expected())
    assert (result.key, result.score) == ("argument_correctness", False)
    assert result.comment.startswith(
        "Missing: check_return_eligibility(order_id=2670, order_item_ids=[3862]). Called: "
    )


@pytest.mark.parametrize(
    ("args", "passed"),
    [
        ({"order_id": 2670, "order_item_ids": [3862]}, True),
        ({"order_id": 2670, "order_item_ids": [3861, 3862]}, False),
        ({"order_id": 2670}, False),
        ({"order_id": "2670", "order_item_ids": [3862]}, False),
    ],
    ids=["exact", "extra-item", "all-items", "string-id"],
)
def test_argument_correctness_compares_values_exactly(
    make_run: RunFactory, make_example: ExampleFactory, args: dict[str, Any], passed: bool
) -> None:
    """Expected values must be equal: other item lists and a string order id fail.

    Args:
        make_run: Run builder.
        make_example: Example builder.
        args: Arguments of the ``check_return_eligibility`` call.
        passed: Expected score.
    """
    messages = turn(("get_order_details", {"order_id": 2670}), ("check_return_eligibility", args))
    result = _grade(ArgumentCorrectnessEvaluator(), make_run, make_example, messages, p8_expected())
    assert result.score is passed


def test_argument_correctness_allows_extra_and_unspecified_arguments(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Extra arguments pass, and an expected call without args matches any arguments.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    expected = [
        {"name": "list_my_orders", "args": {}},
        {"name": "get_order_details", "args": {"order_id": 2670}},
    ]
    messages = turn(
        ("list_my_orders", {"status": "Complete", "limit": 5}),
        ("get_order_details", {"order_id": 2670, "verbose": True}),
    )
    result = _grade(ArgumentCorrectnessEvaluator(), make_run, make_example, messages, expected)
    assert result.score is True


@pytest.mark.parametrize("evaluator_cls", [ToolCorrectnessEvaluator, ArgumentCorrectnessEvaluator])
def test_trajectory_metrics_follow_the_shared_policy(
    evaluator_cls: type[TrajectoryMatchEvaluator],
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """An errored run fails, and a case that doesn't list the metric gets no result.

    Args:
        evaluator_cls: Trajectory evaluator under test.
        make_run: Run builder.
        make_example: Example builder.
    """
    evaluator = evaluator_cls()
    example = make_example(metrics=[evaluator.key])
    (result,) = _results(evaluator(make_run(error="GraphRecursionError"), example))
    assert (result.key, result.score) == (evaluator.key, False)
    assert _results(evaluator(make_run(), make_example())) == []


@pytest.mark.parametrize("evaluator_cls", [ToolCorrectnessEvaluator, ArgumentCorrectnessEvaluator])
def test_langsmith_accepts_trajectory_evaluator(
    evaluator_cls: type[TrajectoryMatchEvaluator],
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """Contract: LangSmith wraps a trajectory evaluator and keeps our key and score.

    Args:
        evaluator_cls: Trajectory evaluator under test.
        make_run: Run builder.
        make_example: Example builder.
    """
    evaluator = evaluator_cls()
    run = make_run(messages=turn(("check_return_eligibility", {"order_id": 7})))
    output = run_evaluator(evaluator).evaluate_run(run, make_example(metrics=[evaluator.key]))
    (result,) = _results(output)
    assert (result.key, result.score) == (evaluator.key, True)


def test_factory_builds_trajectory_metrics_without_judge(
    mocker: MockerFixture, today: datetime
) -> None:
    """Code-only metrics build without a judge, so they need no judge key.

    Args:
        mocker: Pytest-mock fixture.
        today: TODAY of the test shop.
    """
    build_judge = mocker.patch.object(factory_module, "build_judge", autospec=True)
    built = EvaluatorFactory.build(["tool_correctness", "argument_correctness"], today)
    assert [type(e) for e in built] == [ToolCorrectnessEvaluator, ArgumentCorrectnessEvaluator]
    build_judge.assert_not_called()


def test_required_reference_names_reference_fields() -> None:
    """Every metric's required reference fields exist on ``Reference``, so a typo can't hide."""
    for name in EvaluatorFactory.metric_names():
        assert set(EvaluatorFactory.required_reference(name)) <= set(Reference.model_fields)
    assert EvaluatorFactory.required_reference("tool_correctness") == ("expected_tool_calls",)
