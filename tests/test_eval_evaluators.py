"""Evaluators: error policy, every metric, the factory, the LangSmith contract.

The judge model is never called: ``create_llm_as_judge`` and
``create_trajectory_llm_as_judge`` are mocked where the judge base classes look them up.
The code-graded metrics run on hand-built turns (the trajectory metrics with the real
openevals matcher).
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

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
from pydantic import ValidationError
from pytest_mock import MockerFixture, MockType

from evals.cases import Reference
from evals.errors import EvalSetupError
from evals.evaluators import (
    AnswerCorrectnessEvaluator,
    ArgumentCorrectnessEvaluator,
    ArgumentQualityEvaluator,
    EvaluatorFactory,
    JudgeVerdict,
    ReasonCodeEvaluator,
    StepCountEvaluator,
    ToolCorrectnessEvaluator,
    ToolSelectionEvaluator,
    TrajectoryEfficiencyEvaluator,
    TrajectoryJudgeEvaluator,
    TrajectoryMatchEvaluator,
)
from evals.evaluators import factory as factory_module
from evals.evaluators.judges import llm_judge as llm_judge_module
from evals.evaluators.judges import trajectory_judge as trajectory_judge_module
from evals.evaluators.judges.prompts import JUDGE_PROMPTS, JudgePrompt, prompt_for
from tests.eval_runs import ExampleFactory, RunFactory
from tests.stubs import StubRepository


@pytest.fixture
def judge_fn(mocker: MockerFixture) -> MockType:
    """Mock ``create_llm_as_judge`` and return the judge function it builds.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked judge callable; it returns a passing verdict unless changed.
    """
    create: MockType = mocker.patch.object(llm_judge_module, "create_llm_as_judge", autospec=True)
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
    mocker: MockerFixture, judge_fn: MockType, stub_repo: StubRepository
) -> None:
    """The judge is built once for judge metrics, and never for an empty selection.

    Args:
        mocker: Pytest-mock fixture.
        judge_fn: The mocked judge callable (keeps the real judge out of the evaluator).
        stub_repo: Repository on the test shop.
    """
    build_judge = mocker.patch.object(factory_module, "build_judge", autospec=True)
    build_judge.return_value = mocker.create_autospec(BaseChatModel, instance=True)

    assert EvaluatorFactory.build([], stub_repo) == []
    build_judge.assert_not_called()

    (built,) = EvaluatorFactory.build(["answer_correctness"], stub_repo)
    assert isinstance(built, AnswerCorrectnessEvaluator)
    assert built.__name__ == "answer_correctness"
    assert built.today == "2026-11-12"
    build_judge.assert_called_once_with()


def test_factory_rejects_unknown_metric(stub_repo: StubRepository) -> None:
    """An unknown metric name fails with the list of valid metric names.

    Args:
        stub_repo: Repository on the test shop.
    """
    with pytest.raises(EvalSetupError, match="answer_correctness"):
        EvaluatorFactory.build(["answer_corectness"], stub_repo)


def test_factory_reports_judge_prompt_versions() -> None:
    """Prompt versions of the selected judge metrics go into the experiment metadata."""
    assert EvaluatorFactory.prompt_versions(["answer_correctness"]) == {"answer_correctness": "v2"}
    assert EvaluatorFactory.prompt_versions(["tool_correctness", "argument_correctness"]) == {}
    assert EvaluatorFactory.prompt_versions(
        ["tool_selection", "argument_quality", "trajectory_efficiency"]
    ) == {"tool_selection": "v1", "argument_quality": "v2", "trajectory_efficiency": "v1"}
    assert EvaluatorFactory.metric_names() == [
        "answer_correctness",
        "tool_correctness",
        "argument_correctness",
        "reason_code",
        "step_count",
        "tool_selection",
        "argument_quality",
        "trajectory_efficiency",
    ]


def turn(
    *calls: tuple[str, dict[str, Any]], results: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Build one agent turn in OpenAI format, as the target returns it.

    Args:
        *calls: ``(tool name, args)`` of each tool call, one AI message per call.
        results: Tool result content per call; ``{"reason_code": "OK"}`` for each if omitted.

    Returns:
        The question, each call with its tool result, and a final answer.
    """
    contents = list(results) if results is not None else ['{"reason_code": "OK"}'] * len(calls)
    messages: list[AnyMessage] = [HumanMessage("Can I return order 2670?")]
    for index, ((name, args), content) in enumerate(zip(calls, contents, strict=True)):
        call_id = f"call-{index}"
        messages.append(AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}]))
        # No name on the tool message: results are matched to their call by tool_call_id.
        messages.append(ToolMessage(content, tool_call_id=call_id))
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


@pytest.mark.parametrize(
    "evaluator_cls", [ToolCorrectnessEvaluator, ArgumentCorrectnessEvaluator, ReasonCodeEvaluator]
)
def test_code_metrics_follow_the_shared_policy(
    evaluator_cls: type[TrajectoryMatchEvaluator | ReasonCodeEvaluator],
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """An errored run fails, and a case that doesn't list the metric gets no result.

    Args:
        evaluator_cls: Code-graded pass/fail evaluator under test.
        make_run: Run builder.
        make_example: Example builder.
    """
    evaluator = evaluator_cls()
    example = make_example(metrics=[evaluator.key])
    (result,) = _results(evaluator(make_run(error="GraphRecursionError"), example))
    assert (result.key, result.score) == (evaluator.key, False)
    assert _results(evaluator(make_run(), make_example())) == []


@pytest.mark.parametrize(
    ("evaluator_cls", "score"),
    [
        (ToolCorrectnessEvaluator, True),
        (ArgumentCorrectnessEvaluator, True),
        (ReasonCodeEvaluator, True),
        (StepCountEvaluator, 1),
    ],
)
def test_langsmith_accepts_code_evaluator(
    evaluator_cls: type[TrajectoryMatchEvaluator | ReasonCodeEvaluator | StepCountEvaluator],
    score: bool | int,
    make_run: RunFactory,
    make_example: ExampleFactory,
) -> None:
    """Contract: LangSmith wraps a code-graded evaluator and keeps our key and score.

    Args:
        evaluator_cls: Code-graded evaluator under test.
        score: The score expected for one eligible ``check_return_eligibility`` call.
        make_run: Run builder.
        make_example: Example builder.
    """
    evaluator = evaluator_cls()
    run = make_run(
        messages=turn(
            ("check_return_eligibility", {"order_id": 7}), results=['{"reason_code": "ELIGIBLE"}']
        )
    )
    output = run_evaluator(evaluator).evaluate_run(run, make_example(metrics=[evaluator.key]))
    (result,) = _results(output)
    assert (result.key, result.score) == (evaluator.key, score)


def test_factory_builds_code_metrics_without_judge(
    mocker: MockerFixture, stub_repo: StubRepository
) -> None:
    """Code-only metrics build without a judge or a contract, so they need no judge key.

    Args:
        mocker: Pytest-mock fixture.
        stub_repo: Repository on the test shop.
    """
    build_judge = mocker.patch.object(factory_module, "build_judge", autospec=True)
    contract = mocker.spy(factory_module, "agent_contract")
    metrics = ["tool_correctness", "argument_correctness", "reason_code", "step_count"]
    built = EvaluatorFactory.build(metrics, stub_repo)
    assert [type(e) for e in built] == [
        ToolCorrectnessEvaluator,
        ArgumentCorrectnessEvaluator,
        ReasonCodeEvaluator,
        StepCountEvaluator,
    ]
    build_judge.assert_not_called()
    contract.assert_not_called()


def test_required_reference_names_reference_fields() -> None:
    """Every metric's required reference fields exist on ``Reference``, so a typo can't hide."""
    for name in EvaluatorFactory.metric_names():
        assert set(EvaluatorFactory.required_reference(name)) <= set(Reference.model_fields)
    assert EvaluatorFactory.required_reference("tool_correctness") == ("expected_tool_calls",)
    assert EvaluatorFactory.required_reference("reason_code") == ("expected_reason_code",)


def _grade_reason_code(
    make_run: RunFactory,
    make_example: ExampleFactory,
    messages: list[dict[str, Any]],
    expected: str = "ELIGIBLE",
) -> Any:
    """Grade one turn on ``reason_code`` and return its single result.

    Args:
        make_run: Run builder.
        make_example: Example builder.
        messages: The turn in OpenAI format.
        expected: Code ``check_return_eligibility`` must return.

    Returns:
        The one ``EvaluationResult``.
    """
    evaluator = ReasonCodeEvaluator()
    example = make_example(metrics=[evaluator.key], expected_reason_code=expected)
    (result,) = _results(evaluator(make_run(messages=messages), example))
    return result


def _check(order_id: int = 4481) -> tuple[str, dict[str, Any]]:
    """Return a ``check_return_eligibility`` call for one order.

    Args:
        order_id: The order checked.

    Returns:
        ``(tool name, args)`` for ``turn``.
    """
    return ("check_return_eligibility", {"order_id": order_id})


def _code(code: str) -> str:
    """Return tool result content with a reason code.

    Args:
        code: The reason code.

    Returns:
        The JSON content.
    """
    return f'{{"reason_code": "{code}", "order_id": 4481}}'


def test_reason_code_passes_on_expected_code(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """The expected code from the expected tool passes, and the comment names it.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(_check(), results=[_code("WINDOW_EXPIRED")])
    result = _grade_reason_code(make_run, make_example, messages, "WINDOW_EXPIRED")
    assert (result.key, result.score) == ("reason_code", True)
    assert result.comment == "check_return_eligibility returned WINDOW_EXPIRED."


def test_reason_code_fails_on_other_code_and_names_both(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Day 31 reported as eligible fails, with the actual and the expected code.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(_check(), results=[_code("ELIGIBLE")])
    result = _grade_reason_code(make_run, make_example, messages, "WINDOW_EXPIRED")
    assert result.score is False
    assert result.comment == "check_return_eligibility returned ELIGIBLE, expected WINDOW_EXPIRED."


def test_reason_code_grades_the_last_result_of_the_tool(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """A retry after a wrong order id is graded on the final check.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(_check(4418), _check(), results=[_code("NOT_FOUND"), _code("ELIGIBLE")])
    result = _grade_reason_code(make_run, make_example, messages)
    assert result.score is True
    assert "(last of 2 results)" in result.comment


def test_reason_code_ignores_results_of_other_tools(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Another tool's result, even with the right code, is never graded.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = turn(
        _check(),
        ("get_order_details", {"order_id": 4481}),
        results=[_code("ELIGIBLE"), _code("OK")],
    )
    assert _grade_reason_code(make_run, make_example, messages).score is True
    messages = turn(
        ("check_cancellation_eligibility", {"order_id": 1}), results=[_code("ELIGIBLE")]
    )
    assert _grade_reason_code(make_run, make_example, messages).score is False


def test_reason_code_fails_without_a_result_from_the_tool(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """A turn that never called the tool fails and says so.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    result = _grade_reason_code(make_run, make_example, turn())
    assert result.score is False
    assert result.comment == "No check_return_eligibility result."


@pytest.mark.parametrize(
    "content", ["Traceback (most recent call last)", "[]", '{"eligible": true}']
)
def test_reason_code_fails_on_unreadable_tool_content(
    make_run: RunFactory, make_example: ExampleFactory, content: str
) -> None:
    """A result without a readable reason code fails the metric; it is not an evaluator error.

    Args:
        make_run: Run builder.
        make_example: Example builder.
        content: Tool result content that breaks the tool contract.
    """
    result = _grade_reason_code(make_run, make_example, turn(_check(), results=[content]))
    assert (result.key, result.score) == ("reason_code", False)
    assert "no readable reason_code" in result.comment


def test_step_count_counts_tool_calls_and_model_calls(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """The score is the number of tool calls; the comment adds the model calls.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    evaluator = StepCountEvaluator()
    messages = turn(("get_order_details", {"order_id": 2670}), _check(2670))
    example = make_example(metrics=[evaluator.key])
    (result,) = _results(evaluator(make_run(messages=messages), example))
    assert (result.key, result.score) == ("step_count", 2)
    assert result.comment == "2 tool calls in 3 model calls."


def test_measure_emits_nothing_for_an_errored_run(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """A crashed run has no step count: no result, rather than a misleading zero.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    evaluator = StepCountEvaluator()
    example = make_example(metrics=[evaluator.key])
    assert _results(evaluator(make_run(error="GraphRecursionError"), example)) == []


# Trajectory judges: reference-free, run by the openevals trajectory judge (mocked).

TRAJECTORY_JUDGES = [
    ToolSelectionEvaluator,
    ArgumentQualityEvaluator,
    TrajectoryEfficiencyEvaluator,
]


@pytest.fixture
def trajectory_judge_fn(mocker: MockerFixture) -> MockType:
    """Mock ``create_trajectory_llm_as_judge`` and return the judge function it builds.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked judge callable; it returns a passing openevals result unless changed.
    """
    create: MockType = mocker.patch.object(
        trajectory_judge_module, "create_trajectory_llm_as_judge", autospec=True
    )
    judge: MockType = create.return_value
    judge.return_value = {"key": "x", "score": True, "comment": "Called the eligibility tool."}
    return judge


def _trajectory_judge(
    mocker: MockerFixture, evaluator_cls: type[TrajectoryJudgeEvaluator], today: datetime
) -> TrajectoryJudgeEvaluator:
    """Build a trajectory judge with a mocked judge model and a short contract.

    Args:
        mocker: Pytest-mock fixture.
        evaluator_cls: Trajectory judge under test.
        today: TODAY of the test shop.

    Returns:
        The evaluator.
    """
    model = mocker.create_autospec(BaseChatModel, instance=True)
    return evaluator_cls(model, today, "CONTRACT: call an eligibility tool first.")


@pytest.mark.parametrize("evaluator_cls", TRAJECTORY_JUDGES)
@pytest.mark.parametrize("verdict", [True, False])
def test_trajectory_judge_maps_score_and_comment(
    mocker: MockerFixture,
    trajectory_judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
    today: datetime,
    evaluator_cls: type[TrajectoryJudgeEvaluator],
    verdict: bool,
) -> None:
    """The openevals score becomes the result's score, its comment the reasoning.

    Args:
        mocker: Pytest-mock fixture.
        trajectory_judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
        today: TODAY of the test shop.
        evaluator_cls: Trajectory judge under test.
        verdict: The judge's verdict.
    """
    trajectory_judge_fn.return_value = {"key": "x", "score": verdict, "comment": "Checked."}
    evaluator = _trajectory_judge(mocker, evaluator_cls, today)
    run = make_run(messages=turn(("check_return_eligibility", {"order_id": 7})))
    (result,) = _results(evaluator(run, make_example(metrics=[evaluator.key])))
    assert (result.key, result.score, result.comment) == (evaluator.key, verdict, "Checked.")
    assert result.metadata == {"prompt_version": evaluator.prompt_version}


@pytest.mark.parametrize("evaluator_cls", TRAJECTORY_JUDGES)
def test_trajectory_judge_sees_turn_and_contract_but_no_reference(
    mocker: MockerFixture,
    trajectory_judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
    today: datetime,
    evaluator_cls: type[TrajectoryJudgeEvaluator],
) -> None:
    """The judge gets today's date, the contract and the whole turn; never the reference.

    Args:
        mocker: Pytest-mock fixture.
        trajectory_judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
        today: TODAY of the test shop.
        evaluator_cls: Trajectory judge under test.
    """
    evaluator = _trajectory_judge(mocker, evaluator_cls, today)
    messages = turn(("check_return_eligibility", {"order_id": 7}))
    evaluator(make_run(messages=messages), make_example(metrics=[evaluator.key]))
    kwargs = trajectory_judge_fn.call_args.kwargs
    assert set(kwargs) == {"today", "outputs", "contract"}
    assert kwargs["today"] == "2026-11-12"
    assert kwargs["outputs"] == messages
    assert kwargs["contract"] == "CONTRACT: call an eligibility tool first."
    seen = str(kwargs)
    assert all(hidden not in seen for hidden in ("2026-12-04", "P3-01", "expected"))


@pytest.mark.parametrize("evaluator_cls", TRAJECTORY_JUDGES)
def test_trajectory_judges_follow_the_shared_policy(
    mocker: MockerFixture,
    trajectory_judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
    today: datetime,
    evaluator_cls: type[TrajectoryJudgeEvaluator],
) -> None:
    """Not listed: no result; errored run: fails without a judge call; broken judge: error.

    Args:
        mocker: Pytest-mock fixture.
        trajectory_judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
        today: TODAY of the test shop.
        evaluator_cls: Trajectory judge under test.
    """
    evaluator = _trajectory_judge(mocker, evaluator_cls, today)
    example = make_example(metrics=[evaluator.key])
    assert _results(evaluator(make_run(), make_example())) == []
    (failed,) = _results(evaluator(make_run(error="GraphRecursionError"), example))
    assert (failed.key, failed.score) == (evaluator.key, False)
    trajectory_judge_fn.assert_not_called()

    trajectory_judge_fn.side_effect = TimeoutError("judge timed out")
    (error,) = _results(evaluator(make_run(), example))
    assert (error.key, error.metadata) == ("evaluator_error", {"metric": evaluator.key})


@pytest.mark.parametrize("evaluator_cls", TRAJECTORY_JUDGES)
def test_trajectory_judge_prompt_has_every_placeholder(
    evaluator_cls: type[TrajectoryJudgeEvaluator],
) -> None:
    """Each trajectory prompt fills with exactly the variables the judge receives.

    Args:
        evaluator_cls: Trajectory judge under test.
    """
    prompt = prompt_for(evaluator_cls.key, evaluator_cls.prompt_version)
    filled = prompt.format(today="2026-11-12", contract="RULES", outputs="TURN")
    assert all(value in filled for value in ("2026-11-12", "RULES", "TURN"))


def test_langsmith_accepts_trajectory_judge(
    mocker: MockerFixture,
    trajectory_judge_fn: MockType,
    make_run: RunFactory,
    make_example: ExampleFactory,
    today: datetime,
) -> None:
    """Contract: LangSmith wraps a trajectory judge and keeps our key and score.

    Args:
        mocker: Pytest-mock fixture.
        trajectory_judge_fn: The mocked judge callable.
        make_run: Run builder.
        make_example: Example builder.
        today: TODAY of the test shop.
    """
    evaluator = _trajectory_judge(mocker, ToolSelectionEvaluator, today)
    example = make_example(metrics=["tool_selection"])
    (result,) = _results(run_evaluator(evaluator).evaluate_run(make_run(), example))
    assert (result.key, result.score) == ("tool_selection", True)


def test_factory_gives_trajectory_judges_the_contract(
    mocker: MockerFixture, trajectory_judge_fn: MockType, stub_repo: StubRepository
) -> None:
    """The contract is built once from the repository and shared by the trajectory judges.

    Args:
        mocker: Pytest-mock fixture.
        trajectory_judge_fn: The mocked judge callable (keeps the real judge out).
        stub_repo: Repository on the test shop.
    """
    build_judge = mocker.patch.object(factory_module, "build_judge", autospec=True)
    build_judge.return_value = mocker.create_autospec(BaseChatModel, instance=True)
    contract = mocker.spy(factory_module, "agent_contract")
    metrics = ["tool_selection", "argument_quality", "trajectory_efficiency", "step_count"]
    built = EvaluatorFactory.build(metrics, stub_repo)
    assert [type(e) for e in built] == [*TRAJECTORY_JUDGES, StepCountEvaluator]
    contract.assert_called_once_with(stub_repo)
    judges = [e for e in built if isinstance(e, TrajectoryJudgeEvaluator)]
    assert all(e.contract == contract.spy_return and e.today == "2026-11-12" for e in judges)
    build_judge.assert_called_once_with()
