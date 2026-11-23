"""Built failures: each change breaks its metric, keeps the others, and leaves the base alone.

Every mutation is checked against the real code counterpart of the metric it breaks
(``tool_correctness``, ``argument_correctness``, ``step_count``), so a label "fail by
construction" is backed by the same check the experiments run.
"""

import copy
import json
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
    convert_to_openai_messages,
)

from evals.calibration.labels import LabelledTurn
from evals.calibration.mutations import (
    CheckEveryItem,
    ContradictDecision,
    DropEligibilityCall,
    Mutation,
    RepeatCall,
    SwapOrderId,
    all_mutations,
    build_failures,
)
from evals.evaluators import (
    ArgumentCorrectnessEvaluator,
    Evaluator,
    StepCountEvaluator,
    ToolCorrectnessEvaluator,
    Turn,
)
from evals.evaluators.messages import tool_results
from tests.eval_runs import ExampleFactory, RunFactory


def _turn(
    calls: list[tuple[str, dict[str, Any], dict[str, Any]]],
    reference: dict[str, Any],
    question: str = "Can I return just the jeans from order 2670?",
) -> Turn:
    """Build a turn in OpenAI format: one AI message per call, its result, a final answer.

    Args:
        calls: ``(tool name, args, result)`` of each call; results are JSON objects.
        reference: The case's reference outputs.
        question: The customer's message.

    Returns:
        The turn.
    """
    messages: list[AnyMessage] = [HumanMessage(question)]
    for index, (name, args, result) in enumerate(calls):
        call_id = f"call-{index}"
        messages.append(AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}]))
        messages.append(ToolMessage(json.dumps(result), tool_call_id=call_id))
    messages.append(AIMessage("Yes, the jeans can be returned until 2026-12-04."))
    return Turn(
        question=question,
        answer="Yes, the jeans can be returned until 2026-12-04.",
        messages=tuple(convert_to_openai_messages(messages)),
        reference=reference,
    )


@pytest.fixture
def p8_turn() -> Turn:
    """Return a passing partial-return turn: look the order up, then check one item.

    Returns:
        The turn, with a reference expecting both calls and ``ELIGIBLE``.
    """
    details = {
        "reason_code": "OK",
        "order": {
            "order_id": 2670,
            "items": [
                {"order_item_id": 3861, "product_name": "T-shirt"},
                {"order_item_id": 3862, "product_name": "Jeans"},
            ],
        },
    }
    expected = [
        {"name": "get_order_details", "args": {"order_id": 2670}},
        {"name": "check_return_eligibility", "args": {"order_id": 2670, "order_item_ids": [3862]}},
    ]
    return _turn(
        [
            ("get_order_details", {"order_id": 2670}, details),
            (
                "check_return_eligibility",
                {"order_id": 2670, "order_item_ids": [3862]},
                {"reason_code": "ELIGIBLE", "order_id": 2670},
            ),
        ],
        reference={
            "expected_tool_calls": expected,
            "expected_reason_code": {"tool": "check_return_eligibility", "code": "ELIGIBLE"},
        },
    )


@pytest.fixture
def lookup_turn() -> Turn:
    """Return a passing "where's my order" turn: one order listing, no order id.

    Returns:
        The turn, with no expected reason code.
    """
    return _turn(
        [("list_my_orders", {}, {"reason_code": "OK", "orders": []})],
        reference={"expected_tool_calls": [{"name": "list_my_orders", "args": {}}]},
        question="Where's my order?",
    )


def _score(
    evaluator: Evaluator, turn: Turn, make_run: RunFactory, make_example: ExampleFactory
) -> Any:
    """Grade a turn with a code metric, against the turn's expected tool calls.

    Args:
        evaluator: The code evaluator.
        turn: The turn.
        make_run: Run builder.
        make_example: Example builder.

    Returns:
        The metric's score.
    """
    run = make_run(answer=turn.answer, messages=list(turn.messages))
    example = make_example(
        metrics=[evaluator.key], expected_tool_calls=turn.reference["expected_tool_calls"]
    )
    (result,) = evaluator(run, example)["results"]
    return result.score


def _args_of(turn: Turn, tool: str) -> list[dict[str, Any]]:
    """Return the arguments of every call to one tool.

    Args:
        turn: The turn.
        tool: Tool name.

    Returns:
        The parsed arguments, in order.
    """
    return [
        json.loads(call["function"]["arguments"])
        for message in turn.messages
        for call in message.get("tool_calls") or []
        if call["function"]["name"] == tool
    ]


def test_drop_eligibility_call_fails_tool_correctness(
    p8_turn: Turn, make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Without the eligibility call, the expected tools are no longer all called.

    Args:
        p8_turn: Passing partial-return turn.
        make_run: Run builder.
        make_example: Example builder.
    """
    mutated = DropEligibilityCall().apply(p8_turn)
    tool_correctness = ToolCorrectnessEvaluator()
    assert _score(tool_correctness, p8_turn, make_run, make_example) is True
    assert _score(tool_correctness, mutated, make_run, make_example) is False
    assert _args_of(mutated, "check_return_eligibility") == []
    assert tool_results(list(mutated.messages), "check_return_eligibility") == []
    assert mutated.answer == p8_turn.answer
    assert len(mutated.messages) == len(p8_turn.messages) - 2


def test_swap_order_id_fails_argument_correctness(
    p8_turn: Turn, make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """Every call points at the next order, so the expected arguments no longer match.

    Args:
        p8_turn: Passing partial-return turn.
        make_run: Run builder.
        make_example: Example builder.
    """
    mutated = SwapOrderId().apply(p8_turn)
    argument_correctness = ArgumentCorrectnessEvaluator()
    assert _score(argument_correctness, p8_turn, make_run, make_example) is True
    assert _score(argument_correctness, mutated, make_run, make_example) is False
    assert _args_of(mutated, "get_order_details") == [{"order_id": 2671}]


def test_check_every_item_checks_all_looked_up_items(
    p8_turn: Turn, make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """The partial check becomes a check of every item the order lookup returned.

    Args:
        p8_turn: Passing partial-return turn.
        make_run: Run builder.
        make_example: Example builder.
    """
    mutated = CheckEveryItem().apply(p8_turn)
    assert _args_of(mutated, "check_return_eligibility") == [
        {"order_id": 2670, "order_item_ids": [3861, 3862]}
    ]
    assert _score(ArgumentCorrectnessEvaluator(), mutated, make_run, make_example) is False


def test_check_every_item_needs_a_partial_check(lookup_turn: Turn) -> None:
    """A turn that checks no single items can't be changed into checking all of them.

    Args:
        lookup_turn: Passing order-listing turn.
    """
    assert not CheckEveryItem().applies_to(lookup_turn)


def test_repeat_call_adds_one_paired_call(
    p8_turn: Turn, make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """The last call is made again with the same arguments and its own result.

    Args:
        p8_turn: Passing partial-return turn.
        make_run: Run builder.
        make_example: Example builder.
    """
    mutated = RepeatCall().apply(p8_turn)
    step_count = StepCountEvaluator()
    assert _score(step_count, mutated, make_run, make_example) == (
        _score(step_count, p8_turn, make_run, make_example) + 1
    )
    calls = _args_of(mutated, "check_return_eligibility")
    assert calls == [{"order_id": 2670, "order_item_ids": [3862]}] * 2
    assert len(tool_results(list(mutated.messages), "check_return_eligibility")) == 2
    ids = [c["id"] for m in mutated.messages for c in m.get("tool_calls") or []]
    assert len(ids) == len(set(ids))
    assert mutated.messages[-1] == p8_turn.messages[-1]


@pytest.mark.parametrize(
    ("tool", "code", "expected"),
    [
        ("check_return_eligibility", "ELIGIBLE", "order 2670 can no longer be returned"),
        ("check_return_eligibility", "OUTSIDE_RETURN_WINDOW", "order 2670 can still be returned"),
        ("check_cancellation_eligibility", "ALREADY_SHIPPED", "order 2670 can still be cancelled"),
    ],
)
def test_contradict_decision_states_the_opposite(tool: str, code: str, expected: str) -> None:
    """The new answer reverses the expected decision, in the answer and the last message.

    Args:
        tool: The reference's eligibility tool.
        code: The reference's expected reason code.
        expected: Text the new answer must contain.
    """
    turn = _turn(
        [(tool, {"order_id": 2670}, {"reason_code": code})],
        reference={"expected_reason_code": {"tool": tool, "code": code}},
    )
    mutated = ContradictDecision().apply(turn)
    assert expected in mutated.answer
    assert mutated.messages[-1] == {"role": "assistant", "content": mutated.answer}
    assert mutated.messages[:-1] == turn.messages[:-1]


def test_contradict_decision_needs_an_expected_decision(lookup_turn: Turn) -> None:
    """A case without ``expected_reason_code`` has no decision to contradict.

    Args:
        lookup_turn: Passing order-listing turn.
    """
    assert not ContradictDecision().applies_to(lookup_turn)


@pytest.mark.parametrize("mutation", all_mutations(), ids=lambda m: m.name)
def test_mutations_leave_the_base_turn_untouched(p8_turn: Turn, mutation: Mutation) -> None:
    """A change returns a new turn; the stored one can be changed again by the next mutation.

    Args:
        p8_turn: Passing partial-return turn.
        mutation: The change under test.
    """
    before = copy.deepcopy(p8_turn.messages)
    assert mutation.applies_to(p8_turn)
    assert mutation.apply(p8_turn) != p8_turn
    assert p8_turn.messages == before


def test_built_item_breaks_its_metric_and_keeps_the_base_labels(p8_turn: Turn) -> None:
    """The built label fails the broken metric; the kept metrics carry the base's labels.

    Args:
        p8_turn: Passing partial-return turn.
    """
    base = LabelledTurn(
        item_id="P8-01#abcdef12",
        case_id="P8-01",
        source="silver",
        turn=p8_turn,
        labels={"tool_selection": True, "argument_quality": True},
    )
    built = SwapOrderId().build(base)
    assert built is not None
    assert (built.item_id, built.case_id, built.source) == ("P8-01+swap_order_id", "P8-01", "built")
    assert built.labels == {"tool_selection": True, "argument_quality": False}


def test_build_failures_applies_only_the_mutations_that_fit(lookup_turn: Turn) -> None:
    """An order listing has no order id, item check or decision: only a repeat fits.

    Args:
        lookup_turn: Passing order-listing turn.
    """
    base = LabelledTurn("L1-01#abcdef12", "L1-01", "silver", lookup_turn, {"tool_selection": True})
    assert [item.item_id for item in build_failures([base])] == ["L1-01+repeat_call"]
