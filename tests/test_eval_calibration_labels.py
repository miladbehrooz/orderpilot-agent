"""Silver labels: the code counterparts become the judges' expected verdicts."""

from typing import Any

import pytest

from evals.calibration.labels import LabelledTurn, base_turns, silver_items, silver_labels
from evals.calibration.stored import StoredTurn
from evals.evaluators import Turn


def _stored(
    feedback: dict[str, float],
    expected_calls: int = 1,
    calls: int = 1,
    case_id: str = "P3-01",
    run_id: str = "019a1f08-0b2c-7d3e-8f40-a1b2c3d4e5f6",
) -> StoredTurn:
    """Build a stored run with the given feedback and number of tool calls.

    Args:
        feedback: Feedback key to score.
        expected_calls: Number of expected tool calls in the reference.
        calls: Number of tool calls the turn makes.
        case_id: The run's case.
        run_id: The run's id (a UUIDv7, as LangSmith gives).

    Returns:
        The stored run.
    """
    expected = [{"name": "check_return_eligibility", "args": {}}] * expected_calls
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Can I return order 7?"}]
    for index in range(calls):
        function = {"name": "check_return_eligibility", "arguments": '{"order_id": 7}'}
        call = {"type": "function", "id": f"call-{index}", "function": function}
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"call-{index}", "content": '{"reason_code": "OK"}'},
        ]
    messages.append({"role": "assistant", "content": "Yes."})
    turn = Turn("Can I return order 7?", "Yes.", tuple(messages), {"expected_tool_calls": expected})
    return StoredTurn(case_id=case_id, run_id=run_id, turn=turn, feedback=feedback)


def test_silver_labels_follow_the_code_counterparts() -> None:
    """Each judge gets the verdict of its code counterpart; ``answer_correctness`` none."""
    stored = _stored(
        {
            "tool_correctness": 1.0,
            "argument_correctness": 0.0,
            "step_count": 1.0,
            "answer_correctness": 1.0,
        }
    )
    assert silver_labels(stored) == {
        "tool_selection": True,
        "argument_quality": False,
        "trajectory_efficiency": True,
    }


@pytest.mark.parametrize(("calls", "efficient"), [(1, True), (2, True), (3, False)])
def test_efficiency_label_allows_up_to_the_expected_number_of_calls(
    calls: int, efficient: bool
) -> None:
    """Fewer calls than expected are not inefficient; more are.

    Args:
        calls: Tool calls the turn makes.
        efficient: The expected ``trajectory_efficiency`` label.
    """
    stored = _stored({}, expected_calls=2, calls=calls)
    assert silver_labels(stored) == {"trajectory_efficiency": efficient}


def test_efficiency_label_counts_the_turn_not_the_feedback() -> None:
    """Calls are counted from the turn, so runs recorded before ``step_count`` get the label.

    A stale or missing ``step_count`` score doesn't change it.
    """
    assert silver_labels(_stored({}, calls=2)) == {"trajectory_efficiency": False}
    assert silver_labels(_stored({"step_count": 1.0}, calls=2)) == {"trajectory_efficiency": False}


def test_silver_labels_need_feedback_and_expected_calls() -> None:
    """Without counterpart feedback, or without expected calls to count against, no label."""
    assert silver_labels(_stored({}, expected_calls=0)) == {}


def test_silver_items_label_only_runs_with_labels() -> None:
    """A run without counterpart feedback is not an item."""
    items = silver_items(
        [
            _stored({"tool_correctness": 1.0}, expected_calls=0),
            _stored({"answer_correctness": 1.0}, expected_calls=0),
        ]
    )
    assert [(i.item_id, i.source, dict(i.labels)) for i in items] == [
        ("P3-01#a1b2c3d4e5f6", "silver", {"tool_selection": True})
    ]


def test_silver_item_ids_stay_unique_for_runs_made_together() -> None:
    """UUIDv7 run ids made close together share their first digits, never their tail."""
    runs = [
        "019a1f08-0b2c-7d3e-8f40-a1b2c3d4e5f6",
        "019a1f08-0b2d-7a11-9c22-0f1e2d3c4b5a",
    ]
    items = silver_items([_stored({}, run_id=run) for run in runs])
    assert len({item.item_id for item in items}) == 2


def test_base_turns_take_the_first_fully_passing_run_per_case() -> None:
    """A case's base is its first run passing every label; a case without one has none."""
    turn = Turn("q", "a", ())

    def item(case_id: str, run: str, passed: bool) -> LabelledTurn:
        """Build a silver item.

        Args:
            case_id: Its case.
            run: Run suffix of its id.
            passed: Its ``tool_correctness`` label.

        Returns:
            The item.
        """
        labels = {"tool_selection": True, "argument_quality": passed}
        return LabelledTurn(f"{case_id}#{run}", case_id, "silver", turn, labels)

    silver = [
        item("P3-01", "a", False),
        item("P3-01", "b", True),
        item("P3-01", "c", True),
        item("P4-01", "a", False),
    ]
    assert [b.item_id for b in base_turns(silver)] == ["P3-01#b"]
