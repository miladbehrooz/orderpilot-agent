"""Silver labels: the code counterparts' feedback becomes the judges' expected verdicts."""

import pytest

from evals.calibration.labels import LabelledTurn, base_turns, silver_items, silver_labels
from evals.calibration.stored import StoredTurn
from evals.evaluators import Turn


def _stored(
    feedback: dict[str, float], expected_calls: int = 1, case_id: str = "P3-01"
) -> StoredTurn:
    """Build a stored run with the given feedback.

    Args:
        feedback: Feedback key to score.
        expected_calls: Number of expected tool calls in the reference.
        case_id: The run's case.

    Returns:
        The stored run.
    """
    expected = [{"name": "check_return_eligibility", "args": {}}] * expected_calls
    turn = Turn("Can I return order 7?", "Yes.", (), {"expected_tool_calls": expected})
    return StoredTurn(case_id=case_id, run_id="0123456789abcdef", turn=turn, feedback=feedback)


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


@pytest.mark.parametrize(("steps", "efficient"), [(1.0, True), (2.0, True), (3.0, False)])
def test_efficiency_label_allows_up_to_the_expected_number_of_calls(
    steps: float, efficient: bool
) -> None:
    """Fewer calls than expected are not inefficient; more are.

    Args:
        steps: The run's ``step_count``.
        efficient: The expected ``trajectory_efficiency`` label.
    """
    stored = _stored({"step_count": steps}, expected_calls=2)
    assert silver_labels(stored) == {"trajectory_efficiency": efficient}


def test_silver_labels_need_feedback_and_expected_calls() -> None:
    """Without counterpart feedback, or without expected calls to count against, no label."""
    assert silver_labels(_stored({})) == {}
    assert silver_labels(_stored({"step_count": 1.0}, expected_calls=0)) == {}


def test_silver_items_label_only_runs_with_labels() -> None:
    """A run without counterpart feedback is not an item; ids carry the run id prefix."""
    items = silver_items([_stored({"tool_correctness": 1.0}), _stored({"answer_correctness": 1.0})])
    assert [(i.item_id, i.source, dict(i.labels)) for i in items] == [
        ("P3-01#01234567", "silver", {"tool_selection": True})
    ]


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
