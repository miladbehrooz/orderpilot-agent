"""Labelled turns: the verdict each judge should give, from the code metrics.

The code counterparts on stored runs give *silver* labels (``docs/evals.md``, section 6).
A disagreement between a judge and its silver label is a lead for review, not a judge error
by itself: a different path can be equally valid, and the reference can be the one that's
wrong.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evals.calibration.stored import StoredTurn
from evals.evaluators.messages import tool_calls
from evals.evaluators.turn import Turn

LabelSource = Literal["silver", "built"]
"""``silver``: a stored run, labelled by the code metrics. ``built``: a turn changed on
purpose (``mutations.py``), labelled by construction."""


@dataclass(frozen=True, slots=True)
class LabelledTurn:
    """A turn with the verdict each judge should give on it.

    Attributes:
        item_id: Stable id: ``P4-01#<run id tail>`` (silver) or ``P4-01+<mutation>`` (built).
            Run ids are UUIDv7, whose first digits are a timestamp shared by runs made
            close together, so the silver id uses the random last 12 digits.
        case_id: Hand-written case id.
        source: Where the labels come from.
        turn: The turn to grade.
        labels: Judge metric to its expected verdict. Only these metrics are graded.
    """

    item_id: str
    case_id: str
    source: LabelSource
    turn: Turn
    labels: Mapping[str, bool]


def silver_labels(stored: StoredTurn) -> dict[str, bool]:
    """Return the labels the code counterparts give on a stored run.

    - ``tool_selection`` follows ``tool_correctness``;
    - ``argument_quality`` follows ``argument_correctness``;
    - ``trajectory_efficiency`` passes when the turn makes at most as many tool calls as
      the reference expects. Fewer calls are not inefficient: a missing call fails
      ``tool_selection`` instead.

    The tool calls are counted from the stored turn, as ``step_count`` counts them, so
    experiments recorded before ``step_count`` existed get this label too. The other two
    need the reference matcher, so they come from the run's feedback.

    ``answer_correctness`` has no code counterpart, so it gets no silver label.

    Args:
        stored: The stored run with its feedback.

    Returns:
        Judge metric to label, for each counterpart that has feedback on the run (or, for
        ``trajectory_efficiency``, whose reference expects tool calls).
    """
    feedback = stored.feedback
    labels: dict[str, bool] = {}
    if "tool_correctness" in feedback:
        labels["tool_selection"] = feedback["tool_correctness"] == 1
    if "argument_correctness" in feedback:
        labels["argument_quality"] = feedback["argument_correctness"] == 1
    expected_calls = len(stored.turn.reference.get("expected_tool_calls", []))
    if expected_calls:
        labels["trajectory_efficiency"] = len(tool_calls(stored.turn.messages)) <= expected_calls
    return labels


def silver_items(turns: Sequence[StoredTurn]) -> list[LabelledTurn]:
    """Label every stored run that has at least one silver label.

    Args:
        turns: The stored runs.

    Returns:
        One labelled turn per labelled run, in the given order.
    """
    items = []
    for stored in turns:
        labels = silver_labels(stored)
        if labels:
            items.append(
                LabelledTurn(
                    item_id=f"{stored.case_id}#{stored.run_id[-12:]}",
                    case_id=stored.case_id,
                    source="silver",
                    turn=stored.turn,
                    labels=labels,
                )
            )
    return items


def base_turns(silver: Sequence[LabelledTurn]) -> list[LabelledTurn]:
    """Return, per case, the first run that passes every silver label.

    Built failures are made from these, so each change starts from a turn known to pass.

    Args:
        silver: The silver items.

    Returns:
        At most one item per case, in first-seen order. A case without a passing run has
        none.
    """
    bases: dict[str, LabelledTurn] = {}
    for item in silver:
        if item.case_id not in bases and all(item.labels.values()):
            bases[item.case_id] = item
    return list(bases.values())


__all__ = ["LabelSource", "LabelledTurn", "base_turns", "silver_items", "silver_labels"]
