"""Agreement maths and the calibration report: Wilson, kappa, majority verdicts, flips."""

import json

import pytest

from evals.calibration.grading import Grade
from evals.calibration.labels import LabelledTurn
from evals.calibration.report import (
    agreement,
    format_counts,
    format_report,
    summarize,
)
from evals.calibration.stats import cohen_kappa, kappa_interval, wilson
from evals.evaluators import Turn


def _grade(
    label: bool,
    verdicts: tuple[bool, ...],
    source: str = "built",
    metric: str = "tool_selection",
    errors: int = 0,
    item_id: str = "P3-01+x",
) -> Grade:
    """Build a grade.

    Args:
        label: Expected verdict.
        verdicts: The judge's verdicts.
        source: Label source.
        metric: Judge metric.
        errors: Failed judge calls.
        item_id: The item.

    Returns:
        The grade, with one reasoning per verdict.
    """
    return Grade(
        item_id=item_id,
        case_id=item_id[:5],
        source=source,  # type: ignore[arg-type]
        metric=metric,
        label=label,
        verdicts=verdicts,
        reasonings=tuple(f"reason {i}" for i in range(len(verdicts))),
        errors=errors,
    )


def test_wilson_interval_matches_the_formula() -> None:
    """8 of 10 gives the textbook Wilson interval, and 0 of 5 starts at zero."""
    assert wilson(8, 10) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert wilson(0, 5)[0] == 0.0


def test_wilson_needs_trials() -> None:
    """No trials, or more successes than trials, is an error."""
    with pytest.raises(ValueError, match="n > 0"):
        wilson(0, 0)
    with pytest.raises(ValueError, match="successes"):
        wilson(3, 2)


def test_kappa_on_a_hand_computed_table() -> None:
    """TP 20, FN 5, FP 10, TN 15: observed 0.7, chance 0.5, so kappa 0.4."""
    pairs = [(True, True)] * 20 + [(True, False)] * 5 + [(False, True)] * 10 + [(False, False)] * 15
    assert cohen_kappa(pairs) == pytest.approx(0.4)


def test_kappa_is_undefined_without_variation() -> None:
    """All labels and verdicts the same value: chance agreement is 1, kappa undefined."""
    assert cohen_kappa([(True, True)] * 5) is None
    assert cohen_kappa([]) is None
    assert kappa_interval([(True, True)] * 5) is None


def test_kappa_interval_is_reproducible_and_contains_kappa() -> None:
    """The same seed gives the same interval, and it brackets the point estimate."""
    pairs = [(True, True)] * 8 + [(True, False)] * 2 + [(False, False)] * 6 + [(False, True)] * 1
    interval = kappa_interval(pairs)
    kappa = cohen_kappa(pairs)
    assert interval is not None and kappa is not None
    assert interval == kappa_interval(pairs)
    assert interval[0] <= kappa <= interval[1]


def test_perfect_agreement_has_kappa_but_no_interval() -> None:
    """Without a disagreement every resample agrees too, so no collapsed interval is given."""
    pairs = [(True, True)] * 34 + [(False, False)] * 12
    assert cohen_kappa(pairs) == 1.0
    assert kappa_interval(pairs) is None


def test_single_verdicts_have_no_flip_rate() -> None:
    """With one verdict per item nothing can flip, so the flip rate is not reported as 0%."""
    grades = [_grade(True, (True,)), _grade(False, (False,))]
    row = agreement(grades, "tool_selection", "all")
    assert (row.flip_rate, row.kappa, row.kappa_ci) == (None, 1.0, None)
    report = summarize(
        grades,
        experiment="exp",
        judge={"provider": "openai"},
        prompt_versions={"tool_selection": "v1"},
        repeats=1,
    )
    (all_row,) = [line for line in format_report(report).splitlines() if " all " in line]
    assert all_row.split()[-3:] == ["1.00", "n/a", "0"]


def test_grade_uses_the_majority_verdict_and_notices_flips() -> None:
    """Two passes of three is a pass that flipped; incomplete grades are not complete."""
    grade = _grade(label=False, verdicts=(True, False, True))
    assert (grade.verdict, grade.flipped, grade.agrees, grade.complete) == (True, True, False, True)
    assert not _grade(label=True, verdicts=(True, True), errors=1).complete


def test_agreement_counts_majority_verdicts_and_leaves_out_errors() -> None:
    """Confusion counts come from majority verdicts; a grade with errors only counts as one."""
    grades = [
        _grade(True, (True, True, True), source="silver"),
        _grade(False, (False, False, True)),
        _grade(False, (True, True, True)),
        _grade(True, (True, True), errors=1),
        _grade(True, (False, False, False), metric="argument_quality"),
    ]
    row = agreement(grades, "tool_selection", "all")
    assert (row.items, row.tp, row.fn, row.tn, row.fp, row.errors) == (3, 1, 0, 1, 1, 1)
    assert row.tpr is not None and row.tpr.value == 1.0
    assert row.tnr is not None and row.tnr.value == 0.5
    assert row.flip_rate == pytest.approx(1 / 3)
    built = agreement(grades, "tool_selection", "built")
    assert (built.items, built.tpr) == (2, None)


def test_report_has_rows_only_where_grades_exist_and_lists_disagreements() -> None:
    """Per metric: a row per source with grades and one over both; misses built first."""
    grades = [
        _grade(True, (False, False, False), source="silver", item_id="P3-01#aaaa"),
        _grade(False, (True, True, True), item_id="P3-01+swap_order_id"),
        _grade(False, (False, False, False), metric="argument_quality"),
    ]
    report = summarize(
        grades,
        experiment="exp",
        judge={"provider": "openai", "model": "m", "temperature": None},
        prompt_versions={"tool_selection": "v1", "argument_quality": "v2"},
        repeats=3,
    )
    assert [(a.metric, a.source) for a in report.agreements] == [
        ("tool_selection", "silver"),
        ("tool_selection", "built"),
        ("tool_selection", "all"),
        ("argument_quality", "built"),
        ("argument_quality", "all"),
    ]
    assert [g.item_id for g in report.disagreements] == ["P3-01+swap_order_id", "P3-01#aaaa"]
    text = format_report(report)
    assert "tool_selection v1, argument_quality v2" in text
    assert "- tool_selection on P3-01+swap_order_id (built): label fail, verdicts pass" in text
    assert "reason 0" in text
    assert len(json.loads(report.to_json())["grades"]) == 3


def test_counts_show_items_per_source_and_judge_calls() -> None:
    """A dry run shows pass and fail labels per metric and source, and the calls it would make."""
    turn = Turn("q", "a", ())
    items = [
        LabelledTurn("P3-01#a", "P3-01", "silver", turn, {"tool_selection": True}),
        LabelledTurn(
            "P3-01+x", "P3-01", "built", turn, {"tool_selection": False, "argument_quality": True}
        ),
    ]
    text = format_counts(items, ["tool_selection", "argument_quality"], repeats=3)
    assert "tool_selection         silver     1     1     0" in text
    assert "tool_selection         built      1     0     1" in text
    assert "9 judge calls (3 per item and metric)" in text
