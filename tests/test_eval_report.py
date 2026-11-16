"""Experiment report: pass rate, pass^k, pass@k, flaky cases, gates and experiment feedback."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run
from pytest_mock import MockerFixture

from evals.report import (
    MetricSummary,
    ResultRow,
    ScoredRun,
    attach_to_experiment,
    format_report,
    scored_runs,
    summarize,
)


def _runs(case_id: str, scenario_id: str, outcomes: list[bool], metric: str) -> list[ScoredRun]:
    """Return the repetitions of one case.

    Args:
        case_id: Case id; also used as the example id.
        scenario_id: Scenario id.
        outcomes: Pass/fail per repetition.
        metric: Metric key.

    Returns:
        One scored run per outcome.
    """
    return [ScoredRun(case_id, case_id, scenario_id, {metric: ok}) for ok in outcomes]


@pytest.fixture
def runs() -> list[ScoredRun]:
    """Return k=3 runs of three cases: always passing, flaky, and always failing.

    Returns:
        The runs.
    """
    metric = "answer_correctness"
    return [
        *_runs("P3-01", "P3", [True, True, True], metric),
        *_runs("P4-01", "P4", [True, False, True], metric),
        *_runs("P4-02", "P4", [False, False, False], metric),
    ]


def test_pass_rates_per_scenario_and_overall(runs: list[ScoredRun]) -> None:
    """pass^k needs every repetition, pass@k one; the overall row covers all scenarios.

    Args:
        runs: Runs of three cases.
    """
    report = summarize(runs, kinds={"answer_correctness": "score"})
    by_scenario = {m.scenario_id: m for m in report.metrics}

    p4 = by_scenario["P4"]
    assert (p4.cases, p4.runs) == (2, 6)
    assert p4.pass_rate == pytest.approx(2 / 6)
    assert p4.pass_hat_k == 0.0
    assert p4.pass_at_k == 0.5

    overall = report.overall("answer_correctness")
    assert overall is not None
    assert overall.scenario_id == MetricSummary.OVERALL
    assert overall.pass_rate == pytest.approx(5 / 9)
    assert overall.pass_hat_k == pytest.approx(1 / 3)
    assert overall.pass_at_k == pytest.approx(2 / 3)


def test_flaky_case_listed(runs: list[ScoredRun]) -> None:
    """A case that both passed and failed is flaky; stable passes and failures are not.

    Args:
        runs: Runs of three cases.
    """
    overall = summarize(runs, kinds={}).overall("answer_correctness")
    assert overall is not None
    assert overall.flaky_cases == ("P4-01",)


def test_unscored_runs_not_counted_as_failures() -> None:
    """Runs where the metric didn't apply or the judge failed are left out of the denominator."""
    runs = [
        ScoredRun("a", "P3-01", "P3", {"answer_correctness": True}),
        ScoredRun("a", "P3-01", "P3", {}, evaluator_errors=1),
        ScoredRun("b", "AU1-01", "AU1", {}),
    ]
    report = summarize(runs, kinds={})
    overall = report.overall("answer_correctness")
    assert overall is not None
    assert (overall.runs, overall.pass_rate) == (1, 1.0)
    assert report.evaluator_errors == 1


def test_gate_failure_fails_report() -> None:
    """One failing run on a gate fails the report, whatever the scores say."""
    runs = [
        *_runs("S1-01", "S1", [True, True], "no_data_leak"),
        *_runs("S1-02", "S1", [True, False], "no_data_leak"),
        *_runs("P3-01", "P3", [False], "answer_correctness"),
    ]
    report = summarize(runs, kinds={"no_data_leak": "gate", "answer_correctness": "score"})
    assert report.failed_gates == ("no_data_leak",)
    assert report.gate_failed
    assert "FAILED GATES: no_data_leak" in format_report(report)


def test_latency_percentiles_use_nearest_rank() -> None:
    """Latency percentiles use nearest rank over the runs that report a latency."""
    runs = [ScoredRun(str(i), "P3-01", "P3", {"m": True}, latency_s=float(i)) for i in range(1, 11)]
    report = summarize(runs, kinds={})
    assert (report.latency_p50_s, report.latency_p95_s) == (5.0, 10.0)


def test_scored_runs_reads_langsmith_rows() -> None:
    """LangSmith rows become scored runs: scores, evaluator errors, ids and latency."""
    example = Example.model_validate(
        {"id": uuid4(), "metadata": {"case_id": "P4-01", "scenario_id": "P4"}}
    )
    start = datetime(2026, 11, 16, tzinfo=UTC)
    run_id = uuid4()
    run = Run.model_validate(
        {
            "id": run_id,
            "trace_id": run_id,
            "name": "invoke",
            "run_type": "chain",
            "start_time": start,
            "end_time": start + timedelta(seconds=2.5),
        }
    )
    row: ResultRow = {
        "run": run,
        "example": example,
        "evaluation_results": {
            "results": [
                EvaluationResult(key="answer_correctness", score=False),
                EvaluationResult(key="evaluator_error", score=1, metadata={"metric": "x"}),
            ]
        },
    }
    (scored,) = scored_runs([row])
    assert scored.example_id == str(example.id)
    assert (scored.case_id, scored.scenario_id) == ("P4-01", "P4")
    assert scored.scores == {"answer_correctness": False}
    assert scored.evaluator_errors == 1
    assert scored.latency_s == 2.5


def test_attach_to_experiment_writes_overall_feedback(
    mocker: MockerFixture, runs: list[ScoredRun]
) -> None:
    """The overall numbers of each metric are attached to the experiment, not to runs.

    Args:
        mocker: Pytest-mock fixture.
        runs: Runs of three cases.
    """
    client = mocker.create_autospec(Client, instance=True)
    experiment_id = uuid4()
    attach_to_experiment(client, experiment_id, summarize(runs, kinds={}))
    keys = [c.args[1] for c in client.create_feedback.call_args_list]
    assert keys == [
        "answer_correctness:pass_rate",
        "answer_correctness:pass^k",
        "answer_correctness:pass@k",
    ]
    for call in client.create_feedback.call_args_list:
        assert call.args[0] is None
        assert call.kwargs["project_id"] == experiment_id
