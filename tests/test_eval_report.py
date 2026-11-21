"""Experiment report: pass rate, pass^k, pass@k, flaky cases, gates, measures, feedback."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run
from pytest_mock import MockerFixture

from evals.report import (
    MeasureSummary,
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
    (scored,) = scored_runs([row], kinds={"answer_correctness": "score"})
    assert scored.example_id == str(example.id)
    assert (scored.case_id, scored.scenario_id) == ("P4-01", "P4")
    assert scored.scores == {"answer_correctness": False}
    assert scored.evaluator_errors == 1
    assert scored.latency_s == 2.5


def _measured(case_id: str, scenario_id: str, steps: list[float]) -> list[ScoredRun]:
    """Return the repetitions of one case with a ``step_count`` measure and a passing score.

    Args:
        case_id: Case id; also used as the example id.
        scenario_id: Scenario id.
        steps: ``step_count`` per repetition.

    Returns:
        One scored run per value.
    """
    return [
        ScoredRun(case_id, case_id, scenario_id, {"tool_correctness": True}, {"step_count": n})
        for n in steps
    ]


@pytest.fixture
def measured_runs() -> list[ScoredRun]:
    """Return runs of two scenarios with step counts: P3 takes 1 step, P8 takes 2 to 4.

    Returns:
        The runs.
    """
    return [*_measured("P3-01", "P3", [1, 1]), *_measured("P8-01", "P8", [2, 2, 4])]


def test_scored_runs_routes_measures_by_kind() -> None:
    """A measure keeps its number; it is not read as pass/fail (3 steps is not "True")."""
    example = Example.model_validate(
        {"id": uuid4(), "metadata": {"case_id": "P8-01", "scenario_id": "P8"}}
    )
    run_id = uuid4()
    run = Run.model_validate(
        {
            "id": run_id,
            "trace_id": run_id,
            "name": "invoke",
            "run_type": "chain",
            "start_time": datetime(2026, 11, 16, tzinfo=UTC),
        }
    )
    row: ResultRow = {
        "run": run,
        "example": example,
        "evaluation_results": {
            "results": [
                EvaluationResult(key="tool_correctness", score=True),
                EvaluationResult(key="step_count", score=3),
            ]
        },
    }
    (scored,) = scored_runs([row], kinds={"tool_correctness": "score", "step_count": "measure"})
    assert scored.scores == {"tool_correctness": True}
    assert scored.measures == {"step_count": 3.0}


def test_measures_summarized_per_scenario_and_overall(measured_runs: list[ScoredRun]) -> None:
    """Each measure gets mean, nearest-rank p50/p95 and max per scenario and overall.

    Args:
        measured_runs: Runs with step counts.
    """
    report = summarize(measured_runs, kinds={"step_count": "measure"})
    by_scenario = {m.scenario_id: m for m in report.measures}
    assert by_scenario["P8"] == MeasureSummary("P8", "step_count", 3, 8 / 3, 2, 4, 4)
    assert by_scenario[MetricSummary.OVERALL] == MeasureSummary(
        MetricSummary.OVERALL, "step_count", 5, 2.0, 2, 4, 4
    )


def test_measures_never_enter_pass_rates_or_gates(measured_runs: list[ScoredRun]) -> None:
    """Measures are reported apart: no pass rate and no gate, whatever their values.

    Args:
        measured_runs: Runs with step counts.
    """
    report = summarize(measured_runs, kinds={"step_count": "measure"})
    assert {m.metric for m in report.metrics} == {"tool_correctness"}
    assert not report.gate_failed


def test_format_report_shows_measure_table(measured_runs: list[ScoredRun]) -> None:
    """The text report has a measures table with the overall row of each measure.

    Args:
        measured_runs: Runs with step counts.
    """
    text = format_report(summarize(measured_runs, kinds={"step_count": "measure"}))
    assert "measure" in text
    overall = next(line for line in text.splitlines() if line.startswith("ALL") and "step" in line)
    assert overall.split() == ["ALL", "step_count", "5", "2.0", "2", "4", "4"]


def test_attach_to_experiment_writes_measure_feedback(
    mocker: MockerFixture, measured_runs: list[ScoredRun]
) -> None:
    """Each measure's overall mean and p95 are attached to the experiment.

    Args:
        mocker: Pytest-mock fixture.
        measured_runs: Runs with step counts.
    """
    client = mocker.create_autospec(Client, instance=True)
    report = summarize(measured_runs, kinds={"step_count": "measure"})
    attach_to_experiment(client, uuid4(), report)
    scores = {c.args[1]: c.kwargs["score"] for c in client.create_feedback.call_args_list}
    assert scores["step_count:mean"] == 2.0
    assert scores["step_count:p95"] == 4


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
