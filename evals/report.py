"""Experiment report: pass rate, pass^k and pass@k per scenario, flaky cases, latency.

The maths runs on a small internal row type (:class:`ScoredRun`), so it is testable without
LangSmith objects; :func:`scored_runs` adapts LangSmith's experiment rows to it.

LangSmith summary evaluators can't compute pass^k: they receive runs and examples but not
the per-run feedback. So the report computes it here and :func:`attach_to_experiment`
writes the overall numbers back to the experiment as experiment-level feedback.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypedDict
from uuid import UUID

from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResults
from langsmith.schemas import Example, Run

from evals.evaluators.base import EvaluatorKind


class ResultRow(TypedDict):
    """The parts of a LangSmith experiment row the report reads.

    Structurally the same as the rows ``client.evaluate`` returns, without importing
    LangSmith's private runner module.

    Attributes:
        run: The target run.
        example: The dataset example.
        evaluation_results: All evaluator results for the run.
    """

    run: Run
    example: Example
    evaluation_results: EvaluationResults


@dataclass(frozen=True, slots=True)
class ScoredRun:
    """One target run with its metric scores, independent of LangSmith types.

    Attributes:
        example_id: The example; repetitions of one case share it.
        case_id: Hand-written case id, e.g. ``P4-01``.
        scenario_id: Scenario id, e.g. ``P4``.
        scores: Metric key to pass/fail. Metrics that didn't apply or whose evaluator
            failed are absent.
        evaluator_errors: Number of ``evaluator_error`` results on this run.
        latency_s: Wall-clock duration of the run, if known.
        total_tokens: Tokens used by the run, if known.
    """

    example_id: str
    case_id: str
    scenario_id: str
    scores: Mapping[str, bool]
    evaluator_errors: int = 0
    latency_s: float | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """One metric over the runs of one scenario (or of all scenarios).

    Attributes:
        OVERALL: Pseudo scenario id of the summary row over every scenario.
        scenario_id: Scenario id, or ``OVERALL``.
        metric: Metric key.
        cases: Number of distinct cases.
        runs: Number of scored runs (cases times repetitions).
        pass_rate: Passing runs / runs.
        pass_hat_k: Cases where every repetition passed / cases (reliability).
        pass_at_k: Cases where at least one repetition passed / cases (capability).
        flaky_cases: Case ids that both passed and failed across repetitions.
    """

    scenario_id: str
    metric: str
    cases: int
    runs: int
    pass_rate: float
    pass_hat_k: float
    pass_at_k: float
    flaky_cases: tuple[str, ...]

    OVERALL: ClassVar[str] = "ALL"


@dataclass(frozen=True, slots=True)
class Report:
    """Summary of one experiment.

    Attributes:
        metrics: Per-scenario summaries, followed by one ``OVERALL`` row per metric.
        evaluator_errors: Total ``evaluator_error`` results.
        latency_p50_s: Median run latency, if known.
        latency_p95_s: 95th-percentile run latency, if known.
        mean_tokens: Mean tokens per run, if known.
        failed_gates: Gate metrics with at least one failing run.
    """

    metrics: tuple[MetricSummary, ...]
    evaluator_errors: int
    latency_p50_s: float | None
    latency_p95_s: float | None
    mean_tokens: float | None
    failed_gates: tuple[str, ...]

    @property
    def gate_failed(self) -> bool:
        """Return whether any gate metric failed.

        Returns:
            ``True`` if at least one gate had a failing run.
        """
        return bool(self.failed_gates)

    def overall(self, metric: str) -> MetricSummary | None:
        """Return the ``OVERALL`` summary of a metric.

        Args:
            metric: Metric key.

        Returns:
            The summary, or ``None`` if no run was scored on that metric.
        """
        return next(
            (
                m
                for m in self.metrics
                if m.metric == metric and m.scenario_id == MetricSummary.OVERALL
            ),
            None,
        )


def scored_runs(rows: Iterable[ResultRow]) -> list[ScoredRun]:
    """Convert LangSmith experiment rows to :class:`ScoredRun` objects.

    Args:
        rows: Rows of ``client.evaluate`` results (one per run, repetitions included).

    Returns:
        One scored run per row.
    """
    runs = []
    for row in rows:
        run, example = row["run"], row["example"]
        metadata = example.metadata or {}
        results = row["evaluation_results"].get("results", [])
        scores = {
            r.key: bool(r.score)
            for r in results
            if r.key != "evaluator_error" and r.score is not None
        }
        latency = (run.end_time - run.start_time).total_seconds() if run.end_time else None
        runs.append(
            ScoredRun(
                example_id=str(example.id),
                case_id=str(metadata.get("case_id", example.id)),
                scenario_id=str(metadata.get("scenario_id", "?")),
                scores=scores,
                evaluator_errors=sum(r.key == "evaluator_error" for r in results),
                latency_s=latency,
                total_tokens=run.total_tokens,
            )
        )
    return runs


def _percentile(values: Sequence[float], percent: float) -> float | None:
    """Return a nearest-rank percentile.

    Args:
        values: Sample values.
        percent: Percentile between 0 and 100.

    Returns:
        The percentile, or ``None`` for an empty sample.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(math.ceil(percent / 100 * len(ordered)), 1)
    return ordered[rank - 1]


def _summarize_metric(scenario_id: str, metric: str, runs: Sequence[ScoredRun]) -> MetricSummary:
    """Summarize one metric over a group of runs.

    Args:
        scenario_id: Scenario id of the group, or ``OVERALL``.
        metric: Metric key.
        runs: Runs scored on ``metric``.

    Returns:
        The summary.
    """
    by_case: dict[str, list[bool]] = defaultdict(list)
    case_ids: dict[str, str] = {}
    for run in runs:
        by_case[run.example_id].append(run.scores[metric])
        case_ids[run.example_id] = run.case_id
    outcomes = list(by_case.values())
    passed = sum(sum(o) for o in outcomes)
    total = sum(len(o) for o in outcomes)
    return MetricSummary(
        scenario_id=scenario_id,
        metric=metric,
        cases=len(outcomes),
        runs=total,
        pass_rate=passed / total,
        pass_hat_k=sum(all(o) for o in outcomes) / len(outcomes),
        pass_at_k=sum(any(o) for o in outcomes) / len(outcomes),
        flaky_cases=tuple(sorted(case_ids[e] for e, o in by_case.items() if any(o) and not all(o))),
    )


def summarize(runs: Sequence[ScoredRun], kinds: Mapping[str, EvaluatorKind]) -> Report:
    """Summarize an experiment per scenario and overall.

    Args:
        runs: Every run of the experiment, repetitions included.
        kinds: Metric key to ``gate`` or ``score``; a failing run on a gate fails the report.

    Returns:
        The report.
    """
    metrics = sorted({metric for run in runs for metric in run.scores})
    summaries: list[MetricSummary] = []
    for metric in metrics:
        scored = [run for run in runs if metric in run.scores]
        by_scenario: dict[str, list[ScoredRun]] = defaultdict(list)
        for run in scored:
            by_scenario[run.scenario_id].append(run)
        summaries.extend(
            _summarize_metric(scenario, metric, group)
            for scenario, group in sorted(by_scenario.items())
        )
        summaries.append(_summarize_metric(MetricSummary.OVERALL, metric, scored))

    latencies = [run.latency_s for run in runs if run.latency_s is not None]
    tokens = [run.total_tokens for run in runs if run.total_tokens is not None]
    failed_gates = tuple(
        s.metric
        for s in summaries
        if s.scenario_id == MetricSummary.OVERALL
        and kinds.get(s.metric) == "gate"
        and s.pass_rate < 1
    )
    return Report(
        metrics=tuple(summaries),
        evaluator_errors=sum(run.evaluator_errors for run in runs),
        latency_p50_s=_percentile(latencies, 50),
        latency_p95_s=_percentile(latencies, 95),
        mean_tokens=sum(tokens) / len(tokens) if tokens else None,
        failed_gates=failed_gates,
    )


def format_report(report: Report) -> str:
    """Render the report as a plain-text table.

    Args:
        report: The report.

    Returns:
        A table with one row per scenario and metric, followed by run statistics.
    """
    header = f"{'scenario':<9} {'metric':<22} {'cases':>5} {'runs':>5} {'pass':>6} "
    header += f"{'pass^k':>7} {'pass@k':>7}  flaky"
    lines = [header, "-" * len(header)]
    for m in report.metrics:
        if m.scenario_id == MetricSummary.OVERALL:
            lines.append("-" * len(header))
        lines.append(
            f"{m.scenario_id:<9} {m.metric:<22} {m.cases:>5} {m.runs:>5} {m.pass_rate:>6.0%} "
            f"{m.pass_hat_k:>7.0%} {m.pass_at_k:>7.0%}  {', '.join(m.flaky_cases)}"
        )

    def _seconds(value: float | None) -> str:
        """Format an optional duration.

        Args:
            value: Seconds, or ``None``.

        Returns:
            The duration with one decimal, or ``n/a``.
        """
        return "n/a" if value is None else f"{value:.1f}s"

    tokens = "n/a" if report.mean_tokens is None else f"{report.mean_tokens:.0f}"
    lines += [
        "",
        f"latency p50 {_seconds(report.latency_p50_s)}, p95 {_seconds(report.latency_p95_s)}; "
        f"mean tokens {tokens}; evaluator errors {report.evaluator_errors}",
    ]
    if report.gate_failed:
        lines.append(f"FAILED GATES: {', '.join(report.failed_gates)}")
    return "\n".join(lines)


def attach_to_experiment(client: Client, experiment_id: UUID, report: Report) -> None:
    """Write the overall numbers to the experiment as experiment-level feedback.

    Keys are ``<metric>:pass_rate``, ``<metric>:pass^k`` and ``<metric>:pass@k``.

    Args:
        client: LangSmith client.
        experiment_id: The experiment (a LangSmith project) to attach to.
        report: The report.
    """
    for summary in report.metrics:
        if summary.scenario_id != MetricSummary.OVERALL:
            continue
        for name, score in (
            ("pass_rate", summary.pass_rate),
            ("pass^k", summary.pass_hat_k),
            ("pass@k", summary.pass_at_k),
        ):
            client.create_feedback(
                None, f"{summary.metric}:{name}", score=score, project_id=experiment_id
            )


__all__ = [
    "MetricSummary",
    "Report",
    "ResultRow",
    "ScoredRun",
    "attach_to_experiment",
    "format_report",
    "scored_runs",
    "summarize",
]
