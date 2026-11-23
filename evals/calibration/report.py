"""Calibration report: agreement of each judge with its labels, and every disagreement."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from evals.calibration.grading import Grade
from evals.calibration.labels import LabelledTurn
from evals.calibration.stats import cohen_kappa, kappa_interval, wilson

ReportSource = Literal["silver", "built", "all"]
"""Which grades a row covers: one label source, or both."""


@dataclass(frozen=True, slots=True)
class Rate:
    """A proportion with its 95% Wilson interval.

    Attributes:
        value: The proportion.
        low: Lower bound.
        high: Upper bound.
        n: Number of trials it was computed from.
    """

    value: float
    low: float
    high: float
    n: int

    @classmethod
    def of(cls, successes: int, n: int) -> "Rate | None":
        """Build a rate, or nothing if there were no trials.

        Args:
            successes: Number of successes.
            n: Number of trials.

        Returns:
            The rate, or ``None`` if ``n`` is 0.
        """
        if n == 0:
            return None
        low, high = wilson(successes, n)
        return cls(value=successes / n, low=low, high=high, n=n)


@dataclass(frozen=True, slots=True)
class Agreement:
    """How one judge metric agrees with its labels on one label source.

    "Pass" is the positive class: TPR is how often the judge passes a turn labelled pass,
    TNR how often it fails a turn labelled fail.

    Attributes:
        metric: Judge metric.
        source: Label source of the grades, or ``all``.
        items: Complete grades (every repeat gave a verdict).
        tp: Labelled pass, judged pass.
        fn: Labelled pass, judged fail.
        tn: Labelled fail, judged fail.
        fp: Labelled fail, judged pass.
        tpr: True-positive rate, or ``None`` without pass labels.
        tnr: True-negative rate, or ``None`` without fail labels.
        kappa: Cohen's kappa, or ``None`` if undefined.
        kappa_ci: Bootstrap 95% interval of kappa, or ``None``.
        flip_rate: Share of items whose verdicts differed across repeats.
        errors: Grades left out because a judge call failed.
    """

    metric: str
    source: ReportSource
    items: int
    tp: int
    fn: int
    tn: int
    fp: int
    tpr: Rate | None
    tnr: Rate | None
    kappa: float | None
    kappa_ci: tuple[float, float] | None
    flip_rate: float | None
    errors: int


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """The result of one calibration run.

    Attributes:
        experiment: The stored experiment whose turns were graded.
        judge: Judge provider, model and temperature.
        prompt_versions: Prompt version of each graded judge metric.
        repeats: Verdicts per item and metric.
        agreements: One row per metric and label source with grades.
        grades: Every grade, for ``--out`` and for review.
    """

    experiment: str
    judge: Mapping[str, Any]
    prompt_versions: Mapping[str, str]
    repeats: int
    agreements: tuple[Agreement, ...]
    grades: tuple[Grade, ...] = field(repr=False)

    @property
    def disagreements(self) -> list[Grade]:
        """Return the complete grades whose majority verdict differs from the label.

        Returns:
            Those grades, built failures first (a missed built failure is a rubric gap; a
            silver disagreement may be the label's fault).
        """
        missed = [g for g in self.grades if g.complete and not g.agrees]
        return sorted(missed, key=lambda g: (g.source != "built", g.metric, g.item_id))

    def to_json(self) -> str:
        """Serialise the whole report, every grade included.

        Returns:
            Indented JSON.
        """
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def agreement(grades: Sequence[Grade], metric: str, source: ReportSource) -> Agreement:
    """Compute one metric's agreement with its labels.

    Args:
        grades: Grades of any metrics and sources.
        metric: The judge metric.
        source: The label source to keep, or ``all``.

    Returns:
        The agreement, from the majority verdict of each complete grade.
    """
    selected = [g for g in grades if g.metric == metric and source in ("all", g.source)]
    complete = [g for g in selected if g.complete]
    pairs = [(g.label, g.verdict) for g in complete]
    tp = sum(label and verdict for label, verdict in pairs)
    fn = sum(label and not verdict for label, verdict in pairs)
    tn = sum(not label and not verdict for label, verdict in pairs)
    fp = sum(not label and verdict for label, verdict in pairs)
    return Agreement(
        metric=metric,
        source=source,
        items=len(complete),
        tp=tp,
        fn=fn,
        tn=tn,
        fp=fp,
        tpr=Rate.of(tp, tp + fn),
        tnr=Rate.of(tn, tn + fp),
        kappa=cohen_kappa(pairs),
        kappa_ci=kappa_interval(pairs),
        flip_rate=sum(g.flipped for g in complete) / len(complete) if complete else None,
        errors=len(selected) - len(complete),
    )


def summarize(
    grades: Sequence[Grade],
    *,
    experiment: str,
    judge: Mapping[str, Any],
    prompt_versions: Mapping[str, str],
    repeats: int,
) -> CalibrationReport:
    """Build the report: per metric, one row per label source and one over both.

    Args:
        grades: Every grade.
        experiment: The stored experiment graded.
        judge: Judge provider, model and temperature.
        prompt_versions: Prompt version of each graded metric, in report order.
        repeats: Verdicts per item and metric.

    Returns:
        The report. A source without grades for a metric gets no row.
    """
    sources: tuple[ReportSource, ...] = ("silver", "built", "all")
    rows = [
        agreement(grades, metric, source)
        for metric in prompt_versions
        for source in sources
        if any(g.metric == metric and source in ("all", g.source) for g in grades)
    ]
    return CalibrationReport(
        experiment=experiment,
        judge=judge,
        prompt_versions=prompt_versions,
        repeats=repeats,
        agreements=tuple(rows),
        grades=tuple(grades),
    )


def format_report(report: CalibrationReport, reasoning_chars: int = 400) -> str:
    """Render the report as plain text.

    Args:
        report: The report.
        reasoning_chars: How much of the judge's reasoning to show per disagreement.

    Returns:
        A header with the judge and prompts, the agreement table, then each disagreement
        with the judge's first reasoning.
    """
    judge = ", ".join(f"{k}={v}" for k, v in report.judge.items())
    prompts = ", ".join(f"{metric} {version}" for metric, version in report.prompt_versions.items())
    header = f"{'metric':<22} {'source':<6} {'items':>5} {'TP':>4} {'FN':>4} {'TN':>4} {'FP':>4} "
    header += f"{'TPR [95% CI]':>18} {'TNR [95% CI]':>18} {'kappa [95% CI]':>21} "
    header += f"{'flips':>6} {'errors':>6}"
    lines = [
        f"Calibration of {report.experiment}, {report.repeats} verdicts per item",
        f"judge: {judge}",
        f"prompts: {prompts}",
        "",
        header,
        "-" * len(header),
    ]
    for a in report.agreements:
        flips = "n/a" if a.flip_rate is None else f"{a.flip_rate:.0%}"
        lines.append(
            f"{a.metric:<22} {a.source:<6} {a.items:>5} {a.tp:>4} {a.fn:>4} {a.tn:>4} {a.fp:>4} "
            f"{_rate(a.tpr):>18} {_rate(a.tnr):>18} {_kappa(a.kappa, a.kappa_ci):>21} "
            f"{flips:>6} {a.errors:>6}"
        )

    disagreements = report.disagreements
    lines += ["", f"Disagreements ({len(disagreements)}):"]
    for g in disagreements:
        verdicts = " ".join("pass" if v else "fail" for v in g.verdicts)
        reasoning = " ".join(g.reasonings[0].split())[:reasoning_chars]
        lines += [
            f"- {g.metric} on {g.item_id} ({g.source}): label "
            f"{'pass' if g.label else 'fail'}, verdicts {verdicts}",
            f"  {reasoning}",
        ]
    return "\n".join(lines)


def format_counts(items: Sequence[LabelledTurn], metrics: Sequence[str], repeats: int) -> str:
    """Render what a calibration run would grade, without grading it.

    Args:
        items: The labelled items.
        metrics: The judge metrics to grade.
        repeats: Verdicts per item and metric.

    Returns:
        A table of items per metric and source (pass and fail labels), and the number of
        judge calls.
    """
    header = f"{'metric':<22} {'source':<6} {'items':>5} {'pass':>5} {'fail':>5}"
    lines = [header, "-" * len(header)]
    calls = 0
    for metric in metrics:
        for source in ("silver", "built"):
            labels = [i.labels[metric] for i in items if i.source == source and metric in i.labels]
            if labels:
                calls += len(labels) * repeats
                lines.append(
                    f"{metric:<22} {source:<6} {len(labels):>5} {sum(labels):>5} "
                    f"{len(labels) - sum(labels):>5}"
                )
    lines += ["", f"{calls} judge calls ({repeats} per item and metric)"]
    return "\n".join(lines)


def _rate(rate: Rate | None) -> str:
    """Format an optional rate with its interval.

    Args:
        rate: The rate, or ``None``.

    Returns:
        ``0.95 [0.80, 0.99]``, or ``n/a``.
    """
    return "n/a" if rate is None else f"{rate.value:.2f} [{rate.low:.2f}, {rate.high:.2f}]"


def _kappa(kappa: float | None, interval: tuple[float, float] | None) -> str:
    """Format an optional kappa with its interval.

    Args:
        kappa: Kappa, or ``None`` if undefined.
        interval: Its bootstrap interval, or ``None``.

    Returns:
        ``0.81 [0.62, 0.95]``, ``0.81`` without an interval, or ``n/a``.
    """
    if kappa is None:
        return "n/a"
    if interval is None:
        return f"{kappa:.2f}"
    return f"{kappa:.2f} [{interval[0]:.2f}, {interval[1]:.2f}]"


__all__ = [
    "Agreement",
    "CalibrationReport",
    "Rate",
    "ReportSource",
    "agreement",
    "format_counts",
    "format_report",
    "summarize",
]
