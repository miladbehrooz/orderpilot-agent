"""Calibrate the LLM judges: ``uv run python -m evals.calibrate --experiment <name>``.

Grades a stored experiment's turns again, without running the agent, and measures how often
each judge agrees with labels that need no human work: the code metrics' feedback on the
stored runs (silver labels) and failing turns built from passing ones (built failures).
Nothing is written to LangSmith. See ``docs/evals.md`` (Judge calibration).
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evals.calibration.grading import grade_items
from evals.calibration.labels import LabelledTurn, base_turns, silver_items
from evals.calibration.mutations import build_failures
from evals.calibration.report import CalibrationReport, format_counts, format_report, summarize
from evals.calibration.stored import StoredExperiment, load_experiment
from evals.cases import check_reference_now
from evals.client import make_client
from evals.errors import EvalError, EvalSetupError
from evals.evaluators import EvaluatorFactory, LLMJudgeEvaluator
from evals.settings import eval_settings
from orderpilot.errors import OrderPilotError
from orderpilot.repository import OrderRepository, SqliteOrderRepository
from orderpilot.settings import settings


@dataclass(frozen=True, slots=True)
class CalibrateArgs:
    """What to calibrate, from the command line.

    Attributes:
        experiment: Name of the stored experiment whose turns are graded.
        metrics: Judge metrics to calibrate; empty means every LLM-judge metric.
        repeats: Verdicts per item and metric; odd, so the majority verdict has no tie.
        dry_run: Only count the items and judge calls; build no judge.
        out: File to write the full report to as JSON, if given.
    """

    experiment: str
    metrics: tuple[str, ...] = ()
    repeats: int = 3
    dry_run: bool = False
    out: Path | None = None


def parse_args(argv: Sequence[str] | None) -> CalibrateArgs:
    """Parse the command line.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` if ``None``.

    Returns:
        The parsed arguments.
    """

    def _odd(text: str) -> int:
        """Parse a positive odd number.

        Args:
            text: The argument.

        Returns:
            The number.

        Raises:
            argparse.ArgumentTypeError: If it is not a positive odd integer.
        """
        try:
            value = int(text)
        except ValueError:
            value = 0
        if value < 1 or value % 2 == 0:
            raise argparse.ArgumentTypeError(f"must be a positive odd number, got {text!r}")
        return value

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--experiment", required=True, help="stored experiment to grade again")
    parser.add_argument(
        "--metrics", default="", help="comma-separated judge metrics (default: every judge)"
    )
    parser.add_argument(
        "--repeats", type=_odd, default=3, help="verdicts per item and metric (odd, default 3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="count items and judge calls; call no judge"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)
    return CalibrateArgs(
        experiment=args.experiment,
        metrics=tuple(name for name in args.metrics.split(",") if name),
        repeats=args.repeats,
        dry_run=args.dry_run,
        out=args.out,
    )


def judge_metrics(requested: Sequence[str]) -> list[str]:
    """Return the judge metrics to calibrate.

    Args:
        requested: Metric names given with ``--metrics``; empty for every LLM judge.

    Returns:
        ``requested``, or every registered LLM-judge metric in registration order.

    Raises:
        EvalSetupError: If a metric is unknown or not an LLM judge.
    """
    judges = list(EvaluatorFactory.prompt_versions(EvaluatorFactory.metric_names()))
    if not requested:
        return judges
    others = [name for name in requested if name not in EvaluatorFactory.prompt_versions([name])]
    if others:
        raise EvalSetupError(
            f"Not LLM-judge metrics: {', '.join(others)}. Valid: {', '.join(judges)}"
        )
    return list(requested)


def labelled_items(experiment: StoredExperiment, repo: OrderRepository) -> list[LabelledTurn]:
    """Label the experiment's turns and build failures from the passing ones.

    Args:
        experiment: The stored experiment.
        repo: Repository on the local database; the judges read its TODAY and tools.

    Returns:
        The silver items, then the built failures.

    Raises:
        EvalSetupError: If the experiment ran on another database build than the local one,
            or it has no labelled turns.
    """
    check_reference_now(
        experiment.reference_now, repo.reference_now(), source=f"Experiment {experiment.name}"
    )
    silver = silver_items(experiment.turns)
    if not silver:
        raise EvalSetupError(f"Experiment {experiment.name} has no runs with code feedback")
    return silver + build_failures(base_turns(silver))


def calibrate(
    items: Sequence[LabelledTurn],
    metrics: Sequence[str],
    repeats: int,
    *,
    experiment: str,
    repo: OrderRepository,
) -> CalibrationReport:
    """Grade the labelled items with the judges and summarise the agreement.

    Args:
        items: The labelled items.
        metrics: Judge metrics to calibrate.
        repeats: Verdicts per item and metric.
        experiment: Name of the stored experiment, for the report.
        repo: Repository on the database the turns ran on.

    Returns:
        The calibration report.

    Raises:
        LLMConfigError: If the judge is not configured.
    """
    judges = {
        evaluator.key: evaluator
        for evaluator in EvaluatorFactory.build(metrics, repo)
        if isinstance(evaluator, LLMJudgeEvaluator)
    }
    grades = grade_items(items, judges, repeats)
    judge = {
        "provider": eval_settings.JUDGE_PROVIDER,
        "model": eval_settings.JUDGE_MODEL,
        "temperature": eval_settings.JUDGE_TEMPERATURE,
    }
    return summarize(
        grades,
        experiment=experiment,
        judge=judge,
        prompt_versions=EvaluatorFactory.prompt_versions(metrics),
        repeats=repeats,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Calibrate the judges from the command line and print the report.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` if ``None``.

    Returns:
        Process exit code: 0 on success, 2 on a setup or configuration error.
    """
    args = parse_args(argv)
    try:
        metrics = judge_metrics(args.metrics)
        repo = SqliteOrderRepository(settings.DB_PATH)
        experiment = load_experiment(make_client(), args.experiment)
        items = labelled_items(experiment, repo)
        if args.dry_run:
            print(format_counts(items, metrics, args.repeats))
            return 0
        report = calibrate(items, metrics, args.repeats, experiment=experiment.name, repo=repo)
        if args.out is not None:
            args.out.write_text(report.to_json(), encoding="utf-8")
    except (EvalError, OrderPilotError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CalibrateArgs", "calibrate", "judge_metrics", "labelled_items", "main", "parse_args"]
