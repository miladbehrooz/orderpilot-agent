"""Run an experiment: ``uv run python -m evals.run --dataset m1 --split smoke``.

Runs the real agent on the selected LangSmith examples (``EVAL_REPEATS`` times each), grades
every run on the selected metrics, prints the report and attaches its overall numbers
to the experiment. The dataset must have been written for the local database build.
"""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, get_args
from uuid import UUID

from langsmith import Client
from langsmith.schemas import Example

from evals.cases import Split, check_reference_now, langsmith_dataset_name
from evals.client import make_client
from evals.errors import EvalError, EvalSetupError
from evals.evaluators import EvaluatorFactory
from evals.report import Report, attach_to_experiment, format_report, scored_runs, summarize
from evals.settings import eval_settings
from evals.target import AgentTarget
from orderpilot.errors import OrderPilotError
from orderpilot.repository import OrderRepository, SqliteOrderRepository
from orderpilot.settings import PROJECT_ROOT, settings


@dataclass(frozen=True, slots=True)
class RunArgs:
    """What to run, from the command line.

    Attributes:
        dataset: Repo dataset name, e.g. ``"m1"``.
        split: Dataset split to run.
        families: Scenario families to keep (e.g. ``("P",)``); empty keeps all.
        metrics: Metric names; empty means every metric the selected cases list.
        description: Free-text description of the experiment.
    """

    dataset: str
    split: Split
    families: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    description: str | None = None


def parse_args(argv: Sequence[str] | None) -> RunArgs:
    """Parse the command line.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` if ``None``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--dataset", default="m1", help="repo dataset name (default: m1)")
    parser.add_argument("--split", default="smoke", choices=get_args(Split))
    parser.add_argument(
        "--family", action="append", default=[], help="scenario family to run, e.g. P; repeatable"
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="comma-separated metric names (default: the metrics the selected cases list)",
    )
    parser.add_argument("--description", default=None, help="experiment description")
    args = parser.parse_args(argv)
    return RunArgs(
        dataset=args.dataset,
        split=args.split,
        families=tuple(args.family),
        metrics=tuple(name for name in args.metrics.split(",") if name),
        description=args.description,
    )


def select_examples(
    client: Client, dataset_id: UUID, split: Split, families: Sequence[str]
) -> list[Example]:
    """Fetch the examples of one split, optionally only some scenario families.

    Args:
        client: LangSmith client.
        dataset_id: The LangSmith dataset.
        split: Split to fetch.
        families: Scenario families to keep; empty keeps all.

    Returns:
        The examples.

    Raises:
        EvalSetupError: If nothing matches.
    """
    if families:
        examples = [
            example
            for family in families
            for example in client.list_examples(
                dataset_id=dataset_id, splits=[split], metadata={"scenario_family": family}
            )
        ]
    else:
        examples = list(client.list_examples(dataset_id=dataset_id, splits=[split]))
    if not examples:
        raise EvalSetupError(f"No examples in split {split!r} for families {list(families)}")
    return examples


def metrics_for(examples: Sequence[Example], requested: Sequence[str]) -> list[str]:
    """Return the names of the metrics to run.

    Args:
        examples: The selected examples.
        requested: Metric names given with ``--metrics``; empty to use the examples' metadata.

    Returns:
        ``requested``, or every metric the examples list, in first-seen order.
    """
    if requested:
        return list(requested)
    names: dict[str, None] = {}
    for example in examples:
        for name in (example.metadata or {}).get("metrics", []):
            names[name] = None
    return list(names)


def git_revision() -> dict[str, Any]:
    """Return the git commit of the code under test.

    Returns:
        ``git_sha`` (short, or ``"unknown"`` outside git) and ``git_dirty`` (uncommitted
        changes present).
    """

    def _git(*args: str) -> str | None:
        """Run a git command in the project root.

        Args:
            *args: Git arguments.

        Returns:
            Its stripped output, or ``None`` if it failed.
        """
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    sha = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_sha": sha or "unknown", "git_dirty": bool(status)}


def experiment_metadata(
    args: RunArgs, metrics: Sequence[str], repo: OrderRepository
) -> dict[str, Any]:
    """Return the metadata that makes experiments comparable.

    Args:
        args: The run arguments.
        metrics: Names of the metrics being run.
        repo: Repository on the database under test.

    Returns:
        Agent and judge models, judge prompt versions, git revision, DB build, k and split.
    """
    prompt_versions = EvaluatorFactory.prompt_versions(metrics)
    judge = (
        {"judge_provider": eval_settings.JUDGE_PROVIDER, "judge_model": eval_settings.JUDGE_MODEL}
        if prompt_versions
        else {}
    )
    return {
        "agent_provider": settings.LLM_PROVIDER,
        "agent_model": settings.LLM_MODEL,
        **judge,
        "judge_prompts": prompt_versions,
        **git_revision(),
        "reference_now": repo.reference_now().isoformat(),
        "repeats": eval_settings.REPEATS,
        "split": args.split,
        "families": list(args.families),
    }


def run_experiment(args: RunArgs, client: Client, repo: OrderRepository) -> Report:
    """Run one experiment and attach its summary to it.

    Args:
        args: What to run.
        client: LangSmith client.
        repo: Repository on the database under test.

    Returns:
        The experiment report.

    Raises:
        EvalSetupError: If the dataset doesn't match the database, nothing is selected, or
            a metric is unknown.
        LLMConfigError: If the agent or the judge is not configured.
    """
    remote = client.read_dataset(dataset_name=langsmith_dataset_name(args.dataset))
    check_reference_now(
        (remote.metadata or {}).get("reference_now"),
        repo.reference_now(),
        source=f"LangSmith dataset {remote.name}",
    )
    examples = select_examples(client, remote.id, args.split, args.families)
    metrics = metrics_for(examples, args.metrics)
    evaluators = EvaluatorFactory.build(metrics, repo.reference_now())
    results = client.evaluate(
        AgentTarget(repo).invoke,
        data=examples,
        evaluators=evaluators,
        num_repetitions=eval_settings.REPEATS,
        max_concurrency=eval_settings.MAX_CONCURRENCY,
        experiment_prefix=f"{settings.LLM_MODEL or 'agent'}-{args.split}",
        description=args.description,
        metadata=experiment_metadata(args, metrics, repo),
        error_handling="log",
    )
    report = summarize(scored_runs(results), kinds={e.key: e.kind for e in evaluators})
    attach_to_experiment(client, results.experiment_id, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run an experiment from the command line and print its report.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` if ``None``.

    Returns:
        Process exit code: 0 on success, 1 if a gate metric failed, 2 on a setup or
        configuration error.
    """
    args = parse_args(argv)
    try:
        report = run_experiment(args, make_client(), SqliteOrderRepository(settings.DB_PATH))
    except (EvalError, OrderPilotError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(format_report(report))
    return 1 if report.gate_failed else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "RunArgs",
    "experiment_metadata",
    "git_revision",
    "main",
    "metrics_for",
    "parse_args",
    "run_experiment",
    "select_examples",
]
