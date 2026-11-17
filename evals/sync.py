"""Copy a repo dataset to LangSmith: ``uv run python -m evals.sync --dataset m1 [--prune]``.

The YAML files are the source of truth. Examples are matched by their stable id (a UUID5 of
``case_id``), so re-syncing updates examples in place; LangSmith keeps the version history.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from langsmith import Client
from langsmith.schemas import ExampleUpdate

from evals.cases import EvalCase, EvalDataset, check_reference_now, dataset_dir, load_dataset
from evals.client import make_client
from evals.errors import EvalError, EvalSetupError
from evals.evaluators import EvaluatorFactory
from orderpilot.errors import OrderPilotError
from orderpilot.observability import logger
from orderpilot.repository import SqliteOrderRepository
from orderpilot.settings import settings


@dataclass(frozen=True, slots=True)
class SyncSummary:
    """What a sync changed in LangSmith.

    Attributes:
        created: Examples created.
        updated: Examples updated in place.
        pruned: Remote examples deleted because their case is gone from the repo.
        stale: Remote examples whose case is gone from the repo (deleted only with prune).
    """

    created: int
    updated: int
    pruned: int
    stale: int


def check_metric_names(dataset: EvalDataset) -> None:
    """Check that every case only names metrics that have a registered evaluator.

    Args:
        dataset: The repo dataset.

    Raises:
        EvalSetupError: If a case names an unknown metric.
    """
    known = set(EvaluatorFactory.metric_names())
    for case in dataset.cases:
        unknown = sorted(set(case.metrics) - known)
        if unknown:
            raise EvalSetupError(f"Case {case.case_id} names unknown metrics: {unknown}")


def _update(case: EvalCase) -> ExampleUpdate:
    """Build the in-place update of an existing example from its case.

    Args:
        case: The repo case.

    Returns:
        The update.
    """
    example = case.to_example()
    # model_validate instead of the constructor, whose __init__ is untyped.
    return ExampleUpdate.model_validate(
        {
            "id": case.example_id,
            "inputs": example.inputs,
            "outputs": example.reference_outputs,
            "metadata": example.metadata,
            "split": example.split,
        }
    )


def sync_dataset(dataset: EvalDataset, client: Client, *, prune: bool) -> SyncSummary:
    """Create or update the LangSmith copy of a repo dataset.

    Args:
        dataset: The validated repo dataset.
        client: LangSmith client.
        prune: Also delete remote examples whose case is no longer in the repo.

    Returns:
        Counts of created, updated and pruned examples.

    Raises:
        EvalSetupError: If the remote dataset was written for a different DB build.
    """
    name = dataset.langsmith_name
    if client.has_dataset(dataset_name=name):
        remote = client.read_dataset(dataset_name=name)
        check_reference_now(
            (remote.metadata or {}).get("reference_now"),
            dataset.reference_now,
            source=f"LangSmith dataset {name}",
        )
    else:
        remote = client.create_dataset(
            name,
            description=dataset.description,
            metadata={"reference_now": dataset.reference_now.isoformat()},
        )
        logger.info("Created LangSmith dataset {}", name)

    remote_ids = {example.id for example in client.list_examples(dataset_id=remote.id)}
    new = [case for case in dataset.cases if case.example_id not in remote_ids]
    existing = [case for case in dataset.cases if case.example_id in remote_ids]
    stale = remote_ids - {case.example_id for case in dataset.cases}

    if new:
        client.create_examples(dataset_id=remote.id, examples=[c.to_example() for c in new])
    if existing:
        client.update_examples(dataset_id=remote.id, updates=[_update(c) for c in existing])
    if prune and stale:
        client.delete_examples(sorted(stale, key=str))
    return SyncSummary(
        created=len(new),
        updated=len(existing),
        pruned=len(stale) if prune else 0,
        stale=len(stale),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Sync a repo dataset to LangSmith after checking it against the local database.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` if ``None``.

    Returns:
        Process exit code: 0 on success, 2 on a setup or configuration error.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--dataset", default="m1", help="repo dataset name (default: m1)")
    parser.add_argument(
        "--prune", action="store_true", help="delete remote examples no longer in the repo"
    )
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(dataset_dir(args.dataset))
        check_metric_names(dataset)
        repo = SqliteOrderRepository(settings.DB_PATH)
        check_reference_now(
            dataset.reference_now, repo.reference_now(), source=f"Dataset {dataset.name}"
        )
        summary = sync_dataset(dataset, make_client(), prune=args.prune)
    except (EvalError, OrderPilotError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    logger.info(
        "Synced {}: {} created, {} updated, {} pruned ({} remote-only)",
        dataset.langsmith_name,
        summary.created,
        summary.updated,
        summary.pruned,
        summary.stale,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SyncSummary", "check_metric_names", "main", "sync_dataset"]
