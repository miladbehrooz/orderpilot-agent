"""Copy a repo dataset to LangSmith: ``uv run python -m evals.sync --dataset m1 [--prune]``.

The YAML files are the source of truth. Examples are matched by their stable id (a UUID5 of
``case_id``), so re-syncing updates examples in place; LangSmith keeps the version history.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from langchain.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
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
from orderpilot.tools import build_tools


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


def check_metric_references(dataset: EvalDataset) -> None:
    """Check that every case fills the reference fields its metrics need.

    Args:
        dataset: The repo dataset; its metric names must be known (``check_metric_names``).

    Raises:
        EvalSetupError: If a case lists a metric but leaves one of its required fields empty.
    """
    for case in dataset.cases:
        for metric in case.metrics:
            required = EvaluatorFactory.required_reference(metric)
            missing = [field for field in required if not getattr(case.reference, field)]
            if missing:
                raise EvalSetupError(
                    f"Case {case.case_id} lists {metric} but has no {', '.join(missing)}"
                )


def check_expected_tool_calls(dataset: EvalDataset, tools: Sequence[BaseTool]) -> None:
    """Check that every expected tool call names a real tool and arguments the model can set.

    The arguments come from each tool's model-facing schema, so injected ones such as
    ``runtime`` are refused too.

    Args:
        dataset: The repo dataset.
        tools: The agent's tools (``build_tools``).

    Raises:
        EvalSetupError: If a call names an unknown tool or an unknown argument.
    """
    params = {
        tool.name: set(convert_to_openai_tool(tool)["function"]["parameters"].get("properties", {}))
        for tool in tools
    }
    for case in dataset.cases:
        for call in case.reference.expected_tool_calls:
            if call.name not in params:
                valid = ", ".join(params)
                raise EvalSetupError(
                    f"Case {case.case_id} expects unknown tool {call.name!r}. Valid: {valid}"
                )
            unknown = sorted(set(call.args) - params[call.name])
            if unknown:
                raise EvalSetupError(
                    f"Case {case.case_id} expects {call.name} with unknown arguments {unknown}"
                )


def check_expected_reason_codes(dataset: EvalDataset, tools: Sequence[BaseTool]) -> None:
    """Check that every expected reason code names a real tool.

    Unknown codes are already refused when the case is loaded (``ReasonCode``).

    Args:
        dataset: The repo dataset.
        tools: The agent's tools (``build_tools``).

    Raises:
        EvalSetupError: If a case expects a reason code from an unknown tool.
    """
    names = {tool.name for tool in tools}
    for case in dataset.cases:
        expected = case.reference.expected_reason_code
        if expected is not None and expected.tool not in names:
            valid = ", ".join(sorted(names))
            raise EvalSetupError(
                f"Case {case.case_id} expects a reason code from unknown tool "
                f"{expected.tool!r}. Valid: {valid}"
            )


def check_dataset(dataset: EvalDataset, tools: Sequence[BaseTool]) -> None:
    """Run every repo-side check of a dataset before it is uploaded.

    Args:
        dataset: The repo dataset.
        tools: The agent's tools (``build_tools``).

    Raises:
        EvalSetupError: If a check fails.
    """
    check_metric_names(dataset)
    check_metric_references(dataset)
    check_expected_tool_calls(dataset, tools)
    check_expected_reason_codes(dataset, tools)


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
        repo = SqliteOrderRepository(settings.DB_PATH)
        check_dataset(dataset, build_tools(repo))
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


__all__ = [
    "SyncSummary",
    "check_dataset",
    "check_expected_reason_codes",
    "check_expected_tool_calls",
    "check_metric_names",
    "check_metric_references",
    "main",
    "sync_dataset",
]
