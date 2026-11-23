"""Read a stored experiment from LangSmith: each run's turn and the feedback it got.

Calibration grades these turns again without running the agent. The runs are read with
``client.datasets.experiment_runs.query`` (examples with their runs in one query); the older
``list_runs`` and ``get_experiment_results`` are deprecated. That resource is async even on
the sync ``Client``, so it runs in :func:`asyncio.run`.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

from evals.errors import EvalSetupError
from evals.evaluators.turn import Turn


@dataclass(frozen=True, slots=True)
class StoredTurn:
    """One stored run of an experiment, with the feedback it got.

    Attributes:
        case_id: Hand-written case id, e.g. ``P4-01``.
        run_id: The LangSmith run.
        turn: The run's turn.
        feedback: Feedback key to its score on this run, e.g. ``{"tool_correctness": 1.0}``.
    """

    case_id: str
    run_id: str
    turn: Turn
    feedback: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class StoredExperiment:
    """A stored experiment: the turns its runs produced.

    Attributes:
        name: Experiment name, e.g. ``claude-haiku-5-5-full-3897d8a5``.
        reference_now: TODAY of the database it ran on, from its metadata (ISO text), or
            ``None`` if it has none.
        turns: Its runs that did not error, grouped by example in the order LangSmith
            returns them.
    """

    name: str
    reference_now: str | None
    turns: tuple[StoredTurn, ...]


def load_experiment(client: Client, name: str) -> StoredExperiment:
    """Read an experiment's runs and their feedback.

    Runs that errored, or have no outputs, are left out: they have no turn to grade.

    Args:
        client: LangSmith client.
        name: Experiment name.

    Returns:
        The stored experiment.

    Raises:
        EvalSetupError: If there is no experiment of that name, or it has no dataset.
    """
    try:
        project = client.read_project(project_name=name)
    except LangSmithNotFoundError as err:
        raise EvalSetupError(f"No experiment named {name!r} in LangSmith") from err
    if project.reference_dataset_id is None:
        raise EvalSetupError(f"{name!r} is a tracing project, not an experiment")
    metadata = (project.extra or {}).get("metadata", {})
    rows = asyncio.run(_examples_with_runs(client, project.reference_dataset_id, project.id))
    turns = tuple(turn for row in rows for turn in _stored_turns(row))
    return StoredExperiment(name=name, reference_now=metadata.get("reference_now"), turns=turns)


async def _examples_with_runs(client: Client, dataset_id: UUID, experiment_id: UUID) -> list[Any]:
    """Return the dataset's examples, each with the experiment's runs on it.

    Args:
        client: LangSmith client.
        dataset_id: The experiment's dataset.
        experiment_id: The experiment (its tracing project).

    Returns:
        ``ExperimentRunQueryResponse`` rows: example fields plus ``runs``.
    """
    pages = client.datasets.experiment_runs.query(
        str(dataset_id),
        experiment_ids=[str(experiment_id)],
        selects=["ID", "OUTPUTS", "ERROR", "FEEDBACK_STATS"],
        page_size=100,
    )
    return [row async for row in pages]


def _stored_turns(row: Any) -> list[StoredTurn]:
    """Return the gradable runs of one example row.

    Args:
        row: An ``ExperimentRunQueryResponse``: the example's ``inputs``, ``outputs`` and
            ``metadata``, and its ``runs``.

    Returns:
        One stored turn per run that has outputs and no error.
    """
    inputs = cast(Mapping[str, Any], row.inputs or {})
    reference = cast(Mapping[str, Any], row.outputs or {})
    metadata = cast(Mapping[str, Any], row.metadata or {})
    stored = []
    for run in row.runs or []:
        outputs = run.outputs
        if run.error or not outputs:
            continue
        turn = Turn(
            question=inputs["question"],
            answer=outputs.get("answer", ""),
            messages=tuple(outputs.get("messages", [])),
            reference=reference,
        )
        feedback = {
            key: stats.avg
            for key, stats in (run.feedback_stats or {}).items()
            if stats.avg is not None
        }
        stored.append(
            StoredTurn(case_id=metadata["case_id"], run_id=run.id, turn=turn, feedback=feedback)
        )
    return stored


__all__ = ["StoredExperiment", "StoredTurn", "load_experiment"]
