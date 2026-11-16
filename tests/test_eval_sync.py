"""Dataset sync: create vs update by stable id, pruning, and the DB-build checks."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from langsmith import Client
from langsmith.schemas import Example
from pytest_mock import MockerFixture, MockType

from evals.cases import EvalCase, EvalDataset
from evals.errors import EvalSetupError
from evals.sync import check_evaluator_names, sync_dataset


def _case(case_id: str, evaluators: list[str] | None = None) -> EvalCase:
    """Return a minimal valid case.

    Args:
        case_id: Case id, e.g. ``P4-01``.
        evaluators: Metrics of the case.

    Returns:
        The case.
    """
    return EvalCase.model_validate(
        {
            "case_id": case_id,
            "scenario_id": case_id.split("-")[0],
            "splits": ["full"],
            "evaluators": evaluators or ["answer_correctness"],
            "inputs": {"customer_email": "a@example.com", "question": "?"},
            "reference": {"expected_outcome": "x"},
        }
    )


@pytest.fixture
def dataset(today: datetime) -> EvalDataset:
    """Return a dataset with two cases, written for the test shop's reference_now.

    Args:
        today: TODAY of the test shop.

    Returns:
        The dataset.
    """
    return EvalDataset("m9", "test", today, (_case("P3-01"), _case("P4-01")))


@pytest.fixture
def client(mocker: MockerFixture, today: datetime) -> MockType:
    """Return a LangSmith client mock whose dataset exists and matches ``today``.

    Args:
        mocker: Pytest-mock fixture.
        today: TODAY of the test shop.

    Returns:
        The client mock.
    """
    mock: MockType = mocker.create_autospec(Client, instance=True)
    mock.has_dataset.return_value = True
    mock.read_dataset.return_value.id = uuid4()
    mock.read_dataset.return_value.metadata = {"reference_now": today.isoformat()}
    mock.list_examples.return_value = []
    return mock


def _remote(*ids: Any) -> list[Example]:
    """Return remote examples with the given ids.

    Args:
        *ids: Example ids.

    Returns:
        The examples.
    """
    return [Example.model_validate({"id": example_id}) for example_id in ids]


def test_sync_creates_new_and_updates_existing_examples(
    client: MockType, dataset: EvalDataset
) -> None:
    """Cases already in LangSmith are updated in place; the others are created.

    Args:
        client: LangSmith client mock.
        dataset: Two-case dataset.
    """
    existing, new = dataset.cases
    client.list_examples.return_value = _remote(existing.example_id)

    summary = sync_dataset(dataset, client, prune=False)

    (created,) = client.create_examples.call_args.kwargs["examples"]
    (updated,) = client.update_examples.call_args.kwargs["updates"]
    assert created.id == new.example_id
    assert updated.id == existing.example_id
    assert (summary.created, summary.updated, summary.pruned) == (1, 1, 0)


def test_sync_creates_missing_dataset_with_reference_now(
    client: MockType, dataset: EvalDataset, today: datetime
) -> None:
    """A missing dataset is created with the DB build it was written for.

    Args:
        client: LangSmith client mock.
        dataset: Two-case dataset.
        today: TODAY of the test shop.
    """
    client.has_dataset.return_value = False
    sync_dataset(dataset, client, prune=False)
    call = client.create_dataset.call_args
    assert call.args[0] == "orderpilot-m9"
    assert call.kwargs["metadata"] == {"reference_now": today.isoformat()}


@pytest.mark.parametrize(("prune", "deleted"), [(False, 0), (True, 1)])
def test_sync_prune_deletes_remote_only_examples(
    client: MockType, dataset: EvalDataset, prune: bool, deleted: int
) -> None:
    """A remote example whose case is gone is deleted only with --prune.

    Args:
        client: LangSmith client mock.
        dataset: Two-case dataset.
        prune: Whether to prune.
        deleted: Expected number of deleted examples.
    """
    orphan = uuid4()
    client.list_examples.return_value = _remote(orphan)
    summary = sync_dataset(dataset, client, prune=prune)
    assert (summary.pruned, summary.stale) == (deleted, 1)
    if prune:
        client.delete_examples.assert_called_once_with([orphan])
    else:
        client.delete_examples.assert_not_called()


def test_sync_refuses_dataset_of_other_db_build(client: MockType, dataset: EvalDataset) -> None:
    """A remote dataset written for another DB build is never overwritten.

    Args:
        client: LangSmith client mock.
        dataset: Two-case dataset.
    """
    client.read_dataset.return_value.metadata = {
        "reference_now": datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    }
    with pytest.raises(EvalSetupError, match="reference_now"):
        sync_dataset(dataset, client, prune=False)
    client.create_examples.assert_not_called()


def test_check_evaluator_names_rejects_unknown_metric(today: datetime) -> None:
    """A case naming an unregistered metric fails before anything is uploaded.

    Args:
        today: TODAY of the test shop.
    """
    dataset = EvalDataset("m9", "test", today, (_case("P3-01", ["answer_corectness"]),))
    with pytest.raises(EvalSetupError, match="answer_corectness"):
        check_evaluator_names(dataset)
