"""Dataset sync: create vs update by stable id, pruning, the DB-build and dataset checks."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from langchain.tools import BaseTool
from langsmith import Client
from langsmith.schemas import Example
from pytest_mock import MockerFixture, MockType

from evals.cases import EvalCase, EvalDataset
from evals.errors import EvalSetupError
from evals.sync import (
    check_expected_reason_codes,
    check_expected_tool_calls,
    check_metric_names,
    check_metric_references,
    sync_dataset,
)
from orderpilot.tools import build_tools
from tests.stubs import StubRepository


def _case(
    case_id: str, metrics: list[str] | None = None, reference: dict[str, Any] | None = None
) -> EvalCase:
    """Return a minimal valid case.

    Args:
        case_id: Case id, e.g. ``P4-01``.
        metrics: Metrics of the case.
        reference: The case's reference; ``{"expected_outcome": "x"}`` if omitted.

    Returns:
        The case.
    """
    return EvalCase.model_validate(
        {
            "case_id": case_id,
            "scenario_id": case_id.split("-")[0],
            "splits": ["full"],
            "metrics": metrics or ["answer_correctness"],
            "inputs": {"customer_email": "a@example.com", "question": "?"},
            "reference": reference or {"expected_outcome": "x"},
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


def test_check_metric_names_rejects_unknown_metric(today: datetime) -> None:
    """A case naming an unregistered metric fails before anything is uploaded.

    Args:
        today: TODAY of the test shop.
    """
    dataset = EvalDataset("m9", "test", today, (_case("P3-01", ["answer_corectness"]),))
    with pytest.raises(EvalSetupError, match="answer_corectness"):
        check_metric_names(dataset)


def test_check_metric_references_requires_expected_tool_calls(today: datetime) -> None:
    """A case listing a trajectory metric without expected tool calls is refused.

    Args:
        today: TODAY of the test shop.
    """
    case = _case("P3-01", ["answer_correctness", "tool_correctness"])
    dataset = EvalDataset("m9", "test", today, (case,))
    with pytest.raises(EvalSetupError, match=r"P3-01 lists tool_correctness.*expected_tool_calls"):
        check_metric_references(dataset)


def test_check_metric_references_requires_expected_reason_code(today: datetime) -> None:
    """A case listing ``reason_code`` without an expected reason code is refused.

    Args:
        today: TODAY of the test shop.
    """
    dataset = EvalDataset("m9", "test", today, (_case("P4-01", ["reason_code"]),))
    with pytest.raises(EvalSetupError, match=r"P4-01 lists reason_code.*expected_reason_code"):
        check_metric_references(dataset)


def test_check_metric_references_accepts_filled_reference(today: datetime) -> None:
    """A case that fills every required field passes.

    Args:
        today: TODAY of the test shop.
    """
    reference = {"expected_outcome": "x", "expected_tool_calls": [{"name": "list_my_orders"}]}
    case = _case("L1-01", ["answer_correctness", "argument_correctness"], reference)
    check_metric_references(EvalDataset("m9", "test", today, (case,)))


@pytest.fixture
def tools(stub_repo: StubRepository) -> list[BaseTool]:
    """Return the real agent tools, bound to the stub repository.

    Args:
        stub_repo: Stub repository.

    Returns:
        The tool list.
    """
    return build_tools(stub_repo)


def _dataset_expecting(today: datetime, *calls: dict[str, Any]) -> EvalDataset:
    """Return a one-case dataset whose case expects the given tool calls.

    Args:
        today: TODAY of the test shop.
        *calls: Expected calls as ``{"name", "args"}`` dicts.

    Returns:
        The dataset.
    """
    reference = {"expected_outcome": "x", "expected_tool_calls": list(calls)}
    return EvalDataset("m9", "test", today, (_case("P3-01", reference=reference),))


def test_check_expected_tool_calls_accepts_real_tools(
    today: datetime, tools: list[BaseTool]
) -> None:
    """Real tools with arguments the model can set pass.

    Args:
        today: TODAY of the test shop.
        tools: Real agent tools.
    """
    dataset = _dataset_expecting(
        today,
        {"name": "list_my_orders"},
        {"name": "check_return_eligibility", "args": {"order_id": 1, "order_item_ids": [2]}},
    )
    check_expected_tool_calls(dataset, tools)


def test_check_expected_tool_calls_rejects_unknown_tool(
    today: datetime, tools: list[BaseTool]
) -> None:
    """A misspelt tool name fails with the valid names.

    Args:
        today: TODAY of the test shop.
        tools: Real agent tools.
    """
    dataset = _dataset_expecting(today, {"name": "check_return_eligibilty"})
    with pytest.raises(
        EvalSetupError, match=r"check_return_eligibilty.*Valid: .*get_order_details"
    ):
        check_expected_tool_calls(dataset, tools)


@pytest.mark.parametrize("argument", ["orderid", "runtime"])
def test_check_expected_tool_calls_rejects_unknown_argument(
    today: datetime, tools: list[BaseTool], argument: str
) -> None:
    """A misspelt argument fails, and so does an injected one the model can't set.

    Args:
        today: TODAY of the test shop.
        tools: Real agent tools.
        argument: The argument name the case expects.
    """
    dataset = _dataset_expecting(today, {"name": "get_order_details", "args": {argument: 1}})
    with pytest.raises(EvalSetupError, match=rf"get_order_details.*{argument}"):
        check_expected_tool_calls(dataset, tools)


def _dataset_expecting_code_from(today: datetime, tool: str) -> EvalDataset:
    """Return a one-case dataset whose case expects a reason code from ``tool``.

    Args:
        today: TODAY of the test shop.
        tool: Tool name the expected reason code comes from.

    Returns:
        The dataset.
    """
    reference = {
        "expected_outcome": "x",
        "expected_reason_code": {"tool": tool, "code": "WINDOW_EXPIRED"},
    }
    case = _case("P4-01", ["reason_code"], reference)
    return EvalDataset("m9", "test", today, (case,))


def test_check_expected_reason_codes_accepts_real_tool(
    today: datetime, tools: list[BaseTool]
) -> None:
    """A reason code expected from a real tool passes.

    Args:
        today: TODAY of the test shop.
        tools: Real agent tools.
    """
    check_expected_reason_codes(
        _dataset_expecting_code_from(today, "check_return_eligibility"), tools
    )


def test_check_expected_reason_codes_rejects_unknown_tool(
    today: datetime, tools: list[BaseTool]
) -> None:
    """A reason code expected from a misspelt tool fails with the valid names.

    Args:
        today: TODAY of the test shop.
        tools: Real agent tools.
    """
    dataset = _dataset_expecting_code_from(today, "check_return_eligibilty")
    with pytest.raises(
        EvalSetupError, match=r"P4-01.*check_return_eligibilty.*Valid: .*check_return_eligibility"
    ):
        check_expected_reason_codes(dataset, tools)
