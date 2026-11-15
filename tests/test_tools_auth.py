"""verify_customer through a real ToolNode: state updates and model-facing content."""

from typing import Any

import pytest

from orderpilot.domain import User
from orderpilot.settings import settings
from orderpilot.tools import VERIFY_TOOL_NAME
from tests.stubs import StubRepository
from tests.tool_runner import run_tools, tool_contents


def _verify_calls(repo: StubRepository, *calls: dict[str, str], **state: Any) -> dict[str, Any]:
    """Run ``verify_customer`` calls, all in one model step.

    Args:
        repo: Repository the tools use.
        *calls: Arguments of each ``verify_customer`` call.
        **state: Initial state besides the messages, e.g. ``user_id=99``.

    Returns:
        The final graph state.
    """
    return run_tools(repo, [(VERIFY_TOOL_NAME, args) for args in calls], **state)


def test_success_sets_user_id_but_hides_it(
    stub_repo: StubRepository, stub_user: User, stub_details: dict[str, str]
) -> None:
    """Success writes user_id to state, and the model never sees it (rule 2).

    Args:
        stub_repo: Stub repository.
        stub_user: The stub account.
        stub_details: Its correct details.
    """
    result = _verify_calls(stub_repo, stub_details)
    assert result["user_id"] == stub_user.user_id
    assert result["failed_auth_attempts"] == 0
    assert tool_contents(result) == [{"reason_code": "VERIFIED"}]


def test_failure_counts_attempt(stub_repo: StubRepository, stub_details: dict[str, str]) -> None:
    """A mismatch increments the counter and reports attempts left.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
    """
    result = _verify_calls(stub_repo, {**stub_details, "last_name": "Wrong"})
    assert result.get("user_id") is None
    assert result["failed_auth_attempts"] == 1
    assert tool_contents(result) == [
        {"reason_code": "NOT_MATCHED", "attempts_remaining": settings.MAX_AUTH_ATTEMPTS - 1}
    ]


def test_parallel_failures_are_all_counted(
    stub_repo: StubRepository, stub_details: dict[str, str]
) -> None:
    """Two failed calls in one step both count, thanks to the summing reducer.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
    """
    wrong = {**stub_details, "postal_code": "00000"}
    result = _verify_calls(stub_repo, wrong, wrong)
    assert result["failed_auth_attempts"] == 2


def test_already_verified_conversation_cannot_switch_user(
    stub_repo: StubRepository, stub_details: dict[str, str]
) -> None:
    """A verified conversation refuses another verification and keeps its user_id.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
    """
    result = _verify_calls(stub_repo, stub_details, user_id=99)
    assert result["user_id"] == 99
    assert tool_contents(result) == [{"reason_code": "ALREADY_VERIFIED"}]


@pytest.mark.parametrize("missing", ["email", "postal_code"])
def test_invalid_arguments_return_structured_error(
    stub_repo: StubRepository, stub_details: dict[str, str], missing: str
) -> None:
    """Rule 9: a call missing a required argument returns INVALID_ARGUMENTS, not a trace.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
        missing: Argument left out of the call.
    """
    args = {k: v for k, v in stub_details.items() if k != missing}
    (content,) = tool_contents(_verify_calls(stub_repo, args))
    assert content["reason_code"] == "INVALID_ARGUMENTS"
    assert missing in content["detail"]
