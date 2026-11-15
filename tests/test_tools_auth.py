"""verify_customer through a real ToolNode: state updates and model-facing content."""

import json
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from orderpilot.agent.graph import handle_tool_error
from orderpilot.agent.state import AgentState
from orderpilot.settings import settings
from orderpilot.tools import VERIFY_TOOL_NAME, build_tools
from tests.stubs import STUB_USER, StubRepository

CORRECT = {
    "first_name": STUB_USER.first_name,
    "last_name": STUB_USER.last_name,
    "email": STUB_USER.email,
    "postal_code": STUB_USER.postal_code,
}


def _run_tools(repo: StubRepository, calls: list[dict[str, Any]], **state: Any) -> dict[str, Any]:
    """Run tool calls through a one-node graph holding the production ToolNode setup.

    Args:
        repo: Repository the tools use.
        calls: Tool-call args, one dict per ``verify_customer`` call.
        **state: Extra initial state, e.g. ``failed_auth_attempts=2``.

    Returns:
        The final graph state.
    """
    tool_calls = [
        {"name": VERIFY_TOOL_NAME, "args": args, "id": f"call-{i}", "type": "tool_call"}
        for i, args in enumerate(calls)
    ]
    graph = (
        StateGraph(AgentState)
        .add_node("tools", ToolNode(build_tools(repo), handle_tool_errors=handle_tool_error))
        .add_edge(START, "tools")
        .add_edge("tools", END)
        .compile()
    )
    initial = cast(AgentState, {"messages": [AIMessage("", tool_calls=tool_calls)], **state})
    result: dict[str, Any] = graph.invoke(initial)
    return result


def _contents(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the parsed JSON content of every tool message.

    Args:
        result: Final graph state.

    Returns:
        One dict per ToolMessage, in order.
    """
    return [json.loads(m.text) for m in result["messages"] if isinstance(m, ToolMessage)]


def test_au1_success_sets_user_id_but_hides_it(stub_repo: StubRepository) -> None:
    """AU1: success writes user_id to state, and the model never sees it (rule 2).

    Args:
        stub_repo: Stub repository.
    """
    result = _run_tools(stub_repo, [CORRECT])
    assert result["user_id"] == STUB_USER.user_id
    assert result["failed_auth_attempts"] == 0
    assert _contents(result) == [{"reason_code": "VERIFIED"}]


def test_au2_failure_counts_attempt(stub_repo: StubRepository) -> None:
    """AU2: a mismatch increments the counter and reports attempts left.

    Args:
        stub_repo: Stub repository.
    """
    result = _run_tools(stub_repo, [{**CORRECT, "last_name": "Wrong"}])
    assert result.get("user_id") is None
    assert result["failed_auth_attempts"] == 1
    assert _contents(result) == [
        {"reason_code": "NOT_MATCHED", "attempts_remaining": settings.MAX_AUTH_ATTEMPTS - 1}
    ]


def test_au3_parallel_failures_are_all_counted(stub_repo: StubRepository) -> None:
    """AU3: two failed calls in one step both count, thanks to the summing reducer.

    Args:
        stub_repo: Stub repository.
    """
    wrong = {**CORRECT, "postal_code": "00000"}
    result = _run_tools(stub_repo, [wrong, wrong])
    assert result["failed_auth_attempts"] == 2


def test_already_verified_conversation_cannot_switch_user(stub_repo: StubRepository) -> None:
    """A verified conversation refuses another verification and keeps its user_id.

    Args:
        stub_repo: Stub repository.
    """
    result = _run_tools(stub_repo, [CORRECT], user_id=99)
    assert result["user_id"] == 99
    assert _contents(result) == [{"reason_code": "ALREADY_VERIFIED"}]


@pytest.mark.parametrize("missing", ["email", "postal_code"])
def test_invalid_arguments_return_structured_error(stub_repo: StubRepository, missing: str) -> None:
    """Rule 9: a call missing a required argument returns INVALID_ARGUMENTS, not a trace.

    Args:
        stub_repo: Stub repository.
        missing: Argument left out of the call.
    """
    args = {k: v for k, v in CORRECT.items() if k != missing}
    (content,) = _contents(_run_tools(stub_repo, [args]))
    assert content["reason_code"] == "INVALID_ARGUMENTS"
    assert missing in content["detail"]
