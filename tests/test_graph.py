"""Graph wiring, the auth gate and routing. The model is never called."""

import json
from typing import cast

import pytest
from langchain.tools import BaseTool, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState
from pydantic import SecretStr

from orderpilot.agent.graph import (
    AgentGraph,
    build_graph,
    handle_tool_error,
    route_after_agent,
    route_after_tools,
    route_entry,
    select_tools,
)
from orderpilot.agent.prompts import LOCKED_REPLY
from orderpilot.agent.state import AgentState
from orderpilot.settings import settings
from orderpilot.tools import VERIFY_TOOL_NAME, build_tools
from tests.stubs import StubRepository


@tool
def list_my_orders() -> str:
    """Stand-in for a customer tool; never executed in these tests."""
    return "[]"


@pytest.fixture
def tools(stub_repo: StubRepository) -> list[BaseTool]:
    """Return verify_customer plus one customer tool.

    Args:
        stub_repo: Stub repository.

    Returns:
        The tool list.
    """
    return [*build_tools(stub_repo), list_my_orders]


@pytest.fixture
def unreachable_model() -> ChatOpenAI:
    """Return a real chat model pointing at a closed port, so any call fails fast.

    Returns:
        A ChatOpenAI that must never be invoked by these tests.
    """
    return ChatOpenAI(
        model="gpt-test", api_key=SecretStr("sk-test"), base_url="http://127.0.0.1:9", max_retries=0
    )


def _state(*messages: object, **fields: object) -> AgentState:
    """Return a graph state with the given messages and extra fields.

    Args:
        *messages: Conversation messages (defaults to one human message).
        **fields: Extra state fields.

    Returns:
        The state.
    """
    msgs = list(messages) or [HumanMessage("hi")]
    return cast(AgentState, {"messages": msgs, **fields})


def test_au5_only_verify_tool_before_verification(tools: list[BaseTool]) -> None:
    """AU5 / rule 3: before verification the model gets verify_customer only.

    Args:
        tools: Agent tools.
    """
    assert [t.name for t in select_tools(_state(), tools)] == [VERIFY_TOOL_NAME]


def test_verify_tool_dropped_after_verification(tools: list[BaseTool]) -> None:
    """After verification the model gets every tool except verify_customer.

    Args:
        tools: Agent tools.
    """
    assert [t.name for t in select_tools(_state(user_id=1), tools)] == ["list_my_orders"]


@pytest.mark.parametrize(
    ("failed", "expected"),
    [pytest.param(0, "agent", id="open"), pytest.param(3, "locked_reply", id="locked")],
)
def test_au3_route_entry(failed: int, expected: str) -> None:
    """AU3: a locked conversation skips the model at the start of a turn.

    Args:
        failed: Failed attempts in state.
        expected: Expected next node.
    """
    assert route_entry(_state(failed_auth_attempts=failed)) == expected


@pytest.mark.parametrize(
    ("failed", "expected"),
    [pytest.param(1, "agent", id="open"), pytest.param(3, "locked_reply", id="locked")],
)
def test_au3_route_after_tools(failed: int, expected: str) -> None:
    """AU3: the tool call that locks verification goes straight to the fixed reply.

    Args:
        failed: Failed attempts in state after the tools ran.
        expected: Expected next node.
    """
    assert route_after_tools(_state(failed_auth_attempts=failed)) == expected


def test_route_after_agent_runs_requested_tools() -> None:
    """A model message with tool calls goes to the tool node."""
    call = {"name": VERIFY_TOOL_NAME, "args": {}, "id": "1", "type": "tool_call"}
    assert route_after_agent(_state(AIMessage("", tool_calls=[call]))) == "tools"


def test_route_after_agent_ends_on_plain_answer() -> None:
    """A model message without tool calls ends the turn."""
    assert route_after_agent(_state(AIMessage("Hello!"))) == END


def test_unexpected_tool_error_is_structured_and_hides_details() -> None:
    """Rule 9: an unexpected exception becomes INTERNAL_ERROR without its message."""
    content = json.loads(handle_tool_error(RuntimeError("db password is hunter2")))
    assert content == {"reason_code": "INTERNAL_ERROR"}


def test_build_graph_requires_verify_tool(unreachable_model: ChatOpenAI) -> None:
    """The graph refuses to build without verify_customer (the auth gate needs it).

    Args:
        unreachable_model: Model that is never called.
    """
    with pytest.raises(ValueError, match=VERIFY_TOOL_NAME):
        build_graph(unreachable_model, [list_my_orders], system_prompt="x")


def test_graph_has_expected_nodes(unreachable_model: ChatOpenAI, tools: list[BaseTool]) -> None:
    """The compiled graph has the agent, tools and locked_reply nodes.

    Args:
        unreachable_model: Model that is never called.
        tools: Agent tools.
    """
    graph = build_graph(unreachable_model, tools, system_prompt="x")
    assert {"agent", "tools", "locked_reply"} <= set(graph.get_graph().nodes)
    assert graph.config is not None
    assert graph.config["recursion_limit"] == settings.AGENT_RECURSION_LIMIT


def test_au3_locked_conversation_gets_fixed_reply_without_model(
    unreachable_model: ChatOpenAI, tools: list[BaseTool]
) -> None:
    """AU3: a locked conversation answers with LOCKED_REPLY and never calls the model.

    The model points at a closed port, so any call would fail the test.

    Args:
        unreachable_model: Model that is never called.
        tools: Agent tools.
    """
    graph, config = _locked_thread(unreachable_model, tools)
    result = graph.invoke(MessagesState(messages=[HumanMessage("Where is my order?")]), config)
    *_, last = result["messages"]
    assert isinstance(last, AIMessage)
    assert last.content == LOCKED_REPLY
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])


def test_s1_client_cannot_set_user_id_through_input(
    unreachable_model: ChatOpenAI, tools: list[BaseTool]
) -> None:
    """S1 / rule 2: user_id sent in the graph input is dropped (input accepts messages only).

    A locked thread is used so the run never reaches the model.

    Args:
        unreachable_model: Model that is never called.
        tools: Agent tools.
    """
    graph, config = _locked_thread(unreachable_model, tools)
    forged = {"messages": [HumanMessage("hi")], "user_id": 42, "failed_auth_attempts": -3}
    graph.invoke(cast(MessagesState, forged), config)
    values = graph.get_state(config).values
    assert values.get("user_id") is None
    assert values["failed_auth_attempts"] == settings.MAX_AUTH_ATTEMPTS
    assert graph.get_input_jsonschema()["properties"].keys() == {"messages"}


def _locked_thread(model: ChatOpenAI, tools: list[BaseTool]) -> tuple[AgentGraph, RunnableConfig]:
    """Build a graph with a checkpointer and a thread already locked out.

    Args:
        model: Chat model (never called on a locked thread).
        tools: Agent tools.

    Returns:
        The graph and the config of the locked thread.
    """
    graph = build_graph(model, tools, system_prompt="x", checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "locked"}}
    # as_node: on a fresh thread the update must be attributed to a node; "tools" is the
    # node that writes the counter in a real conversation.
    graph.update_state(
        config, {"failed_auth_attempts": settings.MAX_AUTH_ATTEMPTS}, as_node="tools"
    )
    return graph, config
