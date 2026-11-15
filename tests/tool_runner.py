"""Run tool calls through the production ToolNode setup, without a model."""

import json
from typing import Any, cast

from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from orderpilot.agent.graph import handle_tool_error
from orderpilot.agent.state import AgentState
from orderpilot.repository import OrderRepository
from orderpilot.tools import build_tools


def run_tools(
    repo: OrderRepository, calls: list[tuple[str, dict[str, Any]]], **state: Any
) -> dict[str, Any]:
    """Run tool calls, all in one model step, through the production ToolNode setup.

    The one-node graph holds every agent tool and the graph's error handler, so the
    calls see the same argument validation, state injection, state updates and error
    mapping as in the agent.

    Args:
        repo: Repository the tools use.
        calls: ``(tool name, arguments)`` per call, as the model would send them.
        **state: Initial state besides the messages, e.g. ``user_id=1``.

    Returns:
        The final graph state.
    """
    graph = (
        StateGraph(AgentState)
        .add_node("tools", ToolNode(build_tools(repo), handle_tool_errors=handle_tool_error))
        .add_edge(START, "tools")
        .add_edge("tools", END)
        .compile()
    )
    tool_calls = [
        {"name": name, "args": args, "id": f"call-{i}", "type": "tool_call"}
        for i, (name, args) in enumerate(calls)
    ]
    initial = cast(AgentState, {"messages": [AIMessage("", tool_calls=tool_calls)], **state})
    result: dict[str, Any] = graph.invoke(initial)
    return result


def tool_contents(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the parsed JSON content of every ToolMessage in a graph state.

    Args:
        result: Final graph state from :func:`run_tools`.

    Returns:
        One dict per ToolMessage, in order.
    """
    contents: list[dict[str, Any]] = [
        json.loads(m.text) for m in result["messages"] if isinstance(m, ToolMessage)
    ]
    return contents


def run_tool_call(
    repo: OrderRepository, name: str, args: dict[str, Any], **state: Any
) -> dict[str, Any]:
    """Run one tool call through the production ToolNode setup.

    Args:
        repo: Repository the tools use.
        name: Tool to call.
        args: Tool-call arguments, as the model would send them.
        **state: Initial state besides the messages, e.g. ``user_id=1``.

    Returns:
        The parsed JSON content of the resulting ToolMessage.
    """
    (content,) = tool_contents(run_tools(repo, [(name, args)], **state))
    return content


def tools_named(repo: OrderRepository, *names: str) -> list[BaseTool]:
    """Return the agent tools with the given names, in ``build_tools`` order.

    Args:
        repo: Repository the tools use.
        *names: Tool names to keep.

    Returns:
        The matching tools.
    """
    return [t for t in build_tools(repo) if t.name in names]


__all__ = ["run_tool_call", "run_tools", "tool_contents", "tools_named"]
