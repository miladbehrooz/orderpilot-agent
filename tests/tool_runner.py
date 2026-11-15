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


def run_tool_call(
    repo: OrderRepository, name: str, args: dict[str, Any], **state: Any
) -> dict[str, Any]:
    """Run one tool call through a one-node graph holding the production ToolNode setup.

    The ToolNode has every agent tool and the graph's error handler, so the call sees
    the same argument validation, state injection and error mapping as in the agent.

    Args:
        repo: Repository the tools use.
        name: Tool to call.
        args: Tool-call arguments, as the model would send them.
        **state: Initial state besides the messages, e.g. ``user_id=1``.

    Returns:
        The parsed JSON content of the resulting ToolMessage.
    """
    graph = (
        StateGraph(AgentState)
        .add_node("tools", ToolNode(build_tools(repo), handle_tool_errors=handle_tool_error))
        .add_edge(START, "tools")
        .add_edge("tools", END)
        .compile()
    )
    call = {"name": name, "args": args, "id": "call-1", "type": "tool_call"}
    initial = cast(AgentState, {"messages": [AIMessage("", tool_calls=[call])], **state})
    result: dict[str, Any] = graph.invoke(initial)
    (message,) = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    content: dict[str, Any] = json.loads(message.text)
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


__all__ = ["run_tool_call", "tools_named"]
