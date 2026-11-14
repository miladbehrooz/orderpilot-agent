"""The single-agent ReAct loop as a LangGraph ``StateGraph``.

::

    START ─► route_entry ─┬─ locked ─────────────────────────► locked_reply ─► END
                          └─► agent ─┬─ no tool calls ─► END        ▲
                              ▲      └─► tools ─┬─ now locked ──────┘
                              └─────────────────┘

The graph knows only a chat model and a list of tools. Storage lives behind the tools,
so it can change without touching this module.
"""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from langgraph.types import Checkpointer, RetryPolicy

from orderpilot.agent.prompts import LOCKED_REPLY
from orderpilot.agent.state import AgentState, is_locked, is_verified
from orderpilot.logging import logger
from orderpilot.services.results import ReasonCode, ToolResult
from orderpilot.settings import settings
from orderpilot.tools import VERIFY_TOOL_NAME

AgentGraph: TypeAlias = CompiledStateGraph[AgentState, None, AgentState, AgentState]
MessagesUpdate: TypeAlias = dict[str, list[AnyMessage]]


def select_tools(state: AgentState, tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Choose the tools bound to the model: the auth gate (rule 3).

    Args:
        state: Current graph state.
        tools: Every tool the agent has.

    Returns:
        Only ``verify_customer`` before verification; every tool except it afterwards.
    """
    if is_verified(state):
        return [t for t in tools if t.name != VERIFY_TOOL_NAME]
    return [t for t in tools if t.name == VERIFY_TOOL_NAME]


def route_entry(state: AgentState) -> Literal["agent", "locked_reply"]:
    """Route a new turn: a locked conversation never reaches the model.

    Args:
        state: Current graph state.

    Returns:
        ``"locked_reply"`` if verification is locked, otherwise ``"agent"``.
    """
    return "locked_reply" if is_locked(state) else "agent"


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """Run tools if the model asked for them, otherwise finish the turn.

    Args:
        state: Current graph state.

    Returns:
        ``"tools"`` if the last message has tool calls, otherwise ``END``.
    """
    last = state["messages"][-1]
    # "__end__" is END; the literal is needed because END is typed as plain `str`.
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "__end__"


def route_after_tools(state: AgentState) -> Literal["agent", "locked_reply"]:
    """Return to the model, unless the tool calls just locked verification.

    Args:
        state: Current graph state.

    Returns:
        ``"locked_reply"`` if verification is now locked, otherwise ``"agent"``.
    """
    return "locked_reply" if is_locked(state) else "agent"


def locked_reply(state: AgentState) -> MessagesUpdate:
    """Answer with the fixed lockout message, without calling the model (AU3).

    Args:
        state: Current graph state (unused).

    Returns:
        A state update with the fixed AI message.
    """
    return {"messages": [AIMessage(content=LOCKED_REPLY)]}


def handle_tool_error(error: Exception) -> str:
    """Turn a tool exception into structured content for the model (rule 9).

    Args:
        error: The exception raised while running a tool.

    Returns:
        JSON with ``INVALID_ARGUMENTS`` (and the validation message) for bad arguments,
        or ``INTERNAL_ERROR`` for anything else, which is logged with its traceback.
    """
    if isinstance(error, ToolInvocationError):
        return ToolResult(
            reason_code=ReasonCode.INVALID_ARGUMENTS, detail=error.message
        ).to_content()
    logger.opt(exception=error).error("Tool raised an unexpected error")
    return ToolResult(reason_code=ReasonCode.INTERNAL_ERROR).to_content()


def build_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
    checkpointer: Checkpointer = None,
) -> AgentGraph:
    """Assemble and compile the single-agent loop.

    Args:
        model: Chat model that supports tool calling.
        tools: Every tool the agent may use. Must include ``verify_customer``.
        system_prompt: Rendered system prompt.
        checkpointer: Optional checkpointer for multi-turn conversations (``thread_id``).

    Returns:
        The compiled graph, limited to ``settings.AGENT_RECURSION_LIMIT`` steps per run.

    Raises:
        ValueError: If no tool is named ``verify_customer``.
    """
    if not any(t.name == VERIFY_TOOL_NAME for t in tools):
        raise ValueError(f"tools must include {VERIFY_TOOL_NAME!r}")
    system = SystemMessage(content=system_prompt)

    def agent(state: AgentState) -> MessagesUpdate:
        """Call the model with the tools allowed by the auth gate.

        Args:
            state: Current graph state.

        Returns:
            A state update with the model's response.
        """
        allowed = select_tools(state, tools)
        # Some providers reject an empty tool list, so bind only when there is something.
        runnable = model.bind_tools(allowed) if allowed else model
        response = runnable.invoke([system, *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent, retry_policy=RetryPolicy(max_attempts=3))
    # ToolNode holds every tool, so a call to a tool that wasn't bound still reaches that
    # tool's own guard instead of failing silently.
    builder.add_node("tools", ToolNode(list(tools), handle_tool_errors=handle_tool_error))
    builder.add_node("locked_reply", locked_reply)
    builder.add_conditional_edges(START, route_entry, ["agent", "locked_reply"])
    builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
    builder.add_conditional_edges("tools", route_after_tools, ["agent", "locked_reply"])
    builder.add_edge("locked_reply", END)

    graph = builder.compile(checkpointer=checkpointer, name="orderpilot")
    return graph.with_config(recursion_limit=settings.AGENT_RECURSION_LIMIT)


__all__ = [
    "AgentGraph",
    "build_graph",
    "handle_tool_error",
    "locked_reply",
    "route_after_agent",
    "route_after_tools",
    "route_entry",
    "select_tools",
]
