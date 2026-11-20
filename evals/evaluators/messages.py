"""Read a turn for the code-graded metrics.

A turn is what the target returns in ``outputs["messages"]``: the question and everything
after it, as OpenAI-format message dicts.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolCall, ToolMessage, convert_to_messages


def tool_calls(messages: Sequence[dict[str, Any]]) -> list[ToolCall]:
    """Return the tool calls made in a turn, with parsed arguments.

    Args:
        messages: The turn, in OpenAI format.

    Returns:
        The calls of every AI message, in order.
    """
    return [
        call
        for message in convert_to_messages(messages)
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def model_calls(messages: Sequence[dict[str, Any]]) -> int:
    """Count the model calls in a turn.

    Args:
        messages: The turn, in OpenAI format.

    Returns:
        The number of AI messages: one per model call, tool-calling or final.
    """
    return sum(isinstance(message, AIMessage) for message in convert_to_messages(messages))


def tool_results(messages: Sequence[dict[str, Any]], tool: str) -> list[ToolMessage]:
    """Return the results of one tool in a turn.

    A result is matched to its tool through ``tool_call_id`` and the AI message's call, not
    the tool message's optional ``name``.

    Args:
        messages: The turn, in OpenAI format.
        tool: Tool name.

    Returns:
        The tool's results, in order.
    """
    parsed = convert_to_messages(messages)
    names = {
        call["id"]: call["name"]
        for message in parsed
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    return [
        message
        for message in parsed
        if isinstance(message, ToolMessage) and names.get(message.tool_call_id) == tool
    ]


__all__ = ["model_calls", "tool_calls", "tool_results"]
