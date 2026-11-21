"""The agent's contract: the rules and tools a good trajectory follows, as the judges read them.

Built from the agent's real system prompt and tool schemas, so the judges can never grade
against rules that have drifted from the ones the agent was given.
"""

import json
from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_tool

from orderpilot.agent.prompts import render_system_prompt
from orderpilot.repository import OrderRepository
from orderpilot.tools import build_tools


def agent_contract(repo: OrderRepository) -> str:
    """Return the rules and tools the agent works under.

    Args:
        repo: Repository on the database under test; it gives TODAY and binds the tools.

    Returns:
        The agent's system prompt, followed by every tool's name, description and
        model-facing parameters (injected ones such as ``runtime`` are not among them).
    """
    tools = "\n\n".join(
        _describe_tool(convert_to_openai_tool(tool)["function"]) for tool in build_tools(repo)
    )
    return (
        f"<system_prompt>\n{render_system_prompt(repo.reference_now())}</system_prompt>\n\n"
        f"<tools>\n{tools}\n</tools>"
    )


def _describe_tool(function: dict[str, Any]) -> str:
    """Describe one tool for the judge.

    Args:
        function: The ``function`` part of the tool's OpenAI-format schema.

    Returns:
        The tool's name as a heading, its description and its JSON parameters.
    """
    params = json.dumps(function.get("parameters", {}), ensure_ascii=False, indent=2)
    return f"## {function['name']}\n{function.get('description', '')}\nParameters: {params}"


__all__ = ["agent_contract"]
