"""Composition root: wires the repository, tools, model and prompt into the agent graph.

This is the only module that knows about every layer. Evals can pass their own
repository (e.g. one on a fresh database copy) and checkpointer.
"""

from langgraph.types import Checkpointer

from orderpilot.agent.graph import AgentGraph, build_graph
from orderpilot.agent.prompts import render_system_prompt
from orderpilot.errors import LLMConfigError
from orderpilot.llm import LLMFactory
from orderpilot.repository import OrderRepository, SqliteOrderRepository
from orderpilot.settings import settings
from orderpilot.tools import build_tools


def build_agent(
    *, checkpointer: Checkpointer = None, repo: OrderRepository | None = None
) -> AgentGraph:
    """Build the support agent from settings.

    Args:
        checkpointer: Optional checkpointer for multi-turn conversations.
        repo: Data access to use; defaults to the SQLite database at ``settings.DB_PATH``.

    Returns:
        The compiled agent graph.

    Raises:
        LLMConfigError: If ``LLM_PROVIDER`` or ``LLM_MODEL`` is not set, or the provider's
            API key is missing.
    """
    if settings.LLM_PROVIDER is None or settings.LLM_MODEL is None:
        raise LLMConfigError("Set LLM_PROVIDER and LLM_MODEL in .env")
    repo = repo or SqliteOrderRepository(settings.DB_PATH)
    model = LLMFactory.build(
        settings.LLM_PROVIDER,
        model_name=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )
    return build_graph(
        model,
        build_tools(repo),
        system_prompt=render_system_prompt(repo.reference_now()),
        checkpointer=checkpointer,
    )


__all__ = ["build_agent"]
