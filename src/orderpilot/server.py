"""Entry point for the LangGraph dev server (``langgraph dev``), referenced by ``langgraph.json``.

The server loads this module once and runs every conversation (thread) on ``graph``.
No checkpointer is passed: the server persists threads itself.
"""

from orderpilot.agent.graph import AgentGraph
from orderpilot.app import build_agent

graph: AgentGraph = build_agent()

__all__ = ["graph"]
