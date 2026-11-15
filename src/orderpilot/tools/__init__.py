"""Customer-support tools exposed to the agent."""

from langchain.tools import BaseTool

from orderpilot.repository import OrderRepository
from orderpilot.tools.auth import VERIFY_TOOL_NAME, make_verify_customer


def build_tools(repo: OrderRepository) -> list[BaseTool]:
    """Build every agent tool bound to a repository.

    The read tools (orders, eligibility) are added here as they are implemented.

    Args:
        repo: Data access shared by all tools.

    Returns:
        The tools, including ``verify_customer``.
    """
    return [make_verify_customer(repo)]


__all__ = ["VERIFY_TOOL_NAME", "build_tools"]
