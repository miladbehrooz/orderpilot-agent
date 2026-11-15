"""Customer-support tools exposed to the agent."""

from langchain.tools import BaseTool

from orderpilot.repository import OrderRepository
from orderpilot.tools.auth import VERIFY_TOOL_NAME, make_verify_customer
from orderpilot.tools.eligibility import (
    make_check_cancellation_eligibility,
    make_check_return_eligibility,
)
from orderpilot.tools.orders import make_get_order_details, make_list_my_orders


def build_tools(repo: OrderRepository) -> list[BaseTool]:
    """Build every agent tool bound to a repository.

    Args:
        repo: Data access shared by all tools.

    Returns:
        ``verify_customer`` followed by the customer tools.
    """
    return [
        make_verify_customer(repo),
        make_list_my_orders(repo),
        make_get_order_details(repo),
        make_check_cancellation_eligibility(repo),
        make_check_return_eligibility(repo),
    ]


__all__ = ["VERIFY_TOOL_NAME", "build_tools"]
