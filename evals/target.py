"""The system under test: the real agent, run single-turn on a seeded, already-verified thread.

The graph's input schema drops ``user_id`` on purpose (rule 2), and the evals don't add any
eval-only hook to production code. Instead, each example gets a fresh checkpointed thread
whose state is written directly with ``update_state``, as a real verification would have
left it. See ``docs/evals.md`` (Target layer).
"""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from evals.errors import EvalSetupError
from orderpilot.agent.graph import AgentGraph
from orderpilot.app import build_agent
from orderpilot.domain import User
from orderpilot.repository import OrderRepository
from orderpilot.services import ReasonCode
from orderpilot.services.results import VerifyResult
from orderpilot.tools import VERIFY_TOOL_NAME


def verified_history(user: User) -> list[AnyMessage]:
    """Return the messages a successful verification leaves in the conversation.

    The ``ToolMessage`` content is the real ``VERIFIED`` result model, so its format can't
    drift from what ``verify_customer`` returns. Without this history the model would see a
    verified state but no verification, and might ask the customer to verify again.

    Args:
        user: The customer to appear verified.

    Returns:
        Customer details, the ``verify_customer`` call, its result, and a short reply.
    """
    call_id = "seed-verify-customer"
    details = {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "postal_code": user.postal_code,
    }
    return [
        HumanMessage(
            f"Hi, I'm {user.first_name} {user.last_name}, email {user.email}, "
            f"postal code {user.postal_code}."
        ),
        AIMessage("", tool_calls=[{"name": VERIFY_TOOL_NAME, "args": details, "id": call_id}]),
        ToolMessage(
            VerifyResult(reason_code=ReasonCode.VERIFIED).to_content(),
            tool_call_id=call_id,
            name=VERIFY_TOOL_NAME,
        ),
        AIMessage("Thank you, you're verified. How can I help you today?"),
    ]


def seed_verified_thread(graph: AgentGraph, config: RunnableConfig, user: User) -> None:
    """Put a verified customer on a fresh thread without going through the graph input.

    ``as_node="agent"`` is required: on a fresh thread an update without it goes through
    the input schema and ``user_id`` is dropped. Because the last seeded message has no
    tool calls, the thread then ends (``next == ()``) and the next ``invoke`` starts a new
    turn from START.

    Args:
        graph: The compiled agent graph; it must have a checkpointer.
        config: Config naming the fresh thread.
        user: The customer to verify.
    """
    graph.update_state(
        config,
        {"user_id": user.user_id, "messages": verified_history(user)},
        as_node="agent",
    )


def final_answer(messages: Sequence[BaseMessage]) -> str:
    """Return the agent's final reply.

    Args:
        messages: The conversation after the run.

    Returns:
        The text of the last AI message, or ``""`` if there is none.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.text
    return ""


class AgentTarget:
    """The LangSmith target: one agent graph, one fresh seeded thread per example.

    Args:
        repo: Data access for the agent and for looking up the case's customer.
    """

    def __init__(self, repo: OrderRepository) -> None:
        self._repo = repo
        self._graph = build_agent(repo=repo, checkpointer=InMemorySaver())

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one single-turn case.

        LangSmith receives this bound method as the target; unlike an object, it has a
        ``__name__``.

        Args:
            inputs: Example inputs: ``customer_email`` and ``question``.

        Returns:
            ``answer`` (the final reply) and ``messages`` (the full trajectory in OpenAI
            format, for trajectory metrics).

        Raises:
            EvalSetupError: If the case's customer is not in the database; that is a bug in
                the case, not an agent failure.
        """
        user = self._repo.find_user_by_email(inputs["customer_email"])
        if user is None:
            raise EvalSetupError(f"Unknown customer_email {inputs['customer_email']!r}")
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
        seed_verified_thread(self._graph, config, user)
        state = self._graph.invoke({"messages": [HumanMessage(inputs["question"])]}, config)
        messages: list[AnyMessage] = state["messages"]
        return {
            "answer": final_answer(messages),
            "messages": convert_to_openai_messages(messages),
        }


__all__ = ["AgentTarget", "final_answer", "seed_verified_thread", "verified_history"]
