"""Eval target: seeding a verified thread and mapping a run to outputs; no model is called."""

from typing import Any

import pytest
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from pytest_mock import MockerFixture, MockType

from evals import target
from evals.errors import EvalSetupError
from evals.target import (
    AgentTarget,
    final_answer,
    seed_verified_thread,
    this_turn,
    verified_history,
)
from orderpilot.agent.graph import AgentGraph, build_graph, select_tools
from orderpilot.agent.state import AgentState
from orderpilot.domain import User
from orderpilot.tools import VERIFY_TOOL_NAME, build_tools
from tests.stubs import StubRepository
from tests.tool_runner import run_tools


@pytest.fixture
def tools(stub_repo: StubRepository) -> list[BaseTool]:
    """Return every agent tool, bound to the stub repository.

    Args:
        stub_repo: Stub repository.

    Returns:
        The tool list.
    """
    return build_tools(stub_repo)


@pytest.fixture
def graph(tools: list[BaseTool]) -> AgentGraph:
    """Build the real agent graph with a checkpointer and a model that must never be called.

    Args:
        tools: Agent tools.

    Returns:
        The compiled graph.
    """
    model = ChatOpenAI(
        model="gpt-test", api_key=SecretStr("sk-test"), base_url="http://127.0.0.1:9", max_retries=0
    )
    return build_graph(model, tools, system_prompt="x", checkpointer=InMemorySaver())


@pytest.fixture
def config() -> RunnableConfig:
    """Return the config of a fresh thread.

    Returns:
        A config with a thread id.
    """
    return {"configurable": {"thread_id": "eval-seed"}}


def test_seeded_thread_is_verified_and_idle(
    graph: AgentGraph, config: RunnableConfig, stub_user: User
) -> None:
    """After seeding, user_id is set, the history is in place, and no node is pending.

    Args:
        graph: Real agent graph.
        config: Fresh thread config.
        stub_user: The customer to verify.
    """
    seed_verified_thread(graph, config, stub_user)
    snapshot = graph.get_state(config)
    assert snapshot.values["user_id"] == stub_user.user_id
    assert len(snapshot.values["messages"]) == 4
    assert snapshot.next == ()


def test_seeded_thread_offers_eligibility_tools(
    graph: AgentGraph, config: RunnableConfig, stub_user: User, tools: list[BaseTool]
) -> None:
    """The auth gate sees a verified thread: verify_customer is gone, the other tools are bound.

    Args:
        graph: Real agent graph.
        config: Fresh thread config.
        stub_user: The customer to verify.
        tools: Agent tools.
    """
    seed_verified_thread(graph, config, stub_user)
    state: AgentState = graph.get_state(config).values  # type: ignore[assignment]
    names = [t.name for t in select_tools(state, tools)]
    assert VERIFY_TOOL_NAME not in names
    assert "check_return_eligibility" in names


def test_seeded_history_matches_real_verification(
    stub_repo: StubRepository, stub_user: User, stub_details: dict[str, str]
) -> None:
    """The seeded tool call and result are what the real verify_customer produces.

    Args:
        stub_repo: Stub repository.
        stub_user: The customer to verify.
        stub_details: The customer's four verification details.
    """
    real = run_tools(stub_repo, [(VERIFY_TOOL_NAME, stub_details)])
    (real_result,) = [m for m in real["messages"] if isinstance(m, ToolMessage)]

    _, call, seeded_result, _ = verified_history(stub_user)
    assert isinstance(call, AIMessage)
    assert call.tool_calls[0]["args"] == stub_details
    assert isinstance(seeded_result, ToolMessage)
    assert seeded_result.text == real_result.text
    assert seeded_result.tool_call_id == call.tool_calls[0]["id"]
    assert real["user_id"] == stub_user.user_id


def test_final_answer_is_last_ai_text() -> None:
    """The final answer is the last AI message, even when tool messages follow earlier ones."""
    messages = [HumanMessage("q"), AIMessage("first"), HumanMessage("again"), AIMessage("last")]
    assert final_answer(messages) == "last"
    assert final_answer([HumanMessage("q")]) == ""


@pytest.fixture
def mocked_build_agent(mocker: MockerFixture) -> MockType:
    """Replace build_agent with a mock graph whose run appends an AI answer to a seeded thread.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked ``build_agent``.
    """

    def _invoke(graph_input: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        """Return the thread after a run: a seeded history, the input, and an answer.

        Args:
            graph_input: The invoke input, holding the turn's question.
            config: The thread config.

        Returns:
            The final state's messages.
        """
        seeded = [HumanMessage("Hi, I'm Ann Lee."), AIMessage("Thank you, you're verified.")]
        return {
            "messages": [*seeded, *graph_input["messages"], AIMessage("Yes, until 2026-12-01.")]
        }

    build: MockType = mocker.patch.object(target, "build_agent", autospec=True)
    build.return_value.invoke.side_effect = _invoke
    return build


def test_agent_target_maps_final_answer(
    mocked_build_agent: MockType, stub_repo: StubRepository, stub_user: User
) -> None:
    """A run becomes {answer, messages}, on a new seeded thread per call.

    Args:
        mocked_build_agent: The mocked build_agent.
        stub_repo: Stub repository.
        stub_user: The case's customer.
    """
    agent = AgentTarget(stub_repo)
    inputs: dict[str, Any] = {
        "customer_email": stub_user.email,
        "question": "Can I return order 7?",
    }

    first = agent.invoke(inputs)
    agent.invoke(inputs)

    assert first["answer"] == "Yes, until 2026-12-01."
    assert first["messages"][-1] == {"role": "assistant", "content": "Yes, until 2026-12-01."}
    graph = mocked_build_agent.return_value
    seeded = [c.args[0]["configurable"]["thread_id"] for c in graph.update_state.call_args_list]
    invoked = [c.args[1]["configurable"]["thread_id"] for c in graph.invoke.call_args_list]
    assert seeded == invoked
    assert len(set(invoked)) == 2
    assert graph.update_state.call_args.kwargs["as_node"] == "agent"


@pytest.mark.usefixtures("mocked_build_agent")
def test_agent_target_returns_only_this_turn(stub_repo: StubRepository, stub_user: User) -> None:
    """The trajectory starts at the question; the seeded verification is left out.

    Args:
        stub_repo: Stub repository.
        stub_user: The case's customer.
    """
    output = AgentTarget(stub_repo).invoke(
        {"customer_email": stub_user.email, "question": "Can I return order 7?"}
    )
    assert output["messages"] == [
        {"role": "user", "content": "Can I return order 7?"},
        {"role": "assistant", "content": "Yes, until 2026-12-01."},
    ]


def test_this_turn_requires_question_in_thread() -> None:
    """A question missing from the thread is a harness bug, not an empty trajectory."""
    question = HumanMessage("q", id="question-1")
    assert this_turn([AIMessage("a"), question, AIMessage("b")], question)[0] is question
    with pytest.raises(EvalSetupError, match="question"):
        this_turn([HumanMessage("q", id="other")], question)


@pytest.mark.usefixtures("mocked_build_agent")
def test_agent_target_rejects_unknown_customer(stub_repo: StubRepository) -> None:
    """A case naming a customer who isn't in the database is a setup error, not an agent failure.

    Args:
        stub_repo: Stub repository.
    """
    agent = AgentTarget(stub_repo)
    with pytest.raises(EvalSetupError, match=r"nobody@example\.com"):
        agent.invoke({"customer_email": "nobody@example.com", "question": "hi"})
