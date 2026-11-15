"""Terminal chat (``python -m orderpilot``): the input loop around a mocked agent."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pytest_mock import MockerFixture, MockType

from orderpilot import __main__ as chat
from orderpilot.errors import LLMConfigError, RepositoryError


@pytest.fixture
def build_agent(mocker: MockerFixture) -> MockType:
    """Replace build_agent with an autospec mock, so no model or database is used.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked ``build_agent``.
    """
    build: MockType = mocker.patch.object(chat, "build_agent", autospec=True)
    return build


@pytest.fixture
def graph(build_agent: MockType) -> MockType:
    """Return the mocked agent graph, which answers every message with "Hello Ann".

    Args:
        build_agent: The mocked ``build_agent``.

    Returns:
        The graph that ``build_agent`` returns.
    """
    graph: MockType = build_agent.return_value
    graph.invoke.return_value = {"messages": [HumanMessage("hi"), AIMessage("Hello Ann")]}
    return graph


def _type(mocker: MockerFixture, *lines: str | type[BaseException]) -> None:
    """Make ``input()`` return each line in turn, raising the ones that are exceptions.

    Args:
        mocker: Pytest-mock fixture.
        *lines: What the user types, or ``EOFError`` (Ctrl-D) / ``KeyboardInterrupt`` (Ctrl-C).
    """
    mocker.patch("builtins.input", autospec=True, side_effect=list(lines))


def test_sends_message_and_prints_reply(
    mocker: MockerFixture, graph: MockType, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typed line goes to the agent as a human message, and its answer is printed.

    Args:
        mocker: Pytest-mock fixture.
        graph: The mocked agent graph.
        capsys: Pytest output capture.
    """
    _type(mocker, "  Where is my order? ", EOFError)
    assert chat.main() == 0
    (state, _config) = graph.invoke.call_args.args
    assert state == {"messages": [HumanMessage("Where is my order?")]}
    assert "agent> Hello Ann" in capsys.readouterr().out


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(["exit"], id="exit"),
        pytest.param(["QUIT"], id="quit-any-case"),
        pytest.param([EOFError], id="ctrl-d"),
        pytest.param([KeyboardInterrupt], id="ctrl-c"),
        pytest.param(["   ", "exit"], id="blank-line-skipped"),
    ],
)
def test_exits_cleanly_without_calling_agent(
    mocker: MockerFixture, graph: MockType, lines: list[str | type[BaseException]]
) -> None:
    """Exit words, Ctrl-D and Ctrl-C end the chat with 0; blank lines send nothing.

    Args:
        mocker: Pytest-mock fixture.
        graph: The mocked agent graph.
        lines: What the user types.
    """
    _type(mocker, *lines)
    assert chat.main() == 0
    graph.invoke.assert_not_called()


def test_whole_session_is_one_conversation(
    mocker: MockerFixture, build_agent: MockType, graph: MockType
) -> None:
    """The agent gets a checkpointer and every turn uses the same thread, so it remembers.

    Args:
        mocker: Pytest-mock fixture.
        build_agent: The mocked ``build_agent``.
        graph: The mocked agent graph.
    """
    _type(mocker, "first", "second", EOFError)
    chat.main()
    assert isinstance(build_agent.call_args.kwargs["checkpointer"], InMemorySaver)
    first, second = (c.args[1] for c in graph.invoke.call_args_list)
    assert first == second
    assert first["configurable"]["thread_id"]


def test_build_error_exits_1(build_agent: MockType, capsys: pytest.CaptureFixture[str]) -> None:
    """A configuration error while building the agent exits with 1 and a message, no trace.

    Args:
        build_agent: The mocked ``build_agent``.
        capsys: Pytest output capture.
    """
    build_agent.side_effect = LLMConfigError("Set LLM_PROVIDER and LLM_MODEL in .env")
    assert chat.main() == 1
    assert "error: Set LLM_PROVIDER and LLM_MODEL in .env" in capsys.readouterr().err


def test_agent_error_exits_1(
    mocker: MockerFixture, graph: MockType, capsys: pytest.CaptureFixture[str]
) -> None:
    """A data error during a turn exits with 1 and a message, no trace.

    Args:
        mocker: Pytest-mock fixture.
        graph: The mocked agent graph.
        capsys: Pytest output capture.
    """
    graph.invoke.side_effect = RepositoryError("Database not found")
    _type(mocker, "Where is my order?")
    assert chat.main() == 1
    assert "error: Database not found" in capsys.readouterr().err
