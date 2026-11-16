"""Experiment runner: DB-build check, example selection, evaluate() arguments and metadata."""

import subprocess
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from langsmith import Client
from langsmith.schemas import Example
from pytest_mock import MockerFixture, MockType

from evals import run as run_module
from evals.errors import EvalSetupError
from evals.run import RunArgs, experiment_metadata, git_revision, metrics_for, parse_args
from evals.settings import eval_settings
from orderpilot.settings import settings
from tests.stubs import StubRepository


def _example(family: str, evaluators: list[str]) -> Example:
    """Return a remote example.

    Args:
        family: Scenario family in the metadata.
        evaluators: Metrics listed in the metadata.

    Returns:
        The example.
    """
    return Example.model_validate(
        {"id": uuid4(), "metadata": {"scenario_family": family, "evaluators": evaluators}}
    )


@pytest.fixture
def client(mocker: MockerFixture, today: datetime) -> MockType:
    """Return a LangSmith client mock with a dataset matching the stub DB and one example.

    Args:
        mocker: Pytest-mock fixture.
        today: TODAY of the stub repository.

    Returns:
        The client mock.
    """
    mock: MockType = mocker.create_autospec(Client, instance=True)
    mock.read_dataset.return_value.id = uuid4()
    mock.read_dataset.return_value.name = "orderpilot-m1"
    mock.read_dataset.return_value.metadata = {"reference_now": today.isoformat()}
    mock.list_examples.return_value = [_example("P", ["answer_correctness"])]
    results = mocker.MagicMock()
    results.__iter__.return_value = iter([])
    results.experiment_id = uuid4()
    mock.evaluate.return_value = results
    return mock


@pytest.fixture
def no_agent(mocker: MockerFixture) -> MockType:
    """Replace the agent target and the evaluator factory, so no model is ever built.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked ``AgentTarget`` class.
    """
    mocker.patch("evals.run.EvaluatorFactory.build", return_value=[])
    target: MockType = mocker.patch.object(run_module, "AgentTarget", autospec=True)
    return target


def test_parse_args_defaults_and_lists() -> None:
    """Defaults run the smoke split of m1; families repeat and metrics are comma-separated."""
    assert parse_args([]) == RunArgs(dataset="m1", split="smoke")
    args = parse_args(["--split", "full", "--family", "P", "--family", "L", "--evaluators", "a,b"])
    assert (args.split, args.families, args.evaluators) == ("full", ("P", "L"), ("a", "b"))


@pytest.mark.usefixtures("no_agent")
def test_run_refuses_different_db_build(client: MockType, stub_repo: StubRepository) -> None:
    """A dataset written for another DB build is never run.

    Args:
        client: LangSmith client mock.
        stub_repo: Stub repository (TODAY 2026-11-12).
    """
    client.read_dataset.return_value.metadata = {
        "reference_now": datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    }
    with pytest.raises(EvalSetupError, match="reference_now"):
        run_module.run_experiment(RunArgs("m1", "smoke"), client, stub_repo)
    client.evaluate.assert_not_called()


def test_run_passes_repeats_concurrency_and_metadata_to_evaluate(
    mocker: MockerFixture, client: MockType, no_agent: MockType, stub_repo: StubRepository
) -> None:
    """evaluate() gets k, concurrency, the agent target and comparable metadata.

    Args:
        mocker: Pytest-mock fixture.
        client: LangSmith client mock.
        no_agent: The mocked AgentTarget class.
        stub_repo: Stub repository.
    """
    mocker.patch.object(eval_settings, "REPEATS", 2)
    mocker.patch.object(eval_settings, "MAX_CONCURRENCY", 5)
    run_module.run_experiment(RunArgs("m1", "smoke"), client, stub_repo)

    call = client.evaluate.call_args
    assert call.args[0] == no_agent.return_value.invoke
    assert call.kwargs["num_repetitions"] == 2
    assert call.kwargs["max_concurrency"] == 5
    assert call.kwargs["error_handling"] == "log"
    assert call.kwargs["metadata"]["judge_prompts"] == {"answer_correctness": "v1"}
    assert call.kwargs["metadata"]["reference_now"] == "2026-11-12T00:00:00+00:00"


def test_select_examples_filters_by_family_metadata(client: MockType) -> None:
    """Each family is one metadata-filtered listing of the split.

    Args:
        client: LangSmith client mock.
    """
    client.list_examples.side_effect = [[_example("P", [])], [_example("L", [])]]
    examples = run_module.select_examples(client, uuid4(), "full", ["P", "L"])
    assert len(examples) == 2
    filters = [c.kwargs["metadata"] for c in client.list_examples.call_args_list]
    assert filters == [{"scenario_family": "P"}, {"scenario_family": "L"}]


def test_select_examples_rejects_empty_selection(client: MockType) -> None:
    """Selecting nothing is a setup error, not an empty experiment.

    Args:
        client: LangSmith client mock.
    """
    client.list_examples.return_value = []
    with pytest.raises(EvalSetupError, match="No examples"):
        run_module.select_examples(client, uuid4(), "smoke", [])


def test_metrics_default_to_those_the_cases_list() -> None:
    """Without --evaluators, the metrics are the ones the selected cases list, once each."""
    examples = [_example("P", ["a", "b"]), _example("L", ["b", "c"])]
    assert metrics_for(examples, []) == ["a", "b", "c"]
    assert metrics_for(examples, ["c"]) == ["c"]


def test_experiment_metadata_contains_versions_and_git_sha(
    mocker: MockerFixture, stub_repo: StubRepository
) -> None:
    """The metadata names agent, judge, prompt versions, commit, DB build, k and split.

    Args:
        mocker: Pytest-mock fixture.
        stub_repo: Stub repository.
    """
    mocker.patch.object(settings, "LLM_PROVIDER", "anthropic")
    mocker.patch.object(settings, "LLM_MODEL", "claude-agent")
    mocker.patch.object(eval_settings, "JUDGE_MODEL", "gpt-judge")
    mocker.patch.object(
        run_module, "git_revision", autospec=True, return_value={"git_sha": "abc1234"}
    )
    metadata: dict[str, Any] = experiment_metadata(
        RunArgs("m1", "full"), ["answer_correctness"], stub_repo
    )
    assert metadata["agent_model"] == "claude-agent"
    assert metadata["judge_model"] == "gpt-judge"
    assert metadata["judge_prompts"] == {"answer_correctness": "v1"}
    assert metadata["git_sha"] == "abc1234"
    assert metadata["split"] == "full"


def test_git_revision_outside_git(mocker: MockerFixture) -> None:
    """Without git the commit is "unknown" rather than an error.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch("evals.run.subprocess.run", autospec=True, side_effect=OSError)
    assert git_revision() == {"git_sha": "unknown", "git_dirty": False}


def test_git_revision_reports_dirty_tree(mocker: MockerFixture) -> None:
    """Uncommitted changes are flagged, so a result is never attributed to the wrong code.

    Args:
        mocker: Pytest-mock fixture.
    """
    outputs = iter(["abc1234\n", " M evals/run.py\n"])

    def _fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Return the next canned git output.

        Args:
            args: Git command.
            **kwargs: subprocess.run options.

        Returns:
            A completed process with the canned output.
        """
        return subprocess.CompletedProcess(args, 0, stdout=next(outputs))

    mocker.patch("evals.run.subprocess.run", autospec=True, side_effect=_fake_run)
    assert git_revision() == {"git_sha": "abc1234", "git_dirty": True}


def test_main_reports_setup_error(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A setup error is printed without a traceback and exits with code 2.

    Args:
        mocker: Pytest-mock fixture.
        capsys: Pytest output capture.
    """
    mocker.patch.object(
        run_module, "make_client", autospec=True, side_effect=EvalSetupError("no key")
    )
    assert run_module.main([]) == 2
    assert "error: no key" in capsys.readouterr().err
