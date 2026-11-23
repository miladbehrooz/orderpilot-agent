"""The calibration command: turns, stored runs, grading and ``main``.

No real judge or LangSmith call is made: openevals' judge builders, the judge model and
the experiment read are mocked at their edges.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.language_models import BaseChatModel
from langsmith import Client
from langsmith.types.datasets.experiment_run_query_response import ExperimentRunQueryResponse
from langsmith.types.run import FeedbackStats
from langsmith.types.run import Run as QueryRun
from langsmith.utils import LangSmithNotFoundError
from pytest_mock import MockerFixture, MockType

from evals import calibrate as calibrate_module
from evals.calibration import stored as stored_module
from evals.calibration.grading import grade_items
from evals.calibration.labels import LabelledTurn
from evals.calibration.stored import StoredExperiment, StoredTurn, load_experiment
from evals.errors import EvalSetupError
from evals.evaluators import ArgumentQualityEvaluator, JudgeVerdict, Turn
from evals.evaluators import factory as factory_module
from evals.evaluators.judges import llm_judge as llm_judge_module
from evals.evaluators.judges import trajectory_judge as trajectory_judge_module
from tests.eval_runs import ExampleFactory, RunFactory
from tests.stubs import StubRepository


def test_turn_from_langsmith_reads_question_answer_messages_and_reference(
    make_run: RunFactory, make_example: ExampleFactory
) -> None:
    """A run and its example become the turn the judges grade.

    Args:
        make_run: Run builder.
        make_example: Example builder.
    """
    messages = [{"role": "user", "content": "Can I return order 7?"}]
    example = make_example()
    turn = Turn.from_langsmith(make_run(answer="Yes.", messages=messages), example)
    assert (turn.question, turn.answer, turn.messages) == (
        "Can I return order 7?",
        "Yes.",
        (messages[0],),
    )
    assert turn.reference == example.outputs


@pytest.fixture
def trajectory_judge_fn(mocker: MockerFixture) -> MockType:
    """Mock ``create_trajectory_llm_as_judge`` and return the judge function it builds.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked judge callable; it fails every turn unless changed.
    """
    create: MockType = mocker.patch.object(
        trajectory_judge_module, "create_trajectory_llm_as_judge", autospec=True
    )
    judge: MockType = create.return_value
    judge.return_value = {"key": "argument_quality", "score": False, "comment": "Wrong id."}
    return judge


def _item(labels: dict[str, bool], item_id: str = "P3-01+swap_order_id") -> LabelledTurn:
    """Build a labelled item on a minimal turn.

    Args:
        labels: Its labels.
        item_id: Its id.

    Returns:
        The item.
    """
    turn = Turn("Can I return order 7?", "Yes.", ({"role": "user", "content": "q"},))
    return LabelledTurn(item_id, item_id[:5], "built", turn, labels)


def test_grade_items_asks_only_labelled_metrics_with_a_judge(
    mocker: MockerFixture, trajectory_judge_fn: MockType, today: datetime
) -> None:
    """Each labelled metric that has a judge is graded ``repeats`` times; others are not.

    Args:
        mocker: Pytest-mock fixture.
        trajectory_judge_fn: The mocked judge callable.
        today: TODAY of the test shop.
    """
    judge = ArgumentQualityEvaluator(
        mocker.create_autospec(BaseChatModel, instance=True), today, "CONTRACT"
    )
    item = _item({"argument_quality": False, "tool_selection": True})
    (grade,) = grade_items([item], {"argument_quality": judge}, repeats=3)
    assert (grade.metric, grade.label, grade.verdicts) == ("argument_quality", False, (False,) * 3)
    assert grade.reasonings == ("Wrong id.",) * 3
    assert trajectory_judge_fn.call_count == 3
    assert trajectory_judge_fn.call_args.kwargs["outputs"] == list(item.turn.messages)


def test_grade_items_counts_a_failed_judge_call_as_an_error(
    mocker: MockerFixture, trajectory_judge_fn: MockType, today: datetime
) -> None:
    """A judge exception costs one verdict, not the run.

    Args:
        mocker: Pytest-mock fixture.
        trajectory_judge_fn: The mocked judge callable.
        today: TODAY of the test shop.
    """
    trajectory_judge_fn.side_effect = [
        {"key": "argument_quality", "score": True, "comment": "Fine."},
        TimeoutError("judge timed out"),
        {"key": "argument_quality", "score": True, "comment": "Fine."},
    ]
    judge = ArgumentQualityEvaluator(
        mocker.create_autospec(BaseChatModel, instance=True), today, "CONTRACT"
    )
    (grade,) = grade_items([_item({"argument_quality": True})], {"argument_quality": judge}, 3)
    assert (grade.verdicts, grade.errors, grade.complete) == ((True, True), 1, False)


def _row(runs: list[QueryRun], case_id: str = "P3-01") -> ExperimentRunQueryResponse:
    """Build one example row of the experiment-runs query.

    Args:
        runs: The example's runs.
        case_id: The example's case.

    Returns:
        The row.
    """
    return ExperimentRunQueryResponse(
        id=str(uuid4()),
        inputs={"customer_email": "a@example.com", "question": "Can I return order 7?"},
        outputs={"expected_tool_calls": [{"name": "check_return_eligibility", "args": {}}]},
        metadata={"case_id": case_id},
        runs=runs,
    )


def _client(mocker: MockerFixture, metadata: dict[str, Any]) -> MockType:
    """Return a mocked LangSmith client whose experiment has the given metadata.

    Args:
        mocker: Pytest-mock fixture.
        metadata: The experiment's metadata.

    Returns:
        The client.
    """
    client: MockType = mocker.create_autospec(Client, instance=True)
    client.read_project.return_value = mocker.Mock(
        id=uuid4(), reference_dataset_id=uuid4(), extra={"metadata": metadata}
    )
    return client


def test_load_experiment_reads_turns_and_feedback_and_skips_errored_runs(
    mocker: MockerFixture,
) -> None:
    """Each run with outputs becomes a turn with its feedback scores; errored runs don't.

    Args:
        mocker: Pytest-mock fixture.
    """
    good = QueryRun(
        id="run-1",
        outputs={"answer": "Yes.", "messages": [{"role": "user", "content": "q"}]},
        feedback_stats={
            "tool_correctness": FeedbackStats(avg=1.0),
            "notes": FeedbackStats(avg=None),
        },
    )
    failed = QueryRun(id="run-2", error="RecursionError", outputs=None)
    rows = [_row([good, failed])]
    mocker.patch.object(stored_module, "_examples_with_runs", autospec=True, return_value=rows)
    client = _client(mocker, {"reference_now": "2026-11-12T00:00:00+00:00"})

    experiment = load_experiment(client, "exp")

    assert experiment.reference_now == "2026-11-12T00:00:00+00:00"
    (stored,) = experiment.turns
    assert (stored.case_id, stored.run_id, dict(stored.feedback)) == (
        "P3-01",
        "run-1",
        {"tool_correctness": 1.0},
    )
    assert (stored.turn.question, stored.turn.answer) == ("Can I return order 7?", "Yes.")
    assert stored.turn.reference == rows[0].outputs


def test_load_experiment_rejects_an_unknown_experiment(mocker: MockerFixture) -> None:
    """A missing experiment is a setup error naming it.

    Args:
        mocker: Pytest-mock fixture.
    """
    client = mocker.create_autospec(Client, instance=True)
    client.read_project.side_effect = LangSmithNotFoundError("not found")
    with pytest.raises(EvalSetupError, match="'nope'"):
        load_experiment(client, "nope")


def test_examples_with_runs_selects_outputs_errors_and_feedback(
    mocker: MockerFixture,
) -> None:
    """The query asks for the run fields calibration needs and reads every page.

    Args:
        mocker: Pytest-mock fixture.
    """

    async def pages() -> AsyncIterator[str]:
        """Yield two rows, as the async paginator does.

        Yields:
            Row stand-ins.
        """
        for row in ("row-1", "row-2"):
            yield row

    client = mocker.Mock()
    client.datasets.experiment_runs.query.return_value = pages()
    dataset_id, experiment_id = uuid4(), uuid4()

    rows = asyncio.run(stored_module._examples_with_runs(client, dataset_id, experiment_id))

    assert rows == ["row-1", "row-2"]
    args, kwargs = client.datasets.experiment_runs.query.call_args
    assert args == (str(dataset_id),)
    assert kwargs["experiment_ids"] == [str(experiment_id)]
    assert set(kwargs["selects"]) >= {"OUTPUTS", "ERROR", "FEEDBACK_STATS"}


def test_parse_args_wants_an_odd_number_of_repeats() -> None:
    """An even count could tie the majority verdict, so it is refused."""
    assert calibrate_module.parse_args(["--experiment", "e", "--repeats", "5"]).repeats == 5
    with pytest.raises(SystemExit):
        calibrate_module.parse_args(["--experiment", "e", "--repeats", "2"])


def test_judge_metrics_default_to_every_judge_and_refuse_code_metrics() -> None:
    """Without ``--metrics`` every LLM judge is calibrated; a code metric is refused."""
    assert calibrate_module.judge_metrics([]) == [
        "answer_correctness",
        "tool_selection",
        "argument_quality",
        "trajectory_efficiency",
    ]
    with pytest.raises(EvalSetupError, match="Not LLM-judge metrics: tool_correctness"):
        calibrate_module.judge_metrics(["tool_correctness"])


def _experiment(reference_now: str = "2026-11-12T00:00:00+00:00") -> StoredExperiment:
    """Return a stored experiment with one passing P3-style run.

    Args:
        reference_now: TODAY of its database.

    Returns:
        The experiment.
    """
    messages = (
        {"role": "user", "content": "Can I return order 7?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call-0",
                    "function": {
                        "name": "check_return_eligibility",
                        "arguments": '{"order_id": 7}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-0", "content": '{"reason_code": "ELIGIBLE"}'},
        {"role": "assistant", "content": "Yes, until 2026-12-04."},
    )
    reference = {
        "expected_outcome": "Yes.",
        "expected_tool_calls": [{"name": "check_return_eligibility", "args": {"order_id": 7}}],
        "expected_reason_code": {"tool": "check_return_eligibility", "code": "ELIGIBLE"},
    }
    turn = Turn("Can I return order 7?", "Yes, until 2026-12-04.", messages, reference)
    feedback = {"tool_correctness": 1.0, "argument_correctness": 1.0, "step_count": 1.0}
    stored = StoredTurn(case_id="P3-01", run_id="0123456789", turn=turn, feedback=feedback)
    return StoredExperiment(name="exp", reference_now=reference_now, turns=(stored,))


def test_labelled_items_refuse_another_database_build(stub_repo: StubRepository) -> None:
    """Turns graded against another TODAY would get the wrong contract and dates.

    Args:
        stub_repo: Stub repository with TODAY 2026-11-12.
    """
    with pytest.raises(EvalSetupError, match="reference_now"):
        calibrate_module.labelled_items(_experiment("2026-10-01T00:00:00+00:00"), stub_repo)


def test_labelled_items_are_silver_then_built(stub_repo: StubRepository) -> None:
    """The stored run is a silver item, and its built failures follow it.

    Args:
        stub_repo: Stub repository with TODAY 2026-11-12.
    """
    ids = [item.item_id for item in calibrate_module.labelled_items(_experiment(), stub_repo)]
    assert ids == [
        "P3-01#01234567",
        "P3-01+drop_eligibility_call",
        "P3-01+swap_order_id",
        "P3-01+repeat_call",
        "P3-01+contradict_decision",
    ]


@pytest.fixture
def offline_main(mocker: MockerFixture, stub_repo: StubRepository) -> None:
    """Make ``main`` read the stub experiment and repository instead of LangSmith and SQLite.

    Args:
        mocker: Pytest-mock fixture.
        stub_repo: Stub repository with TODAY 2026-11-12.
    """
    mocker.patch.object(calibrate_module, "make_client", autospec=True)
    mocker.patch.object(
        calibrate_module, "load_experiment", autospec=True, return_value=_experiment()
    )
    mocker.patch.object(
        calibrate_module, "SqliteOrderRepository", autospec=True, return_value=stub_repo
    )


@pytest.mark.usefixtures("offline_main")
def test_dry_run_counts_without_building_a_judge(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dry run prints the items and judge calls and never builds the judge.

    Args:
        mocker: Pytest-mock fixture.
        capsys: Pytest output capture.
    """
    build_judge = mocker.patch.object(factory_module, "build_judge", autospec=True)
    assert calibrate_module.main(["--experiment", "exp", "--dry-run"]) == 0
    build_judge.assert_not_called()
    assert "judge calls (3 per item and metric)" in capsys.readouterr().out


@pytest.mark.usefixtures("offline_main")
def test_main_grades_reports_and_writes_json(
    mocker: MockerFixture,
    trajectory_judge_fn: MockType,
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A full run grades every item, prints the agreement table and writes every grade.

    Args:
        mocker: Pytest-mock fixture.
        trajectory_judge_fn: The mocked trajectory judge callable (fails every turn).
        tmp_path: Temporary directory for the JSON report.
        capsys: Pytest output capture.
    """
    mocker.patch.object(
        factory_module,
        "build_judge",
        autospec=True,
        return_value=mocker.create_autospec(BaseChatModel, instance=True),
    )
    answer_judge = mocker.patch.object(llm_judge_module, "create_llm_as_judge", autospec=True)
    answer_judge.return_value.return_value = JudgeVerdict(reasoning="Contradicts.", verdict=False)
    out = tmp_path / "calibration.json"

    code = calibrate_module.main(["--experiment", "exp", "--repeats", "1", "--out", str(out)])

    assert code == 0
    text = capsys.readouterr().out
    assert "Calibration of exp, 1 verdicts per item" in text
    assert "answer_correctness     built" in text
    grades = json.loads(out.read_text(encoding="utf-8"))["grades"]
    # 1 silver item x 3 judges + built: drop 3, swap 3, repeat 3, contradict 4.
    assert len(grades) == 16


def test_main_reports_a_setup_error(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing experiment exits with 2 and says why.

    Args:
        mocker: Pytest-mock fixture.
        capsys: Pytest output capture.
    """
    mocker.patch.object(calibrate_module, "make_client", autospec=True)
    mocker.patch.object(calibrate_module, "SqliteOrderRepository", autospec=True)
    mocker.patch.object(
        calibrate_module,
        "load_experiment",
        autospec=True,
        side_effect=EvalSetupError("No experiment named 'nope' in LangSmith"),
    )
    assert calibrate_module.main(["--experiment", "nope"]) == 2
    assert "No experiment named 'nope'" in capsys.readouterr().err
