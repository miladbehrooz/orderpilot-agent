"""Eval cases: YAML loading, defaults, validation and conversion to LangSmith examples."""

from pathlib import Path
from textwrap import dedent

import pytest

from evals.cases import EvalCase, dataset_dir, load_dataset
from evals.errors import EvalSetupError


def _write(directory: Path, name: str, text: str) -> None:
    """Write a dedented YAML file into a dataset directory.

    Args:
        directory: Dataset directory.
        name: File name.
        text: YAML content (indentation is removed).
    """
    (directory / name).write_text(dedent(text), encoding="utf-8")


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    """Create a small dataset: dataset.yaml and one family file with two cases.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The dataset directory.
    """
    directory = tmp_path / "m9"
    directory.mkdir()
    _write(
        directory,
        "dataset.yaml",
        """\
        description: test dataset
        reference_now: "2026-11-12T00:00:00+00:00"
        """,
    )
    _write(
        directory,
        "P.yaml",
        """\
        defaults:
          splits: [full]
          evaluators: [answer_correctness]
        cases:
          - case_id: P4-01
            scenario_id: P4
            splits: [full, smoke]
            inputs: {customer_email: a@example.com, question: "Can I return order 1?"}
            reference: {expected_outcome: "No, outside the window", key_facts: {order_id: 1}}
          - case_id: AU3-01
            scenario_id: AU3
            inputs: {customer_email: b@example.com, question: "hi"}
            reference: {expected_outcome: "Locked"}
        """,
    )
    return directory


def test_load_dataset_merges_file_defaults(dataset_path: Path) -> None:
    """A case inherits its file's defaults unless it sets the field itself.

    Args:
        dataset_path: Test dataset directory.
    """
    dataset = load_dataset(dataset_path)
    by_id = {case.case_id: case for case in dataset.cases}
    assert by_id["P4-01"].splits == ["full", "smoke"]
    assert by_id["AU3-01"].splits == ["full"]
    assert by_id["AU3-01"].evaluators == ["answer_correctness"]
    assert dataset.name == "m9"
    assert dataset.langsmith_name == "orderpilot-m9"
    assert dataset.reference_now.isoformat() == "2026-11-12T00:00:00+00:00"


def test_case_family_derived_from_scenario_id(dataset_path: Path) -> None:
    """The family is the scenario id without its number, for one- and two-letter families.

    Args:
        dataset_path: Test dataset directory.
    """
    families = {case.case_id: case.family for case in load_dataset(dataset_path).cases}
    assert families == {"P4-01": "P", "AU3-01": "AU"}


def test_example_id_is_stable(dataset_path: Path) -> None:
    """The LangSmith example id depends only on case_id, so re-syncing updates in place.

    Args:
        dataset_path: Test dataset directory.
    """
    first = load_dataset(dataset_path).cases[0]
    again = load_dataset(dataset_path).cases[0]
    changed = first.model_copy(update={"notes": "edited"})
    assert first.example_id == again.example_id == changed.example_id


def test_to_example_keeps_bookkeeping_out_of_inputs(dataset_path: Path) -> None:
    """The agent sees only customer_email and question; scenario ids go to metadata.

    Args:
        dataset_path: Test dataset directory.
    """
    case = load_dataset(dataset_path).cases[0]
    example = case.to_example()
    assert example.inputs == {
        "customer_email": "a@example.com",
        "question": "Can I return order 1?",
    }
    assert example.reference_outputs == {
        "expected_outcome": "No, outside the window",
        "key_facts": {"order_id": 1},
    }
    assert example.metadata is not None
    assert example.metadata["scenario_id"] == "P4"
    assert example.metadata["scenario_family"] == "P"
    assert example.metadata["evaluators"] == ["answer_correctness"]
    assert example.split == ["full", "smoke"]
    assert example.id == case.example_id


def test_load_dataset_rejects_duplicate_case_id(dataset_path: Path) -> None:
    """Two cases with one case_id would share a LangSmith example, so loading fails.

    Args:
        dataset_path: Test dataset directory.
    """
    _write(
        dataset_path,
        "Q.yaml",
        """\
        cases:
          - case_id: P4-01
            scenario_id: P4
            splits: [full]
            evaluators: [answer_correctness]
            inputs: {customer_email: c@example.com, question: "?"}
            reference: {expected_outcome: "x"}
        """,
    )
    with pytest.raises(EvalSetupError, match="Duplicate case_id P4-01"):
        load_dataset(dataset_path)


def test_load_dataset_reports_schema_error_with_case_id(dataset_path: Path) -> None:
    """A schema error names the file and the case, so a typo is easy to find.

    Args:
        dataset_path: Test dataset directory.
    """
    _write(
        dataset_path,
        "S.yaml",
        """\
        cases:
          - case_id: S1-01
            scenario_id: S1
            splits: [nightly]
            evaluators: [answer_correctness]
            inputs: {customer_email: d@example.com, question: "?"}
            reference: {expected_outcome: "x"}
        """,
    )
    with pytest.raises(EvalSetupError, match=r"S\.yaml: case S1-01"):
        load_dataset(dataset_path)


def test_load_dataset_requires_dataset_file(tmp_path: Path) -> None:
    """A directory without dataset.yaml is not a dataset.

    Args:
        tmp_path: Pytest temporary directory.
    """
    with pytest.raises(EvalSetupError, match=r"dataset\.yaml"):
        load_dataset(tmp_path)


def test_case_rejects_unknown_field() -> None:
    """Unknown fields are errors, so a misspelt field never silently falls back to a default."""
    with pytest.raises(ValueError, match="scenario"):
        EvalCase.model_validate(
            {
                "case_id": "P1-01",
                "scenario": "P1",
                "splits": ["full"],
                "evaluators": ["answer_correctness"],
                "inputs": {"customer_email": "e@example.com", "question": "?"},
                "reference": {"expected_outcome": "x"},
            }
        )


def test_repo_datasets_are_valid() -> None:
    """The committed m1 dataset loads, and every scenario has a smoke case."""
    dataset = load_dataset(dataset_dir("m1"))
    scenarios = {case.scenario_id for case in dataset.cases}
    smoke = {case.scenario_id for case in dataset.cases if "smoke" in case.splits}
    assert scenarios == smoke
