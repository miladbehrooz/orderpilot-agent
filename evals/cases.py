"""Hand-written eval cases: schema, YAML loader and conversion to LangSmith examples.

The YAML files in ``evals/datasets/<name>/`` are the source of truth. ``sync.py`` copies
them to LangSmith. A dataset directory holds:

- ``dataset.yaml``: ``description`` and ``reference_now``, the DB build the references
  were written for (they contain absolute dates);
- one file per scenario family (e.g. ``P.yaml``) with optional ``defaults`` and a list of
  ``cases``. Each case is merged over its file's defaults before validation.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

import yaml
from langsmith.schemas import ExampleCreate
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from evals.errors import EvalSetupError

Split = Literal["smoke", "full"]
"""Neutral subsets of a dataset: ``smoke`` (one case per scenario) and ``full``."""


class CaseInputs(BaseModel):
    """What the target receives. Only ``question`` reaches the agent as a message.

    Attributes:
        customer_email: Identifies the already-verified customer the thread is seeded with.
        question: The customer's single-turn message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_email: str
    question: str


class ToolCallSpec(BaseModel):
    """One tool call the agent is expected to make in the turn.

    Attributes:
        name: Tool name, e.g. ``check_return_eligibility``. ``sync`` checks it, and the
            argument names, against the real tools.
        args: Arguments the call must include, with exactly these values. Other arguments
            may be present. Empty matches any arguments.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z_]+$")
    args: dict[str, Any] = Field(default_factory=dict)


class Reference(BaseModel):
    """The hand-written expected result, seen only by the evaluators.

    Attributes:
        expected_outcome: The correct answer in one or two sentences.
        key_facts: Facts a correct answer may state (ids, dates, amounts, day counts).
        expected_tool_calls: Tool calls the turn must contain, for the trajectory metrics.
            The answer judge never sees them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_outcome: str
    key_facts: dict[str, Any] = Field(default_factory=dict)
    expected_tool_calls: list[ToolCallSpec] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One hand-written eval case.

    Attributes:
        case_id: Stable id such as ``P4-01``; the LangSmith example id is derived from it.
        scenario_id: Scenario from ``docs/scenarios.md`` such as ``P4``.
        splits: Dataset splits the case belongs to.
        metrics: Names of the metrics that apply to this case.
        inputs: What the target receives.
        reference: What evaluators compare against.
        notes: Why the case exists, for reviewers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(pattern=r"^[A-Z]+\d+-\d{2}$")
    scenario_id: str = Field(pattern=r"^[A-Z]+\d+$")
    splits: list[Split] = Field(min_length=1)
    metrics: list[str] = Field(min_length=1)
    inputs: CaseInputs
    reference: Reference
    notes: str | None = None

    @property
    def family(self) -> str:
        """Return the scenario family, derived from the scenario id.

        Returns:
            The id without its number, e.g. ``"P"`` for ``P4`` or ``"AU"`` for ``AU3``.
        """
        return self.scenario_id.rstrip("0123456789")

    @property
    def example_id(self) -> UUID:
        """Return the LangSmith example id, stable across syncs.

        Returns:
            A UUID5 derived from ``case_id``, so re-syncing updates instead of duplicating.
        """
        return uuid5(NAMESPACE_URL, f"orderpilot-eval/{self.case_id}")

    def to_example(self) -> ExampleCreate:
        """Convert the case to a LangSmith example.

        Scenario ids and other bookkeeping go into ``metadata``, which the agent and the
        judge never see.

        Returns:
            The example to upload.
        """
        # model_validate instead of the constructor: ExampleCreate.__init__ is untyped and
        # only normalises the deprecated "outputs" alias, which is not used here.
        return ExampleCreate.model_validate(
            {
                "id": self.example_id,
                "inputs": self.inputs.model_dump(),
                "reference_outputs": self.reference.model_dump(),
                "split": list(self.splits),
                "metadata": {
                    "case_id": self.case_id,
                    "scenario_id": self.scenario_id,
                    "scenario_family": self.family,
                    "metrics": list(self.metrics),
                    "notes": self.notes,
                },
            }
        )


class _DatasetFile(BaseModel):
    """Contents of ``dataset.yaml``.

    Attributes:
        description: Human-readable description of the dataset.
        reference_now: TODAY of the DB build the references were written for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    reference_now: AwareDatetime


class _CaseFile(BaseModel):
    """Contents of one family file, before the cases are merged over the defaults.

    Attributes:
        defaults: Fields applied to every case that doesn't set them.
        cases: Raw case mappings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    defaults: dict[str, Any] = Field(default_factory=dict)
    cases: list[dict[str, Any]] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A validated dataset loaded from the repo.

    Attributes:
        name: Directory name, e.g. ``"m1"``.
        description: Human-readable description.
        reference_now: TODAY of the DB build the references were written for.
        cases: All cases, in file and then list order.
    """

    name: str
    description: str
    reference_now: datetime
    cases: tuple[EvalCase, ...]

    @property
    def langsmith_name(self) -> str:
        """Return the dataset's name in LangSmith.

        Returns:
            ``orderpilot-<name>``.
        """
        return langsmith_dataset_name(self.name)


def langsmith_dataset_name(name: str) -> str:
    """Return the LangSmith name of a repo dataset.

    Args:
        name: Repo dataset name, e.g. ``"m1"``.

    Returns:
        ``orderpilot-<name>``.
    """
    return f"orderpilot-{name}"


def check_reference_now(expected: datetime | str | None, actual: datetime, *, source: str) -> None:
    """Check that a dataset was written for the database it is about to run against.

    References contain absolute dates, so a dataset is only valid for the DB build whose
    ``meta.reference_now`` it was written for.

    Args:
        expected: The dataset's ``reference_now`` (a datetime, or ISO text from LangSmith
            metadata), or ``None`` if it has none.
        actual: ``reference_now`` of the database (``repo.reference_now()``).
        source: What ``expected`` came from, for the error message.

    Raises:
        EvalSetupError: If the two differ or ``expected`` is missing.
    """
    expected_dt = datetime.fromisoformat(expected) if isinstance(expected, str) else expected
    if expected_dt != actual:
        raise EvalSetupError(
            f"{source} was written for reference_now={expected}, but the database has "
            f"{actual.isoformat()}. Rebuild the database with that --today, or update the dataset."
        )


def dataset_dir(name: str) -> Path:
    """Return the directory of a repo dataset.

    Args:
        name: Dataset name, e.g. ``"m1"``.

    Returns:
        ``evals/datasets/<name>``.
    """
    return Path(__file__).parent / "datasets" / name


def _read_yaml(path: Path) -> Any:
    """Parse one YAML file.

    Args:
        path: File to read.

    Returns:
        The parsed content.

    Raises:
        EvalSetupError: If the file is not valid YAML.
    """
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvalSetupError(f"{path.name}: invalid YAML: {exc}") from exc


def _load_cases(path: Path) -> list[EvalCase]:
    """Load one family file and validate each case merged over the file's defaults.

    Args:
        path: The family file.

    Returns:
        The validated cases.

    Raises:
        EvalSetupError: If the file or a case doesn't match the schema; the message names
            the file and the case id.
    """
    try:
        case_file = _CaseFile.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise EvalSetupError(f"{path.name}: {exc}") from exc
    cases = []
    for raw in case_file.cases:
        try:
            cases.append(EvalCase.model_validate({**case_file.defaults, **raw}))
        except ValidationError as exc:
            case_id = raw.get("case_id", "<no case_id>")
            raise EvalSetupError(f"{path.name}: case {case_id}: {exc}") from exc
    return cases


def load_dataset(directory: Path) -> EvalDataset:
    """Load and validate a repo dataset.

    Args:
        directory: Dataset directory with ``dataset.yaml`` and the family files.

    Returns:
        The validated dataset.

    Raises:
        EvalSetupError: If ``dataset.yaml`` is missing or invalid, a case is invalid, or
            two cases share a ``case_id``.
    """
    meta_path = directory / "dataset.yaml"
    if not meta_path.is_file():
        raise EvalSetupError(f"{meta_path} not found")
    try:
        meta = _DatasetFile.model_validate(_read_yaml(meta_path))
    except ValidationError as exc:
        raise EvalSetupError(f"{meta_path.name}: {exc}") from exc

    cases: list[EvalCase] = []
    for path in sorted(directory.glob("*.yaml")):
        if path != meta_path:
            cases.extend(_load_cases(path))

    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise EvalSetupError(f"Duplicate case_id {case.case_id} in {directory.name}")
        seen.add(case.case_id)

    return EvalDataset(
        name=directory.name,
        description=meta.description,
        reference_now=meta.reference_now,
        cases=tuple(cases),
    )


__all__ = [
    "CaseInputs",
    "EvalCase",
    "EvalDataset",
    "Reference",
    "Split",
    "ToolCallSpec",
    "check_reference_now",
    "dataset_dir",
    "langsmith_dataset_name",
    "load_dataset",
]
