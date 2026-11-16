"""Evaluator base classes: the shared error policy, and the LLM-judge base.

LangSmith calls an evaluator with ``(run, example)`` when the parameters have exactly those
names. Every evaluator returns ``EvaluationResults`` (``{"results": [...]}``): LangSmith
rejects a bare list of ``EvaluationResult`` objects and any empty result, but accepts an
empty ``results`` list, which is how "not applicable" emits no feedback.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, Literal

from langchain_core.language_models import BaseChatModel
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run
from openevals.llm import create_llm_as_judge
from pydantic import BaseModel, Field

from orderpilot.observability import logger

EvaluatorKind = Literal["gate", "score"]
"""``gate``: any failing run fails the experiment. ``score``: averaged quality metric."""


class Evaluator(ABC):
    """Evaluator for one metric (its ``key``), called by LangSmith with ``(run, example)``.

    ``__call__`` is a template method holding the policy every metric shares:

    - a case whose ``metrics`` don't include this one gets no result (not applicable);
    - a run that errored (exception, recursion limit) fails the metric;
    - a failure inside the evaluator emits ``evaluator_error`` instead of a score, so a
      broken judge is never counted as a failing agent.

    Subclasses set ``key`` (and ``kind``) and implement :meth:`evaluate`.

    Attributes:
        key: Name of the metric: its feedback key, and the key in ``EvaluatorFactory``.
        kind: Whether the metric is a gate or a score.
    """

    key: ClassVar[str]
    kind: ClassVar[EvaluatorKind] = "score"

    def __init__(self) -> None:
        # LangSmith names evaluators (and their traces) by __name__, which objects lack.
        self.__name__ = self.key

    def __call__(self, run: Run, example: Example | None) -> EvaluationResults:
        """Evaluate one run, applying the shared applicability and error policy.

        Args:
            run: The target run, with its outputs or error.
            example: The dataset example, with reference outputs and metadata. LangSmith
                types it as optional; a run without one has nothing to be graded against.

        Returns:
            The metric's results; an empty list when the metric doesn't apply.
        """
        if example is None or self.key not in (example.metadata or {}).get("metrics", []):
            return {"results": []}
        if run.error:
            return {
                "results": [
                    EvaluationResult(key=self.key, score=False, comment=f"Agent error: {run.error}")
                ]
            }
        try:
            return {"results": self.evaluate(run, example)}
        except Exception as exc:  # noqa: BLE001 - evaluator boundary: one broken metric must not hide the others
            logger.exception("Evaluator {} failed on example {}", self.key, example.id)
            return {
                "results": [
                    EvaluationResult(
                        key="evaluator_error",
                        score=1,
                        comment=f"{type(exc).__name__}: {exc}",
                        metadata={"metric": self.key},
                    )
                ]
            }

    @abstractmethod
    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]:
        """Score one applicable run that did not error.

        Args:
            run: The target run.
            example: The dataset example.

        Returns:
            One or more results, each with an explicit ``key``.
        """


class JudgeVerdict(BaseModel):
    """Structured output of a pass/fail judge. Reasoning comes first, so the verdict follows it.

    Attributes:
        reasoning: Step-by-step check of the answer against the rubric.
        verdict: Whether the answer passes the rubric.
    """

    reasoning: str = Field(description="Step-by-step check of the answer against the rubric.")
    verdict: bool = Field(description="True if the answer passes the rubric, otherwise false.")


class LLMJudgeEvaluator(Evaluator):
    """Base for LLM-judge evaluators: a versioned prompt run by an openevals judge.

    Args:
        judge: The judge chat model (from ``build_judge``); it must support structured output.
        reference_now: TODAY of the database the agent ran on. The judge is told this date,
            so it counts days from the same date the agent and the references use.

    Attributes:
        prompt: ``str.format`` template of the judge prompt.
        prompt_version: Version of ``prompt``, recorded in results and experiment metadata.
        today: ``reference_now`` as an ISO date, the ``{today}`` placeholder of the prompt.
    """

    prompt: ClassVar[str]
    prompt_version: ClassVar[str]

    def __init__(self, judge: BaseChatModel, reference_now: datetime) -> None:
        super().__init__()
        self.today = reference_now.date().isoformat()
        self._judge = create_llm_as_judge(
            prompt=self.prompt,
            judge=judge,
            feedback_key=self.key,
            output_schema=JudgeVerdict,
        )

    def _ask(self, **variables: Any) -> JudgeVerdict:
        """Run the judge on the prompt filled with ``variables`` and today's date.

        Args:
            **variables: Values for the prompt's placeholders, other than ``today``.

        Returns:
            The judge's verdict.
        """
        return JudgeVerdict.model_validate(self._judge(today=self.today, **variables))


__all__ = ["Evaluator", "EvaluatorKind", "JudgeVerdict", "LLMJudgeEvaluator"]
