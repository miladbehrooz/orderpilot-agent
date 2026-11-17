"""Evaluator base class: the applicability and error policy every metric shares.

LangSmith calls an evaluator with ``(run, example)`` when the parameters have exactly those
names. Every evaluator returns ``EvaluationResults`` (``{"results": [...]}``): LangSmith
rejects a bare list of ``EvaluationResult`` objects and any empty result, but accepts an
empty ``results`` list, which is how "not applicable" emits no feedback.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run

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

    Subclasses set ``key`` (and ``kind``) and implement :meth:`evaluate`. Families of
    metrics have their own base next to the library they wrap: ``LLMJudgeEvaluator``
    (``llm_judge.py``) and ``TrajectoryMatchEvaluator`` (``trajectory.py``).

    Attributes:
        key: Name of the metric: its feedback key, and the key in ``EvaluatorFactory``.
        kind: Whether the metric is a gate or a score.
        required_reference: Fields of the case ``Reference`` that a case listing this
            metric must fill; ``sync`` refuses the case otherwise.
    """

    key: ClassVar[str]
    kind: ClassVar[EvaluatorKind] = "score"
    required_reference: ClassVar[tuple[str, ...]] = ()

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


__all__ = ["Evaluator", "EvaluatorKind"]
