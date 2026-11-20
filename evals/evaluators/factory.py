"""Evaluator factory: builds the evaluator of each metric from a registry, like ``LLMFactory``.

To add a metric, write an :class:`~evals.evaluators.base.Evaluator` subclass whose ``key`` is
the metric's name, and register it in ``_REGISTRY``.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

from evals.errors import EvalSetupError
from evals.evaluators.answer_correctness import AnswerCorrectnessEvaluator
from evals.evaluators.argument_correctness import ArgumentCorrectnessEvaluator
from evals.evaluators.base import Evaluator
from evals.evaluators.llm_judge import LLMJudgeEvaluator
from evals.evaluators.reason_code import ReasonCodeEvaluator
from evals.evaluators.step_count import StepCountEvaluator
from evals.evaluators.tool_correctness import ToolCorrectnessEvaluator
from evals.judge import build_judge


class EvaluatorFactory:
    """Creates the evaluators of the given metrics; the judge is built only if one needs it."""

    _REGISTRY: ClassVar[dict[str, type[Evaluator]]] = {
        AnswerCorrectnessEvaluator.key: AnswerCorrectnessEvaluator,
        ToolCorrectnessEvaluator.key: ToolCorrectnessEvaluator,
        ArgumentCorrectnessEvaluator.key: ArgumentCorrectnessEvaluator,
        ReasonCodeEvaluator.key: ReasonCodeEvaluator,
        StepCountEvaluator.key: StepCountEvaluator,
    }

    @classmethod
    def metric_names(cls) -> list[str]:
        """Return the metrics that have a registered evaluator.

        Returns:
            Metric names in registration order.
        """
        return list(cls._REGISTRY)

    @classmethod
    def required_reference(cls, name: str) -> tuple[str, ...]:
        """Return the reference fields a case listing this metric must fill.

        Args:
            name: Metric name.

        Returns:
            Field names of the case ``Reference``.

        Raises:
            EvalSetupError: If no evaluator is registered for ``name``.
        """
        return cls._lookup(name).required_reference

    @classmethod
    def _lookup(cls, name: str) -> type[Evaluator]:
        """Return the evaluator class of a metric.

        Args:
            name: Metric name.

        Returns:
            The evaluator class.

        Raises:
            EvalSetupError: If no evaluator is registered for ``name``.
        """
        evaluator_cls = cls._REGISTRY.get(name)
        if evaluator_cls is None:
            valid = ", ".join(cls._REGISTRY)
            raise EvalSetupError(f"Unknown metric {name!r}. Valid: {valid}")
        return evaluator_cls

    @classmethod
    def build(cls, metrics: Sequence[str], reference_now: datetime) -> list[Evaluator]:
        """Build the evaluator of each given metric.

        One judge model is built and shared, and only when a selected metric is an LLM
        judge, so code-only runs need no judge configuration or API key.

        Args:
            metrics: Metric names, in the order the evaluators should run.
            reference_now: TODAY of the database under test, given to the LLM judges.

        Returns:
            One evaluator per metric.

        Raises:
            EvalSetupError: If a metric is unknown.
            LLMConfigError: If a judge metric is selected but the judge isn't configured.
        """
        classes = [cls._lookup(name) for name in metrics]
        needs_judge = any(issubclass(c, LLMJudgeEvaluator) for c in classes)
        judge = build_judge() if needs_judge else None
        evaluators: list[Evaluator] = []
        for evaluator_cls in classes:
            if issubclass(evaluator_cls, LLMJudgeEvaluator) and judge is not None:
                evaluators.append(evaluator_cls(judge, reference_now))
            else:
                evaluators.append(evaluator_cls())
        return evaluators

    @classmethod
    def prompt_versions(cls, metrics: Sequence[str]) -> dict[str, str]:
        """Return the judge prompt version of each selected judge metric.

        Args:
            metrics: Metric names.

        Returns:
            ``{metric: prompt_version}`` for the LLM-judge metrics among ``metrics``.

        Raises:
            EvalSetupError: If a metric is unknown.
        """
        versions: dict[str, str] = {}
        for name in metrics:
            evaluator_cls = cls._lookup(name)
            if issubclass(evaluator_cls, LLMJudgeEvaluator):
                versions[name] = evaluator_cls.prompt_version
        return versions


__all__ = ["EvaluatorFactory"]
