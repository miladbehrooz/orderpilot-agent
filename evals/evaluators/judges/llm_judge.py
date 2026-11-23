"""LLM-judge base: a versioned prompt run by an openevals judge, with a pass/fail verdict."""

from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run
from openevals.llm import create_llm_as_judge
from pydantic import BaseModel, Field

from evals.evaluators.base import Evaluator
from evals.evaluators.judges.prompts import prompt_for
from evals.evaluators.turn import Turn


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

    :meth:`grade` asks the judge about one :class:`~evals.evaluators.turn.Turn`, without the
    shared applicability and error policy; calibration calls it directly. A subclass sets
    ``key`` and ``prompt_version`` and says which prompt variables a turn fills
    (:meth:`_variables`).

    Args:
        judge: The judge chat model (from ``build_judge``); it must support structured output.
        reference_now: TODAY of the database the agent ran on. The judge is told this date,
            so it counts days from the same date the agent and the references use.

    Attributes:
        prompt_version: Version of the judge prompt (see ``evals/prompts.py``), recorded in
            results and experiment metadata. The prompt text is looked up from it.
        today: ``reference_now`` as an ISO date, the ``{today}`` placeholder of the prompt.
    """

    prompt_version: ClassVar[str]

    def __init__(self, judge: BaseChatModel, reference_now: datetime) -> None:
        super().__init__()
        self.today = reference_now.date().isoformat()
        self._judge = self._make_judge(judge, prompt_for(self.key, self.prompt_version))

    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]:
        """Ask the judge whether the run passes this metric's rubric.

        Args:
            run: The target run.
            example: The dataset example.

        Returns:
            One pass/fail result with the judge's reasoning as its comment.
        """
        return [self._result(self.grade(Turn.from_langsmith(run, example)))]

    def grade(self, turn: Turn) -> JudgeVerdict:
        """Ask the judge for its verdict on one turn.

        Args:
            turn: The turn to grade.

        Returns:
            The judge's verdict.
        """
        return self._ask(**self._variables(turn))

    @abstractmethod
    def _variables(self, turn: Turn) -> dict[str, Any]:
        """Return the values of the prompt's placeholders for one turn, other than ``today``.

        Args:
            turn: The turn to grade.

        Returns:
            Placeholder name to value.
        """

    def _make_judge(self, judge: BaseChatModel, prompt: str) -> Callable[..., Any]:
        """Build the openevals judge that runs the prompt.

        The default grades a final answer and returns a :class:`JudgeVerdict`. A subclass
        that needs another openevals judge overrides this and :meth:`_verdict`.

        Args:
            judge: The judge chat model.
            prompt: The ``str.format`` template of this metric's prompt version.

        Returns:
            The openevals judge, called with the prompt's variables as keyword arguments.
        """
        return create_llm_as_judge(
            prompt=prompt, judge=judge, feedback_key=self.key, output_schema=JudgeVerdict
        )

    def _verdict(self, raw: Any) -> JudgeVerdict:
        """Read the judge's output as a verdict.

        Args:
            raw: What the judge built by :meth:`_make_judge` returned.

        Returns:
            The verdict; by default ``raw`` already has the ``JudgeVerdict`` shape.
        """
        return JudgeVerdict.model_validate(raw)

    def _ask(self, **variables: Any) -> JudgeVerdict:
        """Run the judge on the prompt filled with ``variables`` and today's date.

        Args:
            **variables: Values for the prompt's placeholders, other than ``today``.

        Returns:
            The judge's verdict.
        """
        return self._verdict(self._judge(today=self.today, **variables))

    def _result(self, verdict: JudgeVerdict) -> EvaluationResult:
        """Turn a verdict into this metric's result.

        Args:
            verdict: The judge's verdict.

        Returns:
            A pass/fail result with the reasoning as its comment and the prompt version in
            its metadata.
        """
        return EvaluationResult(
            key=self.key,
            score=verdict.verdict,
            comment=verdict.reasoning,
            metadata={"prompt_version": self.prompt_version},
        )


__all__ = ["JudgeVerdict", "LLMJudgeEvaluator"]
