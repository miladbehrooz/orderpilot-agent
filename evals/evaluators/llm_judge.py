"""LLM-judge base: a versioned prompt run by an openevals judge, with a pass/fail verdict."""

from datetime import datetime
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from openevals.llm import create_llm_as_judge
from pydantic import BaseModel, Field

from evals.evaluators.base import Evaluator
from evals.prompts import prompt_for


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
        prompt_version: Version of the judge prompt (see ``evals/prompts.py``), recorded in
            results and experiment metadata. The prompt text is looked up from it.
        today: ``reference_now`` as an ISO date, the ``{today}`` placeholder of the prompt.
    """

    prompt_version: ClassVar[str]

    def __init__(self, judge: BaseChatModel, reference_now: datetime) -> None:
        super().__init__()
        self.today = reference_now.date().isoformat()
        self._judge = create_llm_as_judge(
            prompt=prompt_for(self.key, self.prompt_version),
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


__all__ = ["JudgeVerdict", "LLMJudgeEvaluator"]
