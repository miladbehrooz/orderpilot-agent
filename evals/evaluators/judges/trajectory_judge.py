"""Trajectory-judge base: a reference-free LLM judge of the turn's tool calls."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run
from openevals.trajectory.llm import create_trajectory_llm_as_judge

from evals.evaluators.judges.llm_judge import JudgeVerdict, LLMJudgeEvaluator


class TrajectoryJudgeEvaluator(LLMJudgeEvaluator):
    """Base for reference-free judges of a turn's trajectory, run by openevals.

    The judge sees today's date, the agent's contract (``agent_contract`` in
    ``evals.evaluators.judges.contract``) and the whole turn: the question, every tool call
    with its arguments, every tool result and the final answer. It never sees the
    reference, the example metadata or the step count, so it grades any case, with or
    without a hand-written reference.

    Args:
        judge: The judge chat model (from ``build_judge``); it must support structured output.
        reference_now: TODAY of the database the agent ran on.
        contract: The agent's contract, the ``{contract}`` placeholder of the prompt.

    Attributes:
        contract: The agent's contract given at construction.
    """

    def __init__(self, judge: BaseChatModel, reference_now: datetime, contract: str) -> None:
        self.contract = contract
        super().__init__(judge, reference_now)

    def _make_judge(self, judge: BaseChatModel, prompt: str) -> Callable[..., Any]:
        """Build an openevals trajectory judge, which formats the turn into ``{outputs}``.

        Args:
            judge: The judge chat model.
            prompt: The ``str.format`` template of this metric's prompt version.

        Returns:
            The openevals trajectory judge.
        """
        return create_trajectory_llm_as_judge(prompt=prompt, judge=judge, feedback_key=self.key)

    def _verdict(self, raw: Any) -> JudgeVerdict:
        """Read the trajectory judge's result as a verdict.

        Args:
            raw: ``{"key", "score", "comment"}``: the score is the pass/fail verdict and the
                comment the judge's reasoning.

        Returns:
            The verdict.
        """
        return JudgeVerdict(reasoning=raw["comment"] or "", verdict=bool(raw["score"]))

    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]:
        """Ask the judge whether the turn's trajectory passes this metric's rubric.

        Args:
            run: The target run; ``outputs["messages"]`` is this turn, in OpenAI format.
            example: The example (unused: the judge needs no reference).

        Returns:
            One pass/fail result with the judge's reasoning as its comment.
        """
        messages = (run.outputs or {}).get("messages", [])
        return [self._result(self._ask(outputs=messages, contract=self.contract))]


__all__ = ["TrajectoryJudgeEvaluator"]
