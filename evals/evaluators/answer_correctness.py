"""Answer correctness: is the final answer right, judged against the hand-written reference?"""

import json

from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

from evals.evaluators.base import LLMJudgeEvaluator
from evals.prompts import ANSWER_CORRECTNESS_V1


class AnswerCorrectnessEvaluator(LLMJudgeEvaluator):
    """Evaluator for the ``answer_correctness`` metric.

    An LLM judge compares the final answer with the reference outcome and key facts.

    The judge sees only the question, the answer and the reference. It never sees the tool
    outputs (whether claims are backed by tools is a separate groundedness metric) or the
    example metadata.
    """

    key = "answer_correctness"
    prompt = ANSWER_CORRECTNESS_V1
    prompt_version = "v1"

    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]:
        """Ask the judge whether the answer is correct.

        Args:
            run: The target run; ``outputs["answer"]`` is graded.
            example: The example; ``inputs["question"]`` and the reference are used.

        Returns:
            One pass/fail result with the judge's reasoning as its comment.
        """
        reference = example.outputs or {}
        verdict = self._ask(
            question=(example.inputs or {})["question"],
            answer=(run.outputs or {}).get("answer", ""),
            expected_outcome=reference["expected_outcome"],
            key_facts=json.dumps(reference.get("key_facts", {}), ensure_ascii=False, indent=2),
        )
        return [
            EvaluationResult(
                key=self.key,
                score=verdict.verdict,
                comment=verdict.reasoning,
                metadata={"prompt_version": self.prompt_version},
            )
        ]


__all__ = ["AnswerCorrectnessEvaluator"]
