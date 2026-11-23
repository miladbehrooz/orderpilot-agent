"""Answer correctness: is the final answer right, judged against the hand-written reference?"""

import json
from typing import Any

from evals.evaluators.judges.llm_judge import LLMJudgeEvaluator
from evals.evaluators.turn import Turn


class AnswerCorrectnessEvaluator(LLMJudgeEvaluator):
    """Evaluator for the ``answer_correctness`` metric.

    An LLM judge compares the final answer with the reference outcome and key facts.

    The judge sees only the question, the answer and the reference. It never sees the tool
    outputs (whether claims are backed by tools is the separate ``faithfulness`` metric) or the
    example metadata.
    """

    key = "answer_correctness"
    prompt_version = "v2"

    def _variables(self, turn: Turn) -> dict[str, Any]:
        """Return the question, the answer and the reference for the prompt.

        Args:
            turn: The turn; its ``answer`` is graded against its ``reference``.

        Returns:
            ``question``, ``answer``, ``expected_outcome`` and ``key_facts`` (as JSON).
        """
        return {
            "question": turn.question,
            "answer": turn.answer,
            "expected_outcome": turn.reference["expected_outcome"],
            "key_facts": json.dumps(
                turn.reference.get("key_facts", {}), ensure_ascii=False, indent=2
            ),
        }


__all__ = ["AnswerCorrectnessEvaluator"]
