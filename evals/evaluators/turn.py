"""The unit a judge grades: one turn of the agent, independent of LangSmith types.

Experiments hand the judges a LangSmith ``(run, example)`` pair; calibration hands them
turns that never were runs (built failures, ``evals/calibration/``). Both become a
:class:`Turn`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langsmith.schemas import Example, Run


@dataclass(frozen=True, slots=True)
class Turn:
    """One turn of the agent: the question, what it did, its answer and the reference.

    Attributes:
        question: The customer's message.
        answer: The agent's final reply.
        messages: The turn in OpenAI format: the question and everything after it.
        reference: The case's reference outputs; empty for a reference-free case.
    """

    question: str
    answer: str
    messages: tuple[dict[str, Any], ...]
    reference: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_langsmith(cls, run: Run, example: Example) -> "Turn":
        """Build the turn of a target run and its example.

        Args:
            run: The target run; ``outputs`` holds ``answer`` and ``messages``.
            example: The example; ``inputs["question"]`` and the reference outputs are used.

        Returns:
            The turn.
        """
        outputs = run.outputs or {}
        return cls(
            question=(example.inputs or {})["question"],
            answer=outputs.get("answer", ""),
            messages=tuple(outputs.get("messages", [])),
            reference=example.outputs or {},
        )


__all__ = ["Turn"]
