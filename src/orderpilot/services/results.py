"""Structured results returned by services and shown to the model as JSON.

Every result carries a :class:`ReasonCode`. Evals grade on these codes, and the model
only explains them, so business decisions never depend on the model's wording.
"""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ReasonCode(StrEnum):
    """Machine-readable outcome codes shared by all tools."""

    VERIFIED = "VERIFIED"
    NOT_MATCHED = "NOT_MATCHED"
    MISSING_DETAILS = "MISSING_DETAILS"
    LOCKED = "LOCKED"
    ALREADY_VERIFIED = "ALREADY_VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolResult(BaseModel):
    """Base result returned to the model as JSON.

    Attributes:
        reason_code: Outcome code.
        detail: Optional extra explanation, e.g. why tool arguments were invalid.
    """

    model_config = ConfigDict(frozen=True)

    reason_code: ReasonCode
    detail: str | None = None

    def to_content(self) -> str:
        """Serialize the result as ``ToolMessage`` content.

        Returns:
            Compact JSON without unset (``None``) fields.
        """
        return self.model_dump_json(exclude_none=True)


class VerifyResult(ToolResult):
    """Outcome of an identity check. It never says which detail failed (AU2, AU4).

    Attributes:
        attempts_remaining: Verification attempts left before lockout, when relevant.
    """

    attempts_remaining: int | None = None


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """Full output of the verification service.

    Attributes:
        result: The model-facing result.
        user_id: The verified account, set only on success. Kept out of ``result`` so
            the model never sees it.
        counts_as_failure: Whether this call used up one verification attempt.
    """

    result: VerifyResult
    user_id: int | None = None
    counts_as_failure: bool = False


__all__ = ["ReasonCode", "ToolResult", "VerificationOutcome", "VerifyResult"]
