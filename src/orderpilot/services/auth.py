"""Customer identity verification (scenarios AU1-AU6).

Framework-free: takes a repository and plain values, returns a result. The LangChain
tool in :mod:`orderpilot.tools.auth` is a thin wrapper around it, and a future MCP
server could reuse it unchanged.
"""

from orderpilot.repository import OrderRepository
from orderpilot.services.results import ReasonCode, VerificationOutcome, VerifyResult


def _normalize(value: str) -> str:
    """Normalize a detail for comparison.

    Args:
        value: Raw text, as typed by the customer or stored in the database.

    Returns:
        The text trimmed, with inner whitespace collapsed, case-folded.
    """
    return " ".join(value.split()).casefold()


def verify_customer(
    repo: OrderRepository,
    *,
    first_name: str,
    last_name: str,
    email: str,
    postal_code: str,
    failed_attempts: int,
    already_verified: bool,
    max_attempts: int,
) -> VerificationOutcome:
    """Check the four details against the account found by email.

    Checks run in this order: already verified, locked, missing details (does not use
    an attempt), then lookup and comparison. The failure that reaches ``max_attempts``
    returns ``LOCKED``. A failure never reveals which detail was wrong.

    Args:
        repo: Data access.
        first_name: First name given by the customer.
        last_name: Last name given by the customer.
        email: Email given by the customer.
        postal_code: Postal code given by the customer.
        failed_attempts: Failed attempts so far in this conversation.
        already_verified: Whether the conversation is already verified.
        max_attempts: Failed attempts allowed before lockout.

    Returns:
        The result for the model, the verified ``user_id`` on success, and whether the
        call counts as a failed attempt.
    """
    if already_verified:
        return VerificationOutcome(VerifyResult(reason_code=ReasonCode.ALREADY_VERIFIED))
    if failed_attempts >= max_attempts:
        return VerificationOutcome(VerifyResult(reason_code=ReasonCode.LOCKED))

    details = (first_name, last_name, email, postal_code)
    if any(not detail.strip() for detail in details):
        return VerificationOutcome(
            VerifyResult(
                reason_code=ReasonCode.MISSING_DETAILS,
                attempts_remaining=max_attempts - failed_attempts,
            )
        )

    user = repo.find_user_by_email(email.strip())
    if user is not None and all(
        _normalize(given) == _normalize(stored)
        for given, stored in (
            (first_name, user.first_name),
            (last_name, user.last_name),
            (postal_code, user.postal_code),
        )
    ):
        return VerificationOutcome(VerifyResult(reason_code=ReasonCode.VERIFIED), user.user_id)

    remaining = max(max_attempts - (failed_attempts + 1), 0)
    code = ReasonCode.LOCKED if remaining == 0 else ReasonCode.NOT_MATCHED
    return VerificationOutcome(
        VerifyResult(reason_code=code, attempts_remaining=remaining),
        counts_as_failure=True,
    )


__all__ = ["verify_customer"]
