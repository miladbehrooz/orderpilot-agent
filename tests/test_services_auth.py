"""Verification service on a stub repository, plus one check on the real database."""

from dataclasses import replace
from datetime import datetime
from typing import Any

import pytest

from orderpilot.domain import User
from orderpilot.repository import OrderRepository, SqliteOrderRepository
from orderpilot.services import ReasonCode, VerificationOutcome
from orderpilot.services.auth import verify_customer
from orderpilot.settings import settings
from tests.stubs import StubRepository, verification_details


def _verify(
    repo: OrderRepository,
    details: dict[str, str],
    *,
    failed_attempts: int = 0,
    already_verified: bool = False,
) -> VerificationOutcome:
    """Call the service with the configured maximum number of attempts.

    Args:
        repo: Repository to verify against.
        details: The four verification details.
        failed_attempts: Failed attempts so far.
        already_verified: Whether the conversation is already verified.

    Returns:
        The service outcome.
    """
    kwargs: dict[str, Any] = dict(details)
    return verify_customer(
        repo,
        **kwargs,
        failed_attempts=failed_attempts,
        already_verified=already_verified,
        max_attempts=settings.MAX_AUTH_ATTEMPTS,
    )


def test_correct_details_verify(
    stub_repo: StubRepository, stub_user: User, stub_details: dict[str, str]
) -> None:
    """Correct details verify and return the user id outside the result.

    Args:
        stub_repo: Stub repository.
        stub_user: The stub account.
        stub_details: Its correct details.
    """
    outcome = _verify(stub_repo, stub_details)
    assert outcome.result.reason_code is ReasonCode.VERIFIED
    assert outcome.user_id == stub_user.user_id
    assert not outcome.counts_as_failure


def test_matching_ignores_case_and_whitespace(
    stub_repo: StubRepository, stub_details: dict[str, str]
) -> None:
    """Details match regardless of case and surrounding or repeated spaces.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
    """
    details = {
        **stub_details,
        "first_name": "  aNN ",
        "last_name": "LEE",
        "email": " ANN.LEE@example.com ",
    }
    assert _verify(stub_repo, details).result.reason_code is ReasonCode.VERIFIED


@pytest.mark.parametrize("field", ["first_name", "last_name", "postal_code"])
def test_one_wrong_field_is_generic_failure(
    stub_repo: StubRepository, stub_details: dict[str, str], field: str
) -> None:
    """Any single wrong field gives the same NOT_MATCHED result.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
        field: The field given wrongly.
    """
    outcome = _verify(stub_repo, {**stub_details, field: "wrong"})
    assert outcome.result.reason_code is ReasonCode.NOT_MATCHED
    assert outcome.result.attempts_remaining == settings.MAX_AUTH_ATTEMPTS - 1
    assert outcome.counts_as_failure
    assert outcome.user_id is None
    assert field not in outcome.result.to_content()


def test_unknown_email_looks_like_any_mismatch(
    stub_repo: StubRepository, stub_details: dict[str, str]
) -> None:
    """An unknown email gives exactly the same result as a wrong name.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
    """
    unknown = _verify(stub_repo, {**stub_details, "email": "someone@else.com"})
    wrong_name = _verify(stub_repo, {**stub_details, "first_name": "Bob"})
    assert unknown.result == wrong_name.result


def test_failure_at_limit_locks(stub_repo: StubRepository, stub_details: dict[str, str]) -> None:
    """The failure that reaches the limit returns LOCKED.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
    """
    outcome = _verify(
        stub_repo,
        {**stub_details, "postal_code": "00000"},
        failed_attempts=settings.MAX_AUTH_ATTEMPTS - 1,
    )
    assert outcome.result.reason_code is ReasonCode.LOCKED
    assert outcome.result.attempts_remaining == 0
    assert outcome.counts_as_failure


def test_locked_rejects_even_correct_details(
    stub_repo: StubRepository, stub_details: dict[str, str]
) -> None:
    """Once locked, correct details are refused and no attempt is counted.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
    """
    outcome = _verify(stub_repo, stub_details, failed_attempts=settings.MAX_AUTH_ATTEMPTS)
    assert outcome.result.reason_code is ReasonCode.LOCKED
    assert outcome.user_id is None
    assert not outcome.counts_as_failure


def test_right_email_wrong_postal_code_fails(
    stub_repo: StubRepository, stub_details: dict[str, str]
) -> None:
    """A correct email with a wrong postal code fails without a hint.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
    """
    outcome = _verify(stub_repo, {**stub_details, "postal_code": "99999"})
    assert outcome.result.reason_code is ReasonCode.NOT_MATCHED
    assert "postal" not in outcome.result.to_content()


@pytest.mark.parametrize("field", ["first_name", "last_name", "email", "postal_code"])
def test_missing_detail_does_not_use_attempt(
    stub_repo: StubRepository, stub_details: dict[str, str], field: str
) -> None:
    """A blank detail returns MISSING_DETAILS and doesn't count as a failure.

    Args:
        stub_repo: Stub repository.
        stub_details: Correct details of the stub account.
        field: The field left blank.
    """
    outcome = _verify(stub_repo, {**stub_details, field: "  "}, failed_attempts=1)
    assert outcome.result.reason_code is ReasonCode.MISSING_DETAILS
    assert outcome.result.attempts_remaining == settings.MAX_AUTH_ATTEMPTS - 1
    assert not outcome.counts_as_failure


def test_already_verified_refuses_second_customer(stub_user: User, today: datetime) -> None:
    """After verification, another verification is refused with ALREADY_VERIFIED.

    Args:
        stub_user: The stub account, verified first.
        today: TODAY the stub reports.
    """
    other = replace(stub_user, user_id=7, email="bob@example.com", first_name="Bob")
    repo = StubRepository([stub_user, other], today)
    outcome = _verify(repo, verification_details(other), already_verified=True)
    assert outcome.result.reason_code is ReasonCode.ALREADY_VERIFIED
    assert outcome.user_id is None
    assert not outcome.counts_as_failure


def test_real_user_verifies(sqlite_repo: SqliteOrderRepository, known_user: User) -> None:
    """A real user's correct details verify against the real database.

    Args:
        sqlite_repo: Repository on the real database.
        known_user: A real user.
    """
    outcome = _verify(sqlite_repo, verification_details(known_user))
    assert outcome.result.reason_code is ReasonCode.VERIFIED
    assert outcome.user_id == known_user.user_id
