"""Verification service on a stub repository, plus one check on the real database."""

from dataclasses import replace
from typing import Any

import pytest

from orderpilot.domain import User
from orderpilot.repository import OrderRepository, SqliteOrderRepository
from orderpilot.services import ReasonCode, VerificationOutcome
from orderpilot.services.auth import verify_customer
from tests.stubs import STUB_USER, StubRepository

MAX = 3


def _details(user: User = STUB_USER, /, **overrides: str) -> dict[str, str]:
    """Return the four verification details of a user, with optional overrides.

    Args:
        user: User whose details to use.
        **overrides: Details to replace, e.g. ``postal_code="00000"``.

    Returns:
        Keyword arguments for :func:`verify_customer`.
    """
    details = {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "postal_code": user.postal_code,
    }
    details.update(overrides)
    return details


def _verify(
    repo: OrderRepository,
    details: dict[str, str],
    *,
    failed_attempts: int = 0,
    already_verified: bool = False,
) -> VerificationOutcome:
    """Call the service with the test's max attempts.

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
        max_attempts=MAX,
    )


def test_correct_details_verify(stub_repo: StubRepository) -> None:
    """Correct details verify and return the user id outside the result.

    Args:
        stub_repo: Stub repository.
    """
    outcome = _verify(stub_repo, _details())
    assert outcome.result.reason_code is ReasonCode.VERIFIED
    assert outcome.user_id == STUB_USER.user_id
    assert not outcome.counts_as_failure


def test_matching_ignores_case_and_whitespace(stub_repo: StubRepository) -> None:
    """Details match regardless of case and surrounding or repeated spaces.

    Args:
        stub_repo: Stub repository.
    """
    details = _details(first_name="  aNN ", last_name="LEE", email=" ANN.LEE@example.com ")
    assert _verify(stub_repo, details).result.reason_code is ReasonCode.VERIFIED


@pytest.mark.parametrize("field", ["first_name", "last_name", "postal_code"])
def test_one_wrong_field_is_generic_failure(stub_repo: StubRepository, field: str) -> None:
    """Any single wrong field gives the same NOT_MATCHED result.

    Args:
        stub_repo: Stub repository.
        field: The field given wrongly.
    """
    outcome = _verify(stub_repo, _details(**{field: "wrong"}))
    assert outcome.result.reason_code is ReasonCode.NOT_MATCHED
    assert outcome.result.attempts_remaining == MAX - 1
    assert outcome.counts_as_failure
    assert outcome.user_id is None
    assert field not in outcome.result.to_content()


def test_unknown_email_looks_like_any_mismatch(stub_repo: StubRepository) -> None:
    """An unknown email gives exactly the same result as a wrong name.

    Args:
        stub_repo: Stub repository.
    """
    unknown = _verify(stub_repo, _details(email="someone@else.com"))
    wrong_name = _verify(stub_repo, _details(first_name="Bob"))
    assert unknown.result == wrong_name.result


def test_third_failure_locks(stub_repo: StubRepository) -> None:
    """The failure that reaches the limit returns LOCKED.

    Args:
        stub_repo: Stub repository.
    """
    outcome = _verify(stub_repo, _details(postal_code="00000"), failed_attempts=MAX - 1)
    assert outcome.result.reason_code is ReasonCode.LOCKED
    assert outcome.result.attempts_remaining == 0
    assert outcome.counts_as_failure


def test_locked_rejects_even_correct_details(stub_repo: StubRepository) -> None:
    """Once locked, correct details are refused and no attempt is counted.

    Args:
        stub_repo: Stub repository.
    """
    outcome = _verify(stub_repo, _details(), failed_attempts=MAX)
    assert outcome.result.reason_code is ReasonCode.LOCKED
    assert outcome.user_id is None
    assert not outcome.counts_as_failure


def test_right_email_wrong_postal_code_fails(stub_repo: StubRepository) -> None:
    """A correct email with a wrong postal code fails without a hint.

    Args:
        stub_repo: Stub repository.
    """
    outcome = _verify(stub_repo, _details(postal_code="99999"))
    assert outcome.result.reason_code is ReasonCode.NOT_MATCHED
    assert "postal" not in outcome.result.to_content()


@pytest.mark.parametrize("field", ["first_name", "last_name", "email", "postal_code"])
def test_missing_detail_does_not_use_attempt(stub_repo: StubRepository, field: str) -> None:
    """A blank detail returns MISSING_DETAILS and doesn't count as a failure.

    Args:
        stub_repo: Stub repository.
        field: The field left blank.
    """
    outcome = _verify(stub_repo, _details(**{field: "  "}), failed_attempts=1)
    assert outcome.result.reason_code is ReasonCode.MISSING_DETAILS
    assert outcome.result.attempts_remaining == MAX - 1
    assert not outcome.counts_as_failure


def test_already_verified_refuses_second_customer(stub_repo: StubRepository) -> None:
    """After verification, another verification is refused with ALREADY_VERIFIED.

    Args:
        stub_repo: Stub repository.
    """
    other = replace(STUB_USER, user_id=7, email="bob@example.com", first_name="Bob")
    repo = StubRepository([STUB_USER, other])
    outcome = _verify(repo, _details(other), already_verified=True)
    assert outcome.result.reason_code is ReasonCode.ALREADY_VERIFIED
    assert outcome.user_id is None
    assert not outcome.counts_as_failure


def test_real_user_verifies(sqlite_repo: SqliteOrderRepository, known_user: User) -> None:
    """A real user's correct details verify against the real database.

    Args:
        sqlite_repo: Repository on the real database.
        known_user: A real user.
    """
    outcome = _verify(sqlite_repo, _details(known_user))
    assert outcome.result.reason_code is ReasonCode.VERIFIED
    assert outcome.user_id == known_user.user_id
