"""Domain objects shared by repositories, services and tools.

These are plain dataclasses with no storage details, so any repository implementation
(SQLite, Postgres, a remote service) can return them.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class User:
    """A customer account, with the fields needed for identity verification.

    Attributes:
        user_id: Primary key of the account.
        first_name: First name on the account.
        last_name: Last name on the account.
        email: Account email (unique, case-insensitive).
        postal_code: Postal code of the account address.
    """

    user_id: int
    first_name: str
    last_name: str
    email: str
    postal_code: str


__all__ = ["User"]
