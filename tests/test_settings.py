"""Settings: policy defaults."""

from orderpilot.settings import Settings


def test_policy_defaults() -> None:
    """Policy defaults match the values in CLAUDE.md."""
    settings = Settings()
    assert settings.RETURN_WINDOW_DAYS == 30
    assert settings.STUCK_DAYS == 7
    assert settings.LOST_DAYS == 14
    assert settings.MAX_AUTH_ATTEMPTS == 3
