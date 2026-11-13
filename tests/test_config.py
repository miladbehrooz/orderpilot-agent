from orderpilot import config


def test_policy_constants() -> None:
    """Policy constants match the values in CLAUDE.md."""
    assert config.RETURN_WINDOW_DAYS == 30
    assert config.STUCK_DAYS == 7
    assert config.LOST_DAYS == 14
    assert config.MAX_AUTH_ATTEMPTS == 3
