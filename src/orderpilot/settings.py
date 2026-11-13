"""Policy constants and runtime settings.

Settings are loaded from environment variables and ``.env`` by pydantic-settings. The
policy values are the single source of truth for tools, tests and evals.
"""

from pathlib import Path
from typing import Final

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration and business policy, loaded from environment variables and ``.env``.

    Attributes:
        DB_PATH: Path to the SQLite database built by ``scripts/build_db.py``. Read from
            the ``ORDERPILOT_DB_PATH`` environment variable.
        ANTHROPIC_API_KEY: API key for Anthropic models, if used.
        OPENAI_API_KEY: API key for OpenAI models, if used.
        RETURN_WINDOW_DAYS: Days after delivery during which an order can be returned.
        STUCK_DAYS: Days an order can stay in Processing before it counts as stuck.
        LOST_DAYS: Days an order can stay in Shipped before it counts as lost.
        MAX_AUTH_ATTEMPTS: Failed verification attempts allowed before lockout.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DB_PATH: Path = Field(
        default=PROJECT_ROOT / "ecommerce.sqlite",
        validation_alias="ORDERPILOT_DB_PATH",
    )

    ANTHROPIC_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None

    RETURN_WINDOW_DAYS: int = 30
    STUCK_DAYS: int = 7
    LOST_DAYS: int = 14
    MAX_AUTH_ATTEMPTS: int = 3


settings = Settings()
