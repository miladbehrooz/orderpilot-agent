"""Policy constants and runtime settings.

Settings are loaded from environment variables and ``.env`` by pydantic-settings. The
policy values are the single source of truth for tools, tests and evals.
"""

from pathlib import Path
from typing import Final, Literal, TypeAlias

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]

# --- LLM ---------------------------------------------------------------------
ProviderName: TypeAlias = Literal["openai", "anthropic"]
"""LLM providers supported by :class:`orderpilot.llm.LLMFactory`."""


class Settings(BaseSettings):
    """Runtime configuration and business policy, loaded from environment variables and ``.env``.

    Attributes:
        DB_PATH: Path to the SQLite database built by ``scripts/build_db.py``. Read from
            the ``ORDERPILOT_DB_PATH`` environment variable.
        ANTHROPIC_API_KEY: API key for Anthropic models, if used.
        OPENAI_API_KEY: API key for OpenAI models, if used.
        LLM_PROVIDER: Provider of the agent's chat model. ``None`` until chosen; building
            the agent then fails with a clear error.
        LLM_MODEL: Model name passed to the provider, e.g. ``gpt-...`` or ``claude-...``.
        LLM_TEMPERATURE: Sampling temperature of the agent's chat model. ``None`` (the
            default) sends no temperature, which current Claude models require: they
            reject the parameter.
        AGENT_RECURSION_LIMIT: Maximum graph steps per agent run, so a looping model stops.
        RETURN_WINDOW_DAYS: Days after delivery during which an order can be returned.
        STUCK_DAYS: Days an order can stay in Processing before it counts as stuck.
        LOST_DAYS: Days an order can stay in Shipped before it counts as lost.
        MAX_AUTH_ATTEMPTS: Failed verification attempts allowed before lockout.
        LOG_LEVEL: Minimum Loguru level written to the console.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `LLM_PROVIDER=` in .env means "unset", not ""
        extra="ignore",
    )

    DB_PATH: Path = Field(
        default=PROJECT_ROOT / "ecommerce.sqlite",
        validation_alias="ORDERPILOT_DB_PATH",
    )

    ANTHROPIC_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None

    LLM_PROVIDER: ProviderName | None = None
    LLM_MODEL: str | None = None
    LLM_TEMPERATURE: float | None = None
    AGENT_RECURSION_LIMIT: int = 25

    RETURN_WINDOW_DAYS: int = 30
    STUCK_DAYS: int = 7
    LOST_DAYS: int = 14
    MAX_AUTH_ATTEMPTS: int = 3

    LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"


settings = Settings()
