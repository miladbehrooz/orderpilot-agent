"""Eval-only settings.

Loaded from ``EVAL_*`` environment variables and the project's ``.env``. API keys, policy
values and ``DB_PATH`` are *not* here: evals read them from :mod:`orderpilot.settings`, so
graders and tools always share one policy.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from orderpilot.settings import PROJECT_ROOT, ProviderName


class EvalSettings(BaseSettings):
    """Eval-only configuration, loaded from ``EVAL_*`` environment variables and ``.env``.

    Attributes:
        JUDGE_PROVIDER: Provider of the LLM judge.
        JUDGE_MODEL: Judge model name. ``None`` until chosen; building the judge then fails
            with a clear error.
        JUDGE_TEMPERATURE: Sampling temperature of the judge. ``None`` sends no temperature.
        REPEATS: Runs per example (k), used for pass^k and pass@k.
        MAX_CONCURRENCY: Maximum parallel target runs in one experiment.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    JUDGE_PROVIDER: ProviderName = "openai"
    JUDGE_MODEL: str | None = None
    JUDGE_TEMPERATURE: float | None = None
    REPEATS: int = Field(default=3, ge=1)
    MAX_CONCURRENCY: int = Field(default=4, ge=1)


eval_settings = EvalSettings()

__all__ = ["EvalSettings", "eval_settings"]
