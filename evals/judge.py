"""The LLM judge, built through the project's own LLM factory."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from evals.settings import eval_settings
from orderpilot.errors import LLMConfigError
from orderpilot.llm import LLMFactory
from orderpilot.observability import logger
from orderpilot.settings import settings


def build_judge() -> BaseChatModel:
    """Build the LLM judge from ``eval_settings`` through :class:`~orderpilot.llm.LLMFactory`.

    Logs a warning when the judge is the same model as the agent, because a model tends to
    favour answers written in its own style.

    Returns:
        The judge chat model.

    Raises:
        LLMConfigError: If ``EVAL_JUDGE_MODEL`` is not set, or the provider's API key is missing.
    """
    if eval_settings.JUDGE_MODEL is None:
        raise LLMConfigError("EVAL_JUDGE_MODEL is not set; add it to .env")
    if (eval_settings.JUDGE_PROVIDER, eval_settings.JUDGE_MODEL) == (
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
    ):
        logger.warning(
            "Judge and agent use the same model ({}:{}); the judge may favour its own style",
            eval_settings.JUDGE_PROVIDER,
            eval_settings.JUDGE_MODEL,
        )
    options: dict[str, Any] = {}
    if eval_settings.JUDGE_TEMPERATURE is not None:
        options["temperature"] = eval_settings.JUDGE_TEMPERATURE
    return LLMFactory.build(
        eval_settings.JUDGE_PROVIDER, model_name=eval_settings.JUDGE_MODEL, **options
    )


__all__ = ["build_judge"]
