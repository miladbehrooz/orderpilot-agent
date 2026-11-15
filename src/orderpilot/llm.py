"""LLM factory: builds LangChain chat models from a registry of provider builders.

Usage::

    model = LLMFactory.build("anthropic", model_name="claude-...", temperature=0.0)

Each provider has one builder function. To add a provider, add a ``_build_<name>``
static method, register it in ``_REGISTRY`` and extend
:data:`orderpilot.settings.ProviderName`.
"""

from collections.abc import Callable
from typing import Any, ClassVar, TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from orderpilot.errors import LLMConfigError
from orderpilot.observability.logger import logger
from orderpilot.settings import ProviderName, settings

BuilderFunc: TypeAlias = Callable[..., BaseChatModel]


def _require_key(key: SecretStr | None, name: str) -> SecretStr:
    """Return an API key, or fail with a clear message if it is not configured.

    Args:
        key: The key from settings, or ``None`` if it is unset.
        name: Environment variable name, used in the error message.

    Returns:
        The configured key.

    Raises:
        LLMConfigError: If the key is missing or empty.
    """
    if key is None or not key.get_secret_value():
        raise LLMConfigError(f"{name} is not set; add it to .env")
    return key


class LLMFactory:
    """Creates chat models for the supported providers from a registry of builder functions."""

    @staticmethod
    def _build_openai(*, model_name: str, **kwargs: Any) -> ChatOpenAI:
        """Build a ``ChatOpenAI`` model.

        Args:
            model_name: OpenAI model name.
            **kwargs: Extra ``ChatOpenAI`` arguments (temperature, timeout, ...).

        Returns:
            The configured chat model.

        Raises:
            LLMConfigError: If ``OPENAI_API_KEY`` is not set.
        """
        api_key = _require_key(settings.OPENAI_API_KEY, "OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)

    @staticmethod
    def _build_anthropic(*, model_name: str, **kwargs: Any) -> ChatAnthropic:
        """Build a ``ChatAnthropic`` model.

        Args:
            model_name: Anthropic model name.
            **kwargs: Extra ``ChatAnthropic`` arguments (temperature, timeout, ...).

        Returns:
            The configured chat model.

        Raises:
            LLMConfigError: If ``ANTHROPIC_API_KEY`` is not set.
        """
        api_key = _require_key(settings.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY")
        # `model_name` is the pydantic alias of `model`; mypy only accepts the alias here.
        return ChatAnthropic(model_name=model_name, api_key=api_key, **kwargs)

    _REGISTRY: ClassVar[dict[ProviderName, BuilderFunc]] = {
        "openai": _build_openai,
        "anthropic": _build_anthropic,
    }

    @classmethod
    def build(cls, provider: ProviderName, *, model_name: str, **kwargs: Any) -> BaseChatModel:
        """Build a chat model for a provider.

        Args:
            provider: Registered provider name.
            model_name: The provider's model name.
            **kwargs: Passed through to the provider's chat model class (temperature,
                max_tokens, timeout, max_retries, rate_limiter, ...).

        Returns:
            The configured LangChain chat model.

        Raises:
            LLMConfigError: If the provider is unknown or its API key is missing.
        """
        builder = cls._REGISTRY.get(provider)
        if builder is None:
            valid = ", ".join(cls._REGISTRY)
            raise LLMConfigError(f"Unknown LLM provider {provider!r}. Valid: {valid}")
        logger.debug("Building chat model {}:{}", provider, model_name)
        return builder(model_name=model_name, **kwargs)

    @classmethod
    def list_providers(cls) -> list[ProviderName]:
        """Return the registered provider names.

        Returns:
            Provider names, in registration order.
        """
        return list(cls._REGISTRY)


__all__ = ["BuilderFunc", "LLMFactory"]
