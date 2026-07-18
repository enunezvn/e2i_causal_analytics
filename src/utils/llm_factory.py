"""
LLM Factory for LangChain Models
================================

Provides a centralized factory for creating LangChain LLM instances.
Supports switching between Claude (Anthropic) and OpenAI models via environment variable.

Usage:
    from src.utils.llm_factory import get_chat_llm, get_fast_llm, get_reasoning_llm

    # Get default chat LLM (based on LLM_PROVIDER env var)
    llm = get_chat_llm()

    # Get fast LLM for classification/routing (haiku or gpt-4o-mini)
    fast_llm = get_fast_llm()

    # Get reasoning LLM for complex tasks (sonnet or gpt-4o)
    reasoning_llm = get_reasoning_llm()

Environment Variables:
    LLM_PROVIDER: "openai" (default) or "anthropic"
    LLM_MODEL: Optional override for the OpenAI standard/reasoning model
    ANTHROPIC_API_KEY: Required if using Anthropic
    OPENAI_API_KEY: Required if using OpenAI

Model Mappings:
    Fast (classification/routing):
        - Anthropic: claude-haiku-4-5-20251001
        - OpenAI: gpt-5.6-luna (reasoning_effort="none")

    Standard (general chat):
        - Anthropic: claude-sonnet-5
        - OpenAI: gpt-5.6-terra

    Reasoning (complex analysis):
        - Anthropic: claude-sonnet-5
        - OpenAI: gpt-5.6-terra
"""

import logging
import os
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

from src.utils.llm_usage_callback import UsageRecorderCallback

# Type alias for LLM providers
LLMProvider = Literal["anthropic", "openai"]

# Model mappings for each tier.
# NOTE: the previous Anthropic ids (claude-haiku-4-20250414 / claude-sonnet-4-
# 20250514) were deprecated and return HTTP 404 not_found_error — every chatbot
# LLM call threw, so the CopilotKit chat node fell back to canned keyword
# responses. Every id below was verified callable on the deployment's actual
# API keys (2026-07-18) before being mapped here.
MODEL_MAPPINGS = {
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",
        "standard": "claude-sonnet-5",
        "reasoning": "claude-sonnet-5",
    },
    "openai": {
        "fast": "gpt-5.6-luna",
        "standard": "gpt-5.6-terra",
        "reasoning": "gpt-5.6-terra",
    },
}

# Models that reject a non-default `temperature`. Claude Sonnet 5 / Opus 4.8
# return HTTP 400 ("`temperature` is deprecated for this model"); gpt-5.x
# tolerates it inconsistently (intermittent 401s observed on gpt-5.6-luna with
# temperature=0.3). The creators silently drop `temperature` for these so
# existing callers that pass one keep working across the model upgrade.
_TEMPERATURE_UNSUPPORTED_PREFIXES = (
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-fable-5",
    "gpt-5",
)


def _supports_temperature(model: str) -> bool:
    return not model.startswith(_TEMPERATURE_UNSUPPORTED_PREFIXES)


def get_llm_provider() -> LLMProvider:
    """
    Get the configured LLM provider from environment.

    Returns:
        LLMProvider: "anthropic" or "openai"
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider not in ("anthropic", "openai"):
        logger.warning(f"Unknown LLM_PROVIDER '{provider}', defaulting to 'openai'")
        return "openai"
    return provider  # type: ignore


def get_chat_llm(
    model_tier: Literal["fast", "standard", "reasoning"] = "standard",
    max_tokens: int = 2048,
    temperature: float = 0.3,
    timeout: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
    reasoning_effort: Optional[str] = None,
):
    """
    Get a LangChain chat LLM instance.

    Args:
        model_tier: "fast" for classification, "standard" for general, "reasoning" for complex
        max_tokens: Maximum tokens in response (reasoning/thinking tokens count
            against this budget on gpt-5.x and Claude 5-family models)
        temperature: Sampling temperature (0.0 to 1.0); dropped for models that
            reject it (Claude Sonnet 5 / Opus 4.8, gpt-5.x)
        timeout: Request timeout in seconds
        provider: Override the default provider from environment
        reasoning_effort: OpenAI gpt-5.x reasoning effort ("none"/"low"/"medium"/
            "high"); ignored for Anthropic and non-reasoning OpenAI models

    Returns:
        ChatAnthropic or ChatOpenAI instance

    Raises:
        ImportError: If required package is not installed
        ValueError: If API key is not configured
    """
    if provider is None:
        provider = get_llm_provider()

    model_name = MODEL_MAPPINGS[provider][model_tier]
    if provider == "openai" and model_tier in ("standard", "reasoning"):
        # LLM_MODEL lets the deployment pin the OpenAI workhorse model without
        # a code change (mirrors how ANTHROPIC_MODEL is consumed elsewhere).
        model_name = os.environ.get("LLM_MODEL") or model_name
    logger.debug(f"Creating {provider} LLM: {model_name} (tier={model_tier})")

    if provider == "openai":
        return _create_openai_llm(model_name, max_tokens, temperature, timeout, reasoning_effort)
    else:
        return _create_anthropic_llm(model_name, max_tokens, temperature, timeout)


def _create_anthropic_llm(
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: Optional[int],
):
    """Create a ChatAnthropic instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError(
            "langchain-anthropic is required for Anthropic LLMs. "
            "Install with: pip install langchain-anthropic"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        # Usage capture (spec 2026-07-12): construction-time callbacks fire on
        # invoke AND astream, covering every factory consumer.
        "callbacks": [UsageRecorderCallback(provider="anthropic", default_model=model)],
    }
    if _supports_temperature(model):
        kwargs["temperature"] = temperature
    if timeout is not None:
        kwargs["timeout"] = timeout

    return ChatAnthropic(**kwargs)  # type: ignore[arg-type]


def _create_openai_llm(
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: Optional[int],
    reasoning_effort: Optional[str] = None,
):
    """Create a ChatOpenAI instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is required for OpenAI LLMs. "
            "Install with: pip install langchain-openai"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        # stream_usage: OpenAI omits usage on streamed responses unless asked
        # (Anthropic streams usage by default).
        "stream_usage": True,
        "callbacks": [UsageRecorderCallback(provider="openai", default_model=model)],
    }
    if _supports_temperature(model):
        kwargs["temperature"] = temperature
    if reasoning_effort is not None and model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = reasoning_effort
    if timeout is not None:
        kwargs["request_timeout"] = timeout

    return ChatOpenAI(**kwargs)  # type: ignore[arg-type]


# Convenience functions for common use cases


def get_fast_llm(
    max_tokens: int = 256,
    timeout: int = 5,
    provider: Optional[LLMProvider] = None,
):
    """
    Get a fast LLM for classification and routing tasks.

    Uses claude-haiku or gpt-5.6-luna depending on provider.

    Args:
        max_tokens: Maximum tokens in response (default: 256)
        timeout: Request timeout in seconds (default: 5)
        provider: Override provider from environment

    Returns:
        ChatAnthropic or ChatOpenAI instance
    """
    return get_chat_llm(
        model_tier="fast",
        max_tokens=max_tokens,
        temperature=0.0,  # Deterministic for classification
        timeout=timeout,
        provider=provider,
        # Without this, gpt-5.x default reasoning can consume the entire small
        # max_tokens budget and return empty content (observed in preflight).
        reasoning_effort="none",
    )


def get_standard_llm(
    max_tokens: int = 2048,
    temperature: float = 0.3,
    timeout: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
):
    """
    Get a standard LLM for general chat and synthesis tasks.

    Uses claude-sonnet or gpt-5.6-terra depending on provider.

    Args:
        max_tokens: Maximum tokens in response (default: 2048)
        temperature: Sampling temperature (default: 0.3)
        timeout: Request timeout in seconds
        provider: Override provider from environment

    Returns:
        ChatAnthropic or ChatOpenAI instance
    """
    return get_chat_llm(
        model_tier="standard",
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        provider=provider,
    )


def get_reasoning_llm(
    max_tokens: int = 8192,
    temperature: float = 0.3,
    timeout: int = 120,
    provider: Optional[LLMProvider] = None,
):
    """
    Get a reasoning LLM for complex analysis tasks.

    Uses claude-sonnet or gpt-5.6-terra depending on provider.

    Args:
        max_tokens: Maximum tokens in response (default: 8192)
        temperature: Sampling temperature (default: 0.3)
        timeout: Request timeout in seconds (default: 120)
        provider: Override provider from environment

    Returns:
        ChatAnthropic or ChatOpenAI instance
    """
    return get_chat_llm(
        model_tier="reasoning",
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        provider=provider,
    )
