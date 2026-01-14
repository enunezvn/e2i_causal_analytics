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
    ANTHROPIC_API_KEY: Required if using Anthropic
    OPENAI_API_KEY: Required if using OpenAI

Model Mappings:
    Fast (classification/routing):
        - Anthropic: claude-haiku-4-20250414
        - OpenAI: gpt-4o-mini

    Standard (general chat):
        - Anthropic: claude-sonnet-4-20250514
        - OpenAI: gpt-4o

    Reasoning (complex analysis):
        - Anthropic: claude-sonnet-4-20250514
        - OpenAI: gpt-4o
"""

import logging
import os
from functools import lru_cache
from typing import Literal, Optional, Union

logger = logging.getLogger(__name__)

# Type alias for LLM providers
LLMProvider = Literal["anthropic", "openai"]

# Model mappings for each tier
MODEL_MAPPINGS = {
    "anthropic": {
        "fast": "claude-haiku-4-20250414",
        "standard": "claude-sonnet-4-20250514",
        "reasoning": "claude-sonnet-4-20250514",
    },
    "openai": {
        "fast": "gpt-4o-mini",
        "standard": "gpt-4o",
        "reasoning": "gpt-4o",
    },
}


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
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
):
    """
    Get a LangChain chat LLM instance.

    Args:
        model_tier: "fast" for classification, "standard" for general, "reasoning" for complex
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0 to 1.0)
        timeout: Request timeout in seconds
        provider: Override the default provider from environment

    Returns:
        ChatAnthropic or ChatOpenAI instance

    Raises:
        ImportError: If required package is not installed
        ValueError: If API key is not configured
    """
    if provider is None:
        provider = get_llm_provider()

    model_name = MODEL_MAPPINGS[provider][model_tier]
    logger.debug(f"Creating {provider} LLM: {model_name} (tier={model_tier})")

    if provider == "openai":
        return _create_openai_llm(model_name, max_tokens, temperature, timeout)
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

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout

    return ChatAnthropic(**kwargs)


def _create_openai_llm(
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: Optional[int],
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

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if timeout is not None:
        kwargs["request_timeout"] = timeout

    return ChatOpenAI(**kwargs)


# Convenience functions for common use cases


def get_fast_llm(
    max_tokens: int = 256,
    timeout: int = 5,
    provider: Optional[LLMProvider] = None,
):
    """
    Get a fast LLM for classification and routing tasks.

    Uses claude-haiku or gpt-4o-mini depending on provider.

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
    )


def get_standard_llm(
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
):
    """
    Get a standard LLM for general chat and synthesis tasks.

    Uses claude-sonnet or gpt-4o depending on provider.

    Args:
        max_tokens: Maximum tokens in response (default: 1024)
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
    max_tokens: int = 4096,
    temperature: float = 0.3,
    timeout: int = 120,
    provider: Optional[LLMProvider] = None,
):
    """
    Get a reasoning LLM for complex analysis tasks.

    Uses claude-sonnet or gpt-4o depending on provider.

    Args:
        max_tokens: Maximum tokens in response (default: 4096)
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
