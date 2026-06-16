"""Shared DSPy LM configuration for optimization paths.

Mirrors the only pre-existing config (src/api/routes/chatbot_dspy.py:_ensure_dspy_configured)
so the feedback-loop optimizer and the chatbot use the same LM setup. Idempotent.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_default_dspy_model() -> str:
    """Resolve the DSPy/litellm model string from env, provider-aware.

    DSPy talks to providers through litellm, so the model string must carry a
    ``<provider>/<model>`` prefix. This mirrors ``src.utils.llm_factory``'s
    PROVIDER selection (openai -> ``openai/gpt-4o``, the model the rest of the app
    uses in prod where ``LLM_PROVIDER=openai``). The Anthropic model is taken from
    ``ANTHROPIC_MODEL`` independently of llm_factory's hardcoded mapping, so the
    default is a current model rather than the retired ``claude-sonnet-4-20250514``
    that 404'd the Executive AI Brief.

    Resolution order:
      1. ``DSPY_LM_MODEL`` env override — used verbatim (already prefixed).
      2. ``LLM_PROVIDER=anthropic`` -> ``anthropic/{ANTHROPIC_MODEL}``.
      3. Otherwise (default ``openai``) -> ``openai/gpt-4o`` (the standard tier).
    """
    explicit = os.getenv("DSPY_LM_MODEL")
    if explicit:
        return explicit
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        return model if "/" in model else f"anthropic/{model}"
    return "openai/gpt-4o"


def dspy_provider_api_key_present() -> bool:
    """Whether the API key for the env-configured DSPy provider is set.

    Provider-aware so an OpenAI-configured deployment is not blocked by a missing
    ``ANTHROPIC_API_KEY`` (and vice-versa).
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def ensure_dspy_configured(model: Optional[str] = None, force: bool = False) -> bool:
    """Configure a DSPy LM if one is not already set.

    Returns True if an LM is configured (now or already), False if dspy is
    unavailable or no API key is present.
    """
    try:
        import dspy
    except ImportError:
        logger.warning("dspy not installed; cannot configure LM")
        return False

    if not force and getattr(dspy.settings, "lm", None) is not None:
        return True

    if not dspy_provider_api_key_present():
        logger.warning("No API key for the configured DSPy provider; DSPy LM not configured")
        return False

    model = model or get_default_dspy_model()
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    logger.info("DSPy LM configured: %s", model)
    return True
