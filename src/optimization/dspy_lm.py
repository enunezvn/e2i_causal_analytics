"""Shared DSPy LM configuration for optimization paths.

Mirrors the only pre-existing config (src/api/routes/chatbot_dspy.py:_ensure_dspy_configured)
so the feedback-loop optimizer and the chatbot use the same LM setup. Idempotent.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"


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

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set; DSPy LM not configured")
        return False

    model = model or os.getenv("DSPY_LM_MODEL", DEFAULT_MODEL)
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    logger.info("DSPy LM configured: %s", model)
    return True
