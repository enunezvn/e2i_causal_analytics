"""Shared helpers for page-level strategic-insight generation.

Every insight is grounded in real, caller-provided numbers. When DSPy/the LM is
unavailable (e.g. no OPENAI_API_KEY in CI) run_signature returns None and the
caller renders a deterministic factual fallback — never fabricated content.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def normalize_list(value: Any, cap: int = 5) -> list[str]:
    """DSPy list outputs may arrive as a real list or a newline/`-`-delimited str."""
    if isinstance(value, str):
        items = [ln.strip(" -•\t") for ln in value.splitlines() if ln.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(i) for i in value]
    else:
        items = []
    return [s.strip() for s in items if s and s.strip()][:cap]


def run_signature(signature_cls: Any, **inputs: Any):
    """Run a DSPy ChainOfThought over ``signature_cls``, or return ``None``.

    Returns ``None`` when dspy is unavailable, no LM is configured (no API key),
    the signature is ``None``, or the call raises — the caller then uses its
    factual fallback. BLOCKING: call from a worker thread (``asyncio.to_thread``).
    """
    if signature_cls is None:
        return None
    try:
        import dspy
    except ImportError:
        return None
    try:
        from src.optimization.dspy_lm import ensure_dspy_configured

        if not ensure_dspy_configured():
            logger.info("DSPy LM not configured (no API key); factual fallback")
            return None
        return dspy.ChainOfThought(signature_cls)(**inputs)
    except Exception as e:  # noqa: BLE001 — LLM failure must never break the request
        logger.warning("Strategic-insight LLM call failed (non-fatal): %s", e)
        return None


def cache_key(page: str, scope: str, inputs: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"insight:{page}:{scope}:{digest}"


def cache_get(key: str) -> Optional[dict]:
    try:
        from src.memory.services.factories import get_redis_client

        raw = get_redis_client().get(key)
        return json.loads(raw) if raw else None
    except Exception as e:  # noqa: BLE001 — cache is best-effort
        logger.debug("insight cache_get miss/error: %s", e)
        return None


def cache_set(key: str, value: dict, ttl_seconds: int = 3600) -> None:
    try:
        from src.memory.services.factories import get_redis_client

        get_redis_client().setex(key, ttl_seconds, json.dumps(value, default=str))
    except Exception as e:  # noqa: BLE001 — cache is best-effort
        logger.debug("insight cache_set skipped: %s", e)
