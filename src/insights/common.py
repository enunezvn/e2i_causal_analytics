"""Shared helpers for page-level strategic-insight generation.

Every insight is grounded in real, caller-provided numbers. When DSPy/the LM is
unavailable (e.g. no OPENAI_API_KEY in CI) run_signature returns None and the
caller renders a deterministic factual fallback — never fabricated content.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Optional, overload

logger = logging.getLogger(__name__)

# #1874: the insight LM can emit markdown despite the signatures' plain-prose
# instruction, and pre-fix payloads live in Redis for up to the cache TTL, so
# the routes' _finalize seam flattens every LM-generated field with these
# rules. Conservative by design: paired markers only, and single underscores
# are never touched — snake_case identifiers must survive byte-identical.
_MD_FENCE_RE = re.compile(r"^```[^\n]*\n?", re.MULTILINE)
_MD_HEADING_RE = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^([ \t]*)[-*][ \t]+", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_BOLD_UNDERSCORE_RE = re.compile(r"(?<!\w)__(?!\s)(.+?)(?<!\s)__(?!\w)")
_MD_EMPHASIS_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_MD_INLINE_CODE_RE = re.compile(r"`([^`\n]*)`")


@overload
def flatten_markdown(text: str) -> str: ...
@overload
def flatten_markdown(text: None) -> None: ...
def flatten_markdown(text: str | None) -> str | None:
    """Deterministically strip markdown syntax down to plain prose (#1874).

    Paired ``**bold**``/``__bold__``/`` `code` `` markers keep their inner
    text; fenced-code marker lines, leading heading markers, and leading
    bullet markers (normalized to ``• ``) go per line; ``1.`` numbered
    markers are KEPT (they read as prose under pre-line rendering).
    Single-asterisk emphasis is stripped only when word-boundary-sane, and
    single underscores are NEVER rewritten. Idempotent on plain text;
    None/empty pass through.
    """
    if not text:
        return text
    flat = _MD_FENCE_RE.sub("", text)
    flat = _MD_HEADING_RE.sub("", flat)
    flat = _MD_BULLET_RE.sub(r"\g<1>• ", flat)
    flat = _MD_BOLD_RE.sub(r"\1", flat)
    flat = _MD_BOLD_UNDERSCORE_RE.sub(r"\1", flat)
    flat = _MD_EMPHASIS_RE.sub(r"\1", flat)
    return _MD_INLINE_CODE_RE.sub(r"\1", flat)


def normalize_list(value: Any, cap: int = 5) -> list[str]:
    """DSPy list outputs may arrive as a real list or a newline/`-`-delimited str."""
    if isinstance(value, str):
        items = [ln.strip(" -•\t") for ln in value.splitlines() if ln.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(i) for i in value]
    else:
        items = []
    return [s.strip() for s in items if s and s.strip()][:cap]


def run_signature(signature_cls: Any, *, lm_cache: bool = True, **inputs: Any):
    """Run a DSPy ChainOfThought over ``signature_cls``, or return ``None``.

    Returns ``None`` when dspy is unavailable, no LM is configured (no API key),
    the signature is ``None``, or the call raises — the caller then uses its
    factual fallback. BLOCKING: call from a worker thread (``asyncio.to_thread``).

    ``lm_cache=False`` bypasses DSPy's LM cache for this call so repeated calls
    with identical inputs draw fresh samples. Required by callers that validate
    the output and retry: the API process is long-lived, so the default
    in-memory LM cache would replay the identical rejected completion forever.
    """
    if signature_cls is None:
        return None
    try:
        # Tag this generation's litellm calls as platform-level insights usage
        # (admin observability, spec 2026-07-12): NULL user/session, but a
        # meaningful surface/component in the Platform LLM usage table.
        from src.utils.llm_attribution import set_platform_attribution

        set_platform_attribution("insights", component=signature_cls.__name__)

        import dspy
    except ImportError:
        return None
    try:
        from src.optimization.dspy_lm import ensure_dspy_configured, get_default_dspy_model

        if not ensure_dspy_configured():
            logger.info("DSPy LM not configured (no API key); factual fallback")
            return None
        program = dspy.ChainOfThought(signature_cls)
        if lm_cache:
            return program(**inputs)
        with dspy.context(lm=dspy.LM(get_default_dspy_model(), cache=False)):
            return program(**inputs)
    except Exception as e:  # noqa: BLE001 — LLM failure must never break the request
        logger.warning("Strategic-insight LLM call failed (non-fatal): %s", e)
        return None


def cache_key(page: str, scope: str, inputs: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
    return f"insight:{page}:{scope}:{digest}"


async def cache_get(key: str) -> Optional[dict]:
    # get_redis_client() returns an async client (redis.asyncio.Redis).
    try:
        from src.memory.services.factories import get_redis_client

        raw = await get_redis_client().get(key)
        return json.loads(raw) if raw else None
    except Exception as e:  # noqa: BLE001 — cache is best-effort
        logger.debug("insight cache_get miss/error: %s", e)
        return None


async def cache_set(key: str, value: dict, ttl_seconds: int = 3600) -> None:
    try:
        from src.memory.services.factories import get_redis_client

        await get_redis_client().setex(key, ttl_seconds, json.dumps(value, default=str))
    except Exception as e:  # noqa: BLE001 — cache is best-effort
        logger.debug("insight cache_set skipped: %s", e)
