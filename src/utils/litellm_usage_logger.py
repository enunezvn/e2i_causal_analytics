"""Global litellm usage logger (admin observability, spec 2026-07-12).

DSPy rides litellm, so ONE logger registered at API startup covers every
dspy.LM call site (dspy_lm.py, chatbot_dspy.py, cognitive_rag_dspy.py,
causal_rag.py, causal_role_classifier_loader.py) regardless of where the LM
was instantiated. litellm is imported lazily inside register so this module
stays cheap to import. Fail-open everywhere.
"""

import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)

_registered = False


def _model_and_provider(kwargs: dict, response_obj: Any) -> Tuple[str, str]:
    model = getattr(response_obj, "model", None) or kwargs.get("model") or "unknown"
    lowered = f"{kwargs.get('custom_llm_provider') or ''} {model}".lower()
    provider = "anthropic" if ("anthropic" in lowered or "claude" in lowered) else "openai"
    return str(model), provider


def _usage_tokens(response_obj: Any) -> Tuple[int, int]:
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return 0, 0

    def _get(name: str) -> int:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        return int(value or 0)

    return _get("prompt_tokens"), _get("completion_tokens")


def record_litellm_success(kwargs: dict, response_obj: Any) -> None:
    """Shared body for the sync and async success hooks. Never raises."""
    try:
        # Late imports keep module import free of recorder/attribution cost
        # and let tests monkeypatch the source modules.
        from src.services.llm_usage_recorder import LLMUsageEvent, enqueue
        from src.utils.llm_attribution import get_attribution, record_usage

        if kwargs.get("cache_hit"):
            return  # cached replay: no tokens were spent
        input_tokens, output_tokens = _usage_tokens(response_obj)
        if input_tokens == 0 and output_tokens == 0:
            return
        model, provider = _model_and_provider(kwargs, response_obj)
        record_usage(model, input_tokens, output_tokens)
        attr = get_attribution()
        enqueue(
            LLMUsageEvent(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                surface=attr.surface if attr else "other",
                component=attr.component if attr else None,
                user_id=attr.user_id if attr else None,
                session_id=attr.session_id if attr else None,
                request_id=attr.request_id if attr else None,
            )
        )
    except Exception as e:  # fail-open by contract
        logger.warning("litellm usage logging failed (non-blocking): %s", e)


def register_litellm_usage_logger() -> bool:
    """Idempotent. False when litellm is unavailable (capture disabled)."""
    global _registered
    if _registered:
        return True
    try:
        import litellm
        from litellm.integrations.custom_logger import CustomLogger
    except ImportError:
        logger.warning("litellm not installed; dspy usage capture disabled")
        return False

    class _UsageLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            record_litellm_success(kwargs, response_obj)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            record_litellm_success(kwargs, response_obj)

    litellm.callbacks.append(_UsageLogger())
    _registered = True
    logger.info("litellm usage logger registered")
    return True
