"""LangChain usage-capture callback (admin observability, spec 2026-07-12).

Attached at model construction in src/utils/llm_factory.py, so every factory
consumer is covered with zero call-site edits; construction-time callbacks
fire on .invoke() AND .astream() (verified langchain-anthropic 1.3.1 /
langchain-openai 1.1.14). Fail-open: capture must never break an LLM call.
"""

import logging
from typing import Any, Tuple

from langchain_core.callbacks import BaseCallbackHandler

from src.services.llm_usage_recorder import LLMUsageEvent, enqueue
from src.utils.llm_attribution import get_attribution, record_usage

logger = logging.getLogger(__name__)


def _extract_usage(response: Any, default_model: str) -> Tuple[str, int, int]:
    """(model, input_tokens, output_tokens) across the LLMResult shapes both
    providers emit for invoke and aggregated streams. Zeros when the provider
    reported no usage — the caller then records nothing (never fabricates)."""
    model = default_model
    input_tokens = 0
    output_tokens = 0

    message = None
    try:
        message = getattr(response.generations[0][0], "message", None)
    except (IndexError, AttributeError, TypeError):
        pass

    if message is not None:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
        meta = getattr(message, "response_metadata", None) or {}
        model = meta.get("model_name") or meta.get("model") or model

    if input_tokens == 0 and output_tokens == 0:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        model = llm_output.get("model_name") or llm_output.get("model") or model

    return model, input_tokens, output_tokens


class UsageRecorderCallback(BaseCallbackHandler):
    """One instance per constructed model (carries provider + requested model
    as fallbacks when the response omits them)."""

    def __init__(self, provider: str, default_model: str) -> None:
        self._provider = provider
        self._default_model = default_model

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            model, input_tokens, output_tokens = _extract_usage(response, self._default_model)
            if input_tokens == 0 and output_tokens == 0:
                return
            record_usage(model, input_tokens, output_tokens)
            attr = get_attribution()
            enqueue(
                LLMUsageEvent(
                    provider=self._provider,
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
            logger.warning("UsageRecorderCallback failed (non-blocking): %s", e)
