"""
Chat helper endpoints — conversation- and page-adaptive suggestion pills.

Endpoints:
    POST /api/chat/suggestions   AUTH — generate suggestion pills. Two modes:
                                 non-empty ``messages`` → follow-up pills from
                                 the recent transcript; empty ``messages``
                                 (opener mode, pane just opened) → opener
                                 pills grounded in ``page_context``, a compact
                                 summary of the data visible on the page.

Why this exists
---------------
The CopilotKit sidebar renders follow-up pills above the chat input. The
SDK's built-in LLM suggestion mode (``suggestions="auto"``) is unusable with
our LangGraph runtime: the engine clones the agent and forces a
``copilotkitSuggest`` tool call via ``forwardedProps.toolChoice``, which
``copilotkit.py`` ignores (it binds tools with its own ``tool_choice="auto"``)
— every exchange would burn a full orchestrator run and never yield a pill.
This endpoint is the replacement: ONE fast-tier LLM call over the last few
turns, no orchestrator, no run-log/feedback pollution.

The frontend (``E2IChatSidebar``) calls it after each completed assistant
turn and falls back to static context-aware pills on any error, so this
route fails FAST and LOUD (502) rather than degrade to invented output.

Suggestion topics are constrained to what the chatbot's bound tools
(``E2I_CHATBOT_TOOLS`` in ``chatbot_tools.py``) can actually answer: KPI
queries/calculations, causal analysis, segments, clinical context, agent
status, and document retrieval.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from src.api.dependencies.auth import require_auth
from src.services.chat_capability_catalog import (
    CapabilityCatalog,
    render_catalog_block,
    route_hint,
)
from src.utils.llm_content import normalize_llm_content
from src.utils.llm_factory import get_fast_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# Transcript bounds: enough context for good follow-ups, small enough that
# the fast-tier call stays ~1k input tokens.
MAX_TRANSCRIPT_MESSAGES = 12
MAX_MESSAGE_CHARS = 1500
MAX_PAGE_CONTEXT_CHARS = 4000
MAX_SUGGESTIONS = 4
MAX_TITLE_CHARS = 60
MAX_MESSAGE_TEXT_CHARS = 500

# The prompt is a TEMPLATE: {capability_catalog} and {route_hint} are filled per
# request by build_system_prompt() (str.replace, not str.format - the JSON
# example below has braces). Everything list-shaped comes from
# src.services.chat_capability_catalog, derived from code and data (#1638
# pattern); measured 2026-09-05, the one-sentence capability description this
# replaces let 42% of live pills ask for analyses no tool serves.
_SYSTEM_PROMPT = """You generate suggestion pills for the E2I Assistant, a pharmaceutical \
commercial-analytics chatbot (brands: Remibrutinib, Fabhalta, Kisqali). A pill is a question the \
analyst clicks; the assistant must be able to ANSWER it with the capabilities below. A pill that asks \
for an analysis, grain, axis or metric outside this catalog is a defect.

{capability_catalog}

ON-SCREEN CONTENT: page_content (when present) is ALSO shown to the assistant as on-screen context, \
so a pill MAY ask it to read, compare, rank or summarize values that appear literally in page_content \
(e.g. "Which of the on-screen SHAP features ranks highest for Fabhalta?"). A pill must NOT ask for \
anything beyond those literal values - no recomputing, validating, extending or explaining WHY an \
on-screen SHAP, optimizer, prediction, gap or CATE number is what it is (another segment, more \
features, per-territory detail, robustness, thresholds). Prefer turning an on-screen entity into a \
catalog analysis: an on-screen "sample_drop -> persistent_180d" effect becomes "What drives \
persistent_180d for Remibrutinib, and how confident are those paths?"; an on-screen SHAP feature \
specialty_hematology becomes "Which HCP specialties are most likely to increase Fabhalta \
prescriptions?"; an on-screen territory allocation becomes "What is Fabhalta's TRx by census region?".

{route_hint}

Input is JSON with the current page path, brand filter, page_content (may be empty) and the \
conversation so far (may be empty).

If the conversation is NON-EMPTY, propose exactly 4 follow-ups the analyst most likely wants next - \
deepen or branch from what was discussed, never repeat an answered question, stay with the entities \
in play (brand, KPI, outcome, segment, window).

If the conversation is EMPTY, propose exactly 4 openers grounded in page_content (name the specific \
brand / KPI / outcome / segment on screen); if page_content is empty, ground them in the page hint, \
the page path and the brand filter. Mix capability letters - the 4 pills must cover at least two \
different letters, and at most two pills may share a letter.

Rules:
- "title": at most 42 characters, imperative or noun phrase; MAY start with one emoji (📈 for a chart).
- "message": the full one-sentence question the pill sends.
- When brand_filter is set (not "All"), every "message" MUST name that brand explicitly, unless the \
pill is deliberately about another named brand or a cross-brand comparison. A pill click sends only \
the message text, and a brand-less question forces the assistant to ask which brand was meant.
- When numeric KPIs are on screen or were discussed, at least one pill asks to chart a trend or \
comparison. NEVER propose comparing, summing or ratio-ing two figures whose sources differ or are \
unstated (#1640): page_content marks each KPI "[from <tables>]" or "[source unstated]"; a trend pill \
for ONE figure is always safe, a comparison is safe only within one source.
- Before answering, check each pill against A-H and the NEVER list; replace any pill that fails.

Respond with JSON only, no prose: {"suggestions": [{"title": "...", "message": "..."}, ...]}"""


def build_system_prompt(catalog: CapabilityCatalog, page: Optional[str]) -> str:
    """Fill the prompt template with the rendered catalog and the page's route hint."""
    hint = route_hint(page)
    hint_block = (
        f"PAGE HINT (what this route shows and which catalog letters fit it): {hint}"
        if hint
        else ""
    )
    prompt = _SYSTEM_PROMPT.replace("{capability_catalog}", render_catalog_block(catalog))
    prompt = prompt.replace("{route_hint}", hint_block)
    while "\n\n\n" in prompt:
        prompt = prompt.replace("\n\n\n", "\n\n")
    return prompt


class TranscriptMessage(BaseModel):
    """One user/assistant turn from the sidebar conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)


class SuggestionsRequest(BaseModel):
    """Recent transcript plus the UI context the pills should respect.

    An EMPTY ``messages`` list selects opener mode: the pane was just opened
    and the pills should be grounded in ``page_context`` (what the analyst is
    currently looking at) rather than a conversation.
    """

    messages: List[TranscriptMessage] = Field(
        default_factory=list, max_length=MAX_TRANSCRIPT_MESSAGES
    )
    page: Optional[str] = Field(default=None, max_length=200)
    brand: Optional[str] = Field(default=None, max_length=100)
    page_context: Optional[str] = Field(default=None, max_length=MAX_PAGE_CONTEXT_CHARS)

    @field_validator("messages")
    @classmethod
    def _needs_a_user_turn(cls, v: List[TranscriptMessage]) -> List[TranscriptMessage]:
        if v and not any(m.role == "user" for m in v):
            raise ValueError("non-empty transcript must contain at least one user message")
        return v


class ChatSuggestionItem(BaseModel):
    """One pill: label shown to the user, message sent when clicked."""

    title: str
    message: str


class SuggestionsResponse(BaseModel):
    """Generated pills, most-relevant first, at most MAX_SUGGESTIONS."""

    suggestions: List[ChatSuggestionItem]


def _parse_suggestions(raw: str) -> List[ChatSuggestionItem]:
    """Parse the LLM reply into validated pills.

    Tolerates a markdown code fence and a bare top-level list; everything
    else (junk JSON, empty list, wrong item shapes) raises ValueError so the
    caller can 502 and let the frontend fall back to its static pills.
    """
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        closing = text.rfind("```")
        if first_newline != -1 and closing > first_newline:
            text = text[first_newline + 1 : closing].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"suggestion reply is not valid JSON: {exc}") from exc

    items = parsed.get("suggestions") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise ValueError("suggestion reply has no 'suggestions' list")

    suggestions: List[ChatSuggestionItem] = []
    for item in items[:MAX_SUGGESTIONS]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        message = item.get("message")
        if not isinstance(title, str) or not isinstance(message, str):
            continue
        title, message = title.strip(), message.strip()
        if not title or not message:
            continue
        suggestions.append(
            ChatSuggestionItem(
                title=title[:MAX_TITLE_CHARS],
                message=message[:MAX_MESSAGE_TEXT_CHARS],
            )
        )
    if not suggestions:
        raise ValueError("suggestion reply contained no usable pills")
    return suggestions


@router.post(
    "/suggestions",
    response_model=SuggestionsResponse,
    summary="Generate conversation- and page-adaptive chat suggestion pills",
    description=(
        "One fast-tier LLM call; returns up to four pills. Non-empty "
        "messages → follow-ups from the recent transcript; empty messages "
        "(opener mode) → openers grounded in page_context. 502 on any "
        "generation/parsing failure — the frontend falls back to its static "
        "context-aware pills."
    ),
)
async def generate_chat_suggestions(
    payload: SuggestionsRequest,
    user: Dict[str, Any] = Depends(require_auth),
) -> SuggestionsResponse:
    """Generate follow-up or opener pills for the chat sidebar."""
    context = {
        "page": payload.page or "/",
        "brand_filter": payload.brand or "",
        "page_content": payload.page_context or "",
        "conversation": [{"role": m.role, "content": m.content} for m in payload.messages],
    }
    llm = get_fast_llm(max_tokens=500, timeout=8)
    try:
        reply = await llm.ainvoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(context, ensure_ascii=False)),
            ]
        )
    except Exception as exc:
        logger.warning("chat suggestion LLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail="suggestion generation failed") from exc

    try:
        # AIMessage.content is str | list of content blocks (#1350)
        suggestions = _parse_suggestions(normalize_llm_content(reply.content))
    except ValueError as exc:
        logger.warning("chat suggestion reply unusable: %s", exc)
        raise HTTPException(
            status_code=502, detail="suggestion generation returned no usable pills"
        ) from exc

    return SuggestionsResponse(suggestions=suggestions)
