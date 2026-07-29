"""Normalize LangChain ``AIMessage.content`` to plain text.

LangChain types ``AIMessage.content`` as ``str | list[str | dict]``. With
``ChatAnthropic`` on current Claude models (thinking / content-block
responses) it is a LIST of blocks; call sites that assume ``str`` and feed
it to ``re.search``/``json.loads`` crash with ``TypeError`` (#1350, #1358).
Route every ``response.content`` consumer through
:func:`normalize_llm_content` before parsing.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["normalize_llm_content", "parse_llm_json"]


def normalize_llm_content(content: Any) -> str:
    """Extract the text of an LLM response as a plain string.

    ``str`` passes through unchanged. For content-block lists, plain-``str``
    blocks and the ``text`` of ``type == "text"`` dict blocks are joined;
    non-text blocks (thinking, tool_use, ...) are dropped.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def parse_llm_json(content: Any) -> Any:
    """Parse a JSON payload from an LLM response, tolerating markdown fences.

    Models wrap JSON in code fences even when told not to: claude-haiku-4-5
    (the prod fast tier) fenced 4/4 completions of the intent-classification
    prompt in a 2026-07-29 live capture, so a raw ``json.loads`` failed with
    ``Expecting value: line 1 column 1`` on every turn (#1333). Content-block
    lists are normalized first (#1350), then a leading ````` ```json ````` /
    ````` ``` ````` fence (with or without prose around it) is stripped —
    the same extraction idiom as ``planner._parse_response``, minus its
    unterminated-fence truncation bug.

    Raises ``json.JSONDecodeError`` on an unparseable payload, like
    ``json.loads``.
    """
    text = normalize_llm_content(content)
    for fence in ("```json", "```"):
        idx = text.find(fence)
        if idx != -1:
            start = idx + len(fence)
            end = text.find("```", start)
            text = text[start:end] if end != -1 else text[start:]
            break
    return json.loads(text.strip())
