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
from collections.abc import Iterator
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


def _fenced_candidates(text: str) -> Iterator[str]:
    """Yield the body of each markdown code fence, ``` ```json ``` blocks first.

    An unterminated fence yields everything after its opener (models truncate
    the closing fence when they run out of tokens).
    """
    for fence in ("```json", "```"):
        pos = 0
        while True:
            idx = text.find(fence, pos)
            if idx == -1:
                break
            start = idx + len(fence)
            end = text.find("```", start)
            body = text[start:end] if end != -1 else text[start:]
            yield body.strip()
            if end == -1:
                break
            pos = end + 3


def parse_llm_json(content: Any) -> Any:
    """Parse a JSON payload from an LLM response, tolerating markdown fences.

    Models wrap JSON in code fences even when told not to: claude-haiku-4-5
    (the prod fast tier) fenced 4/4 completions of the intent-classification
    prompt in a 2026-07-29 live capture, so a raw ``json.loads`` failed with
    ``Expecting value: line 1 column 1`` on every turn (#1333). Content-block
    lists are normalized first (#1350).

    Strategy (codex iter-1): try the whole payload as bare JSON first — so
    JSON whose string values legitimately contain ``` never hits fence
    logic — then try each ``` ```json ``` block, then each bare ``` block,
    returning the first candidate that parses. This also survives a non-JSON
    fence appearing before the payload fence, and an unterminated fence
    (the planner idiom's truncation bug).

    Raises the bare-parse ``json.JSONDecodeError`` when nothing parses, like
    ``json.loads``.
    """
    text = normalize_llm_content(content)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as bare_err:
        for candidate in _fenced_candidates(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise bare_err
