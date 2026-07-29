"""Normalize LangChain ``AIMessage.content`` to plain text.

LangChain types ``AIMessage.content`` as ``str | list[str | dict]``. With
``ChatAnthropic`` on current Claude models (thinking / content-block
responses) it is a LIST of blocks; call sites that assume ``str`` and feed
it to ``re.search``/``json.loads`` crash with ``TypeError`` (#1350, #1358).
Route every ``response.content`` consumer through
:func:`normalize_llm_content` before parsing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["normalize_llm_content"]


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
