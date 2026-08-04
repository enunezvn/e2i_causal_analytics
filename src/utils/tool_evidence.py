"""Shared tool-result evidence semantics (#1257, #1458, #1469).

E2I tools fail closed with a ``{"success": false, ...}`` JSON envelope that
still becomes a ToolMessage. Treating mere tool EXECUTION as evidence
therefore grades all-errored turns as grounded:

- copilotkit's learning signal graded an all-errored turn 0.8 — the copilot
  surface maximum — and those rows persist as top-reward training examples
  (#1257).
- chat_bridge's "answered from live platform data" preamble fired on turns
  where every tool FAILED — a stronger trust signal for a weaker answer,
  feeding the same poisoned-reward loop (#1458).

The rule, shared by both surfaces: only results NOT positively marked
``success: false`` count as evidence. Payloads without the envelope (or
unparseable ones) ARE the evidence, not an error marker, so they count.
Extracted here (rather than imported across route modules) because
chat_bridge must stay importable without loading the CopilotKit SDK surface.

#1469 extends the rule past scalar payloads. langchain_core 1.4.0 stringifies
the ``Dict[str, Any]`` all ten E2I chatbot tools return, so today every
payload arrives as a JSON string — but the same version passes list content
(a list of strings, or of typed content blocks) through UNSTRINGIFIED, so one
tool changing its return type would hide envelopes from a dict-only check.
A list is graded per element and is an error marker only when every element
is; ``tests/unit/test_utils/test_tool_evidence.py`` pins both halves of that
reachability assumption.
"""

import json
from typing import Any, Dict, List

# The block types langchain admits into ToolMessage.content, mirrored from
# langchain_core.tools.base.TOOL_MESSAGE_BLOCK_TYPES so this module stays
# stdlib-only; the test suite pins the two sets equal. Only these are
# WRAPPERS around a payload — a structured tool result may carry its own
# unrelated ``type``, and re-grading such a dict by its children would strip
# evidence status from something never positively marked failed.
_CONTENT_BLOCK_TYPES = frozenset(
    {
        "text",
        "image_url",
        "image",
        "json",
        "search_result",
        "custom_tool_call_output",
        "document",
        "file",
    }
)

# Where those wrappers keep the payload: text blocks in ``text``, json blocks
# in ``json``, search_result blocks nest further blocks in ``content``.
_BLOCK_PAYLOAD_KEYS = ("text", "json", "content")

# Nesting is bounded in every shape a tool can actually produce; the cap only
# keeps hostile or malformed input from exhausting the stack. Past it we
# cannot establish a failure marker, so the payload counts as evidence —
# the same direction the #1257 rule takes for anything it cannot parse.
_MAX_DEPTH = 16


def payload_carries_evidence(payload: Any) -> bool:
    """True unless ``payload`` is an envelope positively marked failed.

    ``payload`` is a raw tool result: typically the JSON string a tool
    returned (a ToolMessage's content), but any shape is tolerated. A JSON
    string is parsed; a dict is inspected directly; a list is graded per
    element (#1469); anything else — including unparseable strings — counts
    as evidence per the #1257 rule above.
    """
    return _carries_evidence(payload, 0)


def _carries_evidence(payload: Any, depth: int) -> bool:
    if depth > _MAX_DEPTH:
        return True
    parsed: Any = payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except (ValueError, TypeError):
            parsed = None
    if isinstance(parsed, dict):
        return _dict_carries_evidence(parsed, depth)
    if isinstance(parsed, list):
        # A list is an error marker only when EVERY element is one: one
        # element carrying real content makes the whole payload evidence,
        # and an empty list is not POSITIVELY marked failed either.
        return not parsed or any(_carries_evidence(item, depth + 1) for item in parsed)
    return True


def _dict_carries_evidence(parsed: Dict[str, Any], depth: int) -> bool:
    if "success" in parsed:
        return parsed["success"] is not False
    # No envelope of its own — but a typed content block wraps one a level in.
    # Gated on langchain's block types so a plain result dict that happens to
    # hold a "content" or "text" key is never re-graded by its children.
    block_type = parsed.get("type")
    if isinstance(block_type, str) and block_type in _CONTENT_BLOCK_TYPES:
        return all(
            _carries_evidence(parsed[key], depth + 1)
            for key in _BLOCK_PAYLOAD_KEYS
            if key in parsed
        )
    return True


def evidence_tool_count(tool_results: List[Dict[str, Any]]) -> int:
    """Count tool results that actually carry evidence (#1257).

    ``tool_results`` entries are the copilotkit-route dicts whose ``result``
    key holds the raw tool payload.
    """
    return sum(1 for tr in tool_results if payload_carries_evidence(tr.get("result")))
