"""Shared tool-result evidence semantics (#1257, #1458).

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
"""

import json
from typing import Any, Dict, List


def payload_carries_evidence(payload: Any) -> bool:
    """True unless ``payload`` is an envelope positively marked failed.

    ``payload`` is a raw tool result: typically the JSON string a tool
    returned (a ToolMessage's content), but any shape is tolerated. A JSON
    string is parsed; a dict is inspected directly; anything else — including
    unparseable strings and non-dict JSON — counts as evidence per the #1257
    rule above.
    """
    parsed: Any = payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except (ValueError, TypeError):
            parsed = None
    if isinstance(parsed, dict) and parsed.get("success") is False:
        return False
    return True


def evidence_tool_count(tool_results: List[Dict[str, Any]]) -> int:
    """Count tool results that actually carry evidence (#1257).

    ``tool_results`` entries are the copilotkit-route dicts whose ``result``
    key holds the raw tool payload.
    """
    return sum(1 for tr in tool_results if payload_carries_evidence(tr.get("result")))
