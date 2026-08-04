"""List-shaped tool payloads in the #1257/#1458 evidence rule (#1469).

``payload_carries_evidence`` graded only string and dict payloads, so a
``{"success": false}`` envelope reached through any list-shaped
``ToolMessage.content`` was silently counted as evidence — the exact
defect #1257 fixed for string payloads, re-opened by a shape change.

Reachability today is pinned by ``TestListContentReachabilityPin`` below:
langchain_core 1.4.0 stringifies the ``Dict[str, Any]`` all ten E2I chatbot
tools return, so list content cannot occur *from those tools* — but the
same langchain version passes list payloads through UNSTRINGIFIED, so one
tool changing its return type re-opens the hole. This is defense in depth.
"""

import ast
import json
from pathlib import Path

import pytest
from langchain_core.tools import tool

from src.utils.tool_evidence import evidence_tool_count, payload_carries_evidence

FAILED = {"success": False, "error": "kpi substrate unavailable"}
FAILED_JSON = json.dumps(FAILED)


class TestListContentReachabilityPin:
    """Pins the assumption that makes #1469 defense-in-depth, not a live bug.

    Both halves are asserted through langchain's PUBLIC tool-invocation API
    (no private ``_format_output``/``_stringify`` imports) so the pin tracks
    behaviour rather than internal names.
    """

    def _content(self, fn):
        t = tool(fn)
        return t.invoke({"type": "tool_call", "id": "c1", "name": t.name, "args": {}}).content

    def test_dict_returning_tool_is_stringified(self):
        """Why unreachable today: a dict return becomes a JSON *string*."""

        def dict_tool() -> dict:
            """Returns a plain dict."""
            return FAILED

        content = self._content(dict_tool)
        assert isinstance(content, str)
        assert json.loads(content) == FAILED

    def test_list_returning_tool_is_NOT_stringified(self):
        """Why the hole is one return-type change away: langchain_core 1.4.0
        passes list content through untouched (``_is_message_content_type``
        admits a list of strings or of typed content blocks)."""

        def list_tool() -> list:
            """Returns a list of strings."""
            return [FAILED_JSON]

        assert self._content(list_tool) == [FAILED_JSON]

    def test_all_e2i_chatbot_tools_still_return_dicts(self):
        """The other half of unreachability: every decorated tool in
        chatbot_tools.py annotates a Dict return. Parsed, not imported, to
        avoid pulling the route module's dependencies into a unit test.

        If this fails, a tool changed shape — list content is now REACHABLE
        and the list handling below is load-bearing, not hardening.
        """
        source = Path("src/api/routes/chatbot_tools.py").read_text()
        returns = [
            (node.name, ast.unparse(node.returns) if node.returns else None)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any("tool" in ast.dump(d) for d in node.decorator_list)
        ]
        assert len(returns) == 10, f"tool count changed: {returns}"
        assert all(r is not None and r.startswith("Dict[") for _, r in returns), returns


class TestListPayloadEvidence:
    """A list payload is an error marker only when EVERY element is one.

    Mirrors the module's existing bias: only a payload POSITIVELY marked
    failed loses evidence status; anything else counts.
    """

    def test_list_of_failed_envelope_strings_is_not_evidence(self):
        assert payload_carries_evidence([FAILED_JSON]) is False

    def test_list_with_one_real_string_is_evidence(self):
        assert payload_carries_evidence([FAILED_JSON, "TRx rows: 12,867"]) is True

    def test_list_of_bare_failed_envelope_dicts_is_not_evidence(self):
        assert payload_carries_evidence([FAILED, FAILED]) is False

    def test_json_string_encoding_a_list_of_failed_envelopes_is_not_evidence(self):
        """Reachable TODAY on langchain_core 1.4.0: a tool returning
        ``[{"success": false}]`` has no ``type`` key, so the list is not
        valid message content and is stringified whole — the parsed payload
        is a *list*, which the dict-only check waved through."""
        assert payload_carries_evidence(json.dumps([FAILED])) is False

    def test_empty_list_still_counts_as_evidence(self):
        """No element is positively marked failed, so nothing is. Pins the
        pre-#1469 behaviour against an ``all([]) is True`` regression."""
        assert payload_carries_evidence([]) is True
        assert payload_carries_evidence("[]") is True


class TestContentBlockEvidence:
    """Typed content blocks — the shape a future tool would emit."""

    def test_text_block_wrapping_failed_envelope_is_not_evidence(self):
        assert payload_carries_evidence([{"type": "text", "text": FAILED_JSON}]) is False

    def test_json_block_wrapping_failed_envelope_is_not_evidence(self):
        assert payload_carries_evidence([{"type": "json", "json": FAILED}]) is False

    def test_search_result_block_with_all_failed_content_is_not_evidence(self):
        payload = [{"type": "search_result", "content": [{"type": "text", "text": FAILED_JSON}]}]
        assert payload_carries_evidence(payload) is False

    def test_block_list_with_one_real_block_is_evidence(self):
        payload = [
            {"type": "text", "text": FAILED_JSON},
            {"type": "text", "text": "Kisqali TRx = 12,867"},
        ]
        assert payload_carries_evidence(payload) is True

    def test_non_envelope_block_is_evidence(self):
        assert payload_carries_evidence([{"type": "image", "url": "https://x/y.png"}]) is True

    def test_custom_tool_call_output_block_is_graded(self):
        """Built by the REAL ``langchain_openai.custom_tool`` decorator, which
        wraps every return as ``[{"type": "custom_tool_call_output",
        "output": ...}]`` — list content whose payload key is neither text,
        json nor content. Pins the block-type set against the payload-key
        map: admitting a block type whose key is unmapped silently waves the
        envelope through, since ``all()`` over no keys is True.
        """
        from langchain_openai import custom_tool

        @custom_tool
        def failing_custom_tool(x: str) -> str:
            """Fails closed with the E2I envelope."""
            return FAILED_JSON

        content = failing_custom_tool.invoke(
            {"type": "tool_call", "id": "c1", "name": "failing_custom_tool", "args": {"x": "q"}}
        ).content
        assert content == [{"type": "custom_tool_call_output", "output": FAILED_JSON}]
        assert payload_carries_evidence(content) is False

    def test_block_types_match_langchain(self):
        """The unwrap is keyed to the block types langchain actually admits
        into ToolMessage.content. Pinned rather than imported so the module
        stays stdlib-only; this fails loudly if upstream's set changes."""
        from langchain_core.tools.base import TOOL_MESSAGE_BLOCK_TYPES

        from src.utils.tool_evidence import _CONTENT_BLOCK_TYPES

        assert _CONTENT_BLOCK_TYPES == frozenset(TOOL_MESSAGE_BLOCK_TYPES)

    @pytest.mark.parametrize("bad_type", [["text"], {"a": 1}, 7, None])
    def test_unhashable_or_non_string_type_does_not_raise(self, bad_type):
        """``type`` comes from arbitrary tool JSON, so it need not be a
        string — a set membership test on an unhashable value would raise
        TypeError straight through evidence_tool_count's call sites."""
        assert payload_carries_evidence({"type": bad_type, "content": FAILED_JSON}) is True

    def test_result_dict_with_unrelated_type_is_not_re_graded_by_children(self):
        """A structured E2I result is not a content block: it may legitimately
        carry a ``type`` and nest a failed sub-result under ``content``. Only a
        POSITIVE failure marker on the payload ITSELF removes evidence status
        (#1257), so gating the unwrap on "type is a string" was too loose."""
        payload = {"type": "analytics_result", "content": {"success": False}}
        assert payload_carries_evidence(payload) is True
        assert payload_carries_evidence([payload]) is True

    def _nest(self, depth):
        payload: object = [{"type": "text", "text": FAILED_JSON}]
        for _ in range(depth):
            payload = [{"type": "search_result", "content": payload}]
        return payload

    def test_realistic_nesting_is_graded_not_short_circuited(self):
        """Blocks nest within blocks; the walk must reach the envelope."""
        assert payload_carries_evidence(self._nest(3)) is False

    def test_pathological_nesting_terminates_without_recursion_error(self):
        """Depth past the cap is malformed or hostile input, not a tool
        result: the walk stays bounded and falls back to counting evidence
        rather than raising."""
        assert payload_carries_evidence(self._nest(200)) is True


class TestUnchangedScalarSemantics:
    """#1257 string/dict behaviour must survive the #1469 extension."""

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (FAILED_JSON, False),
            (FAILED, False),
            ('{"success": true, "data": {"trx": 1}}', True),
            ("plain text evidence", True),
            ("", True),
            (None, True),
            (12867, True),
            # Only the `False` singleton is a failure marker: the pre-#1469
            # check was `parsed.get("success") is False`, so these falsy and
            # absent values always counted as evidence and must keep doing so.
            ({"success": None}, True),
            ({"success": 0}, True),
            ({"success": ""}, True),
            ({"success": "no"}, True),
            ({"data": {"trx": 1}}, True),
            ({}, True),
        ],
    )
    def test_scalar_payloads(self, payload, expected):
        assert payload_carries_evidence(payload) is expected


class TestEvidenceToolCount:
    def test_counts_list_shaped_results(self):
        results = [
            {"tool": "kpi", "result": '{"success": true, "data": {"trx": 1}}'},
            {"tool": "blocks_failed", "result": [{"type": "text", "text": FAILED_JSON}]},
            {"tool": "blocks_real", "result": [{"type": "text", "text": "real data"}]},
            {"tool": "list_failed", "result": [FAILED_JSON]},
        ]
        assert evidence_tool_count(results) == 2
