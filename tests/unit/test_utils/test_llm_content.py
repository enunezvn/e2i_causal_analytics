"""Tests for normalize_llm_content (#1350 / #1358).

LangChain's AIMessage.content is str | list[str | dict]. ChatAnthropic on
current Claude models returns content-block LISTS (thinking + text blocks);
consumers that assume str crash in re.search/json.loads.
"""

import json

import pytest

from src.utils.llm_content import normalize_llm_content, parse_llm_json


class TestNormalizeLlmContent:
    def test_str_passthrough(self):
        assert normalize_llm_content('{"a": 1}') == '{"a": 1}'

    def test_empty_str(self):
        assert normalize_llm_content("") == ""

    def test_text_blocks_joined(self):
        content = [
            {"type": "text", "text": "```json\n"},
            {"type": "text", "text": '{"a": 1}\n```'},
        ]
        assert normalize_llm_content(content) == '```json\n{"a": 1}\n```'

    def test_thinking_blocks_dropped(self):
        content = [
            {"type": "thinking", "thinking": "let me reason about this"},
            {"type": "text", "text": '{"a": 1}'},
        ]
        assert normalize_llm_content(content) == '{"a": 1}'

    def test_tool_use_blocks_dropped(self):
        content = [
            {"type": "tool_use", "id": "t1", "name": "calc", "input": {}},
            {"type": "text", "text": "done"},
        ]
        assert normalize_llm_content(content) == "done"

    def test_plain_str_blocks_kept(self):
        assert normalize_llm_content(["part one ", "part two"]) == "part one part two"

    def test_mixed_str_and_dict_blocks(self):
        content = ["prefix ", {"type": "text", "text": "suffix"}]
        assert normalize_llm_content(content) == "prefix suffix"

    def test_empty_list(self):
        assert normalize_llm_content([]) == ""

    def test_none_returns_empty(self):
        assert normalize_llm_content(None) == ""

    def test_text_block_with_non_str_text_skipped(self):
        content = [{"type": "text", "text": None}, {"type": "text", "text": "ok"}]
        assert normalize_llm_content(content) == "ok"

    def test_non_str_non_list_stringified(self):
        assert normalize_llm_content(42) == "42"


class TestParseLlmJson:
    """parse_llm_json: fence-tolerant JSON parsing of LLM completions (#1333).

    claude-haiku-4-5 (the prod fast tier) wraps its JSON in markdown fences on
    every call despite the "Respond with ONLY a JSON object" instruction —
    live-captured 4/4 on 2026-07-29 with the exact _llm_classify prompt/params.
    A raw json.loads therefore failed with "Expecting value: line 1 column 1"
    on every /chat/stream turn, silently killing the LLM intent layer.
    """

    # Verbatim live capture (claude-haiku-4-5-20251001, temperature=0.0,
    # max_tokens=256, the production _llm_classify prompt).
    REAL_CAPTURE = (
        '```json\n{"primary_intent": "system_health", "confidence": 0.95, '
        '"requires_multi_agent": false}\n```'
    )

    def test_real_captured_fenced_completion(self):
        result = parse_llm_json(self.REAL_CAPTURE)
        assert result == {
            "primary_intent": "system_health",
            "confidence": 0.95,
            "requires_multi_agent": False,
        }

    def test_fence_without_language_tag(self):
        assert parse_llm_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_bare_json_passthrough(self):
        assert parse_llm_json('{"a": 1}') == {"a": 1}

    def test_bare_json_with_whitespace(self):
        assert parse_llm_json('  {"a": 1}\n') == {"a": 1}

    def test_prose_preamble_before_fence(self):
        text = 'Here is the classification:\n```json\n{"a": 1}\n```'
        assert parse_llm_json(text) == {"a": 1}

    def test_trailing_prose_after_fence(self):
        text = '```json\n{"a": 1}\n```\nLet me know if you need more.'
        assert parse_llm_json(text) == {"a": 1}

    def test_unterminated_fence_still_parses(self):
        assert parse_llm_json('```json\n{"a": 1}') == {"a": 1}

    def test_content_block_list_with_fenced_text(self):
        content = [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": '```json\n{"a": 1}\n```'},
        ]
        assert parse_llm_json(content) == {"a": 1}

    def test_bare_json_with_backticks_in_value(self):
        # codex iter-1 MEDIUM: fence stripping must not corrupt bare JSON
        # whose string values legitimately contain triple backticks.
        payload = '{"answer": "wrap code in ``` fences ``` like this"}'
        assert parse_llm_json(payload) == {"answer": "wrap code in ``` fences ``` like this"}

    def test_earlier_non_json_fence_skipped(self):
        # codex iter-1 MEDIUM: a non-JSON fence before the payload fence must
        # not shadow it — every fenced block is tried until one parses.
        text = 'Plan:\n```\npseudo code\n```\nResult:\n```\n{"a": 1}\n```'
        assert parse_llm_json(text) == {"a": 1}

    def test_json_fence_preferred_over_earlier_plain_fence(self):
        # ```json blocks are tried before bare ``` blocks regardless of order.
        text = '```\nnot the payload\n```\n```json\n{"a": 1}\n```'
        assert parse_llm_json(text) == {"a": 1}

    def test_non_json_raises_jsondecodeerror(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("I cannot classify this query.")

    def test_empty_content_raises_jsondecodeerror(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("")
