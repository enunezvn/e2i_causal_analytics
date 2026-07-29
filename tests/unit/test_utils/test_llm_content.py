"""Tests for normalize_llm_content (#1350 / #1358).

LangChain's AIMessage.content is str | list[str | dict]. ChatAnthropic on
current Claude models returns content-block LISTS (thinking + text blocks);
consumers that assume str crash in re.search/json.loads.
"""

from src.utils.llm_content import normalize_llm_content


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
