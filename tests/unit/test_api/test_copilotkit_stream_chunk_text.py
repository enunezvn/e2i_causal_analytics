"""Unit coverage for ``_stream_chunk_text`` — the shared chunk-text extractor
for the chat and synthesize streaming loops (2026-08-19 empty-delta review).

sonnet-5 with adaptive thinking streams thinking/signature blocks BEFORE text:
truthy lists whose dict blocks have no ``"text"`` key. The old inline
extraction (``block.get("text", "")`` over every block) mapped them to ``""``,
which the loops then emitted via ``copilotkit_emit_message`` — fatal to the
frontend's Zod validation (see test_copilotkit_empty_delta_guard.py).

Contract: extract ONLY text-bearing blocks (a dict block with an explicit
non-"text" type contributes nothing), preserve the OpenAI plain-string shape
and the untyped-dict shape, and both node loops must route through this one
extractor and skip empty results (source pins below — the loops are closures
inside create_e2i_chat_agent, pinned the same way
test_copilotkit_tool_stream_leak_1547.py pins its dispatcher coupling).
"""

import inspect

import pytest

from src.api.routes.copilotkit import _stream_chunk_text

pytestmark = pytest.mark.unit


class TestThinkingShapes:
    def test_live_thinking_block_yields_empty(self):
        """The exact first-chunk shape captured live (session no90vkf)."""
        assert _stream_chunk_text([{"thinking": "", "type": "thinking", "index": 0}]) == ""

    def test_thinking_block_with_content_yields_empty(self):
        """Thinking TEXT must never leak into the answer stream either."""
        assert (
            _stream_chunk_text(
                [
                    {
                        "thinking": "Let me analyse the TRx shortfall...",
                        "type": "thinking",
                        "index": 0,
                    }
                ]
            )
            == ""
        )

    def test_signature_block_yields_empty(self):
        assert _stream_chunk_text([{"signature": "EqMDCkYIChABGAIqQPT", "type": "signature"}]) == ""

    def test_redacted_thinking_block_yields_empty(self):
        assert _stream_chunk_text([{"data": "opaque", "type": "redacted_thinking"}]) == ""


class TestTextShapes:
    def test_text_block(self):
        assert _stream_chunk_text(
            [{"text": "Remibrutinib is up 12%", "type": "text", "index": 1}]
        ) == ("Remibrutinib is up 12%")

    def test_mixed_thinking_then_text_keeps_only_text(self):
        assert (
            _stream_chunk_text(
                [
                    {"thinking": "hmm", "type": "thinking", "index": 0},
                    {"text": "Answer.", "type": "text", "index": 1},
                ]
            )
            == "Answer."
        )

    def test_untyped_dict_with_text_is_kept(self):
        """OpenAI-ish block without an explicit type: text wins (back-compat
        with the old extraction for every non-thinking provider shape)."""
        assert _stream_chunk_text([{"text": "plain", "index": 0}]) == "plain"

    def test_plain_string_content_passes_through(self):
        assert _stream_chunk_text("OpenAI returns a str") == "OpenAI returns a str"

    def test_string_items_in_list_are_kept(self):
        assert _stream_chunk_text(["part1", "part2"]) == "part1part2"

    def test_empty_shapes(self):
        assert _stream_chunk_text("") == ""
        assert _stream_chunk_text([]) == ""
        assert _stream_chunk_text(None) == ""


class TestLoopCoupling:
    """The extractor only helps while BOTH streaming loops route through it.

    chat_node and synthesize_node are closures inside create_e2i_chat_agent, so
    (per this repo's established pin pattern) couple them via source
    inspection: the factory must call _stream_chunk_text at least twice and
    must no longer contain the raw inline block-join it replaces.
    """

    def _factory_source(self) -> str:
        from src.api.routes.copilotkit import create_e2i_chat_agent

        return inspect.getsource(create_e2i_chat_agent)

    def test_both_loops_use_the_extractor(self):
        src = self._factory_source()
        assert src.count("_stream_chunk_text(") >= 2, (
            "chat_node and synthesize_node must BOTH extract chunk text via _stream_chunk_text"
        )

    def test_inline_extraction_is_gone(self):
        src = self._factory_source()
        assert 'block.get("text", "")' not in src, (
            "raw inline block-join survives in the factory — thinking blocks "
            "will extract to '' outside the guarded helper"
        )

    def test_synthesize_emit_is_empty_guarded(self):
        """The synthesize loop emits per-chunk; an empty extraction must skip
        the emit (the chat loop buffers, so its guard is the append)."""
        src = self._factory_source()
        assert "if not chunk_text" in src, (
            "no empty-chunk guard in the factory loops — "
            "copilotkit_emit_message('') is reachable again"
        )
