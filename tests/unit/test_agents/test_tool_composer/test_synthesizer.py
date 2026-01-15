"""
Tests for Tool Composer Phase 4: Synthesizer

Tests the ResponseSynthesizer class which combines tool outputs
into coherent natural language responses.
"""

import json
from datetime import datetime, timezone

import pytest

from src.agents.tool_composer.models.composition_models import (
    ComposedResponse,
    ExecutionStatus,
    ExecutionTrace,
    StepResult,
    SynthesisInput,
    ToolInput,
    ToolOutput,
)
from src.agents.tool_composer.synthesizer import (
    ResponseSynthesizer,
    synthesize_results,
    synthesize_sync,
)


class TestResponseSynthesizerInit:
    """Tests for ResponseSynthesizer initialization"""

    def test_default_initialization(self, mock_llm_client):
        """Test default initialization"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        assert synthesizer.model == "claude-sonnet-4-20250514"
        assert synthesizer.temperature == 0.4
        assert synthesizer.max_tokens == 2000

    def test_custom_initialization(self, mock_llm_client):
        """Test custom initialization"""
        synthesizer = ResponseSynthesizer(
            llm_client=mock_llm_client,
            model="claude-3-5-haiku-latest",
            temperature=0.7,
            max_tokens=3000,
        )
        assert synthesizer.model == "claude-3-5-haiku-latest"
        assert synthesizer.temperature == 0.7
        assert synthesizer.max_tokens == 3000


class TestBasicSynthesis:
    """Tests for basic synthesis functionality"""

    @pytest.mark.asyncio
    async def test_successful_synthesis(self, mock_llm_client, sample_synthesis_input):
        """Test successful synthesis with all successful results"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(sample_synthesis_input)

        assert isinstance(response, ComposedResponse)
        assert response.answer != ""
        assert 0 <= response.confidence <= 1

    @pytest.mark.asyncio
    async def test_synthesis_with_citations(self, mock_llm_client, sample_synthesis_input):
        """Test that synthesis includes citations"""
        mock_llm_client.set_synthesis_response(
            json.dumps(
                {
                    "answer": "Test answer",
                    "confidence": 0.85,
                    "citations": ["step_1", "step_2"],
                    "caveats": [],
                    "reasoning": "Test reasoning",
                }
            )
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(sample_synthesis_input)

        assert len(response.citations) == 2
        assert "step_1" in response.citations

    @pytest.mark.asyncio
    async def test_synthesis_with_caveats(self, mock_llm_client, sample_synthesis_input):
        """Test that synthesis includes caveats"""
        mock_llm_client.set_synthesis_response(
            json.dumps(
                {
                    "answer": "Test answer",
                    "confidence": 0.7,
                    "caveats": ["Observational data only", "Limited sample size"],
                    "reasoning": "Test reasoning",
                }
            )
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(sample_synthesis_input)

        assert len(response.caveats) == 2
        assert "Observational data only" in response.caveats

    @pytest.mark.asyncio
    async def test_synthesis_with_supporting_data(self, mock_llm_client, sample_synthesis_input):
        """Test that synthesis includes supporting data"""
        mock_llm_client.set_synthesis_response(
            json.dumps(
                {
                    "answer": "Test answer",
                    "confidence": 0.85,
                    "supporting_data": {"effect_size": 0.15, "p_value": 0.02, "sample_size": 1000},
                    "reasoning": "Test reasoning",
                }
            )
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(sample_synthesis_input)

        assert "effect_size" in response.supporting_data
        assert response.supporting_data["effect_size"] == 0.15


class TestMixedResults:
    """Tests for synthesis with mixed (success/failure) results"""

    @pytest.fixture
    def synthesis_input_with_failures(self, sample_decomposition):
        """Create synthesis input with some failed results"""
        step_results = [
            StepResult(
                step_id="step_1",
                sub_question_id="sq_1",
                tool_name="causal_effect_estimator",
                input=ToolInput(tool_name="causal_effect_estimator", parameters={}),
                output=ToolOutput(
                    tool_name="causal_effect_estimator", success=True, result={"effect": 0.15}
                ),
                status=ExecutionStatus.COMPLETED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            ),
            StepResult(
                step_id="step_2",
                sub_question_id="sq_2",
                tool_name="cate_analyzer",
                input=ToolInput(tool_name="cate_analyzer", parameters={}),
                output=ToolOutput(
                    tool_name="cate_analyzer", success=False, error="Model training failed"
                ),
                status=ExecutionStatus.FAILED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            ),
        ]

        trace = ExecutionTrace(plan_id="plan_123")
        for result in step_results:
            trace.add_result(result)

        return SynthesisInput(
            original_query="Test query", decomposition=sample_decomposition, execution_trace=trace
        )

    @pytest.mark.asyncio
    async def test_synthesis_acknowledges_failures(
        self, mock_llm_client, synthesis_input_with_failures
    ):
        """Test that synthesis acknowledges failed components"""
        mock_llm_client.set_synthesis_response(
            json.dumps(
                {
                    "answer": "Partial answer based on available data",
                    "confidence": 0.6,
                    "failed_components": ["sq_2"],
                    "caveats": ["CATE analysis could not be completed"],
                    "reasoning": "One component failed",
                }
            )
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(synthesis_input_with_failures)

        assert response.confidence < 0.8  # Lower confidence due to failures
        assert len(response.failed_components) >= 1 or len(response.caveats) >= 1


class TestResultFormatting:
    """Tests for _format_results method"""

    def test_format_successful_results(self, mock_llm_client, sample_synthesis_input):
        """Test formatting successful results"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        formatted = synthesizer._format_results(sample_synthesis_input)

        assert "Sub-Question:" in formatted
        assert "SUCCESS" in formatted
        assert "causal_effect_estimator" in formatted

    def test_format_failed_results(self, mock_llm_client, sample_decomposition):
        """Test formatting failed results"""
        step_result = StepResult(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="test_tool",
            input=ToolInput(tool_name="test_tool", parameters={}),
            output=ToolOutput(tool_name="test_tool", success=False, error="Tool execution failed"),
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        trace = ExecutionTrace(plan_id="plan_123")
        trace.add_result(step_result)

        synthesis_input = SynthesisInput(
            original_query="Test", decomposition=sample_decomposition, execution_trace=trace
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        formatted = synthesizer._format_results(synthesis_input)

        assert "FAILED" in formatted
        assert "Tool execution failed" in formatted

    def test_format_truncates_long_output(self, mock_llm_client, sample_decomposition):
        """Test that long outputs are truncated"""
        # Create a result with very long output
        long_result = {"data": "x" * 2000}

        step_result = StepResult(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="test_tool",
            input=ToolInput(tool_name="test_tool", parameters={}),
            output=ToolOutput(tool_name="test_tool", success=True, result=long_result),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        trace = ExecutionTrace(plan_id="plan_123")
        trace.add_result(step_result)

        synthesis_input = SynthesisInput(
            original_query="Test", decomposition=sample_decomposition, execution_trace=trace
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        formatted = synthesizer._format_results(synthesis_input)

        assert "(truncated)" in formatted


class TestResponseParsing:
    """Tests for _parse_response method"""

    def test_parse_plain_json(self, mock_llm_client):
        """Test parsing plain JSON response"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = json.dumps(
            {"answer": "Test answer", "confidence": 0.85, "reasoning": "Test reasoning"}
        )

        parsed = synthesizer._parse_response(response)

        assert parsed["answer"] == "Test answer"
        assert parsed["confidence"] == 0.85

    def test_parse_json_code_block(self, mock_llm_client):
        """Test parsing JSON from markdown code block"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = """Here is the synthesized response:

```json
{
    "answer": "Test answer from code block",
    "confidence": 0.9,
    "reasoning": "Parsed from markdown"
}
```"""

        parsed = synthesizer._parse_response(response)

        assert parsed["answer"] == "Test answer from code block"
        assert parsed["confidence"] == 0.9

    def test_parse_generic_code_block(self, mock_llm_client):
        """Test parsing from generic code block (without json tag)"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = """Response:

```
{
    "answer": "Generic block answer",
    "confidence": 0.75
}
```"""

        parsed = synthesizer._parse_response(response)

        assert parsed["answer"] == "Generic block answer"

    def test_parse_invalid_json_returns_raw(self, mock_llm_client):
        """Test that invalid JSON returns raw response as answer"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = "This is not valid JSON at all"

        parsed = synthesizer._parse_response(response)

        assert parsed["answer"] == response
        assert parsed["confidence"] == 0.6  # Lower confidence for fallback


class TestFallbackResponse:
    """Tests for fallback response creation"""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self, mock_llm_client, sample_synthesis_input):
        """Test fallback response when LLM fails"""
        # Use the LangChain interface to inject an error
        mock_llm_client.set_error(Exception("LLM connection error"))

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(sample_synthesis_input)

        assert isinstance(response, ComposedResponse)
        assert response.confidence <= 0.5  # Low confidence for fallback
        assert "error" in response.synthesis_reasoning.lower() or len(response.caveats) > 0

    def test_fallback_extracts_successful_results(self, mock_llm_client, sample_synthesis_input):
        """Test that fallback extracts key values from successful results"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        fallback = synthesizer._create_fallback_response(sample_synthesis_input, "Test error")

        assert isinstance(fallback, ComposedResponse)
        assert fallback.confidence <= 0.5

    def test_fallback_with_all_failures(self, mock_llm_client, sample_decomposition):
        """Test fallback when all results failed"""
        step_result = StepResult(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="test_tool",
            input=ToolInput(tool_name="test_tool", parameters={}),
            output=ToolOutput(tool_name="test_tool", success=False, error="Failed"),
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        trace = ExecutionTrace(plan_id="plan_123")
        trace.add_result(step_result)

        synthesis_input = SynthesisInput(
            original_query="Test", decomposition=sample_decomposition, execution_trace=trace
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        fallback = synthesizer._create_fallback_response(synthesis_input, "All tools failed")

        assert "Unable to" in fallback.answer or "error" in fallback.answer.lower()
        assert "sq_1" in fallback.failed_components


class TestLLMInteraction:
    """Tests for LLM client interaction"""

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_params(self, mock_llm_client, sample_synthesis_input):
        """Test that synthesizer is initialized with correct parameters and calls LLM"""
        synthesizer = ResponseSynthesizer(
            llm_client=mock_llm_client, model="test-model", temperature=0.5, max_tokens=1500
        )
        await synthesizer.synthesize(sample_synthesis_input)

        # Verify LLM was called
        assert mock_llm_client.call_count == 1
        # Verify synthesizer stored the configuration correctly
        # (Note: LangChain's ainvoke doesn't pass model/temp per-call,
        # they're configured on the client. We verify they're stored on synthesizer.)
        assert synthesizer.model == "test-model"
        assert synthesizer.temperature == 0.5
        assert synthesizer.max_tokens == 1500

    @pytest.mark.asyncio
    async def test_query_included_in_message(self, mock_llm_client, sample_synthesis_input):
        """Test that original query is included in message"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        await synthesizer.synthesize(sample_synthesis_input)

        call = mock_llm_client.call_history[0]
        # With LangChain interface, user content is stored directly
        user_content = call["user"]
        assert sample_synthesis_input.original_query in user_content

    @pytest.mark.asyncio
    async def test_system_prompt_used(self, mock_llm_client, sample_synthesis_input):
        """Test that system prompt is included"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        await synthesizer.synthesize(sample_synthesis_input)

        call = mock_llm_client.call_history[0]
        assert call["system"] != ""
        assert "synthesizer" in call["system"].lower()


class TestConvenienceFunction:
    """Tests for synthesize_results convenience function"""

    @pytest.mark.asyncio
    async def test_synthesize_results(
        self, mock_llm_client, sample_decomposition, sample_execution_trace
    ):
        """Test convenience function"""
        response = await synthesize_results(
            query="Test query",
            decomposition=sample_decomposition,
            execution_trace=sample_execution_trace,
            llm_client=mock_llm_client,
        )

        assert isinstance(response, ComposedResponse)

    @pytest.mark.asyncio
    async def test_synthesize_results_with_custom_params(
        self, mock_llm_client, sample_decomposition, sample_execution_trace
    ):
        """Test convenience function with custom parameters"""
        response = await synthesize_results(
            query="Test query",
            decomposition=sample_decomposition,
            execution_trace=sample_execution_trace,
            llm_client=mock_llm_client,
            model="custom-model",
            temperature=0.3,
        )

        assert isinstance(response, ComposedResponse)
        # Verify LLM was called (custom params are stored on synthesizer, not passed to ainvoke)
        assert mock_llm_client.call_count >= 1


@pytest.mark.xdist_group(name="sync_wrappers")
class TestSyncWrapper:
    """Tests for synchronous wrapper function"""

    def test_synthesize_sync(self, mock_llm_client, sample_synthesis_input):
        """Test synchronous synthesis wrapper"""
        response = synthesize_sync(sample_synthesis_input, mock_llm_client)

        assert isinstance(response, ComposedResponse)

    def test_synthesize_sync_with_custom_params(self, mock_llm_client, sample_synthesis_input):
        """Test sync wrapper with custom parameters"""
        response = synthesize_sync(
            sample_synthesis_input, mock_llm_client, model="custom-model", temperature=0.6
        )

        assert isinstance(response, ComposedResponse)
        # Verify LLM was called (custom params are stored on synthesizer, not passed to ainvoke)
        assert mock_llm_client.call_count >= 1


class TestTimestampHandling:
    """Tests for timestamp handling in synthesis"""

    @pytest.mark.asyncio
    async def test_response_has_timestamp(self, mock_llm_client, sample_synthesis_input):
        """Test that response includes timestamp"""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(sample_synthesis_input)

        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_timestamp_is_recent(self, mock_llm_client, sample_synthesis_input):
        """Test that timestamp is recent"""
        before = datetime.now(timezone.utc)
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(sample_synthesis_input)
        after = datetime.now(timezone.utc)

        assert before <= response.timestamp <= after


class TestEdgeCases:
    """Tests for edge cases"""

    @pytest.mark.asyncio
    async def test_empty_execution_trace(self, mock_llm_client, sample_decomposition):
        """Test synthesis with empty execution trace"""
        empty_trace = ExecutionTrace(plan_id="plan_123")

        synthesis_input = SynthesisInput(
            original_query="Test", decomposition=sample_decomposition, execution_trace=empty_trace
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(synthesis_input)

        # Should still produce some response
        assert isinstance(response, ComposedResponse)

    @pytest.mark.asyncio
    async def test_very_long_query(self, mock_llm_client, sample_synthesis_input):
        """Test synthesis with very long query"""
        # Modify the synthesis input to have a long query
        long_query = "What is the effect? " * 100

        long_input = SynthesisInput(
            original_query=long_query,
            decomposition=sample_synthesis_input.decomposition,
            execution_trace=sample_synthesis_input.execution_trace,
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(long_input)

        assert isinstance(response, ComposedResponse)

    @pytest.mark.asyncio
    async def test_special_characters_in_results(self, mock_llm_client, sample_decomposition):
        """Test synthesis with special characters in results"""
        step_result = StepResult(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="test_tool",
            input=ToolInput(tool_name="test_tool", parameters={}),
            output=ToolOutput(
                tool_name="test_tool",
                success=True,
                result={"text": "Special chars: <>&\"'{}[]$#@!"},
            ),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        trace = ExecutionTrace(plan_id="plan_123")
        trace.add_result(step_result)

        synthesis_input = SynthesisInput(
            original_query="Test", decomposition=sample_decomposition, execution_trace=trace
        )

        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)
        response = await synthesizer.synthesize(synthesis_input)

        assert isinstance(response, ComposedResponse)
