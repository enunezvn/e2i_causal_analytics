"""
Tests for Tool Composer Phase 1: Decomposer

Tests the QueryDecomposer class which breaks complex queries
into atomic sub-questions.
"""

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.tool_composer.decomposer import (
    DecompositionError,
    QueryDecomposer,
    decompose_sync,
)
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
)


class TestQueryDecomposerInit:
    """Tests for QueryDecomposer initialization"""

    def test_default_initialization(self, mock_llm_client):
        """Test default initialization"""
        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        assert decomposer.model == "claude-sonnet-4-20250514"
        assert decomposer.temperature == 0.3
        assert decomposer.max_sub_questions == 6
        assert decomposer.min_sub_questions == 2

    def test_custom_initialization(self, mock_llm_client):
        """Test custom initialization"""
        decomposer = QueryDecomposer(
            llm_client=mock_llm_client,
            model="claude-3-5-haiku-latest",
            temperature=0.5,
            max_sub_questions=8,
            min_sub_questions=3,
        )
        assert decomposer.model == "claude-3-5-haiku-latest"
        assert decomposer.temperature == 0.5
        assert decomposer.max_sub_questions == 8
        assert decomposer.min_sub_questions == 3


class TestQueryDecomposition:
    """Tests for query decomposition"""

    @pytest.mark.asyncio
    async def test_basic_decomposition(self, mock_llm_client, simple_query):
        """Test basic query decomposition"""
        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose(simple_query)

        assert isinstance(result, DecompositionResult)
        assert result.original_query == simple_query
        assert len(result.sub_questions) >= 2
        assert result.decomposition_reasoning != ""

    @pytest.mark.asyncio
    async def test_decomposition_with_dependencies(self, mock_llm_client, sample_query):
        """Test decomposition creates proper dependency chains"""
        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose(sample_query)

        # Find questions with dependencies
        dependent_questions = [sq for sq in result.sub_questions if sq.depends_on]
        root_questions = result.get_root_questions()

        assert len(root_questions) >= 1
        # At least one question should have dependencies for complex queries
        # (based on default mock response)
        assert len(dependent_questions) >= 1

    @pytest.mark.asyncio
    async def test_decomposition_extracts_entities(self, mock_llm_client, simple_query):
        """Test that decomposition extracts entities"""
        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose(simple_query)

        # Check that at least some sub-questions have entities
        all_entities = []
        for sq in result.sub_questions:
            all_entities.extend(sq.entities)

        assert len(all_entities) > 0

    @pytest.mark.asyncio
    async def test_decomposition_classifies_intent(self, mock_llm_client, simple_query):
        """Test that decomposition classifies intent"""
        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose(simple_query)

        valid_intents = {"CAUSAL", "COMPARATIVE", "PREDICTIVE", "DESCRIPTIVE", "EXPERIMENTAL"}
        for sq in result.sub_questions:
            assert sq.intent in valid_intents


class TestDecompositionValidation:
    """Tests for decomposition validation"""

    @pytest.mark.asyncio
    async def test_minimum_sub_questions(self, mock_llm_client):
        """Test that minimum sub-questions constraint is enforced"""
        # Configure mock to return too few questions
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Single question",
                    "sub_questions": [
                        {"id": "sq_1", "question": "Only one", "intent": "DESCRIPTIVE"}
                    ],
                }
            )
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client, min_sub_questions=2)

        with pytest.raises(DecompositionError) as exc_info:
            await decomposer.decompose("Test query")

        assert "Too few sub-questions" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_maximum_sub_questions_truncated(self, mock_llm_client):
        """Test that too many sub-questions are truncated"""
        many_questions = [
            {
                "id": f"sq_{i}",
                "question": f"Question {i}",
                "intent": "DESCRIPTIVE",
                "depends_on": [],
            }
            for i in range(10)
        ]
        mock_llm_client.set_decomposition_response(
            json.dumps({"reasoning": "Many questions", "sub_questions": many_questions})
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client, max_sub_questions=6)
        result = await decomposer.decompose("Complex query")

        assert len(result.sub_questions) == 6

    @pytest.mark.asyncio
    async def test_dependency_cycle_detection(self, mock_llm_client):
        """Test that dependency cycles are detected"""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Cyclic dependencies",
                    "sub_questions": [
                        {
                            "id": "sq_1",
                            "question": "Q1",
                            "intent": "CAUSAL",
                            "depends_on": ["sq_2"],
                        },
                        {
                            "id": "sq_2",
                            "question": "Q2",
                            "intent": "CAUSAL",
                            "depends_on": ["sq_1"],
                        },
                    ],
                }
            )
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client)

        with pytest.raises(DecompositionError) as exc_info:
            await decomposer.decompose("Test")

        assert "cycle" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_dependency_reference(self, mock_llm_client):
        """Test that invalid dependency references are caught"""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Invalid reference",
                    "sub_questions": [
                        {"id": "sq_1", "question": "Q1", "intent": "CAUSAL", "depends_on": []},
                        {
                            "id": "sq_2",
                            "question": "Q2",
                            "intent": "CAUSAL",
                            "depends_on": ["sq_99"],
                        },  # Invalid
                    ],
                }
            )
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client)

        with pytest.raises(DecompositionError) as exc_info:
            await decomposer.decompose("Test")

        assert "unknown" in str(exc_info.value).lower()


class TestResponseParsing:
    """Tests for LLM response parsing"""

    @pytest.mark.asyncio
    async def test_parse_markdown_code_block(self, mock_llm_client):
        """Test parsing JSON from markdown code block"""
        mock_llm_client.set_decomposition_response(
            """Here is the decomposition:

```json
{
    "reasoning": "Test",
    "sub_questions": [
        {"id": "sq_1", "question": "Q1", "intent": "CAUSAL", "depends_on": []},
        {"id": "sq_2", "question": "Q2", "intent": "DESCRIPTIVE", "depends_on": []}
    ]
}
```"""
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose("Test")

        assert len(result.sub_questions) == 2

    @pytest.mark.asyncio
    async def test_parse_plain_json(self, mock_llm_client):
        """Test parsing plain JSON response"""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Plain JSON",
                    "sub_questions": [
                        {"id": "sq_1", "question": "Q1", "intent": "CAUSAL"},
                        {"id": "sq_2", "question": "Q2", "intent": "DESCRIPTIVE"},
                    ],
                }
            )
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose("Test")

        assert len(result.sub_questions) == 2

    @pytest.mark.asyncio
    async def test_invalid_json_raises_error(self, mock_llm_client):
        """Test that invalid JSON raises DecompositionError"""
        mock_llm_client.set_decomposition_response("Not valid JSON at all")

        decomposer = QueryDecomposer(llm_client=mock_llm_client)

        with pytest.raises(DecompositionError) as exc_info:
            await decomposer.decompose("Test")

        assert "JSON" in str(exc_info.value)


class TestDefaultValues:
    """Tests for default value handling"""

    @pytest.mark.asyncio
    async def test_default_id_generation(self, mock_llm_client):
        """Test that IDs are auto-generated if not provided"""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "No IDs",
                    "sub_questions": [
                        {"question": "Q1", "intent": "CAUSAL"},
                        {"question": "Q2", "intent": "DESCRIPTIVE"},
                    ],
                }
            )
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose("Test")

        assert result.sub_questions[0].id == "sq_1"
        assert result.sub_questions[1].id == "sq_2"

    @pytest.mark.asyncio
    async def test_default_intent(self, mock_llm_client):
        """Test that intent defaults to DESCRIPTIVE"""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "No intent",
                    "sub_questions": [
                        {"id": "sq_1", "question": "Q1"},
                        {"id": "sq_2", "question": "Q2"},
                    ],
                }
            )
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose("Test")

        for sq in result.sub_questions:
            assert sq.intent == "DESCRIPTIVE"

    @pytest.mark.asyncio
    async def test_default_empty_lists(self, mock_llm_client):
        """Test that entities and depends_on default to empty lists"""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Minimal",
                    "sub_questions": [
                        {"id": "sq_1", "question": "Q1", "intent": "CAUSAL"},
                        {"id": "sq_2", "question": "Q2", "intent": "DESCRIPTIVE"},
                    ],
                }
            )
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        result = await decomposer.decompose("Test")

        for sq in result.sub_questions:
            assert sq.entities == []
            assert sq.depends_on == []


class TestLLMInteraction:
    """Tests for LLM client interaction"""

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_params(self, mock_llm_client, simple_query):
        """Test that LLM is called with correct parameters"""
        decomposer = QueryDecomposer(
            llm_client=mock_llm_client, model="test-model", temperature=0.5
        )
        await decomposer.decompose(simple_query)

        assert mock_llm_client.call_count == 1
        call = mock_llm_client.call_history[0]
        assert call["model"] == "test-model"
        assert call["temperature"] == 0.5
        assert call["max_tokens"] == 2000

    @pytest.mark.asyncio
    async def test_query_included_in_message(self, mock_llm_client, simple_query):
        """Test that query is included in message"""
        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        await decomposer.decompose(simple_query)

        call = mock_llm_client.call_history[0]
        message_content = call["messages"][0]["content"]
        assert simple_query in message_content


@pytest.mark.xdist_group(name="sync_wrappers")
class TestSyncWrapper:
    """Tests for synchronous wrapper function"""

    def test_decompose_sync(self, mock_llm_client, simple_query):
        """Test synchronous decompose wrapper"""
        result = decompose_sync(simple_query, mock_llm_client)

        assert isinstance(result, DecompositionResult)
        assert result.original_query == simple_query

    def test_decompose_sync_with_custom_params(self, mock_llm_client, simple_query):
        """Test sync wrapper with custom parameters"""
        result = decompose_sync(
            simple_query, mock_llm_client, model="custom-model", temperature=0.7
        )

        assert isinstance(result, DecompositionResult)
        assert mock_llm_client.call_history[0]["model"] == "custom-model"


class TestErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_llm_error_wrapped(self, mock_llm_client):
        """Test that LLM errors are wrapped in DecompositionError"""
        # Make LLM raise an error
        mock_llm_client.messages.create = AsyncMock(side_effect=Exception("LLM error"))

        decomposer = QueryDecomposer(llm_client=mock_llm_client)

        with pytest.raises(DecompositionError) as exc_info:
            await decomposer.decompose("Test")

        assert "LLM error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_required_field(self, mock_llm_client):
        """Test handling of missing required fields in response"""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Missing questions",
                    "sub_questions": [{"id": "sq_1", "intent": "CAUSAL"}],  # Missing 'question'
                }
            )
        )

        decomposer = QueryDecomposer(llm_client=mock_llm_client)

        with pytest.raises(DecompositionError):
            await decomposer.decompose("Test")
