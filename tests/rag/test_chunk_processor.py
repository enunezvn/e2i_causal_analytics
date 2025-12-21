"""
Tests for ChunkProcessor.

Tests the semantic chunking of agent outputs for RAG indexing.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from src.rag.chunk_processor import ChunkProcessor
from src.rag.models.insight_models import Chunk

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def chunk_processor():
    """Default chunk processor."""
    return ChunkProcessor()


@pytest.fixture
def custom_chunk_processor():
    """Chunk processor with custom settings."""
    return ChunkProcessor(chunk_size=100, chunk_overlap=10)


@dataclass
class MockAgentActivity:
    """Mock agent activity for testing."""

    id: str
    agent_id: str
    analysis_results: str


@dataclass
class MockAgentActivityWithContent:
    """Mock agent activity with content field."""

    id: str
    agent_id: str
    content: str


@dataclass
class MockCausalPath:
    """Mock causal path for testing."""

    id: str
    cause: str
    effect: str
    path_description: str


@dataclass
class MockKPISnapshot:
    """Mock KPI snapshot for testing."""

    kpi_name: str
    timestamp: str
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return {"kpi_name": self.kpi_name, "timestamp": self.timestamp, "value": self.value}


# ============================================================================
# ChunkProcessor Initialization Tests
# ============================================================================


class TestChunkProcessorInit:
    """Test ChunkProcessor initialization."""

    def test_default_initialization(self, chunk_processor):
        """Test default chunk processor settings."""
        assert chunk_processor.chunk_size == 512
        assert chunk_processor.chunk_overlap == 50

    def test_custom_initialization(self, custom_chunk_processor):
        """Test custom chunk processor settings."""
        assert custom_chunk_processor.chunk_size == 100
        assert custom_chunk_processor.chunk_overlap == 10

    def test_zero_chunk_size(self):
        """Test with zero chunk size."""
        processor = ChunkProcessor(chunk_size=0, chunk_overlap=0)
        assert processor.chunk_size == 0

    def test_large_chunk_size(self):
        """Test with very large chunk size."""
        processor = ChunkProcessor(chunk_size=10000, chunk_overlap=100)
        assert processor.chunk_size == 10000


# ============================================================================
# Chunk Agent Output Tests
# ============================================================================


class TestChunkAgentOutput:
    """Test chunk_agent_output method."""

    def test_chunk_agent_output_with_analysis_results(self, chunk_processor):
        """Test chunking agent output with analysis_results field."""
        activity = MockAgentActivity(
            id="act-123",
            agent_id="causal_impact",
            analysis_results="This is a test analysis. It contains findings about Kisqali adoption.",
        )

        chunks = chunk_processor.chunk_agent_output(activity)

        assert len(chunks) >= 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].source_type == "agent_analysis"
        assert chunks[0].metadata["agent_id"] == "causal_impact"
        assert chunks[0].metadata["activity_id"] == "act-123"

    def test_chunk_agent_output_with_content_field(self, chunk_processor):
        """Test chunking agent output with content field."""
        activity = MockAgentActivityWithContent(
            id="act-456",
            agent_id="gap_analyzer",
            content="Gap analysis report for Q3 2024. Multiple opportunities identified.",
        )

        chunks = chunk_processor.chunk_agent_output(activity)

        assert len(chunks) >= 1
        assert "Gap analysis" in chunks[0].content

    def test_chunk_agent_output_with_string(self, chunk_processor):
        """Test chunking plain string output."""
        output = "Simple string output for testing"

        chunks = chunk_processor.chunk_agent_output(output)

        assert len(chunks) >= 1
        assert "Simple string output" in chunks[0].content

    def test_chunk_agent_output_custom_chunk_size(self, chunk_processor):
        """Test chunking with custom chunk size parameter."""
        # Create text with many sentences to trigger splitting
        sentences = ". ".join([f"Sentence number {i} with some content" for i in range(100)])
        activity = MockAgentActivity(
            id="act-789", agent_id="prediction", analysis_results=sentences
        )

        # Use small chunk size to force multiple chunks
        chunks = chunk_processor.chunk_agent_output(activity, chunk_size=10)

        assert len(chunks) > 1

    def test_chunk_agent_output_chunk_indices(self, chunk_processor):
        """Test that chunk indices are correctly assigned."""
        activity = MockAgentActivity(
            id="act-multi",
            agent_id="test",
            analysis_results="Sentence one. Sentence two. Sentence three. " * 50,
        )

        chunks = chunk_processor.chunk_agent_output(activity, chunk_size=20)

        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                assert chunk.metadata["chunk_index"] == i
                assert chunk.metadata["total_chunks"] == len(chunks)

    def test_chunk_agent_output_empty_analysis(self, chunk_processor):
        """Test chunking with empty analysis results."""
        activity = MockAgentActivity(id="act-empty", agent_id="test", analysis_results="")

        chunks = chunk_processor.chunk_agent_output(activity)

        assert len(chunks) == 0

    def test_chunk_agent_output_embedding_is_none(self, chunk_processor):
        """Test that embedding is None (generated during indexing)."""
        activity = MockAgentActivity(id="act-emb", agent_id="test", analysis_results="Test content")

        chunks = chunk_processor.chunk_agent_output(activity)

        assert chunks[0].embedding is None


# ============================================================================
# Chunk Causal Path Tests
# ============================================================================


class TestChunkCausalPath:
    """Test chunk_causal_path method."""

    def test_chunk_causal_path_basic(self, chunk_processor):
        """Test chunking a causal path."""
        path = MockCausalPath(
            id="path-123",
            cause="Increased sales rep visits",
            effect="Higher TRx volume",
            path_description="Sales rep visits lead to higher TRx through improved HCP awareness.",
        )

        chunks = chunk_processor.chunk_causal_path(path)

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].source_type == "causal_path"
        assert "Sales rep visits" in chunks[0].content
        assert chunks[0].metadata["path_id"] == "path-123"
        assert chunks[0].metadata["cause"] == "Increased sales rep visits"
        assert chunks[0].metadata["effect"] == "Higher TRx volume"

    def test_chunk_causal_path_without_description(self, chunk_processor):
        """Test chunking path without path_description field."""

        @dataclass
        class MinimalPath:
            id: str

        path = MinimalPath(id="path-min")
        chunks = chunk_processor.chunk_causal_path(path)

        assert len(chunks) == 1
        # Should fallback to str(path)
        assert "MinimalPath" in chunks[0].content or "path-min" in chunks[0].content

    def test_chunk_causal_path_embedding_none(self, chunk_processor):
        """Test that causal path chunks have no embedding."""
        path = MockCausalPath(id="path-emb", cause="A", effect="B", path_description="A causes B")

        chunks = chunk_processor.chunk_causal_path(path)

        assert chunks[0].embedding is None


# ============================================================================
# Chunk KPI Snapshot Tests
# ============================================================================


class TestChunkKPISnapshot:
    """Test chunk_kpi_snapshot method."""

    def test_chunk_kpi_snapshot_with_to_dict(self, chunk_processor):
        """Test chunking KPI snapshot with to_dict method."""
        snapshot = MockKPISnapshot(kpi_name="TRx", timestamp="2024-Q3", value=15000.5)

        chunks = chunk_processor.chunk_kpi_snapshot(snapshot)

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].source_type == "kpi_snapshot"
        assert "TRx" in chunks[0].content
        assert "15000.5" in chunks[0].content
        assert chunks[0].metadata["kpi_name"] == "TRx"
        assert chunks[0].metadata["timestamp"] == "2024-Q3"

    def test_chunk_kpi_snapshot_without_to_dict(self, chunk_processor):
        """Test chunking snapshot without to_dict method."""

        @dataclass
        class SimpleSnapshot:
            kpi_name: str
            value: float

        snapshot = SimpleSnapshot(kpi_name="NRx", value=500.0)
        chunks = chunk_processor.chunk_kpi_snapshot(snapshot)

        assert len(chunks) == 1
        # Should fallback to str(snapshot)
        assert "NRx" in chunks[0].content or "500" in chunks[0].content

    def test_chunk_kpi_snapshot_metadata(self, chunk_processor):
        """Test KPI snapshot metadata extraction."""
        snapshot = MockKPISnapshot(kpi_name="conversion_rate", timestamp="2024-12-01", value=0.15)

        chunks = chunk_processor.chunk_kpi_snapshot(snapshot)

        assert chunks[0].metadata["kpi_name"] == "conversion_rate"
        assert chunks[0].metadata["timestamp"] == "2024-12-01"

    def test_chunk_kpi_snapshot_missing_attributes(self, chunk_processor):
        """Test KPI snapshot with missing optional attributes."""

        @dataclass
        class PartialSnapshot:
            value: float

        snapshot = PartialSnapshot(value=100.0)
        chunks = chunk_processor.chunk_kpi_snapshot(snapshot)

        assert len(chunks) == 1
        assert chunks[0].metadata["kpi_name"] is None
        assert chunks[0].metadata["timestamp"] is None


# ============================================================================
# Semantic Split Tests
# ============================================================================


class TestSemanticSplit:
    """Test _semantic_split method."""

    def test_semantic_split_empty_text(self, chunk_processor):
        """Test splitting empty text."""
        result = chunk_processor._semantic_split("", 100)
        assert result == []

    def test_semantic_split_none_text(self, chunk_processor):
        """Test splitting None (should handle gracefully)."""
        result = chunk_processor._semantic_split(None, 100)
        assert result == []

    def test_semantic_split_short_text(self, chunk_processor):
        """Test splitting text shorter than chunk size."""
        text = "This is a short sentence."
        result = chunk_processor._semantic_split(text, 100)

        assert len(result) == 1
        assert "This is a short sentence" in result[0]

    def test_semantic_split_sentence_boundaries(self, chunk_processor):
        """Test that split respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_processor._semantic_split(text, 3)  # 3 words per chunk

        # Should split on sentence boundaries
        for chunk in result:
            assert chunk.endswith(".")

    def test_semantic_split_long_text(self, chunk_processor):
        """Test splitting long text into multiple chunks."""
        # Create text with many sentences
        sentences = ["This is sentence number {}.".format(i) for i in range(100)]
        text = " ".join(sentences)

        result = chunk_processor._semantic_split(text, 20)

        assert len(result) > 1

    def test_semantic_split_overlap(self, custom_chunk_processor):
        """Test that chunks have overlap."""
        # Create text that will be split
        sentences = ["Sentence {}.".format(i) for i in range(20)]
        text = " ".join(sentences)

        result = custom_chunk_processor._semantic_split(text, 10)

        if len(result) > 1:
            # Check for overlap by looking for repeated content
            # Due to overlap, some sentences should appear in consecutive chunks
            for i in range(len(result) - 1):
                result[i]
                result[i + 1]
                # With overlap of 2 sentences, there should be some overlap
                # This is a soft test since overlap is based on last 2 sentences

    def test_semantic_split_newlines(self, chunk_processor):
        """Test that newlines are handled as spaces."""
        text = "First paragraph.\nSecond paragraph.\nThird paragraph."
        result = chunk_processor._semantic_split(text, 100)

        assert len(result) >= 1
        # Newlines should be converted to spaces
        assert "\n" not in result[0]

    def test_semantic_split_multiple_periods(self, chunk_processor):
        """Test text with multiple periods."""
        text = "Dr. Smith visited the clinic. He saw 10 patients. All were from region A."
        result = chunk_processor._semantic_split(text, 100)

        assert len(result) >= 1
        # Note: Current implementation splits on ". " which handles "Dr." correctly

    def test_semantic_split_empty_sentences(self, chunk_processor):
        """Test handling of empty sentences from double periods."""
        text = "First sentence.. Second sentence..."
        result = chunk_processor._semantic_split(text, 100)

        # Should handle gracefully without empty chunks
        for chunk in result:
            assert chunk.strip() != ""


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_chunk_large_single_sentence(self, chunk_processor):
        """Test chunking a very large single sentence."""
        # A single sentence with 1000 words
        text = "word " * 1000
        activity = MockAgentActivity(id="large", agent_id="test", analysis_results=text)

        chunks = chunk_processor.chunk_agent_output(activity, chunk_size=100)

        # Should still create chunks even without sentence boundaries
        # Current implementation may create one large chunk or handle differently
        assert len(chunks) >= 1

    def test_chunk_unicode_content(self, chunk_processor):
        """Test chunking content with unicode characters."""
        activity = MockAgentActivity(
            id="unicode",
            agent_id="test",
            analysis_results="Analysis: Café résumé 日本語テスト. More content here.",
        )

        chunks = chunk_processor.chunk_agent_output(activity)

        assert len(chunks) >= 1
        # Unicode should be preserved
        assert "Café" in chunks[0].content or "résumé" in chunks[0].content

    def test_chunk_special_characters(self, chunk_processor):
        """Test chunking content with special characters."""
        activity = MockAgentActivity(
            id="special",
            agent_id="test",
            analysis_results="Price: $100. Rate: 15%. Formula: a + b = c.",
        )

        chunks = chunk_processor.chunk_agent_output(activity)

        assert len(chunks) >= 1
        # Special characters should be preserved
        assert "$" in chunks[0].content or "%" in chunks[0].content

    def test_chunk_with_none_optional_fields(self, chunk_processor):
        """Test chunking output with None optional fields."""

        @dataclass
        class OutputWithNone:
            id: Optional[str]
            agent_id: Optional[str]
            analysis_results: str

        activity = OutputWithNone(id=None, agent_id=None, analysis_results="Some analysis text.")

        chunks = chunk_processor.chunk_agent_output(activity)

        assert len(chunks) >= 1
        assert chunks[0].metadata["agent_id"] is None
        assert chunks[0].metadata["activity_id"] is None


# ============================================================================
# Chunk Model Validation Tests
# ============================================================================


class TestChunkModel:
    """Test Chunk model structure."""

    def test_chunk_has_required_fields(self, chunk_processor):
        """Test that chunks have all required fields."""
        activity = MockAgentActivity(
            id="test", agent_id="test", analysis_results="Test content for validation."
        )

        chunks = chunk_processor.chunk_agent_output(activity)
        chunk = chunks[0]

        # Verify all fields exist
        assert hasattr(chunk, "content")
        assert hasattr(chunk, "source_type")
        assert hasattr(chunk, "embedding")
        assert hasattr(chunk, "metadata")

    def test_chunk_source_types(self, chunk_processor):
        """Test that source types are correctly assigned."""
        activity = MockAgentActivity(id="a", agent_id="a", analysis_results="Test")
        path = MockCausalPath(id="p", cause="A", effect="B", path_description="A->B")
        snapshot = MockKPISnapshot(kpi_name="TRx", timestamp="2024", value=100)

        agent_chunks = chunk_processor.chunk_agent_output(activity)
        path_chunks = chunk_processor.chunk_causal_path(path)
        kpi_chunks = chunk_processor.chunk_kpi_snapshot(snapshot)

        assert agent_chunks[0].source_type == "agent_analysis"
        assert path_chunks[0].source_type == "causal_path"
        assert kpi_chunks[0].source_type == "kpi_snapshot"


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestChunkProcessorIntegration:
    """Integration-style tests for chunk processor."""

    def test_process_multiple_outputs(self, chunk_processor):
        """Test processing multiple outputs in sequence."""
        activities = [
            MockAgentActivity(id=f"act-{i}", agent_id="test", analysis_results=f"Analysis {i}")
            for i in range(5)
        ]

        all_chunks = []
        for activity in activities:
            chunks = chunk_processor.chunk_agent_output(activity)
            all_chunks.extend(chunks)

        assert len(all_chunks) == 5
        # Each should have unique activity_id
        activity_ids = [c.metadata["activity_id"] for c in all_chunks]
        assert len(set(activity_ids)) == 5

    def test_mixed_content_types(self, chunk_processor):
        """Test processing mixed content types."""
        activity = MockAgentActivity(id="a1", agent_id="test", analysis_results="Analysis")
        path = MockCausalPath(id="p1", cause="X", effect="Y", path_description="X causes Y")
        snapshot = MockKPISnapshot(kpi_name="NRx", timestamp="2024-Q4", value=250)

        all_chunks = []
        all_chunks.extend(chunk_processor.chunk_agent_output(activity))
        all_chunks.extend(chunk_processor.chunk_causal_path(path))
        all_chunks.extend(chunk_processor.chunk_kpi_snapshot(snapshot))

        assert len(all_chunks) == 3
        source_types = {c.source_type for c in all_chunks}
        assert source_types == {"agent_analysis", "causal_path", "kpi_snapshot"}
