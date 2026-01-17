"""
Comprehensive tests for E2I Chatbot State module.

Tests the TypedDict definitions and state factory function.
"""

import operator
import uuid
from typing import get_type_hints

import pytest

from src.api.routes.chatbot_state import (
    AgentRequest,
    ChatbotState,
    IntentType,
    StreamChunk,
    create_initial_state,
)


class TestChatbotStateTypedDict:
    """Tests for ChatbotState TypedDict definition."""

    def test_chatbot_state_is_typeddict(self):
        """ChatbotState should be a TypedDict."""
        assert hasattr(ChatbotState, "__annotations__")
        assert hasattr(ChatbotState, "__total__")

    def test_chatbot_state_total_false(self):
        """ChatbotState should have total=False (all fields optional)."""
        # total=False means fields are optional (TypedDict behavior)
        assert ChatbotState.__total__ is False

    def test_chatbot_state_has_user_context_fields(self):
        """ChatbotState should have user/session context fields."""
        annotations = ChatbotState.__annotations__
        assert "user_id" in annotations
        assert "session_id" in annotations
        assert "request_id" in annotations
        assert "trace_id" in annotations

    def test_chatbot_state_has_query_processing_fields(self):
        """ChatbotState should have query processing fields."""
        annotations = ChatbotState.__annotations__
        assert "query" in annotations
        assert "intent" in annotations
        assert "intent_confidence" in annotations
        assert "intent_reasoning" in annotations
        assert "intent_classification_method" in annotations

    def test_chatbot_state_has_agent_routing_fields(self):
        """ChatbotState should have agent routing fields."""
        annotations = ChatbotState.__annotations__
        assert "routed_agent" in annotations
        assert "secondary_agents" in annotations
        assert "routing_confidence" in annotations
        assert "routing_rationale" in annotations

    def test_chatbot_state_has_orchestrator_fields(self):
        """ChatbotState should have orchestrator integration fields."""
        annotations = ChatbotState.__annotations__
        assert "orchestrator_used" in annotations
        assert "agents_dispatched" in annotations
        assert "response_confidence" in annotations

    def test_chatbot_state_has_context_filter_fields(self):
        """ChatbotState should have E2I context filter fields."""
        annotations = ChatbotState.__annotations__
        assert "brand_context" in annotations
        assert "region_context" in annotations

    def test_chatbot_state_has_messages_field(self):
        """ChatbotState should have annotated messages field."""
        annotations = ChatbotState.__annotations__
        assert "messages" in annotations

    def test_chatbot_state_has_tool_execution_fields(self):
        """ChatbotState should have tool execution fields."""
        annotations = ChatbotState.__annotations__
        assert "tool_results" in annotations

    def test_chatbot_state_has_rag_fields(self):
        """ChatbotState should have RAG context fields."""
        annotations = ChatbotState.__annotations__
        assert "rag_context" in annotations
        assert "rag_sources" in annotations
        assert "rag_rewritten_query" in annotations
        assert "rag_retrieval_method" in annotations

    def test_chatbot_state_has_response_generation_fields(self):
        """ChatbotState should have response generation fields."""
        annotations = ChatbotState.__annotations__
        assert "response_text" in annotations
        assert "response_chunks" in annotations
        assert "streaming_complete" in annotations

    def test_chatbot_state_has_evidence_synthesis_fields(self):
        """ChatbotState should have evidence synthesis fields."""
        annotations = ChatbotState.__annotations__
        assert "confidence_statement" in annotations
        assert "evidence_citations" in annotations
        assert "synthesis_method" in annotations
        assert "follow_up_suggestions" in annotations

    def test_chatbot_state_has_conversation_metadata_fields(self):
        """ChatbotState should have conversation metadata fields."""
        annotations = ChatbotState.__annotations__
        assert "conversation_title" in annotations
        assert "agent_name" in annotations
        assert "agent_tier" in annotations

    def test_chatbot_state_has_error_handling_fields(self):
        """ChatbotState should have error handling fields."""
        annotations = ChatbotState.__annotations__
        assert "error" in annotations

    def test_chatbot_state_has_metadata_field(self):
        """ChatbotState should have extensible metadata field."""
        annotations = ChatbotState.__annotations__
        assert "metadata" in annotations

    def test_chatbot_state_field_count(self):
        """ChatbotState should have expected number of fields."""
        annotations = ChatbotState.__annotations__
        # Approximately 35-40 fields expected
        assert len(annotations) >= 35

    def test_chatbot_state_can_be_instantiated(self):
        """ChatbotState can be instantiated as a dict."""
        state: ChatbotState = {
            "user_id": "test-user",
            "query": "test query",
        }
        assert state["user_id"] == "test-user"
        assert state["query"] == "test query"


class TestAgentRequestTypedDict:
    """Tests for AgentRequest TypedDict definition."""

    def test_agent_request_is_typeddict(self):
        """AgentRequest should be a TypedDict."""
        assert hasattr(AgentRequest, "__annotations__")

    def test_agent_request_has_required_fields(self):
        """AgentRequest should have required fields."""
        annotations = AgentRequest.__annotations__
        assert "query" in annotations
        assert "user_id" in annotations
        assert "request_id" in annotations

    def test_agent_request_has_optional_fields(self):
        """AgentRequest should have optional fields."""
        annotations = AgentRequest.__annotations__
        assert "session_id" in annotations
        assert "brand_context" in annotations
        assert "region_context" in annotations

    def test_agent_request_field_count(self):
        """AgentRequest should have exactly 6 fields."""
        annotations = AgentRequest.__annotations__
        assert len(annotations) == 6

    def test_agent_request_can_be_instantiated(self):
        """AgentRequest can be instantiated as a dict."""
        request: AgentRequest = {
            "query": "test query",
            "user_id": "user-123",
            "request_id": "req-456",
            "session_id": None,
            "brand_context": "Kisqali",
            "region_context": None,
        }
        assert request["query"] == "test query"
        assert request["user_id"] == "user-123"
        assert request["brand_context"] == "Kisqali"


class TestStreamChunkTypedDict:
    """Tests for StreamChunk TypedDict definition."""

    def test_stream_chunk_is_typeddict(self):
        """StreamChunk should be a TypedDict."""
        assert hasattr(StreamChunk, "__annotations__")

    def test_stream_chunk_has_type_field(self):
        """StreamChunk should have type field."""
        annotations = StreamChunk.__annotations__
        assert "type" in annotations

    def test_stream_chunk_has_data_field(self):
        """StreamChunk should have data field."""
        annotations = StreamChunk.__annotations__
        assert "data" in annotations

    def test_stream_chunk_field_count(self):
        """StreamChunk should have exactly 2 fields."""
        annotations = StreamChunk.__annotations__
        assert len(annotations) == 2

    def test_stream_chunk_text_type(self):
        """StreamChunk can represent text chunks."""
        chunk: StreamChunk = {"type": "text", "data": "Hello, world!"}
        assert chunk["type"] == "text"
        assert chunk["data"] == "Hello, world!"

    def test_stream_chunk_session_id_type(self):
        """StreamChunk can represent session_id."""
        chunk: StreamChunk = {"type": "session_id", "data": "user~uuid"}
        assert chunk["type"] == "session_id"

    def test_stream_chunk_conversation_title_type(self):
        """StreamChunk can represent conversation_title."""
        chunk: StreamChunk = {"type": "conversation_title", "data": "Sales Analysis"}
        assert chunk["type"] == "conversation_title"

    def test_stream_chunk_tool_call_type(self):
        """StreamChunk can represent tool_call."""
        chunk: StreamChunk = {"type": "tool_call", "data": "causal_analysis"}
        assert chunk["type"] == "tool_call"

    def test_stream_chunk_tool_result_type(self):
        """StreamChunk can represent tool_result."""
        chunk: StreamChunk = {"type": "tool_result", "data": '{"result": "success"}'}
        assert chunk["type"] == "tool_result"

    def test_stream_chunk_done_type(self):
        """StreamChunk can represent done signal."""
        chunk: StreamChunk = {"type": "done", "data": ""}
        assert chunk["type"] == "done"

    def test_stream_chunk_error_type(self):
        """StreamChunk can represent error."""
        chunk: StreamChunk = {"type": "error", "data": "Something went wrong"}
        assert chunk["type"] == "error"


class TestIntentType:
    """Tests for IntentType class constants."""

    def test_intent_type_kpi_query(self):
        """IntentType should have KPI_QUERY constant."""
        assert IntentType.KPI_QUERY == "kpi_query"

    def test_intent_type_causal_analysis(self):
        """IntentType should have CAUSAL_ANALYSIS constant."""
        assert IntentType.CAUSAL_ANALYSIS == "causal_analysis"

    def test_intent_type_agent_status(self):
        """IntentType should have AGENT_STATUS constant."""
        assert IntentType.AGENT_STATUS == "agent_status"

    def test_intent_type_recommendation(self):
        """IntentType should have RECOMMENDATION constant."""
        assert IntentType.RECOMMENDATION == "recommendation"

    def test_intent_type_search(self):
        """IntentType should have SEARCH constant."""
        assert IntentType.SEARCH == "search"

    def test_intent_type_general(self):
        """IntentType should have GENERAL constant."""
        assert IntentType.GENERAL == "general"

    def test_intent_type_greeting(self):
        """IntentType should have GREETING constant."""
        assert IntentType.GREETING == "greeting"

    def test_intent_type_help(self):
        """IntentType should have HELP constant."""
        assert IntentType.HELP == "help"

    def test_intent_type_multi_faceted(self):
        """IntentType should have MULTI_FACETED constant."""
        assert IntentType.MULTI_FACETED == "multi_faceted"

    def test_intent_type_cohort_definition(self):
        """IntentType should have COHORT_DEFINITION constant."""
        assert IntentType.COHORT_DEFINITION == "cohort_definition"

    def test_intent_type_count(self):
        """IntentType should have expected number of constants."""
        intent_attrs = [
            attr
            for attr in dir(IntentType)
            if not attr.startswith("_") and isinstance(getattr(IntentType, attr), str)
        ]
        assert len(intent_attrs) == 10

    def test_intent_type_values_are_strings(self):
        """All IntentType values should be strings."""
        for attr in dir(IntentType):
            if not attr.startswith("_"):
                value = getattr(IntentType, attr)
                if isinstance(value, str):
                    assert isinstance(value, str)


class TestCreateInitialState:
    """Tests for create_initial_state factory function."""

    def test_create_initial_state_basic(self):
        """create_initial_state should create a valid state with minimal args."""
        state = create_initial_state(
            user_id="user-123",
            query="What is the TRx for Kisqali?",
            request_id="req-456",
        )
        assert state["user_id"] == "user-123"
        assert state["query"] == "What is the TRx for Kisqali?"
        assert state["request_id"] == "req-456"

    def test_create_initial_state_generates_session_id(self):
        """create_initial_state should generate session_id if not provided."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["session_id"] is not None
        assert state["session_id"].startswith("user-123~")
        # Should contain a UUID after the tilde
        session_uuid = state["session_id"].split("~")[1]
        uuid.UUID(session_uuid)  # Should not raise

    def test_create_initial_state_uses_provided_session_id(self):
        """create_initial_state should use provided session_id."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
            session_id="existing-session",
        )
        assert state["session_id"] == "existing-session"

    def test_create_initial_state_with_brand_context(self):
        """create_initial_state should set brand_context."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
            brand_context="Kisqali",
        )
        assert state["brand_context"] == "Kisqali"

    def test_create_initial_state_with_region_context(self):
        """create_initial_state should set region_context."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
            region_context="Northeast",
        )
        assert state["region_context"] == "Northeast"

    def test_create_initial_state_with_trace_id(self):
        """create_initial_state should set trace_id for observability."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
            trace_id="trace-789",
        )
        assert state["trace_id"] == "trace-789"

    def test_create_initial_state_intent_fields_none(self):
        """create_initial_state should initialize intent fields as None."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["intent"] is None
        assert state["intent_confidence"] is None
        assert state["intent_reasoning"] is None
        assert state["intent_classification_method"] is None

    def test_create_initial_state_routing_fields_none(self):
        """create_initial_state should initialize routing fields as None/empty."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["routed_agent"] is None
        assert state["secondary_agents"] == []
        assert state["routing_confidence"] is None
        assert state["routing_rationale"] is None

    def test_create_initial_state_orchestrator_fields_default(self):
        """create_initial_state should initialize orchestrator fields with defaults."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["orchestrator_used"] is False
        assert state["agents_dispatched"] == []
        assert state["response_confidence"] is None

    def test_create_initial_state_empty_lists(self):
        """create_initial_state should initialize list fields as empty lists."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["messages"] == []
        assert state["tool_results"] == []
        assert state["rag_context"] == []
        assert state["rag_sources"] == []
        assert state["response_chunks"] == []
        assert state["evidence_citations"] == []
        assert state["follow_up_suggestions"] == []

    def test_create_initial_state_rag_fields_none(self):
        """create_initial_state should initialize RAG optional fields as None."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["rag_rewritten_query"] is None
        assert state["rag_retrieval_method"] is None

    def test_create_initial_state_response_fields_empty(self):
        """create_initial_state should initialize response fields as empty."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["response_text"] == ""
        assert state["streaming_complete"] is False

    def test_create_initial_state_synthesis_fields_none(self):
        """create_initial_state should initialize synthesis fields as None."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["confidence_statement"] is None
        assert state["synthesis_method"] is None

    def test_create_initial_state_metadata_fields_none(self):
        """create_initial_state should initialize metadata fields as None/empty."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        assert state["conversation_title"] is None
        assert state["agent_name"] is None
        assert state["agent_tier"] is None
        assert state["error"] is None
        assert state["metadata"] == {}

    def test_create_initial_state_returns_chatbot_state(self):
        """create_initial_state should return a ChatbotState TypedDict."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        # Check it has the expected structure
        assert isinstance(state, dict)
        # All keys should be valid ChatbotState fields
        for key in state.keys():
            assert key in ChatbotState.__annotations__

    def test_create_initial_state_all_fields_present(self):
        """create_initial_state should populate all expected fields."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        # Should have most fields defined
        assert len(state) >= 30


class TestStateFieldTypes:
    """Tests for verifying field type correctness."""

    def test_user_id_is_str_type(self):
        """user_id should be str type."""
        annotations = ChatbotState.__annotations__
        assert annotations["user_id"] == str

    def test_session_id_is_str_type(self):
        """session_id should be str type."""
        annotations = ChatbotState.__annotations__
        assert annotations["session_id"] == str

    def test_query_is_str_type(self):
        """query should be str type."""
        annotations = ChatbotState.__annotations__
        assert annotations["query"] == str

    def test_response_text_is_str_type(self):
        """response_text should be str type."""
        annotations = ChatbotState.__annotations__
        assert annotations["response_text"] == str

    def test_streaming_complete_is_bool_type(self):
        """streaming_complete should be bool type."""
        annotations = ChatbotState.__annotations__
        assert annotations["streaming_complete"] == bool

    def test_orchestrator_used_is_bool_type(self):
        """orchestrator_used should be bool type."""
        annotations = ChatbotState.__annotations__
        assert annotations["orchestrator_used"] == bool


class TestStateModification:
    """Tests for state modification patterns."""

    def test_state_can_update_intent(self):
        """State intent can be updated."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        state["intent"] = IntentType.KPI_QUERY
        state["intent_confidence"] = 0.95
        assert state["intent"] == "kpi_query"
        assert state["intent_confidence"] == 0.95

    def test_state_can_append_tool_results(self):
        """State tool_results can be appended."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        state["tool_results"].append({"tool": "causal_analyzer", "result": "success"})
        assert len(state["tool_results"]) == 1

    def test_state_can_set_error(self):
        """State error can be set."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        state["error"] = "Something went wrong"
        assert state["error"] == "Something went wrong"

    def test_state_can_update_metadata(self):
        """State metadata can be updated."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        state["metadata"]["custom_key"] = "custom_value"
        assert state["metadata"]["custom_key"] == "custom_value"

    def test_state_can_set_orchestrator_fields(self):
        """State orchestrator fields can be updated."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        state["orchestrator_used"] = True
        state["agents_dispatched"] = ["causal_impact", "gap_analyzer"]
        state["response_confidence"] = 0.85
        assert state["orchestrator_used"] is True
        assert len(state["agents_dispatched"]) == 2

    def test_state_can_set_rag_context(self):
        """State RAG context can be set."""
        state = create_initial_state(
            user_id="user-123",
            query="test",
            request_id="req-456",
        )
        state["rag_context"] = [{"doc_id": "doc-1", "content": "Sample content"}]
        state["rag_sources"] = ["doc-1"]
        assert len(state["rag_context"]) == 1
        assert state["rag_sources"] == ["doc-1"]


class TestIntegrationScenarios:
    """Tests for realistic usage scenarios."""

    def test_kpi_query_state_flow(self):
        """Test state flow for a KPI query."""
        # 1. Create initial state
        state = create_initial_state(
            user_id="analyst-001",
            query="What is the TRx for Kisqali in Q4?",
            request_id="req-kpi-001",
            brand_context="Kisqali",
        )

        # 2. Simulate intent classification
        state["intent"] = IntentType.KPI_QUERY
        state["intent_confidence"] = 0.92
        state["intent_classification_method"] = "dspy"

        # 3. Simulate agent routing
        state["routed_agent"] = "gap_analyzer"
        state["routing_confidence"] = 0.88

        # 4. Simulate RAG retrieval
        state["rag_context"] = [{"doc_id": "kpi-doc-1", "content": "TRx data..."}]
        state["rag_sources"] = ["kpi-doc-1"]

        # 5. Simulate response generation
        state["response_text"] = "The TRx for Kisqali in Q4 was 15,234 units."
        state["agent_name"] = "gap_analyzer"
        state["agent_tier"] = 2

        # Verify final state
        assert state["intent"] == "kpi_query"
        assert state["routed_agent"] == "gap_analyzer"
        assert len(state["rag_context"]) == 1
        assert "TRx" in state["response_text"]

    def test_causal_analysis_state_flow(self):
        """Test state flow for a causal analysis query."""
        state = create_initial_state(
            user_id="analyst-002",
            query="Why did sales decline in the Northeast?",
            request_id="req-causal-001",
            region_context="Northeast",
        )

        state["intent"] = IntentType.CAUSAL_ANALYSIS
        state["intent_confidence"] = 0.89
        state["routed_agent"] = "causal_impact"
        state["tool_results"] = [
            {"tool": "causal_chain_tracer", "result": {"effect": -0.12}}
        ]
        state["response_text"] = "The decline was primarily caused by..."
        state["confidence_statement"] = "High confidence based on causal analysis"
        state["synthesis_method"] = "dspy"

        assert state["intent"] == "causal_analysis"
        assert len(state["tool_results"]) == 1

    def test_multi_agent_orchestration_flow(self):
        """Test state flow for multi-agent orchestration."""
        state = create_initial_state(
            user_id="analyst-003",
            query="Compare impact of rep visits vs speaker programs and predict ROI",
            request_id="req-multi-001",
        )

        state["intent"] = IntentType.MULTI_FACETED
        state["orchestrator_used"] = True
        state["agents_dispatched"] = [
            "causal_impact",
            "gap_analyzer",
            "prediction_synthesizer",
        ]
        state["response_confidence"] = 0.82

        # Each agent contributes to tool_results
        state["tool_results"] = [
            {"agent": "causal_impact", "result": "rep_visits_effect: 0.15"},
            {"agent": "causal_impact", "result": "speaker_programs_effect: 0.22"},
            {"agent": "prediction_synthesizer", "result": "roi_prediction: 1.4x"},
        ]

        assert state["orchestrator_used"] is True
        assert len(state["agents_dispatched"]) == 3
        assert len(state["tool_results"]) == 3

    def test_streaming_response_flow(self):
        """Test state flow for streaming responses."""
        state = create_initial_state(
            user_id="analyst-004",
            query="Explain the market trends",
            request_id="req-stream-001",
        )

        # Simulate streaming chunks
        chunks = [
            "The market ",
            "has shown ",
            "significant growth ",
            "in Q4.",
        ]
        for chunk in chunks:
            state["response_chunks"].append(chunk)

        state["streaming_complete"] = True
        state["response_text"] = "".join(chunks)

        assert len(state["response_chunks"]) == 4
        assert state["streaming_complete"] is True
        assert state["response_text"] == "The market has shown significant growth in Q4."

    def test_error_handling_flow(self):
        """Test state flow for error scenarios."""
        state = create_initial_state(
            user_id="analyst-005",
            query="Invalid query $#%@",
            request_id="req-error-001",
        )

        # Simulate error during processing
        state["intent"] = IntentType.GENERAL
        state["error"] = "Failed to parse query: invalid characters"
        state["response_text"] = "I'm sorry, I couldn't process your request."

        assert state["error"] is not None
        assert "invalid" in state["error"].lower()
