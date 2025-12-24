"""Tests for Experiment Designer Memory Hooks.

Tests the 4-Memory Architecture integration for experiment design:
- Working Memory: Caching experiment designs
- Episodic Memory: Storing design history for learning
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.experiment_designer.memory_hooks import (
    ExperimentDesignContext,
    ExperimentDesignRecord,
    ValidityThreatRecord,
    ExperimentDesignerMemoryHooks,
    contribute_to_memory,
    get_experiment_designer_memory_hooks,
    reset_memory_hooks,
)


# ============================================================================
# Test Data Structures
# ============================================================================


class TestExperimentDesignContext:
    """Test ExperimentDesignContext data structure."""

    def test_create_context(self):
        """Test context creation with defaults."""
        context = ExperimentDesignContext(session_id="test-session-123")

        assert context.session_id == "test-session-123"
        assert context.working_memory == []
        assert context.episodic_context == []
        assert isinstance(context.retrieval_timestamp, datetime)

    def test_context_with_data(self):
        """Test context creation with populated data."""
        context = ExperimentDesignContext(
            session_id="test-session",
            working_memory=[{"cached_design": "RCT"}],
            episodic_context=[{"past_design": "quasi_experiment"}],
        )

        assert len(context.working_memory) == 1
        assert len(context.episodic_context) == 1

    def test_context_to_dict(self):
        """Test context serialization."""
        context = ExperimentDesignContext(session_id="test-session")
        result = context.to_dict()

        assert result["session_id"] == "test-session"
        assert "retrieval_timestamp" in result
        assert isinstance(result["retrieval_timestamp"], str)


class TestExperimentDesignRecord:
    """Test ExperimentDesignRecord data structure."""

    def test_create_record(self):
        """Test record creation."""
        record = ExperimentDesignRecord(
            record_id="rec-123",
            session_id="sess-456",
            timestamp=datetime.now(timezone.utc),
            business_question="Does increasing rep visits improve engagement?",
            brand="Remibrutinib",
            constraints={"power": 0.80, "alpha": 0.05},
            design_type="RCT",
            design_rationale="RCT provides strongest causal evidence",
            randomization_unit="individual",
            randomization_method="stratified",
            required_sample_size=500,
            achieved_power=0.85,
            duration_estimate_days=90,
            overall_validity_score=0.82,
            validity_confidence="high",
            threat_count=3,
            critical_threat_count=0,
            redesign_iterations=1,
            total_latency_ms=45000,
            warnings=[],
        )

        assert record.record_id == "rec-123"
        assert record.design_type == "RCT"
        assert record.required_sample_size == 500
        assert record.overall_validity_score == 0.82

    def test_record_to_dict(self):
        """Test record serialization."""
        record = ExperimentDesignRecord(
            record_id="rec-123",
            session_id="sess-456",
            timestamp=datetime.now(timezone.utc),
            business_question="Test question",
            brand=None,
            constraints={},
            design_type="quasi_experiment",
            design_rationale="Randomization not feasible",
            randomization_unit="cluster",
            randomization_method="simple",
            required_sample_size=200,
            achieved_power=0.75,
            duration_estimate_days=60,
            overall_validity_score=0.65,
            validity_confidence="medium",
            threat_count=5,
            critical_threat_count=1,
            redesign_iterations=2,
            total_latency_ms=55000,
            warnings=["Limited power"],
        )

        result = record.to_dict()

        assert result["record_id"] == "rec-123"
        assert result["design_type"] == "quasi_experiment"
        assert isinstance(result["timestamp"], str)
        assert result["threat_count"] == 5


class TestValidityThreatRecord:
    """Test ValidityThreatRecord data structure."""

    def test_create_threat_record(self):
        """Test threat record creation."""
        record = ValidityThreatRecord(
            threat_id="threat-123",
            experiment_record_id="exp-456",
            timestamp=datetime.now(timezone.utc),
            threat_type="internal",
            threat_name="Selection Bias",
            severity="high",
            description="Non-random assignment may confound results",
            design_type="quasi_experiment",
            business_question_keywords=["rep", "visits", "engagement"],
            affected_outcomes=["engagement_score", "conversion_rate"],
            mitigation_possible=True,
            mitigation_strategy="Use propensity score matching",
            mitigation_effectiveness="high",
        )

        assert record.threat_id == "threat-123"
        assert record.threat_type == "internal"
        assert record.severity == "high"
        assert len(record.affected_outcomes) == 2

    def test_threat_record_to_dict(self):
        """Test threat record serialization."""
        record = ValidityThreatRecord(
            threat_id="threat-123",
            experiment_record_id="exp-456",
            timestamp=datetime.now(timezone.utc),
            threat_type="external",
            threat_name="Generalizability",
            severity="medium",
            description="Results may not generalize",
            design_type="RCT",
            business_question_keywords=["hcp", "targeting"],
            affected_outcomes=["all"],
            mitigation_possible=False,
            mitigation_strategy=None,
            mitigation_effectiveness=None,
        )

        result = record.to_dict()

        assert result["threat_id"] == "threat-123"
        assert result["threat_type"] == "external"
        assert isinstance(result["timestamp"], str)


# ============================================================================
# Test Memory Hooks Class
# ============================================================================


class TestExperimentDesignerMemoryHooks:
    """Test ExperimentDesignerMemoryHooks class."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    def test_create_hooks(self):
        """Test hooks creation."""
        hooks = ExperimentDesignerMemoryHooks()

        assert hooks is not None
        assert hooks._working_memory is None
        assert hooks._supabase_client is None

    def test_cache_ttl_constant(self):
        """Test cache TTL is set correctly."""
        assert ExperimentDesignerMemoryHooks.CACHE_TTL_SECONDS == 86400

    @pytest.mark.asyncio
    async def test_get_context_no_memory_clients(self):
        """Test get_context when no memory clients available."""
        hooks = ExperimentDesignerMemoryHooks()

        context = await hooks.get_context(
            session_id="test-session",
            business_question="Test question about rep visits",
        )

        assert context.session_id == "test-session"
        assert context.working_memory == []
        assert context.episodic_context == []

    @pytest.mark.asyncio
    async def test_cache_experiment_design_no_client(self):
        """Test caching returns False when no client."""
        hooks = ExperimentDesignerMemoryHooks()

        result = await hooks.cache_experiment_design(
            session_id="test-session",
            result={"design_type": "RCT"},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_cached_experiment_design_no_client(self):
        """Test cached retrieval returns None when no client."""
        hooks = ExperimentDesignerMemoryHooks()

        result = await hooks.get_cached_experiment_design("test-session")

        assert result is None

    @pytest.mark.asyncio
    async def test_store_experiment_design_no_client(self):
        """Test storage returns None when no client."""
        hooks = ExperimentDesignerMemoryHooks()

        result = await hooks.store_experiment_design(
            session_id="test-session",
            result={"design_type": "RCT", "overall_validity_score": 0.8},
            state={"business_question": "Test question"},
        )

        assert result is None

    def test_hash_question(self):
        """Test question hashing for cache keys."""
        hooks = ExperimentDesignerMemoryHooks()

        hash1 = hooks._hash_question("Test question?")
        hash2 = hooks._hash_question("test question?")  # Different case
        hash3 = hooks._hash_question("Different question")

        # Same question, different case should produce same hash
        assert hash1 == hash2
        # Different questions should produce different hashes
        assert hash1 != hash3

    def test_extract_keywords(self):
        """Test keyword extraction from business question."""
        hooks = ExperimentDesignerMemoryHooks()

        keywords = hooks._extract_keywords(
            "Does increasing rep visit frequency improve HCP engagement?"
        )

        assert "increasing" in keywords
        assert "rep" in keywords
        assert "visit" in keywords
        assert "frequency" in keywords
        assert "hcp" in keywords
        # Note: engagement may include trailing punctuation from the question
        assert any("engagement" in kw for kw in keywords)
        # Stopwords should be excluded
        assert "does" not in keywords
        assert "the" not in keywords


class TestExperimentDesignerMemoryHooksWithMocks:
    """Test memory hooks with mocked clients."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    @pytest.mark.asyncio
    async def test_cache_experiment_design_success(self):
        """Test successful caching with mocked Redis."""
        hooks = ExperimentDesignerMemoryHooks()

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        hooks._working_memory = mock_redis

        result = await hooks.cache_experiment_design(
            session_id="test-session",
            result={"design_type": "RCT", "overall_validity_score": 0.85},
            business_question="Does X affect Y?",
        )

        assert result is True
        # Session cache + question cache
        assert mock_redis.set.call_count == 2

    @pytest.mark.asyncio
    async def test_get_cached_experiment_design_success(self):
        """Test successful cache retrieval with mocked Redis."""
        hooks = ExperimentDesignerMemoryHooks()

        cached_data = {
            "result": {"design_type": "RCT"},
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=cached_data)
        hooks._working_memory = mock_redis

        result = await hooks.get_cached_experiment_design("test-session")

        assert result == {"design_type": "RCT"}
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_cache_success(self):
        """Test cache invalidation with mocked Redis."""
        hooks = ExperimentDesignerMemoryHooks()

        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=True)
        hooks._working_memory = mock_redis

        result = await hooks.invalidate_cache(
            session_id="test-session",
            business_question="Test question",
        )

        assert result is True
        assert mock_redis.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_store_experiment_design_success(self):
        """Test successful episodic storage with mocked Supabase."""
        hooks = ExperimentDesignerMemoryHooks()

        mock_response = MagicMock()
        mock_response.data = [{"id": 1}]
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = mock_response
        mock_supabase = MagicMock()
        mock_supabase.table.return_value = mock_table
        hooks._supabase_client = mock_supabase

        result = await hooks.store_experiment_design(
            session_id="test-session",
            result={
                "design_type": "RCT",
                "design_rationale": "Best for causal inference",
                "randomization_unit": "individual",
                "randomization_method": "stratified",
                "power_analysis": {
                    "required_sample_size": 500,
                    "achieved_power": 0.85,
                },
                "duration_estimate_days": 90,
                "overall_validity_score": 0.82,
                "validity_confidence": "high",
                "validity_threats": [
                    {"threat_type": "internal", "threat_name": "Attrition", "severity": "medium"}
                ],
                "treatments": [],
                "outcomes": [],
                "redesign_iterations": 1,
                "total_latency_ms": 45000,
                "warnings": [],
            },
            state={
                "business_question": "Does X improve Y?",
                "constraints": {"power": 0.80},
            },
            brand="Remibrutinib",
        )

        assert result is not None
        mock_supabase.table.assert_called_with("agent_activities")

    @pytest.mark.asyncio
    async def test_get_episodic_context(self):
        """Test episodic context retrieval."""
        hooks = ExperimentDesignerMemoryHooks()

        mock_response = MagicMock()
        mock_response.data = [
            {
                "metadata": {
                    "business_question": "Does rep visit improve engagement?",
                    "design_type": "RCT",
                }
            },
            {
                "metadata": {
                    "business_question": "Unrelated experiment question",
                    "design_type": "quasi_experiment",
                }
            },
        ]
        mock_query = MagicMock()
        mock_query.eq.return_value = mock_query
        mock_query.contains.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.execute.return_value = mock_response

        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value = mock_query
        hooks._supabase_client = mock_supabase

        context = await hooks._get_episodic_context(
            business_question="Does rep visit frequency affect engagement?",
            brand="Remibrutinib",
            design_type="RCT",
            max_records=5,
        )

        # Should return the matching record (keyword overlap)
        assert len(context) >= 1

    @pytest.mark.asyncio
    async def test_get_similar_validity_threats(self):
        """Test retrieving similar validity threats."""
        hooks = ExperimentDesignerMemoryHooks()

        mock_response = MagicMock()
        mock_response.data = [
            {
                "metadata": {
                    "design_type": "RCT",
                    "validity_threats": [
                        {"threat_type": "internal", "threat_name": "Attrition"},
                        {"threat_type": "external", "threat_name": "Generalizability"},
                    ],
                }
            },
            {
                "metadata": {
                    "design_type": "RCT",
                    "validity_threats": [
                        {"threat_type": "internal", "threat_name": "Attrition"},  # Duplicate
                        {"threat_type": "internal", "threat_name": "Selection Bias"},
                    ],
                }
            },
        ]
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.contains.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.execute.return_value = mock_response

        mock_supabase = MagicMock()
        mock_supabase.table.return_value = mock_query
        hooks._supabase_client = mock_supabase

        threats = await hooks.get_similar_validity_threats(
            design_type="RCT",
            max_threats=10,
        )

        # Should deduplicate threats
        assert len(threats) == 3  # Attrition, Generalizability, Selection Bias

    @pytest.mark.asyncio
    async def test_get_working_memory_context(self):
        """Test working memory context retrieval."""
        hooks = ExperimentDesignerMemoryHooks()

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(
            side_effect=[
                {"result": "session_cache"},
                {"result": "question_cache"},
            ]
        )
        hooks._working_memory = mock_redis

        context = await hooks._get_working_memory_context(
            session_id="test-session",
            business_question="Test question",
        )

        assert len(context) == 2


# ============================================================================
# Test Contribute to Memory Function
# ============================================================================


class TestContributeToMemory:
    """Test contribute_to_memory function."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    @pytest.mark.asyncio
    async def test_contribute_no_clients(self):
        """Test contribution when no memory clients available."""
        result = {
            "design_type": "RCT",
            "overall_validity_score": 0.8,
            "validity_threats": [],
            "power_analysis": {"required_sample_size": 100, "achieved_power": 0.8},
            "redesign_iterations": 0,
            "total_latency_ms": 30000,
            "warnings": [],
        }
        state = {
            "business_question": "Test question",
            "constraints": {},
        }

        counts = await contribute_to_memory(result, state, "test-session")

        # Without real clients, all counts should be 0
        assert counts["working"] == 0
        assert counts["episodic"] == 0

    @pytest.mark.asyncio
    async def test_contribute_with_mocked_clients(self):
        """Test contribution with mocked memory clients."""
        reset_memory_hooks()

        # Get and configure singleton
        hooks = get_experiment_designer_memory_hooks()

        # Mock working memory
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        hooks._working_memory = mock_redis

        # Mock supabase for episodic
        mock_response = MagicMock()
        mock_response.data = [{"id": 1}]
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = mock_response
        mock_supabase = MagicMock()
        mock_supabase.table.return_value = mock_table
        hooks._supabase_client = mock_supabase

        result = {
            "design_type": "RCT",
            "design_rationale": "Strongest causal evidence",
            "randomization_unit": "individual",
            "randomization_method": "stratified",
            "overall_validity_score": 0.85,
            "validity_confidence": "high",
            "validity_threats": [
                {"threat_type": "internal", "threat_name": "Attrition", "severity": "low"},
                {"threat_type": "external", "threat_name": "Generalizability", "severity": "medium"},
            ],
            "power_analysis": {
                "required_sample_size": 500,
                "achieved_power": 0.85,
            },
            "duration_estimate_days": 90,
            "treatments": [],
            "outcomes": [],
            "redesign_iterations": 1,
            "total_latency_ms": 45000,
            "warnings": [],
        }
        state = {
            "business_question": "Does increasing rep visits improve engagement?",
            "constraints": {"power": 0.80},
        }

        counts = await contribute_to_memory(
            result, state, "test-session", brand="Remibrutinib"
        )

        assert counts["working"] == 2  # Session + question caches
        assert counts["episodic"] >= 1  # Design record + threats


# ============================================================================
# Test Singleton Access
# ============================================================================


class TestSingletonAccess:
    """Test singleton access pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    def test_get_singleton(self):
        """Test getting singleton instance."""
        hooks1 = get_experiment_designer_memory_hooks()
        hooks2 = get_experiment_designer_memory_hooks()

        assert hooks1 is hooks2

    def test_reset_singleton(self):
        """Test resetting singleton."""
        hooks1 = get_experiment_designer_memory_hooks()
        reset_memory_hooks()
        hooks2 = get_experiment_designer_memory_hooks()

        assert hooks1 is not hooks2
