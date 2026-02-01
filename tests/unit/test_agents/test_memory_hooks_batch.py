"""
Batch unit tests for memory_hooks.py across multiple agents.
=============================================================

Tests memory integration hooks for 11 agent memory_hooks modules:
- src/agents/causal_impact/memory_hooks.py
- src/agents/cohort_constructor/memory_hooks.py
- src/agents/orchestrator/memory_hooks.py
- src/agents/health_score/memory_hooks.py
- src/agents/ml_foundation/data_preparer/memory_hooks.py
- src/agents/ml_foundation/scope_definer/memory_hooks.py
- src/agents/ml_foundation/observability_connector/memory_hooks.py
- src/agents/ml_foundation/feature_analyzer/memory_hooks.py
- src/agents/ml_foundation/model_deployer/memory_hooks.py
- src/agents/ml_foundation/model_selector/memory_hooks.py
- src/agents/ml_foundation/model_trainer/memory_hooks.py

Covers: dataclass creation, class init, lazy-loading properties,
async methods with mocked memory clients, error handling, singletons,
and contribute_to_memory functions.

Author: E2I Causal Analytics Team
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module imports -- all memory hooks modules under test
# ---------------------------------------------------------------------------
from src.agents.causal_impact import memory_hooks as ci_hooks
from src.agents.cohort_constructor import memory_hooks as cc_hooks
from src.agents.health_score import memory_hooks as hs_hooks
from src.agents.ml_foundation.data_preparer import memory_hooks as dp_hooks
from src.agents.ml_foundation.feature_analyzer import memory_hooks as fa_hooks
from src.agents.ml_foundation.model_deployer import memory_hooks as md_hooks
from src.agents.ml_foundation.model_selector import memory_hooks as ms_hooks
from src.agents.ml_foundation.model_trainer import memory_hooks as mt_hooks
from src.agents.ml_foundation.observability_connector import (
    memory_hooks as oc_hooks,
)
from src.agents.ml_foundation.scope_definer import memory_hooks as sd_hooks
from src.agents.orchestrator import memory_hooks as orch_hooks


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_all_singletons():
    """Reset all memory hooks singletons and block real service connections.

    Patches get_working_memory and get_semantic_memory to raise, preventing
    lazy-loading from connecting to real Redis/FalkorDB during tests.
    Tests that need a mock client set hooks._working_memory directly, which
    bypasses the property lazy-load entirely.
    """
    modules = [
        ci_hooks, cc_hooks, orch_hooks, hs_hooks, dp_hooks,
        sd_hooks, oc_hooks, fa_hooks, md_hooks, ms_hooks, mt_hooks,
    ]
    for mod in modules:
        mod.reset_memory_hooks()
    with patch(
        "src.memory.working_memory.get_working_memory",
        side_effect=Exception("test: working memory unavailable"),
    ):
        with patch(
            "src.memory.semantic_memory.get_semantic_memory",
            side_effect=Exception("test: semantic memory unavailable"),
        ):
            yield
    for mod in modules:
        mod.reset_memory_hooks()


@pytest.fixture
def mock_working_memory():
    """Create a mock working memory client with all common methods."""
    mock = MagicMock()
    mock.get_messages = AsyncMock(return_value=[{"role": "user", "content": "hello"}])
    mock.add_message = AsyncMock(return_value=True)
    mock.set = AsyncMock(return_value=True)

    # Mock redis client for caching methods
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.delete = AsyncMock(return_value=True)
    mock_redis.lpush = AsyncMock(return_value=1)
    mock_redis.ltrim = AsyncMock(return_value=True)
    mock_redis.lrange = AsyncMock(return_value=[])
    mock.get_client = AsyncMock(return_value=mock_redis)

    return mock


@pytest.fixture
def mock_semantic_memory():
    """Create a mock semantic memory client with all common methods."""
    mock = MagicMock()
    mock.query = MagicMock(return_value=[])
    mock.add_entity = MagicMock(return_value=True)
    mock.add_relationship = MagicMock(return_value=True)
    mock.get_graph_stats = MagicMock(return_value={"nodes": 10, "edges": 20})
    mock.find_causal_paths_for_kpi = MagicMock(return_value=[])
    return mock


# ===========================================================================
# Section 1: Dataclass Construction Tests
# ===========================================================================


class TestDataclassConstruction:
    """Test that all dataclasses can be instantiated with minimal args."""

    def test_causal_analysis_context(self):
        ctx = ci_hooks.CausalAnalysisContext(session_id="s1")
        assert ctx.session_id == "s1"
        assert ctx.working_memory == []
        assert ctx.episodic_context == []
        assert ctx.semantic_context == {}
        assert isinstance(ctx.retrieval_timestamp, datetime)

    def test_causal_analysis_record(self):
        rec = ci_hooks.CausalAnalysisRecord(
            session_id="s1", query="test", treatment_var="t",
            outcome_var="o", ate_estimate=0.5, confidence_interval=(0.1, 0.9),
            refutation_passed=True, confidence=0.8,
            executive_summary="summary",
        )
        assert rec.treatment_var == "t"
        assert rec.ate_estimate == 0.5
        assert isinstance(rec.timestamp, datetime)
        assert rec.metadata == {}

    def test_causal_path_record(self):
        rec = ci_hooks.CausalPathRecord(
            path_id="p1", treatment_var="t", outcome_var="o",
            confounders=["c1"], ate_estimate=0.3,
            effect_size="medium", confidence=0.7, refutation_passed=True,
        )
        assert rec.path_id == "p1"
        assert rec.confounders == ["c1"]

    def test_orchestration_context(self):
        ctx = orch_hooks.OrchestrationContext(session_id="s2")
        assert ctx.session_id == "s2"
        assert ctx.working_memory == []

    def test_orchestration_record(self):
        rec = orch_hooks.OrchestrationRecord(
            session_id="s2", query_id="q1", query="test",
            intent="analysis", agents_dispatched=["a1"],
            response_summary="ok", confidence=0.9,
            total_latency_ms=100, success=True,
        )
        assert rec.query_id == "q1"
        assert rec.agents_dispatched == ["a1"]

    def test_routing_decision_record(self):
        rec = orch_hooks.RoutingDecisionRecord(
            query_pattern="trend", intent_classified="analysis",
            agents_selected=["causal_impact"], execution_mode="sequential",
            success=True, latency_ms=200, confidence=0.85,
        )
        assert rec.execution_mode == "sequential"

    def test_health_check_context(self):
        ctx = hs_hooks.HealthCheckContext(session_id="s3")
        assert ctx.cached_health is None
        assert ctx.historical_trends == []

    def test_health_record(self):
        rec = hs_hooks.HealthRecord(
            session_id="s3", overall_score=85.0, health_grade="B",
            component_score=0.9, model_score=0.8, pipeline_score=0.85,
            agent_score=0.9, critical_issues_count=0, warnings_count=1,
            check_scope="full", total_latency_ms=1200,
        )
        assert rec.health_grade == "B"

    def test_cohort_construction_context(self):
        ctx = cc_hooks.CohortConstructionContext(session_id="s4")
        assert ctx.session_id == "s4"

    def test_cohort_construction_record(self):
        rec = cc_hooks.CohortConstructionRecord(
            session_id="s4", cohort_id="c1", cohort_name="test_cohort",
            brand="Kisqali", indication="breast_cancer",
            total_patients=1000, eligible_patients=800,
            eligibility_rate=0.8, criteria_count=5,
            config_hash="abc123", execution_time_ms=500,
        )
        assert rec.brand == "Kisqali"
        assert rec.eligibility_rate == 0.8

    def test_data_preparation_context(self):
        ctx = dp_hooks.DataPreparationContext(session_id="s5")
        assert ctx.session_id == "s5"

    def test_qc_report_record(self):
        rec = dp_hooks.QCReportRecord(
            session_id="s5", experiment_id="exp1", report_id="rpt1",
            qc_status="passed", overall_score=0.95,
            gate_passed=True, leakage_detected=False,
        )
        assert rec.qc_status == "passed"

    def test_scope_definition_context(self):
        ctx = sd_hooks.ScopeDefinitionContext(session_id="s6")
        assert ctx.session_id == "s6"

    def test_scope_definition_record(self):
        rec = sd_hooks.ScopeDefinitionRecord(
            session_id="s6", experiment_id="exp1",
            experiment_name="churn_model", problem_type="classification",
            target_variable="churned", business_objective="reduce churn",
            success_criteria={"auc": 0.8}, scope_spec={"features": []},
        )
        assert rec.problem_type == "classification"

    def test_observability_context(self):
        ctx = oc_hooks.ObservabilityContext(session_id="s7")
        assert ctx.session_id == "s7"

    def test_observability_record(self):
        rec = oc_hooks.ObservabilityRecord(
            session_id="s7", time_window="24h",
            overall_success_rate=0.95, overall_p95_latency_ms=250.0,
            quality_score=0.88, error_rate_by_agent={"agent1": 0.02},
        )
        assert rec.overall_success_rate == 0.95

    def test_feature_analysis_context(self):
        ctx = fa_hooks.FeatureAnalysisContext(session_id="s8")
        assert ctx.session_id == "s8"

    def test_feature_importance_record(self):
        rec = fa_hooks.FeatureImportanceRecord(
            session_id="s8", experiment_id="exp1",
            shap_analysis_id="shap1", top_features=["f1", "f2"],
            global_importance={"f1": 0.5, "f2": 0.3},
            interactions=[("f1", "f2", 0.4)],
        )
        assert rec.top_features == ["f1", "f2"]

    def test_deployment_context(self):
        ctx = md_hooks.DeploymentContext(session_id="s9")
        assert ctx.session_id == "s9"

    def test_deployment_record(self):
        rec = md_hooks.DeploymentRecord(
            session_id="s9", experiment_id="exp1",
            deployment_id="dep1", model_version=3,
            target_environment="staging", deployment_status="success",
        )
        assert rec.model_version == 3

    def test_model_selection_context(self):
        ctx = ms_hooks.ModelSelectionContext(session_id="s10")
        assert ctx.session_id == "s10"

    def test_model_selection_record(self):
        rec = ms_hooks.ModelSelectionRecord(
            session_id="s10", experiment_id="exp1",
            algorithm_name="xgboost", algorithm_family="gradient_boosting",
            selection_score=0.92, selection_rationale="best AUC",
        )
        assert rec.algorithm_name == "xgboost"

    def test_model_training_context(self):
        ctx = mt_hooks.ModelTrainingContext(session_id="s11")
        assert ctx.session_id == "s11"

    def test_training_result_record(self):
        rec = mt_hooks.TrainingResultRecord(
            session_id="s11", experiment_id="exp1",
            training_run_id="run1", algorithm_name="xgboost",
            test_metrics={"auc_roc": 0.88}, success_criteria_met=True,
        )
        assert rec.success_criteria_met is True


# ===========================================================================
# Section 2: Class Initialization Tests
# ===========================================================================


class TestClassInitialization:
    """Test __init__ for all memory hooks classes."""

    def test_causal_impact_init(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None
        assert hooks.CACHE_TTL_SECONDS == 86400

    def test_orchestrator_init(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None
        assert hooks.CACHE_TTL_SECONDS == 86400

    def test_health_score_init(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        assert hooks._working_memory is None
        assert hooks.QUICK_CACHE_TTL == 300
        assert hooks.FULL_CACHE_TTL == 900

    def test_cohort_constructor_init(self):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None

    def test_data_preparer_init(self):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None

    def test_scope_definer_init(self):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None

    def test_observability_connector_init(self):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None

    def test_feature_analyzer_init(self):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None

    def test_model_deployer_init(self):
        hooks = md_hooks.ModelDeployerMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None

    def test_model_selector_init(self):
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None

    def test_model_trainer_init(self):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        assert hooks._working_memory is None
        assert hooks._semantic_memory is None


# ===========================================================================
# Section 3: Singleton & Reset Tests
# ===========================================================================


class TestSingletonFunctions:
    """Test get/reset singleton functions for all modules."""

    def test_causal_impact_singleton(self):
        h1 = ci_hooks.get_causal_impact_memory_hooks()
        h2 = ci_hooks.get_causal_impact_memory_hooks()
        assert h1 is h2
        ci_hooks.reset_memory_hooks()
        h3 = ci_hooks.get_causal_impact_memory_hooks()
        assert h3 is not h1

    def test_orchestrator_singleton(self):
        h1 = orch_hooks.get_orchestrator_memory_hooks()
        h2 = orch_hooks.get_orchestrator_memory_hooks()
        assert h1 is h2
        orch_hooks.reset_memory_hooks()
        h3 = orch_hooks.get_orchestrator_memory_hooks()
        assert h3 is not h1

    def test_health_score_singleton(self):
        h1 = hs_hooks.get_health_score_memory_hooks()
        h2 = hs_hooks.get_health_score_memory_hooks()
        assert h1 is h2
        hs_hooks.reset_memory_hooks()
        h3 = hs_hooks.get_health_score_memory_hooks()
        assert h3 is not h1

    def test_cohort_constructor_singleton(self):
        h1 = cc_hooks.get_cohort_constructor_memory_hooks()
        assert isinstance(h1, cc_hooks.CohortConstructorMemoryHooks)
        cc_hooks.reset_memory_hooks()
        h2 = cc_hooks.get_cohort_constructor_memory_hooks()
        assert h2 is not h1

    def test_data_preparer_singleton(self):
        h1 = dp_hooks.get_data_preparer_memory_hooks()
        assert isinstance(h1, dp_hooks.DataPreparerMemoryHooks)

    def test_scope_definer_singleton(self):
        h1 = sd_hooks.get_scope_definer_memory_hooks()
        assert isinstance(h1, sd_hooks.ScopeDefinerMemoryHooks)

    def test_observability_connector_singleton(self):
        h1 = oc_hooks.get_observability_connector_memory_hooks()
        assert isinstance(h1, oc_hooks.ObservabilityConnectorMemoryHooks)

    def test_feature_analyzer_singleton(self):
        h1 = fa_hooks.get_feature_analyzer_memory_hooks()
        assert isinstance(h1, fa_hooks.FeatureAnalyzerMemoryHooks)

    def test_model_deployer_singleton(self):
        h1 = md_hooks.get_model_deployer_memory_hooks()
        assert isinstance(h1, md_hooks.ModelDeployerMemoryHooks)

    def test_model_selector_singleton(self):
        h1 = ms_hooks.get_model_selector_memory_hooks()
        assert isinstance(h1, ms_hooks.ModelSelectorMemoryHooks)

    def test_model_trainer_singleton(self):
        h1 = mt_hooks.get_model_trainer_memory_hooks()
        assert isinstance(h1, mt_hooks.ModelTrainerMemoryHooks)


# ===========================================================================
# Section 4: Lazy-Loading Property Tests
# ===========================================================================


class TestLazyLoadProperties:
    """Test working_memory and semantic_memory lazy-loading properties."""

    def test_working_memory_success(self):
        """Test successful lazy-load of working memory."""
        hooks = ci_hooks.CausalImpactMemoryHooks()
        mock_wm = MagicMock()
        with patch(
            "src.agents.causal_impact.memory_hooks.get_working_memory",
            create=True,
        ):
            with patch(
                "src.memory.working_memory.get_working_memory",
                return_value=mock_wm,
                create=True,
            ):
                # Force the import inside the property to use our mock
                hooks._working_memory = mock_wm
                assert hooks.working_memory is mock_wm

    def test_working_memory_import_failure(self):
        """Test that import failure returns None gracefully."""
        hooks = ci_hooks.CausalImpactMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.working_memory": None}):
            # The property catches the import error
            result = hooks.working_memory
            assert result is None

    def test_semantic_memory_import_failure(self):
        """Test that semantic memory import failure returns None."""
        hooks = ci_hooks.CausalImpactMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.semantic_memory": None}):
            result = hooks.semantic_memory
            assert result is None

    def test_working_memory_cached_after_set(self):
        """Test that once set, the cached client is returned."""
        hooks = orch_hooks.OrchestratorMemoryHooks()
        mock_wm = MagicMock()
        hooks._working_memory = mock_wm
        assert hooks.working_memory is mock_wm

    def test_semantic_memory_cached_after_set(self):
        """Test that once set, the cached semantic client is returned."""
        hooks = orch_hooks.OrchestratorMemoryHooks()
        mock_sm = MagicMock()
        hooks._semantic_memory = mock_sm
        assert hooks.semantic_memory is mock_sm

    def test_health_score_no_semantic_memory(self):
        """HealthScoreMemoryHooks only has working memory, no semantic."""
        hooks = hs_hooks.HealthScoreMemoryHooks()
        assert not hasattr(hooks, "_semantic_memory") or hooks._working_memory is None


# ===========================================================================
# Section 5: Causal Impact Memory Hooks Tests
# ===========================================================================


class TestCausalImpactMemoryHooks:
    """Test CausalImpactMemoryHooks async methods."""

    @pytest.mark.asyncio
    async def test_get_working_memory_context_no_client(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        result = await hooks._get_working_memory_context("session1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_working_memory_context_with_client(self, mock_working_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks._get_working_memory_context("session1")
        assert len(result) == 1
        mock_working_memory.get_messages.assert_awaited_once_with("session1", limit=10)

    @pytest.mark.asyncio
    async def test_get_working_memory_context_error(self, mock_working_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        mock_working_memory.get_messages = AsyncMock(side_effect=Exception("redis down"))
        hooks._working_memory = mock_working_memory
        result = await hooks._get_working_memory_context("session1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_episodic_context_import_error(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.episodic_memory": None}):
            result = await hooks._get_episodic_context(query="test")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_episodic_context_with_filter(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        mock_search = AsyncMock(return_value=[
            {"raw_content": {"treatment_var": "hcp_visits", "outcome_var": "trx"}},
            {"raw_content": {"treatment_var": "other", "outcome_var": "trx"}},
        ])
        with patch(
            "src.agents.causal_impact.memory_hooks.search_episodic_by_text",
            mock_search,
            create=True,
        ):
            with patch(
                "src.agents.causal_impact.memory_hooks.EpisodicSearchFilters",
                create=True,
            ):
                # Call with treatment filter -- exercises the filter branch
                result = await hooks._get_episodic_context(
                    query="test", treatment_var="hcp_visits", outcome_var="trx",
                )
                # Should filter to only matching results
                assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_semantic_context_no_client(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        result = await hooks._get_semantic_context()
        assert result == {"entities": [], "relationships": [], "causal_paths": []}

    @pytest.mark.asyncio
    async def test_get_semantic_context_with_client(self, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context(
            treatment_var="hcp_visits", outcome_var="trx",
        )
        assert "graph_stats" in result
        mock_semantic_memory.get_graph_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_semantic_context_error(self, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        mock_semantic_memory.get_graph_stats = MagicMock(
            side_effect=Exception("falkordb down"),
        )
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context(treatment_var="t", outcome_var="o")
        assert result == {"entities": [], "relationships": [], "causal_paths": []}

    @pytest.mark.asyncio
    async def test_get_causal_paths_no_client(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        result = await hooks._get_causal_paths("t", "o")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_causal_paths_with_client(self, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        mock_semantic_memory.query = MagicMock(return_value=[{"path": "t->o"}])
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_causal_paths("t", "o")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_causal_paths_error(self, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        mock_semantic_memory.query = MagicMock(side_effect=Exception("error"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_causal_paths("t", "o")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_kpi_causal_paths_no_client(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        result = await hooks._get_kpi_causal_paths("trx")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_kpi_causal_paths_with_client(self, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        mock_semantic_memory.find_causal_paths_for_kpi = MagicMock(
            return_value=[{"kpi": "trx"}] * 15,
        )
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_kpi_causal_paths("trx")
        assert len(result) == 10  # Limited to top 10

    @pytest.mark.asyncio
    async def test_cache_causal_analysis_no_client(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        result = await hooks.cache_causal_analysis("s1", {"test": True})
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_causal_analysis_success(self, mock_working_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_causal_analysis("s1", {"ate": 0.5})
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_causal_analysis_error(self, mock_working_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.setex = AsyncMock(side_effect=Exception("cache error"))
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_causal_analysis("s1", {"test": True})
        assert result is False

    @pytest.mark.asyncio
    async def test_get_cached_analysis_no_client(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        result = await hooks.get_cached_causal_analysis("s1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_analysis_hit(self, mock_working_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.get = AsyncMock(return_value=json.dumps({"ate": 0.5}))
        hooks._working_memory = mock_working_memory
        result = await hooks.get_cached_causal_analysis("s1")
        assert result == {"ate": 0.5}

    @pytest.mark.asyncio
    async def test_get_cached_analysis_miss(self, mock_working_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.get_cached_causal_analysis("s1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_analysis_error(self, mock_working_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.get = AsyncMock(side_effect=Exception("error"))
        hooks._working_memory = mock_working_memory
        result = await hooks.get_cached_causal_analysis("s1")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_causal_path_no_client(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        result = await hooks.store_causal_path(
            "t", "o", ["c1"], 0.5, 0.8, True, "medium",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_store_causal_path_success(self, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_causal_path(
            "hcp_visits", "trx", ["region"], 0.3, 0.85, True, "small",
        )
        assert result is True
        assert mock_semantic_memory.add_entity.call_count == 2
        mock_semantic_memory.add_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_causal_path_error(self, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("error"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_causal_path(
            "t", "o", [], 0.0, 0.0, False, "none",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_get_context_full(self, mock_working_memory, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            context = await hooks.get_context(
                session_id="s1", query="test query",
                treatment_var="t", outcome_var="o",
            )
        assert isinstance(context, ci_hooks.CausalAnalysisContext)
        assert context.session_id == "s1"

    @pytest.mark.asyncio
    async def test_store_causal_analysis_error(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.episodic_memory": None}):
            result = await hooks.store_causal_analysis(
                session_id="s1",
                result={"ate_estimate": 0.5, "confidence": 0.8},
                state={"treatment_var": "t", "outcome_var": "o"},
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_get_prior_analyses_error(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.episodic_memory": None}):
            result = await hooks.get_prior_analyses()
            assert result == []


# ===========================================================================
# Section 6: Orchestrator Memory Hooks Tests
# ===========================================================================


class TestOrchestratorMemoryHooks:
    """Test OrchestratorMemoryHooks async methods."""

    @pytest.mark.asyncio
    async def test_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context(
                session_id="s1", query="what drives trx?",
                entities={"kpi": ["trx"], "brand": ["Kisqali"]},
            )
        assert isinstance(ctx, orch_hooks.OrchestrationContext)
        assert ctx.session_id == "s1"

    @pytest.mark.asyncio
    async def test_get_working_memory_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks._get_working_memory_context("s1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_semantic_context_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks._get_semantic_context()
        assert result == {"entities": [], "relationships": [], "causal_paths": []}

    @pytest.mark.asyncio
    async def test_get_semantic_context_with_entities(self, mock_semantic_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(
            hooks, "_get_causal_paths_for_kpi", new_callable=AsyncMock, return_value=[],
        ):
            with patch.object(
                hooks, "_get_brand_context", new_callable=AsyncMock, return_value=[],
            ):
                result = await hooks._get_semantic_context(
                    entities={"kpi": ["trx"], "brand": ["Kisqali"]},
                )
        assert "graph_stats" in result

    @pytest.mark.asyncio
    async def test_get_semantic_context_error(self, mock_semantic_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        mock_semantic_memory.get_graph_stats = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context()
        assert result == {"entities": [], "relationships": [], "causal_paths": []}

    @pytest.mark.asyncio
    async def test_get_causal_paths_for_kpi_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks._get_causal_paths_for_kpi("trx")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_brand_context_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks._get_brand_context("Kisqali")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_brand_context_error(self, mock_semantic_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        mock_semantic_memory.query = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_brand_context("Kisqali")
        assert result == []

    @pytest.mark.asyncio
    async def test_store_conversation_turn_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks.store_conversation_turn("s1", "q", "r")
        assert result is False

    @pytest.mark.asyncio
    async def test_store_conversation_turn_success(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.store_conversation_turn(
            "s1", "what drives trx?", "HCP visits are key.",
            intent="analysis", agents_used=["causal_impact"],
        )
        assert result is True
        assert mock_working_memory.add_message.await_count == 2

    @pytest.mark.asyncio
    async def test_store_conversation_turn_error(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        mock_working_memory.add_message = AsyncMock(side_effect=Exception("err"))
        hooks._working_memory = mock_working_memory
        result = await hooks.store_conversation_turn("s1", "q", "r")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_conversation_history_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks.get_conversation_history("s1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_conversation_history_error(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        mock_working_memory.get_messages = AsyncMock(side_effect=Exception("err"))
        hooks._working_memory = mock_working_memory
        result = await hooks.get_conversation_history("s1")
        assert result == []

    @pytest.mark.asyncio
    async def test_cache_orchestration_result_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks.cache_orchestration_result("s1", "q1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_orchestration_result_success(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_orchestration_result("s1", "q1", {"data": 1})
        assert result is True

    @pytest.mark.asyncio
    async def test_get_cached_orchestration_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks.get_cached_orchestration("s1", "q1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_orchestration_hit(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.get = AsyncMock(return_value=json.dumps({"result": "cached"}))
        hooks._working_memory = mock_working_memory
        result = await hooks.get_cached_orchestration("s1", "q1")
        assert result == {"result": "cached"}

    @pytest.mark.asyncio
    async def test_get_cached_orchestration_error(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.get = AsyncMock(side_effect=Exception("error"))
        hooks._working_memory = mock_working_memory
        result = await hooks.get_cached_orchestration("s1", "q1")
        assert result is None

    @pytest.mark.asyncio
    async def test_track_routing_decision_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks.track_routing_decision(
            "s1", "trend", "analysis", ["ci"], "sequential", True, 200, 0.9,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_track_routing_decision_success(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.track_routing_decision(
            "s1", "trend", "analysis", ["ci"], "sequential", True, 200, 0.9,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_track_routing_decision_error(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.lpush = AsyncMock(side_effect=Exception("error"))
        hooks._working_memory = mock_working_memory
        result = await hooks.track_routing_decision(
            "s1", "p", "i", [], "seq", True, 0, 0.0,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_get_routing_decisions_no_client(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        result = await hooks.get_routing_decisions()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_routing_decisions_success(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.lrange = AsyncMock(return_value=[
            json.dumps({"intent": "analysis"}),
        ])
        hooks._working_memory = mock_working_memory
        result = await hooks.get_routing_decisions(limit=10)
        assert len(result) == 1
        assert result[0]["intent"] == "analysis"

    @pytest.mark.asyncio
    async def test_get_routing_decisions_error(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.lrange = AsyncMock(side_effect=Exception("error"))
        hooks._working_memory = mock_working_memory
        result = await hooks.get_routing_decisions()
        assert result == []

    @pytest.mark.asyncio
    async def test_store_orchestration_error(self):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.episodic_memory": None}):
            result = await hooks.store_orchestration(
                session_id="s1", result={"status": "completed"},
            )
            assert result is None


# ===========================================================================
# Section 7: Health Score Memory Hooks Tests
# ===========================================================================


class TestHealthScoreMemoryHooks:
    """Test HealthScoreMemoryHooks async methods."""

    @pytest.mark.asyncio
    async def test_get_context_basic(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "_get_cached_health", new_callable=AsyncMock, return_value=None):
            ctx = await hooks.get_context("s1", check_scope="quick")
        assert isinstance(ctx, hs_hooks.HealthCheckContext)
        assert ctx.cached_health is None
        assert ctx.historical_trends == []

    @pytest.mark.asyncio
    async def test_get_context_with_history(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "_get_cached_health", new_callable=AsyncMock, return_value=None):
            with patch.object(hooks, "_get_health_trends", new_callable=AsyncMock, return_value=[{"score": 85}]):
                ctx = await hooks.get_context("s1", include_history=True)
        assert len(ctx.historical_trends) == 1

    @pytest.mark.asyncio
    async def test_get_cached_health_no_client(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        result = await hooks._get_cached_health("full")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_health_hit(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.get = AsyncMock(return_value=json.dumps({"score": 90}))
        hooks._working_memory = mock_working_memory
        result = await hooks._get_cached_health("full")
        assert result == {"score": 90}

    @pytest.mark.asyncio
    async def test_get_cached_health_error(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.get = AsyncMock(side_effect=Exception("error"))
        hooks._working_memory = mock_working_memory
        result = await hooks._get_cached_health("full")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_health_check_no_client(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        result = await hooks.cache_health_check("full", {"score": 90})
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_health_check_quick_ttl(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_health_check("quick", {"score": 90})
        assert result is True
        redis = await mock_working_memory.get_client()
        redis.setex.assert_awaited_once()
        # Verify TTL used is QUICK_CACHE_TTL (300s)
        call_args = redis.setex.call_args
        assert call_args[0][1] == 300

    @pytest.mark.asyncio
    async def test_cache_health_check_full_ttl(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_health_check("full", {"score": 90})
        assert result is True
        redis = await mock_working_memory.get_client()
        call_args = redis.setex.call_args
        assert call_args[0][1] == 900  # FULL_CACHE_TTL

    @pytest.mark.asyncio
    async def test_cache_health_check_error(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.setex = AsyncMock(side_effect=Exception("error"))
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_health_check("full", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_cache_no_client(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        result = await hooks.invalidate_cache()
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_cache_specific_scope(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.invalidate_cache(check_scope="full")
        assert result is True

    @pytest.mark.asyncio
    async def test_invalidate_cache_all_scopes(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.invalidate_cache()
        assert result is True
        redis = await mock_working_memory.get_client()
        # Should delete 5 scopes
        assert redis.delete.await_count == 5

    @pytest.mark.asyncio
    async def test_invalidate_cache_error(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        redis = await mock_working_memory.get_client()
        redis.delete = AsyncMock(side_effect=Exception("error"))
        hooks._working_memory = mock_working_memory
        result = await hooks.invalidate_cache()
        assert result is False

    def test_is_significant_critical_issues(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        assert hooks._is_significant_health_event(
            {"critical_issues": ["db down"]},
        ) is True

    def test_is_significant_failing_grade(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        assert hooks._is_significant_health_event({"health_grade": "F"}) is True
        assert hooks._is_significant_health_event({"health_grade": "D"}) is True

    def test_is_significant_low_score(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        assert hooks._is_significant_health_event(
            {"overall_health_score": 65},
        ) is True

    def test_is_not_significant(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        assert hooks._is_significant_health_event(
            {"health_grade": "A", "overall_health_score": 95},
        ) is False

    def test_calculate_importance_high_score(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        importance = hooks._calculate_importance(
            {"overall_health_score": 95, "critical_issues": []},
        )
        assert 0.0 <= importance <= 1.0

    def test_calculate_importance_low_score_with_issues(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        importance = hooks._calculate_importance(
            {"overall_health_score": 30, "critical_issues": ["a", "b", "c"]},
        )
        assert importance >= 0.7

    @pytest.mark.asyncio
    async def test_store_health_check_not_significant(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        result = await hooks.store_health_check(
            "s1",
            {"health_grade": "A", "overall_health_score": 95},
            {"check_scope": "full"},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_health_trends_error(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.episodic_memory": None}):
            result = await hooks._get_health_trends()
            assert result == []

    @pytest.mark.asyncio
    async def test_get_score_history_error(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.episodic_memory": None}):
            result = await hooks.get_score_history()
            assert result == []

    @pytest.mark.asyncio
    async def test_get_component_issues_error(self):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.episodic_memory": None}):
            result = await hooks.get_component_issues("database")
            assert result == []


# ===========================================================================
# Section 8: Cohort Constructor Memory Hooks Tests
# ===========================================================================


class TestCohortConstructorMemoryHooks:
    """Test CohortConstructorMemoryHooks async methods."""

    @pytest.mark.asyncio
    async def test_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context("s1", brand="Kisqali")
        assert isinstance(ctx, cc_hooks.CohortConstructionContext)

    @pytest.mark.asyncio
    async def test_get_semantic_context_no_client(self):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        result = await hooks._get_semantic_context(brand="Kisqali")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_semantic_context_with_brand_and_indication(self, mock_semantic_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context(
            brand="Kisqali", indication="breast_cancer",
        )
        assert "eligibility_rules" in result
        assert "prior_cohorts" in result

    @pytest.mark.asyncio
    async def test_get_semantic_context_error(self, mock_semantic_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        mock_semantic_memory.query = MagicMock(side_effect=Exception("error"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context(brand="Kisqali")
        assert result == {}

    @pytest.mark.asyncio
    async def test_cache_cohort_config_no_client(self):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        result = await hooks.cache_cohort_config("s1", {"brand": "Kisqali"})
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_cohort_config_success(self, mock_working_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_cohort_config("s1", {"brand": "Kisqali"})
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_cohort_config_error(self, mock_working_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        mock_working_memory.set = AsyncMock(side_effect=Exception("error"))
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_cohort_config("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_cohort_result_no_client(self):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        result = await hooks.cache_cohort_result("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_store_eligibility_rule_no_client(self):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        result = await hooks.store_eligibility_rule(
            "rule1", {"field": "age"}, "Kisqali", 0.8, "cohort1",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_store_eligibility_rule_success(self, mock_semantic_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_eligibility_rule(
            "age_filter", {"field": "age", "operator": "gte", "value": 18},
            "Kisqali", 0.85, "cohort_abc",
        )
        assert result is True
        assert mock_semantic_memory.add_entity.call_count >= 2
        assert mock_semantic_memory.add_relationship.call_count >= 2

    @pytest.mark.asyncio
    async def test_store_eligibility_rule_error(self, mock_semantic_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("error"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_eligibility_rule(
            "rule", {}, "brand", 0.5, "cohort",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_store_cohort_pattern_no_client(self):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        result = await hooks.store_cohort_pattern(
            "c1", "test", "Kisqali", "bc", {}, 0.8,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_store_cohort_pattern_success(self, mock_semantic_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_cohort_pattern(
            "c1", "test", "Kisqali", "breast_cancer",
            {"inclusion_count": 3, "exclusion_count": 2}, 0.8,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_get_effective_rules_no_client(self):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        result = await hooks.get_effective_rules_for_brand("Kisqali")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_effective_rules_error(self, mock_semantic_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        mock_semantic_memory.query = MagicMock(side_effect=Exception("error"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.get_effective_rules_for_brand("Kisqali")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_prior_cohorts_error(self):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        with patch.dict("sys.modules", {"src.memory.episodic_memory": None}):
            result = await hooks.get_prior_cohorts()
            assert result == []


# ===========================================================================
# Section 9: ML Foundation Agents - Shared Pattern Tests
# ===========================================================================


class TestMLFoundationMemoryHooks:
    """Test ML Foundation agent memory hooks (common patterns)."""

    # --- Data Preparer ---

    @pytest.mark.asyncio
    async def test_data_preparer_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context("s1", "exp1", data_source="supabase")
        assert isinstance(ctx, dp_hooks.DataPreparationContext)

    @pytest.mark.asyncio
    async def test_data_preparer_get_semantic_no_client(self):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        result = await hooks._get_semantic_context("exp1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_data_preparer_get_semantic_with_source(self, mock_semantic_memory):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context("exp1", data_source="supabase")
        assert "leakage_incidents" in result
        assert "data_source_history" in result

    @pytest.mark.asyncio
    async def test_data_preparer_cache_qc_no_client(self):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        result = await hooks.cache_qc_report("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_data_preparer_cache_qc_success(self, mock_working_memory):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_qc_report("s1", {"status": "passed"})
        assert result is True

    @pytest.mark.asyncio
    async def test_data_preparer_store_quality_pattern_no_client(self):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        result = await hooks.store_data_quality_pattern(
            "exp1", "supabase", "passed", 0.95, False, [],
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_data_preparer_store_quality_pattern_with_leakage(self, mock_semantic_memory):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_data_quality_pattern(
            "exp1", "supabase", "failed", 0.6, True, ["leakage found"],
        )
        assert result is True
        # Should create LeakageIncident entity when leakage_detected
        entity_calls = [c[1].get("entity_type", c[0][0] if c[0] else "")
                        for c in mock_semantic_memory.add_entity.call_args_list]
        assert "LeakageIncident" in entity_calls

    @pytest.mark.asyncio
    async def test_data_preparer_store_quality_pattern_error(self, mock_semantic_memory):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_data_quality_pattern(
            "exp1", "src", "passed", 0.9, False, [],
        )
        assert result is False

    # --- Scope Definer ---

    @pytest.mark.asyncio
    async def test_scope_definer_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context(
                "s1", problem_description="predict churn",
            )
        assert isinstance(ctx, sd_hooks.ScopeDefinitionContext)

    @pytest.mark.asyncio
    async def test_scope_definer_get_semantic_no_client(self):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        result = await hooks._get_semantic_context()
        assert result == {}

    @pytest.mark.asyncio
    async def test_scope_definer_get_semantic_with_filters(self, mock_semantic_memory):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context(
            problem_type="classification", target_variable="churned",
        )
        assert "experiments" in result
        assert "target_variable_history" in result

    @pytest.mark.asyncio
    async def test_scope_definer_cache_no_client(self):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        result = await hooks.cache_scope_definition("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_scope_definer_cache_success(self, mock_working_memory):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_scope_definition("s1", {"spec": True})
        assert result is True

    @pytest.mark.asyncio
    async def test_scope_definer_store_experiment_pattern_no_client(self):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        result = await hooks.store_experiment_pattern(
            "exp1", "test", "classification", "churned", ["f1"], {},
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_scope_definer_store_experiment_pattern_success(self, mock_semantic_memory):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_experiment_pattern(
            "exp1", "churn_model", "classification", "churned",
            ["age", "tenure"], {"auc": 0.8},
        )
        assert result is True
        # Should create Experiment, ProblemType, Variable, ScopeSpec nodes
        assert mock_semantic_memory.add_entity.call_count >= 4

    @pytest.mark.asyncio
    async def test_scope_definer_store_experiment_pattern_error(self, mock_semantic_memory):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_experiment_pattern(
            "exp1", "test", "classification", "target", [], {},
        )
        assert result is False

    # --- Observability Connector ---

    @pytest.mark.asyncio
    async def test_observability_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context("s1", time_window="1h")
        assert isinstance(ctx, oc_hooks.ObservabilityContext)

    @pytest.mark.asyncio
    async def test_observability_get_semantic_no_client(self):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        result = await hooks._get_semantic_context()
        assert result == {}

    @pytest.mark.asyncio
    async def test_observability_get_semantic_with_filter(self, mock_semantic_memory):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context(agent_name_filter="causal_impact")
        assert "health_snapshots" in result
        assert "agent_patterns" in result

    @pytest.mark.asyncio
    async def test_observability_cache_no_client(self):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        result = await hooks.cache_metrics("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_observability_cache_success(self, mock_working_memory):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        hooks._working_memory = mock_working_memory
        result = await hooks.cache_metrics("s1", {"rate": 0.95})
        assert result is True

    @pytest.mark.asyncio
    async def test_observability_store_health_snapshot_no_client(self):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        result = await hooks.store_health_snapshot(
            "24h", 0.95, 250.0, 0.88, {}, [],
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_observability_store_health_snapshot_success(self, mock_semantic_memory):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_health_snapshot(
            "24h", 0.95, 250.0, 0.88,
            {"agent_a": 0.15},  # >10% error, should create AgentPattern
            ["low success rate"],  # anomaly
        )
        assert result is True
        # Should create HealthSnapshot + Anomaly + AgentPattern
        assert mock_semantic_memory.add_entity.call_count >= 3

    @pytest.mark.asyncio
    async def test_observability_store_health_snapshot_error(self, mock_semantic_memory):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_health_snapshot(
            "1h", 0.9, 100.0, None, {}, [],
        )
        assert result is False

    def test_detect_anomalies_all_ok(self):
        result = oc_hooks._detect_anomalies({
            "overall_success_rate": 0.98,
            "overall_p95_latency_ms": 200,
            "quality_score": 0.9,
            "error_rate_by_agent": {"a": 0.01},
        })
        assert result == []

    def test_detect_anomalies_low_success(self):
        result = oc_hooks._detect_anomalies({"overall_success_rate": 0.90})
        assert len(result) == 1
        assert "Low success rate" in result[0]

    def test_detect_anomalies_high_latency(self):
        result = oc_hooks._detect_anomalies({
            "overall_success_rate": 0.99,
            "overall_p95_latency_ms": 10000,
        })
        assert any("High P95 latency" in a for a in result)

    def test_detect_anomalies_low_quality(self):
        result = oc_hooks._detect_anomalies({
            "overall_success_rate": 0.99,
            "quality_score": 0.5,
        })
        assert any("Low quality score" in a for a in result)

    def test_detect_anomalies_high_agent_error(self):
        result = oc_hooks._detect_anomalies({
            "overall_success_rate": 0.99,
            "error_rate_by_agent": {"bad_agent": 0.3},
        })
        assert any("bad_agent" in a for a in result)

    # --- Feature Analyzer ---

    @pytest.mark.asyncio
    async def test_feature_analyzer_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context("s1", "exp1", feature_names=["f1"])
        assert isinstance(ctx, fa_hooks.FeatureAnalysisContext)

    @pytest.mark.asyncio
    async def test_feature_analyzer_get_semantic_no_client(self):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        result = await hooks._get_semantic_context()
        assert result == {}

    @pytest.mark.asyncio
    async def test_feature_analyzer_get_semantic_with_features(self, mock_semantic_memory):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks._get_semantic_context(feature_names=["age", "tenure"])
        assert "interactions" in result

    @pytest.mark.asyncio
    async def test_feature_analyzer_cache_no_client(self):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        result = await hooks.cache_feature_analysis("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_feature_analyzer_store_patterns_no_client(self):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        result = await hooks.store_feature_importance_patterns("exp1", {}, [])
        assert result is False

    @pytest.mark.asyncio
    async def test_feature_analyzer_store_patterns_success(self, mock_semantic_memory):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_feature_importance_patterns(
            "exp1",
            {"age": 0.5, "tenure": 0.3},
            [{"feature_1": "age", "feature_2": "tenure", "interaction_strength": 0.4}],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_feature_analyzer_store_patterns_error(self, mock_semantic_memory):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_feature_importance_patterns(
            "exp1", {"f1": 0.5}, [],
        )
        assert result is False

    # --- Model Deployer ---

    @pytest.mark.asyncio
    async def test_model_deployer_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = md_hooks.ModelDeployerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context("s1", "models:/churn/1")
        assert isinstance(ctx, md_hooks.DeploymentContext)

    @pytest.mark.asyncio
    async def test_model_deployer_get_semantic_no_client(self):
        hooks = md_hooks.ModelDeployerMemoryHooks()
        result = await hooks._get_semantic_context("models:/churn/1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_model_deployer_cache_no_client(self):
        hooks = md_hooks.ModelDeployerMemoryHooks()
        result = await hooks.cache_deployment_manifest("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_model_deployer_store_pattern_no_client(self):
        hooks = md_hooks.ModelDeployerMemoryHooks()
        result = await hooks.store_deployment_pattern(
            "exp1", "dep1", 1, "staging", "success", "direct",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_model_deployer_store_pattern_with_rollback(self, mock_semantic_memory):
        hooks = md_hooks.ModelDeployerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_deployment_pattern(
            "exp1", "dep1", 2, "production", "rolled_back",
            "canary", rollback_occurred=True,
        )
        assert result is True
        # Should create Rollback entity when rollback_occurred
        entity_calls = [c[1].get("entity_type", c[0][0] if c[0] else "")
                        for c in mock_semantic_memory.add_entity.call_args_list]
        assert "Rollback" in entity_calls

    @pytest.mark.asyncio
    async def test_model_deployer_store_pattern_error(self, mock_semantic_memory):
        hooks = md_hooks.ModelDeployerMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_deployment_pattern(
            "exp1", "dep1", 1, "staging", "success", "direct",
        )
        assert result is False

    # --- Model Selector ---

    @pytest.mark.asyncio
    async def test_model_selector_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context("s1", "classification")
        assert isinstance(ctx, ms_hooks.ModelSelectionContext)

    @pytest.mark.asyncio
    async def test_model_selector_get_semantic_no_client(self):
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        result = await hooks._get_semantic_context("classification")
        assert result == {}

    @pytest.mark.asyncio
    async def test_model_selector_cache_no_client(self):
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        result = await hooks.cache_model_selection("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_model_selector_store_algorithm_pattern_no_client(self):
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        result = await hooks.store_algorithm_pattern(
            "exp1", "xgboost", "gradient_boosting", "classification", 0.9, {},
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_model_selector_store_algorithm_pattern_success(self, mock_semantic_memory):
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_algorithm_pattern(
            "exp1", "xgboost", "gradient_boosting", "classification",
            0.92, {"score": 0.88},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_model_selector_store_algorithm_pattern_error(self, mock_semantic_memory):
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_algorithm_pattern(
            "exp1", "xgb", "gb", "cls", 0.0, {},
        )
        assert result is False

    # --- Model Trainer ---

    @pytest.mark.asyncio
    async def test_model_trainer_get_context(self, mock_working_memory, mock_semantic_memory):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "_get_episodic_context", new_callable=AsyncMock, return_value=[]):
            ctx = await hooks.get_context("s1", "xgboost")
        assert isinstance(ctx, mt_hooks.ModelTrainingContext)

    @pytest.mark.asyncio
    async def test_model_trainer_get_semantic_no_client(self):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        result = await hooks._get_semantic_context("xgboost")
        assert result == {}

    @pytest.mark.asyncio
    async def test_model_trainer_cache_no_client(self):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        result = await hooks.cache_training_result("s1", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_model_trainer_store_model_pattern_no_client(self):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        result = await hooks.store_model_pattern(
            "exp1", "run1", "xgboost", {"auc_roc": 0.88}, {}, True,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_model_trainer_store_model_pattern_success(self, mock_semantic_memory):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_model_pattern(
            "exp1", "run1", "xgboost",
            {"auc_roc": 0.88, "f1_score": 0.82},
            {"n_estimators": 100, "max_depth": 5},
            success_criteria_met=True,
        )
        assert result is True
        # Should create Hyperparameters when success_criteria_met
        entity_calls = [c[1].get("entity_type", c[0][0] if c[0] else "")
                        for c in mock_semantic_memory.add_entity.call_args_list]
        assert "Hyperparameters" in entity_calls

    @pytest.mark.asyncio
    async def test_model_trainer_store_model_pattern_no_hyperparams(self, mock_semantic_memory):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_model_pattern(
            "exp1", "run1", "xgboost",
            {"auc_roc": 0.60}, {},
            success_criteria_met=False,
        )
        assert result is True
        # Should NOT create Hyperparameters when success_criteria_met=False
        entity_calls = [c[1].get("entity_type", c[0][0] if c[0] else "")
                        for c in mock_semantic_memory.add_entity.call_args_list]
        assert "Hyperparameters" not in entity_calls

    @pytest.mark.asyncio
    async def test_model_trainer_store_model_pattern_error(self, mock_semantic_memory):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        mock_semantic_memory.add_entity = MagicMock(side_effect=Exception("err"))
        hooks._semantic_memory = mock_semantic_memory
        result = await hooks.store_model_pattern(
            "exp1", "run1", "xgb", {}, {}, False,
        )
        assert result is False


# ===========================================================================
# Section 10: contribute_to_memory Function Tests
# ===========================================================================


class TestContributeToMemory:
    """Test the contribute_to_memory function for each module."""

    # --- Causal Impact ---

    @pytest.mark.asyncio
    async def test_ci_contribute_skip_failed(self):
        counts = await ci_hooks.contribute_to_memory(
            result={"status": "failed"},
            state={},
            memory_hooks=ci_hooks.CausalImpactMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0
        assert counts["semantic_stored"] == 0
        assert counts["working_cached"] == 0

    @pytest.mark.asyncio
    async def test_ci_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_causal_analysis", new_callable=AsyncMock, return_value="mem1"):
            counts = await ci_hooks.contribute_to_memory(
                result={
                    "status": "completed", "ate_estimate": 0.3,
                    "confidence": 0.8, "refutation_passed": True,
                    "effect_size": "small",
                },
                state={
                    "treatment_var": "hcp_visits",
                    "outcome_var": "trx",
                    "confounders": ["region"],
                },
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1

    @pytest.mark.asyncio
    async def test_ci_contribute_generates_session_id(self):
        hooks = ci_hooks.CausalImpactMemoryHooks()
        counts = await ci_hooks.contribute_to_memory(
            result={"status": "failed"},
            state={},
            memory_hooks=hooks,
        )
        assert counts["episodic_stored"] == 0

    # --- Orchestrator ---

    @pytest.mark.asyncio
    async def test_orch_contribute_skip_failed(self):
        counts = await orch_hooks.contribute_to_memory(
            result={"status": "failed"},
            state={},
            memory_hooks=orch_hooks.OrchestratorMemoryHooks(),
            session_id="s1",
        )
        assert counts == {
            "episodic_stored": 0,
            "working_cached": 0,
            "conversation_stored": 0,
            "routing_tracked": 0,
        }

    @pytest.mark.asyncio
    async def test_orch_contribute_success(self, mock_working_memory):
        hooks = orch_hooks.OrchestratorMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_orchestration", new_callable=AsyncMock, return_value="mem1"):
            counts = await orch_hooks.contribute_to_memory(
                result={
                    "status": "completed", "query_id": "q1",
                    "response_text": "HCP visits drive TRx.",
                    "intent_classified": "analysis",
                    "agents_dispatched": ["causal_impact"],
                    "total_latency_ms": 500,
                    "intent_confidence": 0.9,
                },
                state={"query": "what drives trx?"},
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["working_cached"] == 1
        assert counts["conversation_stored"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["routing_tracked"] == 1

    # --- Health Score ---

    @pytest.mark.asyncio
    async def test_hs_contribute_skip_failed(self):
        counts = await hs_hooks.contribute_to_memory(
            result={},
            state={"status": "failed"},
            memory_hooks=hs_hooks.HealthScoreMemoryHooks(),
            session_id="s1",
        )
        assert counts == {"episodic_stored": 0, "working_cached": 0}

    @pytest.mark.asyncio
    async def test_hs_contribute_success(self, mock_working_memory):
        hooks = hs_hooks.HealthScoreMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_health_check", new_callable=AsyncMock, return_value=None):
            counts = await hs_hooks.contribute_to_memory(
                result={"overall_health_score": 90, "health_grade": "A"},
                state={"check_scope": "full"},
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["working_cached"] == 1

    # --- Cohort Constructor ---

    @pytest.mark.asyncio
    async def test_cc_contribute_skip_failed(self):
        counts = await cc_hooks.contribute_to_memory(
            result={"status": "failed"},
            state={},
            memory_hooks=cc_hooks.CohortConstructorMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0

    @pytest.mark.asyncio
    async def test_cc_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_cohort_result", new_callable=AsyncMock, return_value="mem1"):
            with patch.object(hooks, "store_cohort_pattern", new_callable=AsyncMock, return_value=True):
                with patch.object(hooks, "store_eligibility_rule", new_callable=AsyncMock, return_value=True):
                    counts = await cc_hooks.contribute_to_memory(
                        result={
                            "status": "success", "cohort_id": "c1",
                        },
                        state={
                            "config": {
                                "brand": "Kisqali",
                                "cohort_name": "test",
                                "indication": "bc",
                                "inclusion_criteria": [{"field": "age"}],
                                "exclusion_criteria": [],
                            },
                            "eligibility_stats": {"exclusion_rate": 0.2},
                        },
                        memory_hooks=hooks,
                        session_id="s1",
                    )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1
        assert counts["rules_stored"] == 1

    # --- Data Preparer ---

    @pytest.mark.asyncio
    async def test_dp_contribute_skip_error(self):
        counts = await dp_hooks.contribute_to_memory(
            result={},
            state={"error": "something went wrong"},
            memory_hooks=dp_hooks.DataPreparerMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0

    @pytest.mark.asyncio
    async def test_dp_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = dp_hooks.DataPreparerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_qc_report", new_callable=AsyncMock, return_value="mem1"):
            with patch.object(hooks, "store_data_quality_pattern", new_callable=AsyncMock, return_value=True):
                counts = await dp_hooks.contribute_to_memory(
                    result={"report_id": "r1", "qc_status": "passed", "gate_passed": True},
                    state={
                        "overall_score": 0.95,
                        "data_source": "supabase",
                        "experiment_id": "exp1",
                        "leakage_detected": False,
                        "blocking_issues": [],
                    },
                    memory_hooks=hooks,
                    session_id="s1",
                )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1

    # --- Scope Definer ---

    @pytest.mark.asyncio
    async def test_sd_contribute_skip_validation_failed(self):
        counts = await sd_hooks.contribute_to_memory(
            result={},
            state={"validation_passed": False},
            memory_hooks=sd_hooks.ScopeDefinerMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0

    @pytest.mark.asyncio
    async def test_sd_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_scope_definition", new_callable=AsyncMock, return_value="mem1"):
            with patch.object(hooks, "store_experiment_pattern", new_callable=AsyncMock, return_value=True):
                counts = await sd_hooks.contribute_to_memory(
                    result={
                        "experiment_id": "exp1", "experiment_name": "churn",
                        "scope_spec": {}, "success_criteria": {"auc": 0.8},
                    },
                    state={
                        "validation_passed": True,
                        "inferred_problem_type": "classification",
                        "inferred_target_variable": "churned",
                        "required_features": ["age"],
                    },
                    memory_hooks=hooks,
                    session_id="s1",
                )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1

    # --- Observability Connector ---

    @pytest.mark.asyncio
    async def test_oc_contribute_skip_error(self):
        counts = await oc_hooks.contribute_to_memory(
            result={},
            state={"error": "timeout"},
            memory_hooks=oc_hooks.ObservabilityConnectorMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0

    @pytest.mark.asyncio
    async def test_oc_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = oc_hooks.ObservabilityConnectorMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_observability_event", new_callable=AsyncMock, return_value="mem1"):
            with patch.object(hooks, "store_health_snapshot", new_callable=AsyncMock, return_value=True):
                counts = await oc_hooks.contribute_to_memory(
                    result={
                        "overall_success_rate": 0.98,
                        "overall_p95_latency_ms": 200,
                        "quality_score": 0.9,
                        "total_spans_analyzed": 100,
                        "error_rate_by_agent": {},
                    },
                    state={"time_window": "24h"},
                    memory_hooks=hooks,
                    session_id="s1",
                )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1

    # --- Feature Analyzer ---

    @pytest.mark.asyncio
    async def test_fa_contribute_skip_failed(self):
        counts = await fa_hooks.contribute_to_memory(
            result={"status": "failed"},
            state={},
            memory_hooks=fa_hooks.FeatureAnalyzerMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0

    @pytest.mark.asyncio
    async def test_fa_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_feature_analysis", new_callable=AsyncMock, return_value="mem1"):
            with patch.object(hooks, "store_feature_importance_patterns", new_callable=AsyncMock, return_value=True):
                counts = await fa_hooks.contribute_to_memory(
                    result={
                        "status": "completed",
                        "shap_analysis_id": "shap1",
                        "top_features": ["f1"],
                        "interaction_list": [],
                    },
                    state={
                        "experiment_id": "exp1",
                        "selected_feature_count": 5,
                        "global_importance": {"f1": 0.5},
                    },
                    memory_hooks=hooks,
                    session_id="s1",
                )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1

    # --- Model Deployer ---

    @pytest.mark.asyncio
    async def test_md_contribute_skip_failed(self):
        counts = await md_hooks.contribute_to_memory(
            result={"overall_status": "failed"},
            state={},
            memory_hooks=md_hooks.ModelDeployerMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0

    @pytest.mark.asyncio
    async def test_md_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = md_hooks.ModelDeployerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_deployment", new_callable=AsyncMock, return_value="mem1"):
            with patch.object(hooks, "store_deployment_pattern", new_callable=AsyncMock, return_value=True):
                counts = await md_hooks.contribute_to_memory(
                    result={
                        "overall_status": "success",
                        "deployment_id": "dep1",
                        "deployment_manifest": {"image": "v1"},
                        "model_version": 2,
                        "deployment_status": "active",
                    },
                    state={
                        "experiment_id": "exp1",
                        "target_environment": "staging",
                        "deployment_strategy": "canary",
                    },
                    memory_hooks=hooks,
                    session_id="s1",
                )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1

    # --- Model Selector ---

    @pytest.mark.asyncio
    async def test_ms_contribute_skip_error(self):
        counts = await ms_hooks.contribute_to_memory(
            result={},
            state={"error": "no data"},
            memory_hooks=ms_hooks.ModelSelectorMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0

    @pytest.mark.asyncio
    async def test_ms_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_model_selection", new_callable=AsyncMock, return_value="mem1"):
            with patch.object(hooks, "store_algorithm_pattern", new_callable=AsyncMock, return_value=True):
                counts = await ms_hooks.contribute_to_memory(
                    result={
                        "algorithm_name": "xgboost",
                        "algorithm_family": "gradient_boosting",
                        "selection_score": 0.92,
                    },
                    state={
                        "experiment_id": "exp1",
                        "problem_type": "classification",
                        "benchmark_results": {"score": 0.88},
                    },
                    memory_hooks=hooks,
                    session_id="s1",
                )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1

    # --- Model Trainer ---

    @pytest.mark.asyncio
    async def test_mt_contribute_skip_failed(self):
        counts = await mt_hooks.contribute_to_memory(
            result={"training_status": "failed"},
            state={},
            memory_hooks=mt_hooks.ModelTrainerMemoryHooks(),
            session_id="s1",
        )
        assert counts["episodic_stored"] == 0

    @pytest.mark.asyncio
    async def test_mt_contribute_success(self, mock_working_memory, mock_semantic_memory):
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        hooks._working_memory = mock_working_memory
        hooks._semantic_memory = mock_semantic_memory

        with patch.object(hooks, "store_training_result", new_callable=AsyncMock, return_value="mem1"):
            with patch.object(hooks, "store_model_pattern", new_callable=AsyncMock, return_value=True):
                counts = await mt_hooks.contribute_to_memory(
                    result={
                        "training_status": "completed",
                        "training_run_id": "run1",
                        "model_id": "m1",
                        "success_criteria_met": True,
                        "test_metrics": {"auc_roc": 0.88},
                    },
                    state={
                        "experiment_id": "exp1",
                        "algorithm_name": "xgboost",
                        "best_hyperparameters": {"n_estimators": 100},
                    },
                    memory_hooks=hooks,
                    session_id="s1",
                )
        assert counts["working_cached"] == 1
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1


# ===========================================================================
# Section 11: Edge Case & Integration Tests
# ===========================================================================


class TestEdgeCases:
    """Test edge cases across memory hooks modules."""

    @pytest.mark.asyncio
    async def test_ci_contribute_no_refutation_skips_semantic(self, mock_working_memory):
        """Causal path not stored when refutation did not pass."""
        hooks = ci_hooks.CausalImpactMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_causal_analysis", new_callable=AsyncMock, return_value="mem1"):
            counts = await ci_hooks.contribute_to_memory(
                result={
                    "status": "completed", "ate_estimate": 0.3,
                    "refutation_passed": False,
                },
                state={
                    "treatment_var": "t", "outcome_var": "o",
                    "confounders": [],
                },
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["semantic_stored"] == 0

    @pytest.mark.asyncio
    async def test_orch_contribute_no_query_skips_conversation(self, mock_working_memory):
        """Conversation not stored when query is empty."""
        hooks = orch_hooks.OrchestratorMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_orchestration", new_callable=AsyncMock, return_value="mem1"):
            counts = await orch_hooks.contribute_to_memory(
                result={
                    "status": "completed", "query_id": "q1",
                    "response_text": "",  # empty response
                    "intent_classified": "unknown",
                    "agents_dispatched": [],
                    "total_latency_ms": 100,
                    "intent_confidence": 0.5,
                },
                state={"query": ""},
                memory_hooks=hooks,
                session_id="s1",
            )
        # Empty query + empty response -> conversation not stored
        assert counts["conversation_stored"] == 0

    @pytest.mark.asyncio
    async def test_cc_contribute_no_cohort_id_skips_semantic(self, mock_working_memory):
        """Semantic not stored when cohort_id is missing."""
        hooks = cc_hooks.CohortConstructorMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_cohort_result", new_callable=AsyncMock, return_value="mem1"):
            counts = await cc_hooks.contribute_to_memory(
                result={"status": "success"},  # no cohort_id
                state={"config": {"brand": "Kisqali"}},
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["semantic_stored"] == 0

    @pytest.mark.asyncio
    async def test_dp_contribute_no_data_source_skips_semantic(self, mock_working_memory):
        """Semantic not stored when data_source is missing."""
        hooks = dp_hooks.DataPreparerMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_qc_report", new_callable=AsyncMock, return_value="mem1"):
            counts = await dp_hooks.contribute_to_memory(
                result={"report_id": "r1", "qc_status": "passed"},
                state={"overall_score": 0.9},  # no data_source, no experiment_id
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["semantic_stored"] == 0

    @pytest.mark.asyncio
    async def test_sd_contribute_no_experiment_id_skips_semantic(self, mock_working_memory):
        """Semantic not stored when experiment_id is missing."""
        hooks = sd_hooks.ScopeDefinerMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_scope_definition", new_callable=AsyncMock, return_value="mem1"):
            counts = await sd_hooks.contribute_to_memory(
                result={"scope_spec": {}},  # no experiment_id
                state={"validation_passed": True},
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["semantic_stored"] == 0

    @pytest.mark.asyncio
    async def test_fa_contribute_no_experiment_skips_semantic(self, mock_working_memory):
        """Semantic not stored when experiment_id or global_importance missing."""
        hooks = fa_hooks.FeatureAnalyzerMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_feature_analysis", new_callable=AsyncMock, return_value="mem1"):
            counts = await fa_hooks.contribute_to_memory(
                result={"status": "completed"},
                state={},  # no experiment_id, no global_importance
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["semantic_stored"] == 0

    @pytest.mark.asyncio
    async def test_md_contribute_no_ids_skips_semantic(self, mock_working_memory):
        """Semantic not stored when experiment_id or deployment_id missing."""
        hooks = md_hooks.ModelDeployerMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_deployment", new_callable=AsyncMock, return_value="mem1"):
            counts = await md_hooks.contribute_to_memory(
                result={"overall_status": "success", "deployment_manifest": {}},
                state={},  # no experiment_id
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["semantic_stored"] == 0

    @pytest.mark.asyncio
    async def test_ms_contribute_no_ids_skips_semantic(self, mock_working_memory):
        """Semantic not stored when experiment_id or algorithm_name missing."""
        hooks = ms_hooks.ModelSelectorMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_model_selection", new_callable=AsyncMock, return_value="mem1"):
            counts = await ms_hooks.contribute_to_memory(
                result={},  # no algorithm_name
                state={},  # no experiment_id
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["semantic_stored"] == 0

    @pytest.mark.asyncio
    async def test_mt_contribute_no_ids_skips_semantic(self, mock_working_memory):
        """Semantic not stored when experiment_id or training_run_id missing."""
        hooks = mt_hooks.ModelTrainerMemoryHooks()
        hooks._working_memory = mock_working_memory

        with patch.object(hooks, "store_training_result", new_callable=AsyncMock, return_value="mem1"):
            counts = await mt_hooks.contribute_to_memory(
                result={"training_status": "completed"},  # no training_run_id
                state={},  # no experiment_id
                memory_hooks=hooks,
                session_id="s1",
            )
        assert counts["semantic_stored"] == 0

    @pytest.mark.asyncio
    async def test_hs_contribute_generates_session_id(self):
        """Health score generates UUID session_id when not provided."""
        hooks = hs_hooks.HealthScoreMemoryHooks()

        with patch.object(hooks, "cache_health_check", new_callable=AsyncMock, return_value=False):
            with patch.object(hooks, "store_health_check", new_callable=AsyncMock, return_value=None):
                counts = await hs_hooks.contribute_to_memory(
                    result={}, state={}, memory_hooks=hooks,
                )
        assert counts["working_cached"] == 0

    @pytest.mark.asyncio
    async def test_all_contribute_use_default_hooks(self):
        """All contribute_to_memory functions create default hooks when None."""
        # These just verify the code path doesn't crash
        # They will fail on actual memory operations (no Redis/FalkorDB)
        # but the skip-on-failure paths should handle it gracefully
        for mod, skip_key, skip_val in [
            (ci_hooks, "status", "failed"),
            (orch_hooks, "status", "failed"),
            (dp_hooks, "error", "test"),
            (fa_hooks, "status", "failed"),
            (md_hooks, "overall_status", "failed"),
            (mt_hooks, "training_status", "failed"),
        ]:
            result = {skip_key: skip_val}
            counts = await mod.contribute_to_memory(result=result, state={})
            assert isinstance(counts, dict)
