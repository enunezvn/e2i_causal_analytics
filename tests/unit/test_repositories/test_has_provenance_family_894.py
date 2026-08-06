"""#894: predicate pins for the HAS_PROVENANCE family.

Every repository below sits on an ``is_synthetic``-tagged table (migrations
063/067/069) but never set ``HAS_PROVENANCE = True`` — and most expose bespoke
read methods that bypass ``BaseRepository.get_many`` entirely, so the flag
alone is not enough: each bespoke read must call ``apply_provenance_filter``
itself.

These tests pin the ``.eq("is_synthetic", False)`` predicate per read path via
recording supabase-style query builders (mirrors
``test_causal_path_provenance.py``). The faithful live-DB proofs live in
``tests/integration/test_has_provenance_family_894.py``.

Env isolation (#1497, same class as #1495): ``apply_provenance_filter`` is
deliberately gated on ``E2I_INCLUDE_SYNTHETIC`` (WS-SYNTH showcase instances
skip the predicate), and that var IS set on showcase/dev hosts (this repo's
``.env`` plus the find_dotenv walk-up class, PR #1414). Every test here
exercises the kwarg-driven opt-in/real-mode contract — none reads the ambient
env — so an autouse ``delenv`` pins real mode for the whole module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _pin_real_mode_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin real mode for every test in this module regardless of host env.

    Without this, any host exporting ``E2I_INCLUDE_SYNTHETIC`` (showcase/dev
    boxes) makes production legitimately skip the filter and every
    ``*_excludes_synthetic`` pin here (49 tests) fails for an environmental —
    not functional — reason (and the ``*_opt_in`` absence assertions pass
    vacuously).
    """
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


# =============================================================================
# Recording query builders (sync + async execute variants: the A/B-side repos
# follow the sync-client convention, the ML/Twin-side repos await execute)
# =============================================================================


class _ChainableQuery:
    """supabase-style fluent builder that records predicate calls."""

    def __init__(self, sync: bool = False, data: list | None = None, count: int = 0) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._sync = sync
        self._execute_data: list[dict[str, Any]] = data or []
        self._count = count

    def _record(self, name: str, *args: Any) -> "_ChainableQuery":
        self.calls.append((name, args))
        return self

    def select(self, *a: Any, **kw: Any) -> "_ChainableQuery":
        return self._record("select", *a)

    def eq(self, *a: Any) -> "_ChainableQuery":
        return self._record("eq", *a)

    def neq(self, *a: Any) -> "_ChainableQuery":
        return self._record("neq", *a)

    def gt(self, *a: Any) -> "_ChainableQuery":
        return self._record("gt", *a)

    def gte(self, *a: Any) -> "_ChainableQuery":
        return self._record("gte", *a)

    def lt(self, *a: Any) -> "_ChainableQuery":
        return self._record("lt", *a)

    def lte(self, *a: Any) -> "_ChainableQuery":
        return self._record("lte", *a)

    def in_(self, *a: Any) -> "_ChainableQuery":
        return self._record("in_", *a)

    def is_(self, *a: Any) -> "_ChainableQuery":
        return self._record("is_", *a)

    @property
    def not_(self) -> "_ChainableQuery":
        return self._record("not_")

    def order(self, *a: Any, **kw: Any) -> "_ChainableQuery":
        return self._record("order", *a)

    def limit(self, *a: Any) -> "_ChainableQuery":
        return self._record("limit", *a)

    def offset(self, *a: Any) -> "_ChainableQuery":
        return self._record("offset", *a)

    def execute(self) -> Any:
        result = MagicMock()
        result.data = list(self._execute_data)
        result.count = self._count
        if self._sync:
            return result
        return AsyncMock(return_value=result)()


class _RecordingClient:
    """Client whose .table() hands out one recorder per table name."""

    def __init__(self, sync: bool = False, data: dict[str, list] | None = None) -> None:
        self._sync = sync
        self._data = data or {}
        self.queries: dict[str, list[_ChainableQuery]] = {}

    def table(self, name: str) -> _ChainableQuery:
        q = _ChainableQuery(sync=self._sync, data=self._data.get(name))
        self.queries.setdefault(name, []).append(q)
        return q

    def last(self, name: str) -> _ChainableQuery:
        return self.queries[name][-1]


def _eq_calls(query: _ChainableQuery) -> list[tuple[Any, ...]]:
    return [args for (name, args) in query.calls if name == "eq"]


def _assert_excludes(query: _ChainableQuery, where: str) -> None:
    assert ("is_synthetic", False) in _eq_calls(query), (
        f"{where} did not default-exclude synthetic rows. eq calls: {_eq_calls(query)}"
    )


def _assert_no_predicate(query: _ChainableQuery, where: str) -> None:
    assert ("is_synthetic", False) not in _eq_calls(query), (
        f"{where} applied the provenance predicate under include_synthetic=True. "
        f"eq calls: {_eq_calls(query)}"
    )


# =============================================================================
# HAS_PROVENANCE flags
# =============================================================================


def test_all_family_repositories_set_has_provenance() -> None:
    from src.repositories.ab_experiment import ABExperimentRepository
    from src.repositories.ab_results import ABResultsRepository
    from src.repositories.agent_activity import AgentActivityRepository
    from src.repositories.deployment import MLDeploymentRepository
    from src.repositories.ml_experiment import (
        MLExperimentRepository,
        MLModelRegistryRepository,
        MLTrainingRunRepository,
    )
    from src.repositories.observability_span import ObservabilitySpanRepository
    from src.repositories.user_session import UserSessionRepository

    for repo_cls in (
        ABExperimentRepository,
        ABResultsRepository,
        AgentActivityRepository,
        MLDeploymentRepository,
        MLExperimentRepository,
        MLModelRegistryRepository,
        MLTrainingRunRepository,
        ObservabilitySpanRepository,
        UserSessionRepository,
    ):
        assert repo_cls.HAS_PROVENANCE is True, (
            f"{repo_cls.__name__} sits on is_synthetic-tagged table "
            f"{repo_cls.table_name!r} but HAS_PROVENANCE is not True"
        )


# =============================================================================
# ABExperimentRepository (sync-client convention)
# =============================================================================


def _ab_experiment_repo(data: dict[str, list] | None = None):
    from src.repositories.ab_experiment import ABExperimentRepository

    client = _RecordingClient(sync=True, data=data)
    return ABExperimentRepository(supabase_client=client), client


@pytest.mark.asyncio
async def test_ab_get_assignments_excludes_synthetic() -> None:
    repo, client = _ab_experiment_repo()
    await repo.get_assignments(uuid4())
    _assert_excludes(client.last("ab_experiment_assignments"), "get_assignments")


@pytest.mark.asyncio
async def test_ab_get_assignments_opt_in() -> None:
    repo, client = _ab_experiment_repo()
    await repo.get_assignments(uuid4(), include_synthetic=True)
    _assert_no_predicate(client.last("ab_experiment_assignments"), "get_assignments")


@pytest.mark.asyncio
async def test_ab_get_assignment_excludes_synthetic() -> None:
    repo, client = _ab_experiment_repo()
    await repo.get_assignment(uuid4())
    _assert_excludes(client.last("ab_experiment_assignments"), "get_assignment")


@pytest.mark.asyncio
async def test_ab_get_assignment_by_unit_excludes_synthetic() -> None:
    repo, client = _ab_experiment_repo()
    await repo.get_assignment_by_unit(uuid4(), "hcp-1")
    _assert_excludes(client.last("ab_experiment_assignments"), "get_assignment_by_unit")


@pytest.mark.asyncio
async def test_ab_get_assignment_counts_excludes_synthetic() -> None:
    repo, client = _ab_experiment_repo()
    await repo.get_assignment_counts(uuid4())
    _assert_excludes(client.last("ab_experiment_assignments"), "get_assignment_counts")


@pytest.mark.asyncio
async def test_ab_get_enrollments_by_experiment_excludes_synthetic() -> None:
    repo, client = _ab_experiment_repo()
    await repo.get_enrollments_by_experiment(uuid4())
    _assert_excludes(client.last("ab_experiment_enrollments"), "get_enrollments_by_experiment")


@pytest.mark.asyncio
async def test_ab_get_enrollment_excludes_synthetic() -> None:
    repo, client = _ab_experiment_repo()
    await repo.get_enrollment(uuid4())
    _assert_excludes(client.last("ab_experiment_enrollments"), "get_enrollment")


@pytest.mark.asyncio
async def test_ab_get_enrollment_by_assignment_excludes_synthetic() -> None:
    repo, client = _ab_experiment_repo()
    await repo.get_enrollment_by_assignment(uuid4())
    _assert_excludes(client.last("ab_experiment_enrollments"), "get_enrollment_by_assignment")


# =============================================================================
# ABResultsRepository (sync-client convention)
# =============================================================================


def _ab_results_repo():
    from src.repositories.ab_results import ABResultsRepository

    client = _RecordingClient(sync=True)
    return ABResultsRepository(supabase_client=client), client


@pytest.mark.asyncio
async def test_ab_get_results_excludes_synthetic() -> None:
    repo, client = _ab_results_repo()
    await repo.get_results(uuid4())
    _assert_excludes(client.last("ab_experiment_results"), "get_results")


@pytest.mark.asyncio
async def test_ab_get_results_opt_in() -> None:
    repo, client = _ab_results_repo()
    await repo.get_results(uuid4(), include_synthetic=True)
    _assert_no_predicate(client.last("ab_experiment_results"), "get_results")


@pytest.mark.asyncio
async def test_ab_get_latest_results_excludes_synthetic() -> None:
    repo, client = _ab_results_repo()
    await repo.get_latest_results(uuid4())
    _assert_excludes(client.last("ab_experiment_results"), "get_latest_results")


# =============================================================================
# ExperimentOutcomeRepository (assignments leg of the outcome feed)
# =============================================================================


@pytest.mark.asyncio
async def test_outcome_feed_assignments_exclude_synthetic() -> None:
    from src.repositories.experiment_outcome import ExperimentOutcomeRepository

    client = _RecordingClient(sync=True)
    repo = ExperimentOutcomeRepository(supabase_client=client)
    await repo.load_arrays(uuid4(), "trx")
    _assert_excludes(client.last("ab_experiment_assignments"), "load_arrays assignments")


@pytest.mark.asyncio
async def test_outcome_feed_assignments_opt_in() -> None:
    from src.repositories.experiment_outcome import ExperimentOutcomeRepository

    client = _RecordingClient(
        sync=True,
        data={
            "ab_experiment_assignments": [
                {"unit_id": "h1", "variant": "control"},
            ]
        },
    )
    repo = ExperimentOutcomeRepository(supabase_client=client)
    await repo.load_arrays(uuid4(), "trx", include_synthetic=True)
    _assert_no_predicate(client.last("ab_experiment_assignments"), "load_arrays assignments")
    # the business_metrics leg honors the same opt-in (pre-existing behavior)
    _assert_no_predicate(client.last("business_metrics"), "load_arrays business_metrics")


# =============================================================================
# MLExperimentRepository / MLTrainingRunRepository / MLModelRegistryRepository
# =============================================================================


def _ml_repo(repo_cls, data: dict[str, list] | None = None):
    client = _RecordingClient(sync=False, data=data)
    return repo_cls(supabase_client=client), client


@pytest.mark.asyncio
async def test_ml_experiment_get_by_name_excludes_synthetic() -> None:
    from src.repositories.ml_experiment import MLExperimentRepository

    repo, client = _ml_repo(MLExperimentRepository)
    await repo.get_by_name("exp")
    _assert_excludes(client.last("ml_experiments"), "get_by_name")


@pytest.mark.asyncio
async def test_ml_experiment_get_by_name_opt_in() -> None:
    from src.repositories.ml_experiment import MLExperimentRepository

    repo, client = _ml_repo(MLExperimentRepository)
    await repo.get_by_name("exp", include_synthetic=True)
    _assert_no_predicate(client.last("ml_experiments"), "get_by_name")


@pytest.mark.asyncio
async def test_ml_experiment_get_by_mlflow_id_excludes_synthetic() -> None:
    from src.repositories.ml_experiment import MLExperimentRepository

    repo, client = _ml_repo(MLExperimentRepository)
    await repo.get_by_mlflow_id("mlf-1")
    _assert_excludes(client.last("ml_experiments"), "get_by_mlflow_id")


@pytest.mark.asyncio
async def test_ml_experiment_get_many_excludes_synthetic() -> None:
    from src.repositories.ml_experiment import MLExperimentRepository

    repo, client = _ml_repo(MLExperimentRepository)
    await repo.get_many(filters={})
    _assert_excludes(client.last("ml_experiments"), "get_many")


@pytest.mark.asyncio
async def test_training_run_get_by_mlflow_run_id_excludes_synthetic() -> None:
    from src.repositories.ml_experiment import MLTrainingRunRepository

    repo, client = _ml_repo(MLTrainingRunRepository)
    await repo.get_by_mlflow_run_id("run-1")
    _assert_excludes(client.last("ml_training_runs"), "get_by_mlflow_run_id")


@pytest.mark.asyncio
async def test_registry_get_by_name_version_excludes_synthetic() -> None:
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo, client = _ml_repo(MLModelRegistryRepository)
    await repo.get_by_name_version("m", "1")
    _assert_excludes(client.last("ml_model_registry"), "get_by_name_version")


@pytest.mark.asyncio
async def test_registry_get_champion_model_excludes_synthetic() -> None:
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo, client = _ml_repo(MLModelRegistryRepository)
    await repo.get_champion_model()
    _assert_excludes(client.last("ml_model_registry"), "get_champion_model")


@pytest.mark.asyncio
async def test_registry_get_models_for_target_excludes_synthetic() -> None:
    """The serving surface NEVER reads synthetic rows (no opt-in by design:
    a synthetic model name must never reach the prediction ensemble)."""
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo, client = _ml_repo(MLModelRegistryRepository)
    await repo.get_models_for_target("csu_treatment_initiation")
    _assert_excludes(client.last("ml_model_registry"), "get_models_for_target")


@pytest.mark.asyncio
async def test_registry_get_model_performance_for_target_excludes_synthetic() -> None:
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo, client = _ml_repo(MLModelRegistryRepository)
    await repo.get_model_performance_for_target("csu_treatment_initiation")
    _assert_excludes(client.last("ml_model_registry"), "get_model_performance_for_target")


# =============================================================================
# ObservabilitySpanRepository (sync-client custom reads)
# =============================================================================


def _span_repo():
    from src.repositories.observability_span import ObservabilitySpanRepository

    client = _RecordingClient(sync=True)
    return ObservabilitySpanRepository(supabase_client=client), client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("get_spans_by_time_window", {"window": "24h"}),
        ("get_spans_by_trace_id", {"trace_id": "t-1"}),
        ("get_spans_by_agent", {"agent_name": "orchestrator"}),
        ("get_spans_by_tier", {"agent_tier": "coordination"}),
        ("get_error_spans", {}),
        ("get_fallback_spans", {}),
    ],
)
async def test_span_reads_exclude_synthetic(method: str, kwargs: dict) -> None:
    repo, client = _span_repo()
    await getattr(repo, method)(**kwargs)
    _assert_excludes(client.last("ml_observability_spans"), method)


@pytest.mark.asyncio
async def test_span_reads_opt_in() -> None:
    repo, client = _span_repo()
    await repo.get_spans_by_time_window(window="24h", include_synthetic=True)
    _assert_no_predicate(client.last("ml_observability_spans"), "get_spans_by_time_window")


# =============================================================================
# AgentActivityRepository (async custom reads)
# =============================================================================


def _activity_repo():
    from src.repositories.agent_activity import AgentActivityRepository

    client = _RecordingClient(sync=False)
    return AgentActivityRepository(supabase_client=client), client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,args",
    [
        ("get_by_agent", ("orchestrator",)),
        ("get_by_tier", ("coordination",)),
        ("get_analysis_results", ("causal_impact",)),
        ("get_recent_activities", ()),
        ("get_by_workstream", ("WS1",)),
        ("get_agent_activity_summary", ()),
        (
            "get_activities_in_range",
            (datetime.now(timezone.utc), datetime.now(timezone.utc)),
        ),
    ],
)
async def test_agent_activity_reads_exclude_synthetic(method: str, args: tuple) -> None:
    repo, client = _activity_repo()
    await getattr(repo, method)(*args)
    _assert_excludes(client.last("agent_activities"), method)


@pytest.mark.asyncio
async def test_agent_activity_opt_in() -> None:
    repo, client = _activity_repo()
    await repo.get_by_agent("orchestrator", include_synthetic=True)
    _assert_no_predicate(client.last("agent_activities"), "get_by_agent")


# =============================================================================
# UserSessionRepository
# =============================================================================


def _session_repo():
    from src.repositories.user_session import UserSessionRepository

    client = _RecordingClient(sync=False)
    return UserSessionRepository(supabase_client=client), client


@pytest.mark.asyncio
async def test_user_sessions_in_range_excludes_synthetic() -> None:
    repo, client = _session_repo()
    await repo.get_sessions_in_range(datetime.now(timezone.utc), datetime.now(timezone.utc))
    _assert_excludes(client.last("user_sessions"), "get_sessions_in_range")


@pytest.mark.asyncio
async def test_user_session_metrics_excludes_synthetic() -> None:
    repo, client = _session_repo()
    await repo.get_session_metrics()
    _assert_excludes(client.last("user_sessions"), "get_session_metrics")


@pytest.mark.asyncio
async def test_user_engagement_by_role_excludes_synthetic() -> None:
    repo, client = _session_repo()
    await repo.get_engagement_by_role()
    _assert_excludes(client.last("user_sessions"), "get_engagement_by_role")


# =============================================================================
# MLDeploymentRepository
# =============================================================================


def _deployment_repo():
    from src.repositories.deployment import MLDeploymentRepository

    client = _RecordingClient(sync=False)
    return MLDeploymentRepository(supabase_client=client), client


@pytest.mark.asyncio
async def test_deployment_get_active_excludes_synthetic() -> None:
    repo, client = _deployment_repo()
    await repo.get_active_deployment("production")
    _assert_excludes(client.last("ml_deployments"), "get_active_deployment")


@pytest.mark.asyncio
async def test_deployment_get_for_model_excludes_synthetic() -> None:
    repo, client = _deployment_repo()
    await repo.get_deployments_for_model(uuid4())
    _assert_excludes(client.last("ml_deployments"), "get_deployments_for_model")


@pytest.mark.asyncio
async def test_deployment_get_by_name_excludes_synthetic() -> None:
    repo, client = _deployment_repo()
    await repo.get_deployment_by_name("dep-1")
    _assert_excludes(client.last("ml_deployments"), "get_deployment_by_name")


# =============================================================================
# CausalPathRepository: live-schema column mapping (cross-posted to #894)
# =============================================================================


def _causal_repo():
    from src.repositories.causal_path import CausalPathRepository

    client = _RecordingClient(sync=False)
    return CausalPathRepository(supabase_client=client), client


@pytest.mark.asyncio
async def test_get_paths_for_cause_filters_start_node() -> None:
    """Live schema has start_node/end_node, not cause/effect (42703 pre-fix)."""
    repo, client = _causal_repo()
    await repo.get_paths_for_cause("hcp_engagement")
    eqs = _eq_calls(client.last("causal_paths"))
    assert ("start_node", "hcp_engagement") in eqs, f"eq calls: {eqs}"
    assert all(col != "cause" for col, *_ in eqs)


@pytest.mark.asyncio
async def test_get_paths_for_effect_filters_end_node() -> None:
    repo, client = _causal_repo()
    await repo.get_paths_for_effect("trx_growth")
    eqs = _eq_calls(client.last("causal_paths"))
    assert ("end_node", "trx_growth") in eqs, f"eq calls: {eqs}"
    assert all(col != "effect" for col, *_ in eqs)


@pytest.mark.asyncio
async def test_get_path_between_filters_both_nodes() -> None:
    repo, client = _causal_repo()
    await repo.get_path_between("hcp_engagement", "trx_growth")
    eqs = _eq_calls(client.last("causal_paths"))
    assert ("start_node", "hcp_engagement") in eqs and ("end_node", "trx_growth") in eqs


@pytest.mark.asyncio
async def test_causal_path_get_by_id_filters_path_id() -> None:
    """BaseRepository.get_by_id must use the declared PK column (path_id)."""
    repo, client = _causal_repo()
    await repo.get_by_id("CP123")
    eqs = _eq_calls(client.last("causal_paths"))
    assert ("path_id", "CP123") in eqs, f"eq calls: {eqs}"
    assert all(col != "id" for col, *_ in eqs)


# =============================================================================
# MLDataLoader: the tagged-table set is migration-derived (SSOT)
# =============================================================================


def test_provenance_tagged_tables_match_migrations() -> None:
    """PROVENANCE_TAGGED_TABLES must be the 26 tables migrations 063/067/069
    tagged — the loader's stale pre-063 subset hard-excluded causal_paths and
    agent_activities on an obsolete 42703 rationale."""
    from src.repositories.provenance import PROVENANCE_TAGGED_TABLES

    expected = {
        # 063_is_synthetic_provenance.sql
        "triggers",
        "business_metrics",
        "ml_predictions",
        "agent_activities",
        "causal_paths",
        "patient_journeys",
        "treatment_events",
        "hcp_profiles",
        "user_sessions",
        "hcp_intent_surveys",
        "episodic_memories",
        "ab_experiment_assignments",
        # 067_kpi_view_synthetic_exclusion.sql
        "data_source_tracking",
        "etl_pipeline_metrics",
        "ml_annotations",
        # 069_synthetic_provenance_shard09_tables.sql
        "ml_experiments",
        "ml_model_registry",
        "ml_training_runs",
        "ml_deployments",
        "ab_experiment_enrollments",
        "ab_experiment_results",
        "ml_observability_spans",
        "learning_signals",
        "feature_groups",
        "features",
        "feature_values",
    }
    assert set(PROVENANCE_TAGGED_TABLES) == expected


def test_ml_data_loader_uses_shared_tagged_set() -> None:
    from src.repositories import ml_data_loader
    from src.repositories.provenance import PROVENANCE_TAGGED_TABLES

    assert set(ml_data_loader.PROVENANCE_TAGGED_TABLES) == set(PROVENANCE_TAGGED_TABLES)
    assert "causal_paths" in ml_data_loader.PROVENANCE_TAGGED_TABLES
    assert "agent_activities" in ml_data_loader.PROVENANCE_TAGGED_TABLES


def test_apply_provenance_filter_for_table_skips_untagged() -> None:
    """The table-aware helper must NOT add the predicate on untagged tables
    (e.g. executive_insights) — that would be a real 42703."""
    from src.repositories.provenance import apply_provenance_filter_for_table

    tagged = _ChainableQuery(sync=True)
    apply_provenance_filter_for_table(tagged, "causal_paths")
    assert ("is_synthetic", False) in _eq_calls(tagged)

    untagged = _ChainableQuery(sync=True)
    apply_provenance_filter_for_table(untagged, "executive_insights")
    assert _eq_calls(untagged) == []

    opted = _ChainableQuery(sync=True)
    apply_provenance_filter_for_table(opted, "causal_paths", include_synthetic=True)
    assert _eq_calls(opted) == []


# =============================================================================
# Sentinel evaluators (non-repository family)
# =============================================================================


def _patch_sentinel_client(monkeypatch, client: _RecordingClient) -> None:
    import src.memory.sentinels.registry as registry

    monkeypatch.setattr(registry, "get_supabase_client", lambda: client)


@pytest.mark.asyncio
async def test_sentinel_new_causal_path_excludes_synthetic(monkeypatch) -> None:
    from src.memory.sentinels.registry import _eval_new_causal_path

    client = _RecordingClient(sync=True)
    _patch_sentinel_client(monkeypatch, client)
    await _eval_new_causal_path({}, "Kisqali", None)
    _assert_excludes(client.last("causal_paths"), "_eval_new_causal_path")


@pytest.mark.asyncio
async def test_sentinel_new_causal_path_opt_in_strictly_parsed(monkeypatch) -> None:
    """cfg opt-in goes through coerce_provenance_flag: 'false' stays real-mode."""
    from src.memory.sentinels.registry import _eval_new_causal_path

    client = _RecordingClient(sync=True)
    _patch_sentinel_client(monkeypatch, client)
    await _eval_new_causal_path({"include_synthetic": "false"}, "Kisqali", None)
    _assert_excludes(client.last("causal_paths"), "_eval_new_causal_path('false')")

    client2 = _RecordingClient(sync=True)
    _patch_sentinel_client(monkeypatch, client2)
    await _eval_new_causal_path({"include_synthetic": True}, "Kisqali", None)
    _assert_no_predicate(client2.last("causal_paths"), "_eval_new_causal_path(True)")


@pytest.mark.asyncio
async def test_sentinel_threshold_breach_excludes_synthetic_on_tagged(monkeypatch) -> None:
    from src.memory.sentinels.registry import _eval_threshold_breach

    client = _RecordingClient(sync=True)
    _patch_sentinel_client(monkeypatch, client)
    await _eval_threshold_breach(
        {"table": "causal_paths", "column": "confidence_level", "op": ">", "value": 0.5},
        "Kisqali",
    )
    _assert_excludes(client.last("causal_paths"), "_eval_threshold_breach")


@pytest.mark.asyncio
async def test_sentinel_threshold_breach_skips_predicate_on_untagged(monkeypatch) -> None:
    """executive_insights carries no is_synthetic column — predicate would 42703."""
    from src.memory.sentinels.registry import _eval_threshold_breach

    client = _RecordingClient(sync=True)
    _patch_sentinel_client(monkeypatch, client)
    await _eval_threshold_breach(
        {"table": "executive_insights", "column": "confidence_score", "op": ">", "value": 0.5},
        "Kisqali",
    )
    _assert_no_predicate(client.last("executive_insights"), "_eval_threshold_breach untagged")


@pytest.mark.asyncio
async def test_sentinel_freshness_excludes_synthetic_on_tagged(monkeypatch) -> None:
    from src.memory.sentinels.registry import _eval_freshness

    client = _RecordingClient(sync=True)
    _patch_sentinel_client(monkeypatch, client)
    await _eval_freshness(
        {"table": "triggers", "ts_column": "created_at", "max_age_hours": 24}, "Kisqali"
    )
    _assert_excludes(client.last("triggers"), "_eval_freshness")


@pytest.mark.asyncio
async def test_sentinel_invalidation_count_excludes_synthetic_on_tagged(monkeypatch) -> None:
    from src.memory.sentinels.registry import _eval_invalidation_count

    client = _RecordingClient(sync=True)
    _patch_sentinel_client(monkeypatch, client)
    await _eval_invalidation_count({"table": "ml_predictions"}, "Kisqali")
    _assert_excludes(client.last("ml_predictions"), "_eval_invalidation_count")


# =============================================================================
# Memory-lifecycle consolidator (same family as the sentinel scan)
# =============================================================================


@pytest.mark.asyncio
async def test_consolidator_promotion_candidates_exclude_synthetic(monkeypatch) -> None:
    """Synthetic causal paths must never be promoted to the semantic tier."""
    import src.memory.lifecycle.consolidator as consolidator_mod

    client = _RecordingClient(sync=True)
    monkeypatch.setattr(consolidator_mod, "get_supabase_client", lambda: client)

    consolidator = consolidator_mod.Consolidator()
    result = consolidator_mod.ConsolidationResult()
    await consolidator._promote_to_semantic(result, brand=None)
    _assert_excludes(client.last("causal_paths"), "_promote_to_semantic")


# =============================================================================
# Procedural memory learning_signals readers
# =============================================================================


def _patch_procedural_client(monkeypatch, client: _RecordingClient) -> None:
    import src.memory.procedural_memory as pm

    monkeypatch.setattr(pm, "get_supabase_client", lambda: client)


@pytest.mark.asyncio
async def test_get_recent_signals_excludes_synthetic(monkeypatch) -> None:
    from src.memory.procedural_memory import get_recent_signals

    client = _RecordingClient(sync=True)
    _patch_procedural_client(monkeypatch, client)
    await get_recent_signals()
    _assert_excludes(client.last("learning_signals"), "get_recent_signals")


@pytest.mark.asyncio
async def test_get_recent_signals_opt_in(monkeypatch) -> None:
    from src.memory.procedural_memory import get_recent_signals

    client = _RecordingClient(sync=True)
    _patch_procedural_client(monkeypatch, client)
    await get_recent_signals(include_synthetic=True)
    _assert_no_predicate(client.last("learning_signals"), "get_recent_signals")


@pytest.mark.asyncio
async def test_get_training_examples_excludes_synthetic(monkeypatch) -> None:
    from src.memory.procedural_memory import get_training_examples_for_agent

    client = _RecordingClient(sync=True)
    _patch_procedural_client(monkeypatch, client)
    await get_training_examples_for_agent("causal_impact")
    _assert_excludes(client.last("learning_signals"), "get_training_examples_for_agent")


@pytest.mark.asyncio
async def test_feedback_summaries_exclude_synthetic(monkeypatch) -> None:
    from src.memory.procedural_memory import (
        get_feedback_summary_for_agent,
        get_feedback_summary_for_trigger,
    )

    client = _RecordingClient(sync=True)
    _patch_procedural_client(monkeypatch, client)
    await get_feedback_summary_for_trigger("trig-1")
    _assert_excludes(client.last("learning_signals"), "get_feedback_summary_for_trigger")

    client2 = _RecordingClient(sync=True)
    _patch_procedural_client(monkeypatch, client2)
    await get_feedback_summary_for_agent("causal_impact")
    _assert_excludes(client2.last("learning_signals"), "get_feedback_summary_for_agent")


# =============================================================================
# id_column: live PKs pinned per repository (#894 — .eq("id") was a latent
# 42703 on every table whose PK is a natural key; values verified against the
# live schema's pg_index)
# =============================================================================


def test_id_column_matches_live_pk_per_repository() -> None:
    from src.repositories.agent_activity import AgentActivityRepository
    from src.repositories.business_metric import BusinessMetricRepository
    from src.repositories.causal_path import CausalPathRepository
    from src.repositories.expert_review import ExpertReviewRepository
    from src.repositories.patient_journey import PatientJourneyRepository
    from src.repositories.prediction import PredictionRepository
    from src.repositories.trigger import TriggerRepository
    from src.repositories.user_session import UserSessionRepository

    expected = {
        CausalPathRepository: "path_id",
        AgentActivityRepository: "activity_id",
        UserSessionRepository: "session_id",
        TriggerRepository: "trigger_id",
        BusinessMetricRepository: "metric_id",
        PatientJourneyRepository: "patient_journey_id",
        PredictionRepository: "prediction_id",
        # codex R1: create_renewal_review's get_by_id hit a nonexistent "id"
        ExpertReviewRepository: "review_id",
    }
    for repo_cls, pk in expected.items():
        assert repo_cls.id_column == pk, (
            f"{repo_cls.__name__}.id_column must be {pk!r} (live PK), got {repo_cls.id_column!r}"
        )


# =============================================================================
# Codex R1 regressions
# =============================================================================


@pytest.mark.asyncio
async def test_ml_data_loader_allows_and_filters_ml_predictions() -> None:
    """codex R1: ML_TABLES listed a nonexistent "predictions" table — the REAL
    tagged table is ml_predictions; loads of it must carry the predicate."""
    from src.repositories.ml_data_loader import ML_TABLES, MLDataLoader

    assert "ml_predictions" in ML_TABLES
    assert "predictions" not in ML_TABLES

    client = _RecordingClient(sync=True)
    loader = MLDataLoader(supabase_client=client)
    await loader.load_table_sample("ml_predictions", columns=["prediction_id"])
    _assert_excludes(client.last("ml_predictions"), "load_table_sample(ml_predictions)")


@pytest.mark.asyncio
async def test_drift_connector_models_projection_uses_live_columns() -> None:
    """codex R1: get_available_models selected name/version/metrics/created_at
    — none exist on ml_model_registry — so the 6-hourly production sweep
    42703'd into [] ("No production models found") forever. Pin the live
    projection + the provenance predicate."""
    from src.agents.drift_monitor.connectors.supabase_connector import (
        SupabaseDataConnector,
    )

    client = _RecordingClient(sync=True)
    connector = SupabaseDataConnector.__new__(SupabaseDataConnector)
    connector._client = client
    connector._initialized = True

    async def _noop():
        return None

    connector._ensure_initialized = _noop  # type: ignore[method-assign]

    await connector.get_available_models(stage="production")

    query = client.last("ml_model_registry")
    selects = [args for (name, args) in query.calls if name == "select"]
    assert selects, "no select recorded"
    sel = selects[0][0]
    for col in ("model_name", "model_version", "registered_at"):
        assert col in sel, f"projection missing live column {col!r}: {sel}"
    for col in ("metrics", "created_at"):
        assert col not in sel, f"projection still names nonexistent column {col!r}: {sel}"
    _assert_excludes(query, "get_available_models")


class _EnrollmentRecordingRepo:
    """Records the include_synthetic each read receives (codex R1 #3).

    Returns ONE fake assignment so the per-assignment enrollment read actually
    fires (codex R2 #5: an empty list made the second assertion vacuous).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def get_assignments(self, experiment_id, include_synthetic=False):
        self.calls.append(("get_assignments", include_synthetic))
        assignment = MagicMock()
        assignment.id = uuid4()
        assignment.variant = "control"
        return [assignment]

    async def get_enrollment_by_assignment(self, assignment_id, include_synthetic=False):
        self.calls.append(("get_enrollment_by_assignment", include_synthetic))
        return None


@pytest.mark.asyncio
async def test_enrollment_stats_threads_opt_in(monkeypatch) -> None:
    """codex R1: get_enrollment_stats had no opt-in path, so synthetic-only
    validation experiments became zero-count stats with no recourse."""
    import src.repositories.ab_experiment as ab_mod
    from src.services.enrollment import EnrollmentService

    recorder = _EnrollmentRecordingRepo()
    monkeypatch.setattr(ab_mod, "ABExperimentRepository", lambda *a, **kw: recorder)

    service = EnrollmentService.__new__(EnrollmentService)
    stats = await service.get_enrollment_stats(uuid4(), include_synthetic=True)
    assert ("get_assignments", True) in recorder.calls
    assert ("get_enrollment_by_assignment", True) in recorder.calls
    assert stats.total_assigned == 1

    recorder2 = _EnrollmentRecordingRepo()
    monkeypatch.setattr(ab_mod, "ABExperimentRepository", lambda *a, **kw: recorder2)
    await service.get_enrollment_stats(uuid4())
    assert ("get_assignments", False) in recorder2.calls
    assert ("get_enrollment_by_assignment", False) in recorder2.calls


# =============================================================================
# Codex R2 regressions
# =============================================================================


def _bare_connector(client):
    from src.agents.drift_monitor.connectors.supabase_connector import (
        SupabaseDataConnector,
    )

    connector = SupabaseDataConnector.__new__(SupabaseDataConnector)
    connector._client = client
    connector._initialized = True

    async def _noop():
        return None

    connector._ensure_initialized = _noop  # type: ignore[method-assign]
    return connector


@pytest.mark.asyncio
async def test_drift_query_features_excludes_synthetic() -> None:
    """codex R2: feature_values is tagged (069) — real drift checks must not
    ingest planted feature values."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock as _MM

    client = _RecordingClient(sync=True, data={"features": [{"id": "feat-1", "name": "f1"}]})
    connector = _bare_connector(client)

    window = _MM()
    window.start = datetime.now(timezone.utc)
    window.end = datetime.now(timezone.utc)

    await connector.query_features(["f1"], window)
    _assert_excludes(client.last("feature_values"), "query_features")

    client2 = _RecordingClient(sync=True, data={"features": [{"id": "feat-1", "name": "f1"}]})
    connector2 = _bare_connector(client2)
    await connector2.query_features(["f1"], window, include_synthetic=True)
    _assert_no_predicate(client2.last("feature_values"), "query_features opt-in")


@pytest.mark.asyncio
async def test_drift_available_features_excludes_synthetic() -> None:
    """codex R2: features is tagged (069) — the sweep must not auto-select
    planted feature names."""
    client = _RecordingClient(sync=True)
    connector = _bare_connector(client)
    await connector.get_available_features()
    _assert_excludes(client.last("features"), "get_available_features")


@pytest.mark.asyncio
async def test_drift_health_check_probes_live_pk() -> None:
    """codex R2: ml_predictions PK is prediction_id — probing "id" was a
    reachable 42703 that aborted the whole health_check try block."""
    client = _RecordingClient(sync=True)
    connector = _bare_connector(client)
    await connector.health_check()

    pred_query = client.last("ml_predictions")
    selects = [args for (name, args) in pred_query.calls if name == "select"]
    assert selects and selects[0][0] == "prediction_id", (
        f"health_check must probe the live PK prediction_id, got {selects}"
    )


@pytest.mark.asyncio
async def test_algorithm_trends_exclude_synthetic_runs(monkeypatch) -> None:
    """codex R2: ml_training_runs is tagged (069; 720/720 live synthetic) —
    planted runs must not skew model-selection trends."""
    import src.repositories.ml_data_loader as loader_mod
    from src.agents.ml_foundation.model_selector.nodes.historical_analyzer import (
        _query_algorithm_trends,
    )

    client = _RecordingClient(sync=True)

    class _FakeLoader:
        def __init__(self, *a, **kw):
            self.client = client

    monkeypatch.setattr(loader_mod, "MLDataLoader", _FakeLoader)
    await _query_algorithm_trends(["logistic_regression"])
    _assert_excludes(client.last("ml_training_runs"), "_query_algorithm_trends")
