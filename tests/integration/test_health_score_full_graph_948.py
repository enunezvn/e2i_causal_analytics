"""Integration (#952, re-homed from PR #948): the REAL health-score full graph.

These two tests drive the production route helper ``_execute_health_check(FULL)``
with NO agent mock — they instantiate the real ``HealthScoreAgent`` and run the
real LangGraph (component/model/pipeline/agent nodes + score composer). They are
integration tests by nature, so they live in the integration lane where the
harness provisions live Redis + MLflow service backends. They previously lived
in ``tests/unit/test_api/test_routes/test_health_score.py``; in the serviceless
unit shard the real agent's MLflow/Opik/Redis clients hit dead endpoints and
hung the xdist worker (``node down`` -> 20-min job cancel; issue #952).

Opik is intentionally OFF in CI (and stopped in prod). Against a dead Opik
endpoint the SDK's background uploader thread retries ``httpx.ConnectTimeout``
forever rather than no-op'ing, which can hang teardown. The autouse fixture
below pins ``OPIK_ENABLED=false`` and resets the tracer singleton so the real
agent never constructs the Opik client — matching the new prod-correct
``OPIK_ENABLED`` contract in ``src/agents/health_score/opik_tracer.py`` (and the
``OPIK_ENABLED=false`` set on the integration CI lane). The model/pipeline/agent
DATA still comes from patched ``_fetch_*`` readers, so this faithfully proves the
graph computes the three dimensions — the actual PR #948 defect (three nulls) —
without any mock of the health LOGIC.

The pure-logic regressions split out of these tests stay in the unit lane:
``tests/unit/test_agents/test_health_score/test_score_composer.py``
(``TestDegradedModelNullErrorRate``) and the ``_reconcile_full_provenance`` /
adapter tests still in ``test_api/test_routes/test_health_score.py``.
"""

from unittest.mock import patch

import pytest

from src.agents.health_score import opik_tracer as _hs_opik
from src.api.routes.health_score import (
    AgentHealth,
    CheckScope,
    DataProvenance,
    ModelHealth,
    ModelStatus,
    PipelineHealth,
    PipelineStatus,
    _execute_health_check,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _opik_off_for_real_graph(monkeypatch):
    """Run the real agent with Opik disabled so it never constructs the Opik
    client that would retry forever against the dead CI/prod endpoint. Resets
    the process-wide tracer singleton so the agent re-resolves OPIK_ENABLED on
    this run regardless of any earlier test in the xdist worker."""
    monkeypatch.setenv("OPIK_ENABLED", "false")
    _hs_opik.HealthScoreOpikTracer._instance = None
    _hs_opik.HealthScoreOpikTracer._initialized = False
    _hs_opik._tracer_instance = None
    yield
    _hs_opik.HealthScoreOpikTracer._instance = None
    _hs_opik.HealthScoreOpikTracer._initialized = False
    _hs_opik._tracer_instance = None


class _StubHealthClient:
    """Implements the component node's HealthClient Protocol (check(endpoint) ->
    {"ok": ...}) so every component reports healthy and the component dimension
    is measured."""

    async def check(self, endpoint: str):
        return {"ok": True}


@pytest.mark.asyncio
async def test_full_health_score_degraded_model_with_null_error_rate_no_crash():
    """REGRESSION (codex r2 HIGH): a degraded/unhealthy model whose error_rate is
    UNMEASURED (None, as the real ml_model_health_dashboard adapter passes through)
    must NOT crash the score composer's diagnosis (None > 0.1 TypeError). The
    composite must still complete with a real, non-null model score (NOT collapse
    to a failed/unknown composite)."""
    real_models = [
        ModelHealth(model_id="m1", model_name="alpha", status=ModelStatus.DEGRADED),
        ModelHealth(model_id="m2", model_name="beta", status=ModelStatus.UNHEALTHY),
    ]
    real_pipelines = [
        PipelineHealth(
            pipeline_name="rwd_ingest",
            last_run="2026-06-12T00:00:00+00:00",
            last_success="2026-06-12T00:00:00+00:00",
            rows_processed=10000,
            freshness_hours=12.0,
            status=PipelineStatus.HEALTHY,
        )
    ]
    real_agents = [
        AgentHealth(
            agent_name="orchestrator",
            tier=1,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        )
    ]

    with (
        patch(
            "src.api.routes.health_score._fetch_model_health",
            return_value=(real_models, DataProvenance.PARTIAL),
        ),
        patch(
            "src.api.routes.health_score._fetch_pipeline_health",
            return_value=(real_pipelines, DataProvenance.MEASURED),
        ),
        patch(
            "src.api.routes.health_score._fetch_agent_health",
            return_value=(real_agents, DataProvenance.PARTIAL),
        ),
        patch("src.agents.health_score.SupabaseHealthClient", _StubHealthClient),
    ):
        result = await _execute_health_check(CheckScope.FULL)

    # Diagnosis did not crash: model dimension is a REAL measurement (not None),
    # and the composite did not collapse to a failed/unknown 'F'.
    assert result.model_health_score is not None
    # 1 degraded (0.5) + 1 unhealthy (0.0) over 2 -> 0.25
    assert result.model_health_score == 0.25
    assert result.data_provenance == "partial"


@pytest.mark.asyncio
async def test_full_health_score_computes_real_dimensions_end_to_end():
    """END-TO-END (faithful, no agent mock): with the real store adapters wired
    over patched _fetch_* readers returning real-shaped rows, the full health
    SCORE must compute non-null model/pipeline/agent dimensions — proving the
    defect (three nulls) is fixed through the real agent graph, not a stubbed
    agent. Provenance is reconciled to 'partial' (not 'measured') because the
    model + agent readers are PARTIAL: the dimension SCORES are real, but some
    sub-fields (model accuracy/latency, agent runtime telemetry) are unsourced —
    so /full must not overclaim relative to /models and /agents."""
    real_models = [
        ModelHealth(model_id="m1", model_name="alpha", status=ModelStatus.HEALTHY),
        ModelHealth(model_id="m2", model_name="beta", status=ModelStatus.HEALTHY),
    ]
    real_pipelines = [
        PipelineHealth(
            pipeline_name="rwd_ingest",
            last_run="2026-06-12T00:00:00+00:00",
            last_success="2026-06-12T00:00:00+00:00",
            rows_processed=10000,
            freshness_hours=12.0,
            status=PipelineStatus.HEALTHY,
        )
    ]
    real_agents = [
        AgentHealth(
            agent_name="orchestrator",
            tier=1,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        ),
        AgentHealth(
            agent_name="causal_impact",
            tier=2,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        ),
    ]

    with (
        patch(
            "src.api.routes.health_score._fetch_model_health",
            return_value=(real_models, DataProvenance.PARTIAL),
        ),
        patch(
            "src.api.routes.health_score._fetch_pipeline_health",
            return_value=(real_pipelines, DataProvenance.MEASURED),
        ),
        patch(
            "src.api.routes.health_score._fetch_agent_health",
            return_value=(real_agents, DataProvenance.PARTIAL),
        ),
        patch("src.agents.health_score.SupabaseHealthClient", _StubHealthClient),
    ):
        result = await _execute_health_check(CheckScope.FULL)

    # The three previously-null dimensions are now REAL, computed from the tables.
    assert result.model_health_score is not None
    assert result.pipeline_health_score is not None
    assert result.agent_health_score is not None
    # All measured-and-healthy -> scores 1.0 (matches the per-dimension endpoints).
    assert result.model_health_score == 1.0
    assert result.pipeline_health_score == 1.0
    assert result.agent_health_score == 1.0
    # Honest provenance: scores are real, but model/agent sub-fields are unsourced
    # (readers PARTIAL) -> reconciled to 'partial', never overclaiming 'measured'.
    assert result.data_provenance == "partial"
