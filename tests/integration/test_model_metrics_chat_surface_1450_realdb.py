"""#1450 faithful integration: demo 5.3 answered from the LIVE tables.

No fakes anywhere in this file — the real ``factory._create_agent`` seam the
chat path uses, the real ``_ModelMetricsStoreAdapter``, the real readers over
``ml_model_health_dashboard`` / ``ml_performance_metrics`` / ``ml_model_registry``,
the real LangGraph nodes and the real score composer.

What it pins:

* the chat construction seam injects the route's real adapters (before #1450 it
  called ``HealthScoreAgent()`` bare, so the model dimension logged
  "No metrics_store wired - model health is UNKNOWN");
* the demo-5.3 question, routed through the dispatcher's own input resolver,
  comes back with ROC-AUC / calibration slope / Brier for the Kisqali models —
  each value equal to the row in ``ml_performance_metrics`` — with the model
  version, evaluation cohort, cohort size and as-of date attached;
* every reported metric belongs to ONE evaluation event (never a fresh holdout
  ROC-AUC paired with a stale backtest Brier).

Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_model_metrics_chat_surface_1450_realdb.py'
"""

from __future__ import annotations

import os
import re
import uuid

import pytest

from tests.integration._asyncio_compat import run_sync

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]

DEMO_53 = "What is the ROC-AUC and calibration of the current Kisqali model?"

# Metric keys whose live rows the summary must reproduce exactly.
_PINNED = ("auc_roc", "calibration_slope", "brier_score")


@pytest.fixture
def owned_session():
    """Mint a session id for a real ``check_health`` and delete what it deposits.

    ``HealthScoreAgent.check_health`` contributes every completed run to memory
    (#879). A MEASURED grade-A check is not "significant" so it usually stores
    nothing episodic, but that is a property of today's data — the fixture must
    not depend on it. Every session this suite mints is cleaned on teardown so a
    test run can never leave rows in the live episodic store (#1420).
    """
    minted: list[str] = []

    def _mint() -> str:
        session_id = str(uuid.uuid4())
        minted.append(session_id)
        return session_id

    yield _mint

    if not minted:
        return
    try:
        from src.api.dependencies.supabase_client import get_supabase

        db = get_supabase()
        for session_id in minted:
            db.table("episodic_memories").delete().eq("session_id", session_id).execute()
    except Exception as exc:  # pragma: no cover - teardown must not mask failures
        pytest.fail(f"could not clean episodic rows for {minted}: {exc}")


def _real_health_score_agent():
    """The agent EXACTLY as the chat path builds it."""
    from src.agents.factory import AGENT_REGISTRY_CONFIG, _create_agent

    config = AGENT_REGISTRY_CONFIG["health_score"]
    agent = _create_agent(
        module_path=config["module"],
        class_name=config["class_name"],
    )
    assert agent is not None, "health_score must be constructible on the chat path"
    return agent


def test_chat_seam_wires_the_route_adapters():
    from src.api.routes.health_score import (
        _AgentRegistryAdapter,
        _ModelMetricsStoreAdapter,
        _PipelineStoreAdapter,
    )

    agent = _real_health_score_agent()
    assert isinstance(agent.metrics_store, _ModelMetricsStoreAdapter)
    assert isinstance(agent.pipeline_store, _PipelineStoreAdapter)
    assert isinstance(agent.agent_registry, _AgentRegistryAdapter)
    assert agent.health_client is not None


def test_live_adapter_matches_ml_performance_metrics():
    """Every value the adapter reports is the value in the table."""
    from src.api.dependencies.supabase_client import get_supabase
    from src.api.routes.health_score import _ModelMetricsStoreAdapter

    db = get_supabase()
    adapter = _ModelMetricsStoreAdapter()
    model_ids = _await(adapter.get_active_models())
    assert model_ids, "the live registry must expose at least one real model"

    checked = 0
    for model_id in model_ids:
        payload = _await(adapter.get_model_metrics(model_id, "24h"))
        eval_metrics = payload.get("eval_metrics") or {}
        if not eval_metrics:
            continue
        as_of = payload["eval_as_of"]
        cohort = payload["eval_cohort"]
        rows = (
            db.table("ml_performance_metrics")
            .select("metric_name, metric_value, sample_size, measured_at, source")
            .eq("model_id", model_id)
            .eq("source", cohort)
            .execute()
            .data
            or []
        )
        same_event = {
            r["metric_name"]: float(r["metric_value"])
            for r in rows
            if str(r["measured_at"]) == str(as_of)
        }
        assert same_event, f"no live rows for {model_id} @ {as_of} ({cohort})"
        for name, value in eval_metrics.items():
            # Coherence: the metric must come from the SAME evaluation event the
            # adapter attributed it to, not from another row that happens to
            # carry the same metric name.
            assert name in same_event, f"{name} is not part of the {as_of} {cohort} event"
            assert value == pytest.approx(same_event[name]), name
        checked += 1
    assert checked, "no live model carried evaluation metrics"


def test_demo_53_answers_with_the_metrics(owned_session):
    """The full production path: dispatcher input resolution -> real agent."""
    from src.agents.orchestrator.nodes.dispatcher import _resolve_health_score_input

    kwargs = _resolve_health_score_input(
        {"query": DEMO_53, "session_id": owned_session()}, {"parameters": {}}
    )
    assert kwargs["scope"] == "models", "the dispatcher already scopes 5.3 to models"

    agent = _real_health_score_agent()
    output = _await(
        agent.check_health(
            scope=kwargs["scope"], query=kwargs["query"], session_id=kwargs["session_id"]
        )
    )
    summary = output.health_summary

    assert summary.startswith("Model quality metrics (requested: ROC-AUC, calibration slope)")
    lowered = summary.lower()
    assert "roc-auc" in lowered and "calibration slope" in lowered and "brier" in lowered
    assert "holdout" in lowered, "the evaluation cohort must be named"
    assert re.search(r"as of \d{4}-\d{2}-\d{2}", summary), "the as-of date must be stated"
    assert re.search(r"\bn=\d+", summary), "the cohort size must be stated"
    assert re.search(r"\bv\d", summary), "the model version must be stated"

    # Only the models the question names.
    assert "kisqali" in lowered
    assert "fabhalta" not in lowered and "remibrutinib" not in lowered

    # The dimension is genuinely measured — not the #1447 UNKNOWN placeholder.
    assert output.model_health_score is not None
    assert output.data_provenance != "unknown"


def test_reported_numbers_equal_the_live_rows(owned_session):
    from src.api.dependencies.supabase_client import get_supabase

    agent = _real_health_score_agent()
    output = _await(agent.check_health(scope="models", query=DEMO_53, session_id=owned_session()))
    summary = output.health_summary

    db = get_supabase()
    registry = (
        db.table("ml_model_registry")
        .select("id, model_name")
        .eq("is_synthetic", False)
        .eq("model_name", "hcp_adoption_kisqali_goldstd_lr_v1")
        .execute()
        .data
        or []
    )
    assert registry, "the Kisqali champion must be registered"
    model_id = registry[0]["id"]

    rows = (
        db.table("ml_performance_metrics")
        .select("metric_name, metric_value, measured_at, source")
        .eq("model_id", model_id)
        .eq("source", "holdout")
        .in_("metric_name", list(_PINNED))
        .order("measured_at", desc=True)
        .execute()
        .data
        or []
    )
    latest = max(str(r["measured_at"]) for r in rows)
    live = {
        r["metric_name"]: float(r["metric_value"]) for r in rows if str(r["measured_at"]) == latest
    }
    for name in _PINNED:
        assert f"{live[name]:.3f}" in summary, (
            f"{name}={live[name]:.3f} from the live table is not in the answer:\n{summary}"
        )


def _await(coro):
    """Sync -> async boundary via the repo's sanctioned helper.

    Bare ``asyncio.run`` is rejected in ``tests/integration/`` by
    ``test_no_bare_asyncio_run_in_integration_tests.py`` — it is a latent victim
    of the RAGAS event-loop pollution chain (#215/#218/#220). ``run_sync`` builds
    an explicit fresh loop instead.
    """
    return run_sync(coro)
