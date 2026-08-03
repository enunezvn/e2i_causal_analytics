"""#1450 — a model-quality question must return the METRICS, not a grade.

Demo question 5.3 ("What is the ROC-AUC and calibration of the current Kisqali
model?") routes correctly (LLM intent=system_health 0.92; ``health_score`` is
the gold agent in ``benchmark_queries_gold.jsonl``) and still cannot be
answered, for two separate reasons:

1. **No metrics store on the chat path.** ``src/api/routes/health_score.py``
   already builds the REAL adapters (``_build_real_health_stores``) for the REST
   route, but every chat surface reaches the agent through
   ``cognitive.get_orchestrator() -> factory.create_agent_registry() ->
   factory._create_agent()``, which calls ``HealthScoreAgent()`` with no stores.
   The model dimension therefore fail-closes to UNMEASURED
   ("No metrics_store wired - model health is UNKNOWN").
2. **The composer emits a composite grade.** Even fully measured, the summary
   the synthesizer reads (``AGENT_RESPONSE_FIELDS["health_score"] ==
   ["health_summary", "narrative"]``) is "Model health is excellent (Grade: A
   ...)" — not the ROC-AUC / calibration / Brier the reviewer asked for.

Guards here:
  (a) the factory seam the CHAT path uses injects the SAME real adapters the
      REST route uses (no duplicate adapter, no mock);
  (b) the real ``_ModelMetricsStoreAdapter`` carries the named evaluation
      metrics plus model version, evaluation cohort and as-of date;
  (c) a metric-naming query renders those numbers in ``health_summary``;
  (d) when a metric is NOT recorded, or the dimension was not measured at all,
      the prose says so — the #1447 UNKNOWN path — and NEVER prints a number.

Offline: the Supabase boundary is faked with recorded row shapes; the readers,
adapters, graph nodes and composer under test all run for real. A live-DB
counterpart lives in
``tests/integration/test_model_metrics_chat_surface_1450_realdb.py``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.health_score.nodes.score_composer import ScoreComposerNode

EMIT_TARGET = "src.agents.health_score.nodes.score_composer.emit_recipient_signal"

DEMO_53 = "What is the ROC-AUC and calibration of the current Kisqali model?"

# Real values read from the live platform on 2026-08-03 (holdout evaluation of
# the production champion, ml_performance_metrics @ 2026-06-01, n=1000).
KISQALI_HCP_ADOPTION = {
    "model_id": "8d1244df-7c38-435e-820f-1f201e51af24",
    "model_name": "hcp_adoption_kisqali_goldstd_lr_v1",
    "model_version": "1.0",
    "model_stage": "production",
    "auc_roc": 0.767697,
    "brier_score": 0.190392,
    "calibration_slope": 0.925305,
    "eval_cohort": "holdout",
    "eval_sample_size": 1000,
    "eval_as_of": "2026-06-01T00:00:00+00:00",
}


# =============================================================================
# (a) THE CHAT SEAM — factory injects the REAL adapters
# =============================================================================


class TestChatPathWiresRealStores:
    """``factory._create_agent`` is the ONLY construction site the chat path
    uses (cognitive.get_orchestrator -> create_agent_registry -> _create_agent).
    It must inject the same real adapters ``_execute_health_check`` builds."""

    def test_create_agent_injects_the_route_s_real_metrics_store(self):
        from src.agents.factory import AGENT_REGISTRY_CONFIG, _create_agent
        from src.api.routes.health_score import _ModelMetricsStoreAdapter

        config = AGENT_REGISTRY_CONFIG["health_score"]
        agent = _create_agent(
            module_path=config["module"],
            class_name=config["class_name"],
        )

        assert agent is not None, "health_score must be constructible"
        assert isinstance(agent.metrics_store, _ModelMetricsStoreAdapter), (
            "the chat path must reuse the EXISTING route adapter "
            "(_ModelMetricsStoreAdapter), not a duplicate or a mock"
        )

    def test_create_agent_injects_pipeline_agent_and_component_backends(self):
        """All four dimensions, matching ``_execute_health_check`` — a chat
        'full' health check must not be narrower than the REST one."""
        from src.agents.factory import AGENT_REGISTRY_CONFIG, _create_agent
        from src.api.routes.health_score import (
            _AgentRegistryAdapter,
            _PipelineStoreAdapter,
        )

        config = AGENT_REGISTRY_CONFIG["health_score"]
        agent = _create_agent(
            module_path=config["module"],
            class_name=config["class_name"],
        )

        assert isinstance(agent.pipeline_store, _PipelineStoreAdapter)
        assert isinstance(agent.agent_registry, _AgentRegistryAdapter)
        assert agent.health_client is not None, "component health must be measurable too"

    def test_chat_provenance_matches_the_rest_route(self):
        """Wiring the same adapters must not make chat claim a FULLER provenance
        than /health-score/full, which downgrades "measured" -> "partial"
        because the model/agent readers have unsourced sub-fields."""
        from src.agents.factory import AGENT_REGISTRY_CONFIG, _create_agent
        from src.api.routes.health_score import DataProvenance

        config = AGENT_REGISTRY_CONFIG["health_score"]
        agent = _create_agent(
            module_path=config["module"],
            class_name=config["class_name"],
        )
        # Simulate the readers having loaded: model reader is ALWAYS PARTIAL.
        agent.metrics_store.provenance = DataProvenance.PARTIAL
        assert agent._reconcile_provenance("measured") == "partial"
        # Conservative composites are already honest and stay untouched.
        assert agent._reconcile_provenance("unknown") == "unknown"
        assert agent._reconcile_provenance("partial") == "partial"

    def test_reconcile_is_inert_without_provenance_bearing_stores(self):
        """A store that reports no provenance contributes nothing (the unit-test
        doubles and any third-party store)."""
        from src.agents.health_score.agent import HealthScoreAgent

        agent = HealthScoreAgent(enable_mlflow=False, enable_opik=False, enable_memory=False)
        assert agent._reconcile_provenance("measured") == "measured"

    def test_model_health_node_receives_the_store(self):
        """The store must reach the NODE, not just the agent attribute — the
        graph is built once in ``__init__`` from the constructor kwargs."""
        from src.agents.factory import AGENT_REGISTRY_CONFIG, _create_agent

        config = AGENT_REGISTRY_CONFIG["health_score"]
        agent = _create_agent(
            module_path=config["module"],
            class_name=config["class_name"],
        )
        assert agent.metrics_store is not None
        # The full graph is the one the models/full scopes execute.
        assert agent._full_graph is not None


class TestAdapterCacheIsBounded:
    """The chat path holds ONE agent for the process lifetime (the orchestrator
    singleton), so the adapters' load-once cache would pin the first reading
    forever. The REST route builds a fresh adapter per request and never noticed.
    """

    def test_metrics_adapter_cache_expires(self):
        from src.api.routes import health_score as hs

        adapter = hs._ModelMetricsStoreAdapter()
        assert hasattr(hs, "_ADAPTER_CACHE_TTL_SECONDS"), (
            "a process-lifetime adapter needs a bounded cache TTL"
        )
        assert hs._ADAPTER_CACHE_TTL_SECONDS > 0
        # Freshly built -> nothing cached yet, so a load is due.
        assert adapter._cache_is_stale() is True


# =============================================================================
# (b) THE REAL ADAPTER carries the named eval metrics + provenance
# =============================================================================


class _FakeQuery:
    """Minimal supabase-py query-builder stub honouring the filters the readers
    under test actually use. Only the network boundary is faked."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self._eq: List[tuple] = []
        self._in: List[tuple] = []
        self._exclude_null: List[str] = []
        self._negate = False
        self._limit: int | None = None

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self._eq.append((col, val))
        return self

    def in_(self, col, vals):
        self._in.append((col, list(vals)))
        return self

    def gte(self, *a, **k):
        return self

    def order(self, col, desc=False, **k):
        self._rows = sorted(self._rows, key=lambda r: str(r.get(col) or ""), reverse=bool(desc))
        return self

    def limit(self, n):
        self._limit = n
        return self

    @property
    def not_(self):
        self._negate = True
        return self

    def is_(self, col, val):
        if self._negate and str(val).lower() == "null":
            self._exclude_null.append(col)
        self._negate = False
        return self

    def execute(self):
        from unittest.mock import MagicMock

        rows = list(self._rows)
        for col, val in self._eq:
            rows = [r for r in rows if r.get(col, val) == val]
        for col, vals in self._in:
            rows = [r for r in rows if r.get(col) in vals]
        for col in self._exclude_null:
            rows = [r for r in rows if r.get(col) is not None]
        if self._limit is not None:
            rows = rows[: self._limit]
        return MagicMock(data=rows)


class _FakeDB:
    def __init__(self, tables: Dict[str, List[Dict[str, Any]]]):
        self._tables = tables

    def table(self, name):
        return _FakeQuery(self._tables.get(name, []))


def _eval_rows(model_id: str, measured_at: str, source: str, n: int, values: Dict[str, float]):
    return [
        {
            "model_id": model_id,
            "metric_name": name,
            "metric_value": value,
            "sample_size": n,
            "measured_at": measured_at,
            "source": source,
            "data_split": "production",
        }
        for name, value in values.items()
    ]


def _platform_db() -> _FakeDB:
    """Recorded shapes of the three live tables (values are the REAL ones read
    from the platform on 2026-08-03)."""
    mid = KISQALI_HCP_ADOPTION["model_id"]
    other = "4ec55d13-46c8-4df4-9ec8-7723fad67fb3"
    return _FakeDB(
        {
            "ml_model_health_dashboard": [
                {
                    "model_id": mid,
                    "model_name": KISQALI_HCP_ADOPTION["model_name"],
                    "model_stage": "production",
                    "health_status": "healthy",
                    "latest_metric_value": 0.767697,
                    "primary_metric": "auc_roc",
                    "is_synthetic": False,
                    "latest_accuracy": 0.715,
                    "latest_auc_roc": 0.767697,
                    "latest_f1": 0.621514,
                },
                {
                    "model_id": other,
                    "model_name": "initiation_kisqali_goldstd_lr_v1",
                    "model_stage": "staging",
                    "health_status": "healthy",
                    "latest_metric_value": 0.851908,
                    "primary_metric": "auc_roc",
                    "is_synthetic": False,
                    "latest_accuracy": 0.786626,
                    "latest_auc_roc": 0.851908,
                    "latest_f1": 0.664756,
                },
            ],
            "ml_model_registry": [
                {
                    "id": mid,
                    "model_name": KISQALI_HCP_ADOPTION["model_name"],
                    "model_version": "1.0",
                    "stage": "production",
                    "is_synthetic": False,
                },
                {
                    "id": other,
                    "model_name": "initiation_kisqali_goldstd_lr_v1",
                    "model_version": "1.0",
                    "stage": "staging",
                    "is_synthetic": False,
                },
            ],
            "ml_performance_metrics": (
                # An OLDER walk-forward backtest must lose to the newer holdout.
                _eval_rows(
                    mid,
                    "2026-05-01T00:00:00+00:00",
                    "backtest_wf",
                    146,
                    {"auc_roc": 0.725284, "brier_score": 0.24, "calibration_slope": 1.4},
                )
                + _eval_rows(
                    mid,
                    "2026-06-01T00:00:00+00:00",
                    "holdout",
                    1000,
                    {
                        "auc_roc": 0.767697,
                        "brier_score": 0.190392,
                        "calibration_slope": 0.925305,
                        "accuracy": 0.715,
                        "f1": 0.621514,
                    },
                )
                + _eval_rows(
                    other,
                    "2026-07-21T00:00:00+00:00",
                    "holdout",
                    1645,
                    {
                        "auc_roc": 0.851908,
                        "brier_score": 0.145714,
                        "calibration_slope": 1.049494,
                    },
                )
            ),
        }
    )


def _patch_health_client(db):
    return patch("src.api.routes.health_score._health_source_client", return_value=db)


class TestRealAdapterCarriesEvalMetrics:
    def test_adapter_returns_roc_auc_calibration_and_brier(self):
        from src.api.routes.health_score import _ModelMetricsStoreAdapter

        with _patch_health_client(_platform_db()):
            adapter = _ModelMetricsStoreAdapter()
            metrics = _run(adapter.get_model_metrics(KISQALI_HCP_ADOPTION["model_id"], "24h"))

        eval_metrics = metrics.get("eval_metrics") or {}
        assert eval_metrics.get("auc_roc") == pytest.approx(0.767697)
        assert eval_metrics.get("calibration_slope") == pytest.approx(0.925305)
        assert eval_metrics.get("brier_score") == pytest.approx(0.190392)

    def test_adapter_returns_version_cohort_and_as_of(self):
        from src.api.routes.health_score import _ModelMetricsStoreAdapter

        with _patch_health_client(_platform_db()):
            adapter = _ModelMetricsStoreAdapter()
            metrics = _run(adapter.get_model_metrics(KISQALI_HCP_ADOPTION["model_id"], "24h"))

        assert metrics.get("model_version") == "1.0"
        assert metrics.get("model_stage") == "production"
        assert metrics.get("eval_cohort") == "holdout"
        assert metrics.get("eval_sample_size") == 1000
        assert str(metrics.get("eval_as_of") or "").startswith("2026-06-01")

    def test_adapter_reports_ONE_coherent_evaluation_not_a_mix(self):
        """Mixing a fresh holdout AUC with a stale backtest Brier would be a
        fabricated 'evaluation'. All reported values come from the single
        latest evaluation event."""
        from src.api.routes.health_score import _ModelMetricsStoreAdapter

        with _patch_health_client(_platform_db()):
            adapter = _ModelMetricsStoreAdapter()
            metrics = _run(adapter.get_model_metrics(KISQALI_HCP_ADOPTION["model_id"], "24h"))

        eval_metrics = metrics.get("eval_metrics") or {}
        # 0.24 / 1.4 are the older backtest values — they must NOT appear.
        assert eval_metrics.get("brier_score") != pytest.approx(0.24)
        assert eval_metrics.get("calibration_slope") != pytest.approx(1.4)

    def test_model_health_node_carries_eval_metrics_into_state(self):
        """The node is the only path from store -> state; without a passthrough
        the composer can never see the numbers."""
        from src.agents.health_score.nodes.model_health import ModelHealthNode
        from src.api.routes.health_score import _ModelMetricsStoreAdapter

        with _patch_health_client(_platform_db()):
            node = ModelHealthNode(metrics_store=_ModelMetricsStoreAdapter())
            state = _run(node.execute({"check_scope": "models", "query": DEMO_53}))

        assert state["model_health_measured"] is True
        by_name = {m.get("model_name"): m for m in state["model_metrics"]}
        champ = by_name[KISQALI_HCP_ADOPTION["model_name"]]
        assert (champ.get("eval_metrics") or {}).get("auc_roc") == pytest.approx(0.767697)
        assert champ.get("model_version") == "1.0"
        assert champ.get("eval_sample_size") == 1000


# =============================================================================
# (c) THE ANSWER — a metric-naming query renders the metrics
# =============================================================================


def _measured_models_state(query: str = DEMO_53) -> Dict[str, Any]:
    """A models-scoped check where the model dimension WAS measured."""
    return {
        "query": query,
        "check_scope": "models",
        "model_metrics": [
            {
                "model_id": KISQALI_HCP_ADOPTION["model_id"],
                "model_name": KISQALI_HCP_ADOPTION["model_name"],
                "model_version": "1.0",
                "model_stage": "production",
                "accuracy": 0.715,
                "precision": None,
                "recall": None,
                "f1_score": 0.621514,
                "auc_roc": 0.767697,
                "prediction_latency_p50_ms": None,
                "prediction_latency_p99_ms": None,
                "predictions_last_24h": None,
                "error_rate": None,
                "status": "healthy",
                "eval_metrics": {
                    "auc_roc": 0.767697,
                    "calibration_slope": 0.925305,
                    "brier_score": 0.190392,
                    "accuracy": 0.715,
                    "f1": 0.621514,
                },
                "eval_cohort": "holdout",
                "eval_sample_size": 1000,
                "eval_as_of": "2026-06-01T00:00:00+00:00",
            },
            {
                "model_id": "a11a50e2-42ed-4233-8711-bbdf233d3a0d",
                "model_name": "initiation_fabhalta_goldstd_lr_v1",
                "model_version": "1.0",
                "model_stage": "staging",
                "accuracy": 0.777391,
                "precision": None,
                "recall": None,
                "f1_score": 0.660177,
                "auc_roc": 0.84183,
                "prediction_latency_p50_ms": None,
                "prediction_latency_p99_ms": None,
                "predictions_last_24h": None,
                "error_rate": None,
                "status": "healthy",
                "eval_metrics": {"auc_roc": 0.84183, "calibration_slope": 1.01},
                "eval_cohort": "holdout",
                "eval_sample_size": 1645,
                "eval_as_of": "2026-07-21T00:00:00+00:00",
            },
        ],
        "model_health_score": 1.0,
        "model_health_measured": True,
        "total_latency_ms": 0,
        "errors": [],
    }


async def _compose(state: Dict[str, Any]) -> Dict[str, Any]:
    node = ScoreComposerNode()
    with patch(EMIT_TARGET, new=AsyncMock(return_value=None)):
        return await node.execute(state)


def _run(coro):
    import asyncio

    return asyncio.run(coro)


@pytest.mark.asyncio
class TestMetricQuestionReturnsMetrics:
    async def test_summary_states_roc_auc_calibration_and_brier(self):
        out = await _compose(_measured_models_state())
        summary = out["health_summary"]

        assert "0.768" in summary or "0.7677" in summary, summary
        assert "0.925" in summary, summary
        assert "0.190" in summary, summary

    async def test_summary_names_the_metrics_not_just_numbers(self):
        summary = (await _compose(_measured_models_state()))["health_summary"]
        lowered = summary.lower()
        assert "roc-auc" in lowered
        assert "calibration" in lowered
        assert "brier" in lowered

    async def test_summary_carries_version_cohort_and_as_of(self):
        summary = (await _compose(_measured_models_state()))["health_summary"]
        assert "1.0" in summary
        assert "holdout" in summary.lower()
        assert "1000" in summary or "1,000" in summary
        assert "2026-06-01" in summary

    async def test_summary_scopes_to_the_model_the_user_named(self):
        """'Kisqali' names 1 of the 2 measured models; the Fabhalta model must
        not be presented as the answer."""
        summary = (await _compose(_measured_models_state()))["health_summary"]
        assert "hcp_adoption_kisqali_goldstd_lr_v1" in summary
        assert "initiation_fabhalta_goldstd_lr_v1" not in summary

    async def test_metrics_lead_the_summary_not_the_grade(self):
        """The reviewer asked for metrics; the composite grade may follow but
        must not be the answer."""
        summary = (await _compose(_measured_models_state()))["health_summary"]
        first_line = summary.splitlines()[0].lower()
        assert "grade" not in first_line, first_line

    async def test_training_signal_carries_only_the_summary_template_output(self):
        """``summary_template`` is the optimised recipient field; the #1450 block
        comes from its own templates. Emitting the composed string would train
        GEPA on text ``summary_template`` never produced."""
        node = ScoreComposerNode()
        emit = AsyncMock(return_value=None)
        with patch(EMIT_TARGET, new=emit):
            out = await node.execute(_measured_models_state())

        assert emit.await_count == 1, "exactly one summary signal"
        emitted = emit.await_args.kwargs["generated_output"]
        assert "ROC-AUC" not in emitted, emitted
        assert emitted == (
            "Model health is excellent (Grade: A, Score: 100.0/100). All systems operational."
        )
        # ...while the RETURNED summary does carry the metrics.
        assert "ROC-AUC" in out["health_summary"]

    async def test_composite_payload_is_unchanged(self):
        """Narration only — the structured score/grade contract is untouched."""
        out = await _compose(_measured_models_state())
        assert out["model_health_score"] == 1.0
        assert out["data_provenance"] == "partial"
        assert out["health_grade"] == "A"


@pytest.mark.asyncio
class TestHonestyWhenNotMeasured:
    async def test_unrecorded_metric_is_named_and_never_numbered(self):
        """PSI is not among the recorded evaluation metrics. The answer must say
        so instead of substituting a different metric's number for it."""
        state = _measured_models_state("What is the PSI of the Kisqali model?")
        summary = (await _compose(state))["health_summary"]
        assert "psi" in summary.lower()
        assert re.search(r"not\s+recorded|no[t]?\s+available|unknown", summary.lower()), summary

    async def test_unmeasured_model_dimension_says_unknown_not_a_number(self):
        """#1447 path: nothing measured -> UNKNOWN prose, no fabricated metric."""
        state = {
            "query": DEMO_53,
            "check_scope": "models",
            "model_metrics": [],
            "model_health_measured": False,
            "total_latency_ms": 0,
            "errors": [],
        }
        out = await _compose(state)
        summary = out["health_summary"]
        assert "UNKNOWN" in summary
        assert not re.search(r"\b0\.\d{2,}", summary), (
            "no metric-looking number may appear when nothing was measured: " + summary
        )
        # The deliberate F1 anti-fabrication payload is preserved.
        assert out["overall_health_score"] == 0.0
        assert out["health_grade"] == "F"
        assert out["data_provenance"] == "unknown"

    async def test_measured_models_without_eval_metrics_say_not_recorded(self):
        """The dimension is measured (status known) but no evaluation metrics
        exist for the model — say so, do not fall back to the grade."""
        state = _measured_models_state()
        for m in state["model_metrics"]:
            m["eval_metrics"] = {}
            m["eval_cohort"] = None
            m["eval_sample_size"] = None
            m["eval_as_of"] = None
        summary = (await _compose(state))["health_summary"]
        assert re.search(r"not\s+recorded|no\s+evaluation\s+metrics", summary.lower()), summary
        assert not re.search(r"\b0\.\d{3}", summary), summary

    async def test_unavailable_block_precedes_the_1447_unknown_narration(self):
        """Composition order pin: the targeted answer first, then #1447's
        scope-named UNKNOWN narration. Neither may be dropped."""
        state = {
            "query": DEMO_53,
            "check_scope": "models",
            "model_metrics": [],
            "model_health_measured": False,
            "total_latency_ms": 0,
            "errors": [],
        }
        summary = (await _compose(state))["health_summary"]
        assert summary.startswith("Model quality metrics (requested: ROC-AUC, calibration slope)")
        assert "Model health status is UNKNOWN - nothing was measured." in summary
        assert summary.index("Model quality metrics") < summary.index("Model health status is")

    async def test_no_registered_model_matches_is_disclosed(self):
        """A brand with no registered model (Xolair is not in the data model)
        must not silently be answered with another brand's numbers."""
        state = _measured_models_state("What is the ROC-AUC of the Xolair model?")
        summary = (await _compose(state))["health_summary"]
        assert re.search(r"no registered model", summary.lower()), summary


@pytest.mark.asyncio
class TestNonMetricQueriesAreUnchanged:
    async def test_plain_health_question_renders_the_historical_summary(self):
        """Regression pin: the metrics block must appear ONLY for a
        metric-naming question."""
        state = _measured_models_state("Is the system healthy?")
        summary = (await _compose(state))["health_summary"]
        assert summary.startswith("Model health is excellent (Grade: A, Score: 100.0/100).")
        assert "ROC-AUC" not in summary

    async def test_ambiguous_business_words_do_not_trigger_the_block(self):
        """ "recall" can mean a product recall; "accuracy"/"precision" are used
        loosely about forecasts. They only count as a model-quality request when
        the question is visibly about a model."""
        state = _measured_models_state("Was there a recall affecting our supply?")
        summary = (await _compose(state))["health_summary"]
        assert "Model quality metrics" not in summary, summary

    async def test_ambiguous_words_DO_trigger_with_model_context(self):
        state = _measured_models_state("What is the precision and recall of the Kisqali model?")
        summary = (await _compose(state))["health_summary"]
        assert summary.startswith("Model quality metrics (requested: precision, recall)")
        assert "hcp_adoption_kisqali_goldstd_lr_v1" in summary

    async def test_full_scope_summary_is_byte_identical(self):
        state = {
            "query": "How is the platform doing?",
            "check_scope": "full",
            "component_health_score": 0.9,
            "component_health_measured": True,
            "model_health_score": 0.85,
            "model_health_measured": True,
            "pipeline_health_score": 0.8,
            "pipeline_health_measured": True,
            "agent_health_score": 0.9,
            "agent_health_measured": True,
            "total_latency_ms": 0,
            "errors": [],
        }
        summary = (await _compose(state))["health_summary"]
        assert (
            summary == "System health is good (Grade: B, Score: 86.0/100). All systems operational."
        )
