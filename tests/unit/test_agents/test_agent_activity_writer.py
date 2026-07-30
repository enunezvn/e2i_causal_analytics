"""#1355: Tier-2/3 runtime writers persist a real agent_activities row.

Before this fix only experiment_designer's memory hooks inserted
agent_activities rows (#883 §5) — heterogeneous_optimizer, causal_impact and
gap_analyzer completed analyses that no chat agent-analysis query could ever
read. These tests pin:

* the shared ``persist_agent_activity`` helper writes REAL columns only
  (schema SSOT database/core/e2i_ml_complete_v3_schema.sql:610, mirroring the
  experiment_designer idiom), stamps ``is_synthetic=False``, and NEVER raises
  (a failed activity write must never fail the analysis);
* each of the three agents' ``contribute_to_memory`` persists one activity row
  on a completed analysis, with brand resolvable at
  ``analysis_results->>'brand'`` (the field _query_agent_analysis filters on);
* a failed analysis persists nothing.

No MagicMock where hasattr semantics matter: the fake supabase client is a
minimal recorder implementing exactly table().insert().execute().
"""

from typing import Any, Dict, List, Optional

import pytest

from src.agents.activity_writer import persist_agent_activity

# ---------------------------------------------------------------------------
# Fakes (sync supabase client shape used by src.repositories.get_supabase_client)
# ---------------------------------------------------------------------------


class _FakeInsert:
    def __init__(self, sink: List[Dict[str, Any]], record: Dict[str, Any], fail: bool):
        self._sink = sink
        self._record = record
        self._fail = fail

    def execute(self):
        if self._fail:
            raise RuntimeError("insert failed (simulated)")
        self._sink.append(self._record)

        class _Resp:
            data = [self._record]

        return _Resp()


class _FakeTable:
    def __init__(self, sink: List[Dict[str, Any]], fail: bool):
        self._sink = sink
        self._fail = fail

    def insert(self, record: Dict[str, Any]) -> _FakeInsert:
        return _FakeInsert(self._sink, record, self._fail)


class _FakeSupabase:
    """Sync supabase.Client lookalike: table(name).insert(rec).execute()."""

    def __init__(self, fail: bool = False):
        self.rows: List[Dict[str, Any]] = []
        self.tables: List[str] = []
        self._fail = fail

    def table(self, name: str) -> _FakeTable:
        self.tables.append(name)
        return _FakeTable(self.rows, self._fail)


# The live schema's writable columns (migration 063 adds is_synthetic;
# search_vector is GENERATED and must never be written).
_REAL_COLUMNS = {
    "activity_id",
    "agent_name",
    "agent_tier",
    "activity_timestamp",
    "activity_type",
    "workstream",
    "processing_duration_ms",
    "input_data",
    "records_processed",
    "time_window",
    "analysis_results",
    "causal_paths_analyzed",
    "confidence_level",
    "recommendations",
    "actions_initiated",
    "impact_estimate",
    "roi_estimate",
    "status",
    "error_message",
    "resource_usage",
    "data_split",
    "split_config_id",
    "created_at",
    "is_synthetic",
}


@pytest.mark.unit
class TestPersistAgentActivity:
    def test_inserts_real_columns_only(self):
        client = _FakeSupabase()
        activity_id = persist_agent_activity(
            agent_name="heterogeneous_optimizer",
            agent_tier="causal_analytics",
            activity_type="cate_analysis",
            analysis_results={"brand": "Remibrutinib", "overall_ate": 0.3},
            input_data={"session_id": "s1", "query": "q"},
            confidence_level=0.87,
            processing_duration_ms=1234,
            supabase_client=client,
        )
        assert activity_id is not None
        assert client.tables == ["agent_activities"]
        assert len(client.rows) == 1
        row = client.rows[0]
        assert set(row) <= _REAL_COLUMNS, f"non-schema columns: {set(row) - _REAL_COLUMNS}"
        assert row["activity_id"] == activity_id
        assert len(row["activity_id"]) <= 30  # varchar(30) PK
        assert row["agent_name"] == "heterogeneous_optimizer"
        assert row["agent_tier"] == "causal_analytics"
        assert row["is_synthetic"] is False  # runtime rows are REAL provenance
        assert row["status"] == "completed"
        assert row["analysis_results"]["brand"] == "Remibrutinib"

    def test_never_writes_generated_search_vector(self):
        client = _FakeSupabase()
        persist_agent_activity(
            agent_name="causal_impact",
            agent_tier="causal_analytics",
            activity_type="causal_analysis",
            analysis_results={},
            supabase_client=client,
        )
        assert "search_vector" not in client.rows[0]

    def test_insert_failure_is_swallowed(self):
        client = _FakeSupabase(fail=True)
        activity_id = persist_agent_activity(
            agent_name="gap_analyzer",
            agent_tier="causal_analytics",
            activity_type="gap_analysis",
            analysis_results={"brand": "Kisqali"},
            supabase_client=client,
        )
        assert activity_id is None  # log-and-continue, never raises

    def test_no_client_available_returns_none(self):
        # supabase_client=None + factory unavailable must not raise
        activity_id = persist_agent_activity(
            agent_name="gap_analyzer",
            agent_tier="causal_analytics",
            activity_type="gap_analysis",
            analysis_results={},
            supabase_client=None,
            _client_factory=lambda: None,
        )
        assert activity_id is None

    def test_kill_switch_blocks_factory_client(self, monkeypatch):
        """E2I_DISABLE_AGENT_ACTIVITY_WRITER must prevent the writer from
        even CREATING a real client (2026-07-30 incident: pre-existing agent
        unit suites invoked contribute_to_memory, the factory picked up real
        .env service-role creds via tests/conftest load_dotenv, and 16 real
        rows landed in the live agent_activities table)."""
        monkeypatch.setenv("E2I_DISABLE_AGENT_ACTIVITY_WRITER", "1")
        factory_calls: List[int] = []

        def _factory():
            factory_calls.append(1)
            return _FakeSupabase()

        activity_id = persist_agent_activity(
            agent_name="heterogeneous_optimizer",
            agent_tier="causal_analytics",
            activity_type="cate_analysis",
            analysis_results={"brand": "Remibrutinib"},
            supabase_client=None,
            _client_factory=_factory,
        )
        assert activity_id is None
        assert factory_calls == []  # the factory must never be touched

    def test_explicit_client_overrides_kill_switch(self, monkeypatch):
        """An explicitly injected client (tests' fakes; deliberate callers)
        still writes — the kill switch only blocks the IMPLICIT real client."""
        monkeypatch.setenv("E2I_DISABLE_AGENT_ACTIVITY_WRITER", "1")
        client = _FakeSupabase()
        activity_id = persist_agent_activity(
            agent_name="causal_impact",
            agent_tier="causal_analytics",
            activity_type="causal_analysis",
            analysis_results={},
            supabase_client=client,
        )
        assert activity_id is not None
        assert len(client.rows) == 1

    def test_pytest_session_arms_kill_switch(self):
        """The root tests/conftest.py must arm the kill switch for the whole
        session so NO unit test can implicitly write the live table."""
        import os

        assert os.environ.get("E2I_DISABLE_AGENT_ACTIVITY_WRITER") == "1"

    def test_confidence_and_roi_are_bounded(self):
        client = _FakeSupabase()
        persist_agent_activity(
            agent_name="gap_analyzer",
            agent_tier="causal_analytics",
            activity_type="gap_analysis",
            analysis_results={},
            confidence_level=1.7,  # numeric(4,3) holds it, but 0-1 is the contract
            roi_estimate=123456.0,  # numeric(5,2) caps at 999.99
            supabase_client=client,
        )
        row = client.rows[0]
        assert 0.0 <= row["confidence_level"] <= 1.0
        assert abs(row["roi_estimate"]) <= 999.99


# ---------------------------------------------------------------------------
# Per-agent contribute_to_memory wiring
# ---------------------------------------------------------------------------


class _NullHooks:
    """Memory hooks stand-in whose memory backends are unavailable, so
    contribute_to_memory exercises ONLY the activity-writer step."""

    async def cache_cate_analysis(self, *a, **kw):
        return False

    async def store_cate_analysis(self, *a, **kw):
        return None

    async def store_segment_profiles(self, *a, **kw):
        return 0

    async def cache_causal_analysis(self, *a, **kw):
        return False

    async def store_causal_analysis(self, *a, **kw):
        return None

    async def store_causal_path(self, *a, **kw):
        return False

    async def cache_gap_analysis(self, *a, **kw):
        return False

    async def store_gap_analysis(self, *a, **kw):
        return None


def _persisted(monkeypatch, module) -> List[Dict[str, Any]]:
    """Route the module-under-test's persist_agent_activity onto a recorder."""
    rows: List[Dict[str, Any]] = []

    def _fake(
        *,
        agent_name: str,
        agent_tier: str,
        activity_type: str,
        analysis_results: Dict[str, Any],
        **kwargs: Any,
    ) -> Optional[str]:
        rows.append(
            {
                "agent_name": agent_name,
                "agent_tier": agent_tier,
                "activity_type": activity_type,
                "analysis_results": analysis_results,
                **kwargs,
            }
        )
        return "act_test"

    monkeypatch.setattr(module, "persist_agent_activity", _fake)
    return rows


@pytest.mark.unit
class TestHeterogeneousOptimizerWiring:
    @pytest.mark.asyncio
    async def test_completed_analysis_persists_activity(self, monkeypatch):
        from src.agents.heterogeneous_optimizer import memory_hooks as mod

        rows = _persisted(monkeypatch, mod)
        result = {
            "status": "completed",
            "overall_ate": 0.31,
            "heterogeneity_score": 0.42,
            "high_responders": [{"segment_id": "s1"}],
            "low_responders": [],
            "total_latency_ms": 1500,
            "confidence": 0.9,
        }
        state = {"treatment_var": "copay_support", "outcome_var": "adherent_180d"}
        counts = await mod.contribute_to_memory(
            result=result,
            state=state,
            memory_hooks=_NullHooks(),
            session_id="sess",
            brand="Remibrutinib",
            region="northeast",
        )
        assert counts.get("activity_stored") == 1
        assert len(rows) == 1
        row = rows[0]
        assert row["agent_name"] == "heterogeneous_optimizer"
        assert row["agent_tier"] == "causal_analytics"
        assert row["analysis_results"]["brand"] == "Remibrutinib"
        assert row["analysis_results"]["treatment_var"] == "copay_support"
        assert row["analysis_results"]["overall_ate"] == 0.31

    @pytest.mark.asyncio
    async def test_failed_analysis_persists_nothing(self, monkeypatch):
        from src.agents.heterogeneous_optimizer import memory_hooks as mod

        rows = _persisted(monkeypatch, mod)
        counts = await mod.contribute_to_memory(
            result={"status": "failed"},
            state={},
            memory_hooks=_NullHooks(),
            session_id="sess",
        )
        assert counts.get("activity_stored", 0) == 0
        assert rows == []

    @pytest.mark.asyncio
    async def test_writer_failure_does_not_break_contribution(self, monkeypatch):
        from src.agents.heterogeneous_optimizer import memory_hooks as mod

        def _boom(**kwargs):
            raise RuntimeError("writer exploded")

        monkeypatch.setattr(mod, "persist_agent_activity", _boom)
        counts = await mod.contribute_to_memory(
            result={"status": "completed", "high_responders": [], "low_responders": []},
            state={},
            memory_hooks=_NullHooks(),
            session_id="sess",
        )
        assert counts.get("activity_stored") == 0  # log-and-continue


@pytest.mark.unit
class TestCausalImpactWiring:
    @pytest.mark.asyncio
    async def test_completed_analysis_persists_activity(self, monkeypatch):
        from src.agents.causal_impact import memory_hooks as mod

        rows = _persisted(monkeypatch, mod)
        result = {
            "status": "completed",
            "ate_estimate": 0.12,
            "confidence": 0.88,
            "refutation_passed": True,
            "gate_decision": "proceed",
            "effect_size": "medium",
            "total_latency_ms": 900,
        }
        state = {
            "treatment_var": "rep_detailing_high",
            "outcome_var": "treatment_initiated",
            "confounders": ["academic_hcp", "engagement_score"],
            "brand": "Kisqali",
        }
        counts = await mod.contribute_to_memory(
            result=result, state=state, memory_hooks=_NullHooks(), session_id="sess"
        )
        assert counts.get("activity_stored") == 1
        row = rows[0]
        assert row["agent_name"] == "causal_impact"
        assert row["agent_tier"] == "causal_analytics"
        assert row["analysis_results"]["brand"] == "Kisqali"
        assert row["analysis_results"]["ate_estimate"] == 0.12
        assert row["analysis_results"]["refutation_passed"] is True

    @pytest.mark.asyncio
    async def test_failed_analysis_persists_nothing(self, monkeypatch):
        from src.agents.causal_impact import memory_hooks as mod

        rows = _persisted(monkeypatch, mod)
        counts = await mod.contribute_to_memory(
            result={"status": "failed"},
            state={},
            memory_hooks=_NullHooks(),
            session_id="sess",
        )
        assert counts.get("activity_stored", 0) == 0
        assert rows == []


@pytest.mark.unit
class TestGapAnalyzerWiring:
    @pytest.mark.asyncio
    async def test_completed_analysis_persists_activity(self, monkeypatch):
        from src.agents.gap_analyzer import memory_hooks as mod

        rows = _persisted(monkeypatch, mod)
        result = {
            "prioritized_opportunities": [{"id": 1}, {"id": 2}],
            "total_addressable_value": 250000.0,
            "quick_wins": [{"id": 1}],
            "strategic_bets": [],
            "confidence": 0.8,
        }
        state = {
            "status": "completed",
            "brand": "Fabhalta",
            "metrics": ["TRx"],
            "segments": ["west"],
        }
        counts = await mod.contribute_to_memory(
            result=result, state=state, memory_hooks=_NullHooks(), session_id="sess"
        )
        assert counts.get("activity_stored") == 1
        row = rows[0]
        assert row["agent_name"] == "gap_analyzer"
        assert row["agent_tier"] == "causal_analytics"
        assert row["analysis_results"]["brand"] == "Fabhalta"
        assert row["analysis_results"]["total_addressable_value"] == 250000.0
        assert row.get("impact_estimate") == 250000.0

    @pytest.mark.asyncio
    async def test_failed_analysis_persists_nothing(self, monkeypatch):
        from src.agents.gap_analyzer import memory_hooks as mod

        rows = _persisted(monkeypatch, mod)
        counts = await mod.contribute_to_memory(
            result={},
            state={"status": "failed"},
            memory_hooks=_NullHooks(),
            session_id="sess",
        )
        assert counts.get("activity_stored", 0) == 0
        assert rows == []
