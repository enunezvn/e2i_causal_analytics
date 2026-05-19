"""Unit tests for Crystallizer (subsystem 7) — brand strictness, edge wiring."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.memory.crystallization.crystallizer import Crystallizer


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode = None
        self._filters: Dict[str, Any] = {}
        self._in_filters: Dict[str, List[Any]] = {}
        self._gte: Dict[str, Any] = {}
        self._insert_payload: Any = None

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        return self

    def insert(self, payload: Any) -> "_FakeQuery":
        self._mode = "insert"
        self._insert_payload = payload
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        self._gte[col] = val
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        self._in_filters[col] = vals
        return self

    def order(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def limit(self, n: int) -> "_FakeQuery":
        return self

    def execute(self) -> MagicMock:
        if self._mode == "insert":
            payload = self._insert_payload
            rows_to_insert: List[Dict[str, Any]]
            if isinstance(payload, list):
                rows_to_insert = payload
            else:
                rows_to_insert = [payload]
            inserted = []
            for r in rows_to_insert:
                row = dict(r)
                if self.table_name == "executive_insights":
                    row["insight_id"] = row.get("insight_id") or str(uuid.uuid4())
                self.store.rows.setdefault(self.table_name, []).append(row)
                inserted.append(row)
            mock = MagicMock()
            mock.data = inserted
            return mock
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            rows = [r for r in rows if r.get(col) == want]
        for col, allowed in self._in_filters.items():
            rows = [r for r in rows if r.get(col) in allowed]
        for col, threshold in self._gte.items():
            rows = [r for r in rows if (r.get(col) or "") >= threshold]
        mock = MagicMock()
        mock.data = rows
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "episodic_memories": [],
            "executive_insights": [],
            "insight_edges": [],
        }

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()


@pytest.fixture(autouse=True)
def patch_client(fake_supabase):
    with patch(
        "src.memory.crystallization.crystallizer.get_supabase_client", return_value=fake_supabase
    ):
        yield


def _seed_episodic(
    db: FakeSupabase,
    *,
    brand: str,
    causal_path_id: str,
    agents: List[str],
    region: Optional[str] = "northeast",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for i, agent in enumerate(agents):
        db.rows["episodic_memories"].append(
            {
                "memory_id": f"{brand}-{causal_path_id}-{i}",
                "agent_name": agent,
                "brand": brand,
                "region": region,
                "causal_path_id": causal_path_id,
                "event_type": "agent_action",
                "description": f"{agent} on {causal_path_id}",
                "outcome_type": "success",
                "occurred_at": now,
                "raw_content": {},
            }
        )


@pytest.mark.asyncio
async def test_crystallize_requires_two_distinct_agents(fake_supabase: FakeSupabase):
    _seed_episodic(fake_supabase, brand="Kisqali", causal_path_id="cp1", agents=["causal_impact"])
    result = await Crystallizer(min_agents=2).run_for_brand("Kisqali")
    assert result.insights_created == 0


@pytest.mark.asyncio
async def test_crystallize_creates_insight_and_edges(fake_supabase: FakeSupabase):
    _seed_episodic(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer", "heterogeneous_optimizer"],
    )
    result = await Crystallizer().run_for_brand("Kisqali")
    assert result.insights_created == 1
    insights = fake_supabase.rows["executive_insights"]
    assert len(insights) == 1
    assert insights[0]["brand"] == "Kisqali"

    edges = fake_supabase.rows["insight_edges"]
    # 3 source episodic memories + 1 causal_path summarizes edge.
    assert len(edges) == 4
    # All edges are brand-tagged Kisqali.
    assert {e["brand"] for e in edges} == {"Kisqali"}


@pytest.mark.asyncio
async def test_crystallize_never_co_aggregates_across_brands(fake_supabase: FakeSupabase):
    """Add Kisqali AND Fabhalta memories with overlapping cycles. The two
    crystallization runs must each produce exactly one brand-pure insight."""
    _seed_episodic(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    _seed_episodic(
        fake_supabase,
        brand="Fabhalta",
        causal_path_id="cp2",
        agents=["causal_impact", "gap_analyzer"],
    )
    r_k = await Crystallizer().run_for_brand("Kisqali")
    r_f = await Crystallizer().run_for_brand("Fabhalta")
    assert r_k.insights_created == 1
    assert r_f.insights_created == 1

    insights = fake_supabase.rows["executive_insights"]
    assert len(insights) == 2
    brands = {i["brand"] for i in insights}
    assert brands == {"Kisqali", "Fabhalta"}

    # Every edge must carry the same brand as its target insight.
    edges_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for e in fake_supabase.rows["insight_edges"]:
        edges_by_target.setdefault(e["target_id"], []).append(e)
    for insight in insights:
        target_edges = edges_by_target.get(insight["insight_id"], [])
        for e in target_edges:
            assert e["brand"] == insight["brand"]


@pytest.mark.asyncio
async def test_crystallize_rejects_empty_brand():
    with pytest.raises(ValueError):
        await Crystallizer().run_for_brand("")


# =============================================================================
# Issue #376 — Phase 4 schema completion tests
# =============================================================================


def _seed_episodic_with_causal_content(
    db: FakeSupabase,
    *,
    brand: str,
    causal_path_id: str,
    agents: List[str],
    region: Optional[str] = "northeast",
    ate: float = 0.42,
    ci_lower: float = 0.30,
    ci_upper: float = 0.55,
    cohort_size: int = 1200,
    confounders: Optional[List[str]] = None,
    refutation_passed_tests: Optional[List[str]] = None,
    refutation_failed_tests: Optional[List[str]] = None,
    data_version: str = "2026-05-19-snapshot",
) -> None:
    """Seed episodic memories with the raw_content shape the crystallizer
    inspects to derive the 13 deterministic CrystalDigest fields.

    Mirrors the shape that ``src/agents/causal_impact/memory_hooks.py``
    actually writes (lines 442-452).
    """
    confounders = confounders if confounders is not None else ["age", "prior_use"]
    refutation_passed = (
        refutation_passed_tests
        if refutation_passed_tests is not None
        else ["placebo_treatment", "random_common_cause"]
    )
    refutation_failed = (
        refutation_failed_tests if refutation_failed_tests is not None else ["data_subset"]
    )

    now = datetime.now(timezone.utc).isoformat()
    for i, agent in enumerate(agents):
        db.rows["episodic_memories"].append(
            {
                "memory_id": f"{brand}-{causal_path_id}-{i}",
                "agent_name": agent,
                "brand": brand,
                "region": region,
                "causal_path_id": causal_path_id,
                "event_type": "agent_action",
                "description": f"{agent} on {causal_path_id}",
                "outcome_type": "success",
                "occurred_at": now,
                "raw_content": {
                    "ate_estimate": ate,
                    "confidence_interval": [ci_lower, ci_upper],
                    "sample_size": cohort_size,
                    "confounders": confounders,
                    "refutation_passed_tests": refutation_passed,
                    "refutation_failed_tests": refutation_failed,
                    "data_version": data_version,
                    "kpi": "trx",
                },
            }
        )


@pytest.mark.asyncio
async def test_crystallize_derives_numeric_effect_size_from_episodic(fake_supabase):
    """Per sub-decision 2a: effect_size is numeric, derived from
    EstimationResult.ate (surfaced in raw_content.ate_estimate)."""
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
        ate=0.42,
        ci_lower=0.30,
        ci_upper=0.55,
    )
    result = await Crystallizer().run_for_brand("Kisqali")
    assert result.insights_created == 1

    insight = fake_supabase.rows["executive_insights"][0]
    assert insight["effect_size"] == pytest.approx(0.42)
    assert insight["effect_ci_lower"] == pytest.approx(0.30)
    assert insight["effect_ci_upper"] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_crystallize_derives_effect_direction_from_ate_sign(fake_supabase):
    """effect_direction is deterministic from the sign of effect_size +
    the CI bounds. Positive ATE → 'positive'; negative → 'negative';
    CI straddling zero → 'null'."""
    # Positive direction
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp_pos",
        agents=["causal_impact", "gap_analyzer"],
        ate=0.42,
        ci_lower=0.30,
        ci_upper=0.55,
    )
    # Negative direction
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp_neg",
        agents=["causal_impact", "gap_analyzer"],
        ate=-0.30,
        ci_lower=-0.45,
        ci_upper=-0.10,
    )
    # CI straddles zero → 'null'
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp_null",
        agents=["causal_impact", "gap_analyzer"],
        ate=0.05,
        ci_lower=-0.10,
        ci_upper=0.15,
    )
    await Crystallizer().run_for_brand("Kisqali")

    insights = fake_supabase.rows["executive_insights"]
    by_path = {}
    for i in insights:
        cp_id = (i.get("key_metrics") or {}).get("causal_path_id")
        by_path[cp_id] = i

    assert by_path["cp_pos"]["effect_direction"] == "positive"
    assert by_path["cp_neg"]["effect_direction"] == "negative"
    assert by_path["cp_null"]["effect_direction"] == "null"


@pytest.mark.asyncio
async def test_crystallize_derives_cohort_size_and_confounders(fake_supabase):
    """cohort_size from sample_size; confounders_controlled from the
    raw_content confounders list (deduplicated across source memories)."""
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
        cohort_size=1500,
        confounders=["age", "prior_use", "comorbidity_score"],
    )
    await Crystallizer().run_for_brand("Kisqali")

    insight = fake_supabase.rows["executive_insights"][0]
    assert insight["cohort_size"] == 1500
    # confounders_controlled is a deduplicated list
    assert set(insight["confounders_controlled"]) == {"age", "prior_use", "comorbidity_score"}


@pytest.mark.asyncio
async def test_crystallize_derives_sensitivity_check_arrays(fake_supabase):
    """sensitivity_checks_passed / sensitivity_checks_failed reflect
    the refutation-test outcomes captured in raw_content."""
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
        refutation_passed_tests=["placebo_treatment", "random_common_cause"],
        refutation_failed_tests=["data_subset"],
    )
    await Crystallizer().run_for_brand("Kisqali")

    insight = fake_supabase.rows["executive_insights"][0]
    assert set(insight["sensitivity_checks_passed"]) == {
        "placebo_treatment",
        "random_common_cause",
    }
    assert set(insight["sensitivity_checks_failed"]) == {"data_subset"}


@pytest.mark.asyncio
async def test_crystallize_derives_provenance_and_replication_lineage(fake_supabase):
    """provenance_chain_id is a deterministic identifier; provenance_depth
    is the BFS hop count; replication_count = source_count for this v1."""
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer", "heterogeneous_optimizer"],
    )
    await Crystallizer().run_for_brand("Kisqali")

    insight = fake_supabase.rows["executive_insights"][0]
    # provenance_chain_id is a deterministic non-empty string
    assert isinstance(insight["provenance_chain_id"], str)
    assert len(insight["provenance_chain_id"]) > 0
    # provenance_depth is at least 1 (the source memories are direct ancestors)
    assert isinstance(insight["provenance_depth"], int)
    assert insight["provenance_depth"] >= 1
    # replication_count = number of source episodic memories
    assert insight["replication_count"] == 3


@pytest.mark.asyncio
async def test_crystallize_derives_consolidation_tier(fake_supabase):
    """consolidation_tier inherited from the source group's tier.

    Per migration 021, episodic_memories rows default to tier='episodic'.
    Crystallization at v1 stays at 'episodic' until the consolidator
    promotes the underlying causal_path; tier 'semantic' is what the
    consolidator-driven crystallization writes."""
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    await Crystallizer().run_for_brand("Kisqali")

    insight = fake_supabase.rows["executive_insights"][0]
    assert insight["consolidation_tier"] in (
        "working",
        "episodic",
        "semantic",
        "procedural",
    )


@pytest.mark.asyncio
async def test_crystallize_derives_data_version(fake_supabase):
    """data_version comes from the cohort manifest tag in raw_content."""
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
        data_version="2026-05-19-snapshot",
    )
    await Crystallizer().run_for_brand("Kisqali")

    insight = fake_supabase.rows["executive_insights"][0]
    assert insight["data_version"] == "2026-05-19-snapshot"


@pytest.mark.asyncio
async def test_crystallize_uses_deterministic_narrative_when_flag_off(fake_supabase, monkeypatch):
    """When E2I_CRYSTAL_LLM_NARRATIVES_ENABLED is not set / falsey, the
    crystallizer must NOT call the LLM — limitations + recommended_next_analysis
    come from a deterministic heuristic and the row inserts without an
    LLMCrystalNarrativeAudit being constructed.

    Memory `[[feedback-live-lm-skip-must-check-key-shape]]` —
    presence-only checks accidentally accept CI placeholder keys;
    the feature flag is the explicit gate.
    """
    monkeypatch.delenv("E2I_CRYSTAL_LLM_NARRATIVES_ENABLED", raising=False)

    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    result = await Crystallizer().run_for_brand("Kisqali")
    assert result.insights_created == 1

    insight = fake_supabase.rows["executive_insights"][0]
    # Deterministic narrative still populates the fields, just from a
    # heuristic — both must be non-empty so the dashboard does not show
    # blank cells.
    assert insight["limitations"]
    assert insight["recommended_next_analysis"]


@pytest.mark.asyncio
async def test_crystallize_calls_llm_narrator_when_flag_on(fake_supabase, monkeypatch):
    """When E2I_CRYSTAL_LLM_NARRATIVES_ENABLED=1, the crystallizer
    invokes the narrator. We inject a stub narrator and assert it was
    called + the audit fields are captured on the row.
    """
    monkeypatch.setenv("E2I_CRYSTAL_LLM_NARRATIVES_ENABLED", "1")

    from src.data.kg.types import LLMCrystalNarrativeAudit

    # Stub narrator: replace the module-level factory so the crystallizer
    # gets a deterministic audit without a network call.
    stub_audit = LLMCrystalNarrativeAudit(
        narrator_model="claude-haiku-4-5-20251001",
        key_finding="Stub finding: Northeast region shows a +0.42 ATE.",
        limitations="Stub limitation: small pre-period (n=120).",
        recommended_next_analysis="Stub: replicate on Q3 cohort.",
        latency_ms=123.4,
        input_tokens=800,
        output_tokens=200,
        cost_usd=0.0018,
    )

    # Codex iter-1 H1: _invoke_llm_narrator is now async, so the stub
    # MUST be async too (calling site is `await _invoke_llm_narrator(...)`).
    # A plain sync `def` would return a coroutine-shaped value once
    # awaited — not the LLMCrystalNarrativeAudit instance.
    async def stub_narrator(*args, **kwargs):
        return stub_audit

    monkeypatch.setattr(
        "src.memory.crystallization.crystallizer._invoke_llm_narrator",
        stub_narrator,
    )

    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    result = await Crystallizer().run_for_brand("Kisqali")
    assert result.insights_created == 1

    insight = fake_supabase.rows["executive_insights"][0]
    # The stubbed narrator's outputs land on the row.
    assert insight["limitations"] == "Stub limitation: small pre-period (n=120)."
    assert insight["recommended_next_analysis"] == "Stub: replicate on Q3 cohort."


@pytest.mark.asyncio
async def test_crystallize_finding_method_exists():
    """``crystallize_finding(finding_id: str, *, brand: str)`` must be a
    bound method of Crystallizer (#376 DoD §D)."""
    crystallizer = Crystallizer()
    assert hasattr(crystallizer, "crystallize_finding"), (
        "Crystallizer must expose crystallize_finding(finding_id, *, brand)"
    )
    assert callable(crystallizer.crystallize_finding)


@pytest.mark.asyncio
async def test_crystallize_portfolio_method_exists_and_iterates_brands(fake_supabase):
    """``crystallize_portfolio()`` iterates the configured brand list
    and returns a result aggregating across brands."""
    crystallizer = Crystallizer()
    assert hasattr(crystallizer, "crystallize_portfolio"), (
        "Crystallizer must expose crystallize_portfolio()"
    )

    # Seed two brands; the portfolio iteration must visit both.
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="fabhalta",
        causal_path_id="cp2",
        agents=["causal_impact", "gap_analyzer"],
    )

    result = await crystallizer.crystallize_portfolio()
    # Each brand should have at least 1 insight created.
    assert result.insights_created >= 2
    assert set(result.by_brand.keys()) >= {"kisqali", "fabhalta"}

