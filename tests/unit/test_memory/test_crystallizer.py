"""Unit tests for Crystallizer (subsystem 7) — brand strictness, edge wiring."""

from __future__ import annotations

import sys
import types
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
        self._mode: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._in_filters: Dict[str, List[Any]] = {}
        self._gte: Dict[str, Any] = {}
        self._insert_payload: Any = None
        self._range: Optional[tuple] = None

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
        # Record the order column so a test can witness a stable-sort call. A
        # fake can't reproduce real query-plan reordering, so the .order() call
        # IS the faithful witness that offset pagination is deterministic.
        if args:
            self.store.order_calls.append((self.table_name, args[0]))
        return self

    def limit(self, n: int) -> "_FakeQuery":
        return self

    def range(self, start: int, end: int) -> "_FakeQuery":
        # PostgREST .range() is inclusive on both ends.
        self._range = (start, end)
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
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        elif self.store.no_range_cap is not None:
            # Simulate PostgREST's silent default row cap on an un-ranged request.
            rows = rows[: self.store.no_range_cap]
        mock = MagicMock()
        mock.data = rows
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "episodic_memories": [],
            "executive_insights": [],
            "insight_edges": [],
            # #391 box 4 codex iter-2 H1: persistence target for
            # LLMCrystalNarrativeAudit. The crystallizer now writes one
            # row here per LLM-narrated crystal so the PHI scanner has
            # ``input_prompt`` to audit.
            "crystal_narrative_audits": [],
        }
        # When set, simulate PostgREST silently capping an un-ranged SELECT.
        self.no_range_cap: Optional[int] = None
        # (table, column) for every .order() call — lets a test witness a sort.
        self.order_calls: List[tuple] = []

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
# M2-supabase (#694) — blocking Supabase .execute() must run off the event
# loop via asyncio.to_thread so the operator /crystallize path does not stall
# the FastAPI event loop. Characterization test: assert the sync .execute()
# calls are dispatched through asyncio.to_thread AND that the crystallizer
# still produces the same insight + edges with the off-loop dispatch.
# =============================================================================


@pytest.mark.asyncio
async def test_crystallize_dispatches_supabase_execute_off_loop(fake_supabase: FakeSupabase):
    """The synchronous Supabase ``<query>.execute()`` calls (SELECT
    candidates, INSERT executive_insight, INSERT insight_edges) must be
    off-loaded via ``asyncio.to_thread`` so they never block the event
    loop. We patch the crystallizer-module ``asyncio.to_thread`` with a
    spy that still invokes the wrapped callable (so behavior is preserved)
    and assert it was used to run the Supabase ``.execute`` callables.
    """
    import asyncio

    _seed_episodic(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )

    real_to_thread = asyncio.to_thread
    execute_dispatch_count = 0

    async def spy_to_thread(func, /, *args, **kwargs):
        nonlocal execute_dispatch_count
        # The bound method we expect to be off-loaded is ``<query>.execute``.
        if getattr(func, "__name__", "") == "execute":
            execute_dispatch_count += 1
        return await real_to_thread(func, *args, **kwargs)

    with patch(
        "src.memory.crystallization.crystallizer.asyncio.to_thread",
        side_effect=spy_to_thread,
    ):
        result = await Crystallizer().run_for_brand("Kisqali")

    # Behavior preserved: the off-loop dispatch still produces the insight
    # + edges exactly as the synchronous path did.
    assert result.insights_created == 1
    assert len(fake_supabase.rows["executive_insights"]) == 1
    # At minimum the candidate SELECT, the insight INSERT, and the edge
    # INSERT each go through to_thread (3 .execute dispatches).
    assert execute_dispatch_count >= 3


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
    # confounders_controlled MUST be sorted (codex iter-1 M2): the
    # crystallizer applies sorted() to the dedup set so downstream
    # consumers get a stable serialization regardless of seed order.
    # Asserting exact ordered list (not set equality) pins the
    # contract — a regression that returns encounter-order trips here.
    assert insight["confounders_controlled"] == [
        "age",
        "comorbidity_score",
        "prior_use",
    ]


@pytest.mark.asyncio
async def test_crystallize_derives_sensitivity_check_arrays(fake_supabase):
    """sensitivity_checks_passed / sensitivity_checks_failed reflect
    the refutation-test outcomes captured in raw_content, sorted for
    deterministic serialization (codex iter-1 M2)."""
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
    # Exact ordered list (codex iter-1 M2): sorted alphabetically.
    assert insight["sensitivity_checks_passed"] == [
        "placebo_treatment",
        "random_common_cause",
    ]
    assert insight["sensitivity_checks_failed"] == ["data_subset"]


@pytest.mark.asyncio
async def test_crystallize_arrays_sorted_across_multi_member_dedup(fake_supabase):
    """Codex iter-1 M2 regression: when multiple source members carry
    overlapping confounder + sensitivity-check lists in DIFFERENT
    orders, the crystallized arrays must be sorted not encounter-
    ordered.

    Pre-fix: dedup happened in encounter order. The upstream
    episodic_memories query has no stable secondary ordering, so two
    crystallization passes on the same data could emit different
    array orderings — a flaky JSONB diff for downstream consumers.
    """
    import uuid
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    # Member A: confounders in alphabetical order, sensitivity-passed
    # in reverse alphabetical order, sensitivity-FAILED with the
    # alphabetically-LARGER value listed first (codex iter-2 M2:
    # adversarial fixture for sensitivity_checks_failed so encounter-
    # order ≠ sorted-order even on the failed array).
    fake_supabase.rows["episodic_memories"].append(
        {
            "memory_id": str(uuid.uuid4()),
            "agent_name": "causal_impact",
            "brand": "Kisqali",
            "region": "northeast",
            "causal_path_id": "cp1",
            "event_type": "agent_action",
            "description": "A",
            "outcome_type": "success",
            "occurred_at": now,
            "raw_content": {
                "ate_estimate": 0.4,
                "confidence_interval": [0.2, 0.6],
                "confounders": ["a_age", "z_zip"],
                "refutation_passed_tests": ["z_random_cc", "a_placebo"],
                "refutation_failed_tests": ["wald_test", "bootstrap"],
            },
        }
    )
    # Member B: introduces NEW confounder in the middle of the sorted
    # output, a sensitivity-pass test that's lexicographically smaller
    # than member A's passes, and a sensitivity-FAILED test that is
    # alphabetically EARLIEST (must sort to front; encounter-order
    # would put it 3rd after A's two values).
    fake_supabase.rows["episodic_memories"].append(
        {
            "memory_id": str(uuid.uuid4()),
            "agent_name": "gap_analyzer",
            "brand": "Kisqali",
            "region": "northeast",
            "causal_path_id": "cp1",
            "event_type": "agent_action",
            "description": "B",
            "outcome_type": "success",
            "occurred_at": now,
            "raw_content": {
                "confounders": ["m_middle"],
                "refutation_passed_tests": ["aa_first_alphabetical"],
                # Three failed tests across both members:
                #   A: ["wald_test", "bootstrap"]
                #   B: ["anderson_test", "wald_test"]   (wald_test dup)
                # Encounter-order dedup → ["wald_test", "bootstrap", "anderson_test"]
                # sorted() → ["anderson_test", "bootstrap", "wald_test"]
                # The two orderings are distinguishable, so this fixture
                # falsifies any non-sort implementation.
                "refutation_failed_tests": ["anderson_test", "wald_test"],
            },
        }
    )
    await Crystallizer().run_for_brand("Kisqali")

    insight = fake_supabase.rows["executive_insights"][0]
    # Sorted alphabetically across both members — encounter order
    # would have been ["a_age", "z_zip", "m_middle"] (A then B),
    # NOT the alphabetical order asserted here.
    assert insight["confounders_controlled"] == ["a_age", "m_middle", "z_zip"]
    assert insight["sensitivity_checks_passed"] == [
        "a_placebo",
        "aa_first_alphabetical",
        "z_random_cc",
    ]
    # Codex iter-2 M2: now multi-value, deduplicated, AND sorted.
    # Encounter-order dedup would produce
    #   ["wald_test", "bootstrap", "anderson_test"]
    # which fails this assertion. Only sorted() produces the asserted
    # alphabetical order.
    assert insight["sensitivity_checks_failed"] == [
        "anderson_test",
        "bootstrap",
        "wald_test",
    ]


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
async def test_crystallize_inherits_highest_tier_from_sources(fake_supabase):
    """Codex iter-1 M1 (LOAD-BEARING): when source episodic_memories
    rows carry consolidation_tier='semantic' or 'procedural' (set by
    the consolidator post-promotion), the crystal's consolidation_tier
    MUST be the highest tier among sources, NOT the default 'episodic'.

    Failure mode pre-fix: the episodic_memories SELECT did not include
    consolidation_tier, so `m.get("consolidation_tier")` in
    _derive_crystal_digest_fields returned None on every row, and the
    crystal defaulted to 'episodic' silently — even when sources had
    been promoted by the consolidator.

    Tier rank: working < episodic < semantic < procedural.
    """
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    # Override the tier on each source row to a non-episodic value.
    # 'semantic' is the tier the consolidator promotes to; one source
    # 'procedural' should win because tier_rank says it's highest.
    rows = fake_supabase.rows["episodic_memories"]
    assert len(rows) >= 2, "fixture must seed at least 2 source rows"
    rows[0]["consolidation_tier"] = "semantic"
    rows[1]["consolidation_tier"] = "procedural"

    await Crystallizer().run_for_brand("Kisqali")

    insight = fake_supabase.rows["executive_insights"][0]
    # Procedural is the highest tier among sources, so the crystal
    # must inherit it (NOT default to 'episodic').
    assert insight["consolidation_tier"] == "procedural", (
        f"Expected highest source tier 'procedural'; got "
        f"{insight['consolidation_tier']!r}. "
        "The episodic_memories SELECT must include consolidation_tier "
        "for tier inheritance to work."
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
    # Codex iter-2 L2 closure: ``input_prompt`` MUST be non-empty in the
    # stub so the H1 persistence path has a meaningful value to assert
    # against (an empty string is itself a meaningful "we did NOT send
    # anything" audit signal — different from "stub did not populate the
    # field at all"; the explicit non-empty value pins the round-trip).
    stub_audit = LLMCrystalNarrativeAudit(
        narrator_model="claude-haiku-4-5-20251001",
        key_finding="Stub finding: Northeast region shows a +0.42 ATE.",
        limitations="Stub limitation: small pre-period (n=120).",
        recommended_next_analysis="Stub: replicate on Q3 cohort.",
        latency_ms=123.4,
        input_tokens=800,
        output_tokens=200,
        cost_usd=0.0018,
        input_prompt="Audit this Kisqali crystal: ATE=+0.42, CI=[0.30,0.55]",
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

    # Codex iter-2 H1 closure: assert the audit row landed in
    # crystal_narrative_audits with input_prompt populated. The PHI
    # scanner reads input_prompt via SQL JOIN — without persistence here
    # it would find NULL for all real crystals (the iter-2 H1 bug shape).
    audits = fake_supabase.rows["crystal_narrative_audits"]
    assert len(audits) == 1, (
        f"Expected 1 row in crystal_narrative_audits after LLM-path crystallization; "
        f"got {len(audits)}. This is the codex iter-2 H1 closure pin."
    )
    audit_row = audits[0]
    assert audit_row["insight_id"] == insight["insight_id"]
    assert audit_row["narrator_model"] == "claude-haiku-4-5-20251001"
    assert audit_row["key_finding"] == "Stub finding: Northeast region shows a +0.42 ATE."
    assert audit_row["limitations"] == "Stub limitation: small pre-period (n=120)."
    assert audit_row["recommended_next"] == "Stub: replicate on Q3 cohort."
    assert audit_row["input_prompt"] == ("Audit this Kisqali crystal: ATE=+0.42, CI=[0.30,0.55]")
    assert audit_row["input_tokens"] == 800
    assert audit_row["output_tokens"] == 200
    assert audit_row["cost_usd"] == pytest.approx(0.0018)


@pytest.mark.asyncio
async def test_crystallize_audit_persistence_failure_does_not_break_crystallization(
    fake_supabase, monkeypatch
):
    """Codex iter-2 H1 narrow-catch semantics: audit-table insertion is
    BEST-EFFORT. A failure on the ``crystal_narrative_audits`` insert
    MUST NOT propagate up — the crystal itself is still valid,
    insight_edges still get inserted, and the run completes with
    insights_created=1.

    The audit row is a sidecar for offline PHI auditing — its absence
    for a single crystal is preferable to failing the entire
    crystallization pipeline. This pin asserts the narrow-catch +
    log-warning shape and acts as a regression catch if a future PR
    widens the audit insert's failure mode to fatal.
    """
    monkeypatch.setenv("E2I_CRYSTAL_LLM_NARRATIVES_ENABLED", "1")

    from src.data.kg.types import LLMCrystalNarrativeAudit

    stub_audit = LLMCrystalNarrativeAudit(
        narrator_model="claude-haiku-4-5-20251001",
        key_finding="Stub finding.",
        limitations="Stub limitation.",
        recommended_next_analysis="Stub recommendation.",
        input_prompt="Stub prompt.",
    )

    async def stub_narrator(*args, **kwargs):
        return stub_audit

    monkeypatch.setattr(
        "src.memory.crystallization.crystallizer._invoke_llm_narrator",
        stub_narrator,
    )

    # Wrap the FakeSupabase to make the audit-table insert raise but
    # leave executive_insights / insight_edges working normally.
    original_table = fake_supabase.table

    def _failing_table(name: str):
        query = original_table(name)
        if name == "crystal_narrative_audits":
            original_execute = query.execute

            def _boom_execute():
                if query._mode == "insert":
                    raise RuntimeError("audit insert failed (simulated DB outage)")
                return original_execute()

            query.execute = _boom_execute  # type: ignore[method-assign]
        return query

    monkeypatch.setattr(fake_supabase, "table", _failing_table)

    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    result = await Crystallizer().run_for_brand("Kisqali")

    # Best-effort: the crystal IS created even though the audit row
    # failed to persist.
    assert result.insights_created == 1, (
        "audit-table insert failure must NOT prevent the crystal from "
        "being created — audit is sidecar telemetry, not gating"
    )
    assert result.errors == [], (
        "audit-table insert failure must NOT surface as a per-group "
        "error (narrow log-warning catch shape, not fatal)"
    )
    # The executive_insight DID land.
    assert len(fake_supabase.rows["executive_insights"]) == 1
    # The audit row DID NOT land (insert was simulated as failed).
    assert fake_supabase.rows["crystal_narrative_audits"] == []


@pytest.mark.asyncio
async def test_crystallize_audit_persistence_skipped_when_flag_off(fake_supabase, monkeypatch):
    """Codex iter-2 H1 negative pin: when the LLM flag is OFF, no audit
    row should land in ``crystal_narrative_audits`` because no LLM
    narrator ran (and there's no ``LLMCrystalNarrativeAudit`` to
    persist).

    Pins the ``if audit is not None`` guard at the persistence site.
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
    assert fake_supabase.rows["crystal_narrative_audits"] == [], (
        "no crystal_narrative_audits row should land on the flag-off "
        "(deterministic-narrative) path — no LLM audit object was produced"
    )


@pytest.mark.asyncio
async def test_crystallize_narrator_swallows_only_anthropic_errors(fake_supabase, monkeypatch):
    """Codex iter-1 H2 (LOAD-BEARING) + iter-3 H1 (test rigor): the
    Haiku call must catch only the four anthropic.* SDK error classes,
    NOT broad Exception.

    Programming errors (TypeError, AttributeError, KeyError) MUST
    propagate so they surface in CI / DLQ instead of being silently
    swallowed as "empty narrator audit". This mirrors the
    #378-iter-0-M1 / sibling narrow-catch contract.

    Test rigor (codex iter-3 H1): inject a fake client at the SDK
    boundary so the REAL ``_invoke_llm_narrator`` body executes — its
    real catch tuple ``(anthropic.APIConnectionError, APITimeoutError,
    RateLimitError, APIStatusError)`` is the assertion target. If a
    future regression broadens that tuple to ``Exception``, the
    TypeError gets swallowed and ``pytest.raises(TypeError)`` fails —
    regression caught.

    The prior iter-2 shape (monkeypatching ``_invoke_llm_narrator``
    itself) tested the MOCK, not the real catch tuple. The prior
    iter-3 shape monkeypatched ``anthropic.AsyncAnthropic`` globally;
    this version keeps the same SDK-call boundary coverage without
    mutating the SDK module.

    Companion positive-control:
    ``test_crystallize_narrator_falls_back_on_anthropic_api_error``
    below — injects at the SAME SDK boundary, raises an
    ``anthropic.APIConnectionError``, and asserts the real catch DOES
    swallow it (insight created with empty prose). Together the two
    tests bracket the catch boundary.
    """
    monkeypatch.setenv("E2I_CRYSTAL_LLM_NARRATIVES_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key-for-shape")

    from src.memory.crystallization import crystallizer as crystallizer_module

    # SDK-level stub: client.messages.create() raises
    # TypeError. This is one level DEEPER than the prior shape —
    # the real _invoke_llm_narrator body runs, receives the fake
    # client, awaits create, and hits the real catch tuple.
    class _BoomMessages:
        async def create(self, **_kwargs):
            raise TypeError("programmer error not in anthropic catch tuple")

    class _BoomClient:
        def __init__(self, **_kwargs):
            self.messages = _BoomMessages()

    def _boom_client_factory(_api_key: str) -> _BoomClient:
        return _BoomClient()

    # PRIMARY assertion: direct await on the real function must
    # propagate the TypeError. The real catch at
    # ``crystallizer.py:789-807`` includes only the four anthropic.*
    # SDK classes, so a TypeError raised inside the `try` block
    # escapes uncaught. If someone broadens the catch to `Exception`
    # tomorrow, this assertion FAILS (pytest.raises receives no
    # exception) → regression caught at the SDK-call boundary, not
    # at a test-internal mock layer.
    with pytest.raises(TypeError, match="programmer error not in anthropic catch tuple"):
        await crystallizer_module._invoke_llm_narrator(
            brand="Kisqali",
            region="northeast",
            members=[{"memory_id": "m1", "raw_content": {}}],
            derived={
                "effect_size": 0.0,
                "effect_ci_lower": None,
                "effect_ci_upper": None,
                "effect_direction": None,
                "cohort_size": None,
                "confounders_controlled": [],
                "sensitivity_checks_passed": [],
                "sensitivity_checks_failed": [],
            },
            client_factory=_boom_client_factory,
        )

    # SECONDARY observation (kept from iter-1 + iter-2): when the
    # same TypeError surfaces inside the per-group loop, the outer
    # ``_crystallize_group`` try/except catches it and records it
    # in result.errors. No row is inserted; no empty-audit shadowing.
    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    result = await Crystallizer(anthropic_client_factory=_boom_client_factory).run_for_brand(
        "Kisqali"
    )
    assert result.insights_created == 0, (
        "TypeError must escape narrow narrator catch, not be swallowed "
        "as empty narrator audit (which would let the row insert with "
        "empty prose pass through)"
    )
    assert any(
        "programmer error not in anthropic catch tuple" in e or "TypeError" in e
        for e in result.errors
    ), f"Expected TypeError to surface in errors; got {result.errors}"


@pytest.mark.asyncio
async def test_crystallize_narrator_captures_all_4_telemetry_fields_non_none(
    fake_supabase, monkeypatch
):
    """Codex iter-1 DoD-10: pin the live narrator integration —
    when AsyncAnthropic.messages.create() returns a real-shape
    response (text content + usage block with prompt_tokens +
    completion_tokens), _invoke_llm_narrator MUST populate all 4
    telemetry fields (latency_ms, input_tokens, output_tokens,
    cost_usd) to non-None values.

    The cost UTILITY (compute_haiku_cost_usd) is tested in isolation
    at tests/unit/test_data/test_crystal_narrative_audit.py; this
    test pins the END-TO-END capture inside _invoke_llm_narrator so
    a regression that drops e.g. usage extraction trips here.
    """
    monkeypatch.setenv("E2I_CRYSTAL_LLM_NARRATIVES_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key-for-shape")
    if "dspy" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "dspy",
            types.SimpleNamespace(
                Signature=object,
                InputField=lambda **_kwargs: None,
                OutputField=lambda **_kwargs: None,
                ChainOfThought=lambda *_args, **_kwargs: None,
            ),
        )

    # Build a fake client that returns a Haiku-shaped response.
    # The shape mirrors the real anthropic SDK:
    #   response.content[0].text   ← JSON string with the 3 prose fields
    #   response.usage.input_tokens / output_tokens
    class _FakeContent:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeUsage:
        def __init__(self, input_tokens: int, output_tokens: int) -> None:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class _FakeResponse:
        def __init__(self) -> None:
            self.content = [
                _FakeContent(
                    '{"key_finding": "Northeast lift +0.42 ATE",'
                    ' "limitations": "Pre-period n=120 small",'
                    ' "recommended_next_analysis": "Replicate Q3"}'
                )
            ]
            self.usage = _FakeUsage(input_tokens=800, output_tokens=200)

    class _FakeMessages:
        async def create(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        def __init__(self, **_kwargs):
            self.messages = _FakeMessages()

    def _fake_client_factory(_api_key: str) -> _FakeClient:
        return _FakeClient()

    # Capture the audit emitted by _invoke_llm_narrator directly with
    # synthetic inputs.
    from src.memory.crystallization.crystallizer import _invoke_llm_narrator

    audit = await _invoke_llm_narrator(
        brand="kisqali",
        region="northeast",
        members=[
            {
                "memory_id": "m1",
                "agent_name": "causal_impact",
                "raw_content": {"ate_estimate": 0.42},
            }
        ],
        derived={
            "effect_size": 0.42,
            "effect_ci_lower": 0.30,
            "effect_ci_upper": 0.55,
            "effect_direction": "positive",
            "cohort_size": 1200,
            "confounders_controlled": ["age", "prior_use"],
            "sensitivity_checks_passed": ["placebo_treatment"],
            "sensitivity_checks_failed": [],
        },
        client_factory=_fake_client_factory,
    )

    # All 4 telemetry fields must be non-None.
    assert audit.latency_ms is not None, "latency_ms must capture wall-clock"
    assert audit.latency_ms >= 0.0
    assert audit.input_tokens == 800, "input_tokens must come from usage block"
    assert audit.output_tokens == 200, "output_tokens must come from usage block"
    assert audit.cost_usd is not None, "cost_usd must derive from token counts"
    # Cost = (800 * 1.00 + 200 * 5.00) / 1_000_000 = 0.0018
    expected_cost = (800 * 1.00 + 200 * 5.00) / 1_000_000.0
    assert abs(audit.cost_usd - expected_cost) < 1e-9, (
        f"cost_usd must follow Haiku pricing constants; expected "
        f"{expected_cost}, got {audit.cost_usd}"
    )
    # Prose round-trips from JSON response (also non-empty)
    assert audit.key_finding == "Northeast lift +0.42 ATE"
    assert audit.limitations == "Pre-period n=120 small"
    assert audit.recommended_next_analysis == "Replicate Q3"
    # Model identifier pinned
    assert audit.narrator_model == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_crystallize_narrator_falls_back_on_anthropic_api_error(fake_supabase, monkeypatch):
    """The narrow catch DOES swallow anthropic.* SDK errors and emits
    an empty audit (fall-back contract). Pin so the boundary is
    explicit."""
    monkeypatch.setenv("E2I_CRYSTAL_LLM_NARRATIVES_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key-for-shape")

    anthropic_mod = pytest.importorskip("anthropic")

    class _BoomMessages:
        async def create(self, **kwargs):
            # anthropic.APIConnectionError accepts a `message` arg in
            # modern SDK; positional-or-kwarg works across versions.
            raise anthropic_mod.APIConnectionError(request=None)  # type: ignore[arg-type]

    class _BoomClient:
        def __init__(self, **_kwargs):
            self.messages = _BoomMessages()

    def _boom_client_factory(_api_key: str) -> _BoomClient:
        return _BoomClient()

    _seed_episodic_with_causal_content(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    result = await Crystallizer(anthropic_client_factory=_boom_client_factory).run_for_brand(
        "Kisqali"
    )

    # Insight still created — fall-back path emits empty prose, the
    # row insert succeeds. Critical contract: SDK errors do NOT block
    # the crystal pipeline.
    assert result.insights_created == 1
    insight = fake_supabase.rows["executive_insights"][0]
    # Empty prose (length 0 string) confirms the fall-back audit shape
    assert insight["limitations"] == ""
    assert insight["recommended_next_analysis"] == ""


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


@pytest.mark.asyncio
async def test_run_for_brand_paginates_candidate_select(fake_supabase: FakeSupabase):
    """L7 (#694): the candidate SELECT must page through .range() windows so a
    candidate set larger than one PostgREST page isn't silently truncated.

    Faithful: ``no_range_cap`` simulates PostgREST capping an un-ranged request;
    the real ``run_for_brand`` runs against the fake. Without pagination only the
    first page is seen (examined_groups < total)."""
    fake_supabase.no_range_cap = 2  # simulate a 2-row PostgREST cap
    for i in range(5):  # 5 candidates, each its own group (distinct causal_path_id)
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "agent_name": f"agent{i}",
                "brand": "Kisqali",
                "region": None,
                "causal_path_id": f"cp{i}",
                "event_type": "causal_discovery",
                "description": "d",
                "outcome_type": "o",
                "occurred_at": "2999-01-01T00:00:00+00:00",  # always >= cutoff
                "raw_content": "{}",
                "consolidation_tier": "episodic",
            }
        )

    crystallizer = Crystallizer(min_agents=2)
    crystallizer.candidate_page_size = 2  # set post-construction so RED fails on assert
    result = await crystallizer.run_for_brand("Kisqali")

    # Pages [0,1],[2,3],[4] -> all 5 fetched and grouped. Without pagination the
    # un-ranged request is capped at 2, so examined_groups would be 2.
    assert result.examined_groups == 5
    # Witness the stable sort: the candidate query must order by the unique PK
    # before paging (codex). Removing the .order("memory_id") makes this fail.
    assert ("episodic_memories", "memory_id") in fake_supabase.order_calls
