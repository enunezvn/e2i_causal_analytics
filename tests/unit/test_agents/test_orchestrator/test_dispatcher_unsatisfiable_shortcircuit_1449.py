"""#1449 — an UNSATISFIABLE dispatch must refuse without any data round-trip.

WHY THIS FILE IS A GUARD, NOT A FIX
-----------------------------------
Issue #1449 attributes demo 4.3's ~80 s to the dispatcher — *"both agents are
dispatched, spend ~80 s, and THEN fail closed on preconditions that were
knowable before dispatch"* — and proposes promoting those preconditions to a
cheap pre-dispatch check. That premise was MEASURED on this box (prod == dev ==
this host) against the exact 4.3 payload, and it is FALSE. The refusal is
already instant:

    _extract_brand_region                        1.3 ms  -> (None, None)
    kpi_resolution.recognize_kpi               150.1 ms  -> WS1-DQ-002
                                                            (registry load, cached)
    kpi_resolution.resolve_kpi_frame             0.0 ms  -> None (no builder)
    _resolve_heterogeneous_optimizer_input       0.2 ms  -> NeedsStructuredInput
    _resolve_gap_analyzer_input                  0.1 ms  -> NeedsStructuredInput
    (warm re-run: 0.1 ms / 0.1 ms)

The cheap preconditions ALREADY come first, by construction:

  * ``_resolve_gap_analyzer_input`` gates ``_probe_gap_substrate`` behind
    ``if brand:`` — a payload naming no brand never reaches Supabase.
  * ``resolve_kpi_frame`` returns ``None`` at its ``_BUILDERS.get(kpi.id)``
    guard for every KPI without a substrate builder, before any fetch.

And the preconditions that DO run after data work — "the substrate has no rows
for this brand", ">= 100 real rows to bind the causal spec" — are NOT pure
functions of the dispatch payload: answering them requires the query. They
cannot be promoted without doing the same work. So there is no dispatcher
latency to reclaim, and no production change ships for it: 4.3's real ~80 s is
the #1336 conversational bridge, which runs precisely BECAUSE the orchestrator
failed completely — and the routing fix in the companion commit removes that
complete failure.

What survives the disproof is issue #1449's own item 3: a regression test that
the short-circuit holds. It pins the property STRUCTURALLY rather than by
timing (a wall-clock assertion would be a box-load flake here): the two
data-fetch boundaries on this path are replaced by doubles that RAISE, so any
future change that puts a round-trip in front of the cheap gate — a new
``_BUILDERS`` entry that materializes a frame for a treatment-less KPI, or a
gap probe hoisted above the brand check — turns this suite red instead of
silently reintroducing the expensive failure #1449 describes.

The fail-closed MESSAGES are correct #883 behaviour and are pinned verbatim: a
bare chat query genuinely cannot name a treatment/outcome column, and an ask
naming no brand has no business_metrics substrate to scope. Nothing here
weakens them or substitutes a fabricated fallback.

No mocking of the unit under test: the real ``_resolve_*_input`` functions and
the real ``DispatcherNode._dispatch_agent`` run end to end. Doubles sit only at
the external data boundary (``kpi_resolution._fetch_df`` — the single Supabase
read primitive every KPI substrate builder goes through; ``_probe_gap_substrate``
— the business_metrics probe) and at the agent boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator.nodes import dispatcher as D

# The verbatim 4.3 ask that produced the 2026-08-03 report (request 70a4b5d1).
QUERY_43 = "Segment HCPs by prescription volume into high, medium, and low tiers"


def _payload(query: str = QUERY_43) -> Dict[str, Any]:
    """The chat-shaped dispatch payload ``_prepare_agent_input`` builds.

    Deliberately carries no ``parsed_query`` entities and no ``user_context``
    brand: the chat path has no producer for either (#1351), which is exactly
    the unsatisfiable case 4.3 hit.
    """
    return {
        "query": query,
        "user_context": {},
        "parsed_query": None,
        "session_id": "sess-1449",
        "parameters": {},
        "agent_results": [],
    }


def _dispatch(agent_name: str, timeout_ms: int) -> Dict[str, Any]:
    return {
        "agent_name": agent_name,
        "priority": "critical",
        "parameters": {},
        "timeout_ms": timeout_ms,
        "fallback_agent": None,
    }


@pytest.fixture
def no_data_round_trips(monkeypatch) -> List[str]:
    """Make every data round-trip on this dispatch path EXPLODE if reached.

    ``kpi_resolution._fetch_df`` is the single Supabase read primitive every KPI
    substrate builder funnels through, so patching it covers the builders that
    exist today AND any added later — the exact future in which #1449's feared
    "materialize first, discover the precondition second" would appear.
    ``_probe_gap_substrate`` is gap_analyzer's business_metrics read.
    """
    breached: List[str] = []

    def _boom_fetch(*args: Any, **kwargs: Any):
        breached.append("kpi_resolution._fetch_df")
        raise AssertionError(
            "a KPI substrate fetch was issued for an unsatisfiable dispatch — the "
            "cheap precondition must refuse BEFORE any data round-trip"
        )

    def _boom_gap_probe(*args: Any, **kwargs: Any):
        breached.append("_probe_gap_substrate")
        raise AssertionError(
            "_probe_gap_substrate reached for a dispatch that names no brand — the "
            "cheap precondition must refuse BEFORE any data round-trip"
        )

    monkeypatch.setattr("src.services.kpi_resolution._fetch_df", _boom_fetch)
    monkeypatch.setattr(D, "_probe_gap_substrate", _boom_gap_probe)
    return breached


# ---------------------------------------------------------------------------
# The resolvers: refuse on the payload-pure gate, before any round-trip.
# ---------------------------------------------------------------------------
class TestUnsatisfiableResolversDoNoDataWork:
    def test_het_optimizer_refuses_without_a_kpi_substrate_fetch(
        self, no_data_round_trips: List[str]
    ) -> None:
        out = D._resolve_heterogeneous_optimizer_input(
            _payload(), _dispatch("heterogeneous_optimizer", 420000)
        )
        assert isinstance(out, D.NeedsStructuredInput)
        assert no_data_round_trips == []

    def test_causal_impact_refuses_without_a_kpi_substrate_fetch(
        self, no_data_round_trips: List[str]
    ) -> None:
        """causal_impact is the het resolver's template (#1351) — same shape,
        same guard."""
        out = D._resolve_causal_impact_input(_payload(), _dispatch("causal_impact", 300000))
        assert isinstance(out, D.NeedsStructuredInput)
        assert no_data_round_trips == []

    def test_gap_analyzer_refuses_without_a_business_metrics_probe(
        self, no_data_round_trips: List[str]
    ) -> None:
        out = D._resolve_gap_analyzer_input(_payload(), _dispatch("gap_analyzer", 20000))
        assert isinstance(out, D.NeedsStructuredInput)
        assert no_data_round_trips == []

    def test_recognizing_a_kpi_is_not_a_licence_to_materialize(
        self, no_data_round_trips: List[str]
    ) -> None:
        """States the precondition #1449 is really about.

        ``recognize_kpi`` is liberal — on the 4.3 ask its distinctive-token
        fallback matches the DATA-QUALITY KPI *Source Coverage - HCPs* purely
        because "hcps" appears in both. Recognition alone must never trigger a
        substrate read; only a KPI with a real builder may be materialized, and
        that is a static registry fact costing no I/O.
        """
        from src.services import kpi_resolution

        kpi = kpi_resolution.recognize_kpi(QUERY_43)
        assert kpi is not None, "precondition: the 4.3 ask does recognize a KPI"
        assert kpi_resolution.resolve_kpi_frame(kpi, None, None) is None
        assert no_data_round_trips == []


# ---------------------------------------------------------------------------
# The #883 refusal text is a contract — pinned verbatim, not paraphrased.
# ---------------------------------------------------------------------------
class TestRefusalMessagesAreUnweakened:
    def test_het_reason_is_verbatim(self, no_data_round_trips: List[str]) -> None:
        out = D._resolve_heterogeneous_optimizer_input(
            _payload(), _dispatch("heterogeneous_optimizer", 420000)
        )
        assert isinstance(out, D.NeedsStructuredInput)
        assert out.agent_name == "heterogeneous_optimizer"
        assert out.missing == ("treatment_var", "outcome_var", "effect_modifiers")
        assert out.reason == (
            "no recognized KPI substrate with a defined treatment and >=100 real "
            "rows to bind the causal spec; a chat query alone cannot name the "
            "treatment/outcome/effect-modifier columns"
        )
        assert "Failing closed — no values were fabricated." in out.to_error()

    def test_gap_reason_is_verbatim(self, no_data_round_trips: List[str]) -> None:
        out = D._resolve_gap_analyzer_input(_payload(), _dispatch("gap_analyzer", 20000))
        assert isinstance(out, D.NeedsStructuredInput)
        assert out.agent_name == "gap_analyzer"
        assert out.missing == ("metrics", "segments", "brand")
        assert out.reason == (
            "the dispatch names no brand (parameters / parsed_query entities / "
            "user_context), so there is no real business_metrics substrate to "
            "derive metrics/segments from"
        )
        assert "Failing closed — no values were fabricated." in out.to_error()


# ---------------------------------------------------------------------------
# End to end through the real dispatcher: refuse without entering the agent.
# ---------------------------------------------------------------------------
class TestUnsatisfiableDispatchNeverReachesTheAgent:
    @staticmethod
    def _never_run_agent() -> MagicMock:
        agent = MagicMock()
        agent.analyze = AsyncMock(
            side_effect=AssertionError("agent must not run on an unsatisfiable dispatch")
        )
        return agent

    @pytest.mark.asyncio
    async def test_het_optimizer_dispatch_fails_closed_before_the_agent_runs(
        self, no_data_round_trips: List[str]
    ) -> None:
        agent = self._never_run_agent()
        node = D.DispatcherNode(agent_registry={"heterogeneous_optimizer": agent})
        state: Dict[str, Any] = {
            "query": QUERY_43,
            "user_context": {},
            "dispatch_plan": [_dispatch("heterogeneous_optimizer", 420000)],
            "parallel_groups": [["heterogeneous_optimizer"]],
        }

        result = await node.execute(state)  # type: ignore[arg-type]

        res = result["agent_results"][0]
        assert res["success"] is False
        assert res["result"] is None
        assert "no recognized KPI substrate" in res["error"]
        assert "no values were fabricated" in res["error"]
        agent.analyze.assert_not_awaited()
        assert no_data_round_trips == []

    @pytest.mark.asyncio
    async def test_gap_analyzer_dispatch_fails_closed_before_the_agent_runs(
        self, no_data_round_trips: List[str]
    ) -> None:
        """gap_analyzer is heterogeneous_optimizer's configured fallback, so this
        is the SECOND refusal 4.3 collected in the live report."""
        agent = self._never_run_agent()
        node = D.DispatcherNode(agent_registry={"gap_analyzer": agent})
        state: Dict[str, Any] = {
            "query": QUERY_43,
            "user_context": {},
            "dispatch_plan": [_dispatch("gap_analyzer", 20000)],
            "parallel_groups": [["gap_analyzer"]],
        }

        result = await node.execute(state)  # type: ignore[arg-type]

        res = result["agent_results"][0]
        assert res["success"] is False
        assert res["result"] is None
        assert "names no brand" in res["error"]
        agent.analyze.assert_not_awaited()
        assert no_data_round_trips == []


# ---------------------------------------------------------------------------
# The gate must NOT over-fire: a satisfiable payload still does its data work.
# ---------------------------------------------------------------------------
class TestSatisfiablePayloadStillProbesTheSubstrate:
    def test_a_brand_named_in_the_ask_text_still_reaches_the_probe(self, monkeypatch) -> None:
        """The cheap gate is "no brand", not "no structured parsed_query": a
        brand named only in the ask text binds via the #1351 text scan and MUST
        still probe. Guards against "fixing" the latency by refusing more.
        """
        seen: List[Any] = []

        def _record_probe(brand: str, include_synthetic: bool):
            # Only the BRAND is asserted. include_synthetic is governed by the
            # separate #872/#877/#880 provenance-opt-in contract (and by the
            # deployment-wide E2I_INCLUDE_SYNTHETIC badge that the dev box's
            # .env sets), so pinning it here would couple this guard to an
            # unrelated contract and to ambient env.
            seen.append(brand)
            return ["trx"], brand, ["northeast"]

        monkeypatch.setattr(D, "_probe_gap_substrate", _record_probe)

        out = D._resolve_gap_analyzer_input(
            _payload("Where is Kisqali underperforming?"), _dispatch("gap_analyzer", 20000)
        )

        assert seen == ["Kisqali"]
        assert not isinstance(out, D.NeedsStructuredInput)
        assert out["brand"] == "Kisqali"
