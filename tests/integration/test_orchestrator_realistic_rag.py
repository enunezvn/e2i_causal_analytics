"""Phase 5 verification with realistic RAG context — load-bearing RouterNode guard.

Issue #268 (codex LOW-5): the original file shipped under PR #259 contained
``TestSelfDispatchRegressionWithRealisticRag`` and
``TestRoutingLeakRegressionWithRealisticRag`` smoke-style integration tests
that asserted ``agent_used != "orchestrator"`` / ``!= "health_score"`` on
TestClient responses. Per PR #259's own attenuation finding,

  "Hop 1 (#254) alone made the F1 symptom non-reproducible through
   TestClient even with realistic_rag — the multi_faceted tie-break
   produces a real agent name in the dispatch plan, so the
   self-dispatch path is no longer hit in practice."

That is: with the deterministic INTENT_PRIORITY tie-break (PR #247 commit
``bf27ff40``) on main, those tests passed by independent code paths.
Reverting cognitive.py's ``orchestrator_degraded`` marker left them green.

The load-bearing falsifiability-verified tests for the same surface live in:

* ``tests/integration/test_cognitive_degraded_marker.py`` — F1/F2 degraded
  marker contract (``AsyncMock`` fixtures FORCE the empty-dispatch and
  raised-orchestrator branches; reverting cognitive.py:365-372 trips them).
* ``tests/unit/test_agents/test_orchestrator/test_router_default_routing_finalization.py``
  — Issue #269 ``_default_routing`` finalization guard.

The unit test below (``TestRouterNeverEmitsOrchestratorHardGuard``) is kept
because it iterates over every ``INTENT_TO_AGENTS`` entry and the synthetic
"unknown" intent — that test is falsifiability-verified for the
``RouterNode``-level F1 strip and exercises a code path the
``test_cognitive_degraded_marker.py`` AsyncMock tests do not (the strip
inside ``execute()`` for every known intent, plus the
``_default_routing``-pathway via the unknown-intent leg).

See memory ``feedback-falsifiability-asyncmock-isolation`` for the
reusable AsyncMock pattern, and ``feedback-testclient-vs-live-divergence``
for why the original smoke tests under-tested.
"""

from __future__ import annotations

import pytest


class TestRouterNeverEmitsOrchestratorHardGuard:
    """Issue #251 F1 acceptance: RouterNode hard guard.

    The RouterNode must structurally never emit a dispatch plan containing
    'orchestrator' — even if a future intent or fallback path tried to. This
    is defense-in-depth on top of the cognitive.py fix.
    """

    @pytest.mark.asyncio
    async def test_router_never_emits_orchestrator_in_dispatch_plan(self) -> None:
        from src.agents.orchestrator.nodes.router import RouterNode

        router = RouterNode()
        # Try every intent that exists in INTENT_TO_AGENTS plus a synthetic
        # unknown one. None should produce an orchestrator-dispatch.
        intents_to_try = list(router.INTENT_TO_AGENTS.keys()) + [
            "nonexistent_intent_to_force_default"
        ]
        for primary_intent in intents_to_try:
            state = {
                "intent": {
                    "primary_intent": primary_intent,
                    "confidence": 0.9,
                    "secondary_intents": [],
                    "requires_multi_agent": False,
                }
            }
            result = await router.execute(state)
            plan = result["dispatch_plan"]
            dispatched_names = [d["agent_name"] for d in plan]
            assert "orchestrator" not in dispatched_names, (
                f"RouterNode emitted 'orchestrator' for intent={primary_intent!r}: "
                f"dispatch_plan={dispatched_names}"
            )
