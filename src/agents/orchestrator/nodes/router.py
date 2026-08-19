"""Router node for orchestrator agent.

Fast routing decisions based on intent classification.
No LLM calls - pure logic.

V4.4: Added discovery routing to pass DAG data to discovery-aware agents.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, cast

from ..state import AgentDispatch, OrchestratorState
from .intent_classifier import _classifier_mode

logger = logging.getLogger(__name__)


class RouterNode:
    """Fast routing decisions based on intent classification.

    No LLM calls - pure logic.

    V4.4: Added discovery routing to pass DAG data to discovery-aware agents.
    """

    # Priority mapping: critical > high > medium > low
    PRIORITY_ORDER = {"critical": 1, "high": 2, "medium": 3, "low": 4}

    # Active-mode floor: below this the 4-stage pipeline abstains and legacy
    # intent routing keeps authority (quality guarantee ≥ today's behavior).
    MIN_ACTIVE_CONFIDENCE = 0.5

    # V4.4: Agents that can use discovered DAG for validation
    DISCOVERY_AWARE_AGENTS = [
        "causal_impact",
        "gap_analyzer",
        "heterogeneous_optimizer",
        "experiment_designer",
    ]

    # Agent capabilities mapping
    INTENT_TO_AGENTS = {
        "causal_effect": [
            AgentDispatch(
                agent_name="causal_impact",
                priority="critical",
                parameters={"interpretation_depth": "standard"},
                # #1419: raised 120s -> 300s on a MEASURED completion (the
                # #1351 convention: raise past 120s only with a measured run).
                # Measured 2026-07-31 on the live 37,371-row conversion frame:
                # estimation (tournament + full-frame refit + prep) ~93s, then
                # refutation on the 5,000-row stratified subsample (#1419) =
                # reconstruction ~12s + 1-sim calibration ~2s + placebo
                # 30 x 2.13s ~64s + random_common_cause 20 x 2.59s ~52s +
                # analytic e-value ~0s -> critical-gates chat turn ~223s.
                # 300s x _CAUSAL_DEADLINE_FRACTION (0.8) = 240s cooperative
                # deadline covers that with margin; the non-critical
                # data_subset (~8s) runs when headroom remains and the
                # non-critical bootstrap (50 x ~11.7s inference-bearing sims
                # ~585s) degrades to an honest SKIPPED result under the #1419
                # skip policy + heavy-cost gate. 300s was chosen to align with
                # the host-nginx proxy_read_timeout ceiling.
                #
                # #1659 (2026-08-16) CORRECTION: matching the ceiling did NOT
                # make this budget "the binding constraint end-to-end", as this
                # comment previously claimed. proxy_read_timeout bounds the
                # SILENT window, and the chat SSE stream was measured silent for
                # the ENTIRE turn (34,395.7ms client-side vs 34,389.4ms of
                # summed node wall time on a live 2026-08-16 request) — so the
                # real constraint was `total turn wall time < 300s`, of which
                # this dispatch is only one term. A 223s critical-gates turn
                # plus ~28s of retrieve_rag/classification/finalisation was
                # already inside 300s only by luck. The silent window is now
                # bounded by with_sse_keepalive instead (see
                # src/api/utils/sse_keepalive.py), which is what actually makes
                # this budget safe. The resolver
                # still sets the cooperative compute_deadline INSIDE this
                # budget so refutation self-gates instead of orphaning
                # to_thread compute.
                timeout_ms=300000,
                fallback_agent="explainer",
            )
        ],
        "performance_gap": [
            AgentDispatch(
                agent_name="gap_analyzer",
                priority="critical",
                parameters={},
                timeout_ms=20000,
                fallback_agent=None,
            )
        ],
        "segment_analysis": [
            AgentDispatch(
                agent_name="heterogeneous_optimizer",
                priority="critical",
                parameters={},
                # heterogeneous_optimizer runs a real CausalForestDML estimation +
                # CausalML hierarchical uplift; on a few-thousand-row KPI substrate this
                # legitimately exceeds 25s, so the old SLA timed out COMPLETING analyses
                # (surfaced by synthetic-causal-validation gate 11, where the full
                # CATE->segment->hierarchical pipeline finishes but the dispatch was cut
                # at 25s). Raised to a workload-appropriate SLA, in line with the other
                # heavy analytical agents (cohort_constructor 120s, tool_composer 180s).
                # 2026-06-11: 120s re-measured too tight — pre-cleanup the resolver
                # bound the small untagged-legacy frame (~4.3k triggers, ~96s); on
                # the clean gold substrate it binds the full 37,378-row conversion
                # frame and the complete run MEASURES 269.7s serialized
                # (LOKY_MAX_CPU_COUNT=1, real CausalForestDML + per-segment
                # effect_interval + hierarchical uplift). 420s = measured + ~55%
                # headroom; a workload-appropriate SLA, not a latency target.
                #
                # #1659 (2026-08-16): this budget is 120s ABOVE the nginx
                # proxy_read_timeout (300s — docker/nginx/host-nginx.conf
                # locations /api/ and /copilotkit/, mirrored in
                # src/api/utils/sse_keepalive.PROXY_READ_TIMEOUT_SECONDS). That
                # only ever mattered because the chat SSE stream was SILENT for
                # the whole turn, and proxy_read_timeout bounds the silent
                # window rather than the request duration. MEASURED through the
                # live host nginx on 2026-08-16, on a turn that dispatched this
                # very agent: one frame at 860.9ms, then 34,395.7ms of nothing
                # against 34,389.4ms of summed node wall time — 6ms apart, i.e.
                # the silent window WAS the whole turn. On that arithmetic even
                # the measured 269.7s run breached 300s once retrieve_rag
                # (23.5s in that trace, up to ~41s per #1484) and the rest of
                # the graph were counted, so the budget did NOT need to reach
                # 420s to sever a completing analysis.
                #
                # The fix bounds the silent window instead of the budget:
                # /api/copilotkit/chat/stream now wraps its body in
                # with_sse_keepalive, which emits an SSE comment every
                # SSE_KEEPALIVE_INTERVAL_SECONDS while the graph is quiet. This
                # budget is therefore free to exceed the proxy ceiling — but
                # only while that wrapper is in place, which
                # tests/unit/test_tests_meta/test_proxy_ceiling_coherence_1659.py
                # asserts.
                timeout_ms=420000,
                fallback_agent="gap_analyzer",
            )
        ],
        "experiment_design": [
            AgentDispatch(
                agent_name="experiment_designer",
                priority="critical",
                parameters={"preregistration_formality": "medium"},
                # 2026-07-29 (#1337 Step 0 empirical pass, #1351): the old 60s
                # budget cut COMPLETING designs — both forced-route attempts
                # timed out at exactly the budget while the live AG-UI surface
                # answered the same asks in a measured 88-90s (LLM-dominated
                # design phases: power analysis, validity audit, DoWhy codegen).
                # 150s = measured + ~67% headroom for LLM-latency variance — a
                # workload-appropriate SLA in line with the other heavy
                # analytical agents above, not a latency target.
                #
                # #1635 (2026-08-16): 150s was still BELOW the agent's own
                # internal step ceilings, so this dispatch timeout could fire
                # mid-graph and DISCARD a design the agent was about to return.
                # The graph is sequential (context_loader → design_reasoning →
                # power_analysis → validity_audit → template_generator) and its
                # two LLM steps declare, in code, on that serial path:
                #   design_reasoning: asyncio.wait_for(primary, timeout=120),
                #     then a fallback LLM with client timeout=60  → ≤180s
                #   validity_audit:   asyncio.wait_for(audit, timeout=90),
                #     which DEGRADES on expiry (validity_audit_status=
                #     "timed_out") and proceeds rather than failing → ≤90s
                # 120+90 = 210s of declared internal budget alone, i.e. 1.4x the
                # old 150s dispatch budget. Confirmed live in the 2026-08-15
                # eval: 3.6 COMPLETED with a full RCT design + power analysis
                # only because validity_audit hit its internal 90s cap and
                # self-degraded (validity score 0.00), while 3.4 — same agent,
                # same brand, two turns apart — was cut at exactly 150000ms and
                # returned nothing. That is a budget-composition defect, not a
                # transient: there is no retry anywhere on this path
                # (_execute_single_agent returns on the first TimeoutError).
                # 240s covers the realistic degraded path — primary reasoning
                # exhausts its 120s cap, the fast fallback LLM answers in ~20s,
                # validity_audit burns its full 90s, +~5s non-LLM steps ≈ 235s —
                # and sits 60s under the host-nginx proxy_read_timeout 300s
                # ceiling (docker/nginx/host-nginx.conf:119), which bounds the
                # SILENT dispatch window: measured, 3.4 emitted no bytes at all
                # until the end (first_progress_ms == total_ms). It deliberately
                # does NOT cover design_reasoning exhausting both its 120s
                # primary AND its full 60s fallback (=275s): that is a wedge,
                # and timing out is the correct guard.
                timeout_ms=240000,
                # No fallback by design (#1635), with the disproof recorded:
                # explainer is the platform's universal fallback, but
                # _resolve_explainer_input fails CLOSED at step (4) when there
                # are no upstream results to explain — and on an
                # experiment_designer timeout it is the ONLY dispatched agent,
                # so an explainer fallback would re-fail with nothing to explain
                # and reproduce the same dead end (the precedent already
                # documented on cohort_definition below). The grounded
                # methodological fallback is restored by the BUDGET instead: the
                # agent's own graceful-degradation path yields a real design,
                # which 150s was cutting off.
                fallback_agent=None,
            )
        ],
        "prediction": [
            AgentDispatch(
                agent_name="prediction_synthesizer",
                priority="critical",
                parameters={},
                timeout_ms=15000,
                fallback_agent=None,
            )
        ],
        "resource_allocation": [
            AgentDispatch(
                agent_name="resource_optimizer",
                priority="critical",
                parameters={},
                timeout_ms=20000,
                fallback_agent=None,
            )
        ],
        "explanation": [
            AgentDispatch(
                agent_name="explainer",
                priority="critical",
                parameters={"depth": "standard"},
                timeout_ms=45000,
                fallback_agent=None,
            )
        ],
        "system_health": [
            AgentDispatch(
                agent_name="health_score",
                priority="critical",
                parameters={},
                # #1634: the previous 5000ms was the ONLY budget in this table
                # with no justifying comment — it shipped unchanged in the
                # initial platform commit (3e1c70cf4) and was never measured.
                # It became user-visible when the AG-UI brain started routing
                # "system health score" through orchestrator_tool (the #1562
                # prompt change advertising that tool's cohort path, fcfb70a64)
                # instead of e2i_data_query_tool: the old route READ stored
                # agent_analysis rows, the new one EXECUTES the agent.
                # MEASURED 2026-08-16 on the faithful chat-path wiring
                # (factory._health_score_kwargs() — the same four real backends
                # create_agent_registry injects), full graph, all four
                # dimensions measured=True, grade A, 0 errors:
                #   cold (fresh process, n=5): 2311/2342/2922/3000/3673 ms
                #   warm (same process):        107–594 ms
                # A chat dispatch hits the COLD path after a worker respawn, so
                # 3673ms is the binding number; 5000ms left only 1.36x over it
                # on an IDLE box — no gunicorn contention, no concurrent
                # dispatch — which is why the live turn timed out while the
                # agent served the same ask in 14.7s wall under the old route.
                # 20000 = measured cold worst x ~5.4, matching the DB-backed
                # peers (gap_analyzer/resource_optimizer 20s); still tight
                # enough to catch a genuinely wedged health check.
                timeout_ms=20000,
                # No fallback by design: with a budget the agent comfortably
                # meets, an expiry means the health substrate itself is down —
                # and the current fail-closed notice (which fabricates nothing
                # and explains itself) is the correct answer to that.
                fallback_agent=None,
            )
        ],
        "drift_check": [
            AgentDispatch(
                agent_name="drift_monitor",
                priority="critical",
                parameters={},
                timeout_ms=10000,
                fallback_agent=None,
            )
        ],
        "feedback": [
            AgentDispatch(
                agent_name="feedback_learner",
                priority="critical",
                parameters={},
                timeout_ms=30000,
                fallback_agent=None,
            )
        ],
        # Tier 3: A/B experiment health monitoring (SRM, interim, enrollment).
        "experiment_monitor": [
            AgentDispatch(
                agent_name="experiment_monitor",
                priority="critical",
                parameters={},
                timeout_ms=15000,
                fallback_agent=None,
            )
        ],
        # Tier 1: Multi-faceted queries decomposed by the Tool Composer.
        "multi_faceted": [
            AgentDispatch(
                agent_name="tool_composer",
                priority="critical",
                parameters={},
                timeout_ms=180000,  # 3-minute SLA per tool_composer agent contract
                fallback_agent="explainer",
            )
        ],
        # Tier 0: cohort/segment chat queries → population profiling with REAL
        # per-segment counts. Routed to cohort_profiler, NOT cohort_constructor:
        # the latter materializes patient rows for the ML pipeline and cannot run
        # from a chat payload (it fell through to a dead-end whose explainer
        # fallback also failed closed — verified by container replay). No fallback:
        # profiling either has real data or fails closed honestly; an explainer
        # fallback would only re-fail with nothing to explain.
        "cohort_definition": [
            AgentDispatch(
                agent_name="cohort_profiler",
                priority="critical",
                parameters={},
                timeout_ms=30000,  # ≤8 sequential DB-backed KPI calls per brand
                fallback_agent=None,
            )
        ],
    }

    # Multi-agent patterns for complex queries (priority: critical > high > medium > low)
    MULTI_AGENT_PATTERNS = {
        ("causal_effect", "segment_analysis"): [
            ("causal_impact", "critical"),
            ("heterogeneous_optimizer", "high"),
        ],
        ("performance_gap", "resource_allocation"): [
            ("gap_analyzer", "critical"),
            ("resource_optimizer", "high"),
        ],
        ("prediction", "explanation"): [
            ("prediction_synthesizer", "critical"),
            ("explainer", "high"),
        ],
    }

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute routing logic.

        Args:
            state: Current orchestrator state

        Returns:
            Updated state with dispatch plan
        """
        start_time = time.time()

        # #1714: an explicit caller-requested target takes routing authority.
        # orchestrator_tool (src/api/routes/chatbot_tools.py) has always
        # stashed the chat model's explicit choice under
        # ``user_context["target_agent"]``, but NO node ever consumed it — the
        # request was silently ignored for every agent and intent routing
        # substituted its own plan (2026-08-19 eval, turn 5.5: requested
        # 'explainer', dispatched ['heterogeneous_optimizer', 'gap_analyzer'];
        # 3.4/3.6/4.4 only LOOKED honored because intent routing
        # coincidentally agreed with the request). An explicit target that
        # resolves to a router-dispatchable agent is dispatched as the sole
        # critical agent — winning over the classification pipeline,
        # multi-agent patterns, AND intent-classification failure (the
        # ``_default_routing`` path below must not shadow it). An
        # unknown/non-dispatchable target falls through to intent routing with
        # a warning logged — the tool payload's ``target_agent_requested`` vs
        # ``agents_dispatched`` pair keeps that mismatch visible to the caller.
        explicit_plan = self._explicit_target_dispatch(state)
        if explicit_plan is not None:
            return self._finalize_explicit_target(state, explicit_plan, start_time)

        intent = state.get("intent")
        if not intent:
            # No intent classified, default to explainer
            return self._default_routing(state, start_time)

        dispatch_plan: List[AgentDispatch] = []
        parallel_groups: List[List[str]] = []

        # Active-mode 4-stage classifier dispatch: when the pipeline made a
        # confident decision, it takes routing authority; on CLARIFICATION /
        # low confidence / unknown pattern it abstains (returns None) and
        # legacy intent routing below proceeds unchanged. Shadow/off modes
        # never enter this branch, keeping today's routing byte-identical.
        # #1582: which subsystem actually decided THIS turn. Computed HERE,
        # where the plan's origin is known — deriving it from
        # `_classifier_mode()` would report "pipeline" for an active-mode turn
        # on which the pipeline ABSTAINED, which is the very conflation the
        # marker exists to remove.
        routing_authority = "legacy"

        classification = state.get("classification")
        if _classifier_mode() == "active" and classification:
            pipeline_plan = self._dispatch_from_classification(classification)
            if pipeline_plan is not None:
                dispatch_plan, parallel_groups = pipeline_plan
                routing_authority = "pipeline"

        # Check for multi-agent patterns
        if (
            not dispatch_plan
            and intent.get("requires_multi_agent")
            and intent.get("secondary_intents")
        ):
            primary_intent = intent["primary_intent"]
            secondary0 = intent["secondary_intents"][0]
            # Order-insensitive lookup: MULTI_AGENT_PATTERNS is keyed canonically,
            # but the query may surface the pair in either order. Match either
            # direction so the deliberate critical/high priorities are preserved
            # (consistent with the classifier's order-insensitive
            # PARALLEL_INTENT_PAIRS deference).
            pattern_key = next(
                (
                    k
                    for k in ((primary_intent, secondary0), (secondary0, primary_intent))
                    if k in self.MULTI_AGENT_PATTERNS
                ),
                None,
            )
            if pattern_key is not None:
                pattern = self.MULTI_AGENT_PATTERNS[pattern_key]
                for agent_name, priority in pattern:
                    dispatch_plan.append(
                        self._get_dispatch_for_agent(
                            agent_name,
                            cast(Literal["low", "medium", "high", "critical"], priority),
                        )
                    )
                # Group by priority for parallel execution
                parallel_groups = self._group_by_priority(dispatch_plan)
            elif intent["primary_intent"] != "multi_faceted":
                # Fix 1 (audit C3): genuinely multi-intent (2 strong intents) but
                # no hard-coded parallel pattern for this pair. Parallel-delegate
                # primary + top secondary rather than SILENTLY DROPPING the
                # secondary intent (the pre-fix behaviour). Dependent *pipelines*
                # are a different case: the classifier promotes those to
                # ``multi_faceted`` (sequential-composition signal), which falls
                # through to single-agent dispatch → ``tool_composer`` below.
                #
                # ``multi_faceted`` is a META-signal, not an agent domain. When
                # it appears only as a *secondary* (a weak "and also"-style hint
                # under a stronger primary, with NO sequential dependency), it is
                # NOT a pipeline — skip it and delegate to the top real-domain
                # secondary so we don't spuriously spawn tool_composer alongside
                # the primary agent.
                secondary_intent = next(
                    (s for s in intent["secondary_intents"] if s != "multi_faceted"),
                    None,
                )
                if secondary_intent is not None:
                    dispatch_plan = [
                        self._get_dispatch_for_agent(
                            self._agent_for_intent(intent["primary_intent"]), "critical"
                        ),
                        self._get_dispatch_for_agent(
                            self._agent_for_intent(secondary_intent), "high"
                        ),
                    ]
                    parallel_groups = self._group_by_priority(dispatch_plan)
                # else: only a multi_faceted meta-hint as secondary → fall through
                # to single-agent dispatch on the (stronger) primary intent.
            # else: primary == "multi_faceted" → leave dispatch_plan empty so the
            # single-agent dispatch below routes to tool_composer (which owns the
            # real sub-question decomposition + dependency DAG).

        # Single agent dispatch
        if not dispatch_plan:
            primary = intent["primary_intent"]
            if primary in self.INTENT_TO_AGENTS:
                dispatch_plan = self.INTENT_TO_AGENTS[primary]
            else:
                # Default to explainer for general queries
                dispatch_plan = [
                    AgentDispatch(
                        agent_name="explainer",
                        priority="medium",
                        parameters={"depth": "minimal"},
                        timeout_ms=30000,
                        fallback_agent=None,
                    )
                ]

        # V4.4: Apply discovery routing to enhance dispatch parameters
        discovery_routing_applied = False
        discovery_aware_agents: List[str] = []

        if self._should_apply_discovery_routing(state):
            dispatch_plan, discovery_aware_agents = self._enhance_with_discovery_data(
                dispatch_plan, state
            )
            discovery_routing_applied = len(discovery_aware_agents) > 0

        # Issue #251 F1 hard guard (centralized for Issue #269).
        # Funnel through a shared finalization so `_default_routing` cannot
        # bypass the strip. See `_apply_self_dispatch_guard` for the helper.
        # NOTE: `intent` is guaranteed truthy here because the `if not intent`
        # branch above returns early via `_default_routing`.
        dispatch_plan = self._apply_self_dispatch_guard(
            dispatch_plan,
            source=f"execute(primary_intent={intent.get('primary_intent', '(none)')!r})",
        )

        # Re-derive parallel_groups from the CLEANED dispatch_plan to keep
        # the parallel_groups view consistent with the strip output. If
        # parallel_groups was computed pre-strip from a multi-agent pattern
        # (line 210) and the strip removed an entry, the stale view would
        # otherwise still reference the forbidden agent name. Today the
        # current MULTI_AGENT_PATTERNS table does not name `orchestrator`,
        # but Issue #269 is explicitly about by-construction protection
        # against future regressions.
        cleaned_names = [d["agent_name"] for d in dispatch_plan]
        if parallel_groups:
            # Filter each priority bucket to drop any stripped names; drop
            # buckets that become empty so downstream consumers don't see
            # phantom empty groups.
            parallel_groups = [
                [name for name in group if name in cleaned_names] for group in parallel_groups
            ]
            parallel_groups = [group for group in parallel_groups if group]

        routing_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "dispatch_plan": dispatch_plan,
            "parallel_groups": parallel_groups or [cleaned_names],
            "routing_latency_ms": routing_time,
            "current_phase": "dispatching",
            # #1582: additive telemetry — never read back for routing.
            "routing_authority": routing_authority,
            # V4.4: Discovery routing metadata
            "discovery_routing_applied": discovery_routing_applied,
            "discovery_aware_agents": discovery_aware_agents if discovery_aware_agents else None,
        }

    def _apply_self_dispatch_guard(
        self,
        dispatch_plan: List[AgentDispatch],
        *,
        source: str = "unknown",
    ) -> List[AgentDispatch]:
        """Strip ``"orchestrator"`` entries from a dispatch plan (Issue #251 F1).

        The orchestrator routes to OTHER agents and must never appear in its
        own ``dispatch_plan``. None of the INTENT_TO_AGENTS or
        MULTI_AGENT_PATTERNS entries name 'orchestrator' today; this helper
        makes that invariant structurally enforced so a future intent
        addition (or a configurable default agent in ``_default_routing``)
        cannot regress it.

        Issue #269: extracted from the inline guard inside ``execute()`` so
        ``_default_routing`` shares the same finalization. Both call sites
        invoke this helper before constructing the return state.

        Args:
            dispatch_plan: The proposed dispatch plan, possibly containing
                an ``agent_name == "orchestrator"`` entry that violates F1.
            source: Free-form caller tag for the warning log line, e.g.
                ``"execute(primary_intent='multi_faceted')"`` or
                ``"_default_routing"``.

        Returns:
            A dispatch plan with all ``"orchestrator"`` entries removed.
            If the strip leaves the plan empty, falls back to a single
            ``"explainer"`` dispatch so the contract that
            ``len(dispatch_plan) >= 1`` is preserved for downstream
            consumers.
        """
        filtered_plan = [d for d in dispatch_plan if d["agent_name"] != "orchestrator"]
        if len(filtered_plan) != len(dispatch_plan):
            logger.warning(
                "RouterNode #251/#269 guard: dropped 'orchestrator' from dispatch_plan; source=%s",
                source,
            )
        if not filtered_plan:
            # Whole plan was orchestrator entries; fall through to explainer.
            filtered_plan = [
                AgentDispatch(
                    agent_name="explainer",
                    priority="medium",
                    parameters={"depth": "minimal"},
                    timeout_ms=30000,
                    fallback_agent=None,
                )
            ]
        return filtered_plan

    def _dispatchable_agents(self) -> set[str]:
        """Agent names this router knows how to dispatch (#1714).

        Derived from ``INTENT_TO_AGENTS`` — the same table every routing path
        resolves through — so an explicit target can only name an agent with a
        deliberate dispatch config (priority/timeout/fallback). Chat-layer
        ``VALID_AGENTS`` deliberately is NOT the reference set: it contains
        ``cohort_constructor``, which cannot run from a chat payload
        (dispatcher.py — the cohort_definition intent routes to
        cohort_profiler instead for exactly that reason).
        """
        return {
            dispatch["agent_name"]
            for intent_agents in self.INTENT_TO_AGENTS.values()
            for dispatch in intent_agents
        }

    def _finalize_explicit_target(
        self,
        state: OrchestratorState,
        explicit_plan: List[AgentDispatch],
        start_time: float,
    ) -> OrchestratorState:
        """Finalize an explicit-target dispatch plan (#1714).

        Mirrors ``execute()``'s finalization — discovery enhancement, the
        Issue #251 F1 self-dispatch guard, parallel_groups derived from the
        CLEANED plan — so the explicit path cannot drift from the intent
        path's invariants. ``routing_authority`` is ``"explicit_target"``
        (#1582 semantics: names which subsystem actually decided THIS turn).
        """
        dispatch_plan = explicit_plan

        discovery_routing_applied = False
        discovery_aware_agents: List[str] = []
        if self._should_apply_discovery_routing(state):
            dispatch_plan, discovery_aware_agents = self._enhance_with_discovery_data(
                dispatch_plan, state
            )
            discovery_routing_applied = len(discovery_aware_agents) > 0

        dispatch_plan = self._apply_self_dispatch_guard(
            dispatch_plan, source="execute(explicit_target)"
        )
        cleaned_names = [d["agent_name"] for d in dispatch_plan]

        return {
            **state,
            "dispatch_plan": dispatch_plan,
            "parallel_groups": [cleaned_names],
            "routing_latency_ms": int((time.time() - start_time) * 1000),
            "current_phase": "dispatching",
            "routing_authority": "explicit_target",
            "discovery_routing_applied": discovery_routing_applied,
            "discovery_aware_agents": discovery_aware_agents if discovery_aware_agents else None,
        }

    def _explicit_target_dispatch(self, state: OrchestratorState) -> Optional[List[AgentDispatch]]:
        """Resolve ``user_context["target_agent"]`` to a dispatch plan (#1714).

        Returns a single-agent critical-priority plan when the caller's
        explicit target names a router-dispatchable agent, or ``None`` to let
        intent routing proceed (no target, blank target, or an unknown /
        non-dispatchable name — the latter logged, and kept visible to the
        caller via the orchestrator_tool payload's ``target_agent_requested``
        vs ``agents_dispatched`` pair).

        ``"orchestrator"`` can never resolve here: it appears in no
        ``INTENT_TO_AGENTS`` entry, so the Issue #251 F1 self-dispatch
        invariant holds by construction on this path too (and ``execute()``
        still funnels the plan through ``_apply_self_dispatch_guard``).
        """
        user_context = state.get("user_context") or {}
        if not isinstance(user_context, dict):
            return None
        target = user_context.get("target_agent")
        if not isinstance(target, str) or not target.strip():
            return None
        normalized = target.strip().lower()
        if normalized not in self._dispatchable_agents():
            logger.warning(
                "RouterNode #1714: explicit target_agent %r is not a dispatchable "
                "agent (known: %s) — falling back to intent routing",
                target,
                sorted(self._dispatchable_agents()),
            )
            return None
        return [self._get_dispatch_for_agent(normalized, "critical")]

    def _dispatch_from_classification(
        self, classification: Dict[str, Any]
    ) -> Optional[tuple[List[AgentDispatch], List[List[str]]]]:
        """Build a dispatch plan from a 4-stage ClassificationResult dump.

        Returns None to ABSTAIN (CLARIFICATION_NEEDED, low confidence, no
        targets, or unknown pattern) — the caller then falls back to legacy
        intent-based routing. Per-agent timeouts/fallbacks are preserved by
        resolving through ``_get_dispatch_for_agent``.
        """
        pattern = classification.get("routing_pattern")
        targets = [t for t in (classification.get("target_agents") or []) if t]
        confidence = classification.get("confidence") or 0.0

        if pattern == "CLARIFICATION_NEEDED" or confidence < self.MIN_ACTIVE_CONFIDENCE:
            logger.info(
                "classification pipeline abstained (pattern=%s, confidence=%.2f) — legacy routing",
                pattern,
                confidence,
            )
            return None

        if pattern == "SINGLE_AGENT" and targets:
            plan = [self._get_dispatch_for_agent(targets[0], "critical")]
            return plan, self._group_by_priority(plan)

        if pattern == "PARALLEL_DELEGATION" and targets:
            plan = [self._get_dispatch_for_agent(targets[0], "critical")]
            plan += [self._get_dispatch_for_agent(t, "high") for t in targets[1:]]
            return plan, self._group_by_priority(plan)

        if pattern == "TOOL_COMPOSER":
            # Reuse the canonical tool_composer dispatch (180s SLA, explainer
            # fallback). Sub-question/dependency handoff into tool_composer
            # parameters is a follow-up — the dispatcher resolves its own
            # substrate today.
            plan = list(self.INTENT_TO_AGENTS["multi_faceted"])
            return plan, self._group_by_priority(plan)

        logger.info(
            "classification pipeline produced no dispatchable plan "
            "(pattern=%s, targets=%s) — legacy routing",
            pattern,
            targets,
        )
        return None

    def _default_routing(self, state: OrchestratorState, start_time: float) -> OrchestratorState:
        """Default routing when intent classification fails.

        Issue #269: this path used to return early WITHOUT the F1 strip, so
        a future refactor that changed the default agent to a configurable
        value (env var, config) could silently re-introduce the self-dispatch
        leak. The fix routes the dispatch plan through
        ``_apply_self_dispatch_guard`` before returning, matching the
        finalization the main ``execute()`` path applies.

        Args:
            state: Current state
            start_time: Routing start time

        Returns:
            Updated state with default dispatch plan
        """
        dispatch_plan = [
            AgentDispatch(
                agent_name="explainer",
                priority="medium",
                parameters={"depth": "minimal"},
                timeout_ms=30000,
                fallback_agent=None,
            )
        ]

        # Issue #269: funnel through the same finalization as `execute()`.
        # Today this is a no-op because the hard-coded default is "explainer",
        # but the structural by-construction guard prevents a future refactor
        # from leaking "orchestrator" through this path.
        dispatch_plan = self._apply_self_dispatch_guard(dispatch_plan, source="_default_routing")

        routing_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "dispatch_plan": dispatch_plan,
            "parallel_groups": [[d["agent_name"] for d in dispatch_plan]],
            "routing_latency_ms": routing_time,
            "current_phase": "dispatching",
            # #1582: this path never consults the pipeline.
            "routing_authority": "legacy",
        }

    def _agent_for_intent(self, intent_name: str) -> str:
        """Resolve the primary agent name for an intent (Fix 1 helper).

        Falls back to ``explainer`` for unknown intents so the parallel-delegation
        path can never produce an empty/invalid agent name.
        """
        agents = self.INTENT_TO_AGENTS.get(intent_name)
        if agents:
            return agents[0]["agent_name"]
        return "explainer"

    def _get_dispatch_for_agent(
        self, agent_name: str, priority: Literal["low", "medium", "high", "critical"]
    ) -> AgentDispatch:
        """Get dispatch config for a specific agent.

        Args:
            agent_name: Name of agent
            priority: Priority level ("critical", "high", "medium", "low")

        Returns:
            Agent dispatch configuration
        """
        for intent_agents in self.INTENT_TO_AGENTS.values():
            for dispatch in intent_agents:
                if dispatch["agent_name"] == agent_name:
                    return AgentDispatch(**{**dispatch, "priority": priority})

        # Default dispatch
        return AgentDispatch(
            agent_name=agent_name,
            priority=priority,
            parameters={},
            timeout_ms=30000,
            fallback_agent=None,
        )

    def _group_by_priority(self, dispatches: List[AgentDispatch]) -> List[List[str]]:
        """Group agents by priority for parallel execution.

        Args:
            dispatches: List of dispatch configurations

        Returns:
            List of agent groups by priority (critical first, then high, medium, low)
        """
        groups = defaultdict(list)
        for d in dispatches:
            groups[d["priority"]].append(d["agent_name"])
        # Sort by priority order: critical=1, high=2, medium=3, low=4
        return [
            groups[p] for p in sorted(groups.keys(), key=lambda x: self.PRIORITY_ORDER.get(x, 99))
        ]

    # ========================================================================
    # V4.4: Discovery Routing Methods
    # ========================================================================

    def _should_apply_discovery_routing(self, state: OrchestratorState) -> bool:
        """Check if discovery routing should be applied.

        Discovery routing is applied when:
        1. enable_discovery is True OR propagate_discovered_dag is True
        2. Gate decision is NOT 'reject'

        Args:
            state: Current orchestrator state

        Returns:
            True if discovery routing should be applied
        """
        # Check if discovery is enabled or DAG propagation is requested
        enable_discovery = state.get("enable_discovery", False)
        propagate_dag = state.get("propagate_discovered_dag", False)

        if not (enable_discovery or propagate_dag):
            return False

        # Check gate decision - reject means don't use DAG
        gate_decision = state.get("discovery_gate_decision")
        if gate_decision == "reject":
            return False

        return True

    def _enhance_with_discovery_data(
        self,
        dispatch_plan: List[AgentDispatch],
        state: OrchestratorState,
    ) -> tuple[List[AgentDispatch], List[str]]:
        """Enhance dispatch parameters with discovery data for discovery-aware agents.

        Args:
            dispatch_plan: Current dispatch plan
            state: Current orchestrator state with discovery data

        Returns:
            Tuple of (enhanced dispatch plan, list of agents that received DAG data)
        """
        enhanced_plan: List[AgentDispatch] = []
        discovery_aware_agents: List[str] = []

        # Extract discovery data from state
        discovery_config = state.get("discovery_config")
        dag_adjacency = state.get("discovered_dag_adjacency")
        dag_nodes = state.get("discovered_dag_nodes")
        dag_edge_types = state.get("discovered_dag_edge_types")
        gate_decision = state.get("discovery_gate_decision")
        gate_confidence = state.get("discovery_gate_confidence")

        # Check if we have DAG data to propagate
        has_dag_data = dag_adjacency is not None and dag_nodes is not None

        for dispatch in dispatch_plan:
            agent_name = dispatch.get("agent_name", "")

            # Check if this agent is discovery-aware
            if agent_name in self.DISCOVERY_AWARE_AGENTS:
                # Create enhanced parameters
                enhanced_params = dict(dispatch.get("parameters", {}))

                # Add discovery config if available
                if discovery_config:
                    enhanced_params["discovery_config"] = discovery_config

                # Add DAG data if available and propagation is enabled
                if has_dag_data and state.get("propagate_discovered_dag", True):
                    enhanced_params["discovered_dag_adjacency"] = dag_adjacency
                    enhanced_params["discovered_dag_nodes"] = dag_nodes
                    if dag_edge_types:
                        enhanced_params["discovered_dag_edge_types"] = dag_edge_types

                    # Add gate decision for validation
                    if gate_decision:
                        enhanced_params["discovery_gate_decision"] = gate_decision
                    if gate_confidence is not None:
                        enhanced_params["discovery_gate_confidence"] = gate_confidence

                    discovery_aware_agents.append(agent_name)

                # Create enhanced dispatch
                enhanced_dispatch = AgentDispatch(
                    agent_name=agent_name,
                    priority=dispatch.get("priority", "medium"),
                    parameters=enhanced_params,
                    timeout_ms=dispatch.get("timeout_ms", 30000),
                    fallback_agent=dispatch.get("fallback_agent"),
                )
                enhanced_plan.append(enhanced_dispatch)
            else:
                # Non-discovery-aware agent, keep original dispatch
                enhanced_plan.append(dispatch)

        return enhanced_plan, discovery_aware_agents


# Export for use in graph
async def route_to_agents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for routing.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    from src.agents.orchestrator.state import OrchestratorState

    router = RouterNode()
    return cast(Dict[str, Any], await router.execute(cast(OrchestratorState, state)))
