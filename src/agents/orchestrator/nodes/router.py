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
                # #1351: 30s could not hold ANY real causal run — the DoWhy
                # chain (graph build + estimation + refutation + sensitivity +
                # LLM interpretation) measures ~15s for the refutation suite
                # ALONE on linear estimators and ~60s on meta-learners
                # (refutation.py SLA notes), so the old budget only ever cut
                # COMPLETING analyses. Before the #1351 resolver this never
                # surfaced because every chat dispatch crashed in <10ms on
                # missing inputs. 120s is the default per-agent ceiling: a
                # timeout >= the 150s chat surface budget could never fire
                # before the chat itself times out, and no completed run has
                # been measured yet (the route was inoperable) — raise past
                # 120s only with a measured completion time, per the
                # experiment_designer/het_optimizer convention. The resolver
                # also sets a cooperative compute_deadline INSIDE this budget
                # so refutation self-gates instead of orphaning to_thread
                # compute.
                timeout_ms=120000,
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
                timeout_ms=150000,
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
                timeout_ms=5000,
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
        classification = state.get("classification")
        if _classifier_mode() == "active" and classification:
            pipeline_plan = self._dispatch_from_classification(classification)
            if pipeline_plan is not None:
                dispatch_plan, parallel_groups = pipeline_plan

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
