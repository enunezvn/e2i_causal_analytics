"""Graph Builder Node - Causal DAG construction.

Constructs causal DAGs using domain knowledge and LLM assistance.
Identifies treatment/outcome variables and valid adjustment sets.
Computes DAG version hash for expert review workflow.

V4.4 Enhancement: Automatic causal structure learning with multi-algorithm
ensemble (GES, PC) and gated acceptance for discovered DAGs.

Version: 4.4

Split by concern (Lane B item 5, PR #2230 follow-up): the method bodies live
in sibling mixin modules — ``manual_dag`` (domain-knowledge DAG),
``discovery_orchestration`` (discovery run, gate, gate-decision DAG) and
``discovery_reporting`` (warnings, edge provenance) — and this module stays the
node entry: ``GraphBuilderNode.execute``, the adjustment-set handling and the
discovered-DAG persistence, plus re-exports of every name importers read here.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Set, cast

import networkx as nx
import pandas as pd

from src.agents.causal_impact.nodes.adjustment_search import (
    ADJUSTMENT_SEARCH_MAX_CANDIDATES,
    ADJUSTMENT_SEARCH_TIME_BUDGET_S,
    find_adjustment_sets,
)
from src.agents.causal_impact.nodes.discovery_orchestration import DiscoveryOrchestrationMixin
from src.agents.causal_impact.nodes.discovery_reporting import DiscoveryReportingMixin
from src.agents.causal_impact.state import CausalGraph, CausalImpactState, spread_safe
from src.causal_engine import compute_dag_hash
from src.causal_engine.discovery import DiscoveryGateDecision, DiscoveryResult
from src.ml.causal_role_dgp.backdoor import satisfies_backdoor_criterion

logger = logging.getLogger(__name__)

# Re-exports. The split moved the definitions or the last use of these names
# out of this module; importers and tests still read them from here (pinned
# by tests/unit/test_agents/test_causal_impact/test_graph_builder_split_surface.py).
from typing import Tuple  # noqa: F401

from src.agents.causal_impact.nodes.discovery_orchestration import (  # noqa: F401
    DISCOVERY_BOOTSTRAP_RESAMPLES,
    DISCOVERY_MAX_COVARIATES,
    DISCOVERY_MIN_RESAMPLES,
    DISCOVERY_TIME_BUDGET_S,
)
from src.causal_engine.discovery import (  # noqa: F401
    DEFAULT_DISCOVERY_ALGORITHM_NAMES,
    DEFAULT_DISCOVERY_ALGORITHMS,
    CausalPriorKnowledge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryGate,
    DiscoveryRunner,
)
from src.causal_engine.discovery.preflight import (  # noqa: F401
    preflight_discovery_frame,
    preflight_summary,
)
from src.utils.session_ids import coerce_session_uuid  # noqa: F401


class GraphBuilderNode(DiscoveryOrchestrationMixin, DiscoveryReportingMixin):
    """Builds causal DAG from query and domain knowledge.

    V4.4 Enhancement: Supports automatic DAG discovery using structure learning
    algorithms (GES, PC) with gated acceptance criteria.

    Modes:
    - Manual (default): Constructs DAG from domain knowledge
    - Auto-Discovery: Uses ensemble of algorithms to discover structure
    - Hybrid: Augments manual DAG with high-confidence discovered edges

    Performance target: <10s (manual), <30s (discovery)
    Type: Standard (computation-heavy)
    """

    async def execute(self, state: CausalImpactState) -> Dict:
        """Build causal DAG with optional auto-discovery.

        Args:
            state: Current workflow state with query and variables

        Returns:
            Updated state with causal_graph populated

        V4.4: If auto_discover=True, attempts structure learning first,
        then falls back to manual DAG based on gate decision.
        """
        start_time = time.time()

        try:
            # Extract or infer treatment/outcome (contract-aligned field names)
            treatment = state.get("treatment_var")
            outcome = state.get("outcome_var")
            confounders = state.get("confounders", [])

            if not treatment or not outcome:
                treatment, outcome = self._infer_variables_from_query(state.get("query", ""))

            # Lane E item 3(d): the feature-role panel, when the caller supplied
            # one, decides which declared covariates may be adjusted for. Leak-
            # verdict covariates (post-index contract / confident outcome leak)
            # leave ``confounders`` + ``modeled_confounders`` with a NAMED
            # warning instead of being adjusted for blind; approved structure
            # (Lane B's ``approved_structure_roles`` seam) anchors confounders.
            # The narrowed channels are written back so every downstream node
            # and the API response see the same adjustment set.
            panel_warnings: List[str] = []
            excluded_columns: List[str] = []
            panel_payload = state.get("feature_role_panel")
            if panel_payload:
                from src.causal_engine.feature_role_panel import derive_confounder_channels

                _frame0 = (state.get("data_cache") or {}).get("estimation_data")
                channels = derive_confounder_channels(
                    panel_payload,
                    declared_covariates=[str(c) for c in (confounders or [])],
                    approved_structure_roles=state.get("approved_structure_roles"),
                    frame_columns=(
                        [str(c) for c in _frame0.columns]
                        if _frame0 is not None and hasattr(_frame0, "columns")
                        else None
                    ),
                )
                confounders = list(channels.modeled_confounders)
                narrowed: Dict[str, Any] = {
                    "confounders": list(confounders),
                    "modeled_confounders": list(confounders),
                }
                if channels.anchored_confounders is not None:
                    narrowed["anchored_confounders"] = list(channels.anchored_confounders)
                if channels.instruments:
                    existing = [str(i) for i in (state.get("instruments") or [])]
                    narrowed["instruments"] = existing + [
                        i for i in channels.instruments if i not in existing
                    ]
                # Excluded columns must leave the FRAME, not just the declared
                # lists: guided discovery tiers every frame column as a candidate
                # confounder and the estimator's no-backdoor fallback adjusts on
                # every column, so a column left in the frame can re-enter an
                # ACCEPT/AUGMENT DAG or the adjustment set (codex r3).
                excluded_columns = list(
                    dict.fromkeys(
                        [name for name, _why in channels.removed] + channels.excluded_frame_columns
                    )
                )
                _cache = dict(state.get("data_cache") or {})
                _frame = _cache.get("estimation_data")
                if excluded_columns and _frame is not None and hasattr(_frame, "columns"):
                    present = [c for c in excluded_columns if c in _frame.columns]
                    if present:
                        _cache["estimation_data"] = _frame.drop(columns=present)
                        narrowed["data_cache"] = _cache
                # spread_safe: the accumulator channels stay out of the rebound
                # state (the node returns only NEW warnings; see the return).
                state = cast(CausalImpactState, {**spread_safe(state), **narrowed})
                _pp = panel_payload if isinstance(panel_payload, dict) else {}
                panel_warnings = [
                    "feature_role_panel applied (caller-supplied; submit established only: "
                    "registered manifest, exact treatment/outcome, at least one covariate overlap, "
                    "structural invariants, dataset-manifest binding only when the dataset declares "
                    "one; provenance not verified): "
                    f"manifest={_pp.get('manifest_source', '?')}, question "
                    f"{_pp.get('treatment', '?')} -> {_pp.get('outcome', '?')}, "
                    f"{len(_pp.get('records') or {})} covariate(s) in the panel, "
                    f"{len(channels.removed)} removed from the adjustment set and the frame, "
                    f"{len(channels.review_required)} pending temporal review; "
                    "approved instruments are not consumed by estimation"
                ] + list(channels.warnings)
                logger.info(
                    "feature_role_panel applied: modeled=%s removed=%s anchored=%s instruments=%s",
                    confounders,
                    channels.removed,
                    channels.anchored_confounders,
                    channels.instruments,
                )

            # Check if auto-discovery is enabled
            auto_discover = state.get("auto_discover", False)
            discovery_result: Optional[DiscoveryResult] = None
            gate_evaluation: Optional[Dict[str, Any]] = None
            discovery_latency_ms: float = 0.0
            discovery_skip_reason: Optional[str] = None

            if auto_discover:
                logger.info("Auto-discovery enabled, attempting structure learning")
                discovery_start = time.time()

                try:
                    discovery_result, gate_evaluation = await self._run_discovery(
                        state, treatment, outcome
                    )
                    discovery_latency_ms = (time.time() - discovery_start) * 1000
                    # A run in which no algorithm converged (runner: success=False)
                    # is a skip with a cause, not a discovered-empty structure:
                    # surface it exactly like the exception path below, so the
                    # reason reaches the state's warnings and the API response.
                    # The gate still evaluates it (REJECT) and the manual DAG
                    # ships, unchanged.
                    if discovery_result is not None and not discovery_result.success:
                        discovery_skip_reason = (
                            "auto-discovery could not run, falling back to manual DAG: "
                            f"{discovery_result.metadata.get('error') or 'no algorithm converged'}"
                        )
                        logger.warning(discovery_skip_reason)
                except Exception as e:
                    # M-gb1: surface the skip as a distinct, non-swallowed signal
                    # instead of only logging. The pipeline still degrades
                    # gracefully to a manual DAG, but the skip is now observable
                    # in state (discovery_skip_reason + warnings accumulator).
                    discovery_skip_reason = (
                        f"auto-discovery skipped, falling back to manual DAG: {e}"
                    )
                    logger.warning(discovery_skip_reason)
                    discovery_latency_ms = (time.time() - discovery_start) * 1000

            # Build DAG based on discovery results
            dag_overridden = False
            if discovery_result and gate_evaluation:
                # The ACCEPT-path narrowing applies only to callers that OPTED
                # INTO the channel split by setting anchored_confounders; a
                # legacy state (confounders only) keeps the old full re-add.
                dag, augmented_edges, dag_overridden = self._build_dag_with_discovery(
                    treatment,
                    outcome,
                    confounders,
                    discovery_result,
                    gate_evaluation,
                    anchored_confounders=(
                        self._resolve_anchored_confounders(state)
                        if "anchored_confounders" in state
                        else None
                    ),
                )
            else:
                # Manual DAG construction (original behavior)
                dag = self._construct_dag(treatment, outcome, confounders)
                augmented_edges = []

            # Lane D item 1: covariates the pre-flight kept away from the
            # structure learner are still declared confounders. On the manual
            # paths they are already nodes; on ACCEPT the shipped DAG is the
            # ensemble over the CAPPED frame, so add them back as isolated
            # nodes (no structure was learned for them — drawing curated edges
            # here would let the API's dag_source classifier read them as a
            # data contribution) so _apply_adjustment_guarantee unions them.
            preflight_removed = self._preflight_removed(discovery_result)
            for covariate in preflight_removed:
                if covariate not in dag and covariate not in (treatment, outcome):
                    dag.add_node(covariate)

            # Find valid adjustment sets (backdoor criterion), then enforce the
            # adjustment guarantee: declared (modeled) confounders are unioned
            # into every set regardless of what the DAG shows (fix 4).
            # #2233: the size-<=3 enumeration is O(k^3) criterion checks and used to
            # run on the event-loop thread (76,154 checks / 242-316 s at k = 77;
            # the live AUGMENT run was aborted by gunicorn's 120 s heartbeat). It
            # now runs in a worker thread, bounded; a hit bound is a warning.
            search = await asyncio.to_thread(
                find_adjustment_sets,
                dag,
                treatment,
                outcome,
                time_budget_s=state.get(
                    "adjustment_search_time_budget_s", ADJUSTMENT_SEARCH_TIME_BUDGET_S
                ),
                max_candidates=state.get(
                    "adjustment_search_max_candidates", ADJUSTMENT_SEARCH_MAX_CANDIDATES
                ),
                criterion=self._satisfies_backdoor_criterion,
            )
            adjustment_sets = search.adjustment_sets
            search_warning = search.warning()
            if search_warning is not None:
                logger.warning(search_warning)
            adjustment_sets = self._apply_adjustment_guarantee(
                dag, treatment, outcome, state, adjustment_sets
            )
            if excluded_columns:
                # Belt and braces for the panel's exclusions: whatever DAG shipped,
                # no excluded column is adjusted for (spec 3(b)).
                denylist = set(excluded_columns)
                deduped: List[List[str]] = []
                for adj_set in adjustment_sets:
                    kept = [c for c in adj_set if c not in denylist]
                    if kept not in deduped:
                        deduped.append(kept)
                adjustment_sets = deduped

            # Compute confidence based on discovery results
            if (
                gate_evaluation
                and gate_evaluation.get("decision") == DiscoveryGateDecision.ACCEPT.value
            ):
                confidence = gate_evaluation.get("confidence", 0.85)
            elif (
                gate_evaluation
                and gate_evaluation.get("decision") == DiscoveryGateDecision.AUGMENT.value
            ):
                # Hybrid confidence
                confidence = min(0.9, 0.85 + 0.05 * len(augmented_edges))
            else:
                confidence = 0.85 if treatment and outcome else 0.5

            # Convert to CausalGraph
            causal_graph: CausalGraph = {
                "nodes": list(dag.nodes()),
                "edges": list(dag.edges()),
                "treatment_nodes": [treatment],
                "outcome_nodes": [outcome],
                "adjustment_sets": adjustment_sets,
                "dag_dot": self._to_dot_format(dag),
                "confidence": confidence,
                # V4.4: Discovery metadata
                "discovery_enabled": auto_discover,
                "discovery_gate_decision": cast(
                    Literal["accept", "review", "reject", "augment"],
                    gate_evaluation.get("decision") if gate_evaluation else "accept",
                ),
                "discovery_algorithms_used": (
                    [a.value for a in discovery_result.config.algorithms]
                    if discovery_result and discovery_result.config
                    else []
                ),
                "discovery_confidence": gate_evaluation.get("confidence", 0.0)
                if gate_evaluation
                else 0.0,
                "discovery_n_edges": discovery_result.n_edges if discovery_result else 0,
                "augmented_edges": augmented_edges,
                # Honest provenance: True only when an ACCEPTED discovered DAG was
                # discarded for the manual one (curated-confounder contradiction).
                "discovery_dag_overridden": dag_overridden,
                # Fix 4: per-edge provenance (required_prior / discovered / curated).
                "edge_provenance": self._compute_edge_provenance(
                    dag, discovery_result, gate_evaluation, dag_overridden, augmented_edges
                ),
            }

            # Compute DAG version hash for expert review tracking
            dag_version_hash = compute_dag_hash(causal_graph=causal_graph.copy())  # type: ignore[arg-type]
            causal_graph["dag_version_hash"] = dag_version_hash

            # Latent-confounding diagnostic (FCI): surface the annotated
            # payload on causal_graph. Added AFTER the hash: compute_dag_hash
            # keys off nodes/edges/treatment/outcome only, and the payload
            # carries a nondeterministic runtime that must never perturb
            # hashing. This node only annotates and logs — the human-readable
            # warning is raised by InterpretationNode, which can corroborate
            # the flag against the E-value sensitivity result (surfacing
            # policy; see test_structural_recovery docstring item 6).
            new_warnings: List[str] = list(panel_warnings)
            if search_warning is not None:
                new_warnings.append(search_warning)
            if discovery_result is not None:
                new_warnings.extend(
                    self._discovery_honesty_warnings(discovery_result, treatment, outcome)
                )
                latent_diagnostic = discovery_result.metadata.get("latent_diagnostic")
                if isinstance(latent_diagnostic, dict):
                    causal_graph["latent_diagnostic"] = latent_diagnostic
                    # Base-rate observability: the agent-analyze job store TTL
                    # is 8h, so this line (and the MLflow metric) is the
                    # durable record of how often the flag fires live.
                    logger.info(
                        "latent_diagnostic ran=%s converged=%s flag=%s treatment=%s outcome=%s",
                        latent_diagnostic.get("ran"),
                        latent_diagnostic.get("converged"),
                        latent_diagnostic.get("flag"),
                        treatment,
                        outcome,
                    )

            # #1974: durable record of the discovery run (public.discovered_dags).
            # Persist ONLY when discovery actually ran: the discovered_dags row
            # IS a discovery run (n_samples, algorithms_used, ensemble_threshold
            # and alpha are NOT NULL) — a manual DAG has none of those and
            # writing it would fabricate them. The DAG that shipped after the
            # gate (manual fallback included) rides in metadata.shipped_dag with
            # its per-edge provenance, so every discovery run is recorded
            # whatever the gate decided. NEVER silently best-effort: the step
            # returns a state delta — {discovered_dag_id} on success, or
            # {discovered_dag_persist_error, warnings} on failure — and never
            # raises into this node's outer ``except``.
            persist_delta: Dict[str, Any] = {}
            if discovery_result is not None and gate_evaluation is not None:
                persist_delta = await _persist_discovered_dag(
                    state=state,
                    discovery_result=discovery_result,
                    gate_evaluation=gate_evaluation,
                    causal_graph=causal_graph,
                    treatment=treatment,
                    outcome=outcome,
                    discovery_latency_ms=discovery_latency_ms,
                )
                new_warnings.extend(persist_delta.pop("warnings", []))

            latency_ms = (time.time() - start_time) * 1000

            result = {
                **spread_safe(state),
                "causal_graph": causal_graph,
                "dag_version_hash": dag_version_hash,
                "graph_builder_latency_ms": latency_ms,
                "current_phase": "estimating",
                **persist_delta,
            }

            # Add discovery metadata if used
            if auto_discover:
                result["discovery_latency_ms"] = discovery_latency_ms
                if discovery_result:
                    result["discovery_result"] = discovery_result.to_dict()
                if gate_evaluation:
                    result["discovery_gate_evaluation"] = gate_evaluation
                if discovery_skip_reason is not None:
                    result["discovery_skip_reason"] = discovery_skip_reason
                    new_warnings.append(discovery_skip_reason)
            if new_warnings:
                # warnings is an operator.add accumulator (state.py);
                # return ONLY the new entries so LangGraph appends them.
                result["warnings"] = new_warnings

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Graph building failed: {e}")
            return {
                **spread_safe(state),
                "graph_builder_error": str(e),
                "graph_builder_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Graph building failed: {e}",
            }

    def _find_adjustment_sets(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[List[str]]:
        """Find valid backdoor adjustment sets (unbounded legacy contract).

        The search itself lives in ``adjustment_search.find_adjustment_sets``
        (#2233): ``execute`` calls it OFF the event loop under a wall-time budget
        and a candidate cap and surfaces a hit bound in ``warnings``. This
        wrapper keeps the direct callers' list contract, unbounded.
        """
        return find_adjustment_sets(
            dag,
            treatment,
            outcome,
            time_budget_s=None,
            max_candidates=None,
            criterion=self._satisfies_backdoor_criterion,
        ).adjustment_sets

    def _satisfies_backdoor_criterion(
        self, dag: nx.DiGraph, adjustment_set: Set[str], treatment: str, outcome: str
    ) -> bool:
        """Check whether ``adjustment_set`` satisfies the backdoor criterion.

        Pearl backdoor criterion for (treatment, outcome):
          1. No node in the adjustment set is a descendant of treatment.
          2. The adjustment set d-separates treatment from outcome in the
             proper backdoor graph (treatment's OUTGOING edges removed).

        This correctly EXCLUDES colliders (and their descendants); conditioning
        on a collider would open a non-causal path (M-bias).

        Args:
            dag: Causal DAG
            adjustment_set: Candidate set of nodes to adjust for
            treatment: Treatment node
            outcome: Outcome node

        Returns:
            True iff the set is a valid backdoor adjustment set.
        """
        # Lane B (2026-09-22): the criterion lives in the light shared module
        # ``src.ml.causal_role_dgp.backdoor`` so the structural assembler applies
        # the SAME admissibility test without importing this agent package.
        return satisfies_backdoor_criterion(dag, adjustment_set, treatment, outcome)

    def _apply_adjustment_guarantee(
        self,
        dag: nx.DiGraph,
        treatment: str,
        outcome: str,
        state: CausalImpactState,
        adjustment_sets: List[List[str]],
    ) -> List[List[str]]:
        """Union the declared (guarantee-channel) confounders into every
        adjustment set (fix 4).

        ``modeled_confounders`` is the caller's promise that these covariates
        must be adjusted for. Under the old wiring that promise was enforced by
        FORCING conf->treatment/conf->outcome DAG edges for all of them, which
        made the shipped DAG prior-determined (identical for real data and pure
        noise). The guarantee now lives here instead: whatever backdoor set the
        shipped DAG licenses is unioned with the declared covariates, so a
        structural miss by discovery can never silently unadjust the estimate —
        measured on the recovery benchmark, tiers-only discovery drops a true
        confounder from the DAG in 7/20 runs, and this union is what makes
        those misses harmless.

        Skips declared names that are not nodes of the shipped DAG (not frame
        columns — the estimator could not resolve them) and names the DAG shows
        as descendants of treatment (not backdoor variables; conditioning on a
        post-treatment variable would bias the estimate — unreachable under
        guided tiers, guards non-tiered DAG shapes). With nothing declared the
        sets pass through untouched, preserving the validated-empty ``[[]]``
        backdoor semantics for randomized designs.

        DELIBERATE LIMIT (codex iter-1 finding 1, rebutted by design): the
        merged set is NOT re-checked against the backdoor criterion, because
        declaration wins over DAG evidence by design — dropping a declared
        covariate when the (possibly wrong) DAG disagrees is exactly the
        measured-and-rejected 7/20-confounded failure mode. The residual risk
        is a declared pre-treatment COLLIDER whose parents are latent or
        undeclared (M-bias): under the agent API's declare-everything wiring a
        collider's in-frame parents are co-declared and block the opened path,
        and the OLD wiring adjusted for every declared covariate with no guard
        at all, so this is not a regression. The sanctioned mechanism for
        excluding a declared collider/mediator is the trust-gated
        ``adjustment_set_policy`` node (STRICT mode; M-bias detection tracked
        at #359), which runs downstream of this union."""
        declared = [
            str(c) for c in (state.get("modeled_confounders") or state.get("confounders") or [])
        ]
        if not declared:
            return adjustment_sets
        if treatment not in dag or outcome not in dag or treatment == outcome:
            return adjustment_sets
        nodes = set(dag.nodes())
        treatment_descendants = nx.descendants(dag, treatment)
        guaranteed = {
            c
            for c in declared
            if c in nodes and c not in (treatment, outcome) and c not in treatment_descendants
        }
        if not guaranteed:
            return adjustment_sets
        unioned: List[List[str]] = []
        seen = set()
        for adj_set in adjustment_sets or [[]]:
            merged = sorted(set(adj_set) | guaranteed)
            key = tuple(merged)
            if key not in seen:
                seen.add(key)
                unioned.append(merged)
        return unioned


async def _build_discovered_dag_repository() -> Any:
    """Service-role repository for ``public.discovered_dags`` (#1974).

    Mirrors ``refutation._build_expert_review_gate``: ``get_async_supabase_client``
    raises ``ServiceConnectionError`` when no Supabase is configured (dev/test),
    which the caller classifies as "unavailable" (WARNING) as opposed to any
    other failure (ERROR). Resolved through the module global at call time so
    tests substitute a fake factory the way the refutation tests do.
    """
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.discovered_dag import DiscoveredDagRepository

    client = await get_async_supabase_client()
    return DiscoveredDagRepository(supabase_client=client)


async def _persist_discovered_dag(
    *,
    state: CausalImpactState,
    discovery_result: DiscoveryResult,
    gate_evaluation: Dict[str, Any],
    causal_graph: CausalGraph,
    treatment: Optional[str],
    outcome: Optional[str],
    discovery_latency_ms: Optional[float],
) -> Dict[str, Any]:
    """Persist one discovery run; return a STATE DELTA, never raise (#1974).

    Success  -> ``{"discovered_dag_id": <uuid>}``.
    Failure  -> ``{"discovered_dag_persist_error": <msg>, "warnings": [<msg>]}``
    with the dag hash, session id and query id in the message, logged at
    WARNING when Supabase is simply not configured (``ServiceConnectionError``,
    the refutation precedent's degrade branch) and at ERROR for anything else.

    Deliberate departure from ``_build_expert_review_gate``, which re-raises
    unexpected errors: a self-bypassed review gate would mislabel a REVIEW-band
    estimate as approved, so it must fail loud. A lost audit row changes no
    result, so failing the whole analysis (minutes of compute, ``status=failed``)
    over it would be the wrong trade — but it must never be SILENT either,
    which is what the state key + warning + ERROR log guarantee. Under the
    unit tree's dead-Supabase pin the transport raises ``httpx.ConnectError``
    (measured), not ``ServiceConnectionError``, so the broad except is what
    keeps a persistence hiccup out of the node's outer failure path.
    """
    from src.memory.services.factories import ServiceConnectionError
    from src.repositories.discovered_dag import (
        build_discovered_dag_payload,
        resolve_frame_provenance,
    )

    dag_version_hash = causal_graph.get("dag_version_hash")
    session_id = state.get("session_id")
    query_id = state.get("query_id")
    context = f"[dag_version_hash={dag_version_hash} session_id={session_id} query_id={query_id}]"

    try:
        # Inside the boundary on purpose (codex iter-1 LOW): a cache value
        # pandas cannot convert must become a persist error, never a raise.
        frame = (state.get("data_cache") or {}).get("estimation_data")
        if frame is not None and not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        payload = build_discovered_dag_payload(
            discovery_result=discovery_result,
            gate_evaluation=gate_evaluation,
            causal_graph=causal_graph,
            treatment=treatment,
            outcome=outcome,
            query_id=query_id,
            session_id=session_id,
            is_synthetic=resolve_frame_provenance(frame, state),
            frame=frame,
            discovery_latency_ms=discovery_latency_ms,
        )
        repository = await _build_discovered_dag_repository()
        dag_id = await repository.record(payload)
    except ServiceConnectionError as exc:
        message = f"Discovered DAG NOT persisted (Supabase unavailable): {exc} {context}"
        logger.warning(message)
        return {"discovered_dag_persist_error": message, "warnings": [message]}
    except Exception as exc:
        message = f"Discovered DAG persistence FAILED: {exc} {context}"
        logger.error(message, exc_info=True)
        return {"discovered_dag_persist_error": message, "warnings": [message]}

    logger.info(
        "Discovered DAG persisted: dag_id=%s gate=%s n_edges=%s is_synthetic=%s %s",
        dag_id,
        payload.get("gate_decision"),
        payload.get("n_edges"),
        payload.get("is_synthetic"),
        context,
    )
    return {"discovered_dag_id": dag_id}


# Standalone function for LangGraph integration
async def build_causal_graph(state: CausalImpactState) -> Dict:
    """Build causal DAG (standalone function).

    Args:
        state: Current workflow state

    Returns:
        Updated state with causal_graph
    """
    node = GraphBuilderNode()
    return await node.execute(state)
