"""Discovery orchestration for the graph builder node.

Split out of ``graph_builder.py`` by concern (Lane B item 5, PR #2230
follow-up). Every body here is moved verbatim from ``GraphBuilderNode``: the
guided / ensemble discovery run and its gate evaluation, the annotations the
run carries (latent diagnostic, required-edge honesty), and the gate-decision
DAG assembly (ACCEPT / AUGMENT / REVIEW / REJECT). ``GraphBuilderNode``
(``graph_builder.py``, the node entry) mixes it back in.

The mixin extends ``ManualDagMixin`` because the gate-decision assembly falls
back to the manual DAG on every path but a clean ACCEPT.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from src.agents.causal_impact.nodes.manual_dag import ManualDagMixin
from src.agents.causal_impact.state import CausalImpactState
from src.causal_engine.discovery import (
    DEFAULT_DISCOVERY_ALGORITHM_NAMES,
    DEFAULT_DISCOVERY_ALGORITHMS,
    CausalPriorKnowledge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryGate,
    DiscoveryGateDecision,
    DiscoveryResult,
    DiscoveryRunner,
)
from src.causal_engine.discovery.preflight import preflight_discovery_frame
from src.utils.session_ids import coerce_session_uuid

logger = logging.getLogger(__name__)

# Bootstrap resamples for guided (single-algorithm) discovery. The gate can
# only corroborate a single-algorithm run through resample stability; 0 would
# make every guided run gate-uncorroborated (auto-REJECT). Override per run
# with state key ``discovery_bootstrap_resamples``.
DISCOVERY_BOOTSTRAP_RESAMPLES = 20

# Lane D (real-data causal estimation, spec "Lane D — guided discovery on
# claims frames"). Measured on the real Optum persistence frame
# (docs/demos/results/2026-09-22_lane_d_guided_discovery_claims/): 77 resolved
# covariates, correlation rank 63/79, one guided PC fit at 20 covariates 9.5 s
# versus 230 s at 43. The DAG-learning frame is capped here; every capped or
# pruned covariate stays in the adjustment guarantee. Override per run with
# ``discovery_max_covariates``.
DISCOVERY_MAX_COVARIATES = 20
# Wall-clock budget for one guided discovery run (primary fit + bootstrap
# resamples + latent diagnostic). Derived, not invented: the agent's hard cap
# (``_AGENT_HARD_TIMEOUT_S`` = 900 s) minus the refutation node's cooperative
# deadline (``_REFUTATION_COMPUTE_BUDGET_S`` = 720 s from task start), both in
# ``src/api/routes/causal/_common.py`` — discovery runs before estimation and
# refutation, so it may use only the headroom the graph has beyond the
# refutation deadline. Pinned against those constants by
# tests/unit/test_agents/test_causal_impact/test_graph_builder_preflight.py.
# Override per run with ``discovery_time_budget_s`` (None = unbounded).
DISCOVERY_TIME_BUDGET_S = 180.0
# Fewer succeeded resamples than this leaves a guided run uncorroborated
# (the gate cannot ACCEPT or AUGMENT on it). Override with
# ``discovery_min_resamples``.
DISCOVERY_MIN_RESAMPLES = 10


class DiscoveryOrchestrationMixin(ManualDagMixin):
    """Structure learning under the run's priors and budget, the gate's
    verdict on it, and the DAG that verdict ships."""

    def __init__(self):
        """Initialize graph builder with discovery components."""
        self._discovery_runner: Optional[DiscoveryRunner] = None
        self._discovery_gate: Optional[DiscoveryGate] = None

    @property
    def discovery_runner(self) -> DiscoveryRunner:
        """Lazy-initialize discovery runner."""
        if self._discovery_runner is None:
            self._discovery_runner = DiscoveryRunner()
        return self._discovery_runner

    @property
    def discovery_gate(self) -> DiscoveryGate:
        """Lazy-initialize discovery gate."""
        if self._discovery_gate is None:
            self._discovery_gate = DiscoveryGate()
        return self._discovery_gate

    @staticmethod
    def _resolve_anchored_confounders(state: CausalImpactState) -> List[str]:
        """Structural-prior channel (fix 4). When ``anchored_confounders`` is
        PRESENT it alone names the confounders to force as required edges (an
        empty list = no structural priors). When ABSENT, legacy behavior anchors
        the modeled (guarantee-channel) confounders so pre-split callers keep
        their exact prior shape."""
        if "anchored_confounders" in state:
            return [str(c) for c in (state.get("anchored_confounders") or [])]
        return [str(c) for c in (state.get("modeled_confounders") or [])]

    async def _run_discovery(
        self,
        state: CausalImpactState,
        treatment: str,
        outcome: str,
    ) -> Tuple[DiscoveryResult, Dict[str, Any]]:
        """Run structure learning and gate evaluation.

        Args:
            state: Current workflow state
            treatment: Treatment variable name
            outcome: Outcome variable name

        Returns:
            Tuple of (DiscoveryResult, gate evaluation dict)
        """
        # Get data from state. Canonical key is "estimation_data" — the only key
        # ever written to data_cache (agent.py writes
        # {"estimation_data": input_data["data"]}; estimation.py reads it).
        # The previous read of "data" was always None for real callers (M-gb1),
        # silently disabling auto-discovery.
        data_cache = state.get("data_cache", {})
        data = data_cache.get("estimation_data")

        if data is None:
            # No real data passthrough in the cache -> discovery cannot run.
            # Raise a distinct, descriptive error; the caller (execute) records
            # this as a surfaced skip signal rather than swallowing it silently.
            logger.warning("No estimation_data in data_cache; skipping discovery")
            raise ValueError("No estimation_data in data_cache for discovery")

        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # GUIDED discovery: when the treatment and outcome are known (the causal
        # question being asked), anchor their roles so observational structure
        # learning does not emit implausibly-oriented edges. Unconstrained PC/GES
        # only recover a Markov equivalence class — on patient_journeys that
        # reversed the confounder edges (treatment->disease_severity) and
        # flipped the treatment->outcome edge. Tiers [covariates < treatment <
        # outcome] + a required treatment->outcome edge let the DATA decide WHICH
        # covariates are confounders while keeping orientation correct.
        # Guided discovery (priors-constrained PC) is OPT-IN: the agent endpoints
        # set discovery_guided=True. Other consumers of this node keep the legacy
        # multi-algorithm ensemble default (False) so their behavior is unchanged.
        guided = bool(state.get("discovery_guided", False))
        prior_knowledge: Optional[CausalPriorKnowledge] = None
        learning_frame = data
        preflight = None
        time_budget_s: Optional[float] = None
        min_resamples: Optional[int] = None
        if guided and treatment in data.columns and outcome in data.columns:
            from src.repositories.provenance import PROVENANCE_DROP_COLS

            covariate_cols = [
                c
                for c in data.columns
                if c not in (treatment, outcome) and c not in PROVENANCE_DROP_COLS
            ]
            # Seed the ANCHORED confounders (fix 4: the structural-prior channel,
            # falling back to modeled_confounders for pre-split callers) as
            # REQUIRED edges so guided PC anchors them as confounders
            # (confounder->treatment AND confounder->outcome) by construction,
            # while the data still selects the rest. An empty anchored channel —
            # the agent API's production shape — leaves only the estimand edge
            # required, so the shipped DAG is data-responsive; the declared
            # covariates stay adjusted through the guarantee channel instead
            # (see _apply_adjustment_guarantee). Restricted to confounders
            # actually present as frame columns (build_background_knowledge
            # name-matches the frame; keeping required_edges clean keeps the
            # prior honest).
            anchored = [
                c
                for c in self._resolve_anchored_confounders(state)
                if c in data.columns and c not in (treatment, outcome)
            ]
            # Lane D item 1: the pre-flight decides which covariates the
            # structure learner sees — constant and exactly collinear columns
            # go (a real claims frame's correlation matrix is singular
            # otherwise and fisherz refuses it), then the frame is capped by a
            # pre-treatment screen (PC's cost is the number of CI tests). The
            # anchored confounders are protected: their required edges need
            # the node in the frame. Everything removed stays in the
            # adjustment guarantee (execute() adds it back to the shipped DAG
            # as a node, and _apply_adjustment_guarantee unions it). The
            # decisions travel in DiscoveryResult.metadata["preflight"].
            if covariate_cols:
                preflight = preflight_discovery_frame(
                    data,
                    treatment,
                    outcome,
                    covariate_cols,
                    max_covariates=int(
                        state.get("discovery_max_covariates", DISCOVERY_MAX_COVARIATES)
                    ),
                    protected=anchored,
                )
                covariate_cols = list(preflight.kept)
                anchored = [c for c in anchored if c in set(preflight.kept)]
                learning_columns = set(covariate_cols) | {treatment, outcome}
                learning_frame = data[[c for c in data.columns if c in learning_columns]]
            tiers = (
                [covariate_cols, [treatment], [outcome]]
                if covariate_cols
                else [[treatment], [outcome]]
            )
            required_edges: List[Tuple[str, str]] = [(treatment, outcome)]
            for conf in anchored:
                required_edges.append((conf, treatment))
                required_edges.append((conf, outcome))
            prior_knowledge = CausalPriorKnowledge(
                tiers=tiers,
                required_edges=required_edges,
            )
            # Only PC consumes BackgroundKnowledge; restrict to it so the ensemble
            # is not polluted by unconstrained orientations from other algorithms.
            algorithms = [DiscoveryAlgorithmType.PC]
            bootstrap_resamples = int(
                state.get("discovery_bootstrap_resamples", DISCOVERY_BOOTSTRAP_RESAMPLES)
            )
            # Guided runs are single-algorithm PC, which assumes causal
            # sufficiency — default the FCI latent-confounding diagnostic ON
            # here (opt-out via state), mirroring the bootstrap idiom above.
            latent_diagnostic = bool(state.get("discovery_latent_diagnostic", True))
            # Lane D item 2: the run is bounded and its corroboration needs a
            # minimum achieved resample count (the runner reports the count
            # it achieved, never the one requested).
            raw_budget = state.get("discovery_time_budget_s", DISCOVERY_TIME_BUDGET_S)
            time_budget_s = None if raw_budget is None else float(raw_budget)
            min_resamples = int(state.get("discovery_min_resamples", DISCOVERY_MIN_RESAMPLES))
        else:
            algorithms_str = state.get("discovery_algorithms", DEFAULT_DISCOVERY_ALGORITHM_NAMES)
            algorithms = []
            for algo in algorithms_str:
                try:
                    algorithms.append(DiscoveryAlgorithmType(algo.lower()))
                except ValueError:
                    logger.warning(f"Unknown algorithm: {algo}, skipping")
            if not algorithms:
                algorithms = list(DEFAULT_DISCOVERY_ALGORITHMS)
            # Multi-algorithm ensembles are corroborated by cross-algorithm
            # agreement already; bootstrap stability is off by default here to
            # avoid a 20x runtime surprise for existing (unguided) consumers.
            bootstrap_resamples = int(state.get("discovery_bootstrap_resamples", 0))
            # Same opt-in reasoning: other discover_dag consumers (ranker,
            # tool registry) should not pay an extra FCI run by default.
            latent_diagnostic = bool(state.get("discovery_latent_diagnostic", False))

        config = DiscoveryConfig(
            algorithms=algorithms,
            ensemble_threshold=state.get("discovery_ensemble_threshold", 0.5),
            alpha=state.get("discovery_alpha", 0.05),
            prior_knowledge=prior_knowledge,
            bootstrap_resamples=bootstrap_resamples,
            latent_diagnostic=latent_diagnostic,
            time_budget_s=time_budget_s,
            min_resamples=min_resamples,
            indep_test=state.get("discovery_indep_test"),
        )

        # Run discovery. The state's session is the caller's chat id RAW
        # (#2116): a composite ``{user}~{session}`` on the plain routes, which
        # never parses as a uuid. A bare ``UUID(session_id)`` here raised a
        # ValueError inside this method; ``execute`` catches it (``except
        # Exception`` around this call), logs it as a warning and surfaces it
        # as ``discovery_skip_reason`` (also appended to the state's warnings),
        # so with ``auto_discover`` set every plain-route causal turn still
        # answered from the manual DAG -- visible to an operator reading the
        # log or the state, not to the user. The shared coercion recovers the
        # trailing session uuid, returns a bare uuid in canonical form and
        # yields None (an honest null) for a malformed id.
        session_uuid = coerce_session_uuid(state.get("session_id"))

        result = await self.discovery_runner.discover_dag(
            data=learning_frame,
            config=config,
            session_id=session_uuid,
        )

        # Annotate the latent diagnostic with the estimand BEFORE the gate
        # evaluates, so the gate's metadata pass-through carries the flag too.
        self._annotate_latent_diagnostic(result, treatment, outcome)
        if preflight is not None:
            result.metadata["preflight"] = preflight.to_dict()
        self._annotate_required_edges(result, config)

        # Evaluate with gate
        expected_edges = [(treatment, outcome)]  # Minimal expectation
        evaluation = self.discovery_gate.evaluate(result, expected_edges)

        logger.info(
            f"Discovery complete: {result.n_edges} edges, "
            f"gate decision: {evaluation.decision.value}"
        )

        return result, evaluation.to_dict()

    @staticmethod
    def _annotate_latent_diagnostic(
        result: DiscoveryResult,
        treatment: Optional[str],
        outcome: Optional[str],
    ) -> None:
        """Attach the estimand and the warning flag to the latent-diagnostic
        payload (the runner does not know treatment/outcome). Mutates the
        payload in ``result.metadata`` in place; no-op when the diagnostic did
        not run.

        Flag predicate — MEASURED, not assumed (test_structural_recovery.py
        docstring item 6): raised iff FCI marks the ESTIMAND PAIR itself
        bidirected. That mark is the one location with true-positive signal on
        this platform's binary-logit frames (10/10 detection on the null-effect
        latent DGP at n=2000; 0/10 false alarms on the observed-confounder and
        noise controls). Bidirected pairs involving covariates are false alarms
        on every measured control, so they stay in the payload as data but do
        not raise the flag.
        """
        payload = result.metadata.get("latent_diagnostic")
        if not isinstance(payload, dict):
            return
        pairs = payload.get("bidirected_edges") or []
        estimand = {treatment, outcome}
        flag = (
            bool(treatment)
            and bool(outcome)
            and any({str(source), str(target)} == estimand for source, target in pairs)
        )
        payload["treatment"] = treatment
        payload["outcome"] = outcome
        payload["flag"] = flag

    @staticmethod
    def _annotate_required_edges(result: DiscoveryResult, config: DiscoveryConfig) -> None:
        """Lane D item 3 — required-edge honesty. Record which prior-required
        edges the ensemble does NOT carry, with the cause.

        Measured (docs/demos/results/2026-09-22_lane_d_guided_discovery_claims/,
        ``d1_required_edge_mechanism.txt``): causal-learn's PC honours a
        required edge at ORIENTATION only — ``skeleton_discovery`` consults
        ``is_forbidden`` and never ``is_required`` — so a required pair the
        data finds conditionally independent is removed in the skeleton phase
        and never returns. On the real Optum persistence frame the estimand
        pair is marginally independent at alpha 0.05 (fisherz p = 0.21), so
        the (T, Y) edge is absent from every guided ensemble there. Nothing
        of ours drops it (the ensemble keeps every edge a single converged
        algorithm draws), so the fix is honesty: the miss is recorded here,
        surfaced as a warning by ``execute``, and the shipped DAG asserts the
        estimand edge with provenance ``required_prior`` on the discovery
        paths (ACCEPT appends it; AUGMENT's manual base carries it)."""
        prior = config.prior_knowledge
        required = list(prior.required_edges or []) if prior is not None else []
        if not required:
            return
        dag = result.ensemble_dag
        missing = [
            [source, target]
            for source, target in required
            if dag is None or not dag.has_edge(source, target)
        ]
        result.metadata["required_edges_missing"] = missing
        if not missing:
            return
        # The cause is ESTABLISHED from the run, not assumed (codex r1
        # finding 6): a run that did not converge names its failure; an edge
        # the algorithm DID draw but the ensemble no longer carries was
        # removed by post-processing (cycle removal); only an edge a converged
        # run never drew is the skeleton-phase removal.
        converged = [r for r in result.algorithm_results if r.converged]
        if not converged:
            errors = [
                str(r.metadata.get("error"))
                for r in result.algorithm_results
                if r.metadata.get("error")
            ]
            if not errors and result.metadata.get("error"):
                errors = [str(result.metadata["error"])]
            detail = "; ".join(errors) if errors else "no algorithm run converged"
            cause = f"discovery did not converge ({detail}), so no edge was learned"
        else:
            drawn = {(source, target) for r in converged for source, target in (r.edge_list or [])}
            if all(tuple(edge) in drawn for edge in missing):
                cause = (
                    "drawn by the algorithm but removed by the ensemble's post-processing "
                    "(cycle removal keeps the higher-confidence direction)"
                )
            elif any(tuple(edge) in drawn for edge in missing):
                cause = (
                    "part drawn by the algorithm but removed by the ensemble's "
                    "post-processing (cycle removal), part removed in PC's skeleton phase "
                    f"(conditionally independent at alpha={config.alpha})"
                )
            else:
                cause = (
                    "removed in PC's skeleton phase: the data found the pair conditionally "
                    f"independent at alpha={config.alpha} (causal-learn applies required "
                    "edges at orientation only, to edges that survived the skeleton)"
                )
        result.metadata["required_edges_missing_cause"] = cause

    def _build_dag_with_discovery(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str],
        discovery_result: DiscoveryResult,
        gate_evaluation: Dict[str, Any],
        anchored_confounders: Optional[List[str]] = None,
    ) -> Tuple[nx.DiGraph, List[Tuple[str, str]], bool]:
        """Build DAG based on discovery results and gate decision.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            confounders: Confounder variables (the manual/fallback DAG asserts
                ALL of these as common causes — the adjustment guarantee on the
                REVIEW/REJECT/AUGMENT paths)
            discovery_result: Result from discovery runner
            gate_evaluation: Result from discovery gate
            anchored_confounders: fix 4 — the structural-prior channel. On the
                ACCEPT path only these are re-asserted onto the discovered DAG
                (a no-op when the prior already forced their edges); the
                remaining declared covariates stay adjusted through
                ``_apply_adjustment_guarantee`` WITHOUT forcing DAG edges, which
                is what makes the accepted DAG data-responsive. ``None``
                preserves the legacy behavior (re-add every ``confounders``
                entry) for direct callers that predate the channel split.

        Returns:
            Tuple of ``(DAG, augmented_edges, dag_overridden)``. ``dag_overridden``
            is True ONLY when the gate ACCEPTED a discovered DAG but it
            contradicted a curated confounder, so the discovered DAG was discarded
            for the manual domain DAG — the caller surfaces this so the API's
            ``dag_source`` provenance is honest ('domain_knowledge', not
            'discovered'). False on every other path (clean ACCEPT keeps the
            discovered DAG; AUGMENT/REVIEW/REJECT are already labeled by the gate
            decision).
        """
        decision = gate_evaluation.get("decision")
        augmented_edges: List[Tuple[str, str]] = []

        if decision == DiscoveryGateDecision.ACCEPT.value:
            # Use discovered DAG directly
            logger.info("Using discovered DAG (ACCEPT)")
            if discovery_result.ensemble_dag is not None:
                dag = discovery_result.ensemble_dag.copy()
                # Ensure treatment and outcome are present
                if treatment not in dag.nodes():
                    dag.add_node(treatment)
                if outcome not in dag.nodes():
                    dag.add_node(outcome)
                # Discovery oriented the estimand edge BACKWARDS (outcome->treatment),
                # contradicting the anchored roles of the user's question. With outcome
                # a direct PARENT of treatment the two are ADJACENT, so the backdoor
                # criterion can NEVER d-separate them: _find_adjustment_sets returns an
                # empty set and the estimate is left CONFOUNDED — and no curated
                # confounder can rescue an adjacent treatment/outcome (live regression:
                # treatment_arm->adopted = naive 0.289, run fails on 0-feature
                # estimators). The estimand edge treatment->outcome also cannot be added
                # below (it would close a 2-cycle). This is the same class of failure as
                # an unplaced curated confounder, so take the same remedy: discard the
                # discovered DAG for the manual domain DAG, which orients
                # treatment->outcome with every curated confounder a clean common cause
                # (a proper, non-empty backdoor). Provenance is therefore manual.
                if treatment and outcome and dag.has_edge(outcome, treatment):
                    logger.warning(
                        "Discovery ACCEPT oriented %s->%s (outcome->treatment), "
                        "contradicting the anchored estimand; falling back to the "
                        "manual DAG to keep the effect adjustable.",
                        outcome,
                        treatment,
                    )
                    return (
                        self._construct_dag(treatment, outcome, confounders),
                        augmented_edges,
                        True,  # discovered DAG discarded -> provenance is manual
                    )
                # The treatment->outcome edge is the estimand under test (the
                # user's question). Constraint-based discovery may not DRAW it —
                # a conditional-independence test can miss a real effect on
                # binary data — but the agent DOES estimate that effect, so the
                # reported DAG must include the estimand edge for consistency
                # (an empty treatment->outcome path next to a non-zero ATE reads
                # as broken). Add it only if it preserves acyclicity.
                if treatment and outcome and not dag.has_edge(treatment, outcome):
                    dag.add_edge(treatment, outcome)
                    if not nx.is_directed_acyclic_graph(dag):
                        dag.remove_edge(treatment, outcome)
                # Re-assert the ANCHORED confounders on the discovered DAG (fix 4).
                # These are the structural-prior channel: the prior already forced
                # their edges, so this is normally a no-op that exists to catch a
                # contradictory orientation atomically (see below). The rest of the
                # declared covariates are deliberately NOT drawn here — forcing an
                # edge for every declared covariate is what made the shipped DAG
                # identical for real data and pure noise. Their adjustment is
                # guaranteed at adjustment-set assembly instead
                # (_apply_adjustment_guarantee), so a discovery miss still cannot
                # silently confound the estimate (PR #1084 removed the estimator's
                # all-other-columns rescue). Legacy callers (anchored is None)
                # keep the old behavior: every curated confounder is re-added.
                accept_curated = (
                    confounders
                    if anchored_confounders is None
                    else [c for c in anchored_confounders if c in dag]
                )
                unplaced = self._add_curated_confounder_edges(
                    dag, treatment, outcome, accept_curated
                )
                if unplaced:
                    # Discovery oriented a curated confounder contradictorily
                    # (treatment->conf or outcome->conf), so it cannot be a clean
                    # common cause on the discovered DAG — a half-edge confounder
                    # is NOT in the backdoor and would silently UNADJUST the
                    # estimate (the exact regression this guards against). Discard
                    # the discovered DAG for this estimand and use the manual DAG,
                    # which draws every curated confounder as a clean common cause.
                    logger.warning(
                        "Discovery ACCEPT contradicts curated confounder(s) %s; "
                        "falling back to manual DAG to keep them adjusted.",
                        unplaced,
                    )
                    return (
                        self._construct_dag(treatment, outcome, confounders),
                        augmented_edges,
                        True,  # discovered DAG discarded -> provenance is manual
                    )
                return dag, augmented_edges, False

        elif decision == DiscoveryGateDecision.AUGMENT.value:
            # Build manual DAG and augment with high-confidence discovered edges
            logger.info("Augmenting manual DAG with discovered edges (AUGMENT)")
            dag = self._construct_dag(treatment, outcome, confounders)

            # Get high-confidence edges from gate evaluation
            high_conf_edges = gate_evaluation.get("high_confidence_edges", [])
            for edge in high_conf_edges:
                source = edge.get("source")
                target = edge.get("target")
                if source and target:
                    # Only add if doesn't create cycle
                    if not dag.has_edge(source, target):
                        dag.add_edge(source, target)
                        if nx.is_directed_acyclic_graph(dag):
                            augmented_edges.append((source, target))
                            logger.debug(f"Augmented edge: {source} -> {target}")
                        else:
                            dag.remove_edge(source, target)
                            logger.debug(f"Skipped edge (cycle): {source} -> {target}")

            return dag, augmented_edges, False

        elif decision == DiscoveryGateDecision.REVIEW.value:
            # Use manual DAG but flag for review
            logger.info("Using manual DAG, flagged for review (REVIEW)")
            dag = self._construct_dag(treatment, outcome, confounders)
            return dag, augmented_edges, False

        # REJECT or unknown: use manual DAG
        logger.info("Using manual DAG (REJECT or fallback)")
        dag = self._construct_dag(treatment, outcome, confounders)
        return dag, augmented_edges, False
