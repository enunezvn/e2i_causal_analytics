"""Discovery reporting for the graph builder node.

Split out of ``graph_builder.py`` by concern (Lane B item 5, PR #2230
follow-up). Every body here is moved verbatim from ``GraphBuilderNode``: what
``execute`` says about a discovery run once it has run — the pre-flight's
removals, the honesty lines for the warnings channel, the per-edge provenance
labels and the latent-confounding warning text (also read by
``InterpretationNode``). ``GraphBuilderNode`` (``graph_builder.py``, the node
entry) mixes it back in.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.causal_engine.discovery import DiscoveryGateDecision, DiscoveryResult
from src.causal_engine.discovery.preflight import preflight_summary


class DiscoveryReportingMixin:
    """Facts about a discovery run for the response: warnings and provenance."""

    @staticmethod
    def _preflight_removed(discovery_result: Optional[DiscoveryResult]) -> List[str]:
        """Covariates the pre-flight kept away from the learner (constant,
        exactly collinear, capped), in the order it reported them."""
        if discovery_result is None:
            return []
        payload = discovery_result.metadata.get("preflight")
        if not isinstance(payload, dict):
            return []
        removed: List[str] = []
        for key in ("constant", "collinear", "capped"):
            removed.extend(str(c) for c in (payload.get(key) or []))
        return removed

    @staticmethod
    def _discovery_honesty_warnings(
        discovery_result: DiscoveryResult, treatment: str, outcome: str
    ) -> List[str]:
        """The two Lane D lines for the response's warnings channel: what the
        pre-flight pruned and capped (by name), and whether the data drew the
        estimand edge. Both are facts about the run, whatever the gate
        decided, so they are raised on every path discovery ran on."""
        lines: List[str] = []
        payload = discovery_result.metadata.get("preflight")
        if isinstance(payload, dict):
            from src.causal_engine.discovery.preflight import PreflightResult

            preflight = PreflightResult(
                kept=list(payload.get("kept") or []),
                constant=list(payload.get("constant") or []),
                collinear=list(payload.get("collinear") or []),
                capped=list(payload.get("capped") or []),
                protected=list(payload.get("protected") or []),
                max_covariates=int(payload.get("max_covariates") or 0),
                n_offered=int(payload.get("n_offered") or 0),
                n_rows=int(payload.get("n_rows") or 0),
                n_rows_used=int(payload.get("n_rows_used") or 0),
            )
            # Raised only when the learner did not see everything: a line
            # saying "nothing dropped, nothing capped" is noise in the
            # response's prose channel.
            if preflight.constant or preflight.collinear or preflight.capped:
                lines.append(preflight_summary(preflight))
        # The API response carries no gate decision and no corroborated
        # flag (AgentCausalAnalysisResponse), so a bootstrap that fell short
        # of min_resamples is said here, in the only channel the consumer
        # reads (verifier MED-2 on this lane).
        bootstrap = discovery_result.metadata.get("bootstrap")
        if isinstance(bootstrap, dict) and bootstrap.get("corroborated") is False:
            budget = bootstrap.get("time_budget_s")
            budget_note = (
                f", the {float(budget):.0f} s discovery time budget was exhausted"
                if bootstrap.get("budget_exhausted") and budget is not None
                else ""
            )
            lines.append(
                f"Discovery bootstrap achieved {bootstrap.get('n_succeeded')} of "
                f"{bootstrap.get('n_resamples')} resamples (minimum "
                f"{bootstrap.get('min_resamples')}{budget_note}): the discovered "
                "structure is uncorroborated and the gate scored it as a single "
                "unverified run; the shipped DAG is the curated construction."
            )
        missing = discovery_result.metadata.get("required_edges_missing") or []
        if any(list(edge) == [treatment, outcome] for edge in missing):
            cause = discovery_result.metadata.get("required_edges_missing_cause") or ""
            lines.append(
                f"Discovery did not draw the estimand edge {treatment} -> {outcome} "
                f"({cause}). The edge is asserted by the prior on the shipped DAG "
                "(provenance required_prior when the DAG shipped through discovery); "
                "the estimate still tests it."
            )
        return lines

    @staticmethod
    def _compute_edge_provenance(
        dag: nx.DiGraph,
        discovery_result: Optional[DiscoveryResult],
        gate_evaluation: Optional[Dict[str, Any]],
        dag_overridden: bool,
        augmented_edges: List[Tuple[str, str]],
    ) -> List[Dict[str, str]]:
        """Label every shipped edge with WHY it is in the DAG (fix 4).

        'required_prior' and 'discovered' apply only when the shipped DAG came
        through discovery (a clean ACCEPT, or AUGMENT's manual-plus-extras);
        on every other path the DAG is the manual domain construction and each
        edge is honestly 'curated' — even where a prior happened to assert the
        same edge, the shipped graph did not come from it.

        'discovered' additionally requires the edge to be one the ensemble
        actually DREW (codex iter-1 HIGH): the ACCEPT path appends the estimand
        edge for consistency when discovery omitted it, and a legacy full
        re-add can draw curated confounder edges onto the discovered DAG —
        crediting either to the data would be the same overstatement the
        dag_source label used to make. Such appended edges are 'curated'
        (or 'required_prior' where a prior asserted them)."""
        prior_edges: Set[Tuple[str, str]] = set()
        if (
            discovery_result is not None
            and discovery_result.config is not None
            and discovery_result.config.prior_knowledge is not None
        ):
            prior_edges = {
                (source, target)
                for source, target in (discovery_result.config.prior_knowledge.required_edges or [])
            }
        ensemble_edges: Set[Tuple[str, str]] = set()
        if discovery_result is not None and discovery_result.ensemble_dag is not None:
            ensemble_edges = set(discovery_result.ensemble_dag.edges())
        decision = gate_evaluation.get("decision") if gate_evaluation else None
        shipped_via_discovery = not dag_overridden and decision in (
            DiscoveryGateDecision.ACCEPT.value,
            DiscoveryGateDecision.AUGMENT.value,
        )
        accepted = shipped_via_discovery and decision == DiscoveryGateDecision.ACCEPT.value
        augmented = set(augmented_edges)

        def _label(edge: Tuple[str, str]) -> str:
            if shipped_via_discovery and edge in prior_edges:
                return "required_prior"
            if (accepted and edge in ensemble_edges) or edge in augmented:
                return "discovered"
            return "curated"

        return [
            {"source": source, "target": target, "provenance": _label((source, target))}
            for source, target in dag.edges()
        ]

    @staticmethod
    def _latent_confounding_warning(treatment: Optional[str], outcome: Optional[str]) -> str:
        """Warning surfaced through the state's warnings accumulator into the
        analyze response. Worded to be TRUE in every measured world where the
        flag fires: FCI genuinely could not attribute the dependence to the
        treatment — whether because a latent confounder accounts for it (the
        detectable case) or because orientation was uncertain on a real effect
        (the unidentifiable case; both fire, see the benchmark docstring)."""
        return (
            f"Latent-confounding diagnostic (FCI): the data's dependence pattern marks "
            f"{treatment} <-> {outcome} as sharing an unmeasured common cause, and FCI "
            f"could not attribute this dependence to {treatment} itself. Unmeasured "
            f"confounding may account for part or all of the estimated effect."
        )
