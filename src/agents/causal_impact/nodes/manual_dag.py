"""Manual (domain-knowledge) DAG construction for the graph builder node.

Split out of ``graph_builder.py`` by concern (Lane B item 5, PR #2230
follow-up). Every body here is moved verbatim from ``GraphBuilderNode``;
``GraphBuilderNode`` (``graph_builder.py``, the node entry) mixes it back in,
so ``node._construct_dag(...)`` and friends resolve exactly as before.
"""

from typing import List, Tuple

import networkx as nx


class ManualDagMixin:
    """The curated/fallback DAG: treatment, outcome and the declared
    confounders drawn from domain knowledge, plus query-to-variable inference
    and the DOT rendering of whatever DAG shipped."""

    # Domain knowledge: Common causal structures in pharma commercial data
    KNOWN_CAUSAL_RELATIONSHIPS = {
        # HCP engagement → patient outcomes
        ("hcp_engagement_level", "patient_conversion_rate"),
        ("hcp_meeting_frequency", "prescription_volume"),
        ("hcp_sample_provision", "new_patient_starts"),
        # Marketing → HCP behavior
        ("marketing_spend", "hcp_engagement_level"),
        ("digital_campaign_reach", "hcp_meeting_acceptance"),
        ("conference_attendance", "hcp_awareness"),
        # Patient journey
        ("patient_awareness", "patient_conversion_rate"),
        ("prior_authorization_time", "treatment_adherence"),
        ("copay_support", "prescription_abandonment"),
        # Market dynamics
        ("competitor_activity", "market_share"),
        ("formulary_status", "prescription_volume"),
        ("geographic_region", "hcp_engagement_level"),  # Confounder
        ("therapeutic_area_expertise", "prescription_volume"),  # Confounder
    }

    def _infer_variables_from_query(self, query: str) -> Tuple[str, str]:
        """Infer treatment and outcome from query text.

        Args:
            query: Natural language query

        Returns:
            (treatment, outcome) tuple
        """
        query_lower = query.lower()

        # Treatment keywords
        treatment_keywords = {
            "hcp engagement": "hcp_engagement_level",
            "marketing": "marketing_spend",
            "sample": "hcp_sample_provision",
            "meeting": "hcp_meeting_frequency",
            "campaign": "digital_campaign_reach",
            "copay": "copay_support",
        }

        # Outcome keywords
        outcome_keywords = {
            "conversion": "patient_conversion_rate",
            "prescription": "prescription_volume",
            "nrx": "new_patient_starts",
            "trx": "total_prescriptions",
            "adherence": "treatment_adherence",
            "market share": "market_share",
        }

        treatment = None
        outcome = None

        for keyword, var in treatment_keywords.items():
            if keyword in query_lower:
                treatment = var
                break

        for keyword, var in outcome_keywords.items():
            if keyword in query_lower:
                outcome = var
                break

        # Defaults
        if not treatment:
            treatment = "hcp_engagement_level"
        if not outcome:
            outcome = "patient_conversion_rate"

        return treatment, outcome

    def _construct_dag(self, treatment: str, outcome: str, confounders: List[str]) -> nx.DiGraph:
        """Construct the curated/fallback causal DAG from the supplied variables.

        The ``confounders`` are the caller's curated adjustment covariates — they
        were selected precisely because domain knowledge treats them as common
        causes of BOTH treatment and outcome. So this manual DAG (used on the
        AUGMENT / REVIEW / REJECT discovery-gate paths) draws a
        ``confounder -> treatment`` AND ``confounder -> outcome`` edge for EACH of
        them. That makes the graph CONNECTED (no orphan covariate nodes) and
        yields a non-empty backdoor adjustment set that is CONSISTENT with the
        covariates the estimator actually conditions on.

        Previously these edges were gated on a hardcoded allowlist of
        generic-pharma confounder names (``hcp_specialty`` / ``geographic_region``
        …) with zero overlap with real dataset covariates (``disease_severity`` /
        ``egfr`` …). The result was a DAG with a single ``treatment -> outcome``
        edge and every covariate rendered as a disconnected node, plus an empty
        adjustment set — a graph that MISREPRESENTED the (correctly
        all-covariate-adjusted) estimate.

        Numeric note: when ``confounders`` equals the loaded covariate columns
        (the API frame is exactly treatment+outcome+covariates), the resulting
        non-empty backdoor adjustment set is the SAME column set the estimator
        used under the prior empty-adjustment-set fallback, so the ATE is
        unchanged. For a caller passing extra columns OR a confounder name that
        the old allowlist happened to match, the now-complete adjustment is a
        correctness improvement (the old code silently under-adjusted), not a
        regression toward wrong values.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            confounders: Confounding variables to adjust for

        Returns:
            NetworkX directed graph
        """
        dag = nx.DiGraph()

        # Add core nodes
        dag.add_node(treatment)
        dag.add_node(outcome)

        # Add confounders
        for conf in confounders:
            dag.add_node(conf)

        # Domain-known structural edges. No-op unless variable names match the
        # curated pharma-commercial schema; harmless for arbitrary datasets.
        for source, target in self.KNOWN_CAUSAL_RELATIONSHIPS:
            if source in dag.nodes() and target in dag.nodes():
                dag.add_edge(source, target)

        # The estimand edge: treatment -> outcome (the question under test).
        # Drawn UNCONDITIONALLY (Lane D item 3, codex r2 HIGH): it used to be
        # skipped whenever a domain-known path T -> ... -> Y already existed
        # (e.g. marketing_spend -> hcp_engagement_level ->
        # patient_conversion_rate), so every manual-DAG path shipped those
        # estimands without the edge the estimate tests. The estimate is the
        # TOTAL effect of T on Y, and a direct edge beside a mediated path can
        # never close a cycle (a cycle would need a Y -> ... -> T path, which
        # the existing T -> ... -> Y path already rules out in a DAG).
        dag.add_edge(treatment, outcome)

        # Every curated confounder is a common cause of BOTH treatment and
        # outcome — draw both edges (acyclicity-guarded) so the graph is
        # connected and the backdoor adjustment set is non-empty and honest.
        self._add_curated_confounder_edges(dag, treatment, outcome, confounders)

        return dag

    def _add_curated_confounder_edges(
        self, dag: nx.DiGraph, treatment: str, outcome: str, confounders: List[str]
    ) -> List[str]:
        """Draw ``confounder -> treatment`` AND ``confounder -> outcome`` for each
        curated confounder (acyclicity-guarded), mutating ``dag`` in place. Returns
        the confounders that could NOT be placed (see the atomicity note below).

        The ``confounders`` are the caller's curated adjustment covariates — domain
        knowledge (e.g. the API's ``causal_paths.confounders_controlled``) that each
        is a common cause of BOTH treatment and outcome. This is shared by the manual
        DAG (``_construct_dag``) and the discovery-gate ACCEPT path so a supplied
        confounder is NEVER silently dropped: an empty backdoor must mean 'no
        confounder was supplied' (a randomized / exogenous treatment -> correctly
        unadjusted), NOT 'discovery happened to omit a supplied one'. Constraint-based
        discovery on binary data routinely fails to recover these edges, so without
        this the ACCEPT branch returns an empty backdoor and leaves the estimate
        CONFOUNDED — which the estimator's all-other-columns fallback used to mask
        until that fallback was removed for validated-empty backdoors (PR #1084).

        Each confounder is placed ATOMICALLY. A half-edge (only one of the two
        common-cause edges — left when the discovered DAG already orients
        ``treatment -> conf`` or ``outcome -> conf`` so the other edge would be
        cyclic) is an instrument or a precision covariate, NOT a backdoor
        confounder: the backdoor criterion drops it and re-confounds the estimate
        silently. So if both edges cannot coexist acyclically, any edge added for
        that confounder is reverted and the confounder is returned as 'unplaced'
        for the caller to handle (the ACCEPT path falls back to the manual DAG). A
        fresh manual DAG has no contradictory orientation, so every confounder
        places cleanly and the returned list is empty."""
        unplaced: List[str] = []
        for conf in confounders:
            if conf in (treatment, outcome):
                continue
            if conf not in dag:
                dag.add_node(conf)
            added: List[Tuple[str, str]] = []
            for target in (treatment, outcome):
                if not dag.has_edge(conf, target):
                    dag.add_edge(conf, target)
                    if nx.is_directed_acyclic_graph(dag):
                        added.append((conf, target))
                    else:
                        dag.remove_edge(conf, target)
            # A valid common cause needs BOTH edges present. If a contradictory
            # discovered orientation blocked one, revert what we added (never
            # leave a half-edge) and report the confounder for the caller.
            if dag.has_edge(conf, treatment) and dag.has_edge(conf, outcome):
                continue
            for u, v in added:
                if dag.has_edge(u, v):
                    dag.remove_edge(u, v)
            unplaced.append(conf)
        return unplaced

    def _to_dot_format(self, dag: nx.DiGraph) -> str:
        """Convert DAG to DOT format for visualization.

        Args:
            dag: NetworkX DAG

        Returns:
            DOT format string
        """
        lines = ["digraph CausalDAG {", "  rankdir=LR;", "  node [shape=box];", ""]

        for node in dag.nodes():
            label = node.replace("_", " ").title()
            lines.append(f'  "{node}" [label="{label}"];')

        lines.append("")

        for source, target in dag.edges():
            lines.append(f'  "{source}" -> "{target}";')

        lines.append("}")

        return "\n".join(lines)
