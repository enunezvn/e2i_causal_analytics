"""Unit tests for the FalkorDB causal-paths sync edge derivation.

The Knowledge-Graph page rendered the variable layer as an unreadable hairball
because the sync turned a chain ``[treatment_arm, *mediators, outcome]`` into
CONSECUTIVE-pair edges. With mediators chosen in random order per row, that
produced spurious ``mediator -> mediator`` edges in BOTH directions (reciprocal
cycles) — e.g. ``adherence -> prior_therapy`` and ``prior_therapy -> adherence``.

The mediators are *parallel* mediators, not a serial chain: the correct graph is
``treatment_arm -> mediator`` and ``mediator -> outcome`` for each mediator, with
NO ``mediator -> mediator`` edge. ``_mediation_edges`` encodes that, and the
bridge map connects terminal outcomes into the KPI layer so the graph is one
connected component.
"""

from scripts.sync_causal_paths_to_falkordb import (  # type: ignore[import-not-found]
    _VARIABLE_KPI_BRIDGE,
    _mediation_edges,
    _variable_roles,
)


def _edge_pairs(edges):
    """(cause, effect) pairs, dropping the is_terminal flag."""
    return {(c, e) for c, e, _ in edges}


class TestMediationEdges:
    def test_emits_treatment_to_each_mediator_and_each_mediator_to_outcome(self):
        edges = _mediation_edges("treatment_arm", ["adherence", "prior_therapy"], "persistent_180d")
        assert _edge_pairs(edges) == {
            ("treatment_arm", "adherence"),
            ("treatment_arm", "prior_therapy"),
            ("adherence", "persistent_180d"),
            ("prior_therapy", "persistent_180d"),
        }

    def test_never_emits_mediator_to_mediator_edges(self):
        # The de-cycle property: with several mediators, no edge connects two of
        # them in EITHER direction (this is exactly the reciprocal-cycle bug).
        mediators = ["adherence", "engagement_score", "prior_therapy", "disease_severity"]
        edges = _mediation_edges("treatment_arm", mediators, "discontinued_180d")
        med = set(mediators)
        assert not any(c in med and e in med for c, e, _ in edges)

    def test_mediator_to_outcome_edges_are_terminal(self):
        edges = _mediation_edges("treatment_arm", ["adherence"], "persistent_180d")
        terminal = {(c, e) for c, e, is_terminal in edges if is_terminal}
        non_terminal = {(c, e) for c, e, is_terminal in edges if not is_terminal}
        assert terminal == {("adherence", "persistent_180d")}
        assert non_terminal == {("treatment_arm", "adherence")}

    def test_no_mediators_yields_single_direct_terminal_edge(self):
        edges = _mediation_edges("treatment_arm", [], "treatment_initiated")
        assert edges == [("treatment_arm", "treatment_initiated", True)]


class TestVariableRoles:
    """Role stamping: the AI-Insights graph colors nodes by their position
    across ALL validated chains. Pure sources (never an effect) are drivers,
    pure sinks (never a cause) are outcomes, everything else mediates. This is
    derived from the causal_paths SSOT topology — no hand-curated ontology."""

    def test_source_mediator_sink_roles_from_one_chain(self):
        edges = _mediation_edges("treatment_arm", ["adherence"], "persistent_180d")
        assert _variable_roles(edges) == {
            "treatment_arm": "driver",
            "adherence": "mediator",
            "persistent_180d": "outcome",
        }

    def test_variable_that_causes_in_one_chain_and_is_caused_in_another_is_mediator(self):
        # nrx_volume is an OUTCOME of chain 1 but the CAUSE in chain 2 —
        # across the whole SSOT it transmits effects, so it must be a mediator.
        edges = _mediation_edges("rep_detailing_frequency", [], "nrx_volume") + _mediation_edges(
            "nrx_volume", [], "trx_volume"
        )
        roles = _variable_roles(edges)
        assert roles["rep_detailing_frequency"] == "driver"
        assert roles["nrx_volume"] == "mediator"
        assert roles["trx_volume"] == "outcome"

    def test_empty_edge_list_yields_no_roles(self):
        assert _variable_roles([]) == {}


class TestVariableKpiBridge:
    def test_bridges_each_terminal_outcome_into_the_kpi_layer(self):
        # Terminal patient-journey outcomes connect to commercial KPIs (matched
        # by KPI *name*; live KPI nodes carry no id). This makes the variable
        # layer and the KPI layer one connected graph.
        assert _VARIABLE_KPI_BRIDGE == {
            "treatment_initiated": "NRx",
            "persistent_180d": "Patient_Retention",
            "discontinued_180d": "Patient_Retention",
            # Commercial grain (2026-07-07): only KPI nodes verified live in
            # the graph (MATCH (k:KPI)) get a bridge — no NBRx/ROI nodes
            # exist, so those outcomes stay unbridged rather than guessed.
            "trx_volume": "TRx",
            "nrx_volume": "NRx",
            "trx_market_share": "Market_Share",
        }
