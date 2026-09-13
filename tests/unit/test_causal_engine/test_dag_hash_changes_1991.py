"""#1991 debt 3: ``get_dag_changes`` is the diff behind the expert-review
structure timeline, so it must report the ADJUSTMENT-SET delta (a covariate
change is a version of the estimand, spec §7) and it must be DETERMINISTIC --
the lists are serialized into an API response, and a set-difference order would
make two identical diffs render differently between two calls.
"""

import pytest

from src.causal_engine.dag_hash import get_dag_changes


@pytest.mark.unit
class TestGetDagChangesAdjustmentSets:
    """The adjustment set is part of the structure a reviewer approves."""

    def test_get_dag_changes_reports_adjustment_set_delta(self):
        old = {"nodes": ["T", "Y"], "edges": [["T", "Y"]], "adjustment_sets": [["W"]]}
        new = {
            "nodes": ["T", "Y", "W"],
            "edges": [["T", "Y"], ["W", "T"]],
            "adjustment_sets": [["W", "Z"]],
        }
        d = get_dag_changes(old, new)
        assert d["nodes_added"] == ["W"] and d["edges_added"] == [["W", "T"]]
        assert d["adjustment_sets_added"] == [["W", "Z"]] and d["adjustment_sets_removed"] == [
            ["W"]
        ]
        assert d["is_changed"] is True

    def test_adjustment_set_only_change_is_a_change(self):
        """The graph is byte-identical; only the adjustment set moved. Before
        #1991 this reported is_changed False and the version diff was empty."""
        old = {"nodes": ["T", "Y"], "edges": [["T", "Y"]], "adjustment_sets": [["W"]]}
        new = {"nodes": ["T", "Y"], "edges": [["T", "Y"]], "adjustment_sets": [["W", "Z"]]}
        assert get_dag_changes(old, new)["is_changed"] is True

    def test_identical_graphs_report_no_change_and_sorted_empty_lists(self):
        """Node order and within-set covariate order are not structure."""
        g = {"nodes": ["Y", "T"], "edges": [["T", "Y"]], "adjustment_sets": [["Z", "W"]]}
        d = get_dag_changes(
            g,
            {"nodes": ["T", "Y"], "edges": [["T", "Y"]], "adjustment_sets": [["W", "Z"]]},
        )
        assert (
            d["is_changed"] is False and d["nodes_added"] == [] and d["adjustment_sets_added"] == []
        )


@pytest.mark.unit
class TestGetDagChangesDeterminism:
    """Set-difference order is not a stable API contract."""

    def test_lists_are_deterministically_sorted(self):
        old = {"nodes": [], "edges": [], "adjustment_sets": []}
        new = {
            "nodes": ["C", "A", "B"],
            "edges": [["C", "A"], ["A", "B"]],
            "adjustment_sets": [["Q"], ["P"]],
        }
        d = get_dag_changes(old, new)
        assert d["nodes_added"] == ["A", "B", "C"]
        assert d["edges_added"] == [["A", "B"], ["C", "A"]]
        assert d["adjustment_sets_added"] == [["P"], ["Q"]]

    def test_missing_keys_are_treated_as_empty(self):
        assert get_dag_changes({}, {})["is_changed"] is False

    def test_explicit_none_values_are_treated_as_empty(self):
        """A version row's ``dag_structure_json`` can carry an explicit NULL
        ``adjustment_sets`` (mig 141 backfill), which ``.get(k, [])`` returns as
        None -- iterating that is a TypeError, not an empty diff."""
        none_graph = {"nodes": None, "edges": None, "adjustment_sets": None}
        d = get_dag_changes(none_graph, none_graph)
        assert d["is_changed"] is False
        assert d["nodes_added"] == [] and d["edges_added"] == []
        assert d["adjustment_sets_added"] == [] and d["adjustment_sets_removed"] == []
