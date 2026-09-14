"""#1991 debt 3: ``get_dag_changes`` is the diff behind the expert-review
structure timeline, so it must report the ADJUSTMENT-SET delta (a covariate
change is a version of the estimand, spec §7) and it must be DETERMINISTIC --
the lists are serialized into an API response, and a set-difference order would
make two identical diffs render differently between two calls.
"""

import pytest

from src.causal_engine.dag_hash import (
    adjustment_hash_from_snapshot,
    compute_adjustment_set_hash,
    effective_adjustment_hash,
    get_dag_changes,
)


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


@pytest.mark.unit
class TestAdjustmentHashFromSnapshot:
    """What a stored snapshot PROVES about the adjustment set (codex round 4).

    A version row's NULL ``adjustment_set_hash`` means "never computed", not
    "empty" -- but where the row kept a snapshot, the snapshot NAMES the set,
    and the hash derived from it is the same one the writer would have stored.
    """

    def test_a_dict_snapshot_derives_the_writers_hash(self):
        snapshot = {"nodes": ["T", "Y", "W"], "adjustment_sets": [["W"]]}
        assert adjustment_hash_from_snapshot(snapshot) == compute_adjustment_set_hash([["W"]])

    def test_order_inside_the_snapshot_does_not_change_the_hash(self):
        assert adjustment_hash_from_snapshot(
            {"adjustment_sets": [["B", "A"], ["C"]]}
        ) == adjustment_hash_from_snapshot({"adjustment_sets": [["C"], ["A", "B"]]})

    def test_absent_null_and_empty_all_mean_the_empty_set(self):
        empty = compute_adjustment_set_hash([])
        assert adjustment_hash_from_snapshot({}) == empty
        assert adjustment_hash_from_snapshot({"adjustment_sets": None}) == empty
        assert adjustment_hash_from_snapshot({"nodes": ["T"], "adjustment_sets": []}) == empty

    def test_a_null_snapshot_is_unknown_not_the_empty_set(self):
        """The row recorded no structure at all, so it asserts nothing about the
        covariates -- reading it as ``sha256("[]")`` would claim a fact."""
        assert adjustment_hash_from_snapshot(None) is None
        assert compute_adjustment_set_hash([]) is not None

    def test_a_non_dict_snapshot_is_unknown(self):
        for value in ("[]", [["W"]], 7, True):
            assert adjustment_hash_from_snapshot(value) is None

    def test_malformed_nested_data_is_unknown_and_never_raises(self):
        """Codex round 5, finding 2. Migration 141's ``ck_erv_snapshot_object``
        checks the OUTER JSON type only, so ``{"adjustment_sets": [null]}`` is a
        row this table can hold -- and the hash computation raised TypeError on
        it. That derivation now runs in the pending queue, OUTSIDE the 503-
        guarded database read, so one such row aborted the whole queue response.

        Malformed nested data proves nothing, which is exactly what an unknown
        covariate set is. It is never an exception, and never the canonical
        EMPTY set either -- that is a real fact this row does not establish.
        """
        unprovable = [
            {"adjustment_sets": [None]},
            {"adjustment_sets": "x"},
            {"adjustment_sets": 7},
            {"adjustment_sets": {"a": ["W"]}},
            {"adjustment_sets": [["A", 1]]},
            {"adjustment_sets": [["A"], None]},
            {"adjustment_sets": [["A"], "B"]},
            {"adjustment_sets": [[["A"]]]},
        ]
        for snapshot in unprovable:
            assert adjustment_hash_from_snapshot(snapshot) is None, snapshot

    def test_a_well_formed_snapshot_beside_the_malformed_ones_still_derives(self):
        """The positive control: the validation rejects the malformed shapes
        above WITHOUT turning every snapshot into "unknown"."""
        assert adjustment_hash_from_snapshot(
            {"adjustment_sets": [["B", "A"], ["C"]]}
        ) == compute_adjustment_set_hash([["B", "A"], ["C"]])
        assert adjustment_hash_from_snapshot(
            {"adjustment_sets": [[]]}
        ) == compute_adjustment_set_hash([[]])

    def test_the_writers_contract_is_unchanged(self):
        """``compute_adjustment_set_hash`` is the WRITER's function and keeps its
        strict contract -- it is given a real ``CausalGraph.adjustment_sets``,
        not a stored column, and must not start silently swallowing a caller's
        bug. Only the READER of stored data forgives."""
        with pytest.raises(TypeError):
            compute_adjustment_set_hash([None])  # type: ignore[list-item]


@pytest.mark.unit
class TestEffectiveAdjustmentHash:
    """The ONE precedence rule the gate and the API share (#1991 debt 3):
    the stored ``adjustment_set_hash`` wins when the column carries one, else
    whatever the snapshot beside it PROVES via ``adjustment_hash_from_snapshot``,
    else None when neither half knows anything."""

    def test_stored_hash_wins_even_when_the_snapshot_derives_a_different_hash(self):
        stored = compute_adjustment_set_hash([["W"]])
        snapshot = {"adjustment_sets": [["Z"]]}
        assert effective_adjustment_hash(stored, snapshot) == stored
        assert effective_adjustment_hash(stored, snapshot) != adjustment_hash_from_snapshot(
            snapshot
        )

    def test_null_stored_falls_back_to_the_snapshots_derived_hash(self):
        snapshot = {"adjustment_sets": [["W"], ["Z"]]}
        assert effective_adjustment_hash(None, snapshot) == compute_adjustment_set_hash(
            [["W"], ["Z"]]
        )

    def test_null_stored_and_null_snapshot_is_unknown(self):
        assert effective_adjustment_hash(None, None) is None

    def test_null_stored_and_malformed_snapshot_is_unknown(self):
        """A malformed snapshot proves nothing -- it is never mistaken for the
        canonical empty set (codex round 5, finding 2)."""
        assert effective_adjustment_hash(None, {"adjustment_sets": [None]}) is None
