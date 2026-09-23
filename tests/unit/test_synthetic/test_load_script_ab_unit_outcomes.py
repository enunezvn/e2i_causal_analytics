"""Loader/scripts wiring for ab_experiment_unit_outcomes (option d1, Part of #2207).

The per-experiment unit outcome frame FK-references ab_experiment_assignments, so
it must be loaded AFTER assignments and purged BEFORE them; --refresh-ab must
carry it (is_synthetic-stamped) so the weekly reseed fills the feed.
"""

import importlib

from src.ml.synthetic.loaders.batch_loader import LOADING_ORDER, TABLE_COLUMNS

load_mod = importlib.import_module("scripts.load_synthetic_data")

TABLE = "ab_experiment_unit_outcomes"


class _Result:
    def __init__(self, count):
        self.count = count


class _DeleteQuery:
    def __init__(self, table, log):
        self._table, self._log = table, log

    def delete(self, *a, **k):
        self._log.append(("delete", self._table, k))
        return self

    def eq(self, *a, **k):
        self._log.append(("eq", self._table, a))
        return self

    def execute(self):
        self._log.append(("execute", self._table))
        return _Result(0)


class _FakeLoader:
    def __init__(self):
        self.log = []
        self.client = self

    def table(self, name):
        return _DeleteQuery(name, self.log)


def test_purge_deletes_unit_outcomes_first_then_enrollments_results_assignments():
    """(xi) FK-safe purge order: the outcome table references assignments."""
    loader = _FakeLoader()
    load_mod.purge_synthetic_ab_rows(loader)
    order = [t for op, t, *_ in loader.log if op == "execute"]
    assert order == [
        TABLE,
        "ab_experiment_enrollments",
        "ab_experiment_results",
        "ab_experiment_assignments",
    ]
    # every delete is scoped to synthetic rows, minimal-returning, exact-counted
    deletes = [e for e in loader.log if e[0] == "delete"]
    assert len(deletes) == 4
    for _op, t, extra in deletes:
        assert extra == {"returning": "minimal", "count": "exact"}, t
    eqs = [e for e in loader.log if e[0] == "eq"]
    assert len(eqs) == 4 and all(a == ("is_synthetic", True) for _op, _t, a in eqs)


def test_unit_outcomes_load_after_assignments_and_are_column_registered():
    """(xii) LOADING_ORDER places the child after its FK parent; TABLE_COLUMNS
    carries the full column set including is_synthetic (BatchLoader silently
    drops unregistered columns)."""
    assert TABLE in LOADING_ORDER
    assert LOADING_ORDER.index("ab_experiment_assignments") < LOADING_ORDER.index(TABLE)
    assert LOADING_ORDER.index("ml_experiments") < LOADING_ORDER.index(TABLE)
    assert set(TABLE_COLUMNS[TABLE]) == {
        "id",
        "assignment_id",
        "experiment_id",
        "unit_id",
        "metric_name",
        "outcome_value",
        "observed_at",
        "is_synthetic",
    }


def test_refresh_ab_datasets_carry_the_unit_outcome_frame_stamped_synthetic():
    hcp_ids = [f"scvhcp_{i:05d}" for i in range(400)]
    datasets = load_mod.build_ab_refresh_datasets({"trigger": 200}, hcp_ids=hcp_ids)
    assert TABLE in datasets
    uo = datasets[TABLE]
    assert len(uo) == len(datasets["ab_experiment_assignments"])
    assert bool(uo["is_synthetic"].all())
