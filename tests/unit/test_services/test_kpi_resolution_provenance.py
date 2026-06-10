"""Shard 07 R10: kpi_resolution._fetch_df default-excludes is_synthetic on taggable
tables (treatment_events/triggers/hcp_profiles), opts in on include_synthetic=True, and
never adds the predicate on a non-taggable table (no 42703)."""

from src.services import kpi_resolution as kr


class _RecQuery:
    def __init__(self, eq_calls):
        self._eq = eq_calls

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self._eq.append((col, val))
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        class _R:
            data = []

        return _R()


class _RecClient:
    def __init__(self):
        self.eq_calls = []

    def table(self, *a, **k):
        return _RecQuery(self.eq_calls)


def test_fetch_df_default_excludes_synthetic_on_taggable():
    c = _RecClient()
    kr._fetch_df(c, "treatment_events", "patient_id,event_date")
    assert ("is_synthetic", False) in c.eq_calls


def test_fetch_df_opt_in_includes_synthetic():
    c = _RecClient()
    kr._fetch_df(c, "treatment_events", "patient_id,event_date", include_synthetic=True)
    assert ("is_synthetic", False) not in c.eq_calls


def test_fetch_df_non_taggable_table_unfiltered():
    c = _RecClient()
    kr._fetch_df(c, "some_lookup_table", "x")
    assert ("is_synthetic", False) not in c.eq_calls
