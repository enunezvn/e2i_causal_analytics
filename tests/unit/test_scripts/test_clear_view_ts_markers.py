"""Bounded, composite-key dedup-marker clear for the renamed online view (codex r1 HIGH)."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "feature_repo"))
import clear_view_ts_markers as cvm  # noqa: E402


def test_scope_is_bounded_to_the_renamed_view():
    assert cvm.ALLOWED_VIEWS == ("hcp_conversion_features",)
    with pytest.raises(SystemExit):
        cvm.main(["--view", "goldstd_cohort_features", "--dry-run"])


def test_the_ids_query_selects_every_join_key_from_the_views_own_source():
    query = cvm.composite_ids_query(["hcp_id", "hcp_brand_id"], "(SELECT 1)")
    assert query == (
        "SELECT DISTINCT (hcp_id)::text, (hcp_brand_id)::text FROM (SELECT 1) AS _src "
        "WHERE hcp_id IS NOT NULL AND hcp_brand_id IS NOT NULL"
    )


def test_the_key_builder_passes_every_join_key_and_value():
    seen = []

    def make(keys, values):
        seen.append((list(keys), list(values)))
        return b"k"

    build = cvm.composite_key_builder(["hcp_id", "hcp_brand_id"], make)
    assert build(("h1", "h1_Kisqali")) == b"k"
    assert seen == [(["hcp_id", "hcp_brand_id"], ["h1", "h1_Kisqali"])]
    with pytest.raises(ValueError):
        build(("only-one",))


def test_clearing_touches_only_the_views_marker_field():
    calls = []

    class _Pipe:
        def hdel(self, key, field):
            calls.append((key, field))

        def hexists(self, key, field):
            calls.append((key, field))

        def execute(self):
            return [1] * len(calls)

    class _Redis:
        def pipeline(self, transaction=False):
            return _Pipe()

    build = cvm.composite_key_builder(["a", "b"], lambda keys, values: "|".join(values).encode())
    marker = cvm.ts_marker_field("hcp_conversion_features")
    hit, absent = cvm.clear_view_markers(_Redis(), build, [("1", "x"), ("2", "y")], marker)
    assert (hit, absent) == (2, 0)
    assert {field for _, field in calls} == {marker}
    assert [key for key, _ in calls] == [b"1|x", b"2|y"]
