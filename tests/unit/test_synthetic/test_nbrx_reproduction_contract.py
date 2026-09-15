"""nbrx reproduction contract across every entrypoint (canonical TRx lane, codex r1).

For any month present in two frames, the nbrx rows are identical on EVERY column
except data_split, and each frame's nbrx data_split equals its own sibling trx row.

Codex r2: before any value is compared, every frame must hold the COMPLETE id set
(3 brands x 4 regions for every trx month, each id equal to its content address),
and two frames must overlap on exactly the expected months. The DR frame comes from
the loader's own build_business_metrics_dataset (seed 42, n=10000, id prefix
"scv", default start). Its window ends at the run month because the generator
anchors a default start to ``date.today()``.

Codex 5A r1: that clock is frozen at RUN_DAY for the whole module, so the DR
fixture and every expectation read the same day even across a month boundary.
Every entrypoint is proven (at runtime, not by source text) to route its base and
cohort rows through the one seam, ``with_nbrx``.
"""

from contextlib import contextmanager
from datetime import date

import pandas as pd
import pytest

import src.ml.synthetic.frontier_append as fa
import src.ml.synthetic.generators.business_metrics_generator as bm_generator
from src.ml.synthetic.generators import BusinessMetricsGenerator, GeneratorConfig

FRONTIER = date(2026, 9, 15)
RUN_DAY = date(2026, 9, 15)

# Written out here on purpose, NOT read from nbrx_series._COLUMNS: the schema every
# entrypoint's nbrx rows must carry (column order and dtype).
EXPECTED_NBRX_SCHEMA = {
    "metric_id": "object",
    "metric_date": "object",
    "metric_type": "object",
    "metric_name": "object",
    "brand": "object",
    "region": "object",
    "value": "float64",
    "target": "float64",
    "achievement_rate": "float64",
    "year_over_year_change": "float64",
    "month_over_month_change": "float64",
    "roi": "float64",
    "statistical_significance": "float64",
    "confidence_interval_lower": "float64",
    "confidence_interval_upper": "float64",
    "sample_size": "int64",
    "data_split": "object",
}


@contextmanager
def _frozen_clock(day: date):
    """``date.today()`` inside the business_metrics generator returns ``day``."""
    frozen = type("FrozenDate", (date,), {"today": classmethod(lambda cls: day)})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bm_generator, "date", frozen)
        yield


@pytest.fixture(scope="module", autouse=True)
def frozen_run_day():
    with _frozen_clock(RUN_DAY):
        yield


def _ns():
    from src.ml.synthetic.generators import nbrx_series

    return nbrx_series


def _month(value) -> date:
    return date.fromisoformat(str(value)[:10]).replace(day=1)


def _months(frame: pd.DataFrame) -> set:
    return {_month(v) for v in frame.loc[frame["metric_type"] == "trx", "metric_date"]}


def _expected_ids(months) -> set:
    ns = _ns()
    return {ns.nbrx_metric_id(m, b, r) for m in months for b in ns.BRANDS for r in ns.REGIONS}


def _assert_complete(frame: pd.DataFrame) -> None:
    """Exactly one correctly addressed nbrx row per (trx month, brand, region)."""
    nbrx = frame[frame["metric_type"] == "nbrx"]
    ids = list(nbrx["metric_id"])
    assert len(ids) == len(set(ids)), "duplicate nbrx ids"
    months = _months(frame)
    assert months, "frame has no trx months"
    assert set(ids) == _expected_ids(months)
    # Codex r3: per ROW, not per set — two swapped ids leave both sets unchanged.
    mismatched = [
        (r.metric_id, _ns().nbrx_metric_id(_month(r.metric_date), str(r.brand), str(r.region)))
        for r in nbrx.itertuples()
        if r.metric_id != _ns().nbrx_metric_id(_month(r.metric_date), str(r.brand), str(r.region))
    ]
    assert mismatched == [], (
        f"an nbrx id does not match its own month/brand/region: {mismatched[:6]}"
    )


def _assert_schema(frame: pd.DataFrame) -> None:
    columns = [c for c in frame.columns if c != "is_synthetic"]
    assert columns == list(EXPECTED_NBRX_SCHEMA)
    nbrx = frame.loc[frame["metric_type"] == "nbrx", columns]
    assert len(nbrx) > 0
    assert dict(nbrx.dtypes.astype(str)) == EXPECTED_NBRX_SCHEMA


def _values(frame: pd.DataFrame, months: set) -> pd.DataFrame:
    cols = [c for c in _ns()._COLUMNS if c not in ("metric_id", "data_split")]
    rows = frame[frame["metric_type"] == "nbrx"].copy()
    rows = rows[rows["metric_date"].map(_month).isin(months)]
    # metric_date is compared as an ISO day: frames differ only in date vs Timestamp dtype.
    rows["metric_date"] = rows["metric_date"].map(lambda v: str(v)[:10])
    return rows.set_index("metric_id").sort_index()[cols]


def _same_rows(a: pd.DataFrame, b: pd.DataFrame, expected_overlap: set) -> None:
    _assert_complete(a)
    _assert_complete(b)
    overlap = _months(a) & _months(b)
    assert overlap == expected_overlap, sorted(overlap ^ expected_overlap)[:6]
    left, right = _values(a, overlap), _values(b, overlap)
    assert set(left.index) == set(right.index) == _expected_ids(overlap)
    pd.testing.assert_frame_equal(left, right)


def _this_month() -> date:
    return RUN_DAY.replace(day=1)


def _month_range(first: date, last: date) -> set:
    out, m = set(), first
    while m <= last:
        out.add(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


def _sibling_split(frame: pd.DataFrame) -> None:
    nbrx = frame[frame["metric_type"] == "nbrx"]
    trx = frame[frame["metric_type"] == "trx"]
    merged = nbrx.merge(trx, on=["metric_date", "brand", "region"], suffixes=("", "_trx"))
    assert len(merged) == len(nbrx)
    assert (merged["data_split"] == merged["data_split_trx"]).all()


def _single_month_frame() -> pd.DataFrame:
    return BusinessMetricsGenerator(
        GeneratorConfig(
            seed=3, n_records=61, start_date=date(2026, 7, 1), trend_origin=date(2013, 1, 1)
        )
    ).generate()


@pytest.fixture(scope="module")
def reseed() -> pd.DataFrame:
    from scripts.reseed_business_metrics_aggregate import build_reseed_frame

    return build_reseed_frame(frontier=FRONTIER)


@pytest.fixture(scope="module")
def dr() -> pd.DataFrame:
    """The DR loader's real preparation function with the loader's own defaults
    (FULL_SIZES, seed 42, `--tag` default "scv", no explicit start)."""
    import scripts.load_synthetic_data as loader

    with _frozen_clock(RUN_DAY):  # the DR window ends at RUN_DAY's month
        return loader.build_business_metrics_dataset(
            loader.FULL_SIZES["business_metrics"], seed=42, id_prefix="scv"
        )


def test_reseed_frame_holds_every_month_complete(reseed):
    assert _months(reseed) == _month_range(date(2013, 1, 1), FRONTIER.replace(day=1))
    _assert_complete(reseed)
    _sibling_split(reseed)


def test_frontier_cohort_matches_the_reseed_frame(reseed):
    cohort = fa.generate_month_cohort(date(2026, 9, 1))["business_metrics"]
    _same_rows(cohort, reseed, {date(2026, 9, 1)})
    _sibling_split(cohort)


def test_arbiter_frame_matches_the_reseed_frame(reseed):
    from scripts.gap_arbiter_1833 import build_frame

    arbiter = build_frame(date(2026, 9, 1))
    _same_rows(arbiter, reseed, _months(reseed))
    _sibling_split(arbiter)


def test_assert_complete_catches_two_swapped_ids(reseed):
    """Teeth for the per-row identity check (codex r3): swapping two real rows' ids
    keeps the id SET complete, so only the per-row comparison can fail."""
    frame = reseed.copy()
    first, second = frame.index[frame["metric_type"] == "nbrx"][:2]
    a, b = frame.at[first, "metric_id"], frame.at[second, "metric_id"]
    assert a != b
    frame.at[first, "metric_id"], frame.at[second, "metric_id"] = b, a
    assert set(frame.loc[frame["metric_type"] == "nbrx", "metric_id"]) == set(
        reseed.loc[reseed["metric_type"] == "nbrx", "metric_id"]
    )
    with pytest.raises(AssertionError, match="does not match its own month/brand/region"):
        _assert_complete(frame)


def test_dr_loader_frame_ends_at_the_run_month_and_is_complete(dr):
    months = _months(dr)
    assert max(months) == _this_month()
    assert len(months) == 10000 // 61  # 163 monthly dates (§0.5 sizing)
    _assert_complete(dr)
    _sibling_split(dr)


def test_dr_loader_frame_matches_the_reseed_frame_on_overlapping_months(dr, reseed):
    last = min(_this_month(), FRONTIER.replace(day=1))
    _same_rows(dr, reseed, _month_range(min(_months(dr)), last))


def test_dr_loader_frame_matches_this_months_cron_cohort(dr):
    cohort = fa.generate_month_cohort(_this_month())["business_metrics"]
    _same_rows(dr, cohort, {_this_month()})


def test_every_entrypoint_routes_base_and_cohort_rows_through_with_nbrx(monkeypatch):
    """A runtime spy on the seam at each call site's own imported name: which frames
    reach ``with_nbrx``, from where, and never one that already carries nbrx rows."""
    import scripts.gap_arbiter_1833 as arbiter
    import scripts.load_synthetic_data as loader
    import scripts.reseed_business_metrics_aggregate as reseed_script

    real = _ns().with_nbrx
    calls = []

    def spy(site):
        def _with_nbrx(frame):
            prefixes = frozenset(frame["metric_id"].str.split("_").str[0])
            calls.append((site, len(frame), prefixes, bool((frame["metric_type"] == "nbrx").any())))
            return real(frame)

        return _with_nbrx

    for site, module in (
        ("reseed", reseed_script),
        ("arbiter", arbiter),
        ("cohort", fa),
        ("dr", loader),
    ):
        monkeypatch.setattr(module, "with_nbrx", spy(site))

    reseed_script.build_reseed_frame(frontier=date(2026, 8, 30))
    assert calls == [
        ("reseed", 9780, frozenset({"metric"}), False),
        ("cohort", 60, frozenset({"m2608"}), False),
    ]
    calls.clear()
    arbiter.build_frame(date(2026, 8, 1))
    assert calls == [
        ("arbiter", 9780, frozenset({"metric"}), False),
        ("cohort", 60, frozenset({"m2608"}), False),
    ]
    calls.clear()
    fa.generate_month_cohort(date(2026, 9, 1))
    assert calls == [("cohort", 60, frozenset({"m2609"}), False)]
    calls.clear()
    loader.build_business_metrics_dataset(2 * 61, seed=42, id_prefix="scv")
    assert calls == [("dr", 120, frozenset({"metric"}), False)]


def test_reseed_and_arbiter_frames_equal_the_pre_seam_composition():
    """Routing the base through with_nbrx changes no row, column, dtype or order:
    both frames equal the Task 4 composition written out by hand."""
    from scripts.gap_arbiter_1833 import build_frame
    from scripts.reseed_business_metrics_aggregate import build_reseed_frame

    base = fa.base_business_metrics_frame()
    legacy = pd.concat(
        [
            base,
            _ns().generate_nbrx_rows(base),
            fa.generate_month_cohort(date(2026, 8, 1))["business_metrics"],
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(build_frame(date(2026, 8, 1)), legacy)
    legacy["is_synthetic"] = True
    pd.testing.assert_frame_equal(build_reseed_frame(frontier=date(2026, 8, 30)), legacy)


def test_every_entrypoint_carries_the_expected_nbrx_schema(dr, reseed):
    import scripts.load_synthetic_data as loader
    from scripts.gap_arbiter_1833 import build_frame

    _assert_schema(fa.generate_month_cohort(date(2026, 8, 1))["business_metrics"])
    _assert_schema(build_frame(date(2026, 8, 1)))
    _assert_schema(reseed)
    _assert_schema(dr)
    _assert_schema(loader.build_business_metrics_dataset(2 * 61, seed=42, id_prefix="scv"))


def test_each_nbrx_row_copies_its_own_regions_split():
    """Heterogeneous splits inside ONE month: each nbrx row takes its own
    (brand, region) sibling's split, never a month-level one."""
    frame = _single_month_frame()
    splits = {"northeast": "train", "south": "validation", "midwest": "test", "west": "holdout"}
    trx = frame["metric_type"] == "trx"
    frame.loc[trx, "data_split"] = frame.loc[trx, "region"].map(splits)
    nbrx = _ns().with_nbrx(frame)
    nbrx = nbrx[nbrx["metric_type"] == "nbrx"]
    assert len(nbrx) == 12
    assert (nbrx["data_split"] == nbrx["region"].map(splits)).all()


def test_with_nbrx_appends_and_never_mutates():
    frame = _single_month_frame()
    before = frame.copy(deep=True)
    out = _ns().with_nbrx(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert len(out) == len(frame) + 12
    pd.testing.assert_frame_equal(
        out.iloc[: len(frame)].reset_index(drop=True), before.reset_index(drop=True)
    )
    # Order invariant: the frame's own rows first, its nbrx rows after them.
    assert not (out.iloc[: len(frame)]["metric_type"] == "nbrx").any()
    assert (out.iloc[len(frame) :]["metric_type"] == "nbrx").all()


def test_with_nbrx_refuses_a_frame_that_already_carries_nbrx():
    once = _ns().with_nbrx(_single_month_frame())
    with pytest.raises(ValueError, match="already carries nbrx"):
        _ns().with_nbrx(once)
