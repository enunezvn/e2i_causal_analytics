"""nbrx series beside (never inside) the frozen business_metrics RNG stream."""

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.generators import BusinessMetricsGenerator, GeneratorConfig

BASE_CONFIG = {"id_prefix": "scv", "seed": 42, "n_records": 10000, "start_date": date(2013, 1, 1)}


@pytest.fixture(scope="module")
def base():
    return BusinessMetricsGenerator(GeneratorConfig(**BASE_CONFIG)).generate()


def _gen():
    from src.ml.synthetic.generators import nbrx_series

    return nbrx_series


def test_one_row_per_trx_cell_with_the_sibling_split(base):
    rows = _gen().generate_nbrx_rows(base)
    trx = base[base["metric_type"] == "trx"]
    assert len(rows) == len(trx) == 1956
    assert set(rows["metric_type"]) == {"nbrx"} and set(rows["metric_name"]) == {"nbrx"}
    merged = rows.merge(trx, on=["metric_date", "brand", "region"], suffixes=("", "_trx"))
    assert len(merged) == len(rows)
    assert (merged["data_split"] == merged["data_split_trx"]).all()


def test_ids_are_content_addressed_in_their_own_namespace(base):
    rows = _gen().generate_nbrx_rows(base)
    assert rows["metric_id"].is_unique
    assert rows["metric_id"].str.match(r"^nbrx_\d{6}_[a-z]+_[a-z]+$").all()
    assert (rows["metric_id"].str.len() <= 50).all()
    assert not set(rows["metric_id"]) & set(base["metric_id"])
    one = rows[
        (rows["metric_date"] == "2026-07-01")
        & (rows["brand"] == "Kisqali")
        & (rows["region"] == "midwest")
    ].iloc[0]
    assert one["metric_id"] == "nbrx_202607_kisqali_midwest"


def test_the_input_frame_is_not_touched(base):
    before = base.copy(deep=True)
    _gen().generate_nbrx_rows(base)
    pd.testing.assert_frame_equal(base, before)


def test_a_month_is_a_pure_function_of_its_calendar_key(base):
    g = _gen()
    from_base = g.generate_nbrx_rows(base)
    single = BusinessMetricsGenerator(
        GeneratorConfig(
            seed=7, n_records=61, start_date=date(2026, 7, 1), trend_origin=date(2013, 1, 1)
        )
    ).generate()
    from_single = g.generate_nbrx_rows(single)
    cols = [
        "metric_id",
        "metric_date",
        "brand",
        "region",
        "value",
        "target",
        "roi",
        "sample_size",
        "statistical_significance",
    ]
    a = from_base[from_base["metric_date"] == "2026-07-01"][cols].sort_values("metric_id")
    b = from_single[cols].sort_values("metric_id")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_a_missing_cell_does_not_shift_the_others(base):
    g = _gen()
    july = base[base["metric_date"] == "2026-07-01"]
    full = g.generate_nbrx_rows(july)
    holed = g.generate_nbrx_rows(
        july[~((july["brand"] == "Remibrutinib") & (july["region"] == "northeast"))]
    )
    kept = full[full["metric_id"].isin(holed["metric_id"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, holed.reset_index(drop=True))


def test_scale_sits_below_nrx(base):
    rows = _gen().generate_nbrx_rows(base)
    nrx = base[base["metric_type"] == "nrx"]
    for brand in ("Remibrutinib", "Fabhalta", "Kisqali"):
        ratio = (
            rows[rows["brand"] == brand]["value"].sum() / nrx[nrx["brand"] == brand]["value"].sum()
        )
        assert 0.2 < ratio < 0.8, (brand, ratio)


def test_carries_seasonality_and_the_execution_factor(base, monkeypatch):
    g = _gen()
    seasonal = g.generate_nbrx_rows(base)
    from src.ml.synthetic.generators import seasonality

    monkeypatch.setattr(seasonality, "SEASONAL_DEVIATION_BP", dict.fromkeys(range(1, 13), 0))
    flat = g.generate_nbrx_rows(base)
    months = pd.to_datetime(seasonal["metric_date"]).dt.month
    mask = months.isin([1, 12]) & (flat["value"] > 100)
    expected = months[mask].map({1: 0.92, 12: 1.08}).to_numpy()
    ratio = (seasonal["value"] / flat["value"])[mask].to_numpy()
    assert np.allclose(ratio, expected, rtol=5e-4)
    pd.testing.assert_series_equal(seasonal["target"], flat["target"])


def test_trend_origin_is_the_frozen_base_origin():
    import src.ml.synthetic.frontier_append as fa

    assert _gen().NBRX_TREND_ORIGIN == fa.BM_TREND_ORIGIN


def test_brand_and_region_order_is_the_generator_order():
    assert _gen().BRANDS == ("Remibrutinib", "Fabhalta", "Kisqali")
    assert _gen().REGIONS == ("northeast", "south", "midwest", "west")


def test_planted_events_now_cover_nbrx():
    for ev in BusinessMetricsGenerator.BRAND_REGION_EVENTS:
        assert "nbrx" in ev.metric_types, ev


# ---------------------------------------------------------------------------
# Execution factor, anchored events and RNG isolation (codex Task 3 r1). A frame
# spanning calendar 2026 holds every planted step: Kisqali midwest is 0.86
# before 2026-05, x0.88 from 2026-05 and x0.85 more from 2026-10.
# ---------------------------------------------------------------------------
YEAR_2026_CONFIG = {
    "id_prefix": "scv",
    "seed": 11,
    # 12 monthly dates x 60 brand/region/metric combos (+1 headroom per date).
    "n_records": 12 * 61,
    "start_date": date(2026, 1, 1),
    "trend_origin": date(2013, 1, 1),
}
# |round(x * f, 2) - round(x, 2) * f| <= 0.005 + 0.005 * f, f <= 1.10
ROUNDING_TOL = 0.011


@pytest.fixture(scope="module")
def year_2026():
    return BusinessMetricsGenerator(GeneratorConfig(**YEAR_2026_CONFIG)).generate()


def _real_and_flat(frame, monkeypatch):
    """nbrx rows with the planted tables, then with every brand x region term at
    identity (execution matrix all 1.0, no events). Same frame, same RNG."""
    g = _gen()
    real = g.generate_nbrx_rows(frame)
    monkeypatch.setattr(
        BusinessMetricsGenerator,
        "BRAND_REGION_PERFORMANCE",
        {b: dict.fromkeys(g.REGIONS, 1.0) for b in g.BRANDS},
    )
    monkeypatch.setattr(BusinessMetricsGenerator, "BRAND_REGION_EVENTS", ())
    flat = g.generate_nbrx_rows(frame)
    assert list(real["metric_id"]) == list(flat["metric_id"])
    return real, flat


def test_value_carries_the_execution_factor_and_events_target_does_not(year_2026, monkeypatch):
    # Resolve the planted factors BEFORE _real_and_flat patches the tables away.
    cells = _gen().generate_nbrx_rows(year_2026)
    expected = pd.Series(
        [
            BusinessMetricsGenerator.brand_region_factor(b, r, "nbrx", date.fromisoformat(d))
            for b, r, d in zip(cells["brand"], cells["region"], cells["metric_date"], strict=True)
        ]
    )
    real, flat = _real_and_flat(year_2026, monkeypatch)
    assert len(real) == 144
    assert (expected != 1.0).any()
    gap = (real["value"] - flat["value"] * expected).abs()
    assert gap.max() <= ROUNDING_TOL, real.loc[gap.idxmax(), ["metric_id", "value"]]
    pd.testing.assert_series_equal(real["target"], flat["target"])


def test_planted_steps_move_kisqali_midwest_nbrx(year_2026, monkeypatch):
    real, flat = _real_and_flat(year_2026, monkeypatch)
    # Literal pins (not brand_region_factor): execution 0.86, then the two steps.
    steps = {
        "2026-04-01": 0.86,
        "2026-06-01": 0.86 * 0.88,
        "2026-10-01": 0.86 * 0.88 * 0.85,
        "2026-12-01": 0.86 * 0.88 * 0.85,
    }
    for metric_date, factor in steps.items():
        mask = (
            (real["metric_date"] == metric_date)
            & (real["brand"] == "Kisqali")
            & (real["region"] == "midwest")
        )
        assert mask.sum() == 1, metric_date
        gap = abs(
            float(real.loc[mask, "value"].iloc[0]) - float(flat.loc[mask, "value"].iloc[0]) * factor
        )
        assert gap <= ROUNDING_TOL, (metric_date, factor, gap)


def test_input_row_order_does_not_matter(year_2026):
    g = _gen()
    ordered = g.generate_nbrx_rows(year_2026)
    shuffled = g.generate_nbrx_rows(year_2026.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(
        ordered.sort_values("metric_id").reset_index(drop=True),
        shuffled.sort_values("metric_id").reset_index(drop=True),
    )


def test_base_generator_and_global_numpy_rng_are_untouched():
    gen = BusinessMetricsGenerator(GeneratorConfig(**YEAR_2026_CONFIG))
    frame = gen.generate()
    gen_state = copy.deepcopy(gen._rng.bit_generator.state)
    np_before = np.random.get_state()
    _gen().generate_nbrx_rows(frame)
    assert gen._rng.bit_generator.state == gen_state
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    assert np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
