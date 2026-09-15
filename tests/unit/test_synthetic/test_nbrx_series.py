"""nbrx series beside (never inside) the frozen business_metrics RNG stream."""

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
