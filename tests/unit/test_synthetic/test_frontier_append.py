"""Frontier-append business_metrics cohort tests (#1566 D1) and instant-
granularity holdback tests (#1577).

#1566 D1: the frozen base (loaded 2026-07-03) spans 163 months 2013-01..
2026-07, so its July-2026 trx rows carry trend_factor 1 + 0.02*162. Monthly
frontier cohorts (``generate_month_cohort``) are single-date runs, so the
positional ``month_idx = dates.index(metric_date)`` reset to 0 and every
appended month from BM_EPOCH (2026-08-01) forward collapsed to the 2013
baseline (~24% of July's level) — deterministically and permanently. D1
anchors the cohort's trend to the absolute calendar origin ``BM_TREND_ORIGIN
= date(2013, 1, 1)`` instead; ``trend_origin=None`` preserves the positional
behavior for every other caller byte-for-byte.

#1577: the DB enforces ``CHECK (event_timestamp <= now())`` on feature_values
(constraint ``valid_event_timestamp`` — the only now()-CHECK in the schema).
``filter_to_frontier`` filtered at DATE granularity only, so the Mon-3AM cron
(frontier=today) shipped frontier-day rows timed AFTER 03:00 (measured: e.g.
``2026-08-10 18:56:17`` on the 2026-08-10 03:00 run) — 28-459 rejected rows
per weekly run, every cron run since 2026-07-06. The ``as_of`` instant
holdback extends the filter's documented "held back (regenerated next run)"
self-heal contract from date to instant granularity. Rows must be HELD BACK,
never clamped: clamping would change ``event_timestamp`` — part of the
loader's upsert key ``(feature_id, entity_values, event_timestamp)``
(batch_loader TABLE_ON_CONFLICT) — so re-generated rows would duplicate
instead of dedup on the next run.
"""

import json
from datetime import date, datetime

import pandas as pd

import src.ml.synthetic.frontier_append as fa
from src.ml.synthetic.frontier_append import (
    build_frontier_datasets,
    filter_to_frontier,
    generate_month_cohort,
)
from src.ml.synthetic.generators import (
    BusinessMetricsGenerator,
    GeneratorConfig,
    HCPGenerator,
)

# Frozen-base trx model pinned literally (NOT read from METRIC_CONFIGS): the DB
# base was generated from these values on 2026-07-03 and does not retune with
# code edits — if someone retunes the generator config, appended cohorts WOULD
# diverge from the frozen base, and this test should break.
TRX_BASE = {"Remibrutinib": 15000.0, "Fabhalta": 8000.0, "Kisqali": 50000.0}
# Market-size factor (value AND target) — unchanged by #1833.
REGION_FACTORS = {"northeast": 1.15, "south": 0.95, "midwest": 0.90, "west": 1.00}
# #1833: the base in the DB is RESEEDED under the brand x region formula
# (value only; scripts/reseed_business_metrics_aggregate.py), so appended
# cohorts must carry the SAME execution matrix and the SAME anchored events —
# pinned literally here for the same reason as REGION_FACTORS: a retune that
# is not accompanied by a base reseed puts cohorts on a different line than
# the base, and this test must break.
BRAND_REGION_PERFORMANCE = {
    "Kisqali": {"northeast": 1.09, "south": 0.97, "midwest": 0.86, "west": 1.04},
    "Fabhalta": {"northeast": 1.03, "south": 0.86, "midwest": 0.98, "west": 1.10},
    "Remibrutinib": {"northeast": 1.00, "south": 1.08, "midwest": 1.04, "west": 0.88},
}
# Step events active on 2026-08-01 for trx: (brand, region) -> compounded factor.
EVENTS_ACTIVE_AUG_2026 = {
    ("Kisqali", "midwest"): 0.88,
    ("Fabhalta", "south"): 0.88,
    ("Remibrutinib", "west"): 0.88,
}
TRX_TREND = 0.02  # 2%/month
# (2026-08 - 2013-01) in calendar months: (2026-2013)*12 + (8-1) = 163.
AUG_2026_IDX = 163


def _expected_trx(brand: str, region: str, month_idx: int, events: dict) -> float:
    return (
        TRX_BASE[brand]
        * REGION_FACTORS[region]
        * (1 + TRX_TREND * month_idx)
        * BRAND_REGION_PERFORMANCE[brand][region]
        * events.get((brand, region), 1.0)
    )


def _cohort_config(**overrides) -> GeneratorConfig:
    """Mirror generate_month_cohort's GeneratorConfig for August 2026."""
    month_start = date(2026, 8, 1)
    params = {
        "id_prefix": fa.month_prefix(month_start),
        "seed": fa.month_seed(month_start),
        "n_records": fa.MONTHLY_SIZES["business_metrics"],
        "start_date": month_start,
        "end_date": date(2026, 8, 31),
    }
    params.update(overrides)
    return GeneratorConfig(**params)


class TestMonthCohortContinuity:
    """The #1566 red test: appended months must sit at frozen-base scale."""

    def test_august_2026_trx_sits_at_base_scale(self):
        bm = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
        trx = bm[bm["metric_type"] == "trx"]
        # 3 brands x 4 regions on the single cohort date.
        assert len(trx) == 12

        expected = trx.apply(
            lambda r: _expected_trx(r["brand"], r["region"], AUG_2026_IDX, EVENTS_ACTIVE_AUG_2026),
            axis=1,
        )
        ratio = float((trx["value"] / expected).mean())
        # Per-row noise is N(0, 0.15); the 12-row mean ratio must sit near 1.
        # Under the positional month_idx=0 defect it sits at ~0.235.
        assert 0.85 <= ratio <= 1.15, (
            f"appended-month trx collapsed off frozen-base scale: mean ratio {ratio:.3f}"
        )

    def test_bm_trend_origin_frozen_constant(self):
        # FROZEN: 2013-01-01 is the first month of the base actually loaded in
        # the DB (2026-07-03 load) — July 2026 = index 162, first appended
        # month 2026-08 = index 163. NOT re-derivable by re-running the base
        # generator (its default date range re-anchors to the run date).
        assert fa.BM_TREND_ORIGIN == date(2013, 1, 1)

    def test_cohort_months_sit_on_the_regenerated_base_line(self):
        """#1833 acceptance (d): the 2026-08 / 2026-09 cohorts continue the
        RESEEDED base — same market-size line, same brand x region execution
        matrix, same anchored events — not the pre-#1833 flat line.

        Per (brand, region) the cohort row and the base's last 6 months are
        divided by their deterministic expectation (which carries the planted
        factor); the noise-only residuals must then sit on the same level. A
        cohort still on the flat formula would read ~1/0.86 for Kisqali/midwest
        against a base residual of ~1.
        """

        def residual(row) -> float:
            d = date.fromisoformat(row["metric_date"])
            idx = (d.year - 2013) * 12 + (d.month - 1)
            line = (
                TRX_BASE[row["brand"]]
                * REGION_FACTORS[row["region"]]
                * (1 + TRX_TREND * idx)
                * BusinessMetricsGenerator.brand_region_factor(
                    row["brand"], row["region"], "trx", d
                )
            )
            return row["value"] / line

        base = fa.base_business_metrics_frame()
        base_trx = base[(base["metric_type"] == "trx") & (base["metric_date"] >= "2026-02-01")]
        for month_start in (date(2026, 8, 1), date(2026, 9, 1)):
            cohort = fa.generate_month_cohort(month_start)["business_metrics"]
            trx = cohort[cohort["metric_type"] == "trx"]
            assert len(trx) == 12
            for brand, region in EVENTS_ACTIVE_AUG_2026:
                row = trx[(trx["brand"] == brand) & (trx["region"] == region)].iloc[0]
                b = base_trx[(base_trx["brand"] == brand) & (base_trx["region"] == region)]
                base_resid = float(b.apply(residual, axis=1).mean())
                # noise N(0, 0.15) per row: one cohort row vs a 6-row base mean
                assert abs(residual(row) - base_resid) < 0.45, (
                    month_start,
                    brand,
                    region,
                    residual(row),
                    base_resid,
                )


class TestBaseBusinessMetricsFrame:
    """#1833: the frozen-base regeneration identity, shared by the reseed
    script and the gap arbiter. Step 0 measured it byte-identical to the DB."""

    def test_regenerates_the_frozen_base_identity(self):
        base = fa.base_business_metrics_frame()
        assert len(base) == 9780
        assert base["metric_date"].min() == "2013-01-01"
        assert base["metric_date"].max() == "2026-07-01"
        # literal DB fingerprint (metric_id is a seeded draw; target is on the
        # market-size line — neither moves under the #1833 value-only terms)
        row = base[
            (base["metric_date"] == "2026-07-01")
            & (base["brand"] == "Kisqali")
            & (base["region"] == "midwest")
            & (base["metric_name"] == "trx")
        ].iloc[0]
        assert row["metric_id"] == "metric_2ca0d492f13b"
        assert row["target"] == 219010.90
        # base ids are the unprefixed generator namespace, disjoint from cohorts
        assert base["metric_id"].str.startswith("metric_").all()

    def test_base_constants_match_the_loader(self):
        assert fa.BM_BASE_N == 10000
        assert fa.BM_BASE_START == fa.BM_TREND_ORIGIN == date(2013, 1, 1)


class TestMonthCohortDeterminism:
    def test_same_month_twice_is_identical(self):
        a = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
        b = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
        pd.testing.assert_frame_equal(a, b)


class TestTrendOriginChangedColumnInvariance:
    """trend_origin feeds arithmetic only — it must consume no RNG variates.

    Pin the EXACT set of columns that differ between trend_origin=None and
    trend_origin=2013-01-01 on the same cohort config (measured, then frozen):

    - value:  base * region * trend_factor * (1 + noise) — trend_factor moves
      from 1.00 (positional idx 0) to 4.26 (calendar idx 163).
    - target: same trend_factor term times the target multiplier draw.
    - confidence_interval_lower/upper: derived from value (std_error is
      value * volatility / sqrt(sample_size)).
    - achievement_rate: value/target is trend_factor-INVARIANT before rounding
      ((1+noise)/multiplier), but (a) value and target are rounded to 2
      decimals before the ratio, so sub-1.0 metrics (market_share,
      conversion_rate) re-quantize differently at different scales (measured
      max |diff| 0.077 / 0.042 on the Aug-2026 cohort), and (b) the
      hcp_engagement_score value cap at 10.0 engages at calendar scale
      (10 of 12 rows; measured max |diff| 0.289) while its target is uncapped
      (the inherited cap defect noted in #1566 — out of D1 scope). trx/nrx
      agree byte-for-byte (measured max |diff| 0.0).

    Byte-identical (pinned below): metric_id, metric_date, metric_type,
    metric_name, brand, region, year_over_year_change, month_over_month_change,
    roi, statistical_significance, sample_size, data_split — the yoy/mom/roi
    stamps are model-parameter draws (D2), the id/statistical draws precede no
    month_idx-dependent RNG call, and data_split depends on dates only.
    """

    EXPECTED_CHANGED = {
        "value",
        "target",
        "achievement_rate",
        "confidence_interval_lower",
        "confidence_interval_upper",
    }

    def _frames(self):
        df_none = BusinessMetricsGenerator(_cohort_config()).generate()
        df_origin = BusinessMetricsGenerator(
            _cohort_config(trend_origin=date(2013, 1, 1))
        ).generate()
        return df_none, df_origin

    def test_changed_column_set_is_exactly_the_value_leg(self):
        df_none, df_origin = self._frames()
        assert list(df_none.columns) == list(df_origin.columns)
        changed = {col for col in df_none.columns if not df_none[col].equals(df_origin[col])}
        assert changed == self.EXPECTED_CHANGED, (
            f"unexpected trend_origin-sensitive columns: "
            f"extra={changed - self.EXPECTED_CHANGED}, "
            f"missing={self.EXPECTED_CHANGED - changed}"
        )

    def test_rng_stream_columns_byte_identical(self):
        df_none, df_origin = self._frames()
        for col in (
            "metric_id",
            "year_over_year_change",
            "month_over_month_change",
            "roi",
            "statistical_significance",
            "sample_size",
            "data_split",
        ):
            pd.testing.assert_series_equal(df_none[col], df_origin[col]), col

    def test_achievement_rate_agrees_where_value_cap_never_engages(self):
        # value/target is scale-invariant modulo the 2-decimal rounding of
        # value and target; for the large-value prescription metrics that
        # quantization is negligible (~1e-7 relative), so agreement holds
        # within the spec's ~2e-3 rounding tolerance. Sub-1.0 metrics and the
        # capped hcp_engagement_score are excluded (see class docstring).
        df_none, df_origin = self._frames()
        mask = df_none["metric_type"].isin(["trx", "nrx"])
        diff = (
            (df_none.loc[mask, "achievement_rate"] - df_origin.loc[mask, "achievement_rate"])
            .abs()
            .max()
        )
        assert diff <= 2e-3, f"trx/nrx achievement_rate moved beyond rounding: {diff}"


class TestPKStability:
    def test_metric_id_set_identical_with_and_without_trend_origin(self):
        df_none = BusinessMetricsGenerator(_cohort_config()).generate()
        df_origin = BusinessMetricsGenerator(
            _cohort_config(trend_origin=date(2013, 1, 1))
        ).generate()
        assert list(df_none["metric_id"]) == list(df_origin["metric_id"])


# ---------------------------------------------------------------------------
# #1577: instant-granularity (as_of) holdback
# ---------------------------------------------------------------------------

# EPOCH (2026-07-06) is a Monday: with frontier=EPOCH the trailing window is a
# SINGLE weekly cohort (iter_week_starts clamps at EPOCH) and no monthly bm
# cohorts exist yet (BM_EPOCH is 2026-08-01) — the cheapest full pass through
# the real build_frontier_datasets path.
FRONTIER_MONDAY = fa.EPOCH
# The real cron instant: Monday 03:00:50 (the measured 2026-08-10 run started
# 03:00:50).
CRON_AS_OF = datetime(2026, 7, 6, 3, 0, 50)
FAR_AS_OF = datetime(2100, 1, 1)
NEXT_MONDAY = date(2026, 7, 13)
NEXT_CRON_AS_OF = datetime(2026, 7, 13, 3, 0, 50)


def _small_hcp() -> pd.DataFrame:
    """Real HCPGenerator, smaller universe (the factory is an existing
    injection point of build_frontier_datasets — nothing is mocked). Both
    runs of any comparison use the same factory, so determinism and
    self-heal contracts hold exactly as with the full 5000-HCP base."""
    return HCPGenerator(GeneratorConfig(id_prefix="scv", seed=42, n_records=50)).generate()


_BUILD_CACHE: dict = {}


def _build(frontier: date, as_of):
    """Cached build_frontier_datasets(frontier, as_of) — generation is ~2s
    per cohort; tests share read-only results."""
    key = (frontier, as_of)
    if key not in _BUILD_CACHE:
        _BUILD_CACHE[key] = build_frontier_datasets(
            frontier=frontier,
            as_of=as_of,
            include_coverage=False,
            hcp_frame_factory=_small_hcp,
        )
    return _BUILD_CACHE[key]


def _fv_keyed(fv: pd.DataFrame) -> pd.DataFrame:
    """feature_values on the loader's upsert key (feature_entity_timestamp_
    unique) plus the value columns byte-identity is asserted over. The ``id``
    column is uuid4-per-generation (measured: the ONLY non-deterministic
    column) and is resolved by the DB upsert, so it is excluded."""
    out = pd.DataFrame(
        {
            "feature_id": fv["feature_id"].astype(str),
            "entity_values": fv["entity_values"].map(lambda v: json.dumps(v, sort_keys=True)),
            "event_timestamp": fv["event_timestamp"].astype(str),
            "value": fv["value"].map(lambda v: json.dumps(v, sort_keys=True)),
            "freshness_status": fv["freshness_status"].astype(str),
        }
    )
    return out.sort_values(["feature_id", "entity_values", "event_timestamp"]).reset_index(
        drop=True
    )


def _fv_keys(fv: pd.DataFrame) -> set:
    keyed = _fv_keyed(fv)
    return set(
        zip(keyed["feature_id"], keyed["entity_values"], keyed["event_timestamp"], strict=True)
    )


class TestAsOfInstantHoldback:
    """Headline #1577 contract: a Monday-03:00 run must not emit feature_values
    rows the DB CHECK (event_timestamp <= now()) will reject."""

    def test_no_feature_value_after_as_of(self):
        fv = _build(FRONTIER_MONDAY, CRON_AS_OF)["feature_values"]
        assert len(fv) > 0
        # ISO8601: generated isoformat() strings have MIXED precision (rows
        # clipped to a whole second print without the .%f fraction)
        ts = pd.to_datetime(fv["event_timestamp"], format="ISO8601")
        offenders = fv[ts > pd.Timestamp(CRON_AS_OF)]
        assert offenders.empty, (
            f"{len(offenders)} feature_values rows after as_of={CRON_AS_OF} would "
            f"violate the DB valid_event_timestamp CHECK: "
            f"{offenders['event_timestamp'].tolist()[:5]}"
        )

    def test_holdback_is_not_vacuous(self):
        """The hazard is real: without the instant cutoff, frontier-day rows
        timed after 03:00 exist (measured 10 of 28 on the EPOCH cohort) —
        exactly the rows all 6 cron runs since 2026-07-06 failed on."""
        fv = _build(FRONTIER_MONDAY, FAR_AS_OF)["feature_values"]
        ts = pd.to_datetime(fv["event_timestamp"], format="ISO8601")
        intraday_future = (ts > pd.Timestamp(CRON_AS_OF)) & (ts.dt.date == FRONTIER_MONDAY)
        assert intraday_future.any(), (
            "expected frontier-day rows timed after the cron instant; the "
            "holdback test would be vacuous without them"
        )

    def test_boundary_row_at_as_of_is_kept(self):
        """CHECK (event_timestamp <= now()) is inclusive — a row exactly AT
        as_of must load, not be held back."""
        df = pd.DataFrame(
            {
                "feature_id": ["f1", "f2", "f3"],
                "event_timestamp": [
                    "2026-07-06T03:00:50",
                    "2026-07-06T03:00:50.000001",
                    "2026-07-06T02:59:59",
                ],
            }
        )
        out = filter_to_frontier(
            {"feature_values": df}, date(2026, 7, 6), as_of=datetime(2026, 7, 6, 3, 0, 50)
        )["feature_values"]
        assert list(out["feature_id"]) == ["f1", "f3"]

    def test_filter_without_as_of_stays_date_only(self):
        """as_of=None preserves the pre-#1577 date-granularity contract for
        direct filter_to_frontier callers."""
        df = pd.DataFrame(
            {
                "feature_id": ["f1", "f2"],
                "event_timestamp": ["2026-07-06T23:59:59", "2026-07-07T00:00:00"],
            }
        )
        out = filter_to_frontier({"feature_values": df}, date(2026, 7, 6))["feature_values"]
        assert list(out["feature_id"]) == ["f1"]


class TestAsOfSelfHeal:
    """Held-back rows are DEFERRED, never lost: the next run regenerates them
    byte-identically (on the upsert key) and they pass its later as_of."""

    def test_held_back_rows_reappear_next_run(self):
        all_rows = _fv_keys(_build(FRONTIER_MONDAY, FAR_AS_OF)["feature_values"])
        loaded = _fv_keys(_build(FRONTIER_MONDAY, CRON_AS_OF)["feature_values"])
        held_back = all_rows - loaded
        assert held_back, "expected held-back rows (measured 10 on the EPOCH cohort)"

        next_run = _fv_keys(_build(NEXT_MONDAY, NEXT_CRON_AS_OF)["feature_values"])
        missing = held_back - next_run
        assert not missing, (
            f"{len(missing)} held-back rows never reappeared — the self-heal "
            f"contract (held back, regenerated next run) is broken: "
            f"{sorted(missing)[:3]}"
        )

    def test_held_back_rows_byte_identical_beyond_key(self):
        """value + freshness_status must also match — a clamp/shift fix would
        break this (and duplicate rows in the DB instead of dedup)."""
        first = _fv_keyed(_build(FRONTIER_MONDAY, FAR_AS_OF)["feature_values"])
        next_run = _fv_keyed(_build(NEXT_MONDAY, NEXT_CRON_AS_OF)["feature_values"])
        key_cols = ["feature_id", "entity_values", "event_timestamp"]
        merged = first.merge(next_run, on=key_cols, suffixes=("_a", "_b"))
        assert len(merged) == len(first)  # every first-run row regenerates
        assert merged["value_a"].equals(merged["value_b"])
        assert merged["freshness_status_a"].equals(merged["freshness_status_b"])


class TestAsOfDeterminism:
    def test_same_frontier_and_as_of_twice_is_identical(self):
        """Deterministic given (frontier, as_of) — modulo the uuid4 ``id``
        column, which the DB resolves via the natural-key upsert."""
        a = build_frontier_datasets(
            frontier=FRONTIER_MONDAY,
            as_of=CRON_AS_OF,
            include_coverage=False,
            hcp_frame_factory=_small_hcp,
        )
        b = _build(FRONTIER_MONDAY, CRON_AS_OF)
        assert set(a) == set(b)
        for table in a:
            df_a, df_b = a[table], b[table]
            if table == "feature_values":
                df_a = df_a.drop(columns=["id"])
                df_b = df_b.drop(columns=["id"])
            pd.testing.assert_frame_equal(df_a, df_b), table


class TestAsOfRegression:
    """A fully-past frontier must behave exactly as before #1577."""

    def test_none_as_of_matches_far_future_for_past_frontier(self):
        """as_of=None resolves to now(); for a frontier weeks in the past the
        instant filter is a no-op, so output matches a far-future as_of —
        i.e. matches the pre-#1577 date-only behavior (backdated manual runs
        keep loading frontier-day 23:59 rows, which are all in the past)."""
        none_run = _build(FRONTIER_MONDAY, None)
        far_run = _build(FRONTIER_MONDAY, FAR_AS_OF)
        assert set(none_run) == set(far_run)
        for table in none_run:
            df_a, df_b = none_run[table], far_run[table]
            if table == "feature_values":
                df_a = df_a.drop(columns=["id"])
                df_b = df_b.drop(columns=["id"])
            pd.testing.assert_frame_equal(df_a, df_b), table

    def test_as_of_touches_only_feature_values(self):
        """Scope pin: feature_values is the ONLY table with a DB now()-CHECK;
        triggers/ml_predictions/agent_activities carry intraday times too but
        have no constraint and no measured harm — they must pass through
        untouched (opt-in via INSTANT_COLUMNS if that ever changes)."""
        cron_run = _build(FRONTIER_MONDAY, CRON_AS_OF)
        far_run = _build(FRONTIER_MONDAY, FAR_AS_OF)
        assert set(cron_run) == set(far_run)
        for table in far_run:
            if table == "feature_values":
                continue
            pd.testing.assert_frame_equal(cron_run[table], far_run[table]), table

    def test_instant_registry_scope_is_exactly_feature_values(self):
        assert fa.INSTANT_COLUMNS == {"feature_values": "event_timestamp"}
