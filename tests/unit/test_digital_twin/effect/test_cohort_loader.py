"""Phase 2: cohort loader / provider-selection (async).

``build_cohort_provider_or_none`` decides whether a simulation uses the
cohort-estimated effect or falls back to the synthetic uplift. It must NEVER
raise (a DB/shape problem degrades to None → synthetic), and must only return a
provider for a cohort-estimable intervention with enough usable rows.
``cohort_treatment_availability`` reports the same gate PER intervention (it
drives ``available_for_effect`` in ``GET /digital-twin/intervention-types``).
"""

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect import cohort_loader
from src.digital_twin.effect.cohort_loader import (
    build_cohort_provider_or_none,
    cohort_treatment_availability,
    flatten_specialty_relation,
)
from src.digital_twin.effect.errors import EffectCause
from src.digital_twin.effect.provider import (
    COHORT_CONFOUNDERS,
    COHORT_ESTIMABLE_INTERVENTIONS,
    COHORT_MIN_ROWS,
    INTERVENTION_TREATMENT_MAP,
    CohortEffectDataProvider,
)


class _FakeResult:
    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class _FakeQuery:
    """Chainable stand-in for the supabase-py query builder, per TABLE.

    It honours what the loader relies on the server for: ``eq`` filters, ``limit``,
    ``count="exact"`` and ``range`` windows capped at the server's max-rows (PostgREST's
    1,000 on Supabase), so a loader that does not page reads a truncated table here too.
    """

    def __init__(self, client, table):
        self._client = client
        self.table = table
        self.filters: dict = {}
        self.orders: list = []
        self.window = None
        self.row_limit = None
        self.count_mode = None
        self.not_null: list = []

    def select(self, _columns, count=None):
        self.count_mode = count
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def is_(self, column, _value):
        self.not_null.append(column)
        return self

    @property
    def not_(self):
        return self

    def order(self, column, **_kwargs):
        self.orders.append(column)
        return self

    def range(self, start, end):
        self.window = (start, end)
        return self

    def limit(self, n):
        self.row_limit = n
        return self

    async def execute(self):
        self._client.calls.append(self)
        if self.table in self._client.raise_on:
            raise RuntimeError("db unreachable")
        rows = [
            r
            for r in self._client.tables.get(self.table, [])
            if all(r.get(k, v) == v for k, v in self.filters.items())
        ]
        count = len(rows) if self.count_mode == "exact" else None
        if self.orders:
            rows = sorted(rows, key=lambda r: tuple(str(r.get(c)) for c in self.orders))
        if self.window is not None:
            start, end = self.window
            rows = rows[start : min(end + 1, start + self._client.max_rows)]
        else:
            rows = rows[: min(self.row_limit or len(rows), self._client.unranged_cap)]
        return _FakeResult(data=rows, count=count)


class _FakeClient:
    """Two tables: the ``business_metrics`` rollups and ``hcp_brand_adoption``."""

    def __init__(
        self,
        rollups=(),
        adoption=None,
        *,
        raise_on=(),
        max_rows=1000,
        unranged_cap=20000,
    ):
        self.tables = {
            "business_metrics": list(rollups),
            "hcp_brand_adoption": list(_adoption_rows(rollups) if adoption is None else adoption),
        }
        self.raise_on = set(raise_on)
        self.max_rows = max_rows
        self.unranged_cap = unranged_cap
        self.calls: list = []

    def table(self, name):
        return _FakeQuery(self, name)

    def reads(self, table):
        return [q for q in self.calls if q.table == table]


def test_embedded_hcp_specialty_is_flattened_without_inventing_missing_values():
    rows = [
        {"hcp_id": "h1", "hcp_profiles": {"specialty": " oncology "}},
        {"hcp_id": "h2", "hcp_profiles": {"specialty": ""}},
        {"hcp_id": "h3", "hcp_profiles": None},
    ]

    frame = flatten_specialty_relation(pd.DataFrame(rows))

    assert frame["specialty"].tolist()[:1] == ["oncology"]
    assert frame["specialty"].isna().tolist() == [False, True, True]
    assert "hcp_profiles" not in frame.columns


BRAND = "Remibrutinib"


def _cohort_rows(n: int = 600, seed: int = 0, *, with_all_channels: bool = False):
    """``per_hcp_rollup`` rows: one per HCP ``h0000..``, ``brand`` and ``metric_date`` set,
    and NO outcome — the twin's outcome is on ``hcp_brand_adoption`` (see _adoption_rows)."""
    rng = np.random.default_rng(seed)
    regions = rng.choice(["northeast", "south", "midwest", "west"], size=n)
    eng = rng.uniform(0, 10, size=n)
    market = rng.uniform(0, 1, size=n)
    total_rx = rng.poisson(lam=80, size=n).astype(float)
    rows = [
        {
            "hcp_id": f"h{i:04d}",
            "brand": BRAND,
            "metric_date": "2026-06-01",
            "region": str(regions[i]),
            "engagement_score": float(eng[i]),
            "call_frequency": float(rng.uniform(0, 14)),
            # Pre-treatment confounders the direct estimator/gate now require.
            "market_share": float(market[i]),
            "triggers_total_count": float(total_rx[i]),
        }
        for i in range(n)
    ]
    if with_all_channels:
        for row in rows:
            row.update(
                {
                    "email_campaign_count": float(rng.poisson(6)),
                    "speaker_program_count": float(rng.poisson(2)),
                    "sample_volume": float(rng.poisson(15)),
                    "peer_influence_score": float(rng.uniform(0, 10)),
                    "patient_support_enrollment": float(rng.uniform(0, 1)),
                    "rep_training_score": float(rng.uniform(0, 10)),
                }
            )
    return rows


def _adoption_rows(rollups, seed: int = 1):
    """One ``hcp_brand_adoption`` row per (hcp_id, brand) in ``rollups``, adopted 0/1."""
    rng = np.random.default_rng(seed)
    pairs = sorted({(r["hcp_id"], r["brand"]) for r in rollups})
    return [{"hcp_id": h, "brand": b, "adopted": int(rng.random() < 0.4)} for h, b in pairs]


async def test_returns_provider_for_estimable_intervention_with_cohort():
    client = _FakeClient(_cohort_rows(600))
    provider = await build_cohort_provider_or_none(client, "digital_engagement", "Remibrutinib")
    assert isinstance(provider, CohortEffectDataProvider)


async def test_returns_provider_for_new_channel_when_column_present():
    # Revision-2 channel (email_campaign_count planted) → cohort-estimable.
    client = _FakeClient(_cohort_rows(600, with_all_channels=True))
    provider = await build_cohort_provider_or_none(client, "email_campaign", "Remibrutinib")
    assert isinstance(provider, CohortEffectDataProvider)


async def test_returns_none_for_unknown_intervention():
    # Not in the catalog/treatment map → never cohort-estimated.
    client = _FakeClient(_cohort_rows(600, with_all_channels=True))
    provider = await build_cohort_provider_or_none(client, "not_a_real_lever", "Remibrutinib")
    assert provider is None


async def test_returns_none_when_treatment_column_missing():
    # email_campaign is estimable, but this cohort lacks its planted column
    # (e.g. migration applied, backfill not yet run) → honest None, not a guess.
    client = _FakeClient(_cohort_rows(600))
    provider = await build_cohort_provider_or_none(client, "email_campaign", "Remibrutinib")
    assert provider is None


async def test_returns_none_on_empty_cohort():
    client = _FakeClient([])
    provider = await build_cohort_provider_or_none(client, "digital_engagement", "Fabhalta")
    assert provider is None


async def test_returns_none_on_insufficient_rows():
    client = _FakeClient(_cohort_rows(COHORT_MIN_ROWS - 50))
    provider = await build_cohort_provider_or_none(client, "call_frequency_increase", "Kisqali")
    assert provider is None


async def test_returns_none_on_db_error_never_raises():
    client = _FakeClient(_cohort_rows(600), raise_on={"business_metrics"})
    provider = await build_cohort_provider_or_none(client, "digital_engagement", "Remibrutinib")
    assert provider is None


def _frame(n: int = 600) -> pd.DataFrame:
    """The joined frame ``load_cohort_frame`` returns: rollups + ``adopted``."""
    rows = _cohort_rows(n)
    adopted = {a["hcp_id"]: a["adopted"] for a in _adoption_rows(rows)}
    return pd.DataFrame([{**r, "adopted": adopted[r["hcp_id"]]} for r in rows])


def _all_null_treatment() -> pd.DataFrame:
    frame = _frame()
    frame["engagement_score"] = np.nan
    return frame


def _columns(n_rows=600, *, treatment=True, outcome=True, region=True, missing_confounders=0):
    return {
        "n_rows": n_rows,
        "has_treatment_column": treatment,
        "has_outcome_column": outcome,
        "has_region_column": region,
        "n_missing_confounder_columns": missing_confounders,
    }


def _rows(n_rows, n_usable, *, null_treatment=0):
    return {
        "n_rows": n_rows,
        "n_usable_rows": n_usable,
        "n_min_usable_rows": COHORT_MIN_ROWS,
        "n_null_treatment_rows": null_treatment,
        "n_null_outcome_rows": 0,
        "n_null_region_rows": 0,
        "n_null_confounder_rows": 0,
    }


@pytest.mark.parametrize(
    ("build", "intervention", "cause", "details"),
    [
        pytest.param(
            pd.DataFrame, "digital_engagement", EffectCause.EMPTY_COHORT, {"n_rows": 0}, id="empty"
        ),
        # ``DataFrame.empty`` is also true for rows without columns; those rows are not an
        # empty cohort, they are a cohort missing every column.
        pytest.param(
            lambda: pd.DataFrame(index=range(5)),
            "digital_engagement",
            EffectCause.REQUIRED_COLUMN_MISSING,
            _columns(
                5,
                treatment=False,
                outcome=False,
                region=False,
                missing_confounders=len(COHORT_CONFOUNDERS),
            ),
            id="rows-without-columns",
        ),
        pytest.param(
            lambda: _frame().drop(columns="engagement_score"),
            "digital_engagement",
            EffectCause.REQUIRED_COLUMN_MISSING,
            _columns(treatment=False),
            id="no-treatment-column",
        ),
        pytest.param(
            lambda: _frame().drop(columns="market_share"),
            "digital_engagement",
            EffectCause.REQUIRED_COLUMN_MISSING,
            _columns(missing_confounders=1),
            id="no-confounder-column",
        ),
        # Before #2021 9b this raised KeyError from dropna: the outcome was never checked.
        pytest.param(
            lambda: _frame().drop(columns="adopted"),
            "digital_engagement",
            EffectCause.REQUIRED_COLUMN_MISSING,
            _columns(outcome=False),
            id="no-outcome-column",
        ),
        pytest.param(
            lambda: _frame().drop(columns="region"),
            "digital_engagement",
            EffectCause.REQUIRED_COLUMN_MISSING,
            _columns(region=False),
            id="no-region-column",
        ),
        pytest.param(
            lambda: _frame().head(100),
            "digital_engagement",
            EffectCause.TOO_FEW_USABLE_ROWS,
            _rows(100, 100),
            id="too-few-rows",
        ),
        pytest.param(
            _all_null_treatment,
            "digital_engagement",
            EffectCause.TOO_FEW_USABLE_ROWS,
            _rows(600, 0, null_treatment=600),
            id="all-null-treatment",
        ),
        pytest.param(
            _frame,
            "not_a_lever",
            EffectCause.INTERVENTION_NOT_IDENTIFIED,
            {},
            id="not-identified",
        ),
    ],
)
def test_an_unusable_cohort_names_its_cause(build, intervention, cause, details):
    """#2021 9b: one ``None`` used to cover every cause; the refusal's code needs to know which."""
    usability = cohort_loader.assess_cohort_frame(build(), intervention)
    assert usability.provider is None
    assert usability.cause is cause
    assert dict(usability.details) == details


def test_a_usable_cohort_gets_a_provider_and_no_cause():
    frame = _frame()
    usability = cohort_loader.assess_cohort_frame(frame, "digital_engagement")
    assert isinstance(usability.provider, CohortEffectDataProvider)
    assert len(usability.provider._cohort) == len(frame)
    assert usability.cause is None
    assert dict(usability.details) == {}


def test_the_optional_wrapper_refuses_a_cohort_without_its_outcome_column():
    frame = _frame().drop(columns="adopted")
    assert cohort_loader.cohort_provider_from_frame(frame, "digital_engagement") is None


async def test_availability_true_per_intervention_when_the_joined_cohort_is_usable():
    client = _FakeClient(_cohort_rows(600, with_all_channels=True))
    availability = await cohort_treatment_availability(client, BRAND)
    # One entry per catalog intervention; all usable on 600 joined HCPs.
    assert set(availability) == set(COHORT_ESTIMABLE_INTERVENTIONS)
    assert all(availability.values())
    assert availability.n_probe_errors == 0


async def test_availability_false_below_threshold_and_on_error():
    below = await cohort_treatment_availability(
        _FakeClient(_cohort_rows(COHORT_MIN_ROWS - 50, with_all_channels=True)), BRAND
    )
    assert set(below) == set(COHORT_ESTIMABLE_INTERVENTIONS)
    assert not any(below.values())
    erroring = await cohort_treatment_availability(
        _FakeClient(_cohort_rows(600), raise_on={"business_metrics"}), BRAND
    )
    assert not any(erroring.values())


@pytest.mark.asyncio
async def test_availability_tells_a_failed_probe_from_a_measured_shortfall():
    """codex r1 MEDIUM: both read all-False. Only one of them means the cohort data is gone; the
    other is a connection blip, and treating it as data loss recommends a production write."""
    measured = await cohort_treatment_availability(
        _FakeClient(_cohort_rows(10, with_all_channels=True)), BRAND
    )
    errored = await cohort_treatment_availability(
        _FakeClient(_cohort_rows(600), raise_on={"hcp_brand_adoption"}), BRAND
    )
    assert not any(measured.values()) and not any(errored.values())
    assert measured.n_probe_errors == 0
    assert errored.n_probe_errors == len(set(INTERVENTION_TREATMENT_MAP.values()))


# ---------------------------------------------------------------------------------------------
# Lane T2: the outcome is hcp_brand_adoption.adopted, joined per (hcp_id, brand)
# ---------------------------------------------------------------------------------------------


def _multi_row_rollups(n_hcps: int = 700):
    """Two metric_date rows for every even HCP (the live table has 1-6 per pair)."""
    rows = []
    for row in _cohort_rows(n_hcps, with_all_channels=True):
        rows.append(row)
        if int(row["hcp_id"][1:]) % 2 == 0:
            rows.append({**row, "metric_date": "2026-07-01", "engagement_score": 0.0})
    return rows


async def test_the_frame_is_one_row_per_hcp_brand_joined_to_adopted():
    rollups = _multi_row_rollups()
    adoption = _adoption_rows(rollups)
    frame = await cohort_loader.load_cohort_frame(_FakeClient(rollups, adoption), BRAND)

    assert len(rollups) == 1050 and len(frame) == 700
    assert not frame.duplicated(["hcp_id", "brand"]).any()
    # The outcome is `adopted`, 0/1, from the adoption table, row-for-row by (hcp, brand).
    by_hcp = {a["hcp_id"]: a["adopted"] for a in adoption}
    assert frame.set_index("hcp_id")["adopted"].to_dict() == by_hcp
    assert "cohort_conversion_outcome" not in frame.columns
    # The collapse is the backfill's: scores MEAN over the pair's rows, counts SUM.
    h0 = [r for r in rollups if r["hcp_id"] == "h0000"]
    got = frame.set_index("hcp_id").loc["h0000"]
    assert got["engagement_score"] == pytest.approx(np.mean([r["engagement_score"] for r in h0]))
    assert got["triggers_total_count"] == pytest.approx(sum(r["triggers_total_count"] for r in h0))
    assert got["n_metric_rows"] == 2
    assert str(got["max_metric_date"])[:10] == "2026-07-01"


async def test_the_adoption_read_is_paged_past_the_server_row_cap_in_a_total_order():
    """PostgREST caps an un-ranged read at max-rows (1,000); the adoption table holds 5,000 rows
    per brand live. An un-paged read would silently join a fifth of the cohort."""
    rollups = _cohort_rows(2500, with_all_channels=True)
    client = _FakeClient(rollups, max_rows=1000)
    frame = await cohort_loader.load_cohort_frame(client, BRAND)

    assert frame["adopted"].notna().sum() == 2500
    reads = client.reads("hcp_brand_adoption")
    assert [q.window for q in reads] == [(0, 999), (1000, 1999), (2000, 2999)]
    assert all(q.orders == ["hcp_id", "brand"] for q in reads)
    assert all(q.filters == {"brand": BRAND, "is_synthetic": True} for q in reads)


async def test_a_short_adoption_read_fails_loud_rather_than_joining_a_subset():
    class _ShortCount(_FakeClient):
        def table(self, name):
            query = super().table(name)
            if name != "hcp_brand_adoption":
                return query
            real = query.execute

            async def execute():
                result = await real()
                result.count = (result.count or 0) + 1  # server says one more row exists
                return result

            query.execute = execute
            return query

    with pytest.raises(RuntimeError, match="hcp_brand_adoption"):
        await cohort_loader.load_cohort_frame(_ShortCount(_cohort_rows(600)), BRAND)


async def test_rollup_rows_that_carry_no_channel_are_not_collapsed_into_the_confounders():
    """Measured live 2026-09-28: 219 per_hcp_rollup rows dated after the plant (the daily ETL)
    carry NO channel; 168 of their pairs also have planted rows. Summing their trigger counts
    and averaging their market share into the pair would give the estimator confounders the
    DGP never saw. Only rows carrying at least one channel are collapsed."""
    rollups = _cohort_rows(600, with_all_channels=True)
    planted_h0 = dict(rollups[0])
    etl_row = {
        k: v for k, v in planted_h0.items() if k not in set(INTERVENTION_TREATMENT_MAP.values())
    }
    etl_row.update({"metric_date": "2026-09-28", "triggers_total_count": 5000.0})
    only_etl = {**etl_row, "hcp_id": "h9999"}
    frame = await cohort_loader.load_cohort_frame(
        _FakeClient([*rollups, etl_row, only_etl], _adoption_rows([*rollups, only_etl])),
        BRAND,
    )
    h0 = frame.set_index("hcp_id").loc["h0000"]
    assert h0["triggers_total_count"] == planted_h0["triggers_total_count"]
    assert h0["n_metric_rows"] == 1
    assert str(h0["max_metric_date"])[:10] == "2026-06-01"
    assert "h9999" not in set(frame["hcp_id"])


async def test_only_synthetic_gold_rows_are_read_on_both_tables():
    """codex r1 #1: the estimate is labelled synthetic-gold (PROVENANCE_COHORT), so a REAL row
    on either table must never be collapsed or joined into it (0 real rows live today)."""
    rollups = [{**r, "is_synthetic": True} for r in _cohort_rows(600, with_all_channels=True)]
    adoption = [{**a, "is_synthetic": True} for a in _adoption_rows(rollups)]
    real_rollup = {**rollups[0], "is_synthetic": False, "triggers_total_count": 9999.0}
    real_only = {**rollups[1], "hcp_id": "h8888", "is_synthetic": False}
    real_adoption = {"hcp_id": "h8888", "brand": BRAND, "adopted": 1, "is_synthetic": False}
    client = _FakeClient([*rollups, real_rollup, real_only], [*adoption, real_adoption])
    frame = await cohort_loader.load_cohort_frame(client, BRAND)

    assert "h8888" not in set(frame["hcp_id"])
    h0 = frame.set_index("hcp_id").loc["h0000"]
    assert h0["triggers_total_count"] == rollups[0]["triggers_total_count"]
    assert all(q.filters.get("is_synthetic") is True for q in client.calls)


@pytest.mark.parametrize("table", ["business_metrics", "hcp_brand_adoption"])
async def test_a_read_without_an_exact_count_fails_loud(table):
    """codex r1 #3: completeness is checked against the server's count; no count, no check,
    so the read refuses rather than trusting it."""

    class _NoCount(_FakeClient):
        def table(self, name):
            query = super().table(name)
            if name != table:
                return query
            real = query.execute

            async def execute():
                result = await real()
                result.count = None
                return result

            query.execute = execute
            return query

    with pytest.raises(RuntimeError, match=table):
        await cohort_loader.load_cohort_frame(_NoCount(_cohort_rows(600)), BRAND)


async def test_a_cohort_larger_than_the_rollup_read_fails_loud(monkeypatch):
    """codex r1 #3: at more rows than ``_FETCH_LIMIT`` the one read returns exactly the limit,
    which the old check accepted; an arbitrary subset would then be collapsed."""
    monkeypatch.setattr(cohort_loader, "_FETCH_LIMIT", 500)
    with pytest.raises(RuntimeError, match="business_metrics"):
        await cohort_loader.load_cohort_frame(_FakeClient(_cohort_rows(600)), BRAND)


async def test_a_truncated_rollup_read_fails_loud_rather_than_collapsing_a_subset():
    """The rollup read relies on the server honouring ``limit(_FETCH_LIMIT)``; measured live it
    does (4.6k rows per brand, 2026-09-23). If a server cap ever cut it short, the collapse
    would silently drop HCPs, so the read checks itself against the exact count."""
    client = _FakeClient(_cohort_rows(600), unranged_cap=500)
    with pytest.raises(RuntimeError, match="business_metrics"):
        await cohort_loader.load_cohort_frame(client, BRAND)


async def test_an_unjoined_pair_has_no_outcome_and_is_counted_as_such():
    """A rollup pair without an adoption row cannot be estimated on: it reads as a NULL
    outcome, so the refusal says the OUTCOME is what is missing."""
    rollups = _cohort_rows(600, with_all_channels=True)
    adoption = _adoption_rows(rollups)[:400]
    frame = await cohort_loader.load_cohort_frame(_FakeClient(rollups, adoption), BRAND)
    usability = cohort_loader.assess_cohort_frame(frame, "digital_engagement")

    assert usability.provider is None
    assert usability.cause is EffectCause.TOO_FEW_USABLE_ROWS
    assert usability.details["n_usable_rows"] == 400
    assert usability.details["n_null_outcome_rows"] == 200


async def test_availability_is_joined_coverage_not_a_rollup_null_count():
    """Every rollup column is non-null on all 600 HCPs, so the old per-column count gate said
    usable; only 400 of them have an outcome, which is below COHORT_MIN_ROWS."""
    rollups = _cohort_rows(600, with_all_channels=True)
    client = _FakeClient(rollups, _adoption_rows(rollups)[:400])
    availability = await cohort_treatment_availability(client, BRAND)
    assert not any(availability.values())
    assert availability.n_probe_errors == 0


@pytest.mark.parametrize("n_adopted", [400, 600])
async def test_availability_advertises_exactly_what_simulate_accepts(n_adopted):
    """``/intervention-types`` and ``/simulate`` answer from the same frame and the same rule."""
    rollups = _cohort_rows(600, with_all_channels=True)
    rollups[0]["rep_training_score"] = None  # one channel short of a row
    adoption = _adoption_rows(rollups)[:n_adopted]
    availability = await cohort_treatment_availability(_FakeClient(rollups, adoption), BRAND)
    for intervention in COHORT_ESTIMABLE_INTERVENTIONS:
        provider = await build_cohort_provider_or_none(
            _FakeClient(rollups, adoption), intervention, BRAND
        )
        assert availability[intervention] is (provider is not None), intervention


@pytest.mark.parametrize(
    "values",
    [[0.0], [0.0, 5.0, 5.0, 5.0], [0.0] * 599 + [1.0]],
    ids=["constant", "tied-at-max", "one-row-arm"],
)
async def test_a_channel_without_a_median_contrast_is_not_advertised(values):
    """codex r2: 600 usable rows of a channel that never splits at its median (constant, or so
    tied that every row sits at or below it) pass a non-null count, but the estimator refuses
    them (NO_TREATMENT_CONTRAST). The gate applies the estimator's own split, so the channel
    is not advertised and /simulate is not left to 422."""
    rollups = _cohort_rows(600, with_all_channels=True)
    for i, row in enumerate(rollups):
        row["email_campaign_count"] = values[i % len(values)] if len(values) > 1 else values[0]
    usability = cohort_loader.assess_cohort_frame(
        await cohort_loader.load_cohort_frame(_FakeClient(rollups), BRAND), "email_campaign"
    )
    assert usability.provider is None
    assert usability.cause is EffectCause.NO_TREATMENT_CONTRAST
    assert usability.details["n_usable_rows"] == 600

    availability = await cohort_treatment_availability(_FakeClient(rollups), BRAND)
    assert availability["email_campaign"] is False
    assert availability["digital_engagement"] is True


async def test_availability_reads_the_cohort_once_for_all_channels():
    """Eight channels, one merged read: one rollup read + the adoption pages, not 8x that."""
    client = _FakeClient(_cohort_rows(600, with_all_channels=True))
    await cohort_treatment_availability(client, BRAND)
    assert len(client.reads("business_metrics")) == 1
    assert len(client.reads("hcp_brand_adoption")) == 1


def test_blocking_build_opens_and_closes_its_own_client_on_each_call(monkeypatch):
    """#2025: the synchronous callers run the load under ``asyncio.run``. The platform's
    cached async client keeps an httpx pool bound to the first loop that used it, so the
    next ``asyncio.run`` in the process fails with "Event loop is closed" (measured on the
    box against the live database: run 1 ok, run 2 failed, run 3 ok). Each blocking build
    therefore opens a client for its own loop, closes it, and leaves the cache alone."""
    from src.memory.services import factories

    monkeypatch.setenv("SUPABASE_URL", "http://supabase.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setattr(factories, "_async_supabase_client", None)
    created = []

    async def _acreate_client(url, key, options=None):
        created.append({"url": url, "key": key, "options": options})
        return _FakeClient(_cohort_rows(600, with_all_channels=True))

    monkeypatch.setattr("supabase.acreate_client", _acreate_client)
    open_during_load = []
    real_load = cohort_loader.load_cohort_frame

    async def _load(client, brand):
        open_during_load.append(not created[-1]["options"].httpx_client.is_closed)
        return await real_load(client, brand)

    monkeypatch.setattr(cohort_loader, "load_cohort_frame", _load)

    for _ in range(2):
        provider = cohort_loader.build_cohort_provider_or_none_blocking(
            "email_campaign", "Remibrutinib"
        )
        assert isinstance(provider, CohortEffectDataProvider)

    assert [(c["url"], c["key"]) for c in created] == [
        ("http://supabase.invalid", "service-key")
    ] * 2
    assert open_during_load == [True, True]
    assert all(c["options"].httpx_client.is_closed for c in created)
    assert factories._async_supabase_client is None


def test_blocking_build_is_none_when_no_client_can_be_opened(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    assert (
        cohort_loader.build_cohort_provider_or_none_blocking("email_campaign", "Remibrutinib")
        is None
    )
