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
    """Chainable stand-in for the supabase-py query builder."""

    def __init__(self, result, *, raise_on_execute=False):
        self._result = result
        self._raise = raise_on_execute

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def is_(self, *a, **k):
        return self

    @property
    def not_(self):
        return self

    async def execute(self):
        if self._raise:
            raise RuntimeError("db unreachable")
        return self._result


class _FakeClient:
    def __init__(self, result, *, raise_on_execute=False):
        self._result = result
        self._raise = raise_on_execute

    def table(self, *a, **k):
        return _FakeQuery(self._result, raise_on_execute=self._raise)


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


def _cohort_rows(n: int = 600, seed: int = 0, *, with_all_channels: bool = False):
    rng = np.random.default_rng(seed)
    regions = rng.choice(["northeast", "south", "midwest", "west"], size=n)
    eng = rng.uniform(0, 10, size=n)
    conv = 0.4 + 0.06 * eng + rng.normal(0, 0.08, size=n)
    market = rng.uniform(0, 1, size=n)
    total_rx = rng.poisson(lam=80, size=n).astype(float)
    rows = [
        {
            "region": str(regions[i]),
            "engagement_score": float(eng[i]),
            "call_frequency": float(rng.uniform(0, 14)),
            "cohort_conversion_outcome": float(conv[i]),
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


async def test_returns_provider_for_estimable_intervention_with_cohort():
    client = _FakeClient(_FakeResult(data=_cohort_rows(600)))
    provider = await build_cohort_provider_or_none(client, "digital_engagement", "Remibrutinib")
    assert isinstance(provider, CohortEffectDataProvider)


async def test_returns_provider_for_new_channel_when_column_present():
    # Revision-2 channel (email_campaign_count planted) → cohort-estimable.
    client = _FakeClient(_FakeResult(data=_cohort_rows(600, with_all_channels=True)))
    provider = await build_cohort_provider_or_none(client, "email_campaign", "Remibrutinib")
    assert isinstance(provider, CohortEffectDataProvider)


async def test_returns_none_for_unknown_intervention():
    # Not in the catalog/treatment map → never cohort-estimated.
    client = _FakeClient(_FakeResult(data=_cohort_rows(600, with_all_channels=True)))
    provider = await build_cohort_provider_or_none(client, "not_a_real_lever", "Remibrutinib")
    assert provider is None


async def test_returns_none_when_treatment_column_missing():
    # email_campaign is estimable, but this cohort lacks its planted column
    # (e.g. migration applied, backfill not yet run) → honest None, not a guess.
    client = _FakeClient(_FakeResult(data=_cohort_rows(600)))
    provider = await build_cohort_provider_or_none(client, "email_campaign", "Remibrutinib")
    assert provider is None


async def test_returns_none_on_empty_cohort():
    client = _FakeClient(_FakeResult(data=[]))
    provider = await build_cohort_provider_or_none(client, "digital_engagement", "Fabhalta")
    assert provider is None


async def test_returns_none_on_insufficient_rows():
    client = _FakeClient(_FakeResult(data=_cohort_rows(COHORT_MIN_ROWS - 50)))
    provider = await build_cohort_provider_or_none(client, "call_frequency_increase", "Kisqali")
    assert provider is None


async def test_returns_none_on_db_error_never_raises():
    client = _FakeClient(_FakeResult(data=None), raise_on_execute=True)
    provider = await build_cohort_provider_or_none(client, "digital_engagement", "Remibrutinib")
    assert provider is None


def _frame(n: int = 600) -> pd.DataFrame:
    return pd.DataFrame(_cohort_rows(n))


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
            lambda: _frame().drop(columns="cohort_conversion_outcome"),
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
    frame = _frame().drop(columns="cohort_conversion_outcome")
    assert cohort_loader.cohort_provider_from_frame(frame, "digital_engagement") is None


async def test_availability_true_per_intervention_when_counts_meet_threshold():
    client = _FakeClient(_FakeResult(count=COHORT_MIN_ROWS + 10))
    availability = await cohort_treatment_availability(client, "Remibrutinib")
    # One entry per catalog intervention; all usable at this count.
    assert set(availability) == set(COHORT_ESTIMABLE_INTERVENTIONS)
    assert all(availability.values())


async def test_availability_false_below_threshold_and_on_error():
    below = await cohort_treatment_availability(_FakeClient(_FakeResult(count=10)), "X")
    assert set(below) == set(COHORT_ESTIMABLE_INTERVENTIONS)
    assert not any(below.values())
    erroring = await cohort_treatment_availability(
        _FakeClient(_FakeResult(count=None), raise_on_execute=True), "X"
    )
    assert not any(erroring.values())


@pytest.mark.asyncio
async def test_availability_tells_a_failed_probe_from_a_measured_shortfall():
    """codex r1 MEDIUM: both read all-False. Only one of them means the cohort data is gone; the
    other is a connection blip, and treating it as data loss recommends a production write."""
    measured = await cohort_treatment_availability(_FakeClient(_FakeResult(count=10)), "X")
    errored = await cohort_treatment_availability(
        _FakeClient(_FakeResult(count=None), raise_on_execute=True), "X"
    )
    assert not any(measured.values()) and not any(errored.values())
    assert measured.n_probe_errors == 0
    assert errored.n_probe_errors == len(set(INTERVENTION_TREATMENT_MAP.values()))


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
        return _FakeClient(_FakeResult(data=_cohort_rows(600, with_all_channels=True)))

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
