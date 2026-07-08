"""Phase 2: cohort loader / provider-selection (async).

``build_cohort_provider_or_none`` decides whether a simulation uses the
cohort-estimated effect or falls back to the synthetic uplift. It must NEVER
raise (a DB/shape problem degrades to None → synthetic), and must only return a
provider for a cohort-estimable intervention with enough usable rows.
``cohort_treatment_availability`` reports the same gate PER intervention (it
drives ``available_for_effect`` in ``GET /digital-twin/intervention-types``).
"""

import numpy as np

from src.digital_twin.effect.cohort_loader import (
    build_cohort_provider_or_none,
    cohort_treatment_availability,
)
from src.digital_twin.effect.provider import (
    COHORT_ESTIMABLE_INTERVENTIONS,
    COHORT_MIN_ROWS,
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
            "conversion_rate": float(conv[i]),
            # Pre-treatment confounders the direct estimator/gate now require.
            "market_share": float(market[i]),
            "total_rx_count": float(total_rx[i]),
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
