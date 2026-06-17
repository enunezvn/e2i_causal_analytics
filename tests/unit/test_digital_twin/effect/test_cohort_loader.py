"""Phase 2: cohort loader / provider-selection (async).

``build_cohort_provider_or_none`` decides whether a simulation uses the
cohort-estimated effect or falls back to the synthetic uplift. It must NEVER
raise (a DB/shape problem degrades to None → synthetic), and must only return a
provider for a cohort-estimable intervention with enough usable rows.
"""

import numpy as np

from src.digital_twin.effect.cohort_loader import (
    brand_has_cohort,
    build_cohort_provider_or_none,
)
from src.digital_twin.effect.provider import COHORT_MIN_ROWS, CohortEffectDataProvider


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


def _cohort_rows(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    regions = rng.choice(["northeast", "south", "midwest", "west"], size=n)
    eng = rng.uniform(0, 10, size=n)
    conv = 0.4 + 0.06 * eng + rng.normal(0, 0.08, size=n)
    return [
        {
            "region": str(regions[i]),
            "engagement_score": float(eng[i]),
            "call_frequency": float(rng.uniform(0, 14)),
            "conversion_rate": float(conv[i]),
        }
        for i in range(n)
    ]


async def test_returns_provider_for_estimable_intervention_with_cohort():
    client = _FakeClient(_FakeResult(data=_cohort_rows(600)))
    provider = await build_cohort_provider_or_none(client, "digital_engagement", "Remibrutinib")
    assert isinstance(provider, CohortEffectDataProvider)


async def test_returns_none_for_non_estimable_intervention():
    # email_campaign has no cohort treatment column → never cohort-estimated.
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


async def test_brand_has_cohort_true_when_count_meets_threshold():
    client = _FakeClient(_FakeResult(count=COHORT_MIN_ROWS + 10))
    assert await brand_has_cohort(client, "Remibrutinib") is True


async def test_brand_has_cohort_false_below_threshold_and_on_error():
    assert await brand_has_cohort(_FakeClient(_FakeResult(count=10)), "X") is False
    assert (
        await brand_has_cohort(_FakeClient(_FakeResult(count=None), raise_on_execute=True), "X")
        is False
    )
