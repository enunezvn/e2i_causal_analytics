"""Unit: cohort-outcome resolver returns a runnable var-set per cohort.

Hermetic — a fake client returns real-shaped rows; the resolver's column-selection
LOGIC runs (no DB). Fail-closed on unknown cohort / brand is asserted directly.
"""
from __future__ import annotations

import pytest

from src.services import cohort_resolution as cr


class _FakeResp:
    def __init__(self, rows):
        self.data = rows


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return _FakeResp(self._rows)


class _FakeClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, *a, **k):
        return _FakeQuery(self._rows)


_PJ_ROWS = [
    {"patient_id": f"pt_{i}", "brand": "Kisqali", "geographic_region": "northeast",
     "disease_severity": 6.0, "academic_hcp": 1, "treatment_arm": i % 2,
     "discontinued_180d": int(i % 3 == 0), "persistent_180d": int(i % 3 != 0),
     "treatment_initiated": 1, "is_synthetic": True}
    for i in range(30)
]


@pytest.mark.parametrize("cohort,outcome", [
    ("discontinuation", "discontinued_180d"),
    ("persistence", "persistent_180d"),
    ("initiation", "treatment_initiated"),
])
def test_resolver_returns_runnable_varset(cohort, outcome):
    spec = cr.resolve_cohort_outcome_frame(
        cohort, brand="Kisqali", region="northeast",
        supabase_client=_FakeClient(_PJ_ROWS),
    )
    assert spec is not None
    assert spec.outcome_column == outcome
    assert spec.outcome_column in spec.frame.columns
    assert spec.treatment_column in spec.frame.columns
    assert len(spec.covariate_columns) >= 1
    assert all(c in spec.frame.columns for c in spec.covariate_columns)


def test_unknown_cohort_fails_closed():
    assert cr.resolve_cohort_outcome_frame(
        "not_a_cohort", brand="Kisqali", region=None,
        supabase_client=_FakeClient(_PJ_ROWS)) is None


def test_unrecognized_brand_fails_closed():
    assert cr.resolve_cohort_outcome_frame(
        "discontinuation", brand="NotABrand", region=None,
        supabase_client=_FakeClient(_PJ_ROWS)) is None


_HCP_ROWS = [
    {"hcp_id": f"h_{i}", "peer_influence_score": 3.0 + (i % 5),
     "influence_network_size": 10 + i,
     "adoption_category": "ADOPTER" if i % 3 else "NON_ADOPTER",
     "geographic_region": "northeast"}
    for i in range(20)
]


def test_hcp_adoption_resolves_runnable_varset():
    spec = cr.resolve_cohort_outcome_frame(
        "hcp_adoption", brand="Kisqali", region="northeast",
        supabase_client=_FakeClient(_HCP_ROWS))
    assert spec is not None
    assert spec.outcome_column == "adoption_category"
    assert spec.treatment_column == "peer_influence_score"
    assert spec.treatment_column in spec.frame.columns


def test_hcp_adoption_unrecognized_region_fails_closed():
    # codex #06 HIGH FINDING-1: region must fail closed for the hcp grain too.
    assert cr.resolve_cohort_outcome_frame(
        "hcp_adoption", brand="Kisqali", region="NotARegion",
        supabase_client=_FakeClient(_HCP_ROWS)) is None


def test_hcp_adoption_missing_treatment_fails_closed():
    # codex #06 MED FINDING-2: no spec when peer_influence_score is absent.
    rows = [{"hcp_id": "h1", "adoption_category": "ADOPTER",
             "influence_network_size": 10, "geographic_region": "northeast"}]
    assert cr.resolve_cohort_outcome_frame(
        "hcp_adoption", brand="Kisqali", region=None,
        supabase_client=_FakeClient(rows)) is None


def test_persistence_cohort_exposes_non_negative_retention_benefit():
    # codex #06 MED FINDING-3: retention_benefit recomputed at resolve time.
    spec = cr.resolve_cohort_outcome_frame(
        "persistence", brand="Kisqali", region="northeast",
        supabase_client=_FakeClient(_PJ_ROWS))
    assert spec is not None
    assert "retention_benefit" in spec.covariate_columns
    assert (spec.frame["retention_benefit"] >= 0).all()
