"""Lane T2: a stored simulation records the outcome column its ATE was estimated ON.

The twin's outcome moved from ``cohort_conversion_outcome`` to ``adopted``. A draft experiment
made from a stored run must measure what THAT run predicted, so ``save_simulation`` writes the
run's outcome into the row's ``effect_heterogeneity`` JSON and ``stored_outcome_column`` reads it
back; a row saved earlier resolves by provenance and date.
"""

from __future__ import annotations

import asyncio

import pytest
from uuid import uuid4

from src.digital_twin.models.simulation_models import (
    OUTCOME_COLUMN_KEY,
    EffectHeterogeneity,
    InterventionConfig,
    SimulationRecommendation,
    SimulationResult,
    stored_outcome_column,
)
from src.digital_twin.twin_repository import SimulationRepository


class _Insert:
    def __init__(self, captured):
        self._captured = captured

    def insert(self, row):
        self._captured["row"] = row
        return self

    async def execute(self):
        return None


class _Client:
    def __init__(self):
        self.captured: dict = {}

    def table(self, _name):
        return _Insert(self.captured)


def _result(outcome_column):
    return SimulationResult(
        model_id=uuid4(),
        intervention_config=InterventionConfig(intervention_type="email_campaign"),
        twin_count=10,
        simulated_ate=0.05,
        simulated_ci_lower=0.01,
        simulated_ci_upper=0.09,
        simulated_std_error=0.02,
        effect_heterogeneity=EffectHeterogeneity(by_region={"west": {"ate": 0.05, "n": 10}}),
        recommendation=SimulationRecommendation.DEPLOY,
        recommendation_rationale="CI excludes zero",
        simulation_confidence=0.7,
        execution_time_ms=5,
        outcome_column=outcome_column,
    )


def test_save_simulation_records_the_outcome_beside_the_heterogeneity():
    client = _Client()
    asyncio.run(
        SimulationRepository(supabase_client=client).save_simulation(_result("adopted"), "Kisqali")
    )
    eh = client.captured["row"]["effect_heterogeneity"]
    assert eh[OUTCOME_COLUMN_KEY] == "adopted"
    assert eh["by_region"] == {"west": {"ate": 0.05, "n": 10}}  # the heterogeneity is intact
    assert stored_outcome_column(client.captured["row"]) == "adopted"


def test_a_run_without_an_outcome_records_none():
    client = _Client()
    asyncio.run(
        SimulationRepository(supabase_client=client).save_simulation(_result(None), "Kisqali")
    )
    assert OUTCOME_COLUMN_KEY not in client.captured["row"]["effect_heterogeneity"]


COHORT = "cohort_estimated_synthetic_gold_v1"


@pytest.mark.parametrize(
    ("created_at", "expected"),
    [
        # Measured 2026-09-28 (read-only census of all 30 stored runs): 24 cohort runs from
        # 2026-06-18 to 2026-09-18, all before migration 147 (public.schema_migrations
        # applied_at 2026-09-22 01:04:43Z): the twin's outcome was conversion_rate
        # (cohort_causal_estimator._OUTCOME_COL at every revision until 4b5957189).
        ("2026-09-18T10:32:55.951155+00:00", "conversion_rate"),
        ("2026-09-22T01:04:43+00:00", "conversion_rate"),
        # 3 cohort runs at 2026-09-22 07:00Z, after the deploy that applied 147 finished
        # (run 35672404601, 01:11:41Z): cohort_conversion_outcome.
        ("2026-09-22T07:00:12.001592+00:00", "cohort_conversion_outcome"),
        # Between the migration and its deploy finishing, the running image is unknown.
        ("2026-09-22T01:08:00+00:00", None),
        (None, None),
        ("not a date", None),
    ],
)
def test_a_cohort_run_saved_before_lane_t2_resolves_its_outcome_by_date(created_at, expected):
    row = {"data_provenance": COHORT, "created_at": created_at, "effect_heterogeneity": {}}
    assert stored_outcome_column(row) == expected


def test_a_datetime_created_at_resolves_too():
    from datetime import datetime, timezone

    row = {"data_provenance": COHORT, "created_at": datetime(2026, 9, 1, tzinfo=timezone.utc)}
    assert stored_outcome_column(row) == "conversion_rate"


@pytest.mark.parametrize("provenance", ["synthetic_uplift_v1", None, "rwd_uplift"])
def test_a_non_cohort_run_without_a_record_has_no_known_outcome(provenance):
    """The synthetic-uplift runs (3, 2026-06-16) estimated on the synthetic DGP's own
    ``outcome``, which is no measurable column; anything else unrecorded is unknown too."""
    row = {"data_provenance": provenance, "created_at": "2026-06-16T14:45:07+00:00"}
    assert stored_outcome_column(row) is None


def test_a_recorded_outcome_wins_over_the_date_rule():
    row = {
        "data_provenance": COHORT,
        "created_at": "2026-06-18T00:00:00+00:00",
        "effect_heterogeneity": {"outcome_column": "adopted"},
    }
    assert stored_outcome_column(row) == "adopted"
