"""Lane T2: a stored simulation records the outcome column its ATE was estimated ON.

The twin's outcome moved from ``cohort_conversion_outcome`` to ``adopted``. A draft experiment
made from a stored run must measure what THAT run predicted, so ``save_simulation`` writes the
run's outcome into the row's ``effect_heterogeneity`` JSON and ``stored_outcome_column`` reads it
back; rows saved earlier read as the column they have always been labelled with.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from src.digital_twin.models.simulation_models import (
    EffectHeterogeneity,
    InterventionConfig,
    SimulationRecommendation,
    SimulationResult,
)
from src.digital_twin.twin_repository import (
    OUTCOME_COLUMN_KEY,
    SimulationRepository,
    stored_outcome_column,
)


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


def test_a_row_saved_before_lane_t2_reads_as_its_historical_label():
    for row in ({}, {"effect_heterogeneity": None}, {"effect_heterogeneity": {"by_region": {}}}):
        assert stored_outcome_column(row) == "cohort_conversion_outcome"
