"""R1 N2 regression: ResultsAnalysisService.compare_experiment_to_twin is the 2-arg
convenience method BOTH the fidelity route (experiments.py) and the H9 task expected
(they passed only the two ids with a `# type: ignore[call-arg]` and 500'd). It must
orchestrate the real fetch — actual results via ABResultsRepository, twin prediction
via the REAL persisted twin columns (simulated_ate / _ci_*) — and delegate to the
6-arg primitive. NO fabricated effect."""

import inspect
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


@pytest.mark.unit
def test_compare_experiment_to_twin_exists_with_2arg_signature():
    from src.services.results_analysis import ResultsAnalysisService

    assert hasattr(ResultsAnalysisService, "compare_experiment_to_twin")
    sig = inspect.signature(ResultsAnalysisService.compare_experiment_to_twin)
    assert list(sig.parameters) == ["self", "experiment_id", "twin_simulation_id"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_experiment_to_twin_orchestrates_real_fetch():
    from src.repositories.ab_results import ExperimentResultRecord
    from src.services.results_analysis import ResultsAnalysisService

    exp_id = uuid4()
    sim_id = uuid4()
    record = ExperimentResultRecord(
        id=uuid4(),
        experiment_id=exp_id,
        analysis_type="final",
        analysis_method="itt",
        computed_at=datetime.now(timezone.utc),
        primary_metric="nrx",
        control_mean=0.10,
        treatment_mean=0.13,
        effect_estimate=0.03,
        effect_ci_lower=0.01,
        effect_ci_upper=0.05,
        p_value=0.01,
        sample_size_control=500,
        sample_size_treatment=500,
        statistical_power=0.9,
        is_significant=True,
    )
    sim_row = {
        "simulation_id": str(sim_id),
        "simulated_ate": 0.025,
        "simulated_ci_lower": 0.01,
        "simulated_ci_upper": 0.04,
    }

    service = ResultsAnalysisService()
    with (
        patch("src.repositories.ab_results.ABResultsRepository") as MockResults,
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch("src.digital_twin.twin_repository.TwinRepository") as MockTwinRepo,
        patch.object(service, "_persist_fidelity_comparison", new=AsyncMock()),
    ):
        MockResults.return_value.get_results = AsyncMock(return_value=[record])
        MockTwinRepo.return_value.get_simulation = AsyncMock(return_value=sim_row)

        result = await service.compare_experiment_to_twin(exp_id, sim_id)

    assert result.predicted_effect == pytest.approx(0.025)
    assert result.actual_effect == pytest.approx(0.03)
    assert result.prediction_error == pytest.approx(0.03 - 0.025)
    assert result.experiment_id == exp_id
