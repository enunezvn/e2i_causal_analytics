"""R5 + H9 — faithful end-to-end proof of the twin fidelity outcome feed.

NO MOCKING. Builds ONE real A/B experiment over REAL Fabhalta per-HCP
``business_metrics`` rows and drives the WHOLE real chain:

  real assignments ⋈ real business_metrics
    → ExperimentOutcomeRepository.load_arrays           (R5 outcome feed)
    → ResultsAnalysisService.compute_itt_results        (real ATE, real persist)
    → ab_experiment_results                             (real row)
    → SimulationRepository.get_latest_for_experiment    (H9 experiment-scoped)
    → compare_experiment_to_twin(..., twin_simulation_id=None)
    → ab_fidelity_comparisons                           (real row; exercises HIGH-1)

The H9 assertion is decisive: a DECOY simulation (different brand,
``experiment_design_id=NULL``, NEWER created_at) must NOT be picked — the OLD
global "newest simulation" fallback would have returned it.

Run gate: ``E2I_DB_INTEGRATION=1`` (mirrors the other real-DB integration
modules). Without it the module is skipped so CI unit-only lanes stay green.
All rows are isolated by unique ids and torn down in a finally block.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 to run",
)

EXISTING_MODEL_ID = "5141ba73-d926-43ec-b064-9cf717f8a9dd"  # real trained twin model
BRAND = "Fabhalta"
METRIC = "total_rx_count"


def _distinct_fabhalta_hcps(client, n: int) -> list[str]:
    res = (
        client.table("business_metrics")
        .select("hcp_id")
        .eq("metric_type", "per_hcp_rollup")
        .eq("brand", BRAND)
        .not_.is_(METRIC, "null")
        .limit(2000)
        .execute()
    )
    seen: list[str] = []
    for row in res.data or []:
        hcp = row.get("hcp_id")
        if hcp and hcp not in seen:
            seen.append(hcp)
        if len(seen) >= n:
            break
    return seen


@pytest.mark.asyncio
async def test_outcome_feed_and_h9_full_chain():
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories import get_supabase_client
    from src.repositories.experiment_outcome import ExperimentOutcomeRepository
    from src.services.results_analysis import ResultsAnalysisService

    sync = get_supabase_client()
    a_client = await get_async_supabase_client()

    run = uuid.uuid4().hex[:8]
    exp_id = str(uuid.uuid4())
    sim_scoped = str(uuid.uuid4())
    sim_decoy = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    hcps = _distinct_fabhalta_hcps(sync, 8)
    assert len(hcps) >= 6, f"need >=6 real Fabhalta HCPs, got {len(hcps)}"
    control_hcps = hcps[: len(hcps) // 2]
    treat_hcps = hcps[len(hcps) // 2 :]

    try:
        # --- real fixtures -------------------------------------------------
        sync.table("ml_experiments").insert(
            {
                "id": exp_id,
                "experiment_name": f"h9r5-e2e-{run}",
                "prediction_target": METRIC,
                "brand": BRAND,
                "status": "completed",
                "mlflow_experiment_id": f"h9r5-e2e-{run}",
            }
        ).execute()

        rows = [
            {"experiment_id": exp_id, "unit_id": h, "unit_type": "hcp", "variant": "control"}
            for h in control_hcps
        ] + [
            {"experiment_id": exp_id, "unit_id": h, "unit_type": "hcp", "variant": "treatment"}
            for h in treat_hcps
        ]
        sync.table("ab_experiment_assignments").insert(rows).execute()

        # Scoped sim (the right answer) + a NEWER decoy (different brand, no link).
        sync.table("twin_simulations").insert(
            {
                "simulation_id": sim_scoped,
                "model_id": EXISTING_MODEL_ID,
                "experiment_design_id": exp_id,
                "intervention_type": "email_campaign",
                "target_population": "hcp",
                "twin_count": 1000,
                "brand": BRAND,
                "simulated_ate": 0.5,
                "simulated_ci_lower": 0.1,
                "simulated_ci_upper": 0.9,
                "simulation_status": "completed",
                "created_at": (now - timedelta(hours=1)).isoformat(),
            }
        ).execute()
        sync.table("twin_simulations").insert(
            {
                "simulation_id": sim_decoy,
                "model_id": EXISTING_MODEL_ID,
                "experiment_design_id": None,  # unlinked → must never be picked
                "intervention_type": "email_campaign",
                "target_population": "hcp",
                "twin_count": 1000,
                "brand": "Kisqali",
                "simulated_ate": -99.0,
                "simulated_ci_lower": -100.0,
                "simulated_ci_upper": -98.0,
                "simulation_status": "completed",
                "created_at": now.isoformat(),  # NEWER than the scoped one
            }
        ).execute()

        # --- R5: real outcome feed ----------------------------------------
        outcome_repo = ExperimentOutcomeRepository(supabase_client=sync)
        control, treatment = await outcome_repo.load_arrays(uuid.UUID(exp_id), METRIC, brand=BRAND)
        assert control.size == len(control_hcps) and treatment.size == len(treat_hcps)
        assert np.isfinite(control).all() and np.isfinite(treatment).all()
        assert not np.isnan(control).any() and not np.isnan(treatment).any()

        # --- real ITT compute + persist -----------------------------------
        results = await ResultsAnalysisService().compute_itt_results(
            experiment_id=uuid.UUID(exp_id),
            primary_metric=METRIC,
            control_data=control,
            treatment_data=treatment,
        )
        assert np.isfinite(results.effect_estimate)
        persisted = (
            sync.table("ab_experiment_results").select("*").eq("experiment_id", exp_id).execute()
        )
        assert persisted.data, "compute_itt_results must persist an ab_experiment_results row"

        # --- H9: experiment-scoped resolver picks scoped, NOT newer decoy --
        from src.digital_twin.twin_repository import TwinRepository

        repo = TwinRepository(supabase_client=a_client)
        resolved = await repo.simulations.get_latest_for_experiment(uuid.UUID(exp_id))
        assert resolved is not None
        assert str(resolved["simulation_id"]) == sim_scoped, "H9 must pick the scoped sim"
        assert str(resolved["simulation_id"]) != sim_decoy, "decoy (newer, unlinked) must NOT win"

        # --- HIGH-1 + H9: real fidelity comparison persists ---------------
        svc = ResultsAnalysisService()
        comparison = await svc.compare_experiment_to_twin(uuid.UUID(exp_id))  # twin_sim_id=None
        assert str(comparison.twin_simulation_id) == sim_scoped
        assert comparison.predicted_effect == pytest.approx(0.5)
        assert np.isfinite(comparison.prediction_error)

        fid = (
            sync.table("ab_fidelity_comparisons").select("*").eq("experiment_id", exp_id).execute()
        )
        assert fid.data, "compare_experiment_to_twin must persist an ab_fidelity_comparisons row"
        assert str(fid.data[0]["twin_simulation_id"]) == sim_scoped

    finally:
        # --- deterministic teardown (FK-safe order) -----------------------
        sync.table("ab_fidelity_comparisons").delete().eq("experiment_id", exp_id).execute()
        sync.table("ab_experiment_results").delete().eq("experiment_id", exp_id).execute()
        sync.table("ab_experiment_assignments").delete().eq("experiment_id", exp_id).execute()
        for sid in (sim_scoped, sim_decoy):
            sync.table("twin_simulations").delete().eq("simulation_id", sid).execute()
        sync.table("ml_experiments").delete().eq("id", exp_id).execute()


@pytest.mark.asyncio
async def test_interim_analysis_full_chain_real_outcomes():
    """The interim consumer of the same outcome feed: scheduled_interim_analysis
    must load REAL per-HCP outcomes, run the sequential test, and persist a real
    ab_interim_analyses row (force=True bypasses only the milestone gate)."""
    from src.repositories import get_supabase_client
    from src.tasks.ab_testing_tasks import scheduled_interim_analysis

    sync = get_supabase_client()
    run = uuid.uuid4().hex[:8]
    exp_id = str(uuid.uuid4())
    assignment_ids: list[str] = []

    hcps = _distinct_fabhalta_hcps(sync, 8)
    assert len(hcps) >= 6
    control_hcps = hcps[: len(hcps) // 2]
    treat_hcps = hcps[len(hcps) // 2 :]

    try:
        sync.table("ml_experiments").insert(
            {
                "id": exp_id,
                "experiment_name": f"h9r5-interim-{run}",
                "prediction_target": METRIC,
                "brand": BRAND,
                "status": "running",
                "mlflow_experiment_id": f"h9r5-interim-{run}",
            }
        ).execute()

        rows = [
            {"experiment_id": exp_id, "unit_id": h, "unit_type": "hcp", "variant": "control"}
            for h in control_hcps
        ] + [
            {"experiment_id": exp_id, "unit_id": h, "unit_type": "hcp", "variant": "treatment"}
            for h in treat_hcps
        ]
        ins = sync.table("ab_experiment_assignments").insert(rows).execute()
        assignment_ids = [r["id"] for r in ins.data]

        # Real enrollments (one per assignment) so get_enrollment_stats reports
        # real enrollment — required even with force=True.
        sync.table("ab_experiment_enrollments").insert(
            [{"assignment_id": aid, "enrollment_status": "active"} for aid in assignment_ids]
        ).execute()

        result = scheduled_interim_analysis.run(experiment_id=exp_id, force=True)

        assert result["status"] == "completed", result
        assert np.isfinite(result["effect_estimate"])
        assert "decision" in result

        persisted = (
            sync.table("ab_interim_analyses").select("*").eq("experiment_id", exp_id).execute()
        )
        assert persisted.data, "interim analysis must persist an ab_interim_analyses row"

    finally:
        sync.table("ab_interim_analyses").delete().eq("experiment_id", exp_id).execute()
        for aid in assignment_ids:
            sync.table("ab_experiment_enrollments").delete().eq("assignment_id", aid).execute()
        sync.table("ab_experiment_assignments").delete().eq("experiment_id", exp_id).execute()
        sync.table("ml_experiments").delete().eq("id", exp_id).execute()
