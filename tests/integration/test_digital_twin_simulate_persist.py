"""R5: all-real /simulate -> persist integration test.

Unlike test_digital_twin_e2e.py (which mocks the repository and hand-derives the
"actual" outcome), this drives the REAL ``run_simulation`` handler with the REAL
SimulationEngine + effect/ provider+estimator and a REAL TwinRepository over an
in-memory fake Supabase client that records inserts. It would have caught H4 (the
model_id= TypeError → 500) and N1 (a FAILED result persisted + returned 200).

Honest limitation (logged, not faked): the real-world fidelity leg — twin
prediction vs an ACTUAL A/B outcome — is untestable without a ground-truth outcome
feed, which does not exist. That path is covered by POST /validate tests that
supply an explicit actual_ate; we do NOT hand-derive an "actual" here.
"""

import math
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.digital_twin.models.twin_models import Brand, DigitalTwin, TwinPopulation, TwinType

# async def + await (NOT bare asyncio.run, which a meta-test bans in
# tests/integration/ as a RAGAS-pollution victim — #220/#218/#215); asyncio_mode=auto.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.heavy_ml,
    pytest.mark.asyncio,
    pytest.mark.xdist_group(name="digital_twin_e2e"),
]

_OPERATOR = {"sub": "op", "roles": ["operator"], "app_metadata": {"role": "operator"}}


def _training_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 2000
    return pd.DataFrame(
        {
            "specialty": np.random.choice(["rheumatology", "dermatology", "allergy"], n),
            "decile": np.random.randint(1, 11, n),
            "region": np.random.choice(["northeast", "south", "midwest", "west"], n),
            "digital_engagement_score": np.random.uniform(0.1, 0.9, n),
            "adoption_stage": np.random.choice(
                ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"], n
            ),
            "patient_volume": np.random.randint(50, 500, n),
            "prescribing_change": np.random.uniform(-0.1, 0.3, n),
        }
    )


def _trained_generator():
    from src.digital_twin.twin_generator import TwinGenerator

    gen = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.REMIBRUTINIB)
    gen.train(data=_training_data(), target_col="prescribing_change")
    return gen


def _cohort_provider():
    """A synthetic-gold cohort with a real (confounded) region-heterogeneous engagement
    effect, wrapped in the real CohortEffectDataProvider so the direct DML estimator
    produces a real, finite cohort-causal estimate."""
    from src.digital_twin.effect.provider import CohortEffectDataProvider

    rng = np.random.default_rng(7)
    n = 800
    regions = rng.choice(["northeast", "south", "midwest", "west"], size=n)
    tau_map = {"northeast": 0.45, "west": 0.30, "south": 0.18, "midwest": 0.08}
    market = rng.uniform(0.0, 1.0, n)
    eng = 10.0 / (1.0 + np.exp(-(1.5 * (market - 0.5) + rng.normal(0.0, 0.5, n))))
    t_bin = (eng > np.median(eng)).astype(float)
    tau = np.array([tau_map[r] for r in regions])
    conv = np.clip(0.5 + 0.8 * market + tau * t_bin + rng.normal(0.0, 0.25, n), 0.0, None)
    cohort = pd.DataFrame(
        {
            "region": regions,
            "engagement_score": eng,
            "conversion_rate": conv,
            "market_share": market,
            "total_rx_count": rng.poisson(80, n).astype(float),
        }
    )
    return CohortEffectDataProvider(cohort)


class _FakeInsert:
    def __init__(self, store, row):
        self._store = store
        self._row = row

    async def execute(self):
        self._store.append(self._row)
        return MagicMock(data=[self._row])


class _FakeTable:
    def __init__(self, store):
        self._store = store

    def insert(self, row):
        return _FakeInsert(self._store, row)


class _FakeClient:
    """In-memory Supabase stand-in that records twin_simulations inserts."""

    def __init__(self):
        self.inserts: list = []

    def table(self, _name):
        return _FakeTable(self.inserts)


def _request(twin_count: int):
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
    )

    return SimulateRequest(
        # digital_engagement is the IDENTIFIED intervention (real cohort-causal estimate);
        # email_campaign is now honestly unavailable (422) and cannot persist a result.
        intervention=InterventionConfigRequest(intervention_type="digital_engagement"),
        brand=BrandEnum.REMIBRUTINIB,
        twin_type=TwinTypeEnum.HCP,
        twin_count=twin_count,
    )


async def test_simulate_persists_one_real_completed_result():
    from src.api.routes import digital_twin as dt
    from src.api.routes.digital_twin import run_simulation
    from src.digital_twin.twin_repository import TwinRepository

    generator = _trained_generator()
    fake = _FakeClient()
    repo = TwinRepository(supabase_client=fake)
    model_row = {
        "model_id": str(uuid4()),
        "mlflow_model_uri": "models:/x/1",
        "mlflow_run_id": "r",
    }

    with (
        patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)),
        patch.object(dt, "_resolve_active_model_row", AsyncMock(return_value=model_row)),
        patch.object(dt, "_load_trained_generator", AsyncMock(return_value=generator)),
        patch(
            "src.digital_twin.effect.cohort_loader.build_cohort_provider_or_none",
            AsyncMock(return_value=_cohort_provider()),
        ),
    ):
        resp = await run_simulation(_request(twin_count=200), user=_OPERATOR)

    # Real engine produced a real, finite, non-zero cohort-causal ATE.
    assert resp.status.value == "completed"
    assert resp.simulated_ate != 0.0
    assert math.isfinite(resp.simulated_ate)
    assert resp.data_provenance == "cohort_estimated_synthetic_gold_v1"
    # Exactly ONE real twin_simulations row persisted, with completed status.
    # (Persisting data_provenance on the row is H5b/R4b — surfaced on the response
    # here via R1; the row-column round-trip is asserted in R4b's tests.)
    assert len(fake.inserts) == 1
    assert fake.inserts[0]["simulation_status"] == "completed"


async def test_simulate_failed_result_is_422_and_not_persisted():
    from src.api.routes import digital_twin as dt
    from src.api.routes.digital_twin import run_simulation
    from src.digital_twin.twin_repository import TwinRepository

    # Sub-threshold population (<100 after filtering) → engine returns FAILED.
    small_pop = TwinPopulation(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
        twins=[
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={"decile": 1, "digital_engagement_score": 0.5, "patient_volume": 100},
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for _ in range(50)
        ],
        size=50,
    )
    small_gen = MagicMock()
    small_gen.generate.return_value = small_pop

    fake = _FakeClient()
    repo = TwinRepository(supabase_client=fake)
    model_row = {"model_id": str(uuid4()), "mlflow_model_uri": "models:/x/1", "mlflow_run_id": "r"}

    with (
        patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)),
        patch.object(dt, "_resolve_active_model_row", AsyncMock(return_value=model_row)),
        patch.object(dt, "_load_trained_generator", AsyncMock(return_value=small_gen)),
        patch(
            "src.digital_twin.effect.cohort_loader.build_cohort_provider_or_none",
            AsyncMock(return_value=_cohort_provider()),
        ),
    ):
        with pytest.raises(HTTPException) as ei:
            await run_simulation(_request(twin_count=200), user=_OPERATOR)
        assert ei.value.status_code == 422

    # N1: a FAILED result must NOT be persisted.
    assert len(fake.inserts) == 0
