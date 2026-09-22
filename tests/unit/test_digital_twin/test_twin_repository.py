"""
Unit tests for digital_twin/twin_repository.py

Tests cover:
- TwinModelRepository save, get, list, deactivate
- SimulationRepository save, get, list, update, link
- FidelityRepository save, update, get
- TwinRepository facade pattern
- Redis caching
- MLflow integration
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.digital_twin.models.simulation_models import (
    EffectHeterogeneity,
    FidelityRecord,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)
from src.digital_twin.models.twin_models import (
    Brand,
    TwinModelConfig,
    TwinModelMetrics,
    TwinType,
)
from src.digital_twin.twin_repository import (
    FidelityRepository,
    SimulationRepository,
    TwinModelRepository,
    TwinRepository,
)


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    client = MagicMock()
    client.table = MagicMock(return_value=client)
    client.select = MagicMock(return_value=client)
    client.insert = MagicMock(return_value=client)
    client.update = MagicMock(return_value=client)
    client.eq = MagicMock(return_value=client)
    client.order = MagicMock(return_value=client)
    client.limit = MagicMock(return_value=client)
    not_mock = MagicMock()
    not_mock.is_ = MagicMock(return_value=client)
    client.not_ = not_mock
    client.is_ = MagicMock(return_value=client)
    client.execute = AsyncMock()
    return client


@pytest.fixture
def mock_mlflow():
    """Mock MLflow client."""
    return MagicMock()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = MagicMock()
    redis.setex = MagicMock()
    redis.get = MagicMock(return_value=None)
    redis.delete = MagicMock()
    return redis


@pytest.fixture
def twin_model_config():
    """Sample TwinModelConfig."""
    return TwinModelConfig(
        model_name="HCP Twin Model",
        model_description="Test twin model",
        twin_type=TwinType.HCP,
        brand=Brand.KISQALI,
        algorithm="random_forest",
        n_estimators=100,
        max_depth=10,
        training_samples=5000,
        validation_split=0.2,
        cv_folds=5,
        feature_columns=["decile", "specialty", "region"],
        target_column="prescribing_change",
        geographic_scope="US",
    )


@pytest.fixture
def twin_model_metrics():
    """Sample TwinModelMetrics."""
    return TwinModelMetrics(
        model_id=uuid4(),
        r2_score=0.85,
        rmse=0.12,
        mae=0.08,
        cv_scores=[0.82, 0.84, 0.86, 0.85, 0.83],
        cv_mean=0.84,
        cv_std=0.015,
        feature_importances={"decile": 0.45, "specialty": 0.30, "region": 0.25},
        top_features=["decile", "specialty", "region"],
        training_samples=5000,
        training_duration_seconds=120.5,
    )


class TestTwinModelRepository:
    """Tests for TwinModelRepository."""

    @pytest.mark.asyncio
    async def test_save_model(
        self, mock_supabase, mock_mlflow, mock_redis, twin_model_config, twin_model_metrics
    ):
        """Test saving a twin model."""
        repo = TwinModelRepository(mock_supabase, mock_mlflow, mock_redis)
        mock_supabase.execute.return_value = MagicMock(
            data=[{"model_id": str(twin_model_metrics.model_id)}]
        )

        model_id = await repo.save_model(twin_model_config, twin_model_metrics)

        assert model_id == twin_model_metrics.model_id
        mock_supabase.table.assert_called_once_with("digital_twin_models")
        mock_supabase.insert.assert_called_once()
        mock_supabase.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_model_persists_real_mlflow_refs_no_fabrication(
        self, mock_supabase, twin_model_config, twin_model_metrics
    ):
        """save_model must store the REAL mlflow refs the caller passes, never a
        fabricated ``models:/twin_<type>_<brand>/latest`` URI (#705 H4 anti-mock)."""
        repo = TwinModelRepository(mock_supabase)
        mock_supabase.execute.return_value = MagicMock(
            data=[{"model_id": str(twin_model_metrics.model_id)}]
        )

        await repo.save_model(
            twin_model_config,
            twin_model_metrics,
            mlflow_run_id="run-abc123",
            mlflow_model_uri="models:/m-deadbeef",
        )

        row = mock_supabase.insert.call_args.args[0]
        assert row["mlflow_run_id"] == "run-abc123"
        assert row["mlflow_model_uri"] == "models:/m-deadbeef"
        # Anti-mock: never the old fabricated stub pattern.
        assert "/latest" not in (row["mlflow_model_uri"] or "")

    @pytest.mark.asyncio
    async def test_save_model_persists_the_training_frame_identity(
        self, mock_supabase, twin_model_config, twin_model_metrics
    ):
        """#2206: training_config carries the frame that produced the fit so the
        /models fit fingerprint distinguishes fits from different frames."""
        repo = TwinModelRepository(mock_supabase)
        mock_supabase.execute.return_value = MagicMock(
            data=[{"model_id": str(twin_model_metrics.model_id)}]
        )

        await repo.save_model(
            twin_model_config,
            twin_model_metrics,
            data_provenance="synthetic",
            training_frame={"source": "synthetic_training_frame", "seed": 0, "n_rows": 2000},
        )

        row = mock_supabase.insert.call_args.args[0]
        assert row["training_config"]["training_frame"] == {
            "source": "synthetic_training_frame",
            "seed": 0,
            "n_rows": 2000,
        }
        # Legacy callers that pass nothing record nothing (no fabricated frame identity).
        mock_supabase.insert.reset_mock()
        await repo.save_model(twin_model_config, twin_model_metrics)
        assert "training_frame" not in mock_supabase.insert.call_args.args[0]["training_config"]

    @pytest.mark.asyncio
    async def test_save_model_no_mlflow_refs_stores_null_not_fabricated(
        self, mock_supabase, twin_model_config, twin_model_metrics
    ):
        """With no refs supplied, the row stores NULL — not a phantom URI."""
        repo = TwinModelRepository(mock_supabase)
        mock_supabase.execute.return_value = MagicMock(
            data=[{"model_id": str(twin_model_metrics.model_id)}]
        )

        await repo.save_model(twin_model_config, twin_model_metrics)

        row = mock_supabase.insert.call_args.args[0]
        assert row["mlflow_model_uri"] is None
        assert row["mlflow_run_id"] is None

    @pytest.mark.asyncio
    async def test_save_model_no_client(self, twin_model_config, twin_model_metrics):
        """Test save model without client."""
        repo = TwinModelRepository(None, None, None)

        model_id = await repo.save_model(twin_model_config, twin_model_metrics)

        assert model_id == twin_model_metrics.model_id

    @pytest.mark.asyncio
    async def test_get_model_from_cache(self, mock_supabase, mock_redis):
        """Test getting model from Redis cache."""
        repo = TwinModelRepository(mock_supabase, None, mock_redis)
        model_id = uuid4()
        cached_data = '{"model_id": "' + str(model_id) + '", "model_name": "Test"}'
        mock_redis.get.return_value = cached_data

        result = await repo.get_model(model_id)

        assert result["model_id"] == str(model_id)
        mock_redis.get.assert_called_once()
        mock_supabase.table.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_model_from_database(self, mock_supabase):
        """Test getting model from database."""
        repo = TwinModelRepository(mock_supabase, None, None)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(
            data=[{"model_id": str(model_id), "model_name": "Test Model"}]
        )

        result = await repo.get_model(model_id)

        assert result["model_id"] == str(model_id)
        assert result["model_name"] == "Test Model"
        mock_supabase.table.assert_called_with("digital_twin_models")
        mock_supabase.eq.assert_called_with("model_id", str(model_id))

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, mock_supabase):
        """Test getting non-existent model."""
        repo = TwinModelRepository(mock_supabase, None, None)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[])

        result = await repo.get_model(model_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_list_active_models(self, mock_supabase):
        """Test listing active models."""
        repo = TwinModelRepository(mock_supabase, None, None)
        mock_supabase.execute.return_value = MagicMock(
            data=[
                {"model_id": str(uuid4()), "is_active": True},
                {"model_id": str(uuid4()), "is_active": True},
            ]
        )

        result = await repo.list_active_models(twin_type=TwinType.HCP, brand="Kisqali", limit=10)

        assert len(result) == 2
        mock_supabase.eq.assert_any_call("is_active", True)
        mock_supabase.eq.assert_any_call("twin_type", TwinType.HCP.value)
        mock_supabase.eq.assert_any_call("brand", "Kisqali")

    @pytest.mark.asyncio
    async def test_deactivate_model(self, mock_supabase, mock_redis):
        """Test deactivating a model."""
        repo = TwinModelRepository(mock_supabase, None, mock_redis)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock()

        result = await repo.deactivate_model(model_id, "Testing")

        assert result is True
        mock_supabase.update.assert_called_once()
        mock_supabase.eq.assert_called_with("model_id", str(model_id))
        mock_redis.delete.assert_called_with(f"twin_model:{model_id}")

    @pytest.mark.asyncio
    async def test_update_fidelity_score(self, mock_supabase, mock_redis):
        """Test updating model fidelity score."""
        repo = TwinModelRepository(mock_supabase, None, mock_redis)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock()

        result = await repo.update_fidelity_score(model_id, 0.85, 100)

        assert result is True
        update_call = mock_supabase.update.call_args[0][0]
        assert update_call["fidelity_score"] == 0.85
        assert update_call["fidelity_sample_count"] == 100


class TestSimulationRepository:
    """Tests for SimulationRepository."""

    @pytest.fixture
    def simulation_result(self):
        """Sample SimulationResult."""
        return SimulationResult(
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                channel="email",
                frequency="weekly",
                duration_weeks=8,
            ),
            population_filters=PopulationFilter(specialties=["cardiology"], deciles=[1, 2, 3]),
            twin_count=1000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            effect_heterogeneity=EffectHeterogeneity(),
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Positive effect detected",
            recommended_sample_size=500,
            recommended_duration_weeks=8,
            simulation_confidence=0.85,
            status=SimulationStatus.COMPLETED,
            execution_time_ms=1500,
        )

    @pytest.mark.asyncio
    async def test_save_simulation(self, mock_supabase, simulation_result):
        """Test saving simulation result."""
        repo = SimulationRepository(mock_supabase)
        mock_supabase.execute.return_value = MagicMock()

        result_id = await repo.save_simulation(simulation_result, "Kisqali")

        assert result_id == simulation_result.simulation_id
        mock_supabase.table.assert_called_with("twin_simulations")
        mock_supabase.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_simulation(self, mock_supabase):
        """Test getting simulation by ID."""
        repo = SimulationRepository(mock_supabase)
        sim_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[{"simulation_id": str(sim_id)}])

        result = await repo.get_simulation(sim_id)

        assert result["simulation_id"] == str(sim_id)
        mock_supabase.eq.assert_called_with("simulation_id", str(sim_id))

    @pytest.mark.asyncio
    async def test_list_simulations(self, mock_supabase):
        """Test listing simulations with filters."""
        repo = SimulationRepository(mock_supabase)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(
            data=[
                {"simulation_id": str(uuid4())},
                {"simulation_id": str(uuid4())},
            ]
        )

        result = await repo.list_simulations(
            model_id=model_id, brand="Kisqali", status=SimulationStatus.COMPLETED, limit=50
        )

        assert len(result) == 2
        mock_supabase.eq.assert_any_call("model_id", str(model_id))
        mock_supabase.eq.assert_any_call("brand", "Kisqali")
        mock_supabase.eq.assert_any_call("simulation_status", SimulationStatus.COMPLETED.value)

    @pytest.mark.asyncio
    async def test_update_status(self, mock_supabase):
        """Test updating simulation status."""
        repo = SimulationRepository(mock_supabase)
        sim_id = uuid4()
        mock_supabase.execute.return_value = MagicMock()

        result = await repo.update_status(sim_id, SimulationStatus.RUNNING)

        assert result is True
        update_call = mock_supabase.update.call_args[0][0]
        assert update_call["simulation_status"] == SimulationStatus.RUNNING.value
        assert "started_at" in update_call

    @pytest.mark.asyncio
    async def test_link_experiment(self, mock_supabase):
        """Test linking simulation to experiment."""
        repo = SimulationRepository(mock_supabase)
        sim_id = uuid4()
        exp_id = uuid4()
        mock_supabase.execute.return_value = MagicMock()

        result = await repo.link_experiment(sim_id, exp_id)

        assert result is True
        update_call = mock_supabase.update.call_args[0][0]
        assert update_call["experiment_design_id"] == str(exp_id)


class TestFidelityRepository:
    """Tests for FidelityRepository."""

    @pytest.fixture
    def fidelity_record(self):
        """Sample FidelityRecord."""
        return FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            actual_ate=0.09,
            actual_ci_lower=0.06,
            actual_ci_upper=0.12,
            actual_sample_size=450,
        )

    @pytest.mark.asyncio
    async def test_save_fidelity_record(self, mock_supabase, fidelity_record):
        """Test saving fidelity record."""
        repo = FidelityRepository(mock_supabase)
        mock_supabase.execute.return_value = MagicMock()

        result_id = await repo.save_fidelity_record(fidelity_record)

        assert result_id == fidelity_record.tracking_id
        mock_supabase.table.assert_called_with("twin_fidelity_tracking")
        mock_supabase.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_fidelity_validation(self, mock_supabase):
        """Test updating fidelity validation."""
        repo = FidelityRepository(mock_supabase)
        tracking_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(
            data=[
                {
                    "tracking_id": str(tracking_id),
                    "actual_ate": 0.09,
                }
            ]
        )

        result = await repo.update_fidelity_validation(
            tracking_id,
            actual_ate=0.09,
            actual_ci_lower=0.06,
            actual_ci_upper=0.12,
            validated_by="admin",
        )

        assert result is not None
        assert result["actual_ate"] == 0.09
        update_call = mock_supabase.update.call_args[0][0]
        assert update_call["actual_ate"] == 0.09
        assert update_call["validated_by"] == "admin"

    @pytest.mark.asyncio
    async def test_get_fidelity_by_simulation(self, mock_supabase):
        """Test getting fidelity record by simulation ID."""
        repo = FidelityRepository(mock_supabase)
        sim_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(
            data=[
                {
                    "tracking_id": str(uuid4()),
                    "simulation_id": str(sim_id),
                    "simulated_ate": 0.08,
                    "fidelity_grade": "good",
                }
            ]
        )

        result = await repo.get_fidelity_by_simulation(sim_id)

        assert result is not None
        assert result.simulation_id == sim_id
        assert result.simulated_ate == 0.08

    @pytest.mark.asyncio
    async def test_get_model_fidelity_records(self, mock_supabase):
        """Test getting fidelity records for a model."""
        repo = FidelityRepository(mock_supabase)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(
            data=[
                {
                    "tracking_id": str(uuid4()),
                    "simulation_id": str(uuid4()),
                    "simulated_ate": 0.08,
                    "fidelity_grade": "good",
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "tracking_id": str(uuid4()),
                    "simulation_id": str(uuid4()),
                    "simulated_ate": 0.10,
                    "fidelity_grade": "excellent",
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                },
            ]
        )

        result = await repo.get_model_fidelity_records(model_id, validated_only=True, limit=50)

        assert len(result) == 2
        assert all(isinstance(r, FidelityRecord) for r in result)


class TestTwinRepository:
    """Tests for unified TwinRepository facade."""

    def test_initialization(self, mock_supabase, mock_mlflow, mock_redis):
        """Test TwinRepository initialization."""
        repo = TwinRepository(mock_supabase, mock_mlflow, mock_redis)

        assert isinstance(repo.models, TwinModelRepository)
        assert isinstance(repo.simulations, SimulationRepository)
        assert isinstance(repo.fidelity, FidelityRepository)

    @pytest.mark.asyncio
    async def test_save_model_delegation(
        self, mock_supabase, twin_model_config, twin_model_metrics
    ):
        """Test save_model delegates to models repository."""
        repo = TwinRepository(mock_supabase, None, None)
        mock_supabase.execute.return_value = MagicMock()

        with patch.object(repo.models, "save_model", new_callable=AsyncMock) as mock_save:
            mock_save.return_value = twin_model_metrics.model_id
            result = await repo.save_model(twin_model_config, twin_model_metrics)

            mock_save.assert_called_once_with(
                twin_model_config, twin_model_metrics, None, None, None, None
            )
            assert result == twin_model_metrics.model_id

    @pytest.mark.asyncio
    async def test_get_model_delegation(self, mock_supabase):
        """Test get_model delegates to models repository."""
        repo = TwinRepository(mock_supabase, None, None)
        model_id = uuid4()

        with patch.object(repo.models, "get_model", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"model_id": str(model_id)}
            result = await repo.get_model(model_id)

            mock_get.assert_called_once_with(model_id)
            assert result["model_id"] == str(model_id)

    @pytest.mark.asyncio
    async def test_list_active_models_delegation(self, mock_supabase):
        """Test list_active_models delegates to models repository."""
        repo = TwinRepository(mock_supabase, None, None)

        with patch.object(repo.models, "list_active_models", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [{"model_id": str(uuid4())}]
            result = await repo.list_active_models(twin_type=TwinType.HCP, brand="Kisqali")

            mock_list.assert_called_once_with(TwinType.HCP, "Kisqali")
            assert len(result) == 1


class TestStoredSubgroupsBasis:
    """How a stored row's effect_heterogeneity was computed (#2104 item 3).

    The stored JSON is never rewritten; the detail read ANNOTATES it. From #2097 to #2162
    the cohort estimator declared region alone, so an old cohort row with populated
    specialty and no axis provenance remains legacy. #2162 specialty rows carry explicit
    cohort-row provenance and are classified as current. Decile/adoption-stage population
    remains legacy on this path. No created_at cutoff is needed.
    """

    @staticmethod
    def _row(provenance, **axes):
        from src.digital_twin.effect.estimate import PROVENANCE_COHORT

        eh = {"by_specialty": {}, "by_decile": {}, "by_region": {}, "by_adoption_stage": {}}
        eh.update(axes)
        return {
            "data_provenance": (PROVENANCE_COHORT if provenance == "cohort" else provenance),
            "effect_heterogeneity": eh,
        }

    def test_cohort_row_with_a_non_region_axis_is_twin_weighted_legacy(self):
        from src.digital_twin.twin_repository import StoredSubgroupsBasis

        row = self._row(
            "cohort",
            by_region={"northeast": {"ate": 0.1, "std": 0.0, "n": 900}},
            by_specialty={"oncology": {"mean": 0.05, "std": 0.01, "n": 12}},
        )
        assert StoredSubgroupsBasis.from_row(row) is StoredSubgroupsBasis.TWIN_WEIGHTED_LEGACY

    def test_cohort_row_with_region_only_is_cohort_rows(self):
        from src.digital_twin.twin_repository import StoredSubgroupsBasis

        row = self._row("cohort", by_region={"northeast": {"ate": 0.1, "std": 0.0, "n": 900}})
        assert StoredSubgroupsBasis.from_row(row) is StoredSubgroupsBasis.COHORT_ROWS

    def test_new_specialty_axis_provenance_is_cohort_rows_not_legacy(self):
        from src.digital_twin.twin_repository import StoredSubgroupsBasis

        row = self._row(
            "cohort",
            by_specialty={"oncology": {"ate": 0.05, "std": 0.0, "n": 300}},
            axis_provenance={
                "specialty": {
                    "basis": "cohort_rows",
                    "source": "hcp_profiles.specialty",
                }
            },
        )
        assert StoredSubgroupsBasis.from_row(row) is StoredSubgroupsBasis.COHORT_ROWS

    def test_synthetic_row_is_per_twin_even_with_every_axis_populated(self):
        from src.digital_twin.effect.estimate import PROVENANCE_SYNTHETIC
        from src.digital_twin.twin_repository import StoredSubgroupsBasis

        row = self._row(
            PROVENANCE_SYNTHETIC,
            by_specialty={"oncology": {"mean": 0.05, "std": 0.01, "n": 12}},
            by_adoption_stage={"laggard": {"mean": 0.04, "std": 0.02, "n": 30}},
        )
        assert StoredSubgroupsBasis.from_row(row) is StoredSubgroupsBasis.PER_TWIN

    def test_rwd_row_is_per_twin(self):
        from src.digital_twin.effect.estimate import PROVENANCE_RWD
        from src.digital_twin.twin_repository import StoredSubgroupsBasis

        assert (
            StoredSubgroupsBasis.from_row(self._row(PROVENANCE_RWD))
            is StoredSubgroupsBasis.PER_TWIN
        )

    def test_unrecognised_provenance_is_unknown_not_per_twin(self):
        """Fail closed: only the two provenances that take the per-twin path (synthetic, rwd)
        are labelled per_twin; a future or foreign provenance string says nothing about how
        its subgroups were computed."""
        from src.digital_twin.twin_repository import StoredSubgroupsBasis

        row = self._row("some_future_estimator_v9")
        assert StoredSubgroupsBasis.from_row(row) is StoredSubgroupsBasis.UNKNOWN

    def test_detail_response_literal_matches_the_enum(self):
        """Drift pin: a new enum member would pass from_row, fail pydantic on the detail
        response and be swallowed into a 500 by the route's except. The frontend's handwritten
        union is pinned to the generated contract by a vitest ``expectTypeOf`` in
        DigitalTwin.test.tsx (the API client imports the handwritten type, so tsc alone
        never compares the two)."""
        from typing import get_args

        from src.api.routes.digital_twin import SimulationDetailResponse
        from src.digital_twin.twin_repository import StoredSubgroupsBasis

        literal = SimulationDetailResponse.model_fields["subgroups_basis"].annotation
        assert set(get_args(literal)) == {m.value for m in StoredSubgroupsBasis}

    def test_row_without_provenance_is_unknown(self):
        from src.digital_twin.twin_repository import StoredSubgroupsBasis

        assert StoredSubgroupsBasis.from_row(self._row(None)) is StoredSubgroupsBasis.UNKNOWN
        assert StoredSubgroupsBasis.from_row({}) is StoredSubgroupsBasis.UNKNOWN

    def test_legacy_axes_are_every_subgroup_axis_but_region(self):
        """Drift pin: when a lane makes another axis legitimate on the cohort path (specialty,
        #2104 item 2) it edits this one constant; the legacy rows stay recognised through
        by_adoption_stage, which exists in no table and can never be declared."""
        from src.digital_twin.effect.estimate import SUBGROUP_AXES
        from src.digital_twin.twin_repository import LEGACY_TWIN_WEIGHTED_AXES

        assert set(LEGACY_TWIN_WEIGHTED_AXES) == set(SUBGROUP_AXES) - {"region"}


# =============================================================================
# #2206 — model fidelity is the mean of its simulations' A/B comparisons
# =============================================================================


class _FakeQuery:
    """A minimal PostgREST-shaped chain: table().select().eq()/in_().execute()."""

    def __init__(self, tables):
        self._tables = tables
        self._table = None
        self._filters = []

    def table(self, name):
        q = _FakeQuery(self._tables)
        q._table = name
        return q

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._filters.append(lambda r: str(r.get(col)) == str(val))
        return self

    def in_(self, col, vals):
        allowed = {str(v) for v in vals}
        self._filters.append(lambda r: str(r.get(col)) in allowed)
        return self

    def update(self, payload):
        self._update = payload
        return self

    async def execute(self):
        rows = [r for r in self._tables.get(self._table, []) if all(f(r) for f in self._filters)]
        if getattr(self, "_update", None) is not None:
            for r in rows:
                r.update(self._update)
        return MagicMock(data=rows)


@pytest.mark.asyncio
async def test_refresh_model_fidelity_from_comparisons_averages_that_models_comparisons():
    from src.digital_twin.twin_repository import TwinRepository

    model_id = uuid4()
    other_model = uuid4()
    s1, s2, s3 = uuid4(), uuid4(), uuid4()
    tables = {
        "twin_simulations": [
            {"simulation_id": str(s1), "model_id": str(model_id)},
            {"simulation_id": str(s2), "model_id": str(model_id)},
            {"simulation_id": str(s3), "model_id": str(other_model)},
        ],
        "ab_fidelity_comparisons": [
            {"twin_simulation_id": str(s1), "fidelity_score": 0.9},
            {"twin_simulation_id": str(s2), "fidelity_score": 0.5},
            {"twin_simulation_id": str(s2), "fidelity_score": None},  # not a score
            {"twin_simulation_id": str(s3), "fidelity_score": 0.1},  # other model
        ],
        "digital_twin_models": [
            {"model_id": str(model_id), "fidelity_score": None, "fidelity_sample_count": 0}
        ],
    }
    repo = TwinRepository(supabase_client=_FakeQuery(tables))

    out = await repo.refresh_model_fidelity_from_comparisons(model_id)

    assert out == {"model_id": str(model_id), "fidelity_score": 0.7, "sample_count": 2}
    row = tables["digital_twin_models"][0]
    assert row["fidelity_score"] == 0.7
    assert row["fidelity_sample_count"] == 2
    assert row["last_fidelity_update"]


@pytest.mark.asyncio
async def test_refresh_model_fidelity_with_no_comparisons_writes_nothing():
    from src.digital_twin.twin_repository import TwinRepository

    model_id = uuid4()
    tables = {
        "twin_simulations": [{"simulation_id": str(uuid4()), "model_id": str(model_id)}],
        "ab_fidelity_comparisons": [],
        "digital_twin_models": [{"model_id": str(model_id), "fidelity_score": None}],
    }
    repo = TwinRepository(supabase_client=_FakeQuery(tables))

    assert await repo.refresh_model_fidelity_from_comparisons(model_id) is None
    assert tables["digital_twin_models"][0]["fidelity_score"] is None
