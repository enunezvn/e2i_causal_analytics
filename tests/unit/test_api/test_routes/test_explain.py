"""
Unit tests for src/api/routes/explain.py

Tests cover:
- RealTimeSHAPService methods
- All endpoints (explain_prediction, explain_batch, get_explanation_history, list_explainable_models, health_check)
- Happy paths, error paths, edge cases
- Mock all external dependencies (BentoML, Feast, SHAP, Supabase)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.routes.explain import (
    BatchExplainRequest,
    ExplainRequest,
    ExplanationFormat,
    FeatureContribution,
    ModelType,
    RealTimeSHAPService,
    explain_batch,
    explain_prediction,
    get_explanation_history,
    get_shap_service,
    health_check,
    list_explainable_models,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_bentoml_client():
    """Mock BentoML client."""
    client = AsyncMock()
    client.predict = AsyncMock(
        return_value={"predictions": [[0.25, 0.75]], "_metadata": {"model_name": "v2.3.1-prod"}}
    )
    return client


@pytest.fixture
def mock_feast_client():
    """Mock Feast client."""
    client = AsyncMock()
    client.get_online_features = AsyncMock(
        return_value={
            "days_since_last_hcp_visit": [45],
            "total_hcp_interactions_90d": [12],
            "therapy_adherence_score": [0.72],
        }
    )
    return client


@pytest.fixture
def mock_shap_explainer():
    """Mock SHAP explainer."""
    explainer = AsyncMock()
    explainer.compute_shap_values = AsyncMock(
        return_value=MagicMock(
            shap_values={
                "days_since_last_hcp_visit": 0.15,
                "total_hcp_interactions_90d": 0.10,
                "therapy_adherence_score": 0.11,
            },
            base_value=0.42,
            explainer_type=MagicMock(value="TreeExplainer"),
            computation_time_ms=125.5,
        )
    )
    explainer.get_cache_stats = MagicMock(return_value={"hits": 10, "misses": 2})
    return explainer


@pytest.fixture
def mock_shap_repo():
    """Mock SHAP repository."""
    repo = MagicMock()
    repo.client = MagicMock()
    repo.table_name = "ml_shap_analyses"
    repo.store_analysis = AsyncMock(return_value=True)

    # Mock table query
    table_mock = MagicMock()
    execute_mock = MagicMock()
    execute_mock.data = [
        {"id": 1, "experiment_id": "test", "computed_at": datetime.now(timezone.utc).isoformat()}
    ]
    table_mock.select.return_value.order.return_value.limit.return_value.execute.return_value = (
        execute_mock
    )
    repo.client.table.return_value = table_mock

    return repo


@pytest.fixture
def shap_service(mock_bentoml_client, mock_feast_client, mock_shap_explainer, mock_shap_repo):
    """Create RealTimeSHAPService with mocked dependencies."""
    service = RealTimeSHAPService(
        bentoml_client=mock_bentoml_client,
        shap_explainer=mock_shap_explainer,
        shap_repo=mock_shap_repo,
        feast_client=mock_feast_client,
    )
    service._initialized = True
    return service


@pytest.fixture
def sample_explain_request():
    """Sample explain request."""
    return ExplainRequest(
        patient_id="PAT-2024-001234",
        hcp_id="HCP-NE-5678",
        model_type=ModelType.PROPENSITY,
        format=ExplanationFormat.TOP_K,
        top_k=5,
        store_for_audit=True,
    )


@pytest.fixture
def sample_features():
    """Sample feature dictionary."""
    return {
        "days_since_last_hcp_visit": 45,
        "total_hcp_interactions_90d": 12,
        "therapy_adherence_score": 0.72,
        "lab_value_trend": 0.15,
        "prior_brand_experience": 1,
    }


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    return {"user_id": "test_user", "role": "analyst"}


# =============================================================================
# RealTimeSHAPService Tests
# =============================================================================


class TestRealTimeSHAPService:
    """Tests for RealTimeSHAPService class."""

    @pytest.mark.asyncio
    async def test_ensure_initialized_creates_clients(self):
        """Test lazy initialization of async dependencies."""
        service = RealTimeSHAPService()
        assert service._initialized is False

        with (
            patch(
                "src.api.routes.explain.get_bentoml_client", new_callable=AsyncMock
            ) as mock_get_bento,
            patch(
                "src.api.routes.explain.get_feast_client", new_callable=AsyncMock
            ) as mock_get_feast,
            patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo,
        ):
            mock_get_bento.return_value = MagicMock()
            mock_get_feast.return_value = MagicMock()
            mock_get_repo.return_value = MagicMock()

            await service._ensure_initialized()

            assert service._initialized is True
            mock_get_bento.assert_called_once()
            mock_get_feast.assert_called_once()
            mock_get_repo.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_initialized_handles_errors(self):
        """Test initialization handles missing dependencies gracefully."""
        service = RealTimeSHAPService()

        with (
            patch(
                "src.api.routes.explain.get_bentoml_client", side_effect=Exception("BentoML error")
            ),
            patch("src.api.routes.explain.get_feast_client", side_effect=Exception("Feast error")),
            patch(
                "src.api.routes.explain.get_shap_analysis_repository",
                side_effect=Exception("Repo error"),
            ),
        ):
            await service._ensure_initialized()

            assert service._initialized is True
            assert service.bentoml_client is None
            assert service.feast_client is None
            assert service.shap_repo is None

    @pytest.mark.asyncio
    async def test_get_features_from_feast(self, shap_service, mock_feast_client):
        """Test feature retrieval from Feast."""
        features = await shap_service.get_features("PAT-123", ModelType.PROPENSITY)

        assert "days_since_last_hcp_visit" in features
        assert features["days_since_last_hcp_visit"] == 45
        mock_feast_client.get_online_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_features_fails_loud_on_error(self, shap_service, mock_feast_client):
        """#532: a Feast lookup failure must FAIL LOUD, not silently substitute
        fabricated default features.

        The old behavior returned ``_get_default_features()`` (hardcoded plausible
        values like therapy_adherence_score=0.72) which then drove a real SHAP
        explanation and a persistable regulatory-audit record — fabricated data
        presented as real. The route must surface this as a 503 instead.
        """
        mock_feast_client.get_online_features.side_effect = Exception("Feast error")

        with pytest.raises(HTTPException) as exc_info:
            await shap_service.get_features("PAT-123", ModelType.PROPENSITY)

        assert exc_info.value.status_code == 503
        assert "feature store" in str(exc_info.value.detail).lower()

    def test_get_feature_refs_for_model_propensity(self, shap_service):
        """Test feature refs for propensity model."""
        refs = shap_service._get_feature_refs_for_model(ModelType.PROPENSITY)

        assert len(refs) > 0
        assert any("days_since_last_hcp_visit" in ref for ref in refs)

    def test_get_feature_refs_for_model_risk(self, shap_service):
        """Test feature refs for risk stratification model."""
        refs = shap_service._get_feature_refs_for_model(ModelType.RISK_STRATIFICATION)

        assert len(refs) > 0
        assert any("comorbidity_count" in ref for ref in refs)

    def test_get_feature_refs_for_model_unknown(self, shap_service):
        """Test feature refs for unknown model type returns empty list."""
        refs = shap_service._get_feature_refs_for_model("unknown_model")

        assert refs == []

    def test_get_default_features(self, shap_service):
        """Test default features structure."""
        features = shap_service._get_default_features()

        assert "days_since_last_hcp_visit" in features
        assert "total_hcp_interactions_90d" in features
        assert isinstance(features["days_since_last_hcp_visit"], int)

    @pytest.mark.asyncio
    async def test_get_prediction_from_bentoml(self, shap_service, sample_features):
        """Test prediction from BentoML."""
        prediction = await shap_service.get_prediction(
            features=sample_features,
            model_type=ModelType.PROPENSITY,
        )

        assert "prediction_class" in prediction
        assert "prediction_probability" in prediction
        assert "model_version_id" in prediction
        assert prediction["prediction_probability"] == 0.75

    @pytest.mark.asyncio
    async def test_get_prediction_fallback(
        self, shap_service, mock_bentoml_client, sample_features
    ):
        """Test prediction fallback when BentoML fails."""
        mock_bentoml_client.predict.side_effect = Exception("BentoML error")

        prediction = await shap_service.get_prediction(
            features=sample_features,
            model_type=ModelType.PROPENSITY,
        )

        assert "prediction_class" in prediction
        assert prediction["prediction_probability"] == 0.78  # Fallback value

    def test_prepare_numeric_features(self, shap_service):
        """Test feature conversion to numeric."""
        features = {
            "int_feature": 42,
            "float_feature": 3.14,
            "bool_feature": True,
            "str_feature": "category_a",
            "none_feature": None,
        }

        numeric = shap_service._prepare_numeric_features(features)

        assert numeric["int_feature"] == 42.0
        assert numeric["float_feature"] == 3.14
        assert numeric["bool_feature"] == 1.0
        assert 0.0 <= numeric["str_feature"] <= 1.0
        assert numeric["none_feature"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_shap_success(self, shap_service, sample_features):
        """Test SHAP computation."""
        result = await shap_service.compute_shap(
            features=sample_features,
            model_type=ModelType.PROPENSITY,
            model_version_id="v2.3.1",
            top_k=3,
        )

        assert "base_value" in result
        assert "contributions" in result
        assert "shap_sum" in result
        assert len(result["contributions"]) == 3
        assert isinstance(result["contributions"][0], FeatureContribution)

    @pytest.mark.asyncio
    async def test_compute_shap_error_raises_http_exception(
        self, shap_service, mock_shap_explainer, sample_features
    ):
        """Test SHAP computation error handling."""
        mock_shap_explainer.compute_shap_values.side_effect = Exception("SHAP error")

        with pytest.raises(HTTPException) as exc_info:
            await shap_service.compute_shap(
                features=sample_features,
                model_type=ModelType.PROPENSITY,
                model_version_id="v2.3.1",
            )

        assert exc_info.value.status_code == 500
        assert "SHAP computation failed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_generate_narrative(self, shap_service):
        """Test narrative generation."""
        contributions = [
            FeatureContribution(
                feature_name="days_since_last_hcp_visit",
                feature_value=45,
                shap_value=0.15,
                contribution_direction="positive",
                contribution_rank=1,
            ),
            FeatureContribution(
                feature_name="therapy_adherence_score",
                feature_value=0.72,
                shap_value=-0.05,
                contribution_direction="negative",
                contribution_rank=2,
            ),
        ]

        prediction = {
            "prediction_class": "high_propensity",
            "prediction_probability": 0.78,
        }

        narrative = await shap_service.generate_narrative("PAT-123", prediction, contributions)

        assert "high propensity" in narrative
        assert "78%" in narrative
        assert "days since last hcp visit" in narrative

    @pytest.mark.asyncio
    async def test_store_audit_record_writes_canonical_fields(self, shap_service):
        """Issue #321 codex HIGH: audit writes must persist patient_id,
        analysis_type='local_realtime', and signed local_shap_values so that
        /explain/history/{patient_id} can actually retrieve the row.

        Previously the path called ShapAnalysisRepository.store_analysis,
        which writes analysis_type='global' + mean-abs global_importance and
        drops patient_id entirely — the feedback loop that makes the new
        history-filter useful.
        """
        repo = MagicMock()
        repo.client = MagicMock()
        repo.table_name = "ml_shap_analyses"
        # Capture the insert payload
        chain = MagicMock()
        chain.insert.return_value = chain
        chain.execute.return_value = MagicMock(data=[{"id": "row-1"}])
        repo.client.table.return_value = chain
        shap_service.shap_repo = repo

        result = await shap_service.store_audit_record(
            explanation_id="EXPL-123",
            patient_id="PAT-123",
            model_type="propensity",
            model_version_id="v2.3.1",
            features={"feature1": 1.0},
            shap_values={"feature1": 0.15, "feature2": -0.05},
            prediction={"prediction_class": "high", "prediction_probability": 0.8},
        )

        assert result is True
        chain.insert.assert_called_once()
        payload = chain.insert.call_args.args[0]
        assert payload["analysis_type"] == "local_realtime"
        assert payload["patient_id"] == "PAT-123"
        assert payload["entity_type"] == "patient"
        assert payload["entity_id"] == "PAT-123"
        assert payload["explanation_id"] == "EXPL-123"
        assert payload["prediction_class"] == "high"
        assert payload["prediction_probability"] == 0.8
        # Signed contributions stored — sign must survive the round-trip.
        assert payload["local_shap_values"]["feature1"] == 0.15
        assert payload["local_shap_values"]["feature2"] == -0.05

    @pytest.mark.asyncio
    async def test_store_audit_record_writes_only_schema_columns(self, shap_service):
        """Issue #321 codex iter-2 HIGH: PostgREST/Supabase rejects inserts
        that reference columns absent from the schema. The actual columns on
        ``ml_shap_analyses`` are defined by ``database/ml/mlops_tables.sql``
        + migration ``database/ml/011_realtime_shap_audit.sql``.

        Specifically, these columns do NOT exist on the table and MUST NOT
        appear in the insert payload (even though they exist on the FE
        ExplainResponse type — surface them via the migration tracker
        instead): ``model_type``, ``model_version_id``, ``shap_values``,
        ``hcp_id`` (column exists, but unused by this path),
        ``response_time_ms`` (column exists, but unused).
        """
        repo = MagicMock()
        repo.client = MagicMock()
        repo.table_name = "ml_shap_analyses"
        chain = MagicMock()
        chain.insert.return_value = chain
        chain.execute.return_value = MagicMock(data=[{"id": "row-1"}])
        repo.client.table.return_value = chain
        shap_service.shap_repo = repo

        await shap_service.store_audit_record(
            explanation_id="EXPL-X",
            patient_id="PAT-X",
            model_type="propensity",
            model_version_id="v2.3.1",
            features={"feature1": 1.0},
            shap_values={"feature1": 0.15},
            prediction={"prediction_class": "h", "prediction_probability": 0.7},
        )

        payload = chain.insert.call_args.args[0]
        # Hard fail if any unknown column slips into the write — PostgREST
        # would 4xx the insert in production, and audit storage would
        # silently fail in the background task.
        for forbidden in (
            "model_type",
            "model_version_id",
            "shap_values",
        ):
            assert forbidden not in payload, (
                f"forbidden column {forbidden!r} present in insert payload "
                f"(not in checked-in ml_shap_analyses schema): {payload!r}"
            )

    @pytest.mark.asyncio
    async def test_store_audit_record_no_repo(self, shap_service):
        """Test audit storage when repository unavailable."""
        shap_service.shap_repo = None

        result = await shap_service.store_audit_record(
            explanation_id="EXPL-123",
            patient_id="PAT-123",
            model_type="propensity",
            model_version_id="v2.3.1",
            features={},
            shap_values={},
            prediction={},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_store_audit_record_error(self, shap_service):
        """Test audit storage error handling."""
        repo = MagicMock()
        repo.client = MagicMock()
        repo.table_name = "ml_shap_analyses"
        chain = MagicMock()
        chain.insert.return_value = chain
        chain.execute.side_effect = Exception("DB error")
        repo.client.table.return_value = chain
        shap_service.shap_repo = repo

        result = await shap_service.store_audit_record(
            explanation_id="EXPL-123",
            patient_id="PAT-123",
            model_type="propensity",
            model_version_id="v2.3.1",
            features={},
            shap_values={},
            prediction={},
        )

        assert result is False


# =============================================================================
# Endpoint Tests
# =============================================================================


class TestExplainPredictionEndpoint:
    """Tests for /explain/predict endpoint."""

    @pytest.mark.asyncio
    async def test_explain_prediction_success(self, sample_explain_request, mock_user):
        """Test successful prediction explanation."""
        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_features = AsyncMock(return_value={"feat1": 1.0})
            mock_service.get_prediction = AsyncMock(
                return_value={
                    "prediction_class": "high_propensity",
                    "prediction_probability": 0.78,
                    "model_version_id": "v2.3.1",
                }
            )
            mock_service.compute_shap = AsyncMock(
                return_value={
                    "base_value": 0.42,
                    "contributions": [
                        FeatureContribution(
                            feature_name="feat1",
                            feature_value=1.0,
                            shap_value=0.15,
                            contribution_direction="positive",
                            contribution_rank=1,
                        )
                    ],
                    "shap_sum": 0.36,
                }
            )
            mock_service.generate_narrative = AsyncMock(return_value="Test narrative")
            mock_service.store_audit_record = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            background_tasks = BackgroundTasks()

            response = await explain_prediction(sample_explain_request, background_tasks, mock_user)

            # Patient ID should be masked (not the original value)
            assert response.patient_id != "PAT-2024-001234"
            assert response.model_type == ModelType.PROPENSITY
            assert response.prediction_probability == 0.78
            assert len(response.top_features) == 1
            assert response.audit_stored is True

    @pytest.mark.asyncio
    async def test_explain_prediction_with_provided_features(self, mock_user):
        """Test prediction with pre-provided features."""
        request = ExplainRequest(
            patient_id="PAT-123",
            model_type=ModelType.PROPENSITY,
            features={"feat1": 42.0},
        )

        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_features = AsyncMock()
            mock_service.get_prediction = AsyncMock(
                return_value={
                    "prediction_class": "high",
                    "prediction_probability": 0.8,
                    "model_version_id": "v1",
                }
            )
            mock_service.compute_shap = AsyncMock(
                return_value={
                    "base_value": 0.5,
                    "contributions": [],
                    "shap_sum": 0.3,
                }
            )
            mock_get_service.return_value = mock_service

            await explain_prediction(request, BackgroundTasks(), mock_user)

            # Should NOT call get_features since features were provided
            mock_service.get_features.assert_not_called()

    @pytest.mark.asyncio
    async def test_explain_prediction_narrative_format(self, mock_user):
        """Test prediction with narrative format."""
        request = ExplainRequest(
            patient_id="PAT-123",
            model_type=ModelType.PROPENSITY,
            format=ExplanationFormat.NARRATIVE,
        )

        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_features = AsyncMock(return_value={})
            mock_service.get_prediction = AsyncMock(
                return_value={
                    "prediction_class": "high",
                    "prediction_probability": 0.8,
                    "model_version_id": "v1",
                }
            )
            mock_service.compute_shap = AsyncMock(
                return_value={
                    "base_value": 0.5,
                    "contributions": [],
                    "shap_sum": 0.3,
                }
            )
            mock_service.generate_narrative = AsyncMock(return_value="Generated narrative")
            mock_get_service.return_value = mock_service

            response = await explain_prediction(request, BackgroundTasks(), mock_user)

            assert response.narrative_explanation == "Generated narrative"
            mock_service.generate_narrative.assert_called_once()

    @pytest.mark.asyncio
    async def test_explain_prediction_error(self, sample_explain_request, mock_user):
        """Test error handling in prediction endpoint."""
        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_features = AsyncMock(side_effect=Exception("Test error"))
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await explain_prediction(sample_explain_request, BackgroundTasks(), mock_user)

            assert exc_info.value.status_code == 500
            assert "Explanation failed" in exc_info.value.detail


class TestExplainBatchEndpoint:
    """Tests for /explain/predict/batch endpoint."""

    @pytest.mark.asyncio
    async def test_explain_batch_parallel_success(self, mock_user):
        """Test batch explanation in parallel mode."""
        from src.api.routes.explain import ExplainResponse

        requests = [
            ExplainRequest(patient_id=f"PAT-{i}", model_type=ModelType.PROPENSITY) for i in range(3)
        ]
        batch_request = BatchExplainRequest(requests=requests, parallel=True)

        # Create proper ExplainResponse instance
        async def mock_explain(*args, **kwargs):
            return ExplainResponse(
                explanation_id="test-123",
                request_timestamp=datetime.now(timezone.utc),
                patient_id="P******",
                model_type=ModelType.PROPENSITY,
                model_version_id="v1",
                prediction_class="high",
                prediction_probability=0.8,
                top_features=[],
                shap_sum=0.3,
                computation_time_ms=100.0,
                audit_stored=False,
            )

        with patch("src.api.routes.explain.explain_prediction", new=mock_explain):
            response = await explain_batch(batch_request, BackgroundTasks(), mock_user)

            assert response.total_requests == 3
            assert response.successful == 3
            assert response.failed == 0

    @pytest.mark.asyncio
    async def test_explain_batch_sequential(self, mock_user):
        """Test batch explanation in sequential mode."""
        from src.api.routes.explain import ExplainResponse

        requests = [
            ExplainRequest(patient_id=f"PAT-{i}", model_type=ModelType.PROPENSITY) for i in range(2)
        ]
        batch_request = BatchExplainRequest(requests=requests, parallel=False)

        async def mock_explain(*args, **kwargs):
            return ExplainResponse(
                explanation_id="test-123",
                request_timestamp=datetime.now(timezone.utc),
                patient_id="P******",
                model_type=ModelType.PROPENSITY,
                model_version_id="v1",
                prediction_class="high",
                prediction_probability=0.8,
                top_features=[],
                shap_sum=0.3,
                computation_time_ms=100.0,
                audit_stored=False,
            )

        with patch("src.api.routes.explain.explain_prediction", new=mock_explain):
            response = await explain_batch(batch_request, BackgroundTasks(), mock_user)

            assert response.total_requests == 2
            assert response.successful == 2

    @pytest.mark.asyncio
    async def test_explain_batch_with_errors(self, mock_user):
        """Test batch explanation handles individual errors."""
        from src.api.routes.explain import ExplainResponse

        requests = [
            ExplainRequest(patient_id=f"PAT-{i}", model_type=ModelType.PROPENSITY) for i in range(3)
        ]
        batch_request = BatchExplainRequest(requests=requests, parallel=True)

        # Create valid ExplainResponse for successful calls
        success_response = ExplainResponse(
            explanation_id="test-123",
            request_timestamp=datetime.now(timezone.utc),
            patient_id="P******",
            model_type=ModelType.PROPENSITY,
            model_version_id="v1",
            prediction_class="high",
            prediction_probability=0.8,
            top_features=[],
            shap_sum=0.3,
            computation_time_ms=100.0,
            audit_stored=False,
        )

        call_count = 0

        async def mock_explain(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise HTTPException(status_code=500, detail="Test error")
            return success_response

        with patch("src.api.routes.explain.explain_prediction", new=mock_explain):
            response = await explain_batch(batch_request, BackgroundTasks(), mock_user)

            assert response.total_requests == 3
            assert response.failed == 1
            assert len(response.errors) == 1


class TestGetExplanationHistoryEndpoint:
    """Tests for /explain/history/{patient_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_success(self):
        """Test retrieving explanation history."""
        with patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.client = MagicMock()
            mock_repo.table_name = "ml_shap_analyses"

            # Mock the supabase query chain. The route now calls
            # table().select().eq("patient_id", ...).order().limit().execute(),
            # so build a self-returning chain.
            mock_execute = MagicMock(
                data=[
                    {"id": 1, "patient_id": "PAT-123", "explanation_id": "EXPL-1"},
                    {"id": 2, "patient_id": "PAT-123", "explanation_id": "EXPL-2"},
                ]
            )
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.order.return_value = chain
            chain.limit.return_value = chain
            chain.execute.return_value = mock_execute
            mock_repo.client.table.return_value = chain
            mock_get_repo.return_value = mock_repo

            response = await get_explanation_history("PAT-123")

            # Patient ID should be masked
            assert "patient_id" in response
            assert response["patient_id"] != "PAT-123"  # Should be masked
            assert response["total_explanations"] == 2

    @pytest.mark.asyncio
    async def test_get_history_no_client(self):
        """Test history retrieval when DB client unavailable."""
        with patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.client = None
            mock_get_repo.return_value = mock_repo

            response = await get_explanation_history("PAT-123")

            assert response["total_explanations"] == 0
            assert "not available" in response["message"]

    @pytest.mark.asyncio
    async def test_get_history_error(self):
        """Test history retrieval error handling."""
        with patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo:
            mock_get_repo.side_effect = Exception("DB error")

            response = await get_explanation_history("PAT-123")

            assert response["total_explanations"] == 0
            assert "error" in response

    @pytest.mark.asyncio
    async def test_get_history_filters_by_patient_id(self):
        """Issue #321 HIGH: /history/{patient_id} must filter rows by patient_id.

        The schema has a `patient_id` column on ml_shap_analyses (per migration
        database/ml/011_realtime_shap_audit.sql). The endpoint previously ignored
        the path-param and returned recent rows for any patient. Verify the
        supabase query chain calls `.eq("patient_id", patient_id)`.
        """
        with patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.client = MagicMock()
            mock_repo.table_name = "ml_shap_analyses"

            # Build a fake chain: table().select().eq().order().limit().execute()
            chain = MagicMock()
            chain.execute.return_value = MagicMock(data=[])
            chain.limit.return_value = chain
            chain.order.return_value = chain
            chain.eq.return_value = chain
            chain.select.return_value = chain
            mock_repo.client.table.return_value = chain
            mock_get_repo.return_value = mock_repo

            await get_explanation_history("PAT-FILTER-ME")

            # Verify .eq("patient_id", ...) was called with the actual patient_id
            eq_calls = [
                call_args
                for call_args in chain.eq.call_args_list
                if call_args.args and call_args.args[0] == "patient_id"
            ]
            assert eq_calls, (
                f"Expected .eq('patient_id', ...) call but got: {chain.eq.call_args_list!r}"
            )
            assert eq_calls[0].args[1] == "PAT-FILTER-ME"

    @pytest.mark.asyncio
    async def test_get_history_response_shape_matches_fe_type(self):
        """Issue #321 HIGH: /history/{patient_id} response must match the FE type
        `ExplanationHistoryResponse { explanations: ExplainResponse[] }`.

        Each item should carry ExplainResponse fields (explanation_id,
        request_timestamp, patient_id, model_type, model_version_id,
        prediction_class, prediction_probability, top_features, shap_sum,
        computation_time_ms, audit_stored).

        Uses the canonical ``local_shap_values`` column (from
        ``database/ml/mlops_tables.sql`` — local explanations are stored
        there, not in ``global_importance``). Verifies signed contributions
        survive normalization so that direction and shap_sum are preserved.
        """
        from datetime import datetime, timezone

        sample_row = {
            "explanation_id": "EXPL-20251217-abc123",
            "patient_id": "PAT-001",
            "hcp_id": "HCP-7",
            "model_type": "propensity",
            "model_version_id": "v2.3.1",
            "prediction_class": "high_propensity",
            "prediction_probability": 0.78,
            "base_value": 0.42,
            "local_shap_values": {
                "days_since_last_hcp_visit": 0.15,
                "therapy_adherence_score": -0.05,
            },
            "request_timestamp": datetime.now(timezone.utc).isoformat(),
            "response_time_ms": 127.5,
        }

        with patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.client = MagicMock()
            mock_repo.table_name = "ml_shap_analyses"

            chain = MagicMock()
            chain.execute.return_value = MagicMock(data=[sample_row])
            chain.limit.return_value = chain
            chain.order.return_value = chain
            chain.eq.return_value = chain
            chain.select.return_value = chain
            mock_repo.client.table.return_value = chain
            mock_get_repo.return_value = mock_repo

            response = await get_explanation_history("PAT-001")

            assert "explanations" in response
            assert response["total_explanations"] == 1
            assert len(response["explanations"]) == 1

            item = response["explanations"][0]
            # FE ExplainResponse fields (per frontend/src/types/explain.ts)
            for field in (
                "explanation_id",
                "request_timestamp",
                "patient_id",
                "model_type",
                "model_version_id",
                "prediction_class",
                "prediction_probability",
                "top_features",
                "shap_sum",
                "computation_time_ms",
                "audit_stored",
            ):
                assert field in item, f"missing {field!r} in normalized item: {item!r}"

            # top_features should be a list of FeatureContribution-shaped dicts
            assert isinstance(item["top_features"], list)
            if item["top_features"]:
                first = item["top_features"][0]
                for f in (
                    "feature_name",
                    "feature_value",
                    "shap_value",
                    "contribution_direction",
                    "contribution_rank",
                ):
                    assert f in first, f"missing {f!r} in top_features entry"

            # Signed-direction preservation: the negative SHAP value must map
            # to direction='negative'. Falling back to global_importance
            # would silently absolutize signs.
            directions = {
                f["feature_name"]: f["contribution_direction"] for f in item["top_features"]
            }
            assert directions["therapy_adherence_score"] == "negative"
            assert directions["days_since_last_hcp_visit"] == "positive"

    @pytest.mark.asyncio
    async def test_get_history_local_shap_values_preferred_over_global_importance(self):
        """Issue #321 codex MED: when a row has both ``local_shap_values``
        (canonical, signed) and ``global_importance`` (mean-absolute),
        normalization must prefer ``local_shap_values`` so signs survive.
        """
        sample_row = {
            "id": "r1",
            "explanation_id": "EXPL-1",
            "patient_id": "PAT-X",
            "model_type": "propensity",
            "model_version_id": "v1",
            "prediction_class": "high",
            "prediction_probability": 0.8,
            # Mixed presence — local has signs, global lost them.
            "local_shap_values": {"feat_a": -0.3, "feat_b": 0.1},
            "global_importance": {"feat_a": 0.3, "feat_b": 0.1},
        }
        with patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.client = MagicMock()
            mock_repo.table_name = "ml_shap_analyses"
            chain = MagicMock()
            chain.execute.return_value = MagicMock(data=[sample_row])
            chain.limit.return_value = chain
            chain.order.return_value = chain
            chain.eq.return_value = chain
            chain.select.return_value = chain
            mock_repo.client.table.return_value = chain
            mock_get_repo.return_value = mock_repo

            resp = await get_explanation_history("PAT-X")
            item = resp["explanations"][0]
            directions = {
                f["feature_name"]: f["contribution_direction"] for f in item["top_features"]
            }
            assert directions["feat_a"] == "negative"
            # shap_sum reflects signed contributions: -0.3 + 0.1 = -0.2
            assert item["shap_sum"] == pytest.approx(-0.2)

    @pytest.mark.asyncio
    async def test_get_history_model_type_filter_is_client_side(self):
        """Issue #321 codex iter-3 MED: ``model_type`` is not a column on
        ``ml_shap_analyses`` (per the checked-in schema), so the optional
        ``model_type`` query param MUST be applied client-side after the
        supabase fetch — pushing it into a ``.eq("model_type", ...)`` would
        get rejected by PostgREST.

        Falsifiability: the supabase chain's ``.eq`` must be called with
        ``patient_id`` once and never with ``model_type``; the response
        must reflect only rows matching the requested model_type.
        """
        rows = [
            {
                "id": "r1",
                "explanation_id": "EXPL-1",
                "patient_id": "PAT-Y",
                "model_type": "propensity",
                "local_shap_values": {"f1": 0.1},
            },
            {
                "id": "r2",
                "explanation_id": "EXPL-2",
                "patient_id": "PAT-Y",
                "model_type": "churn_prediction",
                "local_shap_values": {"f2": 0.2},
            },
        ]
        with patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.client = MagicMock()
            mock_repo.table_name = "ml_shap_analyses"
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.order.return_value = chain
            chain.limit.return_value = chain
            chain.execute.return_value = MagicMock(data=rows)
            mock_repo.client.table.return_value = chain
            mock_get_repo.return_value = mock_repo

            resp = await get_explanation_history("PAT-Y", model_type=ModelType.PROPENSITY)
            # Filter must be applied client-side, not pushed into supabase.
            eq_keys = [c.args[0] for c in chain.eq.call_args_list if c.args]
            assert "model_type" not in eq_keys, (
                f"unexpected .eq('model_type', ...): {chain.eq.call_args_list!r}"
            )
            assert eq_keys.count("patient_id") == 1
            # Only the propensity row should be returned.
            assert resp["total_explanations"] == 1
            assert resp["explanations"][0]["explanation_id"] == "EXPL-1"

    @pytest.mark.asyncio
    async def test_get_history_awaits_async_execute(self):
        """Issue #321 codex HIGH: ``get_shap_analysis_repository`` installs
        an async Supabase client, so ``.execute()`` returns a coroutine.
        The endpoint must await it; reading ``.data`` on a coroutine raises.
        """

        async def async_execute():
            return MagicMock(data=[])

        with patch("src.api.routes.explain.get_shap_analysis_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.client = MagicMock()
            mock_repo.table_name = "ml_shap_analyses"
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.order.return_value = chain
            chain.limit.return_value = chain
            # IMPORTANT: execute returns a coroutine, simulating async client.
            chain.execute.side_effect = lambda: async_execute()
            mock_repo.client.table.return_value = chain
            mock_get_repo.return_value = mock_repo

            response = await get_explanation_history("PAT-async")
            assert response["total_explanations"] == 0
            assert "error" not in response, (
                f"endpoint did not await async execute (fell into error path): {response!r}"
            )


class TestListExplainableModelsEndpoint:
    """Tests for /explain/models endpoint."""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test listing explainable models."""
        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.shap_explainer = MagicMock()
            mock_service.shap_explainer.get_cache_stats.return_value = {"hits": 100, "misses": 10}
            mock_get_service.return_value = mock_service

            response = await list_explainable_models()

            assert "supported_models" in response
            assert len(response["supported_models"]) == len(ModelType)
            assert response["total_models"] == len(ModelType)
            assert response["cache_stats"]["hits"] == 100

    @pytest.mark.asyncio
    async def test_list_models_includes_latest_version(self):
        """Issue #321 LOW: /api/explain/models must emit `latest_version` per model.

        FE type `ExplainableModelInfo` declares `latest_version: string` but route
        previously emitted nothing for the field. Verify every entry now has it.
        """
        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.shap_explainer = MagicMock()
            mock_service.shap_explainer.get_cache_stats.return_value = {"hits": 0, "misses": 0}
            mock_get_service.return_value = mock_service

            response = await list_explainable_models()

            assert "supported_models" in response
            for entry in response["supported_models"]:
                # field must be present (may be None when no registry data)
                assert "latest_version" in entry, f"missing latest_version in entry {entry!r}"
                # type is Optional[str]
                assert entry["latest_version"] is None or isinstance(entry["latest_version"], str)

    @pytest.mark.asyncio
    async def test_list_models_uses_registered_at_and_picks_newest(self):
        """Issue #321 codex HIGH: ml_model_registry has ``registered_at``, not
        ``created_at`` (see database/ml/mlops_tables.sql:166). The lookup must
        query that column and pick the newest row per model_name.
        """

        async def async_execute():
            return MagicMock(
                data=[
                    {
                        "model_name": "propensity",
                        "model_version": "v3.0",
                        "registered_at": "2026-05-15T00:00:00Z",
                    },
                    {
                        "model_name": "propensity",
                        "model_version": "v2.3.1",
                        "registered_at": "2026-04-01T00:00:00Z",
                    },
                    {
                        "model_name": "churn_prediction",
                        "model_version": "v1.0",
                        "registered_at": "2026-05-01T00:00:00Z",
                    },
                ]
            )

        captured = {}

        def select_capture(cols):
            captured["select"] = cols
            return chain

        def order_capture(col, desc=False):
            captured["order"] = (col, desc)
            return chain

        chain = MagicMock()
        chain.in_.return_value = chain
        chain.select.side_effect = select_capture
        chain.order.side_effect = order_capture
        chain.execute.side_effect = lambda: async_execute()

        fake_client = MagicMock()
        fake_client.table.return_value = chain

        async def fake_get_client():
            return fake_client

        with (
            patch("src.api.routes.explain.get_shap_service") as mock_get_service,
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                side_effect=fake_get_client,
            ),
        ):
            mock_service = MagicMock()
            mock_service.shap_explainer.get_cache_stats.return_value = {}
            mock_get_service.return_value = mock_service

            response = await list_explainable_models()

            # Confirm SELECT and ORDER targeted the canonical column.
            assert "registered_at" in captured.get("select", ""), captured
            assert captured.get("order", (None, None))[0] == "registered_at"

            by_type = {m["model_type"]: m["latest_version"] for m in response["supported_models"]}
            assert by_type["propensity"] == "v3.0"  # newest, not v2.3.1
            assert by_type["churn_prediction"] == "v1.0"
            assert by_type["next_best_action"] is None  # absent from registry
            assert by_type["risk_stratification"] is None


class TestHealthCheckEndpoint:
    """Tests for /explain/health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when all dependencies available."""
        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service._ensure_initialized = AsyncMock()
            mock_service.bentoml_client = MagicMock()
            mock_service.feast_client = MagicMock()
            mock_service.shap_explainer = MagicMock()
            mock_service.shap_explainer.get_cache_stats.return_value = {}
            mock_service.shap_repo = MagicMock()
            mock_service.shap_repo.client = MagicMock()
            mock_get_service.return_value = mock_service

            response = await health_check()

            assert response["status"] == "healthy"
            assert response["service"] == "real-time-shap-api"
            assert response["dependencies"]["bentoml"] == "connected"
            assert response["dependencies"]["feast"] == "connected"
            assert response["dependencies"]["shap_explainer"] == "loaded"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check when SHAP explainer missing."""
        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service._ensure_initialized = AsyncMock()
            mock_service.bentoml_client = None
            mock_service.feast_client = None
            mock_service.shap_explainer = None
            mock_service.shap_repo = None
            mock_get_service.return_value = mock_service

            response = await health_check()

            assert response["status"] == "degraded"
            assert response["dependencies"]["bentoml"] == "not_configured"
            assert response["dependencies"]["shap_explainer"] == "not_loaded"


class TestGetShapServiceFunction:
    """Tests for get_shap_service singleton function."""

    @pytest.mark.asyncio
    async def test_get_shap_service_creates_singleton(self):
        """Test that get_shap_service creates a singleton."""
        # Reset singleton
        import src.api.routes.explain as explain_module

        explain_module._shap_service = None

        service1 = await get_shap_service()
        service2 = await get_shap_service()

        assert service1 is service2

    @pytest.mark.asyncio
    async def test_get_shap_service_returns_existing(self):
        """Test that get_shap_service returns existing instance."""
        import src.api.routes.explain as explain_module

        # Set a specific instance
        test_service = RealTimeSHAPService()
        explain_module._shap_service = test_service

        service = await get_shap_service()

        assert service is test_service


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_explain_with_empty_features(self, mock_user):
        """Test explanation with empty features dict."""
        request = ExplainRequest(
            patient_id="PAT-123",
            model_type=ModelType.PROPENSITY,
            features={},
        )

        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_prediction = AsyncMock(
                return_value={
                    "prediction_class": "high",
                    "prediction_probability": 0.8,
                    "model_version_id": "v1",
                }
            )
            mock_service.compute_shap = AsyncMock(
                return_value={
                    "base_value": 0.5,
                    "contributions": [],
                    "shap_sum": 0.3,
                }
            )
            mock_get_service.return_value = mock_service

            response = await explain_prediction(request, BackgroundTasks(), mock_user)

            assert response is not None

    @pytest.mark.asyncio
    async def test_explain_with_max_top_k(self, mock_user):
        """Test explanation with maximum top_k value."""
        request = ExplainRequest(
            patient_id="PAT-123",
            model_type=ModelType.PROPENSITY,
            top_k=20,  # Maximum allowed
        )

        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_features = AsyncMock(return_value={})
            mock_service.get_prediction = AsyncMock(
                return_value={
                    "prediction_class": "high",
                    "prediction_probability": 0.8,
                    "model_version_id": "v1",
                }
            )
            mock_service.compute_shap = AsyncMock(
                return_value={
                    "base_value": 0.5,
                    "contributions": [],
                    "shap_sum": 0.3,
                }
            )
            mock_get_service.return_value = mock_service

            await explain_prediction(request, BackgroundTasks(), mock_user)

            mock_service.compute_shap.assert_called_once()
            call_args = mock_service.compute_shap.call_args
            assert call_args.kwargs["top_k"] == 20

    @pytest.mark.asyncio
    async def test_batch_explain_empty_list(self, mock_user):
        """Test batch explanation with empty request list."""
        batch_request = BatchExplainRequest(requests=[])

        response = await explain_batch(batch_request, BackgroundTasks(), mock_user)

        assert response.total_requests == 0
        assert response.successful == 0
        assert response.failed == 0

    def test_prepare_numeric_features_with_all_types(self, shap_service):
        """Test numeric feature preparation with all data types."""
        features = {
            "int": 42,
            "float": 3.14,
            "bool_true": True,
            "bool_false": False,
            "string": "test",
            "empty_string": "",
            "none": None,
        }

        numeric = shap_service._prepare_numeric_features(features)

        assert all(isinstance(v, float) for v in numeric.values())
        assert numeric["int"] == 42.0
        assert numeric["bool_true"] == 1.0
        assert numeric["bool_false"] == 0.0
        assert numeric["none"] == 0.0


# =============================================================================
# Priority 1 OOM bounding: heavy-compute slot on SHAP endpoints
# =============================================================================


@pytest.fixture
def _heavy_compute_one_slot(monkeypatch):
    """Bound heavy compute to a single in-flight op and reset the limiter."""
    monkeypatch.setenv("HEAVY_COMPUTE_MAX_CONCURRENCY", "1")
    import src.api.dependencies.compute as compute_mod

    compute_mod._reset_limiter_cache_for_tests()
    yield compute_mod
    compute_mod._reset_limiter_cache_for_tests()


class TestExplainHeavyComputeBounding:
    """SHAP endpoints must hold a single heavy-compute slot per request."""

    @pytest.mark.asyncio
    async def test_explain_prediction_rejects_when_saturated(
        self, sample_explain_request, mock_user, _heavy_compute_one_slot
    ):
        """A saturated heavy-compute slot must reject /explain/predict fast
        (HeavyComputeSaturated) instead of running more SHAP compute."""
        from src.api.dependencies.compute import HeavyComputeSaturated

        limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
        limiter.acquire()  # occupy the single slot

        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            with pytest.raises(HeavyComputeSaturated):
                await explain_prediction(sample_explain_request, BackgroundTasks(), mock_user)

            # No SHAP compute may run while saturated.
            mock_service.compute_shap.assert_not_called()

    @pytest.mark.asyncio
    async def test_explain_prediction_releases_slot_on_success(
        self, sample_explain_request, mock_user, _heavy_compute_one_slot
    ):
        """The slot must be released after a successful prediction, leaving the
        response shape unchanged."""
        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_features = AsyncMock(return_value={"feat1": 1.0})
            mock_service.get_prediction = AsyncMock(
                return_value={
                    "prediction_class": "high_propensity",
                    "prediction_probability": 0.78,
                    "model_version_id": "v2.3.1",
                }
            )
            mock_service.compute_shap = AsyncMock(
                return_value={
                    "base_value": 0.42,
                    "contributions": [
                        FeatureContribution(
                            feature_name="feat1",
                            feature_value=1.0,
                            shap_value=0.15,
                            contribution_direction="positive",
                            contribution_rank=1,
                        )
                    ],
                    "shap_sum": 0.36,
                }
            )
            mock_service.generate_narrative = AsyncMock(return_value="narrative")
            mock_service.store_audit_record = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            response = await explain_prediction(
                sample_explain_request, BackgroundTasks(), mock_user
            )

            assert response.prediction_probability == 0.78
            assert len(response.top_features) == 1

            limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
            assert limiter.in_flight == 0

    @pytest.mark.asyncio
    async def test_explain_batch_rejects_when_saturated(self, mock_user, _heavy_compute_one_slot):
        """A saturated slot must reject the batch endpoint fast."""
        from src.api.dependencies.compute import HeavyComputeSaturated

        limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
        limiter.acquire()

        requests = [
            ExplainRequest(patient_id=f"PAT-{i}", model_type=ModelType.PROPENSITY) for i in range(3)
        ]
        batch_request = BatchExplainRequest(requests=requests, parallel=True)

        with pytest.raises(HeavyComputeSaturated):
            await explain_batch(batch_request, BackgroundTasks(), mock_user)

    @pytest.mark.asyncio
    async def test_explain_batch_holds_single_slot_for_whole_batch(
        self, mock_user, _heavy_compute_one_slot
    ):
        """With max concurrency 1, the batch must hold ONE slot for the whole fan-out
        (its inner per-item explain_prediction calls must NOT each try to re-acquire,
        which would self-reject every item). Proven by a successful 3-item batch under
        a 1-slot limit."""
        from src.api.routes.explain import ExplainResponse

        requests = [
            ExplainRequest(patient_id=f"PAT-{i}", model_type=ModelType.PROPENSITY) for i in range(3)
        ]
        batch_request = BatchExplainRequest(requests=requests, parallel=True)

        async def mock_explain(*args, **kwargs):
            return ExplainResponse(
                explanation_id="test-123",
                request_timestamp=datetime.now(timezone.utc),
                patient_id="P******",
                model_type=ModelType.PROPENSITY,
                model_version_id="v1",
                prediction_class="high",
                prediction_probability=0.8,
                top_features=[],
                shap_sum=0.3,
                computation_time_ms=100.0,
                audit_stored=False,
            )

        with patch("src.api.routes.explain.explain_prediction", side_effect=mock_explain):
            response = await explain_batch(batch_request, BackgroundTasks(), mock_user)

        assert response.successful == 3
        assert response.failed == 0

        limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
        assert limiter.in_flight == 0

    @pytest.mark.asyncio
    async def test_explain_batch_inner_real_explain_does_not_self_reject(
        self, mock_user, _heavy_compute_one_slot
    ):
        """The REAL inner explain_prediction must NOT re-acquire a slot when invoked
        from within a batch that already holds one (max concurrency 1). Exercises the
        actual explain_prediction body (only the SHAP *service* is mocked), proving the
        batch's single slot is shared, not contended item-by-item."""
        requests = [
            ExplainRequest(patient_id=f"PAT-{i}", model_type=ModelType.PROPENSITY) for i in range(2)
        ]
        batch_request = BatchExplainRequest(requests=requests, parallel=True)

        with patch("src.api.routes.explain.get_shap_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_features = AsyncMock(return_value={"feat1": 1.0})
            mock_service.get_prediction = AsyncMock(
                return_value={
                    "prediction_class": "high",
                    "prediction_probability": 0.8,
                    "model_version_id": "v1",
                }
            )
            mock_service.compute_shap = AsyncMock(
                return_value={"base_value": 0.5, "contributions": [], "shap_sum": 0.3}
            )
            mock_get_service.return_value = mock_service

            response = await explain_batch(batch_request, BackgroundTasks(), mock_user)

        # Both items succeed under a 1-slot limit -> the inner calls reused the batch slot.
        assert response.successful == 2
        assert response.failed == 0

        limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
        assert limiter.in_flight == 0
