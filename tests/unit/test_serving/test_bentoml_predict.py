"""Tests for the standalone BentoML serving service (Block 3A).

Covers:
  - Feast endpoint resolution precedence (FEAST_HTTP_ENDPOINT > FEAST_URL > default).
  - PredictionInput's two paths: direct features vs entity_ids + feature_view.
  - The Feast HTTP path: payload shape, response parsing, telemetry tag.
  - Backward compatibility: the existing direct-features path still works and
    produces ``feature_source='user_provided'``.

External boundaries we mock here:
  - ``bentoml`` (test conftest installs a stub — the host venv has no bentoml).
  - ``httpx.AsyncClient`` (Feast HTTP boundary; mocking it satisfies the
    Tier-0 "no mocks of business logic" rule because httpx is an external
    HTTP transport, not project code).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Endpoint resolution
# =============================================================================


class TestResolveFeastEndpoint:
    """Precedence: FEAST_HTTP_ENDPOINT > FEAST_URL > default."""

    def test_default_when_no_env(self, serving_module: Any) -> None:
        assert serving_module._resolve_feast_endpoint() == "http://feast:6566"

    def test_uses_feast_url_when_only_url_set(
        self, serving_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FEAST_URL", "http://feast-staging:6566")
        assert serving_module._resolve_feast_endpoint() == "http://feast-staging:6566"

    def test_prefers_http_endpoint_over_url(
        self, serving_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FEAST_HTTP_ENDPOINT", "http://feast-prod:7000")
        monkeypatch.setenv("FEAST_URL", "http://feast-staging:6566")
        assert serving_module._resolve_feast_endpoint() == "http://feast-prod:7000"


# =============================================================================
# PredictionInput schema
# =============================================================================


class TestPredictionInputSchema:
    """The two-path schema must accept both legacy and Feast-routed payloads."""

    def test_legacy_features_only(self, serving_module: Any) -> None:
        """The original contract — features as a matrix — still validates."""
        inp = serving_module.PredictionInput(features=[[0.1, 0.2, 0.3]])
        assert inp.features == [[0.1, 0.2, 0.3]]
        assert inp.entity_ids is None
        assert inp.feature_view is None

    def test_feast_path_optional_features(self, serving_module: Any) -> None:
        """When entity_ids + feature_view are set, features may be omitted."""
        inp = serving_module.PredictionInput(
            entity_ids=["P001", "P002"],
            feature_view="patient_engagement_features",
        )
        assert inp.entity_ids == ["P001", "P002"]
        assert inp.feature_view == "patient_engagement_features"
        assert inp.features == []  # default
        assert inp.entity_key == "patient_id"

    def test_entity_key_overridable(self, serving_module: Any) -> None:
        inp = serving_module.PredictionInput(
            entity_ids=["HCP1"],
            feature_view="hcp_features",
            entity_key="hcp_id",
        )
        assert inp.entity_key == "hcp_id"


# =============================================================================
# Direct features path (backward compatibility)
# =============================================================================


class TestDirectFeaturesPath:
    """The legacy ``features`` path must remain unchanged."""

    @pytest.mark.asyncio
    async def test_predict_with_features_uses_user_provided_tag(self, serving_module: Any) -> None:
        service = serving_module.E2IModelService()

        # Stub the prediction kernel so we don't need a real model.
        captured: dict[str, Any] = {}

        def _fake_run(features: Any, feature_source: Any = None, **_kwargs: Any) -> Any:
            # **_kwargs absorbs the routed-bundle keywords (#39 multi-model:
            # model=/preprocessor=/feature_columns=/model_tag=) the predict
            # method now threads through; this stub only asserts the tag/features.
            captured["features"] = features
            captured["feature_source"] = feature_source
            return serving_module.PredictionOutput(
                predictions=[0.42],
                probabilities=[0.7],
                model_id="stub",
                prediction_time_ms=1.0,
                is_mock=False,
                feature_source=feature_source,
            )

        service._run_prediction = _fake_run  # type: ignore[method-assign]

        inp = serving_module.PredictionInput(features=[[0.1, 0.2]])
        out = await service.predict(inp)

        assert captured["features"] == [[0.1, 0.2]]
        assert captured["feature_source"] == "user_provided"
        assert out.feature_source == "user_provided"
        assert out.predictions == [0.42]


# =============================================================================
# Feast HTTP path (the new behavior)
# =============================================================================


class _FakeAsyncContext:
    """Minimal async context manager wrapping a mock httpx client."""

    def __init__(self, client: MagicMock) -> None:
        self._client = client

    async def __aenter__(self) -> MagicMock:
        return self._client

    async def __aexit__(self, *_args: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_predict_with_entity_ids_uses_feast(
    serving_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: entity_ids + feature_view → Feast HTTP → model.

    Asserts:
      - The HTTP call is shaped as Feast 0.43 expects (POST /get-online-features
        with features=[view:*], entities={key: [...]}, full_feature_names=False).
      - The endpoint URL respects FEAST_HTTP_ENDPOINT precedence.
      - The returned PredictionOutput carries feature_source='feast_online'.
      - The fetched feature matrix flows into _run_prediction (NOT the empty
        ``features`` list from the request).
    """
    monkeypatch.setenv("FEAST_HTTP_ENDPOINT", "http://feast-test:7777")

    # Mocked Feast response: column-oriented (Feast 0.43 native shape).
    feast_body = {
        "metadata": {
            "feature_names": [
                "patient_id",
                "days_since_last_hcp_visit",
                "total_hcp_interactions_90d",
            ]
        },
        "results": [
            {"values": ["P001", "P002"], "statuses": ["PRESENT", "PRESENT"]},
            {"values": [10.0, 25.0], "statuses": ["PRESENT", "PRESENT"]},
            {"values": [3.0, 7.0], "statuses": ["PRESENT", "PRESENT"]},
        ],
    }

    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock(return_value=None)
    response_mock.json = MagicMock(return_value=feast_body)

    http_client_mock = MagicMock()
    http_client_mock.post = AsyncMock(return_value=response_mock)

    captured: dict[str, Any] = {}

    def _fake_run(features: Any, feature_source: Any = None, **_kwargs: Any) -> Any:
        # **_kwargs absorbs the routed-bundle keywords (#39 multi-model).
        captured["features"] = features
        captured["feature_source"] = feature_source
        return serving_module.PredictionOutput(
            predictions=[0.6, 0.7],
            probabilities=[0.6, 0.7],
            model_id="stub",
            prediction_time_ms=1.0,
            is_mock=False,
            feature_source=feature_source,
        )

    with patch(
        "httpx.AsyncClient",
        return_value=_FakeAsyncContext(http_client_mock),
    ):
        service = serving_module.E2IModelService()
        service._run_prediction = _fake_run  # type: ignore[method-assign]

        inp = serving_module.PredictionInput(
            entity_ids=["P001", "P002"],
            feature_view="patient_engagement_features",
            entity_key="patient_id",
        )
        out = await service.predict(inp)

    # --- HTTP boundary assertions ---
    http_client_mock.post.assert_awaited_once()
    call_args = http_client_mock.post.await_args
    posted_url = call_args.args[0]
    posted_payload = call_args.kwargs["json"]

    assert posted_url == "http://feast-test:7777/get-online-features"
    assert posted_payload == {
        "features": ["patient_engagement_features:*"],
        "entities": {"patient_id": ["P001", "P002"]},
        "full_feature_names": False,
    }

    # --- Inference-input assertions ---
    # patient_id column is dropped from the matrix; only feature columns remain.
    assert captured["features"] == [[10.0, 3.0], [25.0, 7.0]]
    assert captured["feature_source"] == "feast_online"
    assert out.feature_source == "feast_online"
    assert out.predictions == [0.6, 0.7]


@pytest.mark.asyncio
async def test_feast_path_propagates_http_failure(
    serving_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Feast HTTP errors, the service raises RuntimeError (no silent zeros)."""
    monkeypatch.setenv("FEAST_HTTP_ENDPOINT", "http://feast-test:7777")

    http_client_mock = MagicMock()
    http_client_mock.post = AsyncMock(side_effect=Exception("connection refused"))

    with patch(
        "httpx.AsyncClient",
        return_value=_FakeAsyncContext(http_client_mock),
    ):
        service = serving_module.E2IModelService()
        with pytest.raises(RuntimeError, match="Feast online-features call failed"):
            await service._fetch_features_from_feast(
                entity_ids=["P001"],
                feature_view="patient_engagement_features",
                entity_key="patient_id",
            )


@pytest.mark.asyncio
async def test_feast_path_fails_closed_on_null_or_non_numeric_values(
    serving_module: Any,
) -> None:
    """Null / non-numeric Feast values FAIL CLOSED (RuntimeError) — they are NOT
    zero-filled and labeled feast_online (the #576/#532 audit-grade harm). A
    fabricated 'feast_online' vector must never reach the model."""
    feast_body = {
        "metadata": {"feature_names": ["patient_id", "score", "label"]},
        "results": [
            {"values": ["P001"], "statuses": ["PRESENT"]},
            {"values": [None], "statuses": ["NOT_FOUND"]},
            {"values": ["not-a-number"], "statuses": ["PRESENT"]},
        ],
    }
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock(return_value=None)
    response_mock.json = MagicMock(return_value=feast_body)

    http_client_mock = MagicMock()
    http_client_mock.post = AsyncMock(return_value=response_mock)

    with patch(
        "httpx.AsyncClient",
        return_value=_FakeAsyncContext(http_client_mock),
    ):
        service = serving_module.E2IModelService()
        with pytest.raises(RuntimeError) as ei:
            await service._fetch_features_from_feast(
                entity_ids=["P001"],
                feature_view="patient_engagement_features",
                entity_key="patient_id",
            )
    # The error names the offending feature(s) and refuses the fabrication.
    assert "refusing to fabricate" in str(ei.value)
    assert "score" in str(ei.value) and "label" in str(ei.value)


@pytest.mark.asyncio
async def test_feast_path_real_zero_is_not_treated_as_null(
    serving_module: Any,
) -> None:
    """A genuine 0.0 from Feast is a legitimate value — it must NOT trip the
    fail-closed guard (only null/non-numeric do)."""
    feast_body = {
        "metadata": {"feature_names": ["patient_id", "score", "rate"]},
        "results": [
            {"values": ["P001"], "statuses": ["PRESENT"]},
            {"values": [0.0], "statuses": ["PRESENT"]},
            {"values": [0.0], "statuses": ["PRESENT"]},
        ],
    }
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock(return_value=None)
    response_mock.json = MagicMock(return_value=feast_body)

    http_client_mock = MagicMock()
    http_client_mock.post = AsyncMock(return_value=response_mock)

    with patch(
        "httpx.AsyncClient",
        return_value=_FakeAsyncContext(http_client_mock),
    ):
        service = serving_module.E2IModelService()
        matrix = await service._fetch_features_from_feast(
            entity_ids=["P001"],
            feature_view="patient_engagement_features",
            entity_key="patient_id",
        )

    assert matrix == [[0.0, 0.0]]


@pytest.mark.asyncio
async def test_preprocessor_failure_fails_closed_no_raw_predict(
    serving_module: Any,
) -> None:
    """When a bundled preprocessor exists but its transform raises, the service
    FAILS CLOSED — it does NOT silently predict on the raw (un-preprocessed)
    matrix, which would emit a plausible-but-wrong audit-grade prediction."""
    service = serving_module.E2IModelService()

    class _Model:
        def predict(self, arr):
            raise AssertionError("model.predict must not run on raw input")

    class _BadPreprocessor:
        def transform(self, x):
            raise ValueError("transform boom")

    service._model = _Model()
    service._preprocessor = _BadPreprocessor()
    service._feature_columns = None

    import numpy as np

    with pytest.raises(RuntimeError) as ei:
        service._apply_preprocessor(np.array([[1.0, 2.0]]))
    assert "Preprocessor transform failed" in str(ei.value)


@pytest.mark.asyncio
async def test_feast_path_orders_by_model_feature_columns_not_feast_order(
    serving_module: Any,
) -> None:
    """The matrix must be built in the MODEL's feature_columns order, not Feast's
    response order — else a mis-ordered (plausible-but-wrong) vector would be
    labeled feast_online. Feast returns [patient_id, b, a]; the model expects
    [a, b]; the row must be [a_value, b_value]."""
    feast_body = {
        "metadata": {"feature_names": ["patient_id", "b", "a"]},
        "results": [
            {"values": ["P001"], "statuses": ["PRESENT"]},
            {"values": [20.0], "statuses": ["PRESENT"]},  # b
            {"values": [10.0], "statuses": ["PRESENT"]},  # a
        ],
    }
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock(return_value=None)
    response_mock.json = MagicMock(return_value=feast_body)
    http_client_mock = MagicMock()
    http_client_mock.post = AsyncMock(return_value=response_mock)

    with patch("httpx.AsyncClient", return_value=_FakeAsyncContext(http_client_mock)):
        service = serving_module.E2IModelService()
        service._feature_columns = ["a", "b"]  # model's authoritative order
        matrix = await service._fetch_features_from_feast(
            entity_ids=["P001"],
            feature_view="patient_engagement_features",
            entity_key="patient_id",
        )

    # Ordered as the model expects: a (10.0) then b (20.0), NOT Feast's b, a.
    assert matrix == [[10.0, 20.0]]


@pytest.mark.asyncio
async def test_feast_path_fails_closed_when_expected_feature_absent(
    serving_module: Any,
) -> None:
    """A model-expected feature absent from the Feast payload FAILS CLOSED —
    it is not silently dropped or zero-filled."""
    feast_body = {
        "metadata": {"feature_names": ["patient_id", "a"]},  # no "b"
        "results": [
            {"values": ["P001"], "statuses": ["PRESENT"]},
            {"values": [10.0], "statuses": ["PRESENT"]},
        ],
    }
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock(return_value=None)
    response_mock.json = MagicMock(return_value=feast_body)
    http_client_mock = MagicMock()
    http_client_mock.post = AsyncMock(return_value=response_mock)

    with patch("httpx.AsyncClient", return_value=_FakeAsyncContext(http_client_mock)):
        service = serving_module.E2IModelService()
        service._feature_columns = ["a", "b"]
        with pytest.raises(RuntimeError) as ei:
            await service._fetch_features_from_feast(
                entity_ids=["P001"],
                feature_view="patient_engagement_features",
                entity_key="patient_id",
            )
    assert "b" in str(ei.value)
    assert "missing" in str(ei.value)
