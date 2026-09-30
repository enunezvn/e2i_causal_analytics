"""#2343: ``POST /predict/{model_name}/batch`` must score the model NAMED in the URL.

The route used to post ``{"batch_id", "features"}`` with NO ``model_name``. The
goldstd sidecar is multi-model and routes by ``input_data.model_name``; with the
key absent it scores its DEFAULT model, so a batch for a goldstd model returned
plausible probabilities from a different model while the response echoed the
requested name.

These tests drive the REAL route and the REAL ``BentoMLClient`` over the wire.
Only the far side of the HTTP boundary is faked: a small multi-model sidecar on
``httpx.MockTransport`` that mirrors the served contract (``/model_info`` and
``/predict`` / ``/predict_batch`` route by ``input_data.model_name`` and fall
back to a default model when it is absent; goldstd bundles take ``raw_features``
and fail closed with a 200 + ``error`` on an encoded matrix, as PR #2339 does).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies.bentoml_client import (
    BentoMLClient,
    BentoMLClientConfig,
    get_bentoml_client,
)
from src.api.main import app

GOLDSTD = "initiation_kisqali_goldstd_lr_v1"
LEGACY = "tier0_legacy_positional"
KEEP_COLUMNS = ["disease_severity", "academic_hcp", "geographic_region"]
LEGACY_COLUMNS = ["x1", "x2"]

# Per-model constant score so a wrong-model answer is unmistakable.
DEFAULT_SCORE = 0.99
GOLDSTD_BASE = 0.10
LEGACY_SCORE = 0.42


def _goldstd_score(row: Dict[str, Any]) -> float:
    # Deterministic, row-dependent: batch and single must agree row-for-row.
    return round(GOLDSTD_BASE + 0.01 * float(row["disease_severity"]), 6)


class FakeMultiModelSidecar:
    """Boundary fake for the multi-model BentoML sidecar."""

    def __init__(self) -> None:
        self.batch_payloads: List[Dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        inp = body.get("input_data") or {}
        name = inp.get("model_name")
        path = request.url.path

        if path.endswith("/model_info"):
            if name == GOLDSTD:
                return httpx.Response(
                    200,
                    json={
                        "model_id": f"{GOLDSTD}:v1",
                        "model_loaded": True,
                        "keep_columns": KEEP_COLUMNS,
                        "feature_columns": ["enc_0", "enc_1", "enc_2", "enc_3"],
                    },
                )
            if name == LEGACY:
                return httpx.Response(
                    200,
                    json={
                        "model_id": f"{LEGACY}:v1",
                        "model_loaded": True,
                        "feature_columns": LEGACY_COLUMNS,
                    },
                )
            return httpx.Response(
                200,
                json={"model_id": "default:v0", "model_loaded": True, "feature_columns": ["d"]},
            )

        if path.endswith("/predict_batch"):
            self.batch_payloads.append(inp)
            rows_raw = inp.get("raw_features") or []
            rows_enc = inp.get("features") or []
            n = len(rows_raw) or len(rows_enc)
            if name == GOLDSTD:
                if not rows_raw:
                    return httpx.Response(
                        200,
                        json={
                            "batch_id": inp.get("batch_id"),
                            "total_samples": n,
                            "predictions": [],
                            "probabilities": [],
                            "processing_time_ms": 1.0,
                            "error": f"{GOLDSTD} encodes raw covariates; send raw_features",
                        },
                    )
                probs = [_goldstd_score(r) for r in rows_raw]
                return httpx.Response(
                    200,
                    json={
                        "batch_id": inp.get("batch_id"),
                        "total_samples": n,
                        "predictions": [1.0 if p >= 0.5 else 0.0 for p in probs],
                        "probabilities": probs,
                        "processing_time_ms": 1.0,
                        "model_id": f"{GOLDSTD}:v1",
                    },
                )
            score = LEGACY_SCORE if name == LEGACY else DEFAULT_SCORE
            model_id = f"{LEGACY}:v1" if name == LEGACY else "default:v0"
            return httpx.Response(
                200,
                json={
                    "batch_id": inp.get("batch_id"),
                    "total_samples": n,
                    "predictions": [score] * n,
                    "probabilities": [],
                    "processing_time_ms": 1.0,
                    "model_id": model_id,
                },
            )

        if path.endswith("/predict"):
            rows_raw = inp.get("raw_features") or []
            if name == GOLDSTD and rows_raw:
                p = _goldstd_score(rows_raw[0])
                return httpx.Response(
                    200,
                    json={
                        "predictions": [1.0 if p >= 0.5 else 0.0],
                        "probabilities": [p],
                        "model_id": f"{GOLDSTD}:v1",
                    },
                )
            score = LEGACY_SCORE if name == LEGACY else DEFAULT_SCORE
            return httpx.Response(200, json={"predictions": [score], "model_id": "x"})

        return httpx.Response(404, json={"detail": f"unknown path {path}"})


@pytest.fixture
def sidecar_client():
    sidecar = FakeMultiModelSidecar()
    real = BentoMLClient(
        BentoMLClientConfig(base_url="http://sidecar.test", enable_tracing=False, max_retries=1)
    )
    real._client = httpx.AsyncClient(transport=httpx.MockTransport(sidecar.handler))
    real._initialized = True
    app.dependency_overrides[get_bentoml_client] = lambda: real
    try:
        yield sidecar
    finally:
        app.dependency_overrides.clear()


def _goldstd_rows() -> List[Dict[str, Any]]:
    return [
        {"disease_severity": 5.6, "academic_hcp": 0, "geographic_region": "northeast"},
        {"disease_severity": 2.0, "academic_hcp": 1, "geographic_region": "south"},
    ]


@pytest.mark.unit
class TestBatchScoresTheNamedModel:
    def test_goldstd_batch_is_routed_to_the_named_model(self, sidecar_client):
        """The named goldstd model, not the sidecar default, scores the batch."""
        rows = _goldstd_rows()
        resp = TestClient(app).post(
            f"/api/models/predict/{GOLDSTD}/batch",
            json={"instances": [{"features": r} for r in rows]},
        )
        assert resp.status_code == 200, resp.text

        sent = sidecar_client.batch_payloads[-1]
        assert sent.get("model_name") == GOLDSTD
        # goldstd bundles encode raw covariates server-side (mirrors single /predict).
        assert sent.get("raw_features") == rows

        body = resp.json()
        got = [p["probabilities"]["positive_class"] for p in body["predictions"]]
        assert got == [_goldstd_score(r) for r in rows]
        assert DEFAULT_SCORE not in got

    def test_legacy_batch_sends_model_name_with_positional_rows(self, sidecar_client):
        resp = TestClient(app).post(
            f"/api/models/predict/{LEGACY}/batch",
            json={"instances": [{"features": {"x1": 1.0, "x2": 2.0}}]},
        )
        assert resp.status_code == 200, resp.text
        sent = sidecar_client.batch_payloads[-1]
        assert sent.get("model_name") == LEGACY
        assert sent.get("features") == [[1.0, 2.0]]
        assert [p["prediction"] for p in resp.json()["predictions"]] == [LEGACY_SCORE]

    def test_batch_equals_single_for_same_rows_and_model(self, sidecar_client):
        """Issue #2343 item 3, at unit scope: batch and single agree row-for-row."""
        rows = _goldstd_rows()
        tc = TestClient(app)
        batch = tc.post(
            f"/api/models/predict/{GOLDSTD}/batch",
            json={"instances": [{"features": r} for r in rows]},
        )
        assert batch.status_code == 200, batch.text
        singles = []
        for r in rows:
            s = tc.post(f"/api/models/predict/{GOLDSTD}", json={"features": r})
            assert s.status_code == 200, s.text
            singles.append(s.json())
        for b, s in zip(batch.json()["predictions"], singles, strict=True):
            assert b["prediction"] == s["prediction"]
            assert b["probabilities"] == s["probabilities"]

    def test_goldstd_batch_missing_covariate_fails_closed_422(self, sidecar_client):
        bad = {"disease_severity": 5.6, "academic_hcp": 0}  # geographic_region missing
        resp = TestClient(app).post(
            f"/api/models/predict/{GOLDSTD}/batch",
            json={"instances": [{"features": bad}]},
        )
        assert resp.status_code == 422, resp.text
        assert "geographic_region" in resp.text
        assert sidecar_client.batch_payloads == []

    def test_service_error_is_surfaced_not_an_empty_200(self, sidecar_client, monkeypatch):
        """A 200 + ``error`` from the sidecar (e.g. a named bundle refusing an encoded
        matrix, PR #2339) must not become a 200 with zero predictions."""

        async def _info_without_keep_columns(self, model_name):  # noqa: ANN001
            return {"model_id": f"{GOLDSTD}:v1", "feature_columns": LEGACY_COLUMNS}

        monkeypatch.setattr(BentoMLClient, "get_model_info", _info_without_keep_columns)
        resp = TestClient(app).post(
            f"/api/models/predict/{GOLDSTD}/batch",
            json={"instances": [{"features": {"x1": 1.0, "x2": 2.0}}]},
        )
        assert resp.status_code == 422, resp.text
        assert "raw_features" in resp.text
