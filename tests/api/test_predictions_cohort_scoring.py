"""Cohort scoring (#cohort-scoring): score a model's holdout cohort, rank targets.

The predictive-analytics page is data-driven: instead of hand-typing one feature
row, the user scores the model's REAL holdout cohort and gets a ranked list of
targets + a probability distribution. The backend:

  - resolves cohort+brand from the model name,
  - loads the model's OWN holdout split via FeatureBuilder.load_frame(splits=["holdout"]),
  - scores it through the BentoML raw-covariate BATCH path in chunks (<=1000),
  - ranks rows by predicted probability (top-N) and summarizes the distribution.

The pure ranking/distribution + chunked scorer tests import ONLY the light
``predictions`` module. The full-app submit->poll endpoint test imports
``src.api.main.app`` LAZILY (inside the test) so it runs in CI; on a
memory-pressured box it is skipped via -k selection rather than OOM-killing the
whole module import. The raw-batch serving path itself is covered faithfully in
tests/unit/test_serving/test_bentoml_batch_raw_covariates.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.routes import predictions as pred


# =============================================================================
# _native scalar conversion (no mocks, light import)
# =============================================================================
class TestNativeScalar:
    """NaN -> None so a NULL holdout covariate rides the serving contract's
    __isna missingness path instead of crashing httpx's ``allow_nan=False``
    JSON encoding (the 2026-07-14 'Out of range float values' cohort failure)."""

    def test_python_nan_becomes_none(self):
        assert pred._native(float("nan")) is None

    def test_numpy_nan_becomes_none(self):
        import numpy as np

        assert pred._native(np.float64("nan")) is None

    def test_ordinary_values_pass_through(self):
        import numpy as np

        assert pred._native(np.float64(0.5)) == 0.5
        assert pred._native(np.int64(3)) == 3
        assert pred._native("northeast") == "northeast"
        assert pred._native(None) is None


# =============================================================================
# Pure ranking + distribution (no mocks, light import)
# =============================================================================
class TestCohortRanking:
    def test_ranks_desc_and_caps_top_n(self):
        entity_ids = ["a", "b", "c", "d"]
        covariate_rows = [{"x": i} for i in range(4)]
        probabilities = [0.2, 0.95, 0.5, 0.71]
        top_rows, _dist = pred._cohort_ranking(entity_ids, covariate_rows, probabilities, top_n=2)
        assert [r.entity_id for r in top_rows] == ["b", "d"]  # 0.95, 0.71 desc
        assert top_rows[0].probability == 0.95
        assert top_rows[0].covariates == {"x": 1}
        assert len(top_rows) == 2  # capped at top_n

    def test_distribution_summarizes_all_rows(self):
        entity_ids = ["a", "b", "c", "d"]
        covariate_rows = [{} for _ in range(4)]
        probabilities = [0.05, 0.15, 0.95, 0.99]
        _top, dist = pred._cohort_ranking(entity_ids, covariate_rows, probabilities, top_n=10)
        assert dist.n == 4
        assert abs(dist.mean - 0.535) < 1e-9
        assert sum(dist.bin_counts) == 4  # ALL rows, not just top_n
        assert len(dist.bin_edges) == len(dist.bin_counts) + 1
        # 10 fixed [0,1] bins: 0.05 -> bin 0, 0.15 -> bin 1, 0.95 & 0.99 -> last bin.
        assert dist.bin_counts[0] == 1
        assert dist.bin_counts[-1] == 2


# =============================================================================
# Chunked scorer (fake async client, light import)
# =============================================================================
class _FakeClient:
    def __init__(self, prob_per_row: float = 0.5, error: str | None = None):
        self.prob = prob_per_row
        self.error = error
        self.calls: List[int] = []

    async def predict_batch(self, model_name: str, batch_data: Dict[str, Any]):
        rows = batch_data["raw_features"]
        self.calls.append(len(rows))
        if self.error:
            return {"error": self.error, "predictions": [], "probabilities": []}
        n = len(rows)
        return {"predictions": [self.prob] * n, "probabilities": [self.prob] * n}


@pytest.mark.asyncio
class TestScoreCohortChunks:
    async def test_chunks_at_boundary_and_preserves_order(self):
        fake = _FakeClient(prob_per_row=0.3)
        raw = [{"i": i} for i in range(2500)]
        probs = await pred._score_cohort_chunks(fake, "m", raw, chunk_size=1000)
        assert fake.calls == [1000, 1000, 500]  # 3 chunks
        assert len(probs) == 2500
        assert all(p == 0.3 for p in probs)

    async def test_service_error_fails_closed(self):
        fake = _FakeClient(error="Model 'm' not found")
        with pytest.raises(Exception):
            await pred._score_cohort_chunks(fake, "m", [{"i": 0}], chunk_size=1000)

    async def test_length_mismatch_fails_closed(self):
        class _BadClient:
            async def predict_batch(self, model_name, batch_data):
                return {"predictions": [0.5], "probabilities": [0.5]}  # 1 for 2 rows

        with pytest.raises(Exception):
            await pred._score_cohort_chunks(
                _BadClient(), "m", [{"i": 0}, {"i": 1}], chunk_size=1000
            )


# =============================================================================
# T5: serving-schema-drift hardening — the BentoML client (predict_batch) RAISES
# the raw httpx error on a non-2xx response (bentoml_client.py:387-390 does
# response.raise_for_status() then `circuit.record_failure(); raise`). It does
# NOT fold a 400 into the {"error": ...} body, so the old `result.get("error")`
# branch never fires for a stale-schema 400 — the raw httpx.HTTPStatusError used
# to propagate as an un-actionable job failure. The scorer must translate it into
# a 502 whose detail tells the operator the remediation (restart the bentoml
# service so it reloads the current serving schema).
# =============================================================================
def _http_status_error(code: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://bentoml:3000/predict_batch")
    response = httpx.Response(code, request=request, text=body)
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


@pytest.mark.asyncio
class TestScoreCohortChunksHttpError:
    async def test_stale_schema_400_raises_actionable_502(self):
        class _Raises400:
            async def predict_batch(self, model_name, batch_data):
                raise _http_status_error(400, "field required: 'features'")

        with pytest.raises(HTTPException) as ei:
            await pred._score_cohort_chunks(_Raises400(), "m", [{"i": 0}], chunk_size=1000)
        assert ei.value.status_code == 502
        detail = str(ei.value.detail).lower()
        # Actionable: names the stale-schema cause AND the bentoml remediation.
        assert "schema" in detail or "stale" in detail, detail
        assert "bentoml" in detail or "restart" in detail, detail

    async def test_422_validation_also_actionable_502(self):
        class _Raises422:
            async def predict_batch(self, model_name, batch_data):
                raise _http_status_error(422, "unprocessable: schema mismatch")

        with pytest.raises(HTTPException) as ei:
            await pred._score_cohort_chunks(_Raises422(), "m", [{"i": 0}], chunk_size=1000)
        assert ei.value.status_code == 502
        detail = str(ei.value.detail).lower()
        assert "bentoml" in detail or "restart" in detail, detail

    async def test_other_5xx_status_raises_502_without_stale_claim(self):
        """A 503 is NOT a schema-drift signal — it must still 502 but must NOT
        misattribute it to a stale schema (that would send the operator down the
        wrong remediation)."""

        class _Raises503:
            async def predict_batch(self, model_name, batch_data):
                raise _http_status_error(503, "service unavailable")

        with pytest.raises(HTTPException) as ei:
            await pred._score_cohort_chunks(_Raises503(), "m", [{"i": 0}], chunk_size=1000)
        assert ei.value.status_code == 502
        detail = str(ei.value.detail).lower()
        assert "stale" not in detail, detail
        assert "503" in detail

    async def test_unreachable_service_raises_502(self):
        class _ConnError:
            async def predict_batch(self, model_name, batch_data):
                raise httpx.ConnectError(
                    "connection refused",
                    request=httpx.Request("POST", "http://bentoml:3000/predict_batch"),
                )

        with pytest.raises(HTTPException) as ei:
            await pred._score_cohort_chunks(_ConnError(), "m", [{"i": 0}], chunk_size=1000)
        assert ei.value.status_code == 502
        detail = str(ei.value.detail).lower()
        assert "reach" in detail or "unreachable" in detail or "running" in detail, detail


def test_resolve_cohort_spec_rejects_non_goldstd_model():
    """Unresolvable model name -> ValueError (the endpoint maps this to 422)."""
    with pytest.raises(ValueError):
        pred._resolve_cohort_spec("not_a_goldstd_model")


# =============================================================================
# Full-app submit -> poll endpoint (imports app LAZILY -> CI; -k skip locally)
# =============================================================================
def _fake_holdout_frame() -> pd.DataFrame:
    # Full T11 7-covariate set (cohort_spec._BASE7). p2's comorbidity_burden is
    # NaN on purpose: a NULL holdout covariate must ride the NaN->None path in
    # _native (JSON has no NaN) instead of failing the whole job — the
    # 2026-07-14 production incident.
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "disease_severity": [5.6, 2.1, 8.0],
            "academic_hcp": [0, 1, 0],
            "geographic_region": ["northeast", "west", "south"],
            "insurance_type": ["commercial", "medicare", "medicaid"],
            "age_at_diagnosis": [54, 71, 39],
            "comorbidity_burden": [2.0, float("nan"), 1.0],
            "prior_therapy_lines": [0, 3, 1],
        }
    )


class TestCohortScoreEndpoint:
    """Heavy: imports the full FastAPI app. Runs in CI; locally select with
    ``-k Ranking or Chunks`` to avoid the app import on a memory-pressured box."""

    def test_submit_then_poll_completes_with_ranked_rows(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.api.dependencies.bentoml_client import get_bentoml_client
        from src.api.main import app

        async def _fake_load_frame(self, db, *, splits=None, before_month=None, include_real=False):
            assert splits == ["holdout"]
            return _fake_holdout_frame()

        async def _fake_db_client():
            return None

        monkeypatch.setattr(
            "src.mlops.gold_standard_eval.feature_builder.FeatureBuilder.load_frame",
            _fake_load_frame,
        )
        monkeypatch.setattr(pred, "_resolve_db_client", _fake_db_client)

        client_mock = _FakeClient(prob_per_row=0.7)
        app.dependency_overrides[get_bentoml_client] = lambda: client_mock
        try:
            tc = TestClient(app)
            submit = tc.post("/api/models/predict/initiation_kisqali_goldstd_lr_v1/cohort?top_n=2")
            assert submit.status_code == 200, submit.text
            body = submit.json()
            job_id = body["job_id"]
            assert body["cohort"] == "initiation"
            assert body["brand"] == "Kisqali"

            # TestClient runs the BackgroundTask before returning -> poll is done.
            poll = tc.get(f"/api/models/predict/initiation_kisqali_goldstd_lr_v1/cohort/{job_id}")
            assert poll.status_code == 200, poll.text
            done = poll.json()
            assert done["status"] == "completed", done
            assert done["n_scored"] == 3
            assert done["out_of_sample"] is True
            assert done["feature_source"] == "holdout_synthetic"
            assert len(done["top_rows"]) == 2  # top_n
            assert "disease_severity" in done["top_rows"][0]["covariates"]
            assert done["distribution"]["n"] == 3
            # The NaN covariate arrived as JSON null (None), not a crashed job.
            by_id = {r["entity_id"]: r["covariates"] for r in done["top_rows"]}
            if "p2" in by_id:
                assert by_id["p2"]["comorbidity_burden"] is None
        finally:
            app.dependency_overrides.clear()

    def test_unresolvable_model_returns_422(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.api.dependencies.bentoml_client import get_bentoml_client
        from src.api.main import app

        app.dependency_overrides[get_bentoml_client] = lambda: _FakeClient()
        try:
            tc = TestClient(app)
            resp = tc.post("/api/models/predict/not_a_goldstd_model/cohort")
            assert resp.status_code == 422, resp.text
        finally:
            app.dependency_overrides.clear()


# =============================================================================
# Cohort-level SHAP drivers (2026-07-09): mean |SHAP| over the sampled top rows,
# aggregated for the Strategic Interpretation. Pure aggregation + best-effort
# sampler (a SHAP failure must never fail the cohort job or fabricate drivers).
# =============================================================================
class TestAggregateShapDrivers:
    def test_empty_input_yields_no_drivers(self):
        assert pred.aggregate_shap_drivers([]) == []

    def test_ranks_by_mean_abs_and_directions(self):
        rows = [
            {"disease_severity": -1.2, "insurance_type_commercial": 0.4, "academic_hcp": 0.0},
            {"disease_severity": -0.8, "insurance_type_commercial": 0.3, "academic_hcp": 0.0},
        ]
        drivers = pred.aggregate_shap_drivers(rows)
        # academic_hcp contributes ~0 everywhere -> dropped, not a fake driver.
        assert [d.feature for d in drivers] == ["disease_severity", "insurance_type_commercial"]
        assert drivers[0].direction == "decreases"
        assert abs(drivers[0].importance - 1.0) < 1e-9
        assert drivers[1].direction == "increases"

    def test_cancelling_contributions_are_mixed_not_directional(self):
        drivers = pred.aggregate_shap_drivers([{"region": 1.0}, {"region": -1.0}])
        assert drivers[0].direction == "mixed"
        assert drivers[0].importance == 1.0

    def test_caps_at_top_k(self):
        rows = [{f"f{i}": float(i + 1) for i in range(15)}]
        drivers = pred.aggregate_shap_drivers(rows, top_k=10)
        assert len(drivers) == 10
        assert drivers[0].feature == "f14"


class _FakeShapClient:
    def __init__(self, per_row: Dict[str, float] | None = None, error: str | None = None):
        self.per_row = per_row if per_row is not None else {"disease_severity": 0.4}
        self.error = error
        self.calls = 0

    async def get_shap(self, model_name: str, raw_features):
        self.calls += 1
        if self.error:
            return {"error": self.error, "shap_values": {}}
        return {"error": None, "shap_values": dict(self.per_row)}


@pytest.mark.asyncio
class TestSampleCohortDrivers:
    async def test_samples_top_rows_and_aggregates(self):
        fake = _FakeShapClient()
        rows = [
            pred.CohortScoredRow(entity_id=f"e{i}", probability=0.9, covariates={"x": i})
            for i in range(3)
        ]
        drivers, sampled = await pred._sample_cohort_drivers(fake, "m", rows)
        assert sampled == 3 and fake.calls == 3
        assert drivers[0].feature == "disease_severity"

    async def test_caps_at_driver_shap_rows(self):
        fake = _FakeShapClient()
        rows = [
            pred.CohortScoredRow(entity_id=f"e{i}", probability=0.9, covariates={})
            for i in range(pred._DRIVER_SHAP_ROWS + 25)
        ]
        _drivers, sampled = await pred._sample_cohort_drivers(fake, "m", rows)
        assert sampled == pred._DRIVER_SHAP_ROWS
        assert fake.calls == pred._DRIVER_SHAP_ROWS

    async def test_per_row_shap_errors_are_skipped(self):
        fake = _FakeShapClient(error="no bundle for model")
        rows = [pred.CohortScoredRow(entity_id="e0", probability=0.9, covariates={})]
        drivers, sampled = await pred._sample_cohort_drivers(fake, "m", rows)
        assert drivers == [] and sampled == 0

    async def test_client_exception_is_best_effort_empty(self):
        class _Boom:
            async def get_shap(self, model_name, raw_features):
                raise RuntimeError("serving down")

        rows = [pred.CohortScoredRow(entity_id="e0", probability=0.9, covariates={})]
        drivers, sampled = await pred._sample_cohort_drivers(_Boom(), "m", rows)
        assert drivers == [] and sampled == 0
