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

import pandas as pd
import pytest

from src.api.routes import predictions as pred


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


def test_resolve_cohort_spec_rejects_non_goldstd_model():
    """Unresolvable model name -> ValueError (the endpoint maps this to 422)."""
    with pytest.raises(ValueError):
        pred._resolve_cohort_spec("not_a_goldstd_model")


# =============================================================================
# Full-app submit -> poll endpoint (imports app LAZILY -> CI; -k skip locally)
# =============================================================================
def _fake_holdout_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "disease_severity": [5.6, 2.1, 8.0],
            "academic_hcp": [0, 1, 0],
            "geographic_region": ["northeast", "west", "south"],
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
