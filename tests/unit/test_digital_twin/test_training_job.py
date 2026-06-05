"""Offline twin training job (#705 H4).

``train_and_persist_twin`` trains a real ``TwinGenerator`` for a (brand,
twin_type) from synthetic or supplied data, persists the artifact to MLflow, and
records a loadable ``digital_twin_models`` row — the piece that lets ``/simulate``
load a real model instead of failing closed forever.

Hermetic: MLflow points at a ``file://`` store; the repo is an AsyncMock so we
assert the real refs are persisted (no fabrication).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pytest

from src.digital_twin.models.twin_models import Brand, TwinType
from src.digital_twin.twin_generator import TwinGenerator


@pytest.fixture()
def file_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")


@pytest.mark.asyncio
async def test_train_and_persist_twin_synthetic_creates_loadable_model(file_tracking):
    from src.digital_twin import twin_persistence
    from src.digital_twin.training_job import train_and_persist_twin

    repo = AsyncMock()
    repo.save_model = AsyncMock(return_value=uuid4())

    result = await train_and_persist_twin(
        twin_type=TwinType.HCP,
        brand=Brand.KISQALI,
        repo=repo,
        synthetic=True,
        n_rows=1100,
        seed=2,
    )

    # Honest result: real refs + provenance + finite metric.
    assert result["model_id"]
    assert result["model_uri"].startswith(("models:/", "runs:/"))
    assert result["data_provenance"] == "synthetic"
    assert np.isfinite(result["r2_score"])

    # The persisted refs round-trip into a working generator (E2E).
    gen = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.KISQALI)
    assert twin_persistence.hydrate_generator(gen, result["model_uri"], result["run_id"]) is True
    assert gen.model is not None
    assert gen.generate(n=3, seed=1).size == 3

    # The repo row got the REAL mlflow refs (anti-mock: not None, not fabricated).
    repo.save_model.assert_awaited_once()
    kwargs = repo.save_model.await_args.kwargs
    assert kwargs["mlflow_model_uri"] == result["model_uri"]
    assert kwargs["mlflow_run_id"] == result["run_id"]


@pytest.mark.asyncio
async def test_train_and_persist_requires_a_data_source(file_tracking):
    from src.digital_twin.training_job import train_and_persist_twin

    repo = AsyncMock()
    with pytest.raises(ValueError):
        # No data, no data_source, synthetic not set → fail loud, train nothing.
        await train_and_persist_twin(twin_type=TwinType.HCP, brand=Brand.KISQALI, repo=repo)
    repo.save_model.assert_not_called()
