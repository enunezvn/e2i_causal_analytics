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
    # Structured provenance is persisted (synthetic != RWD).
    assert kwargs["data_provenance"] == "synthetic"


@pytest.mark.asyncio
async def test_train_and_persist_requires_a_data_source(file_tracking):
    from src.digital_twin.training_job import train_and_persist_twin

    repo = AsyncMock()
    with pytest.raises(ValueError):
        # No data, no data_source, synthetic not set → fail loud, train nothing.
        await train_and_persist_twin(twin_type=TwinType.HCP, brand=Brand.KISQALI, repo=repo)
    repo.save_model.assert_not_called()


@pytest.mark.asyncio
async def test_train_and_persist_records_the_training_frame_identity(file_tracking):
    """codex r3 #2: the fit fingerprint hashes training_config; the frame that produced
    the fit (source + seed + rows) is recorded there so two fits from different
    frames cannot share a fingerprint on coinciding metrics."""
    from src.digital_twin.training_job import train_and_persist_twin

    repo = AsyncMock()
    repo.save_model = AsyncMock(return_value=uuid4())

    await train_and_persist_twin(
        twin_type=TwinType.HCP, brand=Brand.KISQALI, repo=repo, synthetic=True, n_rows=1100, seed=2
    )

    kwargs = repo.save_model.await_args.kwargs
    assert kwargs["training_frame"] == {
        "source": "synthetic_training_frame",
        "seed": 2,
        "n_rows": 1100,
        "target_column": "outcome",
    }


@pytest.mark.asyncio
async def test_train_and_persist_reaches_the_real_repository_facade(file_tracking):
    """codex r4 #1: the Celery path constructs the TwinRepository facade, whose
    save_model did not accept training_frame — real training raised TypeError after
    fitting. The facade is exercised for real here; only the model store is faked."""
    from unittest.mock import create_autospec

    from src.digital_twin.training_job import train_and_persist_twin
    from src.digital_twin.twin_repository import TwinModelRepository, TwinRepository

    repo = TwinRepository(supabase_client=None)
    repo.models = create_autospec(TwinModelRepository, instance=True)
    saved_id = uuid4()
    repo.models.save_model = AsyncMock(return_value=saved_id)

    result = await train_and_persist_twin(
        twin_type=TwinType.HCP, brand=Brand.KISQALI, repo=repo, synthetic=True, n_rows=1100, seed=2
    )

    assert result["model_id"] == str(saved_id)
    kwargs = repo.models.save_model.await_args.kwargs
    assert kwargs["training_frame"]["seed"] == 2
    assert kwargs["data_provenance"] == "synthetic"
