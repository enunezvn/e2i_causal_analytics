"""H11 residual regression: the three model-DETAIL read GETs that R4b left
unprotected — GET /models/{id}, /models/{id}/fidelity, /models/{id}/fidelity/report —
must require a viewer-tier token and fail-closed brand scoping. A non-admin whose
grant does not include the model's brand must get 404 (no existence leak), mirroring
the get_simulation/{id} ownership contract. Admin / in-grant viewer is unaffected.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

VIEWER_KISQALI = {"app_metadata": {"role": "viewer", "brands": ["Kisqali"]}}
VIEWER_REMI = {"app_metadata": {"role": "viewer", "brands": ["Remibrutinib"]}}
ADMIN = {"app_metadata": {"role": "admin"}}


def _model(brand: str) -> dict:
    """A digital_twin_models row dict (nested JSONB cols) as repo.get_model returns."""
    return {
        "model_id": str(uuid4()),
        "model_name": "hcp_remibrutinib",
        "model_description": None,
        "twin_type": "hcp",
        "brand": brand,
        "feature_columns": ["age_at_index"],
        "target_columns": ["outcome"],
        "performance_metrics": {"r2_score": 0.81, "rmse": 0.1},
        "training_config": {"algorithm": "random_forest", "training_samples": 2000},
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }


def _repo_with_model(brand: str) -> SimpleNamespace:
    return SimpleNamespace(
        get_model=AsyncMock(return_value=_model(brand)),
        get_model_fidelity_records=AsyncMock(return_value=[]),
    )


# ---- GET /models/{id} ------------------------------------------------------
@pytest.mark.unit
def test_get_model_denies_out_of_grant_brand():
    from src.api.routes import digital_twin as dt

    with patch.object(
        dt, "_get_twin_repo", AsyncMock(return_value=_repo_with_model("Remibrutinib"))
    ):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(dt.get_model(model_id=str(uuid4()), user=VIEWER_KISQALI))
    assert ei.value.status_code == 404


@pytest.mark.unit
def test_get_model_admin_allowed():
    from src.api.routes import digital_twin as dt

    with patch.object(
        dt, "_get_twin_repo", AsyncMock(return_value=_repo_with_model("Remibrutinib"))
    ):
        resp = asyncio.run(dt.get_model(model_id=str(uuid4()), user=ADMIN))
    assert resp.brand == "Remibrutinib"


@pytest.mark.unit
def test_get_model_viewer_in_grant_allowed():
    from src.api.routes import digital_twin as dt

    with patch.object(
        dt, "_get_twin_repo", AsyncMock(return_value=_repo_with_model("Remibrutinib"))
    ):
        resp = asyncio.run(dt.get_model(model_id=str(uuid4()), user=VIEWER_REMI))
    assert resp.brand == "Remibrutinib"


# ---- GET /models/{id}/fidelity --------------------------------------------
@pytest.mark.unit
def test_get_model_fidelity_denies_out_of_grant_brand():
    from src.api.routes import digital_twin as dt

    with patch.object(
        dt, "_get_twin_repo", AsyncMock(return_value=_repo_with_model("Remibrutinib"))
    ):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(dt.get_model_fidelity(model_id=str(uuid4()), user=VIEWER_KISQALI))
    assert ei.value.status_code == 404


# ---- GET /models/{id}/fidelity/report -------------------------------------
@pytest.mark.unit
def test_get_fidelity_report_denies_out_of_grant_brand():
    from src.api.routes import digital_twin as dt

    with patch.object(
        dt, "_get_twin_repo", AsyncMock(return_value=_repo_with_model("Remibrutinib"))
    ):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(dt.get_fidelity_report(model_id=str(uuid4()), user=VIEWER_KISQALI))
    assert ei.value.status_code == 404


# ---- fail-closed null-brand + positive in-grant paths ----------------------
@pytest.mark.unit
def test_get_model_denies_null_brand_for_non_admin():
    """Fail-closed: a model row whose brand is NULL is unreadable by a non-admin."""
    from src.api.routes import digital_twin as dt

    m = _model("Remibrutinib")
    m["brand"] = None
    repo = SimpleNamespace(get_model=AsyncMock(return_value=m))
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(dt.get_model(model_id=str(uuid4()), user=VIEWER_KISQALI))
    assert ei.value.status_code == 404


@pytest.mark.unit
def test_get_model_fidelity_viewer_in_grant_allowed():
    from src.api.routes import digital_twin as dt

    with patch.object(
        dt, "_get_twin_repo", AsyncMock(return_value=_repo_with_model("Remibrutinib"))
    ):
        resp = asyncio.run(dt.get_model_fidelity(model_id=str(uuid4()), user=VIEWER_REMI))
    assert resp.total_validations == 0


@pytest.mark.unit
def test_get_fidelity_report_viewer_in_grant_allowed():
    from datetime import datetime, timezone

    from src.api.routes import digital_twin as dt

    tracker = MagicMock()
    tracker.get_model_fidelity_report = MagicMock(
        return_value={
            "validation_count": 3,
            "fidelity_score": 0.82,
            "metrics": {"ci_coverage_rate": 0.9},
            "degradation_alert": False,
            "grade_distribution": {},
            "computed_at": datetime.now(timezone.utc),
        }
    )
    with (
        patch.object(
            dt, "_get_twin_repo", AsyncMock(return_value=_repo_with_model("Remibrutinib"))
        ),
        patch("src.digital_twin.fidelity_tracker.FidelityTracker", return_value=tracker),
    ):
        resp = asyncio.run(dt.get_fidelity_report(model_id=str(uuid4()), user=VIEWER_REMI))
    assert resp.total_validations == 3
