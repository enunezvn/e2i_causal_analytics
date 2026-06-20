"""Coverage for geographic_region one-hot encoding in the causal loader."""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.api.routes import causal as causal_routes
from src.api.routes.causal import (
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_NUMERIC_COLUMNS,
    _one_hot_categoricals,
)

# _load_agent_estimation_frame does a FUNCTION-LOCAL import of
# get_async_supabase_client, so patch the SOURCE module.
_CLIENT_FACTORY = "src.memory.services.factories.get_async_supabase_client"


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    async def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, *_a, **_k):
        return _FakeQuery(self._rows)


@pytest.mark.asyncio
async def test_loader_expands_geographic_region_into_dummies():
    rows = [
        {
            "treatment_arm": 1,
            "persistent_180d": 1,
            "disease_severity": 2,
            "academic_hcp": 1,
            "geographic_region": "south",
        },
        {
            "treatment_arm": 0,
            "persistent_180d": 0,
            "disease_severity": 1,
            "academic_hcp": 0,
            "geographic_region": "west",
        },
        {
            "treatment_arm": 1,
            "persistent_180d": 1,
            "disease_severity": 3,
            "academic_hcp": 1,
            "geographic_region": "midwest",
        },
    ]
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(rows))):
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            covariates=["disease_severity", "academic_hcp", "geographic_region"],
            limit=1500,
        )
    assert "geographic_region" not in df.columns
    assert "geographic_region=south" in df.columns
    assert "geographic_region=west" in df.columns
    assert "geographic_region=midwest" not in df.columns  # reference level
    assert "geographic_region" not in select_cols
    assert "geographic_region=south" in select_cols and "geographic_region=west" in select_cols
    assert df["disease_severity"].dtype == float
    assert df["geographic_region=south"].dtype == float


@pytest.mark.asyncio
async def test_loader_rejects_unallowed_column_still_400():
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient([]))):
        with pytest.raises(causal_routes.HTTPException) as ei:
            await causal_routes._load_agent_estimation_frame(
                dataset="patient_journeys",
                treatment_var="treatment_arm",
                outcome_var="persistent_180d",
                covariates=["totally_made_up_col"],
                limit=10,
            )
    assert ei.value.status_code == 400


def test_geographic_region_registered_as_categorical_covariate():
    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    assert "geographic_region" in spec["covariate"]
    assert "geographic_region" in _CAUSAL_CATEGORICAL_COLUMNS["patient_journeys"]
    assert "geographic_region" not in _CAUSAL_NUMERIC_COLUMNS["patient_journeys"]


def test_one_hot_expands_into_stable_drop_first_float_dummies():
    df = pd.DataFrame(
        {
            "treatment_arm": [1.0, 0.0, 1.0, 0.0],
            "persistent_180d": [1.0, 0.0, 1.0, 1.0],
            "disease_severity": [2.0, 1.0, 3.0, 2.0],
            "geographic_region": ["south", "west", "midwest", "northeast"],
        }
    )
    out, dummy_names = _one_hot_categoricals(df, ["geographic_region"])
    assert dummy_names == [
        "geographic_region=northeast",
        "geographic_region=south",
        "geographic_region=west",
    ]
    assert "geographic_region" not in out.columns
    for name in dummy_names:
        assert name in out.columns
        assert out[name].dtype == float
        assert set(out[name].unique()) <= {0.0, 1.0}
    midwest_row = out.iloc[2]  # midwest = dropped reference level -> all-zero
    assert midwest_row["geographic_region=northeast"] == 0.0
    assert midwest_row["geographic_region=south"] == 0.0
    assert midwest_row["geographic_region=west"] == 0.0
    assert list(out["treatment_arm"]) == [1.0, 0.0, 1.0, 0.0]


def test_one_hot_noop_when_no_categoricals_present():
    df = pd.DataFrame({"treatment_arm": [1.0, 0.0], "disease_severity": [2.0, 1.0]})
    out, dummy_names = _one_hot_categoricals(df, ["geographic_region"])
    assert dummy_names == []
    assert list(out.columns) == ["treatment_arm", "disease_severity"]
