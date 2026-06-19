"""Coverage for geographic_region one-hot encoding in the causal loader."""
import pandas as pd

from src.api.routes.causal import (
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_NUMERIC_COLUMNS,
    _one_hot_categoricals,
)


def test_geographic_region_registered_as_categorical_covariate():
    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    assert "geographic_region" in spec["covariate"]
    assert "geographic_region" in _CAUSAL_CATEGORICAL_COLUMNS["patient_journeys"]
    assert "geographic_region" not in _CAUSAL_NUMERIC_COLUMNS["patient_journeys"]


def test_one_hot_expands_into_stable_drop_first_float_dummies():
    df = pd.DataFrame({
        "treatment_arm": [1.0, 0.0, 1.0, 0.0],
        "persistent_180d": [1.0, 0.0, 1.0, 1.0],
        "disease_severity": [2.0, 1.0, 3.0, 2.0],
        "geographic_region": ["south", "west", "midwest", "northeast"],
    })
    out, dummy_names = _one_hot_categoricals(df, ["geographic_region"])
    assert dummy_names == ["geographic_region=northeast", "geographic_region=south", "geographic_region=west"]
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
