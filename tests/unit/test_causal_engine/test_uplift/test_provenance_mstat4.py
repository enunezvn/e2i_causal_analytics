"""M-stat4 (MED, PROD) — UpliftResult must carry an honesty provenance marker.

The ate/att/atc on an UpliftResult are mean MODEL-PREDICTED uplift, NOT
identification-validated estimands (att==atc under randomized assignment).
These tests pin (a) the module-level provenance constant, (b) the default
data_provenance field on UpliftResult, (c) its presence in to_dict(), and
(d) that the corrected field docstring no longer makes a bare ATE/ATT/ATC
identification claim.
"""

from __future__ import annotations

import numpy as np

from src.causal_engine.uplift.base import (
    PROVENANCE_MODEL_PREDICTED_UPLIFT,
    UpliftModelType,
    UpliftResult,
)


def _make_result(**overrides) -> UpliftResult:
    base = {
        "model_type": UpliftModelType.UPLIFT_RANDOM_FOREST,
        "success": True,
        "uplift_scores": np.array([0.1, 0.2, 0.3]),
        "ate": 0.05,
        "att": 0.08,
        "atc": 0.02,
        "ate_std": 0.02,
    }
    base.update(overrides)
    return UpliftResult(**base)


def test_provenance_constant_value():
    assert (
        PROVENANCE_MODEL_PREDICTED_UPLIFT == "model_predicted_uplift_not_identification_validated"
    )


def test_result_default_data_provenance():
    result = _make_result()
    assert result.data_provenance == PROVENANCE_MODEL_PREDICTED_UPLIFT


def test_to_dict_emits_data_provenance():
    result_dict = _make_result().to_dict()
    assert "data_provenance" in result_dict
    assert result_dict["data_provenance"] == PROVENANCE_MODEL_PREDICTED_UPLIFT


def test_data_provenance_is_overridable():
    result = _make_result(data_provenance="custom_marker")
    assert result.data_provenance == "custom_marker"
    assert result.to_dict()["data_provenance"] == "custom_marker"


def test_field_docstring_disclaims_identification():
    doc = UpliftResult.__doc__ or ""
    # The corrected docstring must state the honesty disclaimer and must NOT
    # carry the bare identification claim for att/atc.
    assert "NOT an identification-validated" in doc
    assert "Average Treatment Effect on Treated" not in doc
    assert "Average Treatment Effect on Control" not in doc
