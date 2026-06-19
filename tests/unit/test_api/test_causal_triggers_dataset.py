# tests/unit/test_api/test_causal_triggers_dataset.py
"""Unit coverage for the nba_triggers dataset spec registration and coercion.

Locks the SSOT maps (brand column, numeric columns, derivation fns, fill-zero
outcomes, physical-table mapping) that teach the loaders how to read the
triggers grain without DB access.
"""
import pytest

from src.api.routes.causal import (
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_FILL_ZERO_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_NUMERIC_DERIVATIONS,
    _CAUSAL_PHYSICAL_TABLE,
)


@pytest.mark.unit
def test_nba_triggers_spec_registered_with_rct_and_modifier_questions():
    spec = _CAUSAL_DATASET_SPECS["nba_triggers"]
    # RCT: control_group_flag -> action_taken; modifier: acceptance_status -> conversion_flag.
    assert "control_group_flag" in spec["treatment"]
    assert "acceptance_status" in spec["treatment"]
    assert "action_taken" in spec["outcome"]
    assert "conversion_flag" in spec["outcome"]


@pytest.mark.unit
def test_nba_triggers_numeric_and_derivation_and_fill_registered():
    numeric = _CAUSAL_NUMERIC_COLUMNS["nba_triggers"]
    # All four question columns coerce to numeric 0/1.
    assert {"control_group_flag", "action_taken", "conversion_flag", "acceptance_status"} <= numeric
    deriv = _CAUSAL_NUMERIC_DERIVATIONS["nba_triggers"]
    # acceptance_status derives to the "is accepted" indicator; action_taken to presence.
    assert deriv["acceptance_status"]("accepted") == 1.0
    assert deriv["acceptance_status"]("rejected") == 0.0
    assert deriv["acceptance_status"](None) == 0.0
    assert deriv["action_taken"]("called_patient") == 1.0
    assert deriv["action_taken"](None) == 0.0
    # Designed-NULL outcomes fill to 0 instead of dropping the row.
    assert {"action_taken", "conversion_flag"} <= _CAUSAL_FILL_ZERO_OUTCOMES["nba_triggers"]


@pytest.mark.unit
def test_nba_triggers_brand_column_is_brand_id():
    # triggers has NO `brand` column — the filter resolves against brand_id.
    assert _CAUSAL_BRAND_COLUMN.get("nba_triggers") == "brand_id"
    # patient_journeys keeps the default `brand` column.
    assert _CAUSAL_BRAND_COLUMN.get("patient_journeys", "brand") == "brand"


@pytest.mark.unit
def test_nba_triggers_physical_table_is_triggers():
    from src.api.routes.causal import _CAUSAL_PHYSICAL_TABLE

    assert _CAUSAL_PHYSICAL_TABLE["nba_triggers"] == "triggers"
