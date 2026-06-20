"""label_segment_columns + resolve_indication: the brand-aware segmentation and
indication-resolution helpers (codex HIGH#2/#3)."""

import pytest

from src.services.clinical_context.label_criteria_provider import (
    label_segment_columns,
    resolve_indication,
)


@pytest.mark.unit
def test_label_segment_columns_are_categorical_inclusion_fields():
    assert label_segment_columns("Remibrutinib") == ["prior_antihistamine_therapy"]
    kis = label_segment_columns("Kisqali")
    assert set(kis) == {"hr_status", "her2_status", "disease_stage"}
    # diagnosis_code (constant) + continuous-threshold fields (age, ecog, ldh) excluded.
    assert "diagnosis_code" not in kis and "age_at_diagnosis" not in kis
    assert label_segment_columns("Fabhalta", "pnh") == ["complement_inhibitor_status"]


@pytest.mark.unit
def test_resolve_indication_from_diagnosis_distribution():
    # D59.5 is PNH-specific -> unambiguous.
    assert resolve_indication("Fabhalta", ["D59.5", "D59.5", "D59.50"]) == "pnh"
    # L50.1 -> CSU.
    assert resolve_indication("Remibrutinib", ["L50.1"]) == "csu"
    # No codes -> None (caller falls back to brand default).
    assert resolve_indication("Fabhalta", []) is None
    # Codes matching no config -> None.
    assert resolve_indication("Fabhalta", ["Z99.9"]) is None
