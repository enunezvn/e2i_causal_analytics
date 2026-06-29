"""Phase 0: the loader's patient_journeys registered columns include the new ones."""
import pytest

from src.ml.synthetic.loaders import batch_loader


@pytest.mark.unit
def test_patient_journeys_registers_commercial_arms_columns():
    registered = set(batch_loader.TABLE_COLUMNS["patient_journeys"])
    for col in (
        "adherent_180d", "low_gap_180d", "adherence_rate", "gap_days",
        "copay_support", "psp_enrolled", "rep_detailing_high", "sample_dropped",
        "copay_support_propensity", "psp_enrolled_propensity",
        "rep_detailing_high_propensity", "sample_dropped_propensity",
        "insurance_access_score",
    ):
        assert col in registered, f"{col} not registered -> loader will drop it"
