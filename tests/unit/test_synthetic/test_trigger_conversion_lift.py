"""Shard 05 Task 2 — TriggerGenerator must inject a KNOWN trigger->prescription
conversion lift (accepted-arm minus rejected-arm, +10-20pp) so the conversion KPI
is non-degenerate. The frame/registry COMPUTES the realized rate; the generator only
seeds the data."""
import pandas as pd

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.trigger_generator import (
    DESIGNED_CONVERSION_LIFT,
    TriggerGenerator,
)
from src.services.kpi_resolution import _compute_conversion_outcome


def _patient_frame(n=400):
    return pd.DataFrame(
        {
            "patient_id": [f"pt{i:05d}" for i in range(n)],
            "hcp_id": [f"hcp{i % 40:03d}" for i in range(n)],
            "journey_start_date": ["2026-05-20"] * n,
            "treatment_initiated": [i % 2 for i in range(n)],
            "engagement_score": [5.0] * n,
            "brand": ["Kisqali"] * n,
        }
    )


def test_accepted_minus_rejected_lift_matches_design():
    cfg = GeneratorConfig(seed=7, n_records=800)
    gen = TriggerGenerator(cfg, patient_df=_patient_frame(), hcp_df=None)
    triggers = gen.generate()
    rx = gen.injected_prescriptions  # treatment_events rows the generator appended

    converted = _compute_conversion_outcome(triggers, rx)
    triggers = triggers.assign(converted=converted.to_numpy())
    acc = triggers["acceptance_status"].eq("accepted")
    rej = triggers["acceptance_status"].eq("rejected")
    lift = triggers.loc[acc, "converted"].mean() - triggers.loc[rej, "converted"].mean()

    # MECHANISM check: the injection must produce a CLEAR, sign-stable accepted>rejected
    # lift. This is the injected-prescriptions-ONLY lift; with ~2 triggers/patient and
    # the patient-level EXISTS conversion, an accepted trigger's rx bleeds into the same
    # patient's other windows, so the injected-only lift over-states DESIGNED_CONVERSION_LIFT.
    # The faithful in-band (+10-20pp) check — where arm-NEUTRAL baseline prescriptions
    # dilute the lift back down — is the Task 5 DB gate, not this unit test.
    assert lift > 0.08, lift          # clearly positive, sign-stable
    assert lift < 0.40, lift          # bounded — not a degenerate all-accepted artifact
    assert DESIGNED_CONVERSION_LIFT == 0.15  # the seed constant is the documented target


def test_injected_prescriptions_land_in_window_and_self_stamp():
    cfg = GeneratorConfig(seed=7, n_records=400)
    gen = TriggerGenerator(cfg, patient_df=_patient_frame(200), hcp_df=None)
    gen.generate()
    rx = gen.injected_prescriptions
    assert len(rx) > 0
    assert {"patient_id", "event_date", "event_type", "brand"}.issubset(rx.columns)
    assert (rx["event_type"] == "prescription").all()
    # Appended AFTER the central is_synthetic stamp path -> must self-stamp, else they
    # leak into real-mode KPIs.
    assert "is_synthetic" in rx.columns
    assert bool(rx["is_synthetic"].all())
