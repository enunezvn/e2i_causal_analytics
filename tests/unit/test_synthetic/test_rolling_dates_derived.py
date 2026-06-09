"""Shard 04 (codex HIGH closure) — date-emitting generators that DERIVE timestamps
from the anchored journey date must honor rolling-window anchoring too: no future
timestamps, bulk recent. Covers prediction_generator, trigger_generator (linked
paths) and feature_value_generator (which previously ignored anchoring entirely)."""
from datetime import date, timedelta

import pandas as pd

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.feature_value_generator import FeatureValueGenerator
from src.ml.synthetic.generators.hcp_generator import HCPGenerator
from src.ml.synthetic.generators.patient_generator import PatientGenerator
from src.ml.synthetic.generators.prediction_generator import PredictionGenerator
from src.ml.synthetic.generators.trigger_generator import TriggerGenerator

REF = date(2026, 6, 9)


def _anchored_patients(n: int = 300) -> pd.DataFrame:
    cfg = GeneratorConfig(
        seed=11, n_records=n, brand=Brand.KISQALI,
        anchor_to_now=True, anchor_reference=REF,
    )
    return PatientGenerator(cfg).generate()


def test_prediction_dates_fresh_and_not_future_under_anchor():
    patients = _anchored_patients()
    cfg = GeneratorConfig(seed=11, n_records=600, anchor_to_now=True, anchor_reference=REF)
    df = PredictionGenerator(cfg, patient_df=patients).generate()
    d = df["prediction_date"].map(date.fromisoformat)
    assert d.max() <= REF, "prediction_date lands in the future"
    assert (d >= REF - timedelta(days=30)).mean() >= 0.30, "predictions not bulk-recent"


def test_trigger_timestamps_fresh_and_not_future_under_anchor():
    patients = _anchored_patients()
    hcp = HCPGenerator(
        GeneratorConfig(seed=11, n_records=50, brand=Brand.KISQALI)
    ).generate()
    cfg = GeneratorConfig(seed=11, n_records=600, anchor_to_now=True, anchor_reference=REF)
    df = TriggerGenerator(cfg, patient_df=patients, hcp_df=hcp).generate()
    ts = pd.to_datetime(df["trigger_timestamp"]).dt.date
    assert ts.max() <= REF, "trigger fires in the future"
    assert (ts >= (REF - timedelta(days=30))).mean() >= 0.30, "triggers not bulk-recent"


def test_feature_value_timestamps_anchor_to_reference():
    cfg = GeneratorConfig(seed=11, n_records=400, anchor_to_now=True, anchor_reference=REF)
    ts = FeatureValueGenerator(cfg)._generate_timestamps(400)
    days = [t.date() for t in ts]
    assert max(days) <= REF
    # newest feature value is recent (proves it is no longer capped at 2024-12-31)
    assert max(days) >= REF - timedelta(days=7)


def test_shift_dates_to_window_clips_future_preserves_past():
    """Direct guard on the clip semantics (codex LOW): future dates collapse onto the
    reference, past/present dates are left untouched (an affine rescale would move
    them and flatten the recency mixture). Also covers the 'YYYY-MM-DD HH:MM:SS'
    time-suffix preservation path."""
    cfg = GeneratorConfig(seed=1, anchor_to_now=True, anchor_reference=REF)
    gen = PatientGenerator(cfg)
    dates = [
        "2026-05-01",            # past -> untouched
        "2026-06-09",            # == ref -> untouched
        "2026-08-15",            # future -> clipped to ref
        "2026-12-31 14:30:00",   # future w/ time -> clipped to ref, time tail kept
    ]
    out = gen._shift_dates_to_window(dates)
    assert out[0] == "2026-05-01"
    assert out[1] == "2026-06-09"
    assert out[2] == "2026-06-09"
    assert out[3] == "2026-06-09 14:30:00"
    # off -> identity
    off = PatientGenerator(GeneratorConfig(seed=1))
    assert off._shift_dates_to_window(dates) == dates


def test_derived_dates_default_off_not_anchored_to_now():
    patients = PatientGenerator(
        GeneratorConfig(seed=11, n_records=200, brand=Brand.KISQALI)
    ).generate()
    df = PredictionGenerator(
        GeneratorConfig(seed=11, n_records=400), patient_df=patients
    ).generate()
    # legacy: journey 2022-2024 + up to 90d offset -> at most spills into early 2025,
    # never anchored to the 2026 reference.
    assert max(int(s[:4]) for s in df["prediction_date"]) <= 2025
