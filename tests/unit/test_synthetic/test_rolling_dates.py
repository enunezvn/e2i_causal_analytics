from datetime import date, timedelta

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator


def test_anchor_to_now_pulls_max_date_within_30d_and_bulk_recent():
    ref = date(2026, 6, 9)
    cfg = GeneratorConfig(
        seed=7, n_records=500, anchor_to_now=True, anchor_reference=ref
    )
    df = PatientGenerator(cfg).generate()
    dates = df["journey_start_date"].map(date.fromisoformat)
    assert dates.max() <= ref
    assert dates.max() >= ref - timedelta(days=30), "max date is stale (>30d old)"
    recent = (dates >= (ref - timedelta(days=30))).mean()
    assert recent >= 0.50, f"only {recent:.0%} of rows inside NOW()-30d window"


def test_anchor_regenerates_per_run_on_later_reference():
    early = PatientGenerator(
        GeneratorConfig(seed=7, n_records=200, anchor_to_now=True,
                        anchor_reference=date(2026, 6, 9))
    ).generate()
    later = PatientGenerator(
        GeneratorConfig(seed=7, n_records=200, anchor_to_now=True,
                        anchor_reference=date(2026, 9, 1))
    ).generate()
    e_max = early["journey_start_date"].map(date.fromisoformat).max()
    l_max = later["journey_start_date"].map(date.fromisoformat).max()
    assert l_max > e_max, "later run did not shift the window forward"


def test_default_off_preserves_legacy_2022_2024_span():
    cfg = GeneratorConfig(seed=7, n_records=200)  # anchor_to_now defaults False
    df = PatientGenerator(cfg).generate()
    years = {int(d[:4]) for d in df["journey_start_date"]}
    assert years <= {2022, 2023, 2024}
