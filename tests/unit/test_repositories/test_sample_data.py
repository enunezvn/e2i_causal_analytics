"""Region-label contract for ``SampleDataGenerator`` (#1517, residual of #1509).

``sample_data.py``'s stated purpose is "Match production table schemas", but its
module-level ``REGIONS`` list drifted to ``["US", "EU", "APAC", "LATAM", "JP"]``
— values the production ``region_type`` enum (northeast/south/midwest/west,
owned by :mod:`src.services.enum_labels`) can never accept, and which
``src.mlops.pandera_schemas.BusinessMetricsSchema`` (``region isin E2I_REGIONS``)
rejects. No consumer inserts these frames into the DB today, but
``data_preparer``'s sample path DOES feed ``business_metrics()`` /
``triggers()`` frames into ML prep, so the first schema-validating or
DB-writing consumer would inherit the drift as a 22P02 or a Pandera failure.

These tests pin every region-emitting generator method to the shared enum
labels so the drift cannot re-open.
"""

import pandas as pd

from src.repositories.sample_data import REGIONS, SampleDataGenerator
from src.services.enum_labels import REGION_ENUM_LABELS


def test_module_regions_are_the_shared_enum_labels():
    # One owner (#1505/#1517): the generator's region list IS the enum label
    # set, not a hand-copied (and driftable) literal.
    assert REGIONS == list(REGION_ENUM_LABELS)


def _assert_regions_are_enum_labels(df: pd.DataFrame, column: str) -> None:
    values = set(df[column].dropna().unique())
    assert values, "generator produced no region values"
    assert values <= set(REGION_ENUM_LABELS), values


def test_business_metrics_regions_are_enum_labels():
    df = SampleDataGenerator(seed=42).business_metrics(n_samples=200)
    _assert_regions_are_enum_labels(df, "region")


def test_triggers_regions_are_enum_labels():
    df = SampleDataGenerator(seed=42).triggers(n_samples=200)
    _assert_regions_are_enum_labels(df, "region")


def test_patient_journeys_regions_are_enum_labels():
    df = SampleDataGenerator(seed=42).patient_journeys(n_patients=50)
    _assert_regions_are_enum_labels(df, "region")


def test_ml_patients_regions_are_enum_labels():
    df = SampleDataGenerator(seed=42).ml_patients(n_patients=100)
    _assert_regions_are_enum_labels(df, "geographic_region")
