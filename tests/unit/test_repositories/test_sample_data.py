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

import random

import numpy as np
import pandas as pd

from src.repositories import sample_data as sample_data_module
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


# ---------------------------------------------------------------------------
# RNG stream isolation (#1542, mechanism behind #1524)
#
# The generator's fields must not be coupled through shared RNG streams:
# resizing a categorical constant (REGIONS 5->4 in #1521) once shifted every
# subsequent seeded date draw and emptied a 30-day temporal-split window.
# These tests pin the decoupled contract: date draws are invariant under
# categorical-list resizes, and constructing/using a generator never mutates
# the process-global `random` / `np.random` state.
# ---------------------------------------------------------------------------

_FIXED_RANGE = {"start_date": "2026-01-01T00:00:00", "end_date": "2026-12-31T00:00:00"}


def test_instantiation_leaves_global_rngs_alone():
    random.seed(123)
    np.random.seed(456)
    expected = (random.random(), np.random.random())

    random.seed(123)
    np.random.seed(456)
    SampleDataGenerator(seed=42)
    assert (random.random(), np.random.random()) == expected


def test_generation_leaves_global_rngs_alone():
    random.seed(123)
    np.random.seed(456)
    expected = (random.random(), np.random.random())

    random.seed(123)
    np.random.seed(456)
    gen = SampleDataGenerator(seed=42)
    gen.business_metrics(n_samples=50, **_FIXED_RANGE)
    gen.ml_patients(n_patients=50, **_FIXED_RANGE)
    assert (random.random(), np.random.random()) == expected


def test_categorical_resize_does_not_shift_business_metrics_dates(monkeypatch):
    baseline = SampleDataGenerator(seed=42).business_metrics(n_samples=10, **_FIXED_RANGE)

    monkeypatch.setattr(sample_data_module, "REGIONS", [*REGION_ENUM_LABELS, "extra_label"])
    resized = SampleDataGenerator(seed=42).business_metrics(n_samples=10, **_FIXED_RANGE)

    assert list(baseline["created_at"]) == list(resized["created_at"])
    assert list(baseline["metric_date"]) == list(resized["metric_date"])


def test_categorical_resize_does_not_shift_ml_patients_dates(monkeypatch):
    baseline = SampleDataGenerator(seed=42).ml_patients(n_patients=100, **_FIXED_RANGE)

    monkeypatch.setattr(sample_data_module, "BRANDS", [*sample_data_module.BRANDS, "ExtraBrand"])
    resized = SampleDataGenerator(seed=42).ml_patients(n_patients=100, **_FIXED_RANGE)

    # journey_start_date is the temporal-split axis; journey_end_date and
    # journey_status legitimately vary (they depend on categorical draws).
    assert list(baseline["journey_start_date"]) == list(resized["journey_start_date"])


def test_same_seed_reproduces_identical_frames():
    a = SampleDataGenerator(seed=7).business_metrics(n_samples=25, **_FIXED_RANGE)
    b = SampleDataGenerator(seed=7).business_metrics(n_samples=25, **_FIXED_RANGE)
    # metric_id is uuid4 (intentionally unseeded); everything else must match.
    assert a.drop(columns=["metric_id"]).equals(b.drop(columns=["metric_id"]))


def test_distinct_seeds_produce_different_data():
    a = SampleDataGenerator(seed=1).business_metrics(n_samples=25, **_FIXED_RANGE)
    b = SampleDataGenerator(seed=2).business_metrics(n_samples=25, **_FIXED_RANGE)
    assert not a["created_at"].equals(b["created_at"])
