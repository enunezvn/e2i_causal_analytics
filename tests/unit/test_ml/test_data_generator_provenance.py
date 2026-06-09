"""Honesty gate (Shard 07 D1): the synthetic generator must NOT label its records
with real-vendor data_source strings, and every emitted record carries
is_synthetic=True.

The harm is dishonest provenance: synthetic records tagged 'IQVIA_APLD' /
'HealthVerity' (real vendor feeds) look real. Replace with an honest synthetic
label and stamp is_synthetic=True so the record's origin is unambiguous.
"""

import src.ml.data_generator as dg
from src.ml.data_generator import DATA_SOURCES, E2IDataGenerator

# Real third-party vendor feeds that synthetic data must never claim to be.
_REAL_VENDOR_LABELS = {"IQVIA_APLD", "IQVIA_LAAD", "HealthVerity", "Komodo", "Veeva"}


def test_data_sources_constant_has_no_real_vendor_labels() -> None:
    assert _REAL_VENDOR_LABELS.isdisjoint(set(DATA_SOURCES)), (
        f"DATA_SOURCES still claims real vendor feeds: {DATA_SOURCES}"
    )
    assert all(s.startswith("synthetic") for s in DATA_SOURCES), DATA_SOURCES


def test_reference_universe_uses_honest_synthetic_source() -> None:
    gen = E2IDataGenerator()
    gen._generate_reference_universe()
    sources = {u["data_source"] for u in gen.reference_universe}
    assert _REAL_VENDOR_LABELS.isdisjoint(sources), (
        f"reference_universe data_source still claims a real vendor: {sources}"
    )
    assert all(s.startswith("synthetic") for s in sources), sources
    assert all(u.get("is_synthetic") is True for u in gen.reference_universe)


def test_patient_journeys_data_source_is_honest_and_tagged() -> None:
    gen = E2IDataGenerator()
    gen._generate_patient_journeys()
    assert gen.patient_journeys, "expected patient journeys to be generated"
    for pj in gen.patient_journeys:
        assert pj["data_source"] not in _REAL_VENDOR_LABELS, pj["data_source"]
        assert pj.get("is_synthetic") is True


def test_module_has_no_real_vendor_data_source_default() -> None:
    # Guard: the module-level DATA_SOURCES is the single source of provenance
    # labels threaded through patient_journeys / data_source_tracking.
    assert not (_REAL_VENDOR_LABELS & set(dg.DATA_SOURCES))
