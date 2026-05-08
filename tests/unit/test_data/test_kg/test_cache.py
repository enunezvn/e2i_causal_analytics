"""Tests for the Phase 2.9 Stage 2 KG cache schema and IO helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


def test_cache_record_round_trips_through_json(tmp_path: Path):
    """Per-feature provenance record serializes losslessly."""
    from src.data.kg.cache import CacheRecord

    record = CacheRecord(
        feature_name="has_atopic_dermatitis",
        manifest_fingerprint_sha8="a3f9c2b1",
        target_codes_fingerprint_sha8="5e2d8f04",
        queried_at=datetime(2026, 5, 8, 12, 30, 0, tzinfo=timezone.utc),
        feature_entity_codes=(("ICD10CM", "L20.9"), ("UMLS", "C0011615")),
        target_entity_codes=(("RXNORM", "479158"),),
        sources_attempted=("umls_uts", "open_targets"),
        status="ok",
        edges=(),
        errors=(),
    )

    payload = record.to_json()
    record2 = CacheRecord.from_json(payload)
    assert record2.feature_name == "has_atopic_dermatitis"
    assert record2.status == "ok"
    assert record2.feature_entity_codes == record.feature_entity_codes


def test_cache_record_status_is_validated():
    """status must be one of the four documented values."""
    from src.data.kg.cache import CacheRecord, CacheRecordValidationError

    with pytest.raises(CacheRecordValidationError):
        CacheRecord(
            feature_name="x",
            manifest_fingerprint_sha8="a",
            target_codes_fingerprint_sha8="b",
            queried_at=datetime.now(timezone.utc),
            feature_entity_codes=(),
            target_entity_codes=(),
            sources_attempted=(),
            status="not_a_real_status",  # type: ignore[arg-type]
            edges=(),
            errors=(),
        )


def test_cache_file_round_trips(tmp_path: Path):
    """A list of records writes deterministically and loads back."""
    from src.data.kg.cache import CacheRecord, load_cache, save_cache

    records = [
        CacheRecord(
            feature_name=f"feat_{i}",
            manifest_fingerprint_sha8="a3f9c2b1",
            target_codes_fingerprint_sha8="5e2d8f04",
            queried_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
            feature_entity_codes=(),
            target_entity_codes=(),
            sources_attempted=(),
            status="queried_no_edges",
            edges=(),
            errors=(),
        )
        for i in range(3)
    ]

    path = tmp_path / "cache.json"
    save_cache(records, path)
    loaded = load_cache(path)

    assert len(loaded) == 3
    assert {r.feature_name for r in loaded} == {"feat_0", "feat_1", "feat_2"}


def test_cache_file_atomic_write(tmp_path: Path):
    """save_cache should write atomically via temp + rename."""
    from src.data.kg.cache import CacheRecord, save_cache

    record = CacheRecord(
        feature_name="x",
        manifest_fingerprint_sha8="a",
        target_codes_fingerprint_sha8="b",
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(),
        target_entity_codes=(),
        sources_attempted=(),
        status="ok",
        edges=(),
        errors=(),
    )

    path = tmp_path / "out.json"
    save_cache([record], path)
    # No leftover .tmp file
    assert path.exists()
    leftovers = list(tmp_path.glob("out.json*.tmp"))
    assert leftovers == []


def test_cache_file_deterministic_sort_for_concurrency(tmp_path: Path):
    """Two concurrent regenerations produce identical bytes."""
    from src.data.kg.cache import CacheRecord, save_cache

    records = [
        CacheRecord(
            feature_name="b",
            manifest_fingerprint_sha8="a",
            target_codes_fingerprint_sha8="b",
            queried_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
            feature_entity_codes=(),
            target_entity_codes=(),
            sources_attempted=(),
            status="ok",
            edges=(),
            errors=(),
        ),
        CacheRecord(
            feature_name="a",
            manifest_fingerprint_sha8="a",
            target_codes_fingerprint_sha8="b",
            queried_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
            feature_entity_codes=(),
            target_entity_codes=(),
            sources_attempted=(),
            status="ok",
            edges=(),
            errors=(),
        ),
    ]

    path1 = tmp_path / "c1.json"
    path2 = tmp_path / "c2.json"
    save_cache(records, path1)
    save_cache(list(reversed(records)), path2)

    # Same content despite input order difference (records sorted by feature_name)
    assert path1.read_bytes() == path2.read_bytes()


def test_compute_manifest_fingerprint_stable():
    """Same manifest module → same fingerprint."""
    from src.data.kg.cache import compute_manifest_fingerprint
    from src.data.manifests import csu_feature_manifest

    fp1 = compute_manifest_fingerprint(csu_feature_manifest.CSU_FEATURES)
    fp2 = compute_manifest_fingerprint(csu_feature_manifest.CSU_FEATURES)
    assert fp1 == fp2
    assert len(fp1) == 8  # sha8 truncation


def test_compute_target_codes_fingerprint_order_independent():
    """Same set of (system, code) tuples in different order → same fp."""
    from src.data.kg.cache import compute_target_codes_fingerprint

    a = [("RXNORM", "479158"), ("RXNORM", "1011295")]
    b = [("RXNORM", "1011295"), ("RXNORM", "479158")]
    assert compute_target_codes_fingerprint(a) == compute_target_codes_fingerprint(b)


def test_compose_cache_filename_no_cohort_in_path():
    """Filename has only the two fingerprints — disease-agnostic invariant."""
    from src.data.kg.cache import compose_cache_filename

    fname = compose_cache_filename("a3f9c2b1", "5e2d8f04")
    assert fname == "a3f9c2b1__5e2d8f04.json"
    assert "csu" not in fname
    assert "optum" not in fname


def test_valid_statuses_derived_from_literal_no_drift():
    """Runtime _VALID_STATUSES stays in sync with the Literal type alias.

    A future status added to CacheRecordStatus auto-updates the validator
    (no parallel hardcoded set to drift).
    """
    from typing import get_args

    from src.data.kg.cache import CacheRecordStatus, _VALID_STATUSES

    assert _VALID_STATUSES == frozenset(get_args(CacheRecordStatus))


def test_kg_edge_serialization_round_trip(tmp_path: Path):
    """KGEdge survives JSON round-trip preserving predicate + score."""
    from src.data.kg.cache import CacheRecord, load_cache, save_cache
    from src.data.kg.types import KGEdge

    edge = KGEdge(
        subject_id="CHEMBL2107858",
        predicate="treats",
        object_id="MONDO_0011918",
        evidence_source="open_targets",
        subject_name="omalizumab",
        object_name="chronic urticaria",
        score=0.87,
        pmids=("28846349", "30000000"),
        datasource="europepmc",
    )
    record = CacheRecord(
        feature_name="primary_diagnosis_code",
        manifest_fingerprint_sha8="a",
        target_codes_fingerprint_sha8="b",
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(("ICD10CM", "L50.9"),),
        target_entity_codes=(("RXNORM", "479158"),),
        sources_attempted=("open_targets",),
        status="ok",
        edges=(edge,),
        errors=(),
    )

    path = tmp_path / "cache.json"
    save_cache([record], path)
    loaded = load_cache(path)
    assert len(loaded) == 1
    e = loaded[0].edges[0]
    assert e.subject_id == "CHEMBL2107858"
    assert e.predicate == "treats"
    assert e.score == 0.87
    assert e.pmids == ("28846349", "30000000")
