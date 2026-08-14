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

    from src.data.kg.cache import _VALID_STATUSES, CacheRecordStatus

    assert _VALID_STATUSES == frozenset(get_args(CacheRecordStatus))


def test_try_load_cache_returns_none_for_missing_file(tmp_path: Path):
    """Soft-fail variant: absent file → None instead of FileNotFoundError."""
    from src.data.kg.cache import try_load_cache

    assert try_load_cache(tmp_path / "does_not_exist.json") is None


def test_try_load_cache_loads_existing_file(tmp_path: Path):
    """try_load_cache returns the records when the file exists."""
    from src.data.kg.cache import CacheRecord, save_cache, try_load_cache

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
    path = tmp_path / "cache.json"
    save_cache([record], path)
    loaded = try_load_cache(path)
    assert loaded is not None
    assert len(loaded) == 1


def test_manifest_fingerprint_fields_explicit_allowlist():
    """Fingerprint allowlist is documented + matches what's hashed.

    A future field added to FeatureContract that affects KG queries
    must be added to _MANIFEST_FINGERPRINT_FIELDS or this test fails
    (loud regression vs silently skipping the field).
    """
    from src.data.kg.cache import _MANIFEST_FINGERPRINT_FIELDS

    expected = (
        "name",
        "knowable_at.reference",
        "knowable_at.offset_days",
        "source",
        "derivation_inputs",
        "aggregation",
        "window_days",
        "kg_entity_codes",
    )
    assert _MANIFEST_FINGERPRINT_FIELDS == expected


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


def test_source_endpoint_ids_round_trip(tmp_path: Path):
    """The pre-normalisation endpoint ids must survive the cache file (#1607).

    ``build_kg_cache._drug_disease_edges_for_cui`` rewrites an Open Targets
    edge's endpoints to the manifest/scope identifiers so the voter's
    ``_connects`` check can match them. The original ChEMBL/EFO ids were kept
    only in ``KGEdge.raw``, which ``_kg_edge_to_json`` does not serialise — so
    the committed artifact recorded a leakage finding with no way to audit which
    disease the fuzzy ``search_disease`` call had actually matched.
    """
    from src.data.kg.cache import CacheRecord, load_cache, save_cache
    from src.data.kg.types import KGEdge

    edge = KGEdge(
        subject_id="302379",  # rewritten to the scope's RxNorm code
        predicate="treats",
        object_id="C0041834",  # rewritten to the feature's UMLS CUI
        evidence_source="open_targets",
        subject_name="OMALIZUMAB",
        object_name="chronic urticaria",
        datasource="chembl_indications",
        source_subject_id="CHEMBL1201589",
        source_object_id="MONDO_0005492",
    )
    record = CacheRecord(
        feature_name="dx_total_csu",
        manifest_fingerprint_sha8="a",
        target_codes_fingerprint_sha8="b",
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(("UMLS", "C0041834"),),
        target_entity_codes=(("RXNORM", "302379"),),
        sources_attempted=("open_targets",),
        status="ok",
        edges=(edge,),
        errors=(),
    )

    path = tmp_path / "cache.json"
    save_cache([record], path)
    loaded = load_cache(path)[0].edges[0]

    assert loaded.source_subject_id == "CHEMBL1201589"
    assert loaded.source_object_id == "MONDO_0005492"
    # ...while the rewritten endpoints are what _connects still sees.
    assert loaded.subject_id == "302379"
    assert loaded.object_id == "C0041834"


def test_pre_1607_cache_files_load_without_source_ids(tmp_path: Path):
    """Older cache files have no source ids; they must load, not explode."""
    import json

    from src.data.kg.cache import load_cache

    payload = [
        {
            "feature_name": "f",
            "manifest_fingerprint_sha8": "a",
            "target_codes_fingerprint_sha8": "b",
            "queried_at": datetime.now(timezone.utc).isoformat(),
            "feature_entity_codes": [["UMLS", "C1"]],
            "target_entity_codes": [["RXNORM", "1"]],
            "sources_attempted": ["umls"],
            "status": "ok",
            "edges": [
                {
                    "subject_id": "C1",
                    "subject_name": "",
                    "predicate": "is_a",
                    "object_id": "C2",
                    "object_name": "",
                    "evidence_source": "umls",
                    "score": None,
                    "pmids": [],
                    "datasource": None,
                }
            ],
            "errors": [],
        }
    ]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))

    edge = load_cache(path)[0].edges[0]
    assert edge.source_subject_id is None
    assert edge.source_object_id is None
