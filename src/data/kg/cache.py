"""Phase 2.9 Stage 2 KG cache — schema and IO helpers.

The cache file persists per-feature provenance records produced by
``scripts/build_kg_cache.py``. Run-time pipeline (Layer 5
``adaptive_validity_check``) reads this file at node entry to obtain
``KGEdge`` lists per feature without making HTTP calls in the hot path.

Provenance schema is explicit (status enum, sources_attempted, errors)
so that an empty edge list with ``status="queried_no_edges"`` is
distinguishable from a missing entry (``cache_missing`` audit event at
run time).

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, get_args

from src.data.feature_contract import FeatureContract
from src.data.kg.types import EvidenceItem, KGEdge

CacheRecordStatus = Literal["ok", "queried_no_edges", "entity_unresolved", "source_error"]

# Derived from the Literal definition so a fifth status added to the
# type alias automatically updates the runtime validator (codex L2).
_VALID_STATUSES: frozenset[str] = frozenset(get_args(CacheRecordStatus))


class CacheRecordValidationError(ValueError):
    pass


class KGCacheStaleError(RuntimeError):
    """Raised when the KG cache's per-record fingerprints don't match
    the current manifest + target_entity_codes fingerprints.

    Pipeline reader (``adaptive_validity_check._load_kg_cache``) raises
    this in ``kg_mode="promoted"`` so the operator must rebuild the
    cache before the build can proceed. ``kg_mode="shadow"`` warns and
    falls through to no-cache (audit-only mode tolerates a stale cache
    because verdicts are advisory in that mode).
    """


@dataclass(frozen=True)
class CacheRecord:
    """One per-feature provenance entry in the KG cache file."""

    feature_name: str
    manifest_fingerprint_sha8: str
    target_codes_fingerprint_sha8: str
    queried_at: datetime
    feature_entity_codes: tuple[tuple[str, str], ...]
    target_entity_codes: tuple[tuple[str, str], ...]
    sources_attempted: tuple[str, ...]
    status: CacheRecordStatus
    edges: tuple[KGEdge, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise CacheRecordValidationError(
                f"status must be one of {sorted(_VALID_STATUSES)}; got {self.status!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "manifest_fingerprint_sha8": self.manifest_fingerprint_sha8,
            "target_codes_fingerprint_sha8": self.target_codes_fingerprint_sha8,
            "queried_at": self.queried_at.isoformat(),
            "feature_entity_codes": [list(t) for t in self.feature_entity_codes],
            "target_entity_codes": [list(t) for t in self.target_entity_codes],
            "sources_attempted": list(self.sources_attempted),
            "status": self.status,
            "edges": [_kg_edge_to_json(e) for e in self.edges],
            "errors": list(self.errors),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CacheRecord:
        return cls(
            feature_name=payload["feature_name"],
            manifest_fingerprint_sha8=payload["manifest_fingerprint_sha8"],
            target_codes_fingerprint_sha8=payload["target_codes_fingerprint_sha8"],
            queried_at=datetime.fromisoformat(payload["queried_at"]),
            feature_entity_codes=tuple(tuple(t) for t in payload["feature_entity_codes"]),
            target_entity_codes=tuple(tuple(t) for t in payload["target_entity_codes"]),
            sources_attempted=tuple(payload["sources_attempted"]),
            status=payload["status"],
            edges=tuple(_kg_edge_from_json(e) for e in payload.get("edges", [])),
            errors=tuple(payload.get("errors", [])),
        )


def _kg_edge_to_json(edge: KGEdge) -> dict[str, Any]:
    return {
        "subject_id": edge.subject_id,
        "subject_name": edge.subject_name,
        "predicate": edge.predicate,
        "object_id": edge.object_id,
        "object_name": edge.object_name,
        "evidence_source": edge.evidence_source,
        "score": edge.score,
        "pmids": list(edge.pmids),
        "datasource": edge.datasource,
        # Issue #245: structured evidence items round-trip too. Older
        # cache files lack this key — see ``_kg_edge_from_json`` for the
        # missing-key default.
        "evidence": [_evidence_item_to_json(e) for e in edge.evidence],
        # Issue #1607: the pre-normalisation endpoint ids. build_kg_cache
        # rewrites Open Targets endpoints to manifest/scope identifiers so the
        # voter's ``_connects`` check can match them; these preserve what the
        # source actually said, so a fuzzy disease match stays auditable from
        # the committed artifact alone. Same missing-key default as ``evidence``.
        "source_subject_id": edge.source_subject_id,
        "source_object_id": edge.source_object_id,
    }


def _kg_edge_from_json(payload: dict[str, Any]) -> KGEdge:
    return KGEdge(
        subject_id=payload["subject_id"],
        subject_name=payload.get("subject_name", ""),
        predicate=payload["predicate"],
        object_id=payload["object_id"],
        object_name=payload.get("object_name", ""),
        evidence_source=payload["evidence_source"],
        score=payload.get("score"),
        pmids=tuple(payload.get("pmids", [])),
        datasource=payload.get("datasource"),
        evidence=tuple(_evidence_item_from_json(e) for e in payload.get("evidence", [])),
        # Absent in pre-#1607 cache files; None means "no normalisation".
        source_subject_id=payload.get("source_subject_id"),
        source_object_id=payload.get("source_object_id"),
    )


def _evidence_item_to_json(item: EvidenceItem) -> dict[str, Any]:
    return {
        "pmid": item.pmid,
        "source": item.source,
        "chembl_target_id": item.chembl_target_id,
        "datasource_score": item.datasource_score,
    }


def _evidence_item_from_json(payload: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        pmid=payload["pmid"],
        source=payload["source"],
        chembl_target_id=payload.get("chembl_target_id"),
        datasource_score=payload.get("datasource_score"),
    )


def save_cache(records: Iterable[CacheRecord], path: Path) -> None:
    """Atomically write a cache file with deterministic record order.

    Records are sorted by feature_name for byte-stable output across
    concurrent regenerations. Write goes to a temp file in the same
    directory, then ``os.replace`` atomically replaces the target. On
    error, the temp file is removed.

    After ``os.replace``, the parent directory is fsynced so the rename
    itself survives a crash on filesystems where the directory entry
    isn't durable until the dir is flushed (ext4 / XFS without
    barriers, codex L1).
    """
    sorted_records = sorted(records, key=lambda r: r.feature_name)
    payload = [r.to_json() for r in sorted_records]
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(parent)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    """Fsync a directory so its rename entries are durable on disk.

    Skipped on platforms (Windows) where directories don't expose a
    descriptor for fsync.
    """
    try:
        dir_fd = os.open(str(directory), os.O_DIRECTORY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def load_cache(path: Path) -> list[CacheRecord]:
    """Read a cache file and reconstruct its records.

    Raises ``FileNotFoundError`` if the cache is missing — callers that
    treat absence as a normal pipeline state (e.g. KG mode = ``shadow``)
    should use :func:`try_load_cache` instead.
    """
    payload = json.loads(path.read_text())
    return [CacheRecord.from_json(entry) for entry in payload]


def try_load_cache(path: Path) -> list[CacheRecord] | None:
    """Soft-fail variant of :func:`load_cache`.

    Returns ``None`` when the file does not exist. Other I/O / parse
    errors still propagate, so a malformed cache is treated as a real
    error rather than absence (codex L3).
    """
    if not path.exists():
        return None
    return load_cache(path)


# Fields enumerated below are the ENTIRE input to the manifest
# fingerprint. A new field on FeatureContract that affects KG queries
# (e.g. a future kg_exclude_predicates filter) MUST be added here too —
# otherwise downstream cache readers will accept a stale cache that no
# longer matches the manifest's KG-query contract. Excluded by design:
# label, description, dtype (presentation only), validation_* (catches
# data quality, not KG semantics).
_MANIFEST_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "name",
    "knowable_at.reference",
    "knowable_at.offset_days",
    "source",
    "derivation_inputs",
    "aggregation",
    "window_days",
    "kg_entity_codes",
)


def compute_manifest_fingerprint(features: Iterable[FeatureContract]) -> str:
    """SHA-256 over a deterministic serialization of the manifest.

    The serialization captures every contract field that affects KG
    queries (see ``_MANIFEST_FINGERPRINT_FIELDS``). Returns the first
    8 hex chars (sha8).
    """
    rows: list[tuple[Any, ...]] = []
    for fc in features:
        rows.append(
            (
                fc.name,
                fc.knowable_at.reference,
                fc.knowable_at.offset_days,
                fc.source,
                tuple(fc.derivation_inputs),
                fc.aggregation,
                fc.window_days,
                tuple(tuple(t) for t in fc.kg_entity_codes),
            )
        )
    rows.sort(key=lambda r: r[0])
    blob = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def compute_target_codes_fingerprint(target_codes: Iterable[tuple[str, str]]) -> str:
    """SHA-256 over a sorted (system, code) tuple list. First 8 hex chars."""
    sorted_codes = sorted(tuple(t) for t in target_codes)
    blob = json.dumps(sorted_codes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def compose_cache_filename(manifest_fp: str, target_fp: str) -> str:
    """Deterministic cache filename — no cohort name in the path.

    Two cohorts with identical (manifest, target) legitimately share a
    cache file. The pipeline reads via ``scope_spec["kg_cache_path"]``
    only.
    """
    return f"{manifest_fp}__{target_fp}.json"
