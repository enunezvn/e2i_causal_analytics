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
from src.data.kg.types import KGEdge

CacheRecordStatus = Literal["ok", "queried_no_edges", "entity_unresolved", "source_error"]

# Derived from the Literal definition so a fifth status added to the
# type alias automatically updates the runtime validator (codex L2).
_VALID_STATUSES: frozenset[str] = frozenset(get_args(CacheRecordStatus))


class CacheRecordValidationError(ValueError):
    pass


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
    """Read a cache file and reconstruct its records."""
    payload = json.loads(path.read_text())
    return [CacheRecord.from_json(entry) for entry in payload]


def compute_manifest_fingerprint(features: Iterable[FeatureContract]) -> str:
    """SHA-256 over a deterministic serialization of the manifest.

    The serialization captures every contract field that affects KG
    queries: name, knowable_at, source, derivation_inputs, aggregation,
    window_days, kg_entity_codes. Returns the first 8 hex chars (sha8).
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
