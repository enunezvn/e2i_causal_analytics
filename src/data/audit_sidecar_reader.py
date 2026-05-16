"""Reader for the adaptive-validity sidecar JSON audit trail.

Plan: .claude/plans/layer4_evaluator_audit_consumer.md.

The producer is ``src/agents/ml_foundation/data_preparer/graph.py:
write_adaptive_verdicts_sidecar``. This reader is the inverse:
it iterates over every ``adaptive_verdicts_*.json`` file in the
configured artifacts directory and yields typed ``VerdictRecord``
rows for downstream consumers.

The reader is intentionally schema-tolerant: sidecars predating the
2026-05-15 evaluator-keys addition (Plan ``layer4_evaluator_audit_signal.md``)
must load without error, with the 5 evaluator fields surfaced as ``None``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Issue #235: schema_version contract between producer (graph.py:
# write_adaptive_verdicts_sidecar) and reader. Bump major on breaking
# changes, minor on additive forward-compatible changes.
_READER_SCHEMA_VERSION = "1.0"
_READER_SCHEMA_MAJOR = 1

# Issue #235 A3: the set of verdict-dict keys the reader knows how to
# surface on ``VerdictRecord``. Any unrecognized key is logged ONCE per
# file (not per-record), with the unknown set sorted for determinism.
# Extend this set in lockstep with new ``VerdictRecord`` fields.
_KNOWN_VERDICT_KEYS: frozenset[str] = frozenset(
    {
        "feature",
        "layer",
        "severity",
        "remediation",
        "evidence",
        "z_score",
        "p_value",
        "delta_auc",
        # Producer-emitted fields surfaced on raw_verdict only:
        "decided_by",
        "disagreements",
        "kg_signal",
        "actual_auc",
        "null_mean",
        "null_std",
        "n_permutations",
        "delta_auc_floor",
        "delta_auc_below_floor",
        "severity_pre_joint_check",
        "ablation_z_score",
        "ablation_delta_auc",
        "ablation_null_mean",
        "ablation_null_std",
        "ablation_severity",
        "contract_source",
        "contract_window_days",
        "llm_role",
        "llm_remediation",
        # 5 evaluator audit keys (Plan layer4_evaluator_audit_signal.md):
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
    }
)


@dataclass(frozen=True)
class VerdictRecord:
    """One verdict row extracted from a sidecar JSON.

    Mirrors the producer's per-verdict dict shape but typed for downstream
    consumers. Missing keys (including the 5 evaluator audit keys from
    pre-2026-05-15 runs) surface as ``None``.
    """

    experiment_id: str
    written_at: datetime
    source_path: Path
    feature: str
    layer: Optional[str]
    severity: Optional[str]
    remediation: Optional[str]
    evidence: Optional[str]
    z_score: Optional[float]
    p_value: Optional[float]
    delta_auc: Optional[float]
    evaluator_satisfied: Optional[bool]
    evaluator_rationale_complete: Optional[bool]
    evaluator_missed_considerations: Optional[list[str]]
    evaluator_notes: Optional[str]
    evaluator_model: Optional[str]
    raw_verdict: dict[str, Any]


class SidecarReader:
    """Iterate over adaptive-validity sidecar JSONs under a directory.

    Optional time-window filter via ``since`` / ``until`` (both
    timezone-aware datetimes); a sidecar is included iff its
    ``written_at`` falls in ``[since, until]`` (closed interval). Pass
    ``None`` for either bound to skip that side.
    """

    def __init__(
        self,
        *,
        artifacts_dir: Path,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> None:
        self._artifacts_dir = Path(artifacts_dir)
        self._since = since
        self._until = until

    def iter_verdict_records(self) -> Iterator[VerdictRecord]:
        if not self._artifacts_dir.exists():
            return
        for path in sorted(self._artifacts_dir.rglob("adaptive_verdicts_*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "SidecarReader: skipping malformed sidecar %s: %s",
                    path,
                    exc,
                )
                continue
            written_at_raw = payload.get("written_at")
            written_at = self._parse_iso8601(written_at_raw)
            if written_at is None:
                logger.warning(
                    "SidecarReader: sidecar %s has unparseable written_at=%r; skipping",
                    path,
                    written_at_raw,
                )
                continue
            if self._since is not None and written_at < self._since:
                continue
            if self._until is not None and written_at > self._until:
                continue
            # Issue #235: schema_version handling (missing → legacy WARN;
            # unknown-major → WARN with both versions; both still parse).
            self._check_schema_version(path, payload.get("schema_version"))
            experiment_id = str(payload.get("experiment_id", "<unknown>"))
            verdicts_raw = payload.get("adaptive_verdicts", [])
            # Issue #235 A3: emit unknown-verdict-key WARN at most ONCE per
            # file (not per-record). PRE-SCAN before yielding so the warning
            # always fires for parsed files even when the caller consumes
            # the generator lazily (``next()``, break-early, or mid-iter
            # error). codex MED-1 (2026-05-15): emitting after the yield
            # loop was caller-controlled — pre-scanning makes the warning
            # unconditional on parse.
            self._warn_unknown_verdict_keys(path, verdicts_raw)
            for raw in verdicts_raw:
                if not isinstance(raw, dict):
                    continue
                yield self._build_record(
                    experiment_id=experiment_id,
                    written_at=written_at,
                    source_path=path,
                    raw=raw,
                )

    @staticmethod
    def _warn_unknown_verdict_keys(path: Path, verdicts_raw: Any) -> None:
        """Pre-scan ``verdicts_raw`` for keys not in ``_KNOWN_VERDICT_KEYS``;
        emit at most ONE WARN per file naming the sorted unknown-key set.
        Tolerant of non-dict entries (they are skipped by the main loop)."""
        if not isinstance(verdicts_raw, list):
            return
        seen_unknown: set[str] = set()
        for raw in verdicts_raw:
            if not isinstance(raw, dict):
                continue
            seen_unknown.update(k for k in raw.keys() if k not in _KNOWN_VERDICT_KEYS)
        if seen_unknown:
            logger.warning(
                "SidecarReader: sidecar %s contains unknown verdict key(s) %s — "
                "producer schema drift or forward-compat additive field. Reader "
                "preserves them in raw_verdict but does not surface them on "
                "VerdictRecord.",
                path,
                sorted(seen_unknown),
            )

    def _check_schema_version(self, path: Path, raw_version: Any) -> None:
        """Issue #235 A2: log WARN on missing or unknown-major schema_version.

        Missing → emit "treating as legacy v0" WARN (one-off per file).
        Unknown major → emit WARN with both the payload's version and the
        reader's expected version. In both cases, parsing continues —
        backward / forward compat is the goal.
        """
        if raw_version is None:
            logger.warning(
                "SidecarReader: sidecar %s missing schema_version — treating as "
                "legacy v0 (pre-Issue-#235). All known keys still parsed.",
                path,
            )
            return
        version_str = str(raw_version)
        try:
            major = int(version_str.split(".", 1)[0])
        except (ValueError, IndexError):
            logger.warning(
                "SidecarReader: sidecar %s has unparseable schema_version=%r; "
                "reader expected major %d (e.g. %s). Parsing known keys only.",
                path,
                raw_version,
                _READER_SCHEMA_MAJOR,
                _READER_SCHEMA_VERSION,
            )
            return
        if major != _READER_SCHEMA_MAJOR:
            logger.warning(
                "SidecarReader: sidecar %s has schema_version=%s (major=%d) but "
                "reader expects major=%d (current=%s). Parsing known keys; some "
                "newer fields may not surface.",
                path,
                version_str,
                major,
                _READER_SCHEMA_MAJOR,
                _READER_SCHEMA_VERSION,
            )

    @staticmethod
    def _parse_iso8601(value: Any) -> Optional[datetime]:
        """Accept both ISO-extended ("2026-05-15T10:30:00Z") and the
        producer's compact form ("20260515T103000Z"). Python 3.12+
        ``datetime.fromisoformat`` handles the compact form natively."""
        if not isinstance(value, str):
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None

    @staticmethod
    def _build_record(
        *,
        experiment_id: str,
        written_at: datetime,
        source_path: Path,
        raw: dict[str, Any],
    ) -> VerdictRecord:
        return VerdictRecord(
            experiment_id=experiment_id,
            written_at=written_at,
            source_path=source_path,
            feature=str(raw.get("feature", "<unknown>")),
            layer=_opt_str(raw.get("layer")),
            severity=_opt_str(raw.get("severity")),
            remediation=_opt_str(raw.get("remediation")),
            evidence=_opt_str(raw.get("evidence")),
            z_score=_opt_float(raw.get("z_score")),
            p_value=_opt_float(raw.get("p_value")),
            delta_auc=_opt_float(raw.get("delta_auc")),
            evaluator_satisfied=_opt_bool(raw.get("evaluator_satisfied")),
            evaluator_rationale_complete=_opt_bool(raw.get("evaluator_rationale_complete")),
            evaluator_missed_considerations=_opt_str_list(
                raw.get("evaluator_missed_considerations")
            ),
            evaluator_notes=_opt_str(raw.get("evaluator_notes")),
            evaluator_model=_opt_str(raw.get("evaluator_model")),
            raw_verdict=raw,
        )


def _opt_str(value: Any) -> Optional[str]:
    return str(value) if isinstance(value, str) else None


def _opt_bool(value: Any) -> Optional[bool]:
    """Strict bool coercion. Non-bool, non-None values log a WARNING and
    coerce to None — this surfaces producer-side schema drift (e.g.
    ``"false"`` strings) instead of silently dropping disagreements.
    Codex Gate-2 MED-1 (2026-05-15)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    logger.warning(
        "SidecarReader: non-bool value %r (type=%s) in a bool field; "
        "coercing to None. Indicates producer schema drift — investigate.",
        value,
        type(value).__name__,
    )
    return None


def _opt_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _opt_str_list(value: Any) -> Optional[list[str]]:
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str)]
    return None


@dataclass(frozen=True)
class DisagreementEvent:
    """A verdict where the evaluator did NOT validate the worker's rationale.

    Carries forward enough context for the curation CLI to format a
    candidate compile-set entry: feature name, worker verdict, evaluator
    critique, source attribution.
    """

    experiment_id: str
    written_at: datetime
    source_path: Path
    feature: str
    worker_severity: Optional[str]
    worker_remediation: Optional[str]
    rationale_complete: Optional[bool]
    missed_considerations: tuple[str, ...]
    notes: str
    evaluator_model: Optional[str]


def extract_disagreements(
    records: Iterator[VerdictRecord] | list[VerdictRecord],
) -> Iterator[DisagreementEvent]:
    """Yield one DisagreementEvent per VerdictRecord where the evaluator
    explicitly said satisfied=False. Records with evaluator_satisfied=None
    (evaluator was disabled or failed) are skipped — they are not
    disagreements, they are absences of signal.
    """
    for r in records:
        if r.evaluator_satisfied is not False:
            continue
        yield DisagreementEvent(
            experiment_id=r.experiment_id,
            written_at=r.written_at,
            source_path=r.source_path,
            feature=r.feature,
            worker_severity=r.severity,
            worker_remediation=r.remediation,
            rationale_complete=r.evaluator_rationale_complete,
            missed_considerations=tuple(r.evaluator_missed_considerations or ()),
            notes=r.evaluator_notes or "",
            evaluator_model=r.evaluator_model,
        )


def _mtime_or_zero(path: Path) -> float:
    """Best-effort file mtime in epoch seconds. Returns 0.0 (not raise)
    when the file is missing or unreadable, so the comparator stays
    total: nonexistent paths sort *below* existing paths, which means
    real artifacts always win over phantom ones."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _dedup_sort_key(e: DisagreementEvent) -> tuple[datetime, float, str, str]:
    """Recency-first composite key (Issue #234).

    Order of precedence:
      1. ``written_at`` (payload timestamp, second-resolution per producer).
      2. ``mtime`` of ``source_path`` — recency fallback when ``written_at``
         ties to the second. Newer file wins. Missing files contribute 0.0
         so they never beat a real artifact.
      3. ``str(source_path)`` — last-resort STABLE lex tiebreaker.
         Documented as deterministic-only, not semantically meaningful.
      4. ``experiment_id`` — final lex tiebreaker for total ordering.

    Prior behaviour (codex Gate-2 MED-2) used lex ``str(source_path)`` as
    the *only* tiebreaker; Issue #234 changes this to recency-based,
    because numeric subpaths (``exp-1`` < ``exp-10`` < ``exp-2``) made the
    lex winner unpredictable and unrelated to actual run recency.
    """
    return (e.written_at, _mtime_or_zero(e.source_path), str(e.source_path), e.experiment_id)


def dedup_disagreements(
    events: Iterator[DisagreementEvent] | list[DisagreementEvent],
) -> Iterator[DisagreementEvent]:
    """Collapse duplicate disagreements by feature name. When multiple
    events name the same feature, the LATEST (per ``_dedup_sort_key``)
    is kept so the curated entry reflects the most recent evaluator
    critique. The composite key is ordered ``written_at`` descending,
    then file mtime descending, then ``source_path``/``experiment_id``
    lex as last-resort deterministic tiebreakers (Issue #234).

    Output ordering is deterministic: feature name ascending. Required
    because the downstream JSON manifest is human-diffed across runs.
    """
    by_feature: dict[str, DisagreementEvent] = {}
    for e in events:
        existing = by_feature.get(e.feature)
        if existing is None or _dedup_sort_key(e) > _dedup_sort_key(existing):
            by_feature[e.feature] = e
    for feature in sorted(by_feature):
        yield by_feature[feature]
