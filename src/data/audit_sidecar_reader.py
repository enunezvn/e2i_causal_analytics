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
                    path, exc,
                )
                continue
            written_at_raw = payload.get("written_at")
            written_at = self._parse_iso8601(written_at_raw)
            if written_at is None:
                logger.warning(
                    "SidecarReader: sidecar %s has unparseable written_at=%r; "
                    "skipping",
                    path, written_at_raw,
                )
                continue
            if self._since is not None and written_at < self._since:
                continue
            if self._until is not None and written_at > self._until:
                continue
            experiment_id = str(payload.get("experiment_id", "<unknown>"))
            for raw in payload.get("adaptive_verdicts", []):
                if not isinstance(raw, dict):
                    continue
                yield self._build_record(
                    experiment_id=experiment_id,
                    written_at=written_at,
                    source_path=path,
                    raw=raw,
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
            evaluator_rationale_complete=_opt_bool(
                raw.get("evaluator_rationale_complete")
            ),
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
        value, type(value).__name__,
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
