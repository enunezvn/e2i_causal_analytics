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
#
# 1.1 (Issue #237 Phase 1): additive ``role_attributions`` list. Reader
# pins MAJOR=1; minor bumps do not WARN.
# 1.2 (Issue #240 Stage 1): additive shadow promotion keys
# (would_promote_severity / would_flag_for_review / rationale_incomplete_flag).
# 1.3 (Issue #240 Stage 3): additive env-gated soft-gate keys
# (gate_rule_fired / worker_severity_pre_gate). Still MAJOR=1 — additive,
# nullable, absent on pre-#240-Stage-3 sidecars (surface as None).
# 1.4 (Issue #501 / #240): additive leakage × role cross-check shadow key
# (would_flag_role_leak_disagreement). Still MAJOR=1 — additive, nullable,
# absent on pre-#501 sidecars (surface as None without a warning).
# 1.5 (Issue #501 / #240): additive M-structure structural-remediation gate
# shadow keys (structural_role / structural_llm_disagreement /
# structural_remediation_override / structural_gate_fired). Still MAJOR=1 —
# additive, nullable, absent on pre-1.5 sidecars (surface as None without a
# warning, mirroring the 1.4 leak-crosscheck-key handling).
# 1.6 (Layer-4 Phase 1): additive run-level ``leakage_fdr`` summary key (the FDR
# firing-driver decision). Run-level (not per-verdict) so the reader does not
# iterate it; bumping the expected version keeps the exact-match contract from
# WARNing on the new minor. Still MAJOR=1.
# 1.7 (Layer-4 Phase 2): additive per-verdict ``structural_unclassifiable`` key
# (True when the deterministic structural decider fired on an unclassifiable
# attestation → review). Additive, nullable, absent on pre-1.7 sidecars (surface
# as None without a warning, mirroring the 1.5 structural-key handling). MAJOR=1.
# 1.8 (Phase 2.6 citation channel, #1608): five additive per-verdict citation
# keys (``citations_checked`` / ``citations_verified`` / ``citations_unverified``
# / ``cited_pmids`` / ``verified_citation_ids``) recording whether the PMIDs an
# LLM verdict cited were actually verified against the abstracts behind them.
# Emitted unconditionally by every verdict path so the schema stays uniform;
# absent on pre-1.8 sidecars (surface as None/[] without a warning). MAJOR=1.
_READER_SCHEMA_VERSION = "1.8"
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
        # Phase 2.6 citation channel (#1608). Registered in lockstep with the
        # writer's 1.8 bump so a current sidecar does not trip the
        # unknown-verdict-key WARN on every file.
        "citations_checked",
        "citations_verified",
        "citations_unverified",
        "cited_pmids",
        "verified_citation_ids",
        # 5 evaluator audit keys (Plan layer4_evaluator_audit_signal.md):
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
        # Issue #240 Stage 1 (shadow mode): three nullable promotion-rule
        # flags emitted by ``_ensemble_to_legacy_dict``. Registered here so
        # they parse onto VerdictRecord (and feed the mirror's dedicated
        # columns) without tripping the unknown-verdict-key WARN. Additive
        # at schema 1.2+ — absent on pre-#240 sidecars (surface as None).
        "would_promote_severity",
        "would_flag_for_review",
        "rationale_incomplete_flag",
        # Issue #240 Stage 3 (env-gated soft-gate): two nullable keys
        # emitted by ``_ensemble_to_legacy_dict``. ``gate_rule_fired`` names
        # the rule that flipped severity (only "R1" today; None when the
        # gate was off or did not fire); ``worker_severity_pre_gate`` is the
        # un-mutated worker severity recorded BEFORE substitution (design §5
        # R-4). Registered so they parse onto VerdictRecord and feed the
        # mirror's dedicated columns. Additive at schema 1.3+; absent on
        # pre-Stage-3 sidecars (surface as None).
        "gate_rule_fired",
        "worker_severity_pre_gate",
        # Issue #501 / #240 (leakage × role cross-check, shadow mode): one
        # nullable boolean key. ``True`` when the LLM assigned a benign
        # keep-as-clean-predictor role (ancestor/confounder/instrument) AND
        # ``detect_leakage`` independently flagged the same feature at
        # critical/high severity. ``None`` otherwise (either input absent,
        # role non-benign, or severity below threshold). Additive at schema
        # 1.4+; absent on pre-#501 sidecars (surface as None without a
        # warning — matches the existing Stage-3 precedent).
        "would_flag_role_leak_disagreement",
        # Issue #501 / #240 (M-structure structural-remediation gate, shadow
        # mode): four nullable keys emitted by the validity-node per-feature
        # loop. ``structural_role`` is the role the M-structure-extended
        # deterministic extractor derives from the feature's authored
        # ``FeatureContract.causal_structure`` edge list (None when un-attested).
        # ``structural_llm_disagreement`` (bool) is True iff that role differs
        # from the LLM role. ``structural_remediation_override`` is the
        # remediation the gate forced when ON (e.g. "drop"), else None.
        # ``structural_gate_fired`` is "R-STRUCT" when the env-gated override
        # fired, else None. Additive at schema 1.5+; absent on pre-1.5 sidecars
        # (surface as None without a warning).
        "structural_role",
        "structural_llm_disagreement",
        "structural_remediation_override",
        "structural_gate_fired",
        # Plan v4 Layer B / Phase 2: True when the voter's structural rule fired
        # on an unclassifiable/malformed attestation (routed to review). Additive
        # at schema 1.7+; absent on pre-1.7 sidecars (surface as None).
        "structural_unclassifiable",
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
    # Issue #241: evaluator telemetry — latency + tokens + cost. ``None``
    # on pre-#241 sidecars (older sidecars don't carry these keys; the
    # reader is intentionally schema-tolerant). Defaults so existing
    # test fixtures that construct VerdictRecord positionally without
    # the telemetry fields keep working.
    evaluator_latency_ms: Optional[float] = None
    evaluator_input_tokens: Optional[int] = None
    evaluator_output_tokens: Optional[int] = None
    evaluator_cost_usd: Optional[float] = None
    # Phase 1 of causal-role propagation (Issue #237, schema 1.1).
    # ``None`` on pre-1.1 sidecars (the role_attributions key is
    # absent). When set, mirrors one row of the sidecar's
    # ``role_attributions`` list: ``{feature, causal_role, source,
    # evaluator_satisfied, evaluator_model}``. The lookup map is built
    # ONCE per file at ``iter_verdict_records`` (O(n) construction,
    # O(1) per-record lookup) — see codex iter-2 fix in plan §1.5.
    role_attribution: Optional[dict[str, Any]] = None
    # Issue #240 Stage 1 (shadow mode): three nullable promotion-rule
    # flags. ``None`` on pre-#240 sidecars (the producer did not yet emit
    # these keys) AND on any row where the rule did not fire. Surfaced so
    # the mirror can populate migration-042's dedicated typed columns
    # (``would_promote_severity`` / ``would_flag_for_review`` /
    # ``rationale_incomplete_flag``) rather than leaving the firing signal
    # buried in the ``verdict`` JSONB blob. See
    # ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 1.
    would_promote_severity: Optional[str] = None
    would_flag_for_review: Optional[bool] = None
    rationale_incomplete_flag: Optional[bool] = None
    # Issue #240 Stage 3 (env-gated soft-gate): two nullable fields. ``None``
    # on pre-Stage-3 sidecars (the producer did not yet emit these keys) AND
    # on any row where the gate was disabled / did not fire.
    # ``gate_rule_fired`` names the rule that modulated severity (only "R1"
    # today); ``worker_severity_pre_gate`` is the un-mutated worker severity
    # recorded before substitution (design §5 R-4) so curation never trains
    # on a gate-escalated label. Surfaced so the mirror can populate
    # migration-043's dedicated typed column. See
    # ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3/§5.
    gate_rule_fired: Optional[str] = None
    worker_severity_pre_gate: Optional[str] = None
    # Issue #501 / #240 (leakage × role cross-check, shadow mode): nullable
    # boolean. ``True`` when the LLM assigned a benign keep-as-clean-predictor
    # role (ancestor/confounder/instrument) AND ``detect_leakage`` independently
    # flagged the same feature at critical/high severity (the statistical
    # detector and the LLM reason on orthogonal inputs — feature values-vs-target
    # vs. name/derivation metadata — so this is a genuinely independent signal).
    # ``None`` when either input is absent, the role is non-benign, or the
    # statistical severity is below the threshold. Absent on pre-#501 sidecars
    # (surface as None without a warning). Additive at schema 1.4+.
    would_flag_role_leak_disagreement: Optional[bool] = None
    # Issue #501 / #240 (M-structure structural-remediation gate, shadow mode):
    # four nullable fields. ``structural_role`` is the deterministic role the
    # M-structure-extended extractor derives from the feature's authored DAG
    # fragment (None when un-attested). ``structural_llm_disagreement`` (bool) is
    # True iff that role disagrees with the LLM role. ``structural_remediation_-
    # override`` is the remediation the gate forced when ON (e.g. "drop"), else
    # None. ``structural_gate_fired`` is "R-STRUCT" when the env-gated override
    # fired, else None. Absent on pre-1.5 sidecars (surface as None without a
    # warning). Additive at schema 1.5+.
    structural_role: Optional[str] = None
    structural_llm_disagreement: Optional[bool] = None
    structural_remediation_override: Optional[str] = None
    structural_gate_fired: Optional[str] = None
    # Plan v4 Layer B / Phase 2: True when the structural rule decided on an
    # unclassifiable/malformed attestation (decided_by="structural", role None →
    # review). Absent on pre-1.7 sidecars (surface as None). Additive at 1.7+.
    structural_unclassifiable: Optional[bool] = None
    # Phase 2.6 citation channel (schema 1.8, #1608). ``None`` on pre-1.8
    # sidecars. ``citations_checked`` distinguishes "the LLM cited nothing"
    # from "everything it cited failed verification" — a distinction the two
    # counts alone cannot express, and the one that matters when auditing
    # whether a cited PMID was ever checked against the abstract behind it.
    citations_checked: Optional[int] = None
    citations_verified: Optional[int] = None
    citations_unverified: Optional[int] = None
    cited_pmids: Optional[list[str]] = None
    verified_citation_ids: Optional[list[str]] = None


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
            # codex pass-2 MED-1 (2026-05-15): normalize non-list payloads
            # to ``[]`` after a WARN — a sidecar carrying ``null`` or a
            # scalar in ``adaptive_verdicts`` must not crash downstream
            # consumers (``for raw in verdicts_raw`` would TypeError).
            if not isinstance(verdicts_raw, list):
                logger.warning(
                    "SidecarReader: sidecar %s has non-list adaptive_verdicts=%r "
                    "(type=%s); treating as empty.",
                    path,
                    verdicts_raw,
                    type(verdicts_raw).__name__,
                )
                verdicts_raw = []
            # Issue #235 A3: emit unknown-verdict-key WARN at most ONCE per
            # file (not per-record). PRE-SCAN before yielding so the warning
            # always fires for parsed files even when the caller consumes
            # the generator lazily (``next()``, break-early, or mid-iter
            # error). codex MED-1 (2026-05-15): emitting after the yield
            # loop was caller-controlled — pre-scanning makes the warning
            # unconditional on parse.
            self._warn_unknown_verdict_keys(path, verdicts_raw)
            # Phase 1 of Issue #237 (schema 1.1): build a per-file
            # {feature: role_attribution} map ONCE so per-record lookup
            # is O(1). The plan §1.5 codex iter-2 fix is explicit: do
            # this at file-load boundary, not per-record (which would be
            # O(n²)).
            role_attributions_raw = payload.get("role_attributions") or []
            role_map: dict[str, dict[str, Any]] = {}
            if isinstance(role_attributions_raw, list):
                for attr in role_attributions_raw:
                    if not isinstance(attr, dict):
                        continue
                    feature_name = attr.get("feature")
                    if isinstance(feature_name, str):
                        role_map[feature_name] = attr
            for raw in verdicts_raw:
                if not isinstance(raw, dict):
                    continue
                feature_name = raw.get("feature")
                attr = role_map.get(feature_name) if isinstance(feature_name, str) else None
                yield self._build_record(
                    experiment_id=experiment_id,
                    written_at=written_at,
                    source_path=path,
                    raw=raw,
                    role_attribution=attr,
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
        role_attribution: Optional[dict[str, Any]] = None,
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
            # Issue #241: telemetry fields. Pre-#241 sidecars don't
            # carry these keys; ``_opt_float`` / ``_opt_int`` return
            # ``None`` on missing input.
            evaluator_latency_ms=_opt_float(raw.get("evaluator_latency_ms")),
            evaluator_input_tokens=_opt_int(raw.get("evaluator_input_tokens")),
            evaluator_output_tokens=_opt_int(raw.get("evaluator_output_tokens")),
            evaluator_cost_usd=_opt_float(raw.get("evaluator_cost_usd")),
            raw_verdict=raw,
            role_attribution=role_attribution,
            # Issue #240 Stage 1 (shadow mode): additive promotion-rule
            # flags. ``_opt_str`` / ``_opt_bool`` return None on missing
            # keys (pre-#240 sidecars) — same schema-tolerant pattern as
            # the evaluator-audit fields above.
            would_promote_severity=_opt_str(raw.get("would_promote_severity")),
            would_flag_for_review=_opt_bool(raw.get("would_flag_for_review")),
            rationale_incomplete_flag=_opt_bool(raw.get("rationale_incomplete_flag")),
            # Issue #240 Stage 3: env-gated soft-gate fields. ``_opt_str``
            # returns None on missing keys (pre-Stage-3 sidecars) — same
            # schema-tolerant pattern as the shadow fields above.
            gate_rule_fired=_opt_str(raw.get("gate_rule_fired")),
            worker_severity_pre_gate=_opt_str(raw.get("worker_severity_pre_gate")),
            # Issue #501 / #240 (leakage × role cross-check, shadow mode).
            # ``_opt_bool`` returns None on missing keys (pre-#501 sidecars)
            # and emits a WARNING on non-bool values (producer schema drift)
            # — same strict-bool-coercion contract as the Stage-1 shadow
            # fields (``would_flag_for_review``, ``rationale_incomplete_flag``).
            would_flag_role_leak_disagreement=_opt_bool(
                raw.get("would_flag_role_leak_disagreement")
            ),
            # Issue #501 / #240 (M-structure structural-remediation gate, shadow
            # mode). ``_opt_str`` / ``_opt_bool`` return None on missing keys
            # (pre-1.5 sidecars) — same schema-tolerant pattern as the fields
            # above. ``structural_role`` / ``structural_remediation_override`` /
            # ``structural_gate_fired`` are strings; ``structural_llm_disagreement``
            # is a bool (strict coercion, WARNING on producer schema drift).
            structural_role=_opt_str(raw.get("structural_role")),
            structural_llm_disagreement=_opt_bool(raw.get("structural_llm_disagreement")),
            structural_remediation_override=_opt_str(raw.get("structural_remediation_override")),
            structural_gate_fired=_opt_str(raw.get("structural_gate_fired")),
            structural_unclassifiable=_opt_bool(raw.get("structural_unclassifiable")),
            citations_checked=_opt_int(raw.get("citations_checked")),
            citations_verified=_opt_int(raw.get("citations_verified")),
            citations_unverified=_opt_int(raw.get("citations_unverified")),
            cited_pmids=_opt_str_list(raw.get("cited_pmids")),
            verified_citation_ids=_opt_str_list(raw.get("verified_citation_ids")),
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


def _opt_int(value: Any) -> Optional[int]:
    """Strict int coercion: token counts are integer counts, not floats.
    Issue #241.

    Returns ``None`` for non-int / non-numeric values (including bools,
    which trivially-subclass int but never represent a token count) and
    for missing keys (pre-#241 sidecars)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


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
    # Issue #240 Stage 2 (curation surfacing): the Stage-1 shadow
    # ``would_promote_severity`` field plus the R1 input signals that drove
    # it. ``would_promote_severity`` is the proposed escalated severity (e.g.
    # ``"high"``) recorded by R1 in shadow mode, or ``None`` when R1 did not
    # fire (or the producer predates #240). ``evaluator_satisfied`` is carried
    # explicitly so the markdown "Promotion candidate" section can show the
    # full R1 trigger context (worker_severity + evaluator_satisfied +
    # len(missed_considerations) >= 1) without the reader having to re-derive
    # it. Additive/nullable with defaults so existing keyword-only
    # constructions in tests keep working. Design ref:
    # ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 2 + §4 R1.
    would_promote_severity: Optional[str] = None
    evaluator_satisfied: Optional[bool] = None


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
            # Issue #240 Stage 2: carry the Stage-1 shadow promotion field
            # (verbatim from the sidecar; ``None`` when R1 did not fire) plus
            # the explicit ``evaluator_satisfied`` driving signal. By
            # construction this loop only yields when
            # ``r.evaluator_satisfied is False``, so the field is recorded
            # for downstream display rather than re-derived.
            would_promote_severity=r.would_promote_severity,
            evaluator_satisfied=r.evaluator_satisfied,
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
